import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms, roi_align
from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

import mobileclip


MODEL_REGISTRY = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "mm-grounding-dino-tiny": "rziga/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det",
    "mm-grounding-dino-base": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
    "mm-grounding-dino-large": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
    "llmdet-tiny": "iSEE-Laboratory/llmdet_tiny",
    "llmdet-base": "iSEE-Laboratory/llmdet_base",
    "llmdet-large": "iSEE-Laboratory/llmdet_large",
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(image_dir: str, recursive: bool = False) -> List[str]:
    image_paths = []
    if recursive:
        walker = os.walk(image_dir)
    else:
        walker = [(image_dir, [], os.listdir(image_dir))]

    for root, _, files in walker:
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, name))

    image_paths.sort()
    return image_paths


def safe_stem_for_output(image_path: str, image_dir: Optional[str] = None) -> str:
    if image_dir is not None:
        rel = os.path.relpath(image_path, image_dir)
        stem = os.path.splitext(rel)[0]
    else:
        stem = os.path.splitext(os.path.basename(image_path))[0]
    return stem.replace(os.sep, "__").replace(" ", "_")


@dataclass
class Detection:
    label: str
    bbox: List[float]  # [x, y, w, h]
    score: float
    refined_score: Optional[float] = None


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum("n,d->nd", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_x, emb_y], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_x, grid_y, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape([2, grid_h * grid_w])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


class DetRefiner(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        num_segments: int = 4,
        temperature_init: float = 0.03,
    ) -> None:
        super().__init__()
        num_patches_e = 196
        dino_dim = 768

        def proj_block(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            )

        self.linear_c = proj_block(dino_dim, model_dim)
        self.linear_d = proj_block(dino_dim, model_dim)
        self.linear_e = proj_block(dino_dim, model_dim)
        self.logit_scale_cls = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))
        self.logit_scale_roi = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

        seg_ids = torch.tensor([0, 1] + [2] * 4 + [3] * num_patches_e, dtype=torch.long)
        self.register_buffer("seg_ids", seg_ids.unsqueeze(0), persistent=False)
        self.segment_embedding = nn.Embedding(num_segments, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        grid_size_e = int(math.sqrt(num_patches_e))
        pos_enc_e = get_2d_sincos_pos_embed(model_dim, grid_size_e, grid_size_e)
        self.register_buffer("positional_encoding_e", pos_enc_e.unsqueeze(0), persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.post_ln = nn.LayerNorm(model_dim)
        self.logit_scale_cls.requires_grad_(False)
        self.logit_scale_roi.requires_grad_(False)

    def forward(self, feat_c: torch.Tensor, feat_d: torch.Tensor, feat_e: torch.Tensor):
        bsz = feat_c.size(0)
        feat_c = self.linear_c(feat_c).unsqueeze(1)
        feat_d = self.linear_d(feat_d)
        feat_e = self.linear_e(feat_e)

        num_patches = feat_e.size(1)
        grid = int(num_patches ** 0.5)
        if grid * grid != num_patches:
            raise ValueError(f"Patch count {num_patches} is not square.")
        if self.positional_encoding_e.size(1) != num_patches:
            pos_e = get_2d_sincos_pos_embed(feat_e.size(-1), grid, grid).to(feat_e.device).unsqueeze(0)
        else:
            pos_e = self.positional_encoding_e.to(feat_e.device)
        feat_e = feat_e + pos_e.to(dtype=feat_e.dtype)

        x = torch.cat([feat_c, feat_d, feat_e], dim=1)
        x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = x + self.segment_embedding(self.seg_ids.expand(bsz, -1).to(x.device))
        x = self.post_ln(self.encoder(x))

        start_e = 1 + 1 + 4
        return x[:, 0, :], x[:, start_e:, :]


class OnlineFeatureExtractor:
    def __init__(self, mobileclip_ckpt: str, dinov3_path: str, device: str):
        self.device = torch.device(device)
        self.mobileclip_model, _, _ = mobileclip.create_model_and_transforms(
            "mobileclip_b", pretrained=mobileclip_ckpt
        )
        self.mobileclip_tokenizer = mobileclip.get_tokenizer("mobileclip_b")
        self.mobileclip_model = self.mobileclip_model.to(self.device).eval()

        self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_path)
        self.dinov3_model = AutoModel.from_pretrained(dinov3_path).to(self.device).eval()

    @torch.no_grad()
    def encode_texts(self, labels: Sequence[str]) -> torch.Tensor:
        tokens = self.mobileclip_tokenizer(list(labels)).to(self.device)
        feats = self.mobileclip_model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        inputs = self.dinov3_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.dinov3_model(**inputs)
        dino = outputs.last_hidden_state.float()  # (1, 1+4+P, 768)
        return {
            "feat_c": dino[:, 0, :],
            "feat_d": dino[:, 1:5, :],
            "feat_e": dino[:, 5:, :],
        }


def build_captions_and_token_span(cat_list: Sequence[str]):
    cat2tokenspan, captions = {}, ""
    for class_name in cat_list:
        spans = []
        for subname in class_name.strip().split(" "):
            if captions:
                captions += " "
            start_idx = len(captions)
            captions += subname.strip()
            spans.append([start_idx, len(captions)])
        if spans:
            captions += " ."
            cat2tokenspan[class_name] = spans
    return captions, cat2tokenspan


def create_positive_map_from_span(
    tokenized,
    token_span,
    max_text_len: int = 256,
) -> torch.Tensor:
    positive_map = torch.zeros(
        (len(token_span), max_text_len),
        dtype=torch.float,
    )

    def first_not_none(*values):
        return next(
            (value for value in values if value is not None),
            None,
        )

    for j, tok_list in enumerate(token_span):
        for beg, end in tok_list:
            beg_pos = first_not_none(
                tokenized.char_to_token(beg),
                tokenized.char_to_token(beg + 1),
                tokenized.char_to_token(beg + 2),
            )
            end_pos = first_not_none(
                tokenized.char_to_token(end - 1),
                tokenized.char_to_token(end - 2),
                tokenized.char_to_token(end - 3),
            )

            if beg_pos is None or end_pos is None:
                continue

            if beg_pos >= max_text_len:
                continue

            end_pos = min(end_pos, max_text_len - 1)

            if end_pos < beg_pos:
                continue

            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    row_sums = positive_map.sum(dim=-1, keepdim=True)
    positive_map = positive_map / row_sums.clamp_min(1e-6)

    return positive_map


def build_text_prompt_and_positive_map(processor, class_names: Sequence[str]):
    cat_list = list(class_names)
    captions, cat2tokenspan = build_captions_and_token_span(cat_list)
    token_spans = [cat2tokenspan[c] for c in cat_list]
    positive_map = create_positive_map_from_span(processor.tokenizer(captions), token_spans)
    text_prompt = " . ".join(cat_list) + " ."
    return text_prompt, positive_map


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


class RealtimeDetectorRunner:
    def __init__(
        self,
        class_names: Sequence[str],
        detector_name: str,
        model_root: str,
        device: str,
        max_dets: int,
        candidate_score_thr: float,
    ) -> None:
        self.class_names = list(class_names)
        self.device = device
        self.max_dets = max_dets
        self.candidate_score_thr = candidate_score_thr

        repo_id = MODEL_REGISTRY[detector_name]
        model_id = os.path.join(model_root, repo_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

        text_prompt, positive_map = build_text_prompt_and_positive_map(self.processor, self.class_names)
        self.text_prompt = text_prompt
        self.positive_map = positive_map.to(device)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> List[Detection]:
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        logits = torch.sigmoid(outputs.logits[0]) @ self.positive_map.T
        boxes = box_cxcywh_to_xyxy(outputs.pred_boxes[0])

        if self.max_dets < 0:
            k = logits.numel()
        else:
            k = min(self.max_dets, logits.numel())
        values, indexes = torch.topk(logits.reshape(-1), k, dim=0)
        box_indexes = indexes // logits.shape[1]
        label_indexes = indexes % logits.shape[1]

        width, height = image.size
        detections = []
        for score, box_idx, label_idx in zip(values, box_indexes, label_indexes):
            score = float(score.item())
            if score < self.candidate_score_thr:
                continue
            x1, y1, x2, y2 = (
                boxes[box_idx].detach().cpu().numpy() * np.array([width, height, width, height])
            ).tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    label=self.class_names[int(label_idx)],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=score,
                )
            )
        return detections


def roi_features(
    patch_vectors: torch.Tensor,
    image_size: Tuple[int, int],
    detections: Sequence[Detection],
) -> torch.Tensor:
    width, height = image_size
    num_patches, dim = patch_vectors.shape
    grid = int(num_patches ** 0.5)

    if len(detections) == 0:
        return torch.empty(0, dim, device=patch_vectors.device)

    fmap = patch_vectors.view(1, grid, grid, dim).permute(0, 3, 1, 2)
    rois = []
    for det in detections:
        x, y, w, h = det.bbox
        rois.append(torch.tensor([0.0, x * 224 / width, y * 224 / height, (x + w) * 224 / width, (y + h) * 224 / height], device=patch_vectors.device))
    rois = torch.stack(rois)
    pooled = roi_align(fmap, rois, output_size=2, spatial_scale=grid / 224.0, aligned=True)

    return pooled.mean(dim=[2, 3])


@torch.no_grad()
def refine_detections(
    detrefiner: DetRefiner,
    image_feats: Dict[str, torch.Tensor],
    text_bank: torch.Tensor,
    label2idx: Dict[str, int],
    detections: List[Detection],
    image_size: Tuple[int, int],
    w_det: float,
    w_cls: float,
    w_patch: float,
) -> List[Detection]:
    if len(detections) == 0:
        return []

    class_vector, patch_vectors = detrefiner(image_feats["feat_c"], image_feats["feat_d"], image_feats["feat_e"])
    class_vector = F.normalize(class_vector, dim=-1)
    patch_vectors = patch_vectors[0]
    text_bank = F.normalize(text_bank, dim=-1)

    cls_scale = detrefiner.logit_scale_cls.exp().clamp(1e-2, 100.0)
    roi_scale = detrefiner.logit_scale_roi.exp().clamp(1e-2, 100.0)

    class_probs = torch.sigmoid(cls_scale * (class_vector @ text_bank.T))[0]
    roi_feats = F.normalize(roi_features(patch_vectors, image_size, detections), dim=-1)
    patch_probs = torch.sigmoid(roi_scale * (roi_feats @ text_bank.T)) if len(roi_feats) else None

    refined = []
    for i, det in enumerate(detections):
        if det.label not in label2idx:
            # keep unknown-label detections unchanged instead of crashing
            refined.append(Detection(det.label, det.bbox, det.score, refined_score=det.score))
            continue
        cidx = label2idx[det.label]
        cls_score = float(class_probs[cidx].item())
        patch_score = float(patch_probs[i, cidx].item()) if patch_probs is not None else cls_score
        fused = w_det * det.score + w_cls * cls_score + w_patch * patch_score
        refined.append(Detection(det.label, det.bbox, det.score, refined_score=float(fused)))
    return refined


def filter_sort_and_classwise_nms(
    detections: Sequence[Detection],
    score_key: str,
    score_thr: float,
    topk: int,
    nms_iou_thr: Optional[float],
) -> List[Detection]:
    def score_of(d: Detection) -> float:
        return d.score if score_key == "score" else float(d.refined_score if d.refined_score is not None else d.score)

    out = [d for d in detections if score_of(d) >= score_thr]
    if not out:
        return []

    if nms_iou_thr is not None and nms_iou_thr >= 0:
        boxes = torch.tensor(
            [[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] for d in out],
            dtype=torch.float32,
        )
        scores = torch.tensor([score_of(d) for d in out], dtype=torch.float32)
        label_to_idx = {label: i for i, label in enumerate(sorted({d.label for d in out}))}
        labels = torch.tensor([label_to_idx[d.label] for d in out], dtype=torch.int64)
        keep = batched_nms(boxes, scores, labels, float(nms_iou_thr)).tolist()
        out = [out[i] for i in keep]

    out.sort(key=score_of, reverse=True)
    return out[:topk]


def draw_detections(image: Image.Image, detections: Sequence[Detection], path: str, use_refined_score: bool = False) -> None:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    img_w, img_h = img.size
    pad = 2

    for det in detections:
        x, y, w, h = det.bbox
        score = det.refined_score if use_refined_score and det.refined_score is not None else det.score
        text = f"{det.label} {score:.3f}"

        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        tb = draw.textbbox((0, 0), text, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        box_w = text_w + pad * 2
        box_h = text_h + pad * 2

        label_x1 = max(0, min(float(x), img_w - box_w))
        label_x2 = label_x1 + box_w

        if y - box_h >= 0:
            label_y1 = y - box_h
            label_y2 = y
        else:
            label_y1 = y
            label_y2 = min(y + box_h, img_h)

        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill="red")
        draw.text(
            (label_x1 + pad, label_y1 + pad),
            text,
            fill="white",
            font=font,
        )

    img.save(path)


def save_concatenated_image(left_path: str, right_path: str, output_path: str) -> None:
    """Save two images side by side: left image first, right image second."""
    with Image.open(left_path) as left_src, Image.open(right_path) as right_src:
        left = left_src.convert("RGB")
        right = right_src.convert("RGB")

        canvas_width = left.width + right.width
        canvas_height = max(left.height, right.height)
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(output_path)


def save_json(detections: Sequence[Detection], path: str) -> None:
    rows = []

    for d in detections:
        if d.refined_score is None:
            raise ValueError(
                f"refined_score is missing for label={d.label!r}"
            )

        rows.append({
            "label": d.label,
            "bbox": [float(v) for v in d.bbox],
            "detector_score": float(d.score),
            "refined_score": float(d.refined_score),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Run inference on this image. Use either --image or --image-dir.")
    p.add_argument("--image-dir", default=None, help="Run inference on all images in this folder. Use either --image or --image-dir.")
    p.add_argument("--recursive", action="store_true", help="When --image-dir is used, also search subfolders.")
    p.add_argument("--class-names", nargs="+", required=True, help="Detection/refinement vocabulary. Must include detector labels.")
    p.add_argument("--detector-model-root", default="./data/huggingface")
    p.add_argument("--detector-name", choices=list(MODEL_REGISTRY.keys()), default="llmdet-large")
    p.add_argument("--candidate-score-thr", type=float, default=0.00, help="Minimum detector score used only for candidate generation.")
    p.add_argument("--max-dets", type=int, default=-1, help="Maximum number of detector candidate box-class pairs to keep before refinement (-1 to keep all candidates).")
    p.add_argument("--detrefiner-ckpt", default="./trained_models/detrefiner_lvis.pth", help="Path to the trained DetRefiner checkpoint.")
    p.add_argument("--mobileclip-ckpt", default="./models/mobileclip_blt.pt", help="Path to the MobileCLIP-B checkpoint.")
    p.add_argument("--dinov3-path", default="./models/facebook/dinov3-vitb16-pretrain-lvd1689m", help="Path to the DINOv3 ViT-B/16 model.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--w-det", type=float, default=0.8, help="Weight for the original detector score in the final fused score.")
    p.add_argument("--w-cls", type=float, default=0.1, help="Weight for the DetRefiner image-level class score in the final fused score.")
    p.add_argument("--w-patch", type=float, default=0.1, help="Weight for the DetRefiner ROI-level patch score in the final fused score.")
    p.add_argument("--score-thr-before", type=float, default=0.30, help="Display threshold for original detector scores.")
    p.add_argument("--score-thr-after", type=float, default=0.30, help="Display threshold for DetRefiner fused scores.")
    p.add_argument("--topk", type=int, default=300, help="Maximum number of detections to visualize per image after thresholding and NMS.")
    p.add_argument("--nms-iou-thr", type=float, default=0.30, help="Class-wise NMS IoU threshold used for before/after visualization and JSON output. Set negative to disable NMS.")
    p.add_argument("--out-dir", default="outputs_detrefiner", help="Directory where output images, JSON files, and logs will be saved.")
    p.add_argument("--overwrite", action="store_true", help="If set, remove the existing output directory before saving new results. Use with caution.")
    return p.parse_args()


def process_one_image(
    image_path: str,
    output_root: str,
    output_stem: str,
    args: argparse.Namespace,
    label2idx: Dict[str, int],
    extractor: OnlineFeatureExtractor,
    detrefiner: DetRefiner,
    text_bank: torch.Tensor,
    detector_runner: RealtimeDetectorRunner,
) -> Tuple[int, int, int]:
    with Image.open(image_path) as src:
        image = src.convert("RGB")

    detections = detector_runner(image)

    image_feats = extractor.encode_image(image)
    refined = refine_detections(
        detrefiner=detrefiner,
        image_feats=image_feats,
        text_bank=text_bank,
        label2idx=label2idx,
        detections=detections,
        image_size=image.size,
        w_det=args.w_det,
        w_cls=args.w_cls,
        w_patch=args.w_patch,
    )

    before = filter_sort_and_classwise_nms(
        refined,
        score_key="score",
        score_thr=args.score_thr_before,
        topk=args.topk,
        nms_iou_thr=args.nms_iou_thr,
    )
    after = filter_sort_and_classwise_nms(
        refined,
        score_key="refined_score",
        score_thr=args.score_thr_after,
        topk=args.topk,
        nms_iou_thr=args.nms_iou_thr,
    )

    before_png_dir = os.path.join(output_root, "before_png")
    after_png_dir = os.path.join(output_root, "after_png")
    concat_png_dir = os.path.join(output_root, "concat_png")
    json_dir = os.path.join(output_root, "json")
    for d in (before_png_dir, after_png_dir, concat_png_dir, json_dir):
        os.makedirs(d, exist_ok=True)

    before_png = os.path.join(before_png_dir, f"{output_stem}.png")
    after_png = os.path.join(after_png_dir, f"{output_stem}.png")
    concat_png = os.path.join(concat_png_dir, f"{output_stem}.png")
    result_json = os.path.join(json_dir, f"{output_stem}.json")

    draw_detections(image, before, before_png, use_refined_score=False)
    draw_detections(image, after, after_png, use_refined_score=True)
    save_concatenated_image(before_png, after_png, concat_png)
    save_json(refined, result_json)

    print(f"[{os.path.basename(image_path)}] candidates={len(detections)} before={len(before)} after={len(after)}")
    print(f"  saved: {output_stem}")
    return len(detections), len(before), len(after)


def main() -> None:
    args = parse_args()
    if args.overwrite:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    if (args.image is None) == (args.image_dir is None):
        raise ValueError("Specify exactly one of --image or --image-dir.")

    if args.image is not None:
        image_paths = [args.image]
        image_root = None
    else:
        image_paths = list_images(args.image_dir, recursive=args.recursive)
        image_root = args.image_dir
        if not image_paths:
            raise ValueError(f"No images found in {args.image_dir}")

    class_names = list(dict.fromkeys(args.class_names))  # stable unique order

    detector_runner = RealtimeDetectorRunner(
        class_names=class_names,
        detector_name=args.detector_name,
        model_root=args.detector_model_root,
        device=args.device,
        max_dets=args.max_dets,
        candidate_score_thr=args.candidate_score_thr,
    )

    label2idx = {label: i for i, label in enumerate(class_names)}

    extractor = OnlineFeatureExtractor(args.mobileclip_ckpt, args.dinov3_path, args.device)
    text_bank = extractor.encode_texts(class_names)

    detrefiner = DetRefiner().to(device).eval()
    state = torch.load(args.detrefiner_ckpt, map_location=device)
    detrefiner.load_state_dict(state, strict=True)

    print(f"candidate_score_thr: {args.candidate_score_thr}")
    print(f"score_thr_before: {args.score_thr_before}")
    print(f"score_thr_after: {args.score_thr_after}")
    print(f"nms_iou_thr: {args.nms_iou_thr}")
    print(f"num_images: {len(image_paths)}")

    total_candidates = 0
    total_before = 0
    total_after = 0

    for image_path in image_paths:
        output_stem = safe_stem_for_output(image_path, image_root)

        num_candidates, num_before, num_after = process_one_image(
            image_path=image_path,
            output_root=args.out_dir,
            output_stem=output_stem,
            args=args,
            label2idx=label2idx,
            extractor=extractor,
            detrefiner=detrefiner,
            text_bank=text_bank,
            detector_runner=detector_runner,
        )
        total_candidates += num_candidates
        total_before += num_before
        total_after += num_after

    print("done")
    print(f"total_candidate_detections: {total_candidates}")
    print(f"total_before_displayed: {total_before}")
    print(f"total_after_displayed: {total_after}")
    print(f"output_root: {args.out_dir}")

if __name__ == "__main__":
    main()