# torch==2.4.1, torchvision==0.19.0, transformers = 4.56.2
from collections import defaultdict
import glob
import json
import math
import os
import pickle
import sys
from copy import deepcopy
from typing import Optional
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import roi_align, roi_pool
import torch.multiprocessing as mp
import warnings
warnings.simplefilter('ignore')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVIS, LVISEval, LVISResults


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int):
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_x, grid_y, indexing="xy")  # (2, H, W)
    grid = torch.stack(grid, dim=0)  # (2, H, W)
    grid = grid.reshape([2, grid_h * grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_x, emb_y], dim=1)  # (N, D)


def get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.)
    omega = 1. / (10000**omega)
    pos = pos.reshape(-1)
    out = torch.einsum('n,d->nd', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def ensure_label2idx(dataset, pathname):
    """
    Build a stable label->index mapping.
    """
    if dataset in ["coco", "lvis"]:
        with open(f"./data/textfeatures_{dataset}.pkl", "rb") as f:
            textfeatures = pickle.load(f)
    elif dataset == "odinw13":
        with open(f"./data/textfeatures_{dataset}/textfeatures_" + pathname + ".pkl", "rb") as f:
            textfeatures = pickle.load(f)

    # stable order: sort by name
    labels = sorted(list(textfeatures.keys()))
    label2idx = {l: i for i, l in enumerate(labels)}

    return label2idx, textfeatures


class DetRefiner(nn.Module):
    def __init__(
        self, model_dim, num_layers, num_heads, dropout, 
        num_segments, temperature_init, dino_version, device
        ) -> None:
        super().__init__()

        if "v3" in dino_version:
            num_patches_e = 196
        elif "v2" in dino_version:
            num_patches_e = 256

        dino_dim = 768
        
        def proj_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout)
            )

        self.linear_c = proj_block(dino_dim, model_dim)
        self.linear_d = proj_block(dino_dim, model_dim)
        self.linear_e = proj_block(dino_dim, model_dim)

        self.logit_scale_cls = nn.Parameter(torch.tensor(math.log(1/temperature_init)))
        self.logit_scale_roi = nn.Parameter(torch.tensor(math.log(1/temperature_init)))

        seg_ids = torch.tensor(
            [0, 1] + [2] * 4 + [3] * num_patches_e,
            dtype=torch.long
        )  # (202/262,)
        self.register_buffer("seg_ids", seg_ids.unsqueeze(0), persistent=False)

        self.segment_embedding = nn.Embedding(num_segments, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Positional Encoding for e
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


    def forward(
        self,
        feat_c: torch.Tensor,   # (B, 768->proj)
        feat_d: torch.Tensor,   # (B, 4, 768->proj)
        feat_e: torch.Tensor,   # (B, 196/256, 768->proj)
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = feat_c.size(0)

        feat_c = self.linear_c(feat_c).unsqueeze(1)    # (B,1,512)
        feat_d = self.linear_d(feat_d)                 # (B,4,512)
        feat_e = self.linear_e(feat_e)                 # (B,P,512)

        P = feat_e.size(1)
        H_feat = int(P ** 0.5)
        if H_feat * H_feat != P:
            raise ValueError(f"Patch count {P} is not square.")
        # adjust PE if needed
        if self.positional_encoding_e.size(1) != P:
            pe = get_2d_sincos_pos_embed(feat_e.size(-1), H_feat, H_feat).to(feat_e.device)
            pos_e = pe.unsqueeze(0)
        else:
            pos_e = self.positional_encoding_e.to(feat_e.device)
        pos_e = pos_e.to(feat_e.device, dtype=feat_e.dtype)
        feat_e = feat_e + pos_e

        x = torch.cat([feat_c, feat_d, feat_e], dim=1)  # (B,203/263,512)

        cls_token_expanded = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token_expanded, x], dim=1)  # (B,204/264,512)
        x = x + self.segment_embedding(self.seg_ids.expand(B, -1).to(x.device))

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.post_ln(x)

        num_tokens_d = 4
        start_e = 1 + 1 + num_tokens_d   # cls_token + feat_c + feat_d
        class_vector = x[:, 0, :]
        patch_vectors = x[:, start_e:, :]

        return class_vector, patch_vectors


class VectorDataset(Dataset):
    def __init__(self, train, dataset, imgdir_path, clip_version, dino_version) -> None:

        if dataset=="coco":
            self.train_inputs = sorted(glob.glob("./data/visionfeatures_train2017/*.pkl"))
            self.evaluate_inputs = sorted(glob.glob("./data/visionfeatures_val2017/*.pkl"))
            self.train_gts = sorted(glob.glob("./data/gt_train2017/*.pkl"))
            self.evaluate_gts = sorted(glob.glob("./data/gt_val2017/*.pkl"))
        elif dataset=="lvis":
            self.train_inputs = sorted(glob.glob("./data/visionfeatures_train2017_lvis/*.pkl"))
            self.evaluate_inputs = sorted(glob.glob("./data/visionfeatures_val2017/*.pkl"))
            self.train_gts = sorted(glob.glob("./data/gt_train2017_lvis/*.pkl"))
            self.evaluate_gts = sorted(glob.glob("./data/gt_val2017/*.pkl"))
        elif dataset == "odinw13":
            self.train_inputs = sorted(glob.glob(os.path.join("./data/visionfeatures_odinw13", str(imgdir_path), "*.pkl")))
            self.evaluate_inputs = sorted(glob.glob(os.path.join("./data/visionfeatures_odinw13", str(imgdir_path), "*.pkl")))
            self.train_gts = sorted(glob.glob(os.path.join("./data/gt_odinw13", str(imgdir_path), "*.pkl")))
            self.evaluate_gts = sorted(glob.glob(os.path.join("./data/gt_odinw13", str(imgdir_path), "*.pkl")))

        self.train = train
        self.dataset = dataset
        self.clip_version = clip_version
        self.dino_version = dino_version

    def __len__(self) -> int:
        return len(self.train_inputs) if self.train else len(self.evaluate_inputs)

    def __getitem__(self, idx: int):
        if self.train:
            with open(self.train_inputs[idx], 'rb') as f:
                inputs = pickle.load(f)
            with open(self.train_gts[idx], 'rb') as f:
                gts = pickle.load(f)
        else:
            with open(self.evaluate_inputs[idx], 'rb') as f:
                inputs = pickle.load(f)
            with open(self.evaluate_gts[idx], 'rb') as f:
                gts = pickle.load(f)

        file_name = gts["file_name"]
        image_size = gts["image_size"]
        if self.dataset in ["coco", "odinw13"]:
            text_labels_whole = gts["coco_labels"]
            text_labels_bboxes = gts["coco_bboxes"]
        elif self.dataset == "lvis":
            text_labels_whole = gts["lvis_labels"]
            text_labels_bboxes = gts["lvis_bboxes"]
        lvis_neg_category_ids = gts["lvis_neg_category_ids"]
        lvis_not_exhaustive_category_ids = gts["lvis_not_exhaustive_category_ids"]

        feat_b = torch.as_tensor(inputs[str(self.clip_version)+"_visual_features"],dtype=torch.float32).squeeze(0)
        feat_c = torch.as_tensor(inputs[str(self.dino_version)+"_visual_features"][:,0,:],dtype=torch.float32).squeeze(0)
        feat_d = torch.as_tensor(inputs[str(self.dino_version)+"_visual_features"][:,1:5,:],dtype=torch.float32).squeeze(0)
        feat_e = torch.as_tensor(inputs[str(self.dino_version)+"_visual_features"][:,5:,:],dtype=torch.float32).squeeze(0)  

        return {
            "file_name": file_name,
            "image_size": image_size,
            "text_labels_whole": text_labels_whole,
            "text_labels_bboxes": text_labels_bboxes,
            "lvis_neg_category_ids": lvis_neg_category_ids,
            "lvis_not_exhaustive_category_ids": lvis_not_exhaustive_category_ids,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "feat_d": feat_d,
            "feat_e": feat_e,
        }


def custom_collate(batch):
    collated = {}

    stack_keys = ["image_size", 
                  "feat_b", 
                  "feat_c", 
                  "feat_d", 
                  "feat_e"]
    list_keys = ["file_name",
                 "text_labels_whole",
                 "text_labels_bboxes",
                 "lvis_neg_category_ids",
                 "lvis_not_exhaustive_category_ids"]

    for key in batch[0]:
        values = [d[key] for d in batch]

        if key in stack_keys:
            if key == "image_size":
                values = [torch.tensor(v, dtype=torch.long) for v in values]
            collated[key] = torch.stack(values, dim=0)
        elif key in list_keys:
            collated[key] = values
        else:
            collated[key] = values

    return collated


class MultiGranularityContrastiveLoss(nn.Module):
    def __init__(self, train, device, roi_mode, dataset, 
                 clip_version, dino_version, ls_eps, label2idx, textfeatures):
        super().__init__()

        self.train = train
        self.device = device
        self.roi_mode = roi_mode
        self.dataset = dataset
        self.clip_version = clip_version
        self.dino_version = dino_version

        self.textfeatures = textfeatures

        self.label2idx = label2idx
        self.labels = [None] * len(self.label2idx)
        for l, i in self.label2idx.items():
            self.labels[i] = l

        if self.dataset in ["coco", "lvis"]:
            with open(f"./data/annotations_created/{str(self.dataset)}_seen_classes.json", 'rb') as f:
                self.seen_classes = json.load(f)

            seen_idx_list = [label2idx[l] for l in self.seen_classes]
            self.register_buffer(
                "seen_idx",
                torch.tensor(seen_idx_list, dtype=torch.long),
                persistent=False
            )

            with open(f"./data/annotations_created/{str(self.dataset)}_unseen_classes.json", 'rb') as f:
                self.unseen_classes = json.load(f)

            unseen_idx_list = [label2idx[l] for l in self.unseen_classes]
            self.register_buffer(
                "unseen_idx",
                torch.tensor(unseen_idx_list, dtype=torch.long),
                persistent=False
            )

        self.num_classes = len(self.labels)

        text_bank = []
        for l in self.labels:
            text_bank.append(torch.as_tensor(self.textfeatures[l][str(self.clip_version)+"_text_features"], dtype=torch.float32))  # (dim)
        self.register_buffer("text_bank", torch.stack(text_bank).to(device=self.device))       # (C, dim)

        self.eps = ls_eps   # BCE with label smoothing

    def _normalize(self, x):
        return F.normalize(x, dim=-1)

    def _grid_from_P(self, P):
        H = int(P ** 0.5)
        if H * H != P:
            raise ValueError(f"Patch count {P} not square")
        patch_size = 224 // H
        return H, H, patch_size

    def _get_patch_coords(self, P):
        H, W, patch_size = self._grid_from_P(P)
        coords = []
        for j in range(H):
            for i in range(W):
                x1 = i * patch_size
                y1 = j * patch_size
                x2 = (i+1) * patch_size
                y2 = (j+1) * patch_size
                coords.append([x1, y1, x2, y2])
        return torch.tensor(coords, dtype=torch.float32)  # (P,4)


    def _roi_from_bbox(self, patch_vectors, image_size, bboxes, patch_coords, batch_idx):
        """
        patch_vectors: (P, dim)
        image_size: (W,H)
        bboxes: list of [label, [x1,y1,w,h]]
        patch_coords: (P,4)
        """
        roi_feats = []
        W, H = image_size

        if self.roi_mode == "inclusion":
            scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=patch_vectors.device)
            for label, (x1, y1, w, h) in bboxes:
                x2, y2 = x1 + w, y1 + h
                box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=patch_vectors.device) * (224.0 / scale)

                mask = ((patch_coords[:, 2] > box[0]) & (patch_coords[:, 0] < box[2]) &
                        (patch_coords[:, 3] > box[1]) & (patch_coords[:, 1] < box[3]))
                if mask.sum() == 0:
                    roi_feat = patch_vectors.mean(0, keepdim=True)
                else:
                    roi_feat = patch_vectors[mask].mean(0, keepdim=True)
                roi_feats.append((label, roi_feat))

        else:
            P, dim = patch_vectors.shape
            H_feat = W_feat = int(P ** 0.5)
            fmap = patch_vectors.view(1, H_feat, W_feat, dim).permute(0, 3, 1, 2)  # (1, dim, H_feat, W_feat)

            roi_list, labels_list = [], []
            for label, (x1, y1, w, h) in bboxes:
                x2, y2 = x1 + w, y1 + h
                # map to 224 grid
                x1_, y1_ = x1 * 224.0 / W, y1 * 224.0 / H
                x2_, y2_ = x2 * 224.0 / W, y2 * 224.0 / H
                roi = torch.tensor([0.0, x1_, y1_, x2_, y2_], device=patch_vectors.device)
                roi_list.append(roi)
                labels_list.append(label)

            if len(roi_list) > 0:
                rois = torch.stack(roi_list)
                if self.roi_mode == "roi_align":
                    pooled = roi_align(
                        fmap, rois,
                        output_size=2,
                        spatial_scale=H_feat / 224.0,
                        aligned=True
                    )  # (N, dim, 2, 2)
                    pooled = pooled.mean(dim=[2,3])  # (N, dim)
                elif self.roi_mode == "roi_pooling":
                    pooled = roi_pool(
                        fmap, rois,
                        output_size=2,
                        spatial_scale=H_feat / 224.0
                    )  # (N, dim, 2, 2)
                    pooled = pooled.mean(dim=[2,3])  # (N, dim)
                for j, label in enumerate(labels_list):
                    roi_feats.append((label, pooled[j:j+1]))
        return roi_feats

    def _bce_ls(self, logits, targets, reduction='mean'):
        # label smoothing on BCE targets: y'=(1-ε)*y + ε*0.5
        targets = (1 - self.eps) * targets + self.eps * 0.5
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, class_vector, patch_vectors, image_size, 
                feat_b, text_list, bbox_list,
                lvis_not_exhaustive_category_ids, lvis_neg_category_ids, 
                logit_scale_cls, logit_scale_roi):
        """
        class_vector: (B,dim)
        patch_vectors: (B,P,dim)
        image_size: list[(W,H)] tensors
        text_list: list[list[str]]
        bbox_list: list[list[[label,[x1,y1,w,h]]]]
        lvis_not_exhaustive_category_ids: list[list[int]]
        lvis_neg_category_ids: list[list[int]]
        logit_scale_cls: torch.tensor(math.log(1/temperature_init))
        logit_scale_roi: torch.tensor(math.log(1/temperature_init))
        """
        B, dim = class_vector.shape
        C = self.num_classes
        P = patch_vectors.shape[1]
        patch_coords = self._get_patch_coords(P).to(self.device)

        # normalize embeddings
        self.text_bank = self.text_bank.to(device=self.device, dtype=class_vector.dtype)
        class_vec_norm = self._normalize(class_vector)
        text_bank_norm = self._normalize(self.text_bank)

        scale_cls = logit_scale_cls.exp().clamp(1e-2, 100.0)
        scale_roi = logit_scale_roi.exp().clamp(1e-2, 100.0)

        # logits
        class_logits_main = scale_cls * (class_vec_norm @ text_bank_norm.t())    # (B,C)

        # targets
        target_main = torch.zeros(B, C, device=self.device)
        mask_main = torch.zeros_like(target_main)

        pairs = [(batch_idx, self.label2idx[l])
                for batch_idx, labels in enumerate(text_list)
                for l in labels]

        if pairs:
            bi, ci = torch.tensor(pairs, device=self.device).T  # (K,), (K,)
            
            target_main[bi, ci], mask_main[bi, ci] = 1, 1

        if self.dataset=="lvis":
            for i in range(B):
                for label in lvis_neg_category_ids[i]:
                    cidx = self.label2idx[label]
                    mask_main[i, cidx] = 1
        else:
            mask_main = torch.ones_like(target_main)

        if self.train:
            unseen_idx = self.unseen_idx.to(device=self.device)
            mask_main[:, unseen_idx] = 0

        # cosine kd loss for class vector
        with torch.no_grad():
            t = self._normalize(feat_b)  # (B,1,512)
        s = self._normalize(class_vector)  # (B,1,512)
        loss_class_kd = (1.0 - (s * t).sum(dim=-1)).mean()

        # cosine kd loss for patch vector
        with torch.no_grad():
            t = self._normalize(feat_b)  # (B,1,512)
        s = patch_vectors.mean(dim=1, keepdim=True)   # (B,196/256,512) to (B,1,512)
        s = self._normalize(s)
        loss_patch_kd = (1.0 - (s * t).sum(dim=-1)).mean()

        # losses
        loss_cls_main = self._bce_ls(class_logits_main, target_main, reduction='none') * mask_main
        class_loss_main = loss_cls_main.sum() / mask_main.sum().clamp_min(1.0) + 0.1 * loss_class_kd + 0.1 * loss_patch_kd

        # probs for downstream fusion
        class_probs_main = torch.sigmoid(class_logits_main)

        # --- patch / ROI ---
        patch_logits_main_list = []
        patch_probs_main_list  = []

        all_logits_main = []
        all_targets_main = []
        all_masks_main = []
        all_weights = []  # optional weighting by box area or det score if provided upstream

        for i in range(B):
            W, H = image_size[i]
            rois = self._roi_from_bbox(patch_vectors[i], (W, H), bbox_list[i], patch_coords, i)

            logits_main_i = []
            probs_main_i = []

            for label, roi_feat in rois:
                roi_feat_norm = self._normalize(roi_feat)  # (1,dim)
                logits_main = scale_roi * (roi_feat_norm @ text_bank_norm.t())    # (1,C)

                logits_main_i.append(logits_main)
                probs_main_i.append(torch.sigmoid(logits_main))

                # targets per ROI
                target_main_roi = torch.zeros(1, C, device=self.device)
                mask_main_roi   = torch.zeros(1, C, device=self.device)

                if label in self.label2idx:
                    cidx = self.label2idx[label]
                    target_main_roi[0, cidx] = 1
                    mask_main_roi[0, cidx] = 1

                if self.dataset=="lvis":
                    for label in lvis_neg_category_ids[i]:
                        cidx = self.label2idx[label]
                        mask_main_roi[0, cidx] = 1
                else:
                    mask_main_roi = torch.ones_like(target_main_roi)

                if self.train:
                    unseen_idx = self.unseen_idx.to(device=self.device)
                    mask_main_roi[:, unseen_idx] = 0

                all_logits_main.append(logits_main)
                all_targets_main.append(target_main_roi)
                all_masks_main.append(mask_main_roi)
                all_weights.append(torch.ones(1,1, device=self.device))  # replace by IoU/det score if available

            if len(logits_main_i) > 0:
                patch_logits_main_list.append(torch.cat(logits_main_i, dim=0))
                patch_probs_main_list.append(torch.cat(probs_main_i, dim=0))
            else:
                patch_logits_main_list.append(torch.empty(0, C, device=self.device))
                patch_probs_main_list.append(torch.empty(0, C, device=self.device))

        if len(all_logits_main) > 0:
            all_logits_main   = torch.cat(all_logits_main, dim=0)
            all_targets_main  = torch.cat(all_targets_main, dim=0)
            all_masks_main    = torch.cat(all_masks_main, dim=0)
            weights           = torch.cat(all_weights, dim=0)  # (N,1)

            loss_roi_main = self._bce_ls(all_logits_main, all_targets_main, reduction='none') * all_masks_main

            loss_roi_main = (loss_roi_main * weights).sum() / (all_masks_main * weights).sum().clamp_min(1.0)

        else:
            loss_roi_main = torch.tensor(0.0, device=self.device)

        return {
            "class_vector_loss_main": class_loss_main,
            "patch_vector_loss_main": loss_roi_main,

            "class_vector_logits_main": class_logits_main,
            "patch_vector_logits_main": patch_logits_main_list,

            "class_vector_probs_main": class_probs_main,
            "patch_vector_probs_main": patch_probs_main_list,
        }


def get_ap_for_cat(cat_idx, precisions):
    precision = precisions[:, :, cat_idx, 0, -1]
    precision = precision[precision > -1]
    return np.mean(precision) if precision.size else float('nan')


def evaluate(
    model, evaluate_dataloader, criterion_val, device, dataset, 
    override_category, annotation_path, pathname, 
    topk_per_image, score_thr, W_DET, W_CLS, W_PATCH, 
    JSON_OUTPUT_DIR, JSON_LOAD_FILE
):

    model.to(device)

    with open(os.path.join(JSON_OUTPUT_DIR, JSON_LOAD_FILE), 'rb') as f:
        predicted_bboxes = json.load(f)

    # Load annotations
    with open("./data/annotations/instances_val2017.json", 'r') as f:
        coco_data = json.load(f)
    with open("./data/annotations_created/lvis_v1_minival_inserted_image_name.json", 'r') as f:
        lvis_data = json.load(f)
    if annotation_path:
        with open("./data/"+annotation_path, 'r') as f:
            odinw_data = json.load(f)

    with open('./data/annotations_created/coco_seen_classes.json', 'r') as fin:
        labels_seen_coco = json.load(fin)
    with open('./data/annotations_created/coco_unseen_classes.json', 'r') as fin:
        labels_unseen_coco = json.load(fin)
    with open('./data/annotations_created/coco_other_classes.json', 'r') as fin:
        labels_other_coco = json.load(fin)
    with open('./data/annotations_created/lvis_frequent_classes.json', 'r') as fin:
        labels_frequent_lvis = json.load(fin)
    with open('./data/annotations_created/lvis_common_classes.json', 'r') as fin:
        labels_common_lvis = json.load(fin)
    with open('./data/annotations_created/lvis_rare_classes.json', 'r') as fin:
        labels_rare_lvis = json.load(fin)

    coco_seen_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_seen_coco}
    coco_unseen_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_unseen_coco}
    coco_other_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_other_coco}

    lvis_frequent_cat_ids = {cat["id"] for cat in lvis_data["categories"] if cat["name"] in labels_frequent_lvis}
    lvis_common_cat_ids = {cat["id"] for cat in lvis_data["categories"] if cat["name"] in labels_common_lvis}
    lvis_rare_cat_ids = {cat["id"] for cat in lvis_data["categories"] if cat["name"] in labels_rare_lvis}

    category_id_to_name_coco = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    category_id_to_name_lvis = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}
    if override_category:
        category_id_to_name_odinw = {
            i + 1: "None" if content["name"].strip() == "" else content["name"]
            for i, content in enumerate(override_category)
        }

    name_to_category_id_coco = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
    name_to_category_id_lvis = {cat["name"]: cat["id"] for cat in lvis_data["categories"]}
    if override_category:
        name_to_category_id_odinw = {
            "None" if content["name"].strip() == "" else content["name"]: i + 1
            for i, content in enumerate(override_category)
        }

    image_id_to_filename_coco = {img["id"]: img["file_name"] for img in coco_data["images"]}
    image_id_to_filename_lvis = {img["id"]: img["coco_url"].split("/")[-1] for img in lvis_data["images"] if "val" in img["coco_url"]}
    if annotation_path:
        image_id_to_filename_odinw = {img["id"]: img["file_name"] for img in odinw_data["images"]}

    filename_to_image_id_coco = {img["file_name"]: img["id"] for img in coco_data["images"]}
    filename_to_image_id_lvis = {img["coco_url"].split("/")[-1]: img["id"] for img in lvis_data["images"] if "val" in img["coco_url"]}
    if annotation_path:
        filename_to_image_id_odinw = {img["file_name"]: img["id"] for img in odinw_data["images"]}

    predicted_bboxes_dict = defaultdict((lambda: defaultdict(list)))
    for content in predicted_bboxes:
        if dataset == "coco":
            filename = image_id_to_filename_coco[content["image_id"]]
            category_name = category_id_to_name_coco[content["category_id"]]
        elif dataset == "lvis":
            filename = image_id_to_filename_lvis[content["image_id"]]
            category_name = category_id_to_name_lvis[content["category_id"]]
        elif dataset == "odinw13":
            filename = image_id_to_filename_odinw[content["image_id"]]
            category_name = category_id_to_name_odinw[content["category_id"]]

        predicted_bboxes_dict[filename]["image_id"] = content["image_id"]
        predicted_bboxes_dict[filename]["text_labels_whole"].append(category_name)
        predicted_bboxes_dict[filename]["text_labels_bboxes"].append([category_name, content["bbox"], content["score"]])

    # --- Re-score detections & Evaluate detection mAP---
    model.eval()
    new_predicted_bboxes_dict = deepcopy(predicted_bboxes_dict)
    new_predicted_bboxes_dict_list = list(new_predicted_bboxes_dict.keys())

    with torch.no_grad():
        for idx, batch in enumerate(evaluate_dataloader):
            if batch["file_name"][0] not in new_predicted_bboxes_dict_list:
                continue
            class_vector, patch_vectors = model(
                batch["feat_c"].to(device),
                batch["feat_d"].to(device),
                batch["feat_e"].to(device)
            )
            # Build one-image inputs using detector labels for fusion
            tmp_a = [predicted_bboxes_dict[batch["file_name"][0]]["text_labels_whole"]]
            det_list = predicted_bboxes_dict[batch["file_name"][0]]["text_labels_bboxes"]  # [(label,bbox,score), ...]
            tmp_b = [[ (label, box) for (label, box, *_) in det_list]]

            results = criterion_val(
                class_vector, patch_vectors, batch["image_size"], 
                batch["feat_b"].to(device), 
                tmp_a, tmp_b,
                lvis_not_exhaustive_category_ids=batch["lvis_not_exhaustive_category_ids"],
                lvis_neg_category_ids=batch["lvis_neg_category_ids"], 
                logit_scale_cls=model.logit_scale_cls,
                logit_scale_roi=model.logit_scale_roi
            )

            # fusion per detection
            cls_probs = results["class_vector_probs_main"][0].tolist()
            patch_probs_list = results["patch_vector_probs_main"][0].tolist()  # list per bbox

            for j, (cname, bbox, det_score) in enumerate(predicted_bboxes_dict[batch["file_name"][0]]["text_labels_bboxes"]):
                i = criterion_val.label2idx[cname]
                class_score = cls_probs[i]
                patch_score = patch_probs_list[j][i] if j < len(patch_probs_list) else class_score
                fused_score = W_DET * det_score + W_CLS * float(class_score) + W_PATCH * float(patch_score)

                new_predicted_bboxes_dict[batch["file_name"][0]]["text_labels_bboxes"][j][2] = float(fused_score)

    # Build filtered JSON with per-image topK and score threshold
    new_predicted_bboxes = []
    for filename, content in new_predicted_bboxes_dict.items():
        if dataset == "coco":
            image_id = filename_to_image_id_coco[filename]
            name_to_category_id = name_to_category_id_coco
        elif dataset == "lvis":
            image_id = filename_to_image_id_lvis[filename]
            name_to_category_id = name_to_category_id_lvis
        elif dataset == "odinw13":
            image_id = filename_to_image_id_odinw[filename]
            name_to_category_id = name_to_category_id_odinw            

        preds = []
        for cname, bbox, score in content["text_labels_bboxes"]:
            if score >= score_thr:
                preds.append((cname, bbox, score))
        preds.sort(key=lambda x: x[2], reverse=True)
        preds = preds[:topk_per_image]

        for cname, bbox, score in preds:
            new_predicted_bboxes.append({
                "image_id": image_id,
                "category_id": name_to_category_id[cname],
                "bbox": bbox,
                "score": float(score),
            })

    if dataset == "coco":
        gt_info = COCO("./data/annotations_created/instances_val2017_to_cocoformat.json")
        pred_info = gt_info.loadRes(new_predicted_bboxes)
        coco_eval = COCOeval(gt_info, pred_info, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval['precision']  # [IoU, recall, category, area, maxDets]

        per_class_ap = {cat['id']: get_ap_for_cat(idx, precisions)
                        for idx, cat in enumerate(gt_info.loadCats(gt_info.getCatIds()))}
        all_ap_cat = coco_seen_cat_ids.union(coco_unseen_cat_ids)
        type1_ap_cat = coco_seen_cat_ids
        type2_ap_cat = coco_unseen_cat_ids
        type3_ap_cat = coco_other_cat_ids
        all_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in all_ap_cat])
        type1_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in type1_ap_cat])
        type2_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in type2_ap_cat])
        type3_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in type3_ap_cat])

        print("COCO mAP(all):", all_ap)
        print("COCO mAP(seen):", type1_ap)
        print("COCO mAP(unseen):", type2_ap)
        print("COCO mAP(other):", type3_ap)

    elif dataset == "lvis":
        gt_api = LVIS("./data/annotations_created/lvis_v1_minival_inserted_image_name.json")
        lvis_results = LVISResults(gt_api, new_predicted_bboxes, max_dets=100000)
        lvis_eval = LVISEval(gt_api, lvis_results, iou_type="bbox")
        lvis_eval.run()
        lvis_eval.print_results()
        print("")

    elif dataset == "odinw13":
        gt_info = COCO("./data/"+annotation_path)
        pred_info = gt_info.loadRes(new_predicted_bboxes)
        coco_eval = COCOeval(gt_info, pred_info, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval['precision']  # [IoU, recall, category, area, maxDets]

        per_class_ap = {cat['id']: get_ap_for_cat(idx, precisions) 
                        for idx, cat in enumerate(gt_info.loadCats(gt_info.getCatIds()))}
        all_ap = np.nanmean([per_class_ap[cid] for cid in per_class_ap.keys()])

        print(all_ap)
        total_ap_list.append(all_ap)
        total_name_list[pathname] = all_ap

    sys.stdout.flush()


if __name__ == "__main__":
    torch.manual_seed(0)
    mp.set_start_method("spawn", force=True)

    dataset = "coco"   # "coco" or "lvis" or "odinw13"
    device = "cuda"
    clip_version = "mobileclip"
    dino_version = "dinov3"
    roi_mode = "roi_align"

    model = DetRefiner(
        model_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.5,
        num_segments=4,   # class, c, d, e
        temperature_init=0.03,
        dino_version=dino_version,
        device=device
        )

    if dataset == "coco":
        trained_model_path = './trained_models/detrefiner_coco.pth'
    elif dataset in ["lvis", "odinw13"]:
        trained_model_path = "./trained_models/detrefiner_lvis.pth"
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    odinw_configs = sorted(glob.glob("./data/odinw13_config/*.yaml"))

    if dataset in ["coco", "lvis"]:
        label2idx, textfeatures = ensure_label2idx(dataset, pathname=None)

        evaluate_dataset = VectorDataset(
            train=False, dataset=dataset, imgdir_path=None, 
            clip_version=clip_version, dino_version=dino_version
            )

        evaluate_dataloader = DataLoader(
            evaluate_dataset, batch_size=1, shuffle=False, pin_memory=True, 
            num_workers=16, collate_fn=custom_collate
            )

        criterion_val = MultiGranularityContrastiveLoss(
            train=False, device=device, roi_mode=roi_mode,
            dataset=dataset, clip_version=clip_version, dino_version=dino_version,
            ls_eps=0.2, label2idx=label2idx, textfeatures=textfeatures
        )

        jsonlist = [
            "results_"+dataset+"_glip_tiny.json", 
            "results_"+dataset+"_glip_large.json", 
            "results_"+dataset+"_IDEA-Research_grounding-dino-tiny.json", 
            "results_"+dataset+"_IDEA-Research_grounding-dino-base.json", 
            "results_"+dataset+"_rziga_mm_grounding_dino_tiny_o365v1_goldg_grit_v3det.json", 
            "results_"+dataset+"_openmmlab-community_mm_grounding_dino_base_o365v1_goldg_v3det.json", 
            "results_"+dataset+"_openmmlab-community_mm_grounding_dino_large_o365v2_oiv6_goldg.json",
            "results_"+dataset+"_iSEE-Laboratory_llmdet_tiny.json", 
            "results_"+dataset+"_iSEE-Laboratory_llmdet_base.json", 
            "results_"+dataset+"_iSEE-Laboratory_llmdet_large.json"
            ]

        if dataset == "coco":
            topk_per_image = 300
        elif dataset == "lvis":
            topk_per_image = 100000

        for jsonfile in jsonlist:
            print(jsonfile)
            evaluate(
                model, evaluate_dataloader, criterion_val,
                device=device, dataset=dataset, 
                override_category=None, 
                annotation_path=None, pathname=None, 
                topk_per_image=topk_per_image, score_thr=0.0, 
                W_DET=0.8, W_CLS=0.1, W_PATCH=0.1, 
                JSON_OUTPUT_DIR="eval_results", 
                JSON_LOAD_FILE=jsonfile
            )
    elif dataset == "odinw13":
        json_output_dirs = [
            "eval_results/odinw13_glip_tiny",
            "eval_results/odinw13_glip_large",
            "eval_results/odinw13_grounding-dino-tiny",
            "eval_results/odinw13_grounding-dino-base",
            "eval_results/odinw13_mm_grounding_dino_tiny",
            "eval_results/odinw13_mm_grounding_dino_base",
            "eval_results/odinw13_mm_grounding_dino_large",
            "eval_results/odinw13_llmdet_tiny",
            "eval_results/odinw13_llmdet_base",
            "eval_results/odinw13_llmdet_large",
        ]

        json_load_files = [
            "results_{pathname}_odinw13_glip_tiny.json",
            "results_{pathname}_odinw13_glip_large.json",
            "results_{pathname}_odinw13_IDEA-Research_grounding-dino-tiny.json",
            "results_{pathname}_odinw13_IDEA-Research_grounding-dino-base.json",
            "results_{pathname}_odinw13_rziga_mm_grounding_dino_tiny_o365v1_goldg_grit_v3det.json",
            "results_{pathname}_odinw13_openmmlab-community_mm_grounding_dino_base_o365v1_goldg_v3det.json",
            "results_{pathname}_odinw13_openmmlab-community_mm_grounding_dino_large_o365v2_oiv6_goldg.json",
            "results_{pathname}_odinw13_iSEE-Laboratory_llmdet_tiny.json",
            "results_{pathname}_odinw13_iSEE-Laboratory_llmdet_base.json",
            "results_{pathname}_odinw13_iSEE-Laboratory_llmdet_large.json",
        ]

        for json_output_dir, json_load_file in zip(json_output_dirs, json_load_files):
            total_ap_list = []
            total_name_list = defaultdict(str)
            print(json_output_dir)

            for path in odinw_configs:
                pathname = str(path.split("/")[-1].split(".")[0])
                print(pathname)

                with open(path, "r") as yml:
                    config = yaml.safe_load(yml)

                annotation_path = config["DATASETS"]["REGISTER"][
                    str(eval(config["DATASETS"]["TEST"])[0])
                ]["ann_file"]

                imgdir_path = config["DATASETS"]["REGISTER"][
                    str(eval(config["DATASETS"]["TEST"])[0])
                ]["img_dir"]

                override_category = eval(config["DATASETS"]["OVERRIDE_CATEGORY"])

                label2idx, textfeatures = ensure_label2idx(dataset, pathname)

                evaluate_dataset = VectorDataset(
                    train=False,
                    dataset=dataset,
                    imgdir_path=imgdir_path,
                    clip_version=clip_version,
                    dino_version=dino_version
                )

                evaluate_dataloader = DataLoader(
                    evaluate_dataset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=16,
                    collate_fn=custom_collate
                )

                criterion_val = MultiGranularityContrastiveLoss(
                    train=False,
                    device=device,
                    roi_mode=roi_mode,
                    dataset=dataset,
                    clip_version=clip_version,
                    dino_version=dino_version,
                    ls_eps=0.2,
                    label2idx=label2idx,
                    textfeatures=textfeatures
                )

                evaluate(
                    model, evaluate_dataloader, criterion_val,
                    device=device, dataset=dataset,
                    override_category=override_category, 
                    annotation_path=annotation_path, pathname=pathname, 
                    topk_per_image=100000, score_thr=0.0,
                    W_DET=0.8, W_CLS=0.1, W_PATCH=0.1,
                    JSON_OUTPUT_DIR=json_output_dir,
                    JSON_LOAD_FILE=json_load_file.format(pathname=pathname)
                )

            print(np.nanmean(total_ap_list))
            print(total_name_list)
            print("------------------------------------")