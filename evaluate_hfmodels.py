import argparse
from collections import defaultdict
import gc
import glob
import json
import os
import pickle
import shutil
import warnings

os.environ["CURL_CA_BUNDLE"] = ""

warnings.simplefilter("ignore")

import numpy as np
import torch
import tqdm
import yaml
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVIS, LVISEval, LVISResults


MODEL_REGISTRY = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "llmdet-tiny": "iSEE-Laboratory/llmdet_tiny",
    "llmdet-base": "iSEE-Laboratory/llmdet_base",
    "llmdet-large": "iSEE-Laboratory/llmdet_large",
    "mm-grounding-dino-tiny": "rziga/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det",
    "mm-grounding-dino-base": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
    "mm-grounding-dino-large": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_ovod_model(model_name="grounding-dino-tiny", model_root="./data/huggingface", device="cuda:0"):
    repo_id = MODEL_REGISTRY[model_name]
    model_id = os.path.join(model_root, repo_id)
    tag = repo_id.replace("/", "_")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, tag


def build_captions_and_token_span(cat_list):
    """
    Return:
        captions: str
        cat2tokenspan: dict, e.g. {"dog": [[0, 2]], ...}
    """
    cat2tokenspan = {}
    captions = ""

    for class_name in cat_list:
        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]

        for subname in subnamelist:
            if len(captions) > 0:
                captions += " "
            start_idx = len(captions)
            end_idx = start_idx + len(subname)
            tokens_positive_i.append([start_idx, end_idx])
            captions += subname

        if len(tokens_positive_i) > 0:
            captions += " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan


def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for beg, end in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def build_text_prompt_and_positive_map(processor, class_names):
    text_prompt = ""
    for name in class_names:
        text_prompt += name + " . "
    text_prompt = text_prompt[:-2]

    text_prompt = text_prompt.lower()
    text_prompt = text_prompt.replace("_", " ").replace("-", " ").replace("(", "").replace(")", "")

    cat_list = text_prompt.split(" . ")
    cat_list[-1] = cat_list[-1][:-1]
    print(cat_list)

    captions, cat2tokenspan = build_captions_and_token_span(cat_list)
    tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
    positive_map = create_positive_map_from_span(processor.tokenizer(captions), tokenspanlist)
    return text_prompt, positive_map


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h,
    ]
    return torch.stack(b, dim=-1)


def get_ap_for_cat(precisions, cat_idx):
    precision = precisions[:, :, cat_idx, 0, -1]
    precision = precision[precision > -1]
    return np.mean(precision) if precision.size else float("nan")


def load_lvis_valid_class_names(lvis_data, gt_dir="./data/gt_val2017"):
    gts = defaultdict(dict)
    for fpath in sorted(glob.glob(os.path.join(gt_dir, "*.pkl"))):
        parts = os.path.basename(fpath).split(".")
        key = ".".join(parts[:2]) if len(parts) >= 2 else os.path.basename(fpath)
        with open(fpath, "rb") as f:
            gts[key] = pickle.load(f)

    image_id_to_filename = {}
    for img in lvis_data["images"]:
        file_name = img.get("file_name")
        if file_name is None and "coco_url" in img:
            file_name = img["coco_url"].split("/")[-1]
        if file_name is not None:
            image_id_to_filename[img["id"]] = file_name

    valid_names_by_image_id = {}
    for image_id, filename in image_id_to_filename.items():
        gt = gts.get(filename)
        if not gt:
            continue
        valid_names_by_image_id[image_id] = set(
            gt.get("lvis_labels", []) + gt.get("lvis_neg_category_ids", [])
        )

    return valid_names_by_image_id


def run_detector_on_images(
    processor,
    model,
    gt_info,
    image_ids,
    image_path_fn,
    class_items,
    device,
    max_dets=300,
    category_id_to_name=None,
    valid_class_names_by_image_id=None,
):

    class_names = [item["name"] for item in class_items]
    text_prompt, positive_map = build_text_prompt_and_positive_map(processor, class_names)
    print(text_prompt)

    text_prompt_id_to_category_id = {i: item["id"] for i, item in enumerate(class_items)}
    positive_map = positive_map.to(device)

    all_preds = []
    for image_id in tqdm.tqdm(image_ids):
        img_info = gt_info.loadImgs(image_id)[0]
        image_path = image_path_fn(img_info)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = torch.sigmoid(outputs.logits[0])
        logits = logits @ positive_map.T
        boxes = box_cxcywh_to_xyxy(outputs.pred_boxes[0])

        topk_values, topk_indexes = torch.topk(logits.view(-1), max_dets, dim=0)
        topk_boxes = topk_indexes // logits.shape[1]
        labels = topk_indexes % logits.shape[1]
        boxes_filt = boxes[topk_boxes]

        for box, score, label in zip(boxes_filt, topk_values, labels):
            x1, y1, x2, y2 = box.cpu().numpy() * torch.tensor(image.size * 2).numpy()
            pred = round(score.max().item(), 2)
            category_id = text_prompt_id_to_category_id[label.item()]

            if valid_class_names_by_image_id is not None:
                if category_id_to_name is None:
                    raise ValueError("category_id_to_name is required when valid_class_names_by_image_id is set.")
                valid_class_names = valid_class_names_by_image_id.get(image_id)
                if valid_class_names is not None and category_id_to_name[category_id] not in valid_class_names:
                    continue

            tmp_result = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": pred,
            }
            all_preds.append(tmp_result)

    return all_preds


def run_coco_or_lvis(processor, model, dataset="coco", model_tag_name="grounding_dino_tiny", device="cuda:0"):
    coco_data = load_json("./data/annotations/instances_val2017.json")
    lvis_data = load_json("./data/annotations/lvis_v1_val.json")

    labels_seen_coco = load_json("./data/annotations_created/coco_seen_classes.json")
    labels_unseen_coco = load_json("./data/annotations_created/coco_unseen_classes.json")
    labels_other_coco = load_json("./data/annotations_created/coco_other_classes.json")

    coco_categories = coco_data["categories"]
    lvis_categories = lvis_data["categories"]

    category_id_to_name_lvis = {cat["id"]: cat["name"] for cat in lvis_categories}

    coco_seen_cat_ids = {cat["id"] for cat in coco_categories if cat["name"] in labels_seen_coco}
    coco_unseen_cat_ids = {cat["id"] for cat in coco_categories if cat["name"] in labels_unseen_coco}
    coco_other_cat_ids = {cat["id"] for cat in coco_categories if cat["name"] in labels_other_coco}

    if dataset == "coco":
        gt_info = COCO("./data/annotations_created/instances_val2017_to_cocoformat.json")
        image_ids = gt_info.getImgIds()
        class_items = [{"id": cat["id"], "name": cat["name"]} for cat in coco_categories]
        out_json = f"./results_{dataset}_{model_tag_name}.json"

        preds = run_detector_on_images(
            processor=processor,
            model=model,
            gt_info=gt_info,
            image_ids=image_ids,
            image_path_fn=lambda img_info: os.path.join("./data/val2017", img_info["file_name"]),
            class_items=class_items,
            device=device,
            max_dets=300,
        )
        with open(out_json, "w") as f:
            json.dump(preds, f)

        pred_info = gt_info.loadRes(out_json)
        coco_eval = COCOeval(gt_info, pred_info, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval["precision"]

        per_class_ap = {
            cat["id"]: get_ap_for_cat(precisions, idx)
            for idx, cat in enumerate(gt_info.loadCats(gt_info.getCatIds()))
        }
        all_ap_cat = coco_seen_cat_ids.union(coco_unseen_cat_ids)
        all_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in all_ap_cat])
        seen_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in coco_seen_cat_ids])
        unseen_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in coco_unseen_cat_ids])
        other_ap = np.nanmean([per_class_ap.get(cid, np.nan) for cid in coco_other_cat_ids])

        print("COCO mAP(all):", all_ap)
        print("COCO mAP(seen):", seen_ap)
        print("COCO mAP(unseen):", unseen_ap)
        print("COCO mAP(other):", other_ap)
        return

    if dataset != "lvis":
        raise ValueError("dataset must be 'coco', 'lvis', or 'odinw13'.")

    gt_info = COCO("./data/annotations_created/lvis_minival_to_cocoformat.json")
    image_ids = gt_info.getImgIds()
    class_groups = [
        [{"id": cat["id"], "name": cat["name"]} for cat in lvis_categories[i : i + 40]]
        for i in range(0, len(lvis_categories), 40)
    ]
    valid_lvis_class_names_by_image_id = load_lvis_valid_class_names(lvis_data)

    pred_info = []
    for group_idx, class_items in enumerate(class_groups):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out_json = f"./results_{dataset}_{group_idx}_{model_tag_name}.json"
        preds = run_detector_on_images(
            processor=processor,
            model=model,
            gt_info=gt_info,
            image_ids=image_ids,
            image_path_fn=lambda img_info: os.path.join("./data/val2017", img_info["file_name"]),
            class_items=class_items,
            device=device,
            max_dets=300,
            category_id_to_name=category_id_to_name_lvis,
            valid_class_names_by_image_id=valid_lvis_class_names_by_image_id,
        )
        with open(out_json, "w") as f:
            json.dump(preds, f)

        with open(out_json, "r") as f:
            pred_info.extend(json.load(f))

    aggregated_json = f"./results_{dataset}_{model_tag_name}.json"
    with open(aggregated_json, "w") as f:
        json.dump(pred_info, f)

    lvis_gt = LVIS("./data/annotations_created/lvis_v1_minival_inserted_image_name.json")
    lvis_results = LVISResults(lvis_gt, aggregated_json, max_dets=100000)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type="bbox")
    lvis_eval.run()
    lvis_eval.print_results()
    print("")


def run_odinw(processor, model, dataset="odinw13", model_tag_name="grounding_dino_tiny", device="cuda:0"):
    odinw_configs = sorted(glob.glob(f"./data/{dataset}_config/*.yaml"))
    total_ap_list = []
    total_name_list = defaultdict(str)

    output_dir = f"{dataset}_{model_tag_name}"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    for path in odinw_configs:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dataset_name = os.path.basename(path).split(".")[0]
        print(dataset_name)

        with open(path, "r") as yml:
            config = yaml.safe_load(yml)

        test_dataset_key = str(eval(config["DATASETS"]["TEST"])[0])
        annotation_path = config["DATASETS"]["REGISTER"][test_dataset_key]["ann_file"]
        imgdir_path = config["DATASETS"]["REGISTER"][test_dataset_key]["img_dir"]
        override_category = eval(config["DATASETS"]["OVERRIDE_CATEGORY"])

        class_items = []
        for idx, content in enumerate(override_category):
            name = content["name"]
            if name == "  ":
                name = "None"
            class_items.append({"id": idx + 1, "name": name})

        gt_info = COCO(os.path.join("./data", annotation_path))
        image_ids = gt_info.getImgIds()

        out_json = os.path.join(output_dir, f"results_{dataset_name}_{dataset}_{model_tag_name}.json")
        preds = run_detector_on_images(
            processor=processor,
            model=model,
            gt_info=gt_info,
            image_ids=image_ids,
            image_path_fn=lambda img_info, imgdir_path=imgdir_path: os.path.join(
                "./data", imgdir_path, img_info["file_name"]
            ),
            class_items=class_items,
            device=device,
            max_dets=300,
        )
        with open(out_json, "w") as f:
            json.dump(preds, f)

        pred_info = gt_info.loadRes(out_json)
        coco_eval = COCOeval(gt_info, pred_info, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval["precision"]

        per_class_ap = {
            cat["id"]: get_ap_for_cat(precisions, idx)
            for idx, cat in enumerate(gt_info.loadCats(gt_info.getCatIds()))
        }
        all_ap = np.nanmean([per_class_ap[cid] for cid in per_class_ap.keys()])

        print(all_ap)
        total_ap_list.append(all_ap)
        total_name_list[dataset_name] = all_ap

    print(np.nanmean(total_ap_list))
    print(total_name_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HuggingFace OVOD models on COCO, LVIS, or ODinW13 datasets."
    )
    parser.add_argument("--dataset", default="coco", choices=["coco", "lvis", "odinw13"])
    parser.add_argument(
        "--model-name",
        default="grounding-dino-tiny",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model alias registered in MODEL_REGISTRY.",
    )
    parser.add_argument("--model-root", default="./data/huggingface")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    processor, model, tag = build_ovod_model(
        model_name=args.model_name,
        model_root=args.model_root,
        device=args.device,
    )

    if args.dataset in {"coco", "lvis"}:
        run_coco_or_lvis(
            processor=processor,
            model=model,
            dataset=args.dataset,
            model_tag_name=tag,
            device=args.device,
        )
    else:
        run_odinw(
            processor=processor,
            model=model,
            dataset=args.dataset,
            model_tag_name=tag,
            device=args.device,
        )


if __name__ == "__main__":
    main()