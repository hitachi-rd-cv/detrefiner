import argparse
from collections import defaultdict
import gc
import glob
import json
import os
import shutil

os.environ["CURL_CA_BUNDLE"] = ""

import pickle
import warnings

warnings.simplefilter("ignore")

import tqdm
import yaml
from PIL import Image
import numpy as np
import torch

# mmengine
from mmengine.structures import InstanceData

# GLIP
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVIS, LVISEval, LVISResults


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_glip_model(model_size="tiny", device="cuda:0", glip_resize=800, box_thr=0.0):
    if model_size == "tiny":
        det_config = "configs/glip_Swin_T_O365_GoldG.yaml"
        det_weight = "models/glip_tiny_model_o365_goldg_cc_sbu.pth"
    elif model_size == "large":
        det_config = "configs/glip_Swin_L.yaml"
        det_weight = "models/glip_large_model.pth"
    else:
        raise ValueError(f"Unsupported model_size: {model_size}")

    cfg.merge_from_file(det_config)
    cfg.merge_from_list(["MODEL.WEIGHT", det_weight])
    cfg.merge_from_list(["MODEL.DEVICE", device])
    det_model = GLIPDemo(
        cfg,
        min_image_size=glip_resize,
        confidence_threshold=box_thr,
        show_mask_heatmaps=False,
    )
    det_model.model = det_model.model.to(device)
    det_model.device = device
    return det_model


def get_ap_for_cat(precisions, cat_idx):
    precision = precisions[:, :, cat_idx, 0, -1]
    precision = precision[precision > -1]  # remove nan
    return np.mean(precision) if precision.size else float("nan")


def run_coco_or_lvis(det_model, dataset="coco", model_size="tiny"):
    coco_data = load_json("./data/annotations/instances_val2017.json")
    lvis_data = load_json("./data/annotations_created/lvis_v1_minival_inserted_image_name.json")

    labels_seen_coco = load_json("./data/annotations_created/coco_seen_classes.json")
    labels_unseen_coco = load_json("./data/annotations_created/coco_unseen_classes.json")
    labels_other_coco = load_json("./data/annotations_created/coco_other_classes.json")

    coco_seen_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_seen_coco}
    coco_unseen_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_unseen_coco}
    coco_other_cat_ids = {cat["id"] for cat in coco_data["categories"] if cat["name"] in labels_other_coco}

    category_id_to_name_coco = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    category_id_to_name_lvis = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

    name_to_category_id_coco = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
    name_to_category_id_lvis = {cat["name"]: cat["id"] for cat in lvis_data["categories"]}

    image_id_to_filename_lvis = {
        img["id"]: img["coco_url"].split(os.sep)[-1]
        for img in lvis_data["images"]
        if "val" in img["coco_url"]
    }
    filename_to_image_id_coco = {img["file_name"]: img["id"] for img in coco_data["images"]}
    filename_to_image_id_lvis = {
        img["coco_url"].split(os.sep)[-1]: img["id"]
        for img in lvis_data["images"]
        if "val" in img["coco_url"]
    }

    if dataset == "coco":
        gt_info = COCO("./data/annotations_created/instances_val2017_to_cocoformat.json")
        iterations = 1
    elif dataset == "lvis":
        gt_info = COCO("./data/annotations_created/lvis_v1_minival_inserted_image_name.json")
        iterations = 31
    else:
        raise ValueError("dataset must be 'coco', 'lvis', or an ODINW dataset name such as 'odinw13'.")

    new_predicted_bboxes = []
    image_ids = gt_info.getImgIds()

    gts = defaultdict(list)
    gt_list = sorted(glob.glob("./data/gt_val2017/*.pkl"))
    for fpath in gt_list:
        fname = fpath.split(os.sep)[-1].split(".")[0]
        sfx = fpath.split(os.sep)[-1].split(".")[1]
        with open(fpath, "rb") as f:
            gts[str(fname) + "." + sfx] = pickle.load(f)

    for i in range(iterations):

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if dataset == "coco":
            class_names = list(category_id_to_name_coco.values())
        else:
            class_names = list(category_id_to_name_lvis.values())[i * 40 : (i + 1) * 40]

        det_text_prompt = " .".join(class_names).lower().strip().replace("_", " ").replace("-", " ").replace("(", "").replace(")", "")
        if not det_text_prompt.endswith("."):
            det_text_prompt += " ."
        print(det_text_prompt)

        for image_id in tqdm.tqdm(image_ids):
            img_info = gt_info.loadImgs(image_id)[0]
            if dataset == "coco":
                image_path = os.path.join("./data/val2017", img_info["file_name"])
            else:
                image_path = os.path.join("./data", img_info["file_name"])

            image_new = Image.open(image_path).convert("RGB")
            top_predictions = det_model.inference(
                np.asarray(image_new)[:, :, ::-1], det_text_prompt, use_other_text=False
            )

            pred_instances = InstanceData()
            pred_instances.bboxes = top_predictions.bbox
            pred_instances.labels = top_predictions.get_field("labels") - 1
            pred_instances.scores = top_predictions.get_field("scores")

            filename = image_path.split(os.sep)[-1]
            bboxes = pred_instances.bboxes.detach().numpy().astype("float64")
            scores = pred_instances.scores.detach().numpy().astype("float64")
            cids = pred_instances.labels.detach().numpy().astype("int64")

            preds = []
            for cid, bbox, score in zip(cids, bboxes, scores):
                if dataset == "coco":
                    category_id = name_to_category_id_coco[class_names[cid]]
                else:
                    category_id = name_to_category_id_lvis[class_names[cid]]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                if score >= 0:
                    preds.append((category_id, bbox, score))
            preds.sort(key=lambda x: x[2], reverse=True)
            preds = preds[:300]

            if dataset == "coco":
                result_image_id = filename_to_image_id_coco[filename]
            else:
                result_image_id = filename_to_image_id_lvis[filename]

            for category_id, bbox, score in preds:
                if dataset == "coco":
                    new_predicted_bboxes.append(
                        {
                            "image_id": result_image_id,
                            "category_id": category_id,
                            "bbox": bbox.tolist(),
                            "score": float(score),
                        }
                    )
                else:
                    valid_lvis_classes = (
                        gts[image_id_to_filename_lvis[result_image_id]]["lvis_labels"]
                        + gts[image_id_to_filename_lvis[result_image_id]]["lvis_neg_category_ids"]
                    )
                    if category_id_to_name_lvis[category_id] in valid_lvis_classes:
                        new_predicted_bboxes.append(
                            {
                                "image_id": result_image_id,
                                "category_id": category_id,
                                "bbox": bbox.tolist(),
                                "score": float(score),
                            }
                        )

        iter_out_json = f"./results_{dataset}_{i}_glip_{model_size}.json"
        with open(iter_out_json, "w") as f:
            json.dump(new_predicted_bboxes, f)

    out_json = f"./results_{dataset}_glip_{model_size}.json"
    with open(out_json, "w") as f:
        json.dump(new_predicted_bboxes, f)

    if dataset == "coco":
        gt_info = COCO("./data/annotations_created/instances_val2017_to_cocoformat.json")
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
    else:
        gt_info = LVIS("./data/annotations_created/lvis_v1_minival_inserted_image_name.json")
        lvis_results = LVISResults(gt_info, out_json, max_dets=100000)
        lvis_eval = LVISEval(gt_info, lvis_results, iou_type="bbox")
        lvis_eval.run()
        lvis_eval.print_results()
        print("")


def run_odinw(det_model, dataset="odinw13", model_size="tiny"):
    odinw_configs = sorted(glob.glob(f"./data/{dataset}_config/*.yaml"))
    total_ap_list = []
    total_name_list = defaultdict(str)
    OUTPUT_DIR = str(dataset)+"_glip_"+str(model_size)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in odinw_configs:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_predicted_bboxes = []
        dataset_name = path.split("/")[-1].split(".")[0]
        print(dataset_name)

        with open(path, "r") as yml:
            config = yaml.safe_load(yml)

        test_dataset_key = str(eval(config["DATASETS"]["TEST"])[0])
        annotation_path = config["DATASETS"]["REGISTER"][test_dataset_key]["ann_file"]
        imgdir_path = config["DATASETS"]["REGISTER"][test_dataset_key]["img_dir"]
        override_category = eval(config["DATASETS"]["OVERRIDE_CATEGORY"])

        text_prompt = ""
        for content in override_category:
            if content["name"] == "  ":
                content["name"] = "None"
            text_prompt += content["name"] + " . "
        text_prompt = text_prompt[:-2]

        gt_info = COCO(os.path.join("./data", annotation_path))
        text_prompt_id_to_category_id = {i: i + 1 for i in range(len(override_category))}

        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.replace("_", " ").replace("-", " ").replace("(", "").replace(")", "")

        image_ids = gt_info.getImgIds()
        for image_id in tqdm.tqdm(image_ids):
            img_info = gt_info.loadImgs(image_id)[0]
            image_path = os.path.join("./data", imgdir_path, img_info["file_name"])
            image_new = Image.open(image_path).convert("RGB")

            top_predictions = det_model.inference(
                np.asarray(image_new)[:, :, ::-1], text_prompt, use_other_text=False
            )

            pred_instances = InstanceData()
            pred_instances.bboxes = top_predictions.bbox
            pred_instances.labels = top_predictions.get_field("labels") - 1
            pred_instances.scores = top_predictions.get_field("scores")

            bboxes = pred_instances.bboxes.detach().numpy().astype("float64")
            scores = pred_instances.scores.detach().numpy().astype("float64")
            cids = pred_instances.labels.detach().numpy().astype("int64")

            preds = []
            for cid, bbox, score in zip(cids, bboxes, scores):
                category_id = text_prompt_id_to_category_id[cid]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                if score >= 0:
                    preds.append((category_id, bbox, score))
            preds.sort(key=lambda x: x[2], reverse=True)
            preds = preds[:300]

            for category_id, bbox, score in preds:
                new_predicted_bboxes.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox.tolist(),
                        "score": float(score),
                    }
                )

        out_json = f"results_{dataset_name}_{dataset}_glip_{model_size}.json"
        with open(os.path.join(OUTPUT_DIR, out_json), "w") as f:
            json.dump(new_predicted_bboxes, f)

        pred_info = gt_info.loadRes(os.path.join(OUTPUT_DIR, out_json))
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
    parser = argparse.ArgumentParser(description="Evaluate GLIP on COCO, LVIS, or ODINW datasets.")
    parser.add_argument("--dataset", default="coco", choices=["coco", "lvis", "odinw13"])
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "large"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-thr", type=float, default=0.0)
    parser.add_argument("--glip-resize", type=int, default=800)
    return parser.parse_args()


def main():
    args = parse_args()
    det_model = build_glip_model(
        model_size=args.model_size,
        device=args.device,
        glip_resize=args.glip_resize,
        box_thr=args.box_thr,
    )

    if args.dataset in {"coco", "lvis"}:
        run_coco_or_lvis(det_model, dataset=args.dataset, model_size=args.model_size)
    else:
        run_odinw(det_model, dataset=args.dataset, model_size=args.model_size)


if __name__ == "__main__":
    main()
