from collections import defaultdict
import glob
import json
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pickle
import warnings
warnings.simplefilter('ignore')
import yaml


if __name__ == "__main__":

    ### coco&lvis gt generation ###
    coco_lvis_list = [
        ("train", "./data/gt_train2017"), 
        ("val", "./data/gt_val2017")
        ]
    for (MODE, OUTPUT_DIR) in coco_lvis_list:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open("./data/annotations/instances_"+str(MODE)+"2017.json", 'r') as f:
            coco_data = json.load(f)
        with open("./data/annotations/lvis_v1_"+str(MODE)+".json", 'r') as f:
            lvis_data = json.load(f)

        category_id_to_name_coco = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        category_id_to_name_lvis = {cat["id"]: cat["name"] for cat in lvis_data["categories"]}

        image_id_to_filename_coco = {img["id"]: img["file_name"] for img in coco_data["images"]}
        image_id_to_filename_lvis = {img["id"]: img["coco_url"].split("/")[-1] for img in lvis_data["images"] if str(MODE) in img["coco_url"]}

        image_annotations = defaultdict(dict)

        ### image_size (width, height), other list
        for img in coco_data["images"]:

            width, height = img["width"], img["height"]
            image_id = img["id"]
            file_name = image_id_to_filename_coco[image_id]

            image_annotations[file_name]["file_name"] = file_name
            image_annotations[file_name]["image_size"] = (width, height)
            image_annotations[file_name]["coco_labels"] = []
            image_annotations[file_name]["coco_bboxes"] = []
            image_annotations[file_name]["lvis_labels"] = []
            image_annotations[file_name]["lvis_bboxes"] = []
            image_annotations[file_name]["lvis_neg_category_ids"] = []
            image_annotations[file_name]["lvis_not_exhaustive_category_ids"] = []

        ### lvis_neg_category_ids, lvis_not_exhaustive_category_ids
        for img in lvis_data["images"]:

            if MODE=="train" or (MODE=="val" and img["id"] in list(image_id_to_filename_lvis.keys())):

                image_id = img["id"]
                file_name = image_id_to_filename_lvis[image_id]

                for category_id in img["neg_category_ids"]:
                    image_annotations[file_name]["lvis_neg_category_ids"].append(category_id_to_name_lvis[category_id])

                for category_id in img["not_exhaustive_category_ids"]:
                    image_annotations[file_name]["lvis_not_exhaustive_category_ids"].append(category_id_to_name_lvis[category_id])

        ### coco_labels, coco_bboxes
        for ann in coco_data["annotations"]:

            image_id = ann["image_id"]
            file_name = image_id_to_filename_coco[image_id]
            category_id = ann["category_id"]
            category_name = category_id_to_name_coco[category_id]
            bbox = ann["bbox"]  # [x, y, width, height]

            image_annotations[file_name]["coco_labels"].append(category_name)
            image_annotations[file_name]["coco_bboxes"].append([category_name, bbox])

        ### lvis_labels, lvis_bboxes
        for ann in lvis_data["annotations"]:

            if MODE=="train" or (MODE=="val" and ann["image_id"] in list(image_id_to_filename_lvis.keys())):

                image_id = ann["image_id"]
                file_name = image_id_to_filename_lvis[image_id]
                category_id = ann["category_id"]
                category_name = category_id_to_name_lvis[category_id]
                bbox = ann["bbox"]  # [x, y, width, height]
                
                image_annotations[file_name]["lvis_labels"].append(category_name)
                image_annotations[file_name]["lvis_bboxes"].append([category_name, bbox])

        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        image_annotations_keys = sorted(list(image_annotations.keys()))
        for idx, file_name in enumerate(image_annotations_keys):
            if len(image_annotations[file_name]['lvis_labels'])!=0:
                count1+=1
            if len(image_annotations[file_name]['lvis_bboxes'])!=0:
                count2+=1
            if len(image_annotations[file_name]["lvis_neg_category_ids"])!=0:
                count3+=1
            if len(image_annotations[file_name]["lvis_not_exhaustive_category_ids"])!=0:
                count4+=1
            with open(os.path.join(OUTPUT_DIR, str(file_name) + ".pkl"),"wb") as f:
                pickle.dump(dict(image_annotations[file_name]), f)
        print(count1)   # train: 99338, val: 4752
        print(count2)   # train: 99338, val: 4752
        print(count3)   # train: 100170, val: 4805
        print(count4)   # train: 26213, val: 919

    ### odinw13 gt generation ###
    odinw_configs_list = [
        ("./data/gt_odinw13/", sorted(glob.glob("./data/odinw13_config/*.yaml")))
        ]
    for (base_dir, odinw_configs) in odinw_configs_list:
        for path in odinw_configs:
            with open(path, 'r') as yml:
                config = yaml.safe_load(yml)

            annotation_path = config["DATASETS"]["REGISTER"][str(eval(config["DATASETS"]["TEST"])[0])]["ann_file"]
            imgdir_path = config["DATASETS"]["REGISTER"][str(eval(config["DATASETS"]["TEST"])[0])]["img_dir"]
            override_category = eval(config["DATASETS"]["OVERRIDE_CATEGORY"])

            print(annotation_path)

            OUTPUT_DIR = base_dir + imgdir_path
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            with open(os.path.join("./data/", annotation_path), 'r') as f:
                odinw_data = json.load(f)

            category_id_to_name_odinw = {cat["id"]: cat["name"] for cat in odinw_data["categories"]}
            image_id_to_filename_odinw = {img["id"]: img["file_name"] for img in odinw_data["images"]}
            image_annotations = defaultdict(dict)

            ### image_size (width, height), other list
            for img in odinw_data["images"]:

                width, height = img["width"], img["height"]
                image_id = img["id"]
                file_name = image_id_to_filename_odinw[image_id]

                image_annotations[file_name]["file_name"] = file_name
                image_annotations[file_name]["image_size"] = (width, height)
                image_annotations[file_name]["coco_labels"] = []
                image_annotations[file_name]["coco_bboxes"] = []
                image_annotations[file_name]["lvis_labels"] = []
                image_annotations[file_name]["lvis_bboxes"] = []
                image_annotations[file_name]["lvis_neg_category_ids"] = []
                image_annotations[file_name]["lvis_not_exhaustive_category_ids"] = []

            ### odinw_labels, odinw_bboxes
            for ann in odinw_data["annotations"]:

                image_id = ann["image_id"]
                file_name = image_id_to_filename_odinw[image_id]
                category_id = ann["category_id"]
                category_name = category_id_to_name_odinw[category_id]
                bbox = ann["bbox"]  # [x, y, width, height]

                image_annotations[file_name]["coco_labels"].append(category_name)
                image_annotations[file_name]["coco_bboxes"].append([category_name, bbox])

            image_annotations_keys = sorted(list(image_annotations.keys()))
            for idx, file_name in enumerate(image_annotations_keys):
                with open(os.path.join(OUTPUT_DIR, str(file_name) + ".pkl"),"wb") as f:
                    pickle.dump(dict(image_annotations[file_name]), f)