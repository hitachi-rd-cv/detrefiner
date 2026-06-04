import json
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import shutil
import tqdm


os.makedirs("./data/visionfeatures_train2017_lvis/", exist_ok=True)
os.makedirs("./data/gt_train2017_lvis/", exist_ok=True)


with open("./data/annotations/lvis_v1_train.json", 'r') as f:
    lvis_data = json.load(f)
filename_to_image_id_lvis = {img["coco_url"].split("/")[-1]: img["id"] for img in lvis_data["images"] if "train" in img["coco_url"]}

for filename in tqdm.tqdm(sorted(list(filename_to_image_id_lvis.keys()))):
    shutil.copy("./data/visionfeatures_train2017/"+filename+".pkl", "./data/visionfeatures_train2017_lvis/"+filename+".pkl")
    shutil.copy("./data/gt_train2017/"+filename+".pkl", "./data/gt_train2017_lvis/"+filename+".pkl")