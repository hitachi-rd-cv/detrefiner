# torch==2.4.1, torchvision==0.19.0, transformers==4.56.2
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

import torch

import mobileclip


class ExtractInputFeatures:
    def __init__(self):

        # setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # mobileclip
        self.mobileclip_model, _, self.mobileclip_preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='./models/mobileclip_blt.pt')
        self.mobileclip_tokenizer = mobileclip.get_tokenizer('mobileclip_b')
        self.mobileclip_model = self.mobileclip_model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, text, alttext, flag):

        # mobileclip text feature extract
        inputs = self.mobileclip_tokenizer([text]).to(self.device)
        mobileclip_embed = self.mobileclip_model.encode_text(inputs)
        mobileclip_embed /= mobileclip_embed.norm(dim=-1, keepdim=True)

        savelist = {
            "text": text,
            "mobileclip_text_features": mobileclip_embed[0].cpu().numpy(),
        }

        return savelist


if __name__ == "__main__":

    torch.manual_seed(0)
    feature_model = ExtractInputFeatures()

    with open('./data/annotations/image_info_test2017.json') as f:
        coco_annotations = json.load(f)
    with open('./data/annotations/lvis_v1_image_info_test_dev.json') as f:
        lvis_annotations = json.load(f)

    coco_labels, lvis_labels = defaultdict(str), defaultdict(str)
    for content in coco_annotations["categories"]:
        coco_labels[str(content["name"])] = str(content["supercategory"])
    for content in lvis_annotations["categories"]:
        lvis_labels[str(content["name"])] = str(content["def"])
    
    coco_labels_keys = list(coco_labels.keys())
    lvis_labels_keys = list(lvis_labels.keys())
    savedict_coco = defaultdict(dict)
    savedict_lvis = defaultdict(dict)

    ### coco text features ###
    for idx, label in enumerate(coco_labels_keys):
        print(f"Label: {label}")
        print(f"Supercategory: {coco_labels[label]}")
        savelist = feature_model.extract(label, coco_labels[label], flag="coco")
        savedict_coco[label] = savelist
    with open("./data/textfeatures_coco.pkl","wb") as f:
        pickle.dump(dict(savedict_coco), f)

    print("---------------------------------")

    ### lvis text features ###
    for idx, label in enumerate(lvis_labels_keys):
        print(f"Label: {label}")
        print(f"Definition: {lvis_labels[label]}")
        savelist = feature_model.extract(label, lvis_labels[label], flag="lvis")
        savedict_lvis[label] = savelist
    with open("./data/textfeatures_lvis.pkl","wb") as f:
        pickle.dump(dict(savedict_lvis), f)

    print("---------------------------------")

    ### odinw13 text features ###
    os.makedirs("./data/textfeatures_odinw13", exist_ok=True)
    odinw13_configs = sorted(glob.glob("./data/odinw13_config/*.yaml"))
    for path in odinw13_configs:
        savedict_odinw13 = defaultdict(dict)
        pathname = str(path.split("/")[-1].split(".")[0])
        print(pathname)
        with open(path, 'r') as yml:
            config = yaml.safe_load(yml)
        override_category = eval(config["DATASETS"]["OVERRIDE_CATEGORY"])

        odinw_labels_keys = []
        odinw_superlabels_keys = []

        for content in override_category:
            if content["name"] == "  ":
                content["name"] = "None"
            if content["supercategory"] == "VOC":
                content["supercategory"] = content["name"]
            odinw_labels_keys.append(content["name"])
            odinw_superlabels_keys.append(content["supercategory"])

        for label, supercategory in zip(odinw_labels_keys, odinw_superlabels_keys):
            print(f"Label: {label}")
            print(f"Supercategory: {supercategory}")
            savelist = feature_model.extract(label, supercategory, flag="coco")
            savedict_odinw13[label] = savelist
        with open("./data/textfeatures_odinw13/textfeatures_"+str(pathname)+".pkl","wb") as f:
            pickle.dump(dict(savedict_odinw13), f)

        print("---------------------------------")