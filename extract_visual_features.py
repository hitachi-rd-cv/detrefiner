# torch==2.4.1, torchvision==0.19.0, transformers==4.56.2
from collections import defaultdict
import gc
import glob
import json
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pickle
import warnings
warnings.simplefilter('ignore')
import yaml

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

import mobileclip


class ExtractInputFeatures:
    def __init__(self):

        # setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # mobileclip
        self.mobileclip_model, _, self.mobileclip_preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='./models/mobileclip_blt.pt', device=self.device)
        self.mobileclip_tokenizer = mobileclip.get_tokenizer('mobileclip_b')
        self.mobileclip_model = self.mobileclip_model.to(self.device).eval()

        # dinov3
        self.dinov3_processor = AutoImageProcessor.from_pretrained('./models/facebook/dinov3-vitb16-pretrain-lvd1689m')
        self.dinov3_model = AutoModel.from_pretrained('./models/facebook/dinov3-vitb16-pretrain-lvd1689m')
        self.dinov3_model = self.dinov3_model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, image):
                
        # mobileclip visual feature extract
        image_mobileclip = self.mobileclip_preprocess(image).unsqueeze(0).to(self.device)
        mobileclip_image_features = self.mobileclip_model.encode_image(image_mobileclip)
        mobileclip_image_features = mobileclip_image_features/mobileclip_image_features.norm(dim=-1, keepdim=True)

        # dinov3 visual feature extract
        inputs = self.dinov3_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.dinov3_model(**inputs)
        dinov3_embed = outputs.last_hidden_state

        savelist = {
            "mobileclip_visual_features": mobileclip_image_features.cpu().numpy(),
            "dinov3_visual_features": dinov3_embed.cpu().numpy(),
        }

        return savelist


if __name__ == "__main__":

    torch.manual_seed(0)
    feature_model = ExtractInputFeatures()

    ### coco & lvis visual features ###
    coco_lvis_list = ["train2017", "val2017"]
    for dataset in coco_lvis_list:
        OUTPUT_DIR = "./data/visionfeatures_" + dataset
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        imagefolder = sorted(glob.glob("./data/"+dataset+"/*"))
        for idx, image_path in enumerate(imagefolder):
            print(f"File: {image_path}")
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            savelist = feature_model.extract(image)

            with open(os.path.join(OUTPUT_DIR, str(image_path.split("/")[-1])+".pkl"),"wb") as f:
                pickle.dump(savelist, f)

            if idx%10000 == 0:
                del image, savelist
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    ### odinw13 visual features ###
    odinw_configs_list = [
        ("./data/visionfeatures_odinw13", sorted(glob.glob("./data/odinw13_config/*.yaml"))), 
        ]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for (base_dir, odinw_configs) in odinw_configs_list:
        for path in odinw_configs:
            print(str(path.split("/")[-1].split(".")[0]))
            pathname = str(path.split("/")[-1].split(".")[0])
            with open(path, 'r') as yml:
                config = yaml.safe_load(yml)

            imgdir_path = config["DATASETS"]["REGISTER"][str(eval(config["DATASETS"]["TEST"])[0])]["img_dir"]
            annotation_path = config["DATASETS"]["REGISTER"][str(eval(config["DATASETS"]["TEST"])[0])]["ann_file"]
            with open(os.path.join("./data", annotation_path), 'r') as f:
                odinw_data = json.load(f)
            filenames_set = {img["file_name"] for img in odinw_data["images"]}
            imagefolder = [p for p in glob.glob(os.path.join("./data", imgdir_path, "*")) if p.lower().endswith(exts)]
                        
            OUTPUT_DIR = os.path.join(base_dir, imgdir_path) 
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            for idx, image_path in enumerate(imagefolder):
                if image_path.split("/")[-1] in filenames_set:
                    print(f"File: {image_path}")
                    with Image.open(image_path) as img:
                        image = img.convert("RGB")
                    savelist = feature_model.extract(image)
                    new_image_path = "/".join(image_path.split("/")[2:])

                    with open(os.path.join(base_dir, str(new_image_path)+".pkl"),"wb") as f:
                        pickle.dump(savelist, f)
            print("----------------------------")