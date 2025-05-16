
#export PYTHONPATH=$PYTHONPATH:/scratch3/kat049/Grounded-Segment-Anything/GroundingDINO:/scratch3/kat049/Grounded-Segment-Anything/segment_anything:/scratch3/kat049/concept-graphs:/scratch3/kat049/Grounded-Segment-Anything/recognize-anything
#export GSA_PATH="/scratch3/kat049/Grounded-Segment-Anything"
import os
os.environ["GSA_PATH"] = "/scratch3/kat049/Grounded-Segment-Anything"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":/scratch3/kat049/Grounded-Segment-Anything/GroundingDINO" + ":/scratch3/kat049/Grounded-Segment-Anything/segment_anything" + ":/scratch3/kat049/concept-graphs" + ":/scratch3/kat049/Grounded-Segment-Anything/recognize-anything"


import argparse
from pathlib import Path
import re
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pickle
import gzip
import open_clip

from ultralytics import YOLO
import torch
import torchvision
from torch.utils.data import Dataset
import supervision as sv
from tqdm import trange

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from conceptgraph.utils.model_utils import compute_clip_features
import torch.nn.functional as F

from conceptgraph.scripts.generate_gsa_results import get_sam_predictor, process_tag_classes, get_sam_mask_generator, get_sam_segmentation_from_xyxy, get_sam_segmentation_dense

import torchvision.transforms as TS
from ram.models import ram
from ram.models import tag2text
from ram import inference_tag2text, inference_ram

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")

import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)
# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

device_number = 2
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

sam_variant = "sam"
save_video = True
add_bg_classes = True
accumu_classes = True
detector = "yolo"

color_path = "/scratch3/kat049/datasets/DARPA/p14_fr/results/frame00391.jpg"

for class_set in ["none", "ram"]:
    if class_set == "none":
        mask_generator = get_sam_mask_generator(sam_variant, device)
    else:
        sam_predictor = get_sam_predictor(sam_variant, device)


    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to(device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    global_classes = set()

    yolo_model_w_classes = YOLO('/scratch3/kat049/concept-graphs/yolov8l-world.pt')  # or choose yolov8m/l-world.pt


    if class_set == "none":
        classes = ['item']
        print("Skipping tagging and detection models. ")
    elif class_set == "ram":
        tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,image_size=384,vit='swin_l')
        tagging_model = tagging_model.eval().to(device)
        tagging_transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), 
                    TS.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ])
        classes = None
        print(f"{class_set} will be used to detect classes. ")
    else:
        raise ValueError("Unknown args.class_set: ", class_set)

    save_name = f"{class_set}"
    save_name += f"_{sam_variant}"
    # if save_video:
    #     video_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}.mp4"
    #     frames = []

    color_path = Path(color_path)

    vis_save_path = f"/scratch3/kat049/sam2/my_scripts/gsa_vis_{save_name}.png"
    color_path = str(color_path)
    image = cv2.imread(color_path) # This will in BGR color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
    image_pil = Image.fromarray(image_rgb)


    if class_set == "ram":
        raw_image = image_pil.resize((384, 384))
        raw_image = tagging_transform(raw_image).unsqueeze(0).to(device)

        res = inference_ram(raw_image , tagging_model)
        caption="NA"

        text_prompt=res[0].replace(' |', ',')

        with open('/scratch3/kat049/concept-graphs/conceptgraph/ram_classes_4500.txt', 'r') as file:
            lines = [line.strip() for line in file]
        lines = []
        add_classes = lines + ["other item",  "yellow 4-legged robot", "orange tank-like robot", "orange drill", "red_backpack",  "wall lamp", "entrance" "wall lamp"]
        
        remove_classes = [
            "room", "kitchen", "office", "house", "home", "building", "corner",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
            "apartment", "image", "city", "skylight", "hallway", "bureau", "modern", "salon", "doorway"
        ]

        bg_classes = ["wall", "floor", "ceiling"]

        if add_bg_classes:
            add_classes += bg_classes
        else:
            remove_classes += bg_classes

        classes = process_tag_classes(
                        text_prompt, 
                        add_classes = add_classes,
                        remove_classes = remove_classes,
                    )
        
        # add classes to global classes
    global_classes.update(classes)

    if accumu_classes:
        # Use all the classes that have been seen so far
        classes = list(global_classes)

    if class_set == "none":
        mask, xyxy, conf = get_sam_segmentation_dense(
                    sam_variant, mask_generator, image_rgb)
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=np.zeros_like(conf).astype(int),
                    mask=mask,
                )
        image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)
        annotated_image, labels = vis_result_fast(
                    image, detections, classes, instance_random_color=True)
        cv2.imwrite(vis_save_path, annotated_image)

    else:
        if detector == "dino":
            raise RuntimeError("DINO detector is not supported yet. Import _C has an error ")
        elif detector == "yolo":
            yolo_model_w_classes.set_classes(classes)
            yolo_results_w_classes = yolo_model_w_classes.predict(color_path)
            yolo_results_w_classes[0].save("/scratch3/kat049/sam2/test_YOLO.png")
            xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
            xyxy_np = xyxy_tensor.cpu().numpy()
            confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()

            detections = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
                mask=None,
            )

        if len(detections.class_id) > 0:
            ### Segment Anything ###
            detections.mask = get_sam_segmentation_from_xyxy(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=detections.xyxy
            )

            # Compute and save the clip features of detections  
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)
        else:
            image_crops, image_feats, text_feats = [], [], []

        ### Visualize results ###
        annotated_image, labels = vis_result_fast(image, detections, classes)
        cv2.imwrite(vis_save_path, annotated_image)

    # if save_video:
    #     frames.append(annotated_image)
        

    # Convert the detections to a dict. The elements are in np.array
    results = {
        "xyxy": detections.xyxy,
        "confidence": detections.confidence,
        "class_id": detections.class_id,
        "mask": detections.mask,
        "classes": classes,
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
    }

    if class_set == "ram":
        results["tagging_caption"] = caption
        results["tagging_text_prompt"] = text_prompt


    # with gzip.open(detections_save_path, "wb") as f:
    #     pickle.dump(results, f)