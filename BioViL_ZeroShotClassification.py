import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import List, Tuple, Union
import torch
import numpy as np
import pandas as pd
import random

from src.health_multimodal.image import ImageInferenceEngine
from src.health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from src.health_multimodal.image.model.model import get_biovil_resnet, \
    get_biovil_resnet_from_download
from src.health_multimodal.text.utils import get_cxr_bert_inference
from src.health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 512

#################################################################
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize BioViL model
def _get_vlp_inference_engine() -> ImageTextInferenceEngine:
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_resnet(pretrained=True),
        transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE))
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=get_cxr_bert_inference(),
    )
    return img_txt_inference

# Get pneumonia prompts
def _get_default_text_prompts_for_pneumonia() -> Tuple[List, List]:
    """
    Get the default text prompts for presence and absence of pneumonia
    """
    pos_query = ['Findings consistent with pneumonia', 'Findings suggesting pneumonia',
                 'This opacity can represent pneumonia', 'Findings are most compatible with pneumonia']
    neg_query = ['There is no pneumonia', 'No evidence of pneumonia',
                 'No evidence of acute pneumonia', 'No signs of pneumonia']
    return pos_query, neg_query

##########################################################################################################
##########################################################################################################
# Load RSNA pneumonia dataset: Zero-shot classification
# Read RSNA_pneumonia dataset
# RSNA-zscls.csv: 30% validation 8006 samples
df = pd.read_csv('/mntnfs/med_data2/yuzhouhuang/JIHT_v1/RSNA-zscls.csv')
print(df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BioViL model
img_txt_inference = _get_vlp_inference_engine()

# Generate medical class prompt
positive_prompts, negative_prompts = _get_default_text_prompts_for_pneumonia()

# Load RSNA images
processed_imgs_path = df['Path'].tolist()

# Classification accuracy
RSNA_COMPETITION_TASKS = [
    'Pneumonia',
    'No Pneumonia'
]

# Compute scores
prediction = []
for i in range(len(processed_imgs_path)):
    print('Image', i)
    img_path = Path(processed_imgs_path[i])
    positive_score = img_txt_inference.get_similarity_score_from_raw_data(
        image_path=img_path,
        query_text=positive_prompts)
    negative_score = img_txt_inference.get_similarity_score_from_raw_data(
        image_path=img_path,
        query_text=negative_prompts)
    print('positive:', positive_score)
    print('negative:', negative_score)

    if positive_score >= negative_score:
        # Pneumonia == 0
        prediction.append(0)
    else:
        # No Pneumonia == 1
        prediction.append(1)

# Compute Zero-shot classification
pred = np.array(prediction)
labels = df[RSNA_COMPETITION_TASKS].to_numpy().argmax(axis=1)
acc = len(labels[labels == pred]) / len(labels)
print(acc)
# 0.6353463968304958 ->> 70% training 18678 samples
# 0.6468898326255309 ->> 30% validation 8006 samples
