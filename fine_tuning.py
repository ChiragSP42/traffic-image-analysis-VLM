import boto3
import os
import pandas as pd
from datasets import load_dataset
import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, TrainingArguments, Trainer
from dotenv import load_dotenv
from utils.helpers import _local_or_sagemaker
load_dotenv()

RUNNING_LOCALLY = False

if not _local_or_sagemaker():
    RUNNING_LOCALLY = True

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

if not RUNNING_LOCALLY:
    S3_BUCKET = 'signal-8-data-creation-testing'
    IMAGE_FOLDER = 'test_images'

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name='us-east-1')

if RUNNING_LOCALLY:
    dataset = load_dataset("csv", data_files='input.csv')["train"]
else:
    dataset =load_dataset("csv", data_files=f"{S3_BUCKET}")

class CLIPForContrastiveLearning(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, ground_truth=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  pixel_values=pixel_values,
                                  return_dict=True)
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        

        if not ground_truth:
            ground_truth = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

        loss_image = self.loss_fn(logits_per_image, ground_truth)
        loss_text = self.loss_fn(logits_per_text, ground_truth)
        loss = (loss_image + loss_text) / 2

        if self.training:
            return {"loss": loss, **outputs}
        else:
            return outputs
        
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32", token = HUGGINGFACE_ACCESS_TOKEN)        
model = CLIPForContrastiveLearning.from_pretrained("openai/clip-vit-base-patch32", config = config, token = HUGGINGFACE_ACCESS_TOKEN)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token = HUGGINGFACE_ACCESS_TOKEN)





