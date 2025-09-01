from typing import (
    Dict,
    List, 
)
import logging
from io import BytesIO
import torch
import json
from aws_helpers import helpers
from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = helpers._setup_logger(level=logging.DEBUG)

MODEL_ID = 'OpenGVLab/InternVL3-8B-hf'
S3_BUCKET = 'signal-8-flock'

# model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def create_template(json_obj: Dict) -> str:
    template = ""
    if json_obj["year"] != "None":
        template += json_obj['year'] + " "
    if json_obj["car_type"] != "None":
        template += json_obj["car_type"] + " "
    if json_obj["color"] != "None":
        template += json_obj['color'] + " "
    if json_obj["make"] != "None":
        template += json_obj["make"] + " "
    if json_obj["model"] != "None":
        template += json_obj["model"] + " "
    if json_obj["license_plate"] != "None":
        template += f"with license plate number {json_obj['license_plate']} "
    if json_obj['unique_identifiers']:
        template += f"has the following unique identifiers: " + ", ".join(json_obj["unique_identifiers"])

    return template
        
def load_image_from_s3(s3_uri: str) -> Image.Image:
    s3_client = helpers._get_s3_client()
    path = s3_uri[5:]
    bucket, key = path.split("/", 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    img_bytes = obj["Body"].read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img

def create_conversation(text):
    SYSTEM_PROMPT = None
    with open('user_prompt.txt', 'r') as f:
        USER_PROMPT = f.read()
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": USER_PROMPT})
    messages.append({"role": "assistant", "content": text})
    return {"messages": messages}

class ImageTextDataset(Dataset):
    def __init__(self, s3_bucket, json_file: str):
        self.records = []
        s3_client = helpers._get_s3_client()
        input_json_file = s3_client.get_object(Bucket=s3_bucket, Key=json_file)["Body"].read().decode('utf-8')
        input_json_file = json.loads(input_json_file)

        logger.debug(json.dumps(input_json_file["output"][0], indent=2))

        for json_obj in input_json_file['output']:
            template = create_template(json_obj)
            image_uri = json_obj["s3_uri"]
            self.records.append({"image_uri": image_uri, "text": template})

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        image = load_image_from_s3(s3_uri=record["image_uri"])
        conversation = create_conversation(record["text"])
        return {"image": image, "conversation": conversation}


logger.info("\x1b[31mCreating dataset\x1b[0m")
dataset = ImageTextDataset(json_file='created_data.json',
                           s3_bucket=S3_BUCKET)
logger.info("\x1b[32mCreated dataset\x1b[0m")
# for item in dataset:
#     print(item["conversation"]["messages"][1]["content"])
#     break