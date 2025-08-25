from aws_helpers.helpers import _local_or_sagemaker, list_obj_s3, _get_s3_client
import os
import torch
import boto3
from torch import nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import random
import json
import time
from PIL import Image
from io import BytesIO
from typing import Any, Optional
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
MODEL_ID = "openai/clip-vit-base-patch32"
S3_BUCKET = 'signal-8-data-creation-testing'
# S3_BUCKET = 'bravo-foxtrot-data'
INPUT_FILE = 'created_data.json'
TRAIN_JSON_FILE = 'train.json'
TEST_JSON_FILE = 'test.json'
IMAGE_FOLDER = 'Data'
# IMAGE_FOLDER = 'bravo_foxtrot_images_data'
OUTPUT_DIR = ''
CHECKPOINT_DIR = ''

train_dataset = None
test_dataset = None

def preprocess_sample(sample):
    # print(type(sample["s3_uri"]))
    # print(sample["s3_uri"][0])
    def _load_image_from_s3(filename: str):
        # print(filename)
        path = filename.replace("s3://", "")
        # print(path)
        bucket, key = path.split('/', 1)

        s3_client = _get_s3_client()

        response = s3_client.get_object(Bucket=bucket,
                                            Key=key)
        image_bytes = response["Body"].read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    
    if isinstance(sample['year'], list):
        # Batched processing
        descriptions = []
        all_images = []
        
        for i in range(len(sample['year'])):
            description = f"A {sample['year'][i]} {sample['car_type'][i]} {sample['color'][i]} {sample['make'][i]} {sample['model'][i]} with license plate number {sample['license_plate'][i]} has the following unique identifiers {sample['unique_identifiers'][i]}"
            descriptions.append(description)
            
            # Load images for this sample
            images = [_load_image_from_s3(image) for image in sample["s3_uri"]]
            all_images = images
        
        # Process all texts and images together
        preprocessed = processor(text=descriptions,
                                images=all_images,
                                padding='max_length',
                                return_tensors='pt',
                                truncation=True)
        
    else:
        # Single sample processing (fallback)
        description = f"A {sample['year']} {sample['car_type']} {sample['color']} {sample['make']} {sample['model']} with license plate number {sample['license_plate']} has the following unique identifiers {sample['unique_identifiers']}"
        images = [_load_image_from_s3(image) for image in sample["s3_uri"]]
        
        preprocessed = processor(text=description,
                                images=images,
                                padding='max_length',
                                return_tensors='pt',
                                truncation=True)

    images = [_load_image_from_s3(image) for image in sample["s3_uri"]]

    if processor is None:
        raise ValueError("Processor is None. Please provide a valid processor before calling preprocess.")
    
    # print(f"Shape of input_ids: {preprocessed['input_ids'].shape}")
    # print(f"Shape of input_ids squeezed: {preprocessed['input_ids'].squeeze(0).shape}")

    return {"input_ids": preprocessed['input_ids'],
            "attention_mask": preprocessed['attention_mask'],
            "pixel_values": preprocessed['pixel_values']
        }

class CLIPForContrastiveLearning(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None): # type: ignore
        outputs = super().forward(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                return_dict=True) # type: ignore
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        

        if not labels:
            labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

        loss_image = self.loss_fn(logits_per_image, labels)
        loss_text = self.loss_fn(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        return {"loss": loss, **outputs}

class CustomLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if logs:
                if 'loss' in logs:
                    print(f"Train loss: {logs['loss']}")
                if 'eval_loss' in logs:
                    print(f"Eval loss: {logs['eval_loss']}")
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"Evaluation metrics: {metrics}")

print("\x1b[31mInitializing CLIP model and processor\x1b[0m")    
login(token=HUGGINGFACE_ACCESS_TOKEN)    
config = CLIPConfig.from_pretrained(MODEL_ID)        
model = CLIPForContrastiveLearning.from_pretrained(MODEL_ID, config = config)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
print("\x1b[32mInitialized CLIP model and processor\x1b[0m")   
    
local_sagemaker = _local_or_sagemaker()
if local_sagemaker == False:
    print("\x1b[32mRunning locally\x1b[0m")
    OUTPUT_DIR = 'outputs/model/'
    CHECKPOINT_DIR = 'outputs/model/checkpoint-last'
    train_dataset = load_dataset('json', data_files='train.json', split='train', field='output', streaming=False)
    test_dataset = load_dataset('json', data_files='test.json', split='train', field='output', streaming=False)
else:
    print("\x1b[32mRunning on SageMaker\x1b[0m")
    OUTPUT_DIR = '/opt/ml/model'
    CHECKPOINT_DIR = '/opt/ml/model/checkpoint-last'
    train_json_s3_path = f"s3://{S3_BUCKET}/{TRAIN_JSON_FILE}"
    test_json_s3_path = f"s3://{S3_BUCKET}/{TEST_JSON_FILE}"

    print("\x1b[33mChecking if Train and Test JSONs exist in S3\x1b[0m")
    response = list_obj_s3(s3_client=_get_s3_client(),
                        bucket_name=S3_BUCKET,
                        folder_name='')
    if 'train.json' in response and 'test.json' in response:
        print("\x1b[32mTrain and test JSON files present, loading datasets.\x1b[0m")
        train_dataset = load_dataset('json', data_files=train_json_s3_path, field='output', streaming=True, split='train')
        test_dataset = load_dataset('json', data_files=test_json_s3_path, field='output', streaming=True, split='train')
    elif 'train.json' not in response and 'test.json' not in response:
        print("\x1b[33mTrain and test JSON files not present, checking if original created_data.json file is present\x1b[0m")
        if 'created_data.json' in response:
            print("\x1b[32mOriginal JSON file present, creating train and test JSON files.\x1b[0m")
            s3_client = _get_s3_client()
            response_binary = s3_client.get_object(Bucket=S3_BUCKET,
                                            Key="created_data.json")["Body"]
            response = response_binary.read().decode("utf-8")
            train_list = []
            test_list = []
            for obj in json.loads(response)["output"]:
                if random.random() < 0.8:
                    train_list.append(obj)
                else:
                    test_list.append(obj)
            train_json = {"output" : train_list}
            test_json = {"output": test_list}
            s3_client.put_object(Bucket=S3_BUCKET,
                                Key='train.json',
                                Body=json.dumps(train_json, indent=2),
                                ContentType='application/json')
            s3_client.put_object(Bucket=S3_BUCKET,
                                Key='test.json',
                                Body=json.dumps(test_json, indent=2),
                                ContentType='application/json')
            time.sleep(1)
            train_dataset = load_dataset('json', data_files=train_json_s3_path, split='train', field='output', streaming=True)
            test_dataset = load_dataset('json', data_files=test_json_s3_path, split='train', field='output', streaming=True)
            print(f"\x1b[32mCreated Train and Test JSON files and uploaded to S3 bucket {S3_BUCKET}\x1b[0m")
        else:
            raise FileNotFoundError("\x1b[31mOriginal JSON not present in S3 bucket\x1b[0m")
        

train_stream = train_dataset.map(preprocess_sample, batched=False, batch_size = 4) # type: ignore
test_stream = test_dataset.map(preprocess_sample, batched=True, batch_size = 4) # type: ignore

print("\x1b[31mSetting up Training Arguments\x1b[0m")
num_epochs = 10
length_of_dataset = len(list_obj_s3(s3_client=_get_s3_client(),
                                    bucket_name=S3_BUCKET,
                                    folder_name=IMAGE_FOLDER))

batch_size = 16
steps_per_epoch = (length_of_dataset + batch_size - 1) // batch_size

training_arguments = {
    "output_dir": OUTPUT_DIR,
    "learning_rate": 2e-5,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "logging_strategy": "steps",
    "max_steps": steps_per_epoch * num_epochs,
    "eval_steps": steps_per_epoch * 0.25,
    "logging_steps": 10,
    "seed": 42,
    "fp16": True,
    "load_best_model_at_end": False,
    "disable_tqdm": True
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(**training_arguments)
# training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,  # SageMaker default model directory for saving artifacts
#         learning_rate=2e-5,
#         eval_strategy="steps",
#         save_strategy="steps",
#         logging_strategy="steps",
#         max_steps = steps_per_epoch * num_epochs,
#         eval_steps = steps_per_epoch * 0.25,
#         logging_steps=10,
#         seed=42,
#         fp16=True,  # use mixed precision if supported
#         load_best_model_at_end=False,
#         disable_tqdm=True
#     )

print("\x1b[31mSetting up Trainer\x1b[0m")
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_stream, # type: ignore
        eval_dataset=test_stream, # type: ignore
        callbacks=[CustomLoggingCallback()]
    )

print("\x1b[31mStarting to train model\x1b[0m")

if os.path.exists(CHECKPOINT_DIR):
    trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)
else:
    trainer.train()

# Save final model (redundant as Trainer saves models during save_strategy)
trainer.save_model("/opt/ml/model")