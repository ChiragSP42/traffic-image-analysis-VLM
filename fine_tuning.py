import boto3
import os
import pandas as pd
from datasets import load_dataset
import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, TrainingArguments, Trainer, TrainerCallback
from dotenv import load_dotenv
from aws_helpers.helpers import _local_or_sagemaker
from PIL import Image
from io import BytesIO
from huggingface_hub import login
load_dotenv()

# Defining GLOBAL VARIABLES

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
MODEL_ID = "openai/clip-vit-base-patch32"
S3_BUCKET = 'signal-8-data-creation-testing'
INPUT_FILE = 'input.csv'
IMAGE_FOLDER = 'Data'
OUTPUT_DIR = ''
CHECKPOINT_DIR = ''
RUNNING_LOCALLY = None

# Are you running the script locally or as a training job.
if not _local_or_sagemaker():
    print("\x1b[32mRunning locally\x1b[0m")
    RUNNING_LOCALLY = True
    OUTPUT_DIR = 'outputs/model/'
    CHECKPOINT_DIR = 'outputs/model/checkpoint-last'
else:
    OUTPUT_DIR = '/opt/ml/model'
    CHECKPOINT_DIR = '/opt/ml/model/checkpoint-last'
    print("\x1b[32mRunning on SageMaker\x1b[0m")

def main():

    def load_images_from_s3(image_file_name):
        """
        Function to retrieve image from S3.
        """

        s3_client = session.client("s3")

        response = s3_client.get_object(Bucket=S3_BUCKET,
                                        Key=f"{IMAGE_FOLDER}/{image_file_name}")
        image_bytes = response["Body"].read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image

    def preprocess(examples):
        images = [load_images_from_s3(image) for image in examples["File_name"]]

        preprocessed = processor(text=examples["Description"],
                                images=images,
                                padding='max_length',
                                return_tensors='pt',
                                truncation=True)

        return {"input_ids": preprocessed['input_ids'],
                "attention_mask": preprocessed['attention_mask'],
                "pixel_values": preprocessed['pixel_values']
            }

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')
    
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
            
    print("\x1b[31mInitializing CLIP model and processor\x1b[0m")    
    login(token=HUGGINGFACE_ACCESS_TOKEN)    
    config = CLIPConfig.from_pretrained(MODEL_ID)        
    model = CLIPForContrastiveLearning.from_pretrained(MODEL_ID, config = config)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("\x1b[32mInitialized CLIP model and processor\x1b[0m")        

    print("\x1b[31mLoading dataset\x1b[0m")
    if RUNNING_LOCALLY:
        dataset = load_dataset("csv", data_files=INPUT_FILE)["train"]
    else:
        dataset = load_dataset("csv", data_files=f"/opt/ml/input/data/train/{INPUT_FILE}")["train"]
    print("\x1b[32mSuccessfully loaded dataset\x1b[0m")

    split = dataset.train_test_split(train_size = 0.8) # type: ignore
    train_split, test_split = split["train"], split["test"]

    print("\x1b[31mMapping preprocessing function to dataset\x1b[0m")
    train_dataset = train_split.map(preprocess, batched = True, batch_size = 4)
    test_dataset = test_split.map(preprocess, batched = True, batch_size = 4)

    train_dataset.set_format(type='torch',
                             columns=['input_ids', 'attention_mask', 'pixel_values'])
    test_dataset.set_format(type='torch',
                            columns=['input_ids', 'attention_mask', 'pixel_values'])
    print("\x1b[32mMapped dataset with preprocessfunction\x1b[0m")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\x1b[31mSetting up Training Arguments\x1b[0m")
    training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,  # SageMaker default model directory for saving artifacts
            learning_rate=2e-5,
            num_train_epochs=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=10,
            seed=42,
            fp16=True,  # use mixed precision if supported
            load_best_model_at_end=False,
            disable_tqdm=True
        )

    print("\x1b[31mSetting up Trainer\x1b[0m")
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[CustomLoggingCallback()]
        )
    
    print("\x1b[31mStarting to train model\x1b[0m")

    if os.path.exists(CHECKPOINT_DIR):
        trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)
    else:
        trainer.train()

    # Save final model (redundant as Trainer saves models during save_strategy)
    trainer.save_model("/opt/ml/model")

if __name__ == "__main__":
    main()
