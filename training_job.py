from aws_helpers.utils import FineTuning
from aws_helpers.helpers import _local_or_sagemaker
import os
import torch
import boto3
from torch import nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset, IterableDataset
from huggingface_hub import login
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
MODEL_ID = "openai/clip-vit-base-patch32"
S3_BUCKET = 'signal-8-data-creation-testing'
INPUT_FILE = 'input.csv'
IMAGE_FOLDER = 'Data'
OUTPUT_DIR = ''
CHECKPOINT_DIR = ''
dataset = None

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')

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

if not _local_or_sagemaker:
    print("\x1b[32mRunning locally\x1b[0m")
    OUTPUT_DIR = 'outputs/model/'
    CHECKPOINT_DIR = 'outputs/model/checkpoint-last'
else:
    print("\x1b[32mRunning on SageMaker\x1b[0m")
    OUTPUT_DIR = '/opt/ml/model'
    CHECKPOINT_DIR = '/opt/ml/model/checkpoint-last'
    dataset = load_dataset("json", data_files=f's3://{S3_BUCKET}/{INPUT_FILE}', streaming=True, split='train')


fine_tuner = FineTuning(model=model,
                        processor=processor,
                        config=config,
                        dataset=dataset,
                        batch_size=4)