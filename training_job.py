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
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
MODEL_ID = "openai/clip-vit-base-patch32"
S3_BUCKET = 'signal-8-data-creation-testing'
INPUT_FILE = 'created_data.json'
IMAGE_FOLDER = 'Data'
OUTPUT_DIR = ''
CHECKPOINT_DIR = ''
dataset = Dataset

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')

s3_client = session.client('s3')

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

if not _local_or_sagemaker:
    print("\x1b[32mRunning locally\x1b[0m")
    OUTPUT_DIR = 'outputs/model/'
    CHECKPOINT_DIR = 'outputs/model/checkpoint-last'
else:
    print("\x1b[32mRunning on SageMaker\x1b[0m")
    OUTPUT_DIR = '/opt/ml/model'
    CHECKPOINT_DIR = '/opt/ml/model/checkpoint-last'
    dataset = load_dataset("json", data_files=f's3://{S3_BUCKET}/{INPUT_FILE}', field='output', streaming=True, split='train')
    # dataset = load_dataset("json", data_files='temp_json.jsonl', split='train', streaming=False)


fine_tuner = FineTuning(model=model,
                        processor=processor,
                        dataset=dataset,
                        batch_size=4,
                        s3_client=s3_client,
                        bucket_name=S3_BUCKET,
                        folder_name=IMAGE_FOLDER)

train_dataset, test_dataset = fine_tuner.split(train_size=0.8)

train_dataset.set_format(type='torch',
                             columns=['input_ids', 'attention_mask', 'pixel_values'])
test_dataset.set_format(type='torch',
                        columns=['input_ids', 'attention_mask', 'pixel_values'])

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