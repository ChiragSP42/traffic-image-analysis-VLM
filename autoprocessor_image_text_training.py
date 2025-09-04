from typing import Dict, List, Any, Optional
import logging
import gc
from io import BytesIO
import torch
import json
from aws_helpers import helpers
from transformers import InternVLProcessor, AutoModelForImageTextToText
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = helpers._setup_logger(level=logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = 'OpenGVLab/InternVL3-14B-hf'
S3_BUCKET = 'signal-8-flock'

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

def build_conversation(text):
    messages = []
    with open('user_prompt.txt', 'r') as f:
        user_prompt = f.read()
    user_prompt = "<image>"
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": text})
    return {"messages": messages}

class ManualImageTextDataset(Dataset):
    def __init__(self, s3_bucket: str, json_file: str):
        self.records = []
        s3_client = helpers._get_s3_client()
        input_json_file = s3_client.get_object(Bucket=s3_bucket, Key=json_file)["Body"].read().decode('utf-8')
        input_json_file = json.loads(input_json_file)
        
        for json_obj in input_json_file['output'][:16]:
            template = create_template(json_obj)
            image_uri = json_obj["s3_uri"]
            self.records.append({"image_uri": image_uri, "text": template})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Load and process image manually
        image = load_image_from_s3(s3_uri=record["image_uri"])
        conversation = build_conversation(record['text'])
        
        return {
            "image": image,
            "conversation": conversation
        }

class ManualCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [b['image'] for b in batch]
        texts = [b['conversation']['messages'] for b in batch]
        
        processed = self.processor(
            images=images,
            text=texts,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        logger.debug(type(processed))
        
        return {k: v for k, v in processed.items()}

# Manual forward pass function
def manual_forward_pass(model, batch):
    """
    Manually handle model forward pass and loss computation
    """
    pixel_values = batch['pixel_values'].to(DEVICE)
    input_ids = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)
    
    # Forward pass - note: this might need adjustment based on actual model API
    # InternVL3-hf should accept these arguments
    outputs = model(**batch)
    
    return outputs.loss if hasattr(outputs, 'loss') else None

def main():
    # Load tokenizer only (no processor)
    processor = InternVLProcessor.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # Load model
    # model = AutoModelForImageTextToText.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    # model.train()
    
    logger.info("Creating dataset")
    dataset = ManualImageTextDataset(json_file='created_data.json', s3_bucket=S3_BUCKET)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    logger.info("Creating collator")
    collator = ManualCollator(processor)
    
    logger.info("Creating dataloader")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    
    # Test single batch first
    logger.info("Testing single batch...")
    try:
        sample_batch = next(iter(dataloader))
        logger.info("✅ Batch creation successful!")
        logger.info(f"Batch keys: {sample_batch.keys()}")
        logger.info(f"Pixel values shape: {sample_batch['pixel_values'].shape}")
        logger.info(f"Input IDs shape: {sample_batch['input_ids'].shape}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # Full training loop
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # logger.info("Starting training...")
    # for epoch in range(10):  # Reduced for testing
    #     for step, batch in enumerate(dataloader):
    #         print(batch)
    #         try:
    #             loss = manual_forward_pass(model, batch)
                
    #             if loss is not None:
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #                 optimizer.step()
    #                 optimizer.zero_grad()
                    
    #                 logger.info(f"✅ Step {step}, Loss: {loss.item():.4f}")
    #             else:
    #                 logger.warning(f"⚠️ Step {step}, No loss returned")
                
    #             if step >= 2:  # Test just a few steps
    #                 break
                    
    #         except Exception as e:
    #             logger.error(f"❌ Step {step} failed: {e}")
    #             break
        
    #     break

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
