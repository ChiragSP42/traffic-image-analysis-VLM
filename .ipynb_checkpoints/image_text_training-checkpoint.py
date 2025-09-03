from typing import Dict, List, Any, Optional
import logging
from io import BytesIO
import torch
import json
from aws_helpers import helpers
from transformers import AutoTokenizer, AutoModelForImageTextToText
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

logger = helpers._setup_logger(level=logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = 'OpenGVLab/InternVL3-8B-hf'
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

# Manual image processing function
def process_image(image: Image.Image, image_size: int = 448):
    """
    Manually process image without AutoProcessor
    InternVL uses 448x448 base resolution with specific normalization
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)

def create_conversation_text(text: str) -> tuple[str, str, str]:
    """
    Create conversation without special tokens - just plain text
    Returns: (full_conversation, user_part, assistant_part)
    """
    with open("user_prompt.txt", 'r') as f:
        user_prompt = f.read()
    assistant_response = text
    
    # Create a simple conversation format
    conversation = f"User: {user_prompt}\nAssistant: {assistant_response}"
    
    return conversation, user_prompt, assistant_response

def tokenize_conversation(tokenizer, conversation: str, user_prompt: str, assistant_response: str, max_length: int = 512):
    """
    Manually tokenize conversation and create labels
    Only assistant tokens contribute to loss
    """
    # Tokenize full conversation
    full_tokens = tokenizer.encode(conversation, add_special_tokens=True, max_length=max_length, truncation=True)
    
    # Find where assistant response starts
    user_part = f"User: {user_prompt}\nAssistant: "
    user_tokens = tokenizer.encode(user_part, add_special_tokens=False)
    
    # Create labels - mask non-assistant tokens with -100
    labels = [-100] * len(full_tokens)
    
    # Find assistant start position
    assistant_start = len(user_tokens)
    if assistant_start < len(full_tokens):
        # Only apply loss to assistant response tokens
        for i in range(assistant_start, len(full_tokens)):
            labels[i] = full_tokens[i]
    
    return {
        'input_ids': torch.tensor(full_tokens),
        'labels': torch.tensor(labels),
        'attention_mask': torch.ones(len(full_tokens))
    }

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
        pixel_values = process_image(image)
        
        # Create conversation text
        conversation, user_prompt, assistant_response = create_conversation_text(record["text"])
        
        return {
            "pixel_values": pixel_values,
            "conversation": conversation,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response
        }

class ManualCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack image tensors
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # Tokenize conversations
        tokenized_batch = []
        for item in batch:
            tokenized = tokenize_conversation(
                self.tokenizer, 
                item['conversation'], 
                item['user_prompt'], 
                item['assistant_response'],
                self.max_length
            )
            tokenized_batch.append(tokenized)
        
        # Pad sequences to same length
        max_len = max(len(item['input_ids']) for item in tokenized_batch)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for item in tokenized_batch:
            input_ids = item['input_ids']
            labels = item['labels']
            attention_mask = item['attention_mask']
            
            # Pad sequences
            pad_len = max_len - len(input_ids)
            
            padded_input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            padded_labels = F.pad(labels, (0, pad_len), value=-100)
            padded_attention = F.pad(attention_mask, (0, pad_len), value=0)
            
            batch_input_ids.append(padded_input_ids)
            batch_labels.append(padded_labels)
            batch_attention_mask.append(padded_attention)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_attention_mask)
        }

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
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels  # Model should compute loss automatically
    )
    
    return outputs.loss if hasattr(outputs, 'loss') else None

def main():
    # Load tokenizer only (no processor)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.train()
    
    logger.info("Creating dataset")
    dataset = ManualImageTextDataset(json_file='created_data.json', s3_bucket=S3_BUCKET)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    logger.info("Creating collator")
    collator = ManualCollator(tokenizer, max_length=256)
    
    logger.info("Creating dataloader")
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
    
    # Test single batch first
    # logger.info("Testing single batch...")
    # try:
    #     sample_batch = next(iter(dataloader))
    #     logger.info("✅ Batch creation successful!")
    #     logger.info(f"Batch keys: {sample_batch.keys()}")
    #     logger.info(f"Pixel values shape: {sample_batch['pixel_values'].shape}")
    #     logger.info(f"Input IDs shape: {sample_batch['input_ids'].shape}")
        
    # except Exception as e:
    #     logger.error(f"❌ Error: {e}")
    #     return
    
    # Full training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    logger.info("Starting training...")
    for epoch in range(10):  # Reduced for testing
        for step, batch in enumerate(dataloader):
            try:
                loss = manual_forward_pass(model, batch)
                
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    logger.info(f"✅ Step {step}, Loss: {loss.item():.4f}")
                else:
                    logger.warning(f"⚠️ Step {step}, No loss returned")
                
                if step >= 2:  # Test just a few steps
                    break
                    
            except Exception as e:
                logger.error(f"❌ Step {step} failed: {e}")
                break
        
        break

if __name__ == "__main__":
    main()
