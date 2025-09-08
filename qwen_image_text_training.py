from typing import Dict, List, Any
import logging
import gc
from io import BytesIO
import torch
import json

from aws_helpers import helpers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = helpers._setup_logger(level=logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use a Qwen2.5-VL Instruct checkpoint (change size if needed, e.g., 3B or 72B)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # 3B/7B/72B are available
S3_BUCKET = "signal-8-flock"

def create_template(json_obj: Dict) -> str:
    template = ""
    if json_obj.get("year") and json_obj["year"] != "None":
        template += json_obj["year"] + " "
    if json_obj.get("car_type") and json_obj["car_type"] != "None":
        template += json_obj["car_type"] + " "
    if json_obj.get("color") and json_obj["color"] != "None":
        template += json_obj["color"] + " "
    if json_obj.get("make") and json_obj["make"] != "None":
        template += json_obj["make"] + " "
    if json_obj.get("model") and json_obj["model"] != "None":
        template += json_obj["model"] + " "
    if json_obj.get("license_plate") and json_obj["license_plate"] != "None":
        template += f"with license plate number {json_obj['license_plate']} "
    if json_obj.get("unique_identifiers"):
        template += "has the following unique identifiers: " + ", ".join(json_obj["unique_identifiers"])
    return template.strip()

def load_image_from_s3(s3_uri: str) -> Image.Image:
    s3_client = helpers._get_s3_client()
    path = s3_uri[5:]
    bucket, key = path.split("/", 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    img_bytes = obj["Body"].read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img

def build_conversation(target_text: str) -> Dict[str, Any]:
    """
    Build a Qwen-style chat with one user turn (image + instruction)
    and one assistant turn (ground-truth answer).
    """
    instruction = (
        "Describe the traffic camera image of the car, including year, type, color, make, model, "
        "license plate, and any unique identifiers."
    )
    # The image is indicated by a placeholder here; the actual PIL image is passed separately to the processor.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # placeholder for one image
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        },
    ]
    return {"messages": messages}

class ManualImageTextDataset(Dataset):
    def __init__(self, s3_bucket: str, json_file: str):
        self.records = []
        s3_client = helpers._get_s3_client()
        input_json_file = s3_client.get_object(Bucket=s3_bucket, Key=json_file)["Body"].read().decode("utf-8")
        input_json_file = json.loads(input_json_file)
        for json_obj in input_json_file["output"][:16]:
            template = create_template(json_obj)
            image_uri = json_obj["s3_uri"]
            self.records.append({"image_uri": image_uri, "text": template})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = load_image_from_s3(s3_uri=record["image_uri"])
        conversation = build_conversation(record["text"])
        return {
            "image": image,
            "conversation": conversation,
        }

class ManualCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        # Marker used by the Qwen chat template before assistant content
        self.assistant_tag = "<|im_start|>assistant\n"

    def _mask_labels_to_assistant(self, texts: List[str], input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels by masking everything up to (and including) the assistant tag for each sequence.
        This focuses loss on the assistant reply only.
        """
        labels = input_ids.clone()
        labels[:] = labels  # copy
        # Compute prefix lengths per sample by tokenizing the text up to assistant tag
        prefix_lens = []
        for t in texts:
            pos = t.find(self.assistant_tag)
            if pos == -1:
                # Fallback: no assistant tag found, mask nothing (train on all tokens)
                prefix_lens.append(0)
                continue
            prefix_text = t[: pos + len(self.assistant_tag)]
            toks = self.processor.tokenizer(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prefix_lens.append(int(toks["input_ids"].shape[1]))
        # Apply mask
        for i, pref in enumerate(prefix_lens):
            labels[i, :pref] = -100
        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        messages_list = [b["conversation"]["messages"] for b in batch]

        # Build the chat text for each sample (we keep the assistant answer in the template for SFT)
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # we are training with ground-truth answer present
            )
            for messages in messages_list
        ]

        # Tokenize text and preprocess images in one call
        processed = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Build labels that supervise only the assistant answer
        labels = self._mask_labels_to_assistant(texts, processed["input_ids"])

        processed["labels"] = labels
        logger.debug(type(processed))
        return {k: v for k, v in processed.items()}

def manual_forward_pass(model, batch):
    """
    Model forward pass and loss computation for Qwen2.5-VL.
    """
    # Move tensors to device
    batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
    outputs = model(**batch)  # expects input_ids, attention_mask, pixel_values, labels
    return outputs.loss if hasattr(outputs, "loss") else None

def main():
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # If training, set device_map and dtype appropriate for hardware; here we just sanity check the batch
    # For larger models, consider int4/int8 or LoRA/QLoRA during training.
    # attn_implementation can be set when available (e.g., "sdpa" or "flash_attention_2").
    # See Qwen docs for suggested setups.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    logger.info("Creating dataset")
    dataset = ManualImageTextDataset(json_file="created_data.json", s3_bucket=S3_BUCKET)
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
        logger.info(f"Batch keys: {list(sample_batch.keys())}")
        logger.info(f"Pixel values shape: {sample_batch['pixel_values'].shape}")
        logger.info(f"Input IDs shape: {sample_batch['input_ids'].shape}")
        logger.info(f"Labels shape: {sample_batch['labels'].shape}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

    # Example training loop (commented)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # model.train()
    # logger.info("Starting training...")
    # for epoch in range(1):
    #     for step, batch in enumerate(dataloader):
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
    #             if step >= 2:
    #                 break
    #         except Exception as e:
    #             logger.error(f"❌ Step {step} failed: {e}")
    #             break

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
