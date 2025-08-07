import boto3
import os
import pandas as pd
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name='us-east-1')

dataset = load_dataset("csv", data_files='input.csv')["train"]




