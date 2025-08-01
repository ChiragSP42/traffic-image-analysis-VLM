import boto3
import base64
import json
import os
from dotenv import load_dotenv
load_dotenv()

# image_path = '2000-2003-Gold-Nissan-Maxima.jpg'
image_path = '2001-2006-Blue-Chrysler-Sebring-Convertible.jpg'
# image_path = '2005-2006-Burgundy-Honda-CRV.jpg'

model_id = 'amazon.nova-pro-v1:0'
model_id = 'anthropic.claude-3-7-sonnet-20250219-v1:0'
model_id = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
model_id = 'meta.llama3-2-11b-instruct-v1:0'
model_id = 'meta.llama3-2-90b-instruct-v1:0'

with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

system_list = [
    {
        "text": "You are a helpful assistant that can analyze images and provide detailed descriptions. You will be given an image of a car, and your task is to describe the car in detail including color, make, model, year, license plate number, and any other unique relevant features that might be useful for law enforcement."
    }
]

message_list = [
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpg",
                    "source": {
                        "bytes": image_data
                    }
                }
            },
            {
                "text": "Describe this car in detail including color, make, model, year, and license plate number and any other unique relevant features that might be useful for law enforcement."
            }
        ]
    }
]

inf_params = {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0.3}

request = {
    "schemaVersion": "messages-v1",
    "messages": message_list,
    "system": system_list,
    "inferenceConfig": inf_params
}

if not os.getenv("AWS_ACCESS_KEY", None) or not os.getenv("AWS_SECRET_KEY", None):
    raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")

session = boto3.Session(aws_access_key_id = os.getenv("AWS_ACCESS_KEY"),
                        aws_secret_access_key= os.getenv("AWS_SECRET_KEY"),
                        region_name='us-east-1')

bedrock_runtime = session.client("bedrock-runtime")
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=json.dumps(request),
    contentType="application/json",
    accept="application/json"
)

model_response = json.loads(response.get("body").read())
# print(json.dumps(model_response, indent=2))
print(model_response["output"]["message"]["content"][0]["text"])