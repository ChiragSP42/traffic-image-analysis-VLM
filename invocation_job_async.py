import sys
import boto3
import os
from dotenv import load_dotenv
from utils.helpers import list_obj_s3, create_input_jsonl, poll_invocation_job
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY
                        )
bedrock = session.client('bedrock', region_name='us-east-1')

s3_client = session.client('s3', region_name='us-east-1')

BUCKET_NAME = 'signal-8-data-creation-testing'
FOLDER_NAME = 'Data'
# First check if input.jsonl file already exits.
input_jsonl_yes_no = list_obj_s3(s3_client=s3_client,
                                 bucket_name=BUCKET_NAME,
                                 folder_name='input.jsonl')

if not input_jsonl_yes_no:
    print("\x1b[31mInput jsonl file does not exist. Creating new one...\x1b[0m")

    system_prompt = ("You are a helpful assistant that can analyze images and provide detailed and accurate description of the car in the image for law enforcement.\n"
                    "You will be given an image of a car, and your task is to ONLY describe any unique indentifyable features of the car like stickers, dents, color, scratches, modifications, spoilers, custom paint jobs, car type (SUV, pickup truck, sedan, etc.) etc.\n"
                    "If and only if the car's license plate is clearly visible, include it in your description.\n"
                    "This information will be used by law enforcement to identify cars of interest, so be as detailed and accurate as possible.\n"
                    "Here is an example of a good response:\n\nUnique identifers:\n*Custom paint job with flames on the side\n*Large dent on the rear bumper\n*Sticker of a dog on the back window\n*License plate: ABC1234\n\n"
                    )
    
    create_input_jsonl(s3_client=s3_client,
                       bucket_name=BUCKET_NAME,
                       folder_name=FOLDER_NAME,
                       system_prompt=system_prompt)

else:
    print("\x1b[32mInput jsonl file already exists. No need to create a new one.\x1b[0m")

inputDataConfig = {
    "s3InputDataConfig": {
        "s3InputFormat": "JSONL",
        "s3Uri": f"s3://{BUCKET_NAME}/input.jsonl"
    }
}

outputDataConfig = {
    's3OutputDataConfig': {
        's3Uri': f's3://{BUCKET_NAME}/output/'
    }
}

model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'

print("\x1b[34mStarting model invocation job...\x1b[0m")

response = bedrock.create_model_invocation_job(
    jobName='car-image-analysis-job-200-images',
    modelId=model_id,
    inputDataConfig=inputDataConfig,
    outputDataConfig=outputDataConfig,
    roleArn='arn:aws:iam::381492026108:role/BatchInferenceJobRole',  # Replace with your IAM role ARN
)
print(f"Model invocation job created with ARN: {response['jobArn']}")
print("\x1b[31mPolling for job completion...\x1b[0m")
status = poll_invocation_job(bedrock=bedrock, jobArn=response['jobArn'])
if status:
    print("\x1b[32mJob completed.\x1b[0m")
else:
    print("\x1b[31mJob failed.\x1b[0m")