from typing import List, Optional, Any
import boto3
import os
import sys
from dotenv import load_dotenv
import time
import json
load_dotenv()

def _list_foundational_models(byOutputModality: Optional[str] = None,
                 byProvider: Optional[str] = None) -> None:
    """
    Function to list all models available in the session.
    
    Parameters:
        byOutputModality (Optional[str]): Filter models by output modality, 'TEXT'|'IMAGE'|'EMBEDDING'.
        byProvider (Optional[str]): Filter models by provider.
    
    Returns:
        None
    """

    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")
    
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')
    if not session:
        raise ValueError("Failed to create a Boto3 session. Check your AWS credentials and region.")
    
    bedrock = session.client('bedrock')
    if byOutputModality and byProvider:
        response = bedrock.list_foundation_models(
            byOutputModality=byOutputModality,
            byProvider=byProvider
        )
    elif byOutputModality or byProvider:
        if byOutputModality:
            response = bedrock.list_foundation_models(
                byOutputModality=byOutputModality
            )
        else:
            response = bedrock.list_foundation_models(
                byProvider=byProvider
            )
    else:
        response = bedrock.list_foundation_models()
    
    if 'modelSummaries' not in response:
        raise ValueError("No models found in the response. Check your AWS credentials and permissions.")
    
    for model in response['modelSummaries']:
        print(f"Provider name: {model['providerName']}\nModel Name: {model['modelName']}\nModel ID: {model['modelId']}")
        print(f"Input Modalities: {model['inputModalities']}\nOutput Modalities: {model['outputModalities']}")
        print("-" * 30)

def _list_inference_profiles():
    """
    Function to list all inference profiles available in the session.
    
    Returns:
        None
    """
    
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')
    if not session:
        raise ValueError("Failed to create a Boto3 session. Check your AWS credentials and region.")
    
    bedrock = session.client('bedrock')
    response = bedrock.list_inference_profiles()
    for profile in response.get('inferenceProfileSummaries', []):
        print(f"Profile Name: {profile['inferenceProfileName']}\nProfile ID: {profile['inferenceProfileId']}")
        print("-" * 30)

def _local_or_sagemaker():
    """
    Checks if the current Python script is running within an Amazon SageMaker environment or locally.
    """
    # As a fallback, check for other common SageMaker environment variables
    sagemaker_env_vars = ['SM_CHANNEL_TRAIN', 'SM_MODEL_DIR', 'SAGEMAKER_PROGRAM']
    for var in sagemaker_env_vars:
        if var in os.environ:
            return True

    return False

def list_obj_s3(s3_client: Any,
                bucket_name: Optional[str],
                folder_name: Optional[str],
                delimiter: Optional[str] = '')-> List[str]:
    """
    Function to return list of objects present in bucket. There is an optional
    delimiter parameter to toggle between folder and file names. If delimiter is empty, it will return all files in the bucket.

    Parameters:
        s3_client (Any): S3 client object
        bucket_name (str): Name of S3 bucket where concerned docs are present.
        foldername (str): Name of folder in which pdfs are present.
        delimiter (str): Delimiter to toggle between folder and file names. Default is '/'.

    Returns:
        pdf_list (list[str]): List of pdf names with folder path included.
    """

    pdf_list = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name,
                                   Prefix=folder_name,
                                   Delimiter=delimiter):
        if delimiter:
            if 'CommonPrefixes' in page:
                pdf_list = [obj["Prefix"] for obj in page.get('CommonPrefixes', [])]
        else:
            for obj in page.get('Contents', []):
                key = obj['Key']
                pdf_list.append(key)

    return pdf_list

def create_input_jsonl(s3_client: Any,
                        bucket_name: str,
                        folder_name: str,
                        system_prompt: str) -> None:
    """
    Function to create input.jsonl file for invoking the model. Check if the input.jsonl file already exists in the S3 bucket first.

    Parameters:
        s3_client (Any): S3 client object
        bucket_name (str): Name of S3 bucket where concerned docs are present.
        folder_name (str): Name of folder in which images are present.
        system_prompt (str): System prompt for the model.

    Returns:
        None
    """
    
    list_of_images = list_obj_s3(s3_client=s3_client,
                                 bucket_name=bucket_name,
                                 folder_name=folder_name)

    input_json_file = []
    for idx, image_filename in enumerate(list_of_images):
        content = [
            {
                "image": {
                    "format": "jpg",
                    "source": {
                        "s3Uri": f"s3://{bucket_name}/{image_filename}"
                    }
                }
            }
        ]

        json_obj = {
            "recordId": idx,
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
        }
        input_json_file.append(json_obj)
    
    with open('input.jsonl', 'w') as f:
        for json_obj in input_json_file:
            f.write(json.dumps(json_obj) + "\n")
    print("\x1b[32mCreated JSONL file and store local copy as input.jsonl\x1b[0m")
    
    # Upload JSONL file to S3.
    try:
        print(f"\x1b[31mUploading input.jsonl file to S3 bucket at path {bucket_name}/input.jsonl\x1b[0m")
        with open('input.jsonl', 'rb') as f:
            s3_client.put_object(Bucket=bucket_name,
                                Key='input.jsonl',
                                Body=f)
        print("\x1b[32mUploaded file\x1b[0m")
    except Exception as e:
        print(e)

def poll_invocation_job(bedrock: Any,
                        jobArn: str):
    """Function to poll the status of the model invocation job.

     Parameters:
         bedrock (Any): Bedrock client object.
         jobArn (str): ARN of the model invocation job.

    Returns:
         str: Status of the job.
    """
    black_flag = True
    white_flag = False
    counter = 0
    while True:
        status = bedrock.get_model_invocation_job(jobIdentifier=jobArn)['status']
        # print(f"Status: {status}")
        dots = "." * (counter % 4)
        sys.stdout.write(f"\r{status}{dots}".ljust(len(status) + 4))
        sys.stdout.flush()
        time.sleep(0.5)
        counter += 1
        if status == 'Completed':
            return True
        elif status == 'Failed':
            return False
        time.sleep(5)