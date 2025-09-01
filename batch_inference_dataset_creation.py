import os
import boto3
from aws_helpers.utils import BatchInference
from aws_helpers.helpers import list_obj_s3
from dotenv import load_dotenv
load_dotenv()

"""
The BatchInference tool is used to create, poll and post process a batch inference job. In this case
it's used for dataset creation of image-text pairs. The source will be an S3 folder with the structure 
<S3 bucket name>/<Image folder name>/<All the images>. The outputs will be stored in the same S3 bucket 
which you can name as you deem. 

Prerequisites:
1. Creation of a role for batch inference. Here is an example used for this.
TRUSTED ENTITIES:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "123456789123"
                },
                "ArnEquals": {
                    "aws:SourceArn": "arn:aws:bedrock:us-east-1:123456789123:model-invocation-job/*"
                }
            }
        }
    ]
}

Policies:
- AmazonBedrockFullAccess
- AmazonS3FullAccess
- Custom policy:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:InvokeModel"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::signal-8-data-creation-testing",
                "arn:aws:s3:::signal-8-data-creation-testing/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl"
            ],
            "Resource": [
                "arn:aws:s3:::signal-8-data-creation-testing",
                "arn:aws:s3:::signal-8-data-creation-testing/output/*"
            ]
        }
    ]
}

Process starts by calling the `start_batch_inference_job` function. It checks for the presence of a 'input.jsonl' file. 
If it isn't present, it will first create it based on images present in the aforementioned folder and then start the job 
and returns the ARN of the job. (Note each job must have a unique name not previously used). The prompt should be stored in 
a text file called 'creation_prompt.txt'.

Once the job is started, you can poll it using the `poll_invocation_job` function. It will return True is job is SUCCESSFUL 
or False if FAILED. 

The outputs of the results will be in the output folder you mentioned. Inside it is a randomly generated folder name within which 
2 json(l) files are preesent. A 'manifest.json' containing details about the job. The input.jsonl now containing a key of 'modelOutput'.

A post processing function `process_batch_inference_output` is used to extract contents and format the data into 'image-s3-URIs-and-text pairs.
"""

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY
                        )
bedrock = session.client('bedrock', region_name='us-east-1')

s3_client = session.client('s3', region_name='us-east-1')

BUCKET_NAME = 'signal-8-flock'
# BUCKET_NAME = 'bravo-foxtrot-data'
FOLDER_NAME = 'Daytime'
# FOLDER_NAME = 'bravo_foxtrot_images_data'
OUTPUT_FOLDER = 'output-sonnet-4/'
# MODEL_ID = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
MODEL_ID = 'us.anthropic.claude-sonnet-4-20250514-v1:0'
ROLE_ARN = 'arn:aws:iam::381492026108:role/BatchInferenceJobRole'

with open("creation_prompt.txt", 'r') as f:
    creation_prompt = f.read()

batch_inference = BatchInference(bedrock_client=bedrock,
                                 s3_client=s3_client,
                                 bucket_name=BUCKET_NAME,
                                 folder_name=FOLDER_NAME,
                                 output_folder=OUTPUT_FOLDER,
                                 model_id=MODEL_ID,
                                 creation_prompt=creation_prompt,
                                 role_arn = ROLE_ARN,
                                 job_name='s8-2')


job_arn = batch_inference.start_batch_inference_job()
status = batch_inference.poll_invocation_job(jobArn=job_arn)
# batch_inference.poll_invocation_job(jobArn='arn:aws:bedrock:us-east-1:381492026108:model-invocation-job/pmaeef7fviwk')
# status = True
if status:
    batch_inference.process_batch_inference_output(local_copy=True)

