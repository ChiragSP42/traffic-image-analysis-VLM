import os
import boto3
from aws_helpers.utils import BatchInference
from aws_helpers.helpers import list_obj_s3
from dotenv import load_dotenv
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
# BUCKET_NAME = 'bravo-foxtrot-data'
FOLDER_NAME = 'Data'
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
                                 job_name='s8-1')


job_arn = batch_inference.start_batch_inference_job()
status = batch_inference.poll_invocation_job(jobArn=job_arn)
# batch_inference.poll_invocation_job(jobArn='arn:aws:bedrock:us-east-1:381492026108:model-invocation-job/pmaeef7fviwk')
# status = True
if status:
    batch_inference.process_batch_inference_output(local_copy=True)

