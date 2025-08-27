from aws_helpers import helpers
import os
import boto3
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

"""
if images present:
    if JSON present:
        if images == JSON:
            NOTHING
        else:
            Download missing images (VIF)
    else:
        Read images and create JSON (VIF)
else:
    if JSON present:
        Download images (VIF)
    else:
        Read EXCEL and create dataset and download images.
"""

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None)
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None)
IMAGE_FOLDER = 'Images'
DATASET_JSON_FILE = 'dataset.json'
S3_BUCKET = 'signal-8-evox'

if AWS_SECRET_KEY is None or AWS_SECRET_KEY is None:
    raise ValueError("AWS access credentials are missing/wrong, check")

session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name='us-east-1')

s3_client = session.client("s3")
images_response = helpers.list_obj_s3(s3_client=s3_client,
                       bucket_name=S3_BUCKET,
                       folder_name=IMAGE_FOLDER)
json_response = helpers.list_obj_s3(s3_client=s3_client,
                                bucket_name=S3_BUCKET,
                                folder_name=DATASET_JSON_FILE)
# Check if images are present
if len(images_response):
    # Check if dataset.json is present
    if len(json_response):
        #TODO: Compare number of images to number of entries in dataset.json
        pass
    # If json isn't present, read images and create JSON
    else:
        #TODO: Read the images which will contain the VIFID and create dataset.json
        pass
# If images are not present
else:
    # Check if JSON is present
    if len(json_response):
        # TODO: Read JSON file and download images.
        pass
    # If nothing is present, create JSON and download images.
    else:
        helpers.create_dataset(bucket_name=S3_BUCKET,
                                image_folder=IMAGE_FOLDER,
                                no_vif=2,
                                no_images=2,
                                vif_shuffle=True,
                                images_shuffle=True,
                                seed=42)

        
 