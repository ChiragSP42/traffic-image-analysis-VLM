import boto3
import base64
import json
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
import time
load_dotenv()

WAIT_TIME = 30


model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'

# system_prompt = ("You are a helpful assistant that can analyze images and provide detailed and accurate description of the car in the image for law enforcement.\n"
#                 "You will be given an image of a car, and your task is to ONLY describe any unique indentifiable features of the car like stickers, dents, color, scratches, modifications, spoilers, custom paint jobs, car type (SUV, pickup truck, sedan, sports car, covertible, coupe, van, etc.) etc.\n"
#                 "If there are multiple dents, stickers, etc. mention and describe all of them.\n"
#                 "Do not mention the make or model of the car, or any other general information about the car.\n"
#                 "This information will be used by law enforcement to identify cars of interest, so be as detailed and accurate as possible when describing unique features.\n"
#                 "Here is an example of how the response should look like:\n\nUnique identifers:\n*Custom green paint job\n*Large dent on the rear bumper\n*Sticker of a dog on the back window\n*White sticker of what appears to be a figure\n*Lettered sticker or decal on the side\n\n"
#                 )

system_prompt_2 = ("You are a helpful assistant that can analyze images and provide detailed and accurate description of the car in the image for law enforcement.\n"
                   "Your job is to ONLY describe the following features of the car.\n"
                   "1. Year the car was manufactured.\n\tExample: 2000-2003, 2001-2006, 2016-2022,2017-2019\n"
                   "2. The color of the car.\n\tExample: red, blue, black, white, silver, green, grey\n"
                   "3. The make of the car.\n\tExample: Toyota, Honda, Ford, Chevrolet, BMW, Mercedes-Benz\n"
                   "4. The model of the car.\n\tExample: Camry, Accord, F-150, Silverado, 3 Series, C-Class\n"
                   "5. Any unique identifiable features of the car like stickers, dents, color, scratches, modifications, spoilers, custom paint\n"
                   "6. Identify the type of car (SUV, pickup truck, sedan, sports car, convertible, coupe, van, etc.)\n\n"
                   "Your answer should be in a JSON format. Here is an example of how the response should look like:\n\n"
                   "{\n"
                   "  \"year\": \"2016-2022\",\n"
                   "  \"color\": \"red\",\n"
                   "  \"make\": \"Toyota\",\n"
                   "  \"model\": \"Camry\",\n"
                   "  \"unique_identifiers\": [\n"
                   "    \"Custom green paint job\",\n"
                   "    \"Large dent on the rear bumper\",\n"
                   "    \"Sticker of a dog on the back window\",\n"
                   "    \"White sticker of what appears to be figure\",\n"
                   "    \"Blue circular sticker on the bumper\",\n"
                   "    \"Tinted rear windows\",\n"
                   "    \"Lettered sticker or decal on the side\"\n"
                   "  ],\n"
                   "  \"car_type\": \"sedan\"\n"
                   "}\n\n"
                   "Make sure to include all the features mentioned above in your response. If any of the features are not present in the image, leave the field empty\n")

if not os.getenv("AWS_ACCESS_KEY", None) or not os.getenv("AWS_SECRET_KEY", None):
    raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")

session = boto3.Session(aws_access_key_id = os.getenv("AWS_ACCESS_KEY"),
                        aws_secret_access_key= os.getenv("AWS_SECRET_KEY"),
                        region_name='us-east-1')

bedrock_runtime = session.client("bedrock-runtime")

# List all files in the Data directory
files_list = sorted(os.listdir('Data'))
# Filter out only the images (assuming they are .jpg files)
images = [image for image in files_list if image.endswith('.jpg')]
# Remove the file extension from the image names
image_names = [os.path.splitext(image)[0] for image in images]
# Split and record the year, color, make, and model from the image names
years, colors, makes, models = [], [], [], []
for image in image_names:
    parts = image.split('-', maxsplit = 4)
    if len(parts) == 5:
        years.append(parts[0]+"-"+parts[1])
        colors.append(parts[2])
        makes.append(parts[3])
        models.append(parts[4])
    else:
        years.append(parts[0]+"-"+parts[1])
        colors.append("")
        makes.append(parts[2])
        models.append(parts[3])

# Initialize lists to store the year, color, make, model, unique identifiers, and car type from the model responses
response_years, response_colors, \
response_makes, response_models, \
response_identifiers, response_car_type = [], [], [], [], [], []

# Initialize counters for the number of failed responses
counter_years, counter_colors, \
counter_makes, counter_models, \
counter_identifiers, counter_car_type = 0, 0, 0, 0, 0, 0

# Initialize a counter for the number of processed images
counter_processed = 0

for image in tqdm(images, desc="Processing images"):
    with open(os.path.join("Data", image), 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt_2,
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                "type": "image",
                "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                }
                ]
                }
            ]
        }
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request),
            contentType="application/json",
            accept="application/json"
        )
        time.sleep(WAIT_TIME)  # Wait for a while before processing the next image
    except Exception as e:
        print("Hit the exception:", e)
        break

    model_response_body = json.loads(response.get("body").read().decode("utf-8"))
    # print("\x1b[31mResponse for image:\x1b[0m", image)
    try:
        json_response = json.loads(model_response_body["content"][0]["text"])
        counter_processed += 1
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError for string: {model_response_body['content'][0]['text']}")
        continue
    try:
        response_years.append(json_response["year"])
    except KeyError:
        counter_years += 1
        response_years.append("")
    try:
        response_colors.append(json_response["color"])
    except KeyError:
        counter_colors += 1
        response_colors.append("")
    try:
        response_makes.append(json_response["make"])
    except KeyError:
        counter_makes += 1
        response_makes.append("")
    try:
        response_models.append(json_response["model"])
    except KeyError:
        counter_models += 1
        response_models.append("")
    try:
        response_identifiers.append(json_response["unique_identifiers"])
    except KeyError:
        counter_identifiers += 1
        response_identifiers.append([])
    try:
        response_car_type.append(json_response["car_type"])
    except KeyError:
        counter_car_type += 1
        response_car_type.append("")


print("Processed", counter_processed, "images.")
print(f"Failed to record {counter_years} years, {counter_colors} colors, \
      {counter_makes} makes, {counter_models} models, {counter_identifiers} identifiers, \
       and {counter_car_type} car types.")

response_df = pd.DataFrame()
response_df["Truth_years"] = years[:counter_processed]
response_df["Response_years"] = response_years
response_df["Truth_colors"] = colors[:counter_processed]
response_df["Response_colors"] = response_colors
response_df["Truth_makes"] = makes[:counter_processed]
response_df["Response_makes"] = response_makes
response_df["Truth_models"] = models[:counter_processed]
response_df["Response_models"] = response_models
response_df["Response_car_type"] = response_car_type
response_df["Response_identifiers"] = response_identifiers

response_df.to_csv("response.csv", index=False)
    # print(model_response_body["content"][0]["text"])
    # print()
    # print(json.dumps(model_response, indent=2))
    # input()
# print(model_response["output"]["message"]["content"][0]["text"])


#%%
import pandas as pd

response_df = pd.read_csv("response.csv")
response_df.head()

files_list = sorted(os.listdir('Data'))
# Filter out only the images (assuming they are .jpg files)
images = [image for image in files_list if image.endswith('.jpg')]
image_names = [os.path.splitext(image)[0] for image in images]

response_df.insert(0, "File_name", image_names)

response_df.head()
# %%
