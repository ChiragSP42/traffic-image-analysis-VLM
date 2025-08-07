# ReadMe

This project aims to analyze traffic images of cars or take the description of the car and output similar cars present in the database.
To accomplish this, this project leverages multimodal LLM to curate a dataset based off real life images. These text-image pairs are used to fine tune a image embedding model called CLIP (Contrastive Language-Image Pretraining). The embedded images are fed into a vector database to be retrieved.

Before going over the different programs, their functions and outcomes, let's go over the environment setup.

## Environment Set up.

This code was created with python==3.10.18 in a conda environment.

1. Run `conda create -n <your python environment name> python=3.10.18`
2. To activate environment, run `conda activate <your python environment name>`
3. To set up libraries, run `pip install -r requirements.txt`
4. Follow env-example.txt file to set up .env file.

Optional, but if you're running code locally, especially the fine_tuning.py. Set up the aws cli with the correct credentials.

## Dataset creation

For now a sync job of invoking a multimodal LLM to generate a detailed description of the car with the help of a curated system prompt.
TODO: Batch inference job to create a dataset in the order of 10,000 of images.
Image file names and the corresponding text description are saved offline as a csv file. This csv file is to be used later for fine tuning.

## 