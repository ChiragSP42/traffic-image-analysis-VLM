# ReadMe

This project aims to analyze traffic images of cars or take the description of the car and output similar cars present in the database.
To accomplish this, this project leverages multimodal LLM to curate a dataset based off real life images. These text-image pairs are used to fine tune a image embedding model called CLIP (Contrastive Language-Image Pretraining). The embedded images are fed into a vector database along with the text metadata.

Before going over the different programs, their functions and outcomes, let's go over the environment setup.

## Environment Set up

This code was created with python==3.10.18 in a conda environment.

1. Run the following command to create python conda environment.

    ```bash
    conda create -n <your python environment name> python=3.10.18
    ```

2. To activate environment, run

    ```bash
    conda activate <your python environment name>
    ```

3. To set up libraries, run

    ```python
    pip install -r requirements.txt
    ```

4. Follow env-example.txt file to set up .env file.

***
Optional, but if you're running code locally, especially the fine_tuning.py. Set up the aws cli with the correct credentials.

## Dataset creation

Use the `invocation_job_async.py` script to create a batch inference job to create the image-text pairs. The batch inference job uses a `JSONL` file. For the format of `JSONL` file, refer to [Format and upload your batch inference data](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-data.html). Here is a [code example for a batch job](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-example.html). One of the biggest hurdles faced was getting the right JSONL format. Here is an example of how the file should look like.

```python
{"recordId":"08","modelInput":{"anthropic_version":"bedrock-2023-05-31","system":"(system text)","messages":[{"role":"user","content":[{"image":{"format":"jpg","source":{"s3Uri":"s3://signal-8-data-creation-testing/Data/image8.jpg"}}}]}]}}
{"recordId":"09","modelInput":{"anthropic_version":"bedrock-2023-05-31","system":"(system text)","messages":[{"role":"user","content":[{"image":{"format":"jpg","source":{"s3Uri":"s3://signal-8-data-creation-testing/Data/image9.jpg"}}}]}]}}
```

Refer to the [start_invocation_job](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_model_invocation_job.html) function to start a batch inference job. You can poll the status using the [get_model_invocation_job](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/get_model_invocation_job.html). The correct order of statuses is Submitted>Validating>Scheduled>inProgress>Completed.

Output of the inference job is a jsonl file, `input.jsonl.out` located in a S3 directory mentioned in the S3DataConfig parameter in the `create_model_invocation_job`. Outputs are under 'modelOutput' at the same level as 'modelInput' of the input JSONL file.

## Fine tuning

You can either run the `fine_tuning.py` script directly (of course you need to alter the directory paths to your required paths) or run it through a training job using sagemaker. Edit the `OUTPUT_DIR` and `CHECKPOINT_DIR` according to your needs if you're not using a training job.

## Miscellneaous
