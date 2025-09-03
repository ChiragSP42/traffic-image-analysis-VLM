from sagemaker.huggingface import HuggingFace # type: ignore
from sagemaker.pytorch import PyTorch # type: ignore
from aws_helpers.helpers import list_obj_s3, _get_s3_client

# IAM role with SageMaker and S3 access permissions
role = "arn:aws:iam::381492026108:role/SageMakerTrainingJobRole"

# Hyperparameters passed as command-line arguments to your script
S3_BUCKET = 'signal-8-flock'
IMAGE_FOLDER = 'Daytime'
# num_epochs = 10
# length_of_dataset = len(list_obj_s3(s3_client=_get_s3_client(),
#                                     bucket_name=S3_BUCKET,
#                                     folder_name=IMAGE_FOLDER))

# batch_size = 16
# steps_per_epoch = (length_of_dataset + batch_size - 1) // batch_size
# hyperparameters = {
#     "max_steps": steps_per_epoch * num_epochs,
#     "per_device_train_batch_size": 16,
#     "per_device_eval_batch_size": 16,
# }

huggingface_estimator = PyTorch(
    entry_point="image_text_training.py",      # Your training script name
    source_dir=".",                             # Directory containing the script and files
    instance_type="ml.g5.2xlarge",              # GPU instance for training
    instance_count=1,
    role=role,
    transformers_version='4.53.1',
    framework_version="2.0.1",
    py_version='py312',
)

# Input data channels - specify S3 URI(s) where your data is
train_input = "s3://signal-8-flock/"
val_input = "s3://signal-8-flock/"

huggingface_estimator.fit({'train': train_input, 'validation': val_input})
