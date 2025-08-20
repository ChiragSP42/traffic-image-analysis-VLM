from sagemaker.huggingface import HuggingFace # type: ignore

# IAM role with SageMaker and S3 access permissions
role = "arn:aws:iam::381492026108:role/SageMakerTrainingJobRole"

# Hyperparameters passed as command-line arguments to your script
hyperparameters = {
    "epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
}

huggingface_estimator = HuggingFace(
    entry_point="training_job.py",      # Your training script name
    source_dir=".",                             # Directory containing the script and files
    instance_type="ml.p3.2xlarge",              # GPU instance for training
    instance_count=1,
    role=role,
    transformers_version='4.49',
    pytorch_version='2.5',
    py_version='py311',
    hyperparameters=hyperparameters, # type: ignore
    metric_definitions=[
        {'Name': 'train_loss', 'Regex': r'train_loss = ([0-9\.]+)'},
        {'Name': 'eval_loss', 'Regex': r'eval_loss = ([0-9\.]+)'},
        {'Name': 'epoch', 'Regex': r'epoch = ([0-9\.]+)'},
    ],
)

# Input data channels - specify S3 URI(s) where your data is
train_input = "s3://signal-8-data-creation-testing/"
val_input = "s3://signal-8-data-creation-testing/"

huggingface_estimator.fit({'train': train_input, 'validation': val_input})
