# azure_submit.py
import argparse

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes

def parse_arguments():
    parser = argparse.ArgumentParser(description="Submit job to Azure ML")
    parser.add_argument("--large_node", action="store_true",  help="Run on a large (and expensive!) node for full training. If not set, runs on a smaller node for testing.")
    return parser.parse_args()

args = parse_arguments()

# 1. Connects automatically using the credentials from `az login` and targets from `config.json`
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

if args.large_node:
    print("Running in production mode. Using larger compute instance for full training.")
    cluster_name = "a100-cluster"
    node_size = "Standard_NC24ads_A100_v4" # 1x A100 80GB
    tier = "LowPriority"
else:
    print("Running in test mode. Using smaller compute instance for faster execution.")
    cluster_name = "debug-cluster"
    node_size = "Standard_D2s_v3"  # For testing, use a smaller instance type
    tier = "dedicated"

print(f"Using cluster: {cluster_name} with node size: {node_size} and tier: {tier}")

compute_target = AmlCompute(
    name=cluster_name,
    size=node_size,
    tier=tier,
    min_instances=0,
    max_instances=1,
)
ml_client.compute.begin_create_or_update(compute_target).result()

# get the data asset we created in azure_upload_data.py
data_asset = ml_client.data.get("cifar10_dataset", version="1.0.0")

# 2. Package and configure the command
job = command(
    code=".",   
    command="python -m src.experiments.train --task gan --experiment_name GAN-CIFAR10 --dataset cifar10 --norm scale_neg1_1 --dropout 0.0 --epochs 3 --optimizer adam --learning_rate 0.00001 --nruns 1 --latent_dim 100 --batch_size 64 --data_path ${{inputs.cifar_data}}",
    environment="azureml:AzureML-acpt-pytorch-2.2-cuda12.1@latest",
    compute=cluster_name,
    inputs={
            "cifar_data": Input(
                path=data_asset.id,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT
            )
        },
    # FORCE Python to stream logs to Azure instantly
    environment_variables={
        "PYTHONUNBUFFERED": "1"
    }
)

# 3. Submit and block terminal until finished to see logs stream live
returned_job = ml_client.jobs.create_or_update(job)
print(f"Tracking URL: {returned_job.studio_url}")

# This line links your terminal to the cloud container's stdout streams
ml_client.jobs.stream(returned_job.name)
