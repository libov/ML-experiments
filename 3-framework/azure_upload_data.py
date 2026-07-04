from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

cifar_data = Data(
    name="cifar10_dataset",
    version="1.0.0",
    description="Local CIFAR-10 batch files",
    path="./data",
    type=AssetTypes.URI_FOLDER,
)

ml_client.data.create_or_update(cifar_data)
print("Data Asset created successfully!")
