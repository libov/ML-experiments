import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

for exp in ["Autoencoder-CIFAR10", "Autoencoder-MNIST", "ResNet-CIFAR10", "ResNet-MNIST"]:
    exp = mlflow.get_experiment_by_name(exp)
    client.rename_experiment(
        experiment_id=exp.experiment_id,
        new_name=f"_{exp.name}-ARCHIVE"
    )
