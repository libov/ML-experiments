from torchsummary import summary
import mlflow
import matplotlib.pyplot as plt
import logging

from ..utils.datasets import cifar10
from ..utils.metrics import get_accuracy
from ..models.ResNet import ResNet 
from ..training.train_classifier import train_classifier

#logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri(
    "http://localhost:5000"
)
mlflow.set_experiment("ResNet-CIFAR10")
# mlflow.enable_system_metrics_logging()

train_loader, val_loader, test_loader = cifar10()

all_params = []
base_params = {
    "dropout_blocks": 0.0,
    "epochs": 51,
    "learning_rate": 1e-3,
    "reduce_lr": "cosine_annealing"
}

dropout_values = [0.0, 0.1, 0.3, 0.5]
for dropout in dropout_values:
    params = base_params.copy()
    params["dropout_blocks"] = dropout
    all_params.append(params)

for params in all_params:
    with mlflow.start_run(run_name = f"dropout-{params['dropout_blocks']}"):

        mlflow.log_params(params)

        resnet = ResNet(dropout_blocks=params["dropout_blocks"], nodes_final_layer=10)
        final_model_id = train_classifier(resnet, train_loader, val_loader, num_epochs=params["epochs"], lr=params["learning_rate"], reduce_lr = params["reduce_lr"])

        test_accuracy = get_accuracy(resnet, test_loader)
        print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')
        mlflow.log_metric("test_accuracy", test_accuracy, model_id=final_model_id)
