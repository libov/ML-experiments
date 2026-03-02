import mlflow

from ..utils.datasets import cifar10
from ..utils.metrics import get_accuracy
from ..models.Autoencoder import Autoencoder 
from ..training.train_autoencoder import train_autoencoder

#logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri(
    "http://localhost:5000"
)
mlflow.set_experiment("Autoencoder-CIFAR10")
# mlflow.enable_system_metrics_logging()

train_loader, val_loader, test_loader = cifar10()

all_params = []
base_params = {
    "latent_dim": 256,
    "dropout": 0.0,
    "epochs": 101,
    "learning_rate": 1e-3,
    "reduce_lr": "cosine_annealing"
}

#latent_dim_values = [10, 100, 256] # 10 doesn't really work; 100 and 256 seem to be similar, at least until epoch = 20
latent_dim_values = [100, 256]
for latent_dim in latent_dim_values:
    params = base_params.copy()
    params["latent_dim"] = latent_dim
    all_params.append(params)

for params in all_params:
    with mlflow.start_run(run_name = f"latent_dim-{params['latent_dim']}-lr-{params['learning_rate']}-dropout-{params['dropout']}-epochs-{params['epochs']}"):

        mlflow.log_params(params)
        autoencoder = Autoencoder(dropout=params["dropout"], latent_dim=params["latent_dim"])
        final_model_id = train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=params["epochs"], lr=params["learning_rate"], reduce_lr = params["reduce_lr"])
