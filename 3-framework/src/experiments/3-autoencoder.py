import mlflow

from ..utils.datasets import cifar10, mnist
from ..utils.metrics import get_accuracy
from ..models.Autoencoder import Autoencoder 
from ..training.train_autoencoder import train_autoencoder

#logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment("Autoencoder-CIFAR10")
train_loader, val_loader, test_loader = cifar10()
rgb = True

#mlflow.set_experiment("Autoencoder-MNIST")
#train_loader, val_loader, test_loader = mnist()
#rgb = False

# mlflow.enable_system_metrics_logging()

all_params = []
base_params = {
    "latent_dim": 256,
    "dropout": 0.0,
    "epochs": 21,
    "learning_rate": 1e-3,
    "reduce_lr": "cosine_annealing"
}

#latent_dim_values = [10, 100, 256] # 10 doesn't really work; 100 and 256 seem to be similar, at least until epoch = 20
#latent_dim_values = [100, 256] # 100m vs 256 seems to be very similar, even with 100 epochs
latent_dim_values = [100]
for latent_dim in latent_dim_values:
    params = base_params.copy()
    params["latent_dim"] = latent_dim
    all_params.append(params)

for params in all_params:
    with mlflow.start_run(run_name = f"latent_dim-{params['latent_dim']}-lr-{params['learning_rate']}-dropout-{params['dropout']}-epochs-{params['epochs']}-reduce_lr-{params['reduce_lr']}"):
        mlflow.log_params(params)
        autoencoder = Autoencoder(dropout=params["dropout"], latent_dim=params["latent_dim"], rgb=rgb)
        final_model_id = train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=params["epochs"], lr=params["learning_rate"], reduce_lr = params["reduce_lr"])
