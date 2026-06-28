from torchsummary import summary
import mlflow
import matplotlib.pyplot as plt
import logging
import argparse

from ..utils.datasets import cifar10, mnist
from ..utils.metrics import get_accuracy
from ..models.ResNet import ResNetCIFAR10, ResNetMNIST
from ..models.Autoencoder import AutoencoderCIFAR10, AutoencoderMNIST
from ..models.VariationalAutoencoder import VariationalAutoencoderCIFAR10, VariationalAutoencoderMNIST
from ..models.GAN import GANMNIST, GANCIFAR10
from ..training.train_classifier import train_classifier
from ..training.train_autoencoder import train_autoencoder
from ..training.train_gan import train_gan

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classifier training")
    parser.add_argument("--experiment_name",    type=str,   default="ResNet-CIFAR10",   help="Name of the MLflow experiment")
    parser.add_argument("--task",               type=str,   default="classification",   help="Task to perform: classification, autoencoder, ...")
    parser.add_argument("--dataset",            type=str,   default="cifar10",          help="Dataset to use for training")
    parser.add_argument("--norm",               type=str,   default="standard",         help="Normalization type. Possible values: 'standard', 'scale_0_1', 'scale_neg1_1'. Generally, use 'standard' for classification, 'scale_0_1' for MNIST AE/VAE,  'scale_neg1_1' for GANs.")
    parser.add_argument("--dropout",            type=float, default=0.0,                help="Dropout rate for residual blocks")
    parser.add_argument("--epochs",             type=int,   default=51,                 help="Number of training epochs")
    parser.add_argument("--optimizer",          type=str,   default="adam",             help="Optimizer")
    parser.add_argument("--learning_rate",      type=float, default=1e-3,               help="Learning rate")
    parser.add_argument("--lr_scheduler",       type=str,   default="cosine_annealing", help="LR reduction strategy")
    parser.add_argument("--eta_min",            type=float, default=0.0,                help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight_decay",       type=float, default=0.0,                help="Weight decay for AdamW and SGD optimizers")
    parser.add_argument("--latent_dim",         type=int,   default=100,                help="Latent dimension for autoencoder, variational autoencoder, GAN.")
    parser.add_argument("--nruns",              type=int,   default=1,                  help="Number of experiment runs")

    return parser.parse_args()

def main():
    args = parse_arguments()

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )
    mlflow.set_experiment(args.experiment_name)

    # For now we will use random crops only for classification; in future we might want to pass this as an argument, along with more complete set of data augmentation options.
    # Check the data loaders in the datasets.py file for exact details on data augmentation for different datasets and normalization types.
    if args.task == "classification":
        include_crops = True
    else:
        include_crops = False

    # Get the data
    if args.dataset == "cifar10":
        train_loader, val_loader, test_loader = cifar10(norm=args.norm, include_crops = include_crops)
    elif args.dataset == "mnist":
        train_loader, val_loader, test_loader = mnist(norm=args.norm, include_crops = include_crops)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Execute for each run
    for run in range(args.nruns):
        with mlflow.start_run(run_name = f"{args.dataset}-{args.norm}-dropout_{args.dropout}-{args.optimizer}-lr_{args.learning_rate}-weight_decay_{args.weight_decay}-run{run}"):

            # Log parameters to MLflow
            params = {
                "task": args.task,
                "dataset": args.dataset,
                "norm": args.norm,
                "include_crops": include_crops,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "weight_decay": args.weight_decay,
                "eta_min": args.eta_min,
                "latent_dim": args.latent_dim,
                "run": run
            }
            mlflow.log_params(params)

            # Initialize the model based on the task and dataset
            if args.task == "classification":
                if args.dataset == "cifar10":
                    model = ResNetCIFAR10(dropout=args.dropout)
                elif args.dataset == "mnist":
                    model = ResNetMNIST(dropout=args.dropout)
                else:
                    raise ValueError(f"Unsupported dataset: {args.dataset}")

            elif args.task == "autoencoder":
                if args.dataset == "cifar10":
                    model = AutoencoderCIFAR10(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                elif args.dataset == "mnist":
                    model = AutoencoderMNIST(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                else:
                    raise ValueError(f"Unsupported dataset: {args.dataset}")

            elif args.task == "vae":
                if args.dataset == "cifar10":
                    model = VariationalAutoencoderCIFAR10(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                elif args.dataset == "mnist":
                    model = VariationalAutoencoderMNIST(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                else:
                    raise ValueError(f"Unsupported dataset: {args.dataset}")

            elif args.task == "gan":
                if args.dataset == "cifar10":
                    model = GANCIFAR10(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                elif args.dataset == "mnist":
                    model = GANMNIST(dropout=args.dropout, latent_dim=args.latent_dim, norm=args.norm)
                else:
                    raise ValueError(f"Unsupported dataset: {args.dataset}")

            else:
                raise ValueError(f"Unsupported task: {args.task}")

            # Run the training function based on the task
            if args.task in ["classification","autoencoder", "vae"]:
                final_model_id = train_classifier(model,
                                train_loader,
                                val_loader,
                                num_epochs=args.epochs,
                                optimizer_name=args.optimizer,
                                lr=args.learning_rate,
                                lr_scheduler = args.lr_scheduler,
                                eta_min=args.eta_min,
                                weight_decay=args.weight_decay,
                                task=args.task)
            elif args.task == "gan":
                final_model_id = train_gan(model,
                                           train_loader,
                                           num_epochs=args.epochs,
                                           lr_g=args.learning_rate,
                                           lr_d=args.learning_rate,
                                           n_discriminator_steps=5,
                                           optimizer_name=args.optimizer)

            # For classification calculate also the test set accuracy and log it to MLflow
            if args.task == "classification":
                test_accuracy = get_accuracy(model, test_loader)
                print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')
                mlflow.log_metric("test_accuracy", test_accuracy, model_id=final_model_id)


if __name__ == "__main__":
    main()
