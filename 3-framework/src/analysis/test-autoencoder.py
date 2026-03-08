import argparse
import os
import mlflow
import torch
from torchvision.utils import save_image, make_grid

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from ..utils.datasets import cifar10, mnist, denormalize_cifar10
from ..utils.mlflow import load_latest_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classifier training")
    parser.add_argument("--experiment_name",    type=str,   default="ResNet-CIFAR10",                                   help="Name of the MLflow experiment")
    parser.add_argument("--run_name",           type=str,   default="mnist-dropout-0.0-optimizer-adam-lr-0.001-run0",   help="Name of the MLflow run")
    parser.add_argument("--plots_dir",          type=str,   default="plots/autoencoder",                                help="Directory to save plots and generated images")
    parser.add_argument("--task",               type=str,   default="autoencoder",                                      help="Task type (e.g., 'autoencoder', 'vae')")

    return parser.parse_args()

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

args = parse_arguments()

model, params = load_latest_model(args.experiment_name, args.run_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

output_dir = args.plots_dir
os.makedirs(output_dir, exist_ok=True)

dataset = params.get("dataset", "unknown")
print(f"Loaded model trained on dataset: {dataset}")
if dataset == "mnist":
    _, _, test_loader = mnist()
    denormalize = lambda x: x  # do nothing
elif dataset == "cifar10":
    _, _, test_loader = cifar10()
    denormalize = denormalize_cifar10
else:
    raise ValueError(f"Unknown dataset '{dataset}' in loaded model parameters")

latent_vectors = []
labels = []
first_batch = True
with torch.no_grad():
    for images, labels_ in test_loader:
        images = images.to(device)
        if args.task == "autoencoder":
            reconstructed_images = model(images)
        elif args.task == "vae":
            _, _, reconstructed_images = model(images)
        images_denorm = denormalize(images)
        reconstructed_images_denorm = denormalize(reconstructed_images)
        if first_batch:
            for idx, (original, reconstructed) in enumerate(zip(images, reconstructed_images)):
                # Concatenate original and reconstructed horizontally
                combined = torch.cat([original.unsqueeze(0), reconstructed.unsqueeze(0)], dim=3)
                save_image(combined, os.path.join(output_dir, f"reconstruction_pair_{idx:04d}.png"))
            for idx, (original, reconstructed) in enumerate(zip(images_denorm, reconstructed_images_denorm)):
                # Concatenate original and reconstructed horizontally
                combined = torch.cat([original.unsqueeze(0), reconstructed.unsqueeze(0)], dim=3)
                save_image(combined, os.path.join(output_dir, f"reconstruction_denorm_pair_{idx:04d}.png"))

            # Interleave originals and reconstructions: [orig1, recon1, orig2, recon2, ...]
            paired_tensors = torch.stack([images_denorm, reconstructed_images_denorm], dim=1).view(-1, *images.shape[1:])

            # make_grid handles layout and padding automatically (nrow=2 creates 2 columns)
            grid = make_grid(paired_tensors, nrow=8, padding=2)
            save_image(grid, os.path.join(output_dir, "reconstruction_denorm_batch.png"))

            first_batch = False

        if args.task == "autoencoder":
            z = model.encode(images)
        elif args.task == "vae":
            z, _ = model.encode(images)

        latent_vectors.append(z.cpu().numpy())
        labels.append(labels_.numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.concatenate(labels, axis=0)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(latent_vectors)

# Plot and save with color coding by label
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter, label='True Label')
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of Latent Representations")
plt.savefig(os.path.join(output_dir, "tsne_latent.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"Processed {len(latent_vectors)} images. t-SNE plot saved.")

# generate some images
with torch.no_grad():
    generated_image = model.decode(z[0].unsqueeze(0)) # add batch dimension
    save_image(generated_image, os.path.join(output_dir, f"generated_from_encoder_latent1.png"))
    generated_image = model.decode(z[1].unsqueeze(0)) # add batch dimension
    save_image(generated_image, os.path.join(output_dir, f"generated_from_encoder_latent2.png"))
    generated_image = model.decode(((z[0]+z[1])/2).unsqueeze(0)) # interpolate between two latent vectors
    save_image(generated_image, os.path.join(output_dir, f"generated_from_interpolated_latent.png"))
    z = torch.randn(100, z.shape[1]).to(device) # random latent vectors
    generated_image = model.decode(z)
    grid = make_grid(generated_image, nrow=10, padding=0)
    save_image(grid, os.path.join(output_dir, "generated_from_random_latents.png"))
