import os
import mlflow
import torch
from torchvision.utils import save_image

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from ..utils.datasets import cifar10
from ..utils.datasets import mnist

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

#client = mlflow.MlflowClient()
#logged_model = client.get_logged_model("m-79592f90f043407dbead3d80e10a6cc5")
#model_id="m-79592f90f043407dbead3d80e10a6cc5"
#model_id="m-495e3705491643afaaf222642c4341ac"
model_id="m-b714940d28f04b36baa6e3958a74611a"
model_uri = f"models:/{model_id}"
model = mlflow.pytorch.load_model(model_uri)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

output_dir = "plots/autoencoder"
os.makedirs(output_dir, exist_ok=True)

#_, _, test_loader = cifar10()
_, _, test_loader = mnist()
latent_vectors = []
labels = []

first_batch = True

with torch.no_grad():
    for images, labels_ in test_loader:
        images = images.to(device)
        reconstructed_images = model(images)
        if first_batch:
            for idx, (original, reconstructed) in enumerate(zip(images, reconstructed_images)):
                # Concatenate original and reconstructed horizontally
                combined = torch.cat([original.unsqueeze(0), reconstructed.unsqueeze(0)], dim=3)
                save_image(combined, os.path.join(output_dir, f"pair_{idx:04d}.png"))
            first_batch = False

        z = model.encode(images)
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