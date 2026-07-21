import torch
import torch.nn as nn

import mlflow

import time as time

from ..models.DDPM import log_ddpm_images

def train_classifier(model, train_loader, val_loader, num_epochs, optimizer_name="adam", lr=1e-3, lr_scheduler = None, eta_min=0.0, weight_decay=0.0, task="classification"):

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task == "autoencoder":
        criterion = nn.MSELoss()
    elif task == "vae":
        criterion = nn.MSELoss(reduction='sum') # to make sure that scaling of the reconstruction loss is consistent with the KL divergence term
                                                # (i.e. we sum over pixels, not average over them). However we have to take care to average over the batch dimension.
    elif task == 'ddpm':
        criterion = None   # DDPM computes its own loss
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Add learning rate scheduler
    scheduler = None
    if lr_scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    elif lr_scheduler == "multi_step":
        milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError(f"Unsupported LR scheduler: {lr_scheduler}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start = time.time()

    for epoch in range(num_epochs):
        ### training pass
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if task == "classification":
                pred_labels = model(images)
                loss = criterion(pred_labels, labels)
            elif task == "autoencoder":
                reconstructed_images = model(images)
                loss = criterion(reconstructed_images, images)  # Use images as both input and target for autoencoder
            elif task == "vae":
                mu, log_var, reconstructed_images = model(images)
                recon_loss = criterion(reconstructed_images, images)/images.size(0) # Perform averaging over batch dim, since we used reduction='sum'
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean() # sum over latent dim (not average!), then average over the batch
                loss = recon_loss + kl_loss
            elif task == 'ddpm':
                loss = model.loss(images)
            else:
                raise ValueError(f"Unsupported task: {task}")

            loss.backward()
            optimizer.step()

        ### check performance on the validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if task == "classification":
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                elif task == "autoencoder":
                    reconstructed_images = model(images)
                    loss = criterion(reconstructed_images, images)  # Use images as both input and target for autoencoder
                elif task == "vae":
                    mu, log_var, reconstructed_images = model(images)
                    recon_loss = criterion(reconstructed_images, images)/images.size(0) # Perform averaging over batch dim, since we used reduction='sum'
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean() # sum over latent dim (not average!), then average over the batch
                    loss = recon_loss + kl_loss
                elif task == 'ddpm':
                    loss = model.loss(images)

                val_loss += loss.item()
        val_loss /= len(val_loader)

        if task == "classification":
            val_acc = correct / total
        else:
            val_acc = 0.0  # For autoencoder and VAE, there is no classification accuracy

        ### check performance on the training set
        model.eval()
        train_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                if task == "classification":
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                elif task == "autoencoder":
                    reconstructed_images = model(images)
                    loss = criterion(reconstructed_images, images)  # Use images as both input and target for autoencoder
                elif task == "vae":
                    mu, log_var, reconstructed_images = model(images)
                    recon_loss = criterion(reconstructed_images, images)/images.size(0) # Perform averaging over batch dim, since we used reduction='sum'
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean() # sum over latent dim (not average!), then average over the batch
                    loss = recon_loss + kl_loss
                elif task == 'ddpm':
                    loss = model.loss(images)

                train_loss += loss.item()
        train_loss /= len(train_loader)

        if task == "classification":
            train_acc = correct / total
        else:
            train_acc = 0.0  # For autoencoder and VAE, there is no classification accuracy

        # Log a checkpoint every 10 epochs
        model_id = None
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Log model checkpoint with step parameter
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"model-checkpoint-{epoch}",
                step=epoch
            )
            print(f"Epoch {epoch}, Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {val_acc * 100:.2f}%")
            model_id = model_info.model_id

        if task== 'ddpm' and (epoch % 10 == 0 or epoch == num_epochs - 1 or 0 < epoch < 10):
            log_ddpm_images(model, epoch, device)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]['lr'],
            },
            step=epoch,
            model_id=model_id,
        )

        scheduler.step()

    print(f'Training complete in {time.time()-start:.4f}s.')
    return model_id
