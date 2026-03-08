import torch
import torch.nn as nn

import mlflow

import time as time

def train_autoencoder(model, train_loader, val_loader, num_epochs, lr=1e-3, reduce_lr = None, eta_min=0.0, task="autoencoder"):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    criterion = nn.MSELoss()

    # Add cosine annealing scheduler
    scheduler = None
    if reduce_lr == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start = time.time()
    
    for epoch in range(num_epochs):
        ### training pass

        # check if need to reduce the lr
        if reduce_lr is not None and isinstance(reduce_lr, list) and len(reduce_lr) > 0:
            if epoch in reduce_lr:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.1
                print(f"Epoch {epoch}: LR reduced to {optimizer.param_groups[0]['lr']}")
        
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            if task == "autoencoder":
                reconstructed_images = model(images)
                loss = criterion(reconstructed_images, images)  # Use images as both input and target for autoencoder
            elif task == "vae":
                mu, log_var, reconstructed_images = model(images)
                recon_loss = criterion(reconstructed_images, images)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()  # Average KL loss over the batch
                loss = recon_loss + kl_loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()             

        ### check performance on the validation set
        model.eval()
        val_loss = 0.0
    
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                if task == "autoencoder":
                    reconstructed_images = model(images)
                    loss = criterion(reconstructed_images, images)  # Use images as both input and target for autoencoder
                elif task == "vae":
                    mu, log_var, reconstructed_images = model(images)
                    recon_loss = criterion(reconstructed_images, images)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()  # Average KL loss over the batch
                    loss = recon_loss + kl_loss

                val_loss += loss.item()             

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        # Log a checkpoint every 10 epochs
        model_id = None
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Log model checkpoint with step parameter
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"model-checkpoint-{epoch}",
                step=epoch
            )
            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}")
            model_id = model_info.model_id

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            },
            step=epoch,
            model_id=model_id,
        )

        if reduce_lr == "cosine_annealing":
            scheduler.step()

    print(f'Training complete in {time.time()-start:.4f}s.')
    return model_id
