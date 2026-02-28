import torch
import torch.nn as nn

import mlflow

import time as time

def train_classifier(model, train_loader, val_loader, num_epochs, lr=1e-3, reduce_lr = None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    criterion = nn.CrossEntropyLoss()

    # Add cosine annealing scheduler
    scheduler = None
    if reduce_lr == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

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
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
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
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
    
        val_loss /= len(val_loader)
        val_acc = correct / total     
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
            print(f"Epoch {epoch}, Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")
            model_id = model_info.model_id

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]['lr'],
            },
            step=epoch,
            model_id=model_id,
        )

        if reduce_lr == "cosine_annealing":
            scheduler.step()

    print(f'Training complete in {time.time()-start:.4f}s.')
    return model_id
