import torch
import torch.nn as nn

import time as time

def train(model, train_loader, val_loader, num_epochs, lr=1e-3, reduce_lr = None):
    training_loss = []
    validation_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start = time.time()
    
    for epoch in range(num_epochs):
        ### training pass

        # check if need to reduce the lr
        if reduce_lr is not None:
            if (epoch+1) in reduce_lr:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.1
                print(f"Epoch {epoch+1}: LR reduced to {optimizer.param_groups[0]['lr']}")
        
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
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")
    
    print(f'Training complete in {time.time()-start:.4f}s.')
    return training_loss, validation_loss
