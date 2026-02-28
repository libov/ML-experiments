import torch

def get_accuracy(model, loader):

    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)             # Number of samples in batch
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
