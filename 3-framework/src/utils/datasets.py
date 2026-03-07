import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def cifar10():
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, transform=train_tf, download=True)
    test_dataset = datasets.CIFAR10(root='data', train=False, transform=test_tf, download=True)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    print(f"Splitting training data into {train_size} training and {val_size} validation samples.")
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def denormalize_cifar10(tensor):
    """
    Reverses the CIFAR-10 normalization applied during data loading.
    Works for both single images (3, H, W) and batches (B, 3, H, W).
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(-1, 1, 1).to(tensor.device)

    # 1. Reverse the normalization math
    denormalized = (tensor * std) + mean

    # 2. Clamp values strictly to [0.0, 1.0] to remove floating-point rounding errors
    return torch.clamp(denormalized, 0.0, 1.0)


def mnist():
    mnist_mean = (0.1307,)
    mnist_std  = (0.3081,)

    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=train_tf, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=test_tf, download=True)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    print(f"Splitting training data into {train_size} training and {val_size} validation samples.")
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
