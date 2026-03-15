import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def cifar10(norm="standard"):

    if norm == "standard":
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std  = (0.2470, 0.2435, 0.2616)
    elif norm == "gan":
        cifar_mean = (0.5, 0.5, 0.5)
        cifar_std  = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard' or 'gan'.")

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


def denormalize_cifar10(tensor, norm="standard"):
    """
    Reverses the CIFAR-10 normalization applied during data loading.
    Works for both single images (3, H, W) and batches (B, 3, H, W).
    """
    if norm == "standard":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(-1, 1, 1).to(tensor.device)
    elif norm == "gan":
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard' or 'gan'.")

    # 1. Reverse the normalization math
    denormalized = (tensor * std) + mean

    # 2. Clamp values strictly to [0.0, 1.0] to remove floating-point rounding errors
    return torch.clamp(denormalized, 0.0, 1.0)


def mnist(norm="standard", include_crops = True):
    if norm == "standard":
        mnist_mean = (0.1307,)
        mnist_std  = (0.3081,)
    elif norm == "gan":
        mnist_mean = (0.5,)
        mnist_std  = (0.5,)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard' or 'gan'.")

    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=4) if include_crops else transforms.Identity(),
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

def denormalize_mnist(tensor, norm="standard"):
    """
    Reverses the MNIST normalization applied during data loading.
    Works for both single images (1, H, W) and batches (B, 1, H, W).
    """
    if norm == "standard":
        mean = torch.tensor([0.1307]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.3081]).view(-1, 1, 1).to(tensor.device)
    elif norm == "gan":
        mean = torch.tensor([0.5]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.5]).view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard' or 'gan'.")

    # 1. Reverse the normalization math
    denormalized = (tensor * std) + mean

    # 2. Clamp values strictly to [0.0, 1.0] to remove floating-point rounding errors
    return torch.clamp(denormalized, 0.0, 1.0)
