import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def cifar10(norm="standard", include_crops = True, batch_size=64, data_path="./data"):

    if norm == "standard":
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std  = (0.2470, 0.2435, 0.2616)
    elif norm == "scale_0_1":
        cifar_mean = (0.0, 0.0, 0.0)
        cifar_std  = (1.0, 1.0, 1.0)
    elif norm == "scale_neg1_1":
        cifar_mean = (0.5, 0.5, 0.5)
        cifar_std  = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'.")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if include_crops else nn.Identity(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=train_tf, download=False)
    val_dataset = datasets.CIFAR10(root=data_path, train=True, transform=test_tf, download=False)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=test_tf, download=False)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    print(f"Splitting training data into {train_size} training and {val_size} validation samples.")

    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=g).tolist()

    train_subset = Subset(train_dataset, indices[:train_size])
    val_subset = Subset(val_dataset, indices[train_size:])

    print(f"Creating DataLoaders with batch size {batch_size}...")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def denormalize_cifar10(tensor, norm="standard"):
    """
    Reverses the CIFAR-10 normalization applied during data loading.
    Works for both single images (3, H, W) and batches (B, 3, H, W).
    """
    if norm == "standard":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(-1, 1, 1).to(tensor.device)
    elif norm == "scale_0_1":
        return tensor
    elif norm == "scale_neg1_1":
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'.")

    # 1. Reverse the normalization math
    denormalized = (tensor * std) + mean

    # 2. Clamp values strictly to [0.0, 1.0] to remove floating-point rounding errors
    return torch.clamp(denormalized, 0.0, 1.0)


def mnist(norm="standard", include_crops = True, batch_size=64):
    if norm == "standard":
        mnist_mean = (0.1307,)
        mnist_std  = (0.3081,)
    elif norm == "scale_0_1":
        mnist_mean = (0.0,)
        mnist_std  = (1.0,)
    elif norm == "scale_neg1_1":
        mnist_mean = (0.5,)
        mnist_std  = (0.5,)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'.")

    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=4) if include_crops else nn.Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std),
    ])

    # Train/Validation dataset and loader
    # NB. In order to prevent validation set having random crop augmentation, we need to create two datasets from the same training set, but different transforms....
    train_dataset = datasets.MNIST(root='data', train=True, transform=train_tf, download=True)
    val_dataset   = datasets.MNIST(root='data', train=True, transform=test_tf,  download=True)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    print(f"Splitting training data into {train_size} training and {val_size} validation samples.")
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=g).tolist()

    train_subset = Subset(train_dataset, indices[:train_size])
    val_subset = Subset(val_dataset, indices[train_size:])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Test dataset and loader
    test_dataset = datasets.MNIST(root='data', train=False, transform=test_tf, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def denormalize_mnist(tensor, norm="standard"):
    """
    Reverses the MNIST normalization applied during data loading.
    Works for both single images (1, H, W) and batches (B, 1, H, W).
    """
    if norm == "standard":
        mean = torch.tensor([0.1307]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.3081]).view(-1, 1, 1).to(tensor.device)
    elif norm == "scale_0_1":
        return tensor
    elif norm == "scale_neg1_1":
        mean = torch.tensor([0.5]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.5]).view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'.")

    # 1. Reverse the normalization math
    denormalized = (tensor * std) + mean

    # 2. Clamp values strictly to [0.0, 1.0] to remove floating-point rounding errors
    return torch.clamp(denormalized, 0.0, 1.0)


def imagenet(norm="standard", include_crops = True, batch_size=64, data_path="./data/imagenet"):

    if norm == "standard":
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
    elif norm == "scale_0_1":
        mean = (0.0, 0.0, 0.0)
        std  = (1.0, 1.0, 1.0)
    elif norm == "scale_neg1_1":
        mean = (0.5, 0.5, 0.5)
        std  = (0.5, 0.5, 0.5)
    else:
        raise ValueError(
            f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'."
        )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224) if include_crops else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_path}/train", transform=train_tf)
    val_dataset = datasets.ImageFolder(root=f"{data_path}/val", transform=val_tf)

    print(f"Creating DataLoaders with batch size {batch_size}...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,            # Mandatory for fast CPU-to-GPU transfer
        persistent_workers=True,    # Prevents delay between epochs
        prefetch_factor=2,          # Buffer size per worker
        drop_last=True              # Prevents batchnorm errors on incomplete final batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    return train_loader, val_loader, val_loader


def denormalize_imagenet(tensor, norm="standard"):
    """
    Reverses the ImageNet normalization applied during data loading.
    Works for both single images (3, H, W) and batches (B, 3, H, W).
    """
    if norm == "scale_0_1":
        return torch.clamp(tensor, 0.0, 1.0)
    elif norm == "standard":
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
    elif norm == "scale_neg1_1":
        mean_vals = [0.5, 0.5, 0.5]
        std_vals = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'standard', 'scale_0_1' or 'scale_neg1_1'.")

    # .view(-1, 1, 1) creates a (3, 1, 1) tensor.
    # PyTorch right-aligns broadcasting for (B, 3, H, W) automatically.
    mean = torch.tensor(
        mean_vals, device=tensor.device, dtype=tensor.dtype
    ).view(-1, 1, 1)
    std = torch.tensor(
        std_vals, device=tensor.device, dtype=tensor.dtype
    ).view(-1, 1, 1)

    return torch.clamp((tensor * std) + mean, 0.0, 1.0)

# call this function manually to download the Imagenette dataset before training, as torchvision's ImageFolder does not support automatic downloading
def download_imagenette():
    """
    Downloads the Imagenette dataset using torchvision's built-in functionality.
    """
    print("Downloading Imagenette dataset...")
    datasets.Imagenette(root="./data", size="full", split="train", download=True)
    datasets.Imagenette(root="./data", size="full", split="val", download=True)
