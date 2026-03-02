##############
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

# Load pretrained Vision Transformer (trained on ImageNet-21K)
weights = ViT_B_16_Weights.IMAGENET21K_1K_V1  # Pretrained on 14M images!
model = models.vit_b_16(weights=weights)

print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: ~86 million parameters already learned from massive dataset!

# Inspect the model
print(model)
# Last layer: Linear(768, 1000)  ← Outputs 1000 ImageNet classes
############################################
# The pretrained model has:
# - Transformer blocks: Learned general features ✓
# - Classification head: 1000 ImageNet classes ✗

# Replace classification head
num_classes = 10  # CIFAR-10 has 10 classes

# Option 1: Simple replacement
model.heads = nn.Linear(model.heads.head.in_features, num_classes)

# Option 2: Add a small adapter
model.heads = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

print(model)
# Last layer now: Linear(768, 10) ← Outputs 10 CIFAR classes
############################################
# Freeze transformer backbone
for param in model.encoder.parameters():
    param.requires_grad = False

# Only train new classification head
model.heads.requires_grad_(True)

optimizer = torch.optim.Adam(model.heads.parameters(), lr=1e-3)
# ↑ Normal learning rate (only training small head)

for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Head quickly learns CIFAR-10 decision boundaries
############################################
# Start with frozen backbone, train head
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.heads.parameters(), lr=1e-3)

# Train head for 5 epochs
for epoch in range(5):
    train_epoch()

# Then unfreeze and fine-tune everything
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Very low LR

# Fine-tune for 10 more epochs
for epoch in range(10):
    train_epoch()
############################################
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader, random_split
import time

# ============= DATA =============
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=28),  # ViT expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='.', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform)

val_size = 5000
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ============= LOAD PRETRAINED MODEL =============
print("Loading pretrained ViT (trained on ImageNet-21K)...")
weights = ViT_B_16_Weights.IMAGENET21K_1K_V1
model = models.vit_b_16(weights=weights)

print(f"Pretrained parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============= ADAPT TO CIFAR-10 =============
# Replace classification head
model.heads = nn.Linear(model.heads.head.in_features, num_classes=10)

print(f"Model adapted for 10 classes")

# ============= TRANSFER LEARNING STRATEGY =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Phase 1: Train head only (frozen backbone)
print("\n" + "="*60)
print("PHASE 1: Train head only (backbone frozen)")
print("="*60)

for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.heads.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
for epoch in range(5):
    model.train()
    train_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    print(f"Phase 1 Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_phase1.pth')

# Phase 2: Fine-tune entire model (low learning rate)
print("\n" + "="*60)
print("PHASE 2: Fine-tune entire model (backbone unfrozen)")
print("="*60)

# Unfreeze all parameters
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Very low!
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()
    train_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    print(f"Phase 2 Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_phase2.pth')
    
    scheduler.step()

# ============= EVALUATE =============
model.load_state_dict(torch.load('best_model_phase2.pth'))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total

print(f"\n{'='*60}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"{'='*60}")
# Expected: ~97-99%!
############################################
def train_with_early_stopping(model, train_loader, val_loader, num_epochs, 
                              lr=1e-3, patience=10, reduce_lr=None):
    # ...existing code...
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # ...training code...
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model
############################################
def evaluate_model(model, train_loader, val_loader, test_loader, model_name):
    """Comprehensive model evaluation"""
    metrics = {}
    
    # Accuracies
    train_acc = get_accuracy_detailed(model, train_loader)
    val_acc = get_accuracy_detailed(model, val_loader)
    test_acc = get_accuracy_detailed(model, test_loader)
    
    metrics['train_acc'] = train_acc
    metrics['val_acc'] = val_acc
    metrics['test_acc'] = test_acc
    metrics['overfit_gap'] = train_acc - val_acc
    metrics['generalization_gap'] = val_acc - test_acc
    
    # Model complexity
    metrics['num_params'] = sum(p.numel() for p in model.parameters())
    
    # Stability (run multiple times in production)
    # For now, just note it
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Train Acc:        {train_acc*100:.2f}%")
    print(f"Val Acc:          {val_acc*100:.2f}%  ← PRIMARY METRIC")
    print(f"Test Acc:         {test_acc*100:.2f}%")
    print(f"Overfit Gap:      {metrics['overfit_gap']*100:.2f}%  (ideal: <5%)")
    print(f"Generalization:   {metrics['generalization_gap']*100:.2f}%")
    print(f"Parameters:       {metrics['num_params']:,}")
    print(f"{'='*50}\n")
    
    return metrics

def get_accuracy_detailed(model, loader):
    """Get accuracy and optionally confusion matrix"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total
############################################
# Create comparison table
comparison_df = pd.DataFrame({
    'Model': model_names,
    'Val Acc': val_accuracies,
    'Test Acc': test_accuracies,
    'Overfit Gap': overfit_gaps,
    'Params': param_counts
})
comparison_df = comparison_df.sort_values('Val Acc', ascending=False)
print(comparison_df)
############################################
def train_with_lr_scheduling(model, train_loader, val_loader, num_epochs, 
                             lr=1e-3, patience=5, factor=0.5):
    """
    Reduce LR when validation accuracy plateaus
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_acc = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                val_acc += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc /= sum(len(batch[1]) for batch in val_loader)
        train_loss /= len(train_loader)
        
        # **LR scheduling based on VALIDATION ACCURACY**
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Epoch {epoch+1}: Val Acc improved to {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Reduce learning rate
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = old_lr * factor
                optimizer.param_groups[0]['lr'] = new_lr
                patience_counter = 0
                print(f"Epoch {epoch+1}: Val Acc plateaued. LR: {old_lr:.2e} → {new_lr:.2e}")
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    return model

# Usage:
# model = SimpleCNNv3(dropout_conv_block=0.5, dropout_final_conv=0.1, dropout_fc=0.0)
# train_with_lr_scheduling(model, train_loader, val_loader, num_epochs=100, 
#                          lr=1e-3, patience=5, factor=0.5)
############################################
# More robust approach
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',          # **Maximize accuracy** (not minimize loss)
    factor=0.5,
    patience=5,
    verbose=True,
    min_lr=1e-7
)

for epoch in range(num_epochs):
    # ... training code ...
    
    # Step scheduler based on VAL ACCURACY
    scheduler.step(val_acc)  # Not val_loss!
############################################
############################################