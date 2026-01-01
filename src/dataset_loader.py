from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir="data/caltech101/101_ObjectCategories", batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    
    return train_loader, val_loader, len(full_dataset.classes)
