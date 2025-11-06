# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split
# from PIL import Image

# class ToRGB:
#     def __call__(self, img):
#         return img.convert("RGB")

# def get_dataloaders(batch_size=32, val_split=0.2):
#     print("Torch version:", torch.__version__)
#     print("CUDA available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("GPU name:", torch.cuda.get_device_name(0))   
#     """
#     Loads Caltech-101 dataset with preprocessing and returns train/val loaders
#     """
#     # ImageNet normalization (mean/std)
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         ToRGB(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     dataset = datasets.Caltech101(root="data", download=False, transform=preprocess)
#     total_size = len(dataset)

#     print("Total dataset size:", total_size)
#     val_size = int(val_split * total_size)
#     train_size = total_size - val_size

#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#     print(f"✅ Loaded Caltech-101: {total_size} images ({train_size} train / {val_size} val)")
#     return train_loader, val_loader, len(train_dataset.classes)

# # if __name__ == "__main__":
# #     get_dataloaders()

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

    # Load entire dataset first
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    # Split into train and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Return class count using the original dataset
    return train_loader, val_loader, len(full_dataset.classes)
