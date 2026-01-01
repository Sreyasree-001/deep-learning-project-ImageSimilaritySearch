import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_loader import get_dataloaders
from tqdm import tqdm
import numpy as np
import os

MODEL_SAVE_PATH = "model/resnet50_finetuned_20epochs.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    train_loader, val_loader, num_classes = get_dataloaders(batch_size=batch_size)
    print(f"Loaded Caltech-101: {len(train_loader.dataset)+len(val_loader.dataset)} images")


    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_val_acc = 0.0
    patience = 5  # stop if val acc doesn’t improve for 5 epochs
    wait = 0

    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved (val acc improved to {val_acc:.4f})")
            wait = 0
        else:
            wait += 1
            print(f"No improvement for {wait} epochs.")
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
