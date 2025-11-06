import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_loader import get_dataloaders

def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    batch_size = 32
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    train_loader, val_loader, num_classes = get_dataloaders(batch_size=batch_size)

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze base layers

    # Replace classifier (fully connected layer)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ----------------------------
    # TRAINING SETUP
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"\n✅ Validation Accuracy: {val_acc:.2f}%")

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    torch.save(model.state_dict(), "model/resnet50_finetuned.pth")
    print("💾 Fine-tuned model saved as resnet50_finetuned.pth")

if __name__ == "__main__":
    main()