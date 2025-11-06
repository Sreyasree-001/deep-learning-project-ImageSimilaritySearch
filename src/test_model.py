import torch
from dataset_loader import get_dataloaders
from model import EmbeddingModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_dataloaders(batch_size=8)

    model = EmbeddingModel().to(device)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    embeddings = model(images)

    print("Embeddings shape:", embeddings.shape)  # expect (8, 512)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # ✅ safe for Windows
    main()