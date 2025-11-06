import torch
from tqdm import tqdm
import numpy as np
import os
from dataset_loader import get_dataloaders
from model import EmbeddingModel

def extract_embeddings(output_dir="embeddings", batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingModel().to(device)
    model.eval()

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    all_loaders = [("train", train_loader), ("val", val_loader)]

    with torch.no_grad():
        for split, loader in all_loaders:
            embeddings_list, labels_list = [], []
            for images, labels in tqdm(loader, desc=f"Extracting {split} embeddings"):
                images = images.to(device)
                embs = model(images)
                embeddings_list.append(embs.cpu().numpy())
                labels_list.append(labels.numpy())

            embeddings = np.concatenate(embeddings_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)

            np.save(os.path.join(output_dir, f"{split}_embeddings.npy"), embeddings)
            np.save(os.path.join(output_dir, f"{split}_labels.npy"), labels)
            print(f"✅ Saved {split} embeddings: {embeddings.shape}")

if __name__ == "__main__":
    extract_embeddings()
