import torch
import numpy as np
from model import EmbeddingModel
from dataset_loader import get_dataloaders
from search_engine import TorchImageSearchEngine
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load one sample image
    train_loader, _ = get_dataloaders(batch_size=1)
    images, labels = next(iter(train_loader))
    image = images[0].unsqueeze(0).to(device)

    # Get embedding
    model = EmbeddingModel().to(device)
    model.eval()
    with torch.no_grad():
        query_emb = model(image).cpu().numpy()

    # Initialize search engine
    engine = TorchImageSearchEngine("embeddings/train_embeddings.npy", "embeddings/train_labels.npy", device=device)

    # Search for top-5 similar images
    indices, scores = engine.search(query_emb, k=5)
    print("Top 5 similar image indices:", indices)
    print("Similarity scores:", scores)

    # dataset = datasets.Caltech101(root="data", download=False, transform=transforms.ToTensor())
    # plt.figure(figsize=(12, 3))
    # plt.subplot(1, 6, 1)
    # plt.imshow(dataset[indices[0]][0].permute(1, 2, 0))
    # plt.title("Query")

    # for i, idx in enumerate(indices):
    #     plt.subplot(1, 6, i+2)
    #     img, _ = dataset[idx]
    #     plt.imshow(img.permute(1, 2, 0))
    #     plt.axis('off')

    # plt.show()

if __name__ == "__main__":
    main()
    