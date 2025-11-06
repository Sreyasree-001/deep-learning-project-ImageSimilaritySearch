import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dataset_loader import get_dataloaders
from search_engine import TorchImageSearchEngine

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=32)

    # Load embeddings and labels
    engine = TorchImageSearchEngine(
        "embeddings/train_embeddings.npy",
        "embeddings/train_labels.npy",
        device=device
    )

    # Load validation embeddings & labels for evaluation
    val_embeddings = np.load("embeddings/val_embeddings.npy")
    val_labels = np.load("embeddings/val_labels.npy")

    # Compute retrieval accuracy
    correct_top1, correct_top5 = 0, 0
    total = len(val_embeddings)

    print(f"🔍 Evaluating on {total} validation samples...")
    for i in tqdm(range(total)):
        query_emb = val_embeddings[i].reshape(1, -1)
        true_label = val_labels[i]

        indices, _ = engine.search(query_emb, k=5)
        retrieved_labels = engine.labels[indices]

        if retrieved_labels[0] == true_label:
            correct_top1 += 1
        if true_label in retrieved_labels:
            correct_top5 += 1

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total

    print(f"\n✅ Retrieval Accuracy Results:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")



if __name__ == "__main__":
    main()