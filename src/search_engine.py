import torch
import numpy as np

class TorchImageSearchEngine:
    def __init__(self, embeddings_path, labels_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float32).to(self.device)
        self.labels = np.load(labels_path)
        # Normalize once for cosine similarity
        self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
        print(f"✅ Torch search engine initialized on {self.device} with {self.embeddings.shape[0]} embeddings.")

    def search(self, query_embedding, k=5):
        query = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
        query = torch.nn.functional.normalize(query, p=2, dim=1)
        # Compute cosine similarities (dot product)
        similarities = torch.mm(query, self.embeddings.T).squeeze(0)
        topk = torch.topk(similarities, k=k)
        indices = topk.indices.detach().cpu().numpy()
        scores = topk.values.detach().cpu().numpy()
        return indices, scores
