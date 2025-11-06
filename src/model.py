import torch
import torch.nn as nn
from torchvision import models

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Load pretrained ResNet50 backbone
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classifier
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        # Add our custom projection head
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        return x
