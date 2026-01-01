import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings():
    vectors = np.load("embeddings_20epochs/train_embeddings.npy")
    paths = np.load("embeddings_20epochs/train_image_paths.npy")
    return vectors, paths


def get_feature_extractor():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(
        torch.load("model/resnet50_finetuned_20epochs.pth", map_location=DEVICE)
    )

    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)


def find_similar_images(query_img, top_k=5):
    model = get_feature_extractor()
    vectors, paths = load_embeddings()

    img_tensor = preprocess_image(query_img)

    with torch.no_grad():
        q_vec = model(img_tensor).squeeze().numpy()

    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec)
    scores = np.dot(vectors, q_vec) / norms

    best_idx = np.argsort(scores)[::-1][:top_k]
    return [(paths[i], scores[i]) for i in best_idx]