import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === CONFIG ===
    DATA_DIR = "data/caltech101/101_ObjectCategories"
    MODEL_PATH = "model/resnet50_finetuned_20epochs.pth"
    SAVE_DIR = "embeddings_20epochs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === LOAD MODEL ===
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()

    # === TRANSFORMS ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # === LOAD DATA ===
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings = []
    image_paths = []

    print("Extracting train embeddings...")
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            emb = model(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings)
    image_paths = [path for path, _ in dataset.samples]

    np.save(os.path.join(SAVE_DIR, "train_embeddings.npy"), embeddings)
    np.save(os.path.join(SAVE_DIR, "train_image_paths.npy"), np.array(image_paths))

    print(f"✅ Saved train embeddings: {embeddings.shape}")

if __name__ == "__main__":
    extract_embeddings()