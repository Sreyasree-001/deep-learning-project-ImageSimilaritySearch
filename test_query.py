import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

def main():
    # ---------------------
    # CONFIG
    # ---------------------
    MODEL_PATH = "model/resnet50_finetuned_20epochs.pth"
    EMB_PATH = "embeddings_20epochs/train_embeddings.npy"
    IMG_PATHS_PATH = "embeddings_20epochs/train_image_paths.npy"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------
    # LOAD MODEL
    # ---------------------
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)  # Caltech-101 has 101 classes
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final classifier
    model.eval().to(DEVICE)

    # ---------------------
    # TRANSFORM FUNCTION
    # ---------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ---------------------
    # LOAD QUERY IMAGE
    # ---------------------
    query_path = "test7.jpg"  # your uploaded image
    image = Image.open(query_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_emb = model(img_tensor).squeeze().cpu().numpy()

    # ---------------------
    # LOAD TRAIN EMBEDDINGS
    # ---------------------
    train_emb = np.load(EMB_PATH)
    train_paths = np.load(IMG_PATHS_PATH, allow_pickle=True)

    # ---------------------
    # COMPUTE SIMILARITY
    # ---------------------
    sims = cosine_similarity([query_emb], train_emb)[0]
    top_idx = np.argsort(sims)[::-1][:5]  # Top 5 similar images

    # ---------------------
    # SHOW RESULTS
    # ---------------------
    fig, axes = plt.subplots(1, 6, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for i, idx in enumerate(top_idx):
        img_path = train_paths[idx]
        sim_img = Image.open(img_path)
        axes[i+1].imshow(sim_img)
        axes[i+1].set_title(f"Sim: {sims[idx]:.3f}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()