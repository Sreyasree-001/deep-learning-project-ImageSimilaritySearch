# AI-Powered Image Similarity Search & Recommendation System
A deep learning–based Content-Based Image Retrieval (CBIR) system that retrieves and recommends visually similar images using **ResNet50 embeddings** and **cosine similarity**.  
Built on the **Caltech-101 dataset** and powered by **PyTorch**.

## Project Overview
Traditional keyword-based image search fails to capture actual visual similarity.  
This project uses **deep convolutional neural networks (CNNs)** to learn semantic visual embeddings from images and perform **accurate similarity search & recommendation**.

Key features:
- Fine-tuned ResNet50 model
- 2048-dimensional deep embeddings
- Fast Top-K similarity retrieval
- Cosine similarity ranking
- Highly accurate (≈ **96% validation accuracy**)

## System Architecture


## Dataset Details
- Dataset Name: Caltech101
- Source: California Institute of Technology
- Number of Images: 8,677
- Images vary between 40–800 per class
- Number of Classes: 101 object categories
- Includes 1 additional background class
- Content Description: Diverse real-world objects
- Examples: insects, vehicles, instruments, everyday items etc.

## Technologies Used
- Language : Python 3 
- Framework : PyTorch 
- Model : ResNet50 
- Libraries : torchvision, numpy, PIL, tqdm 
- GPU : NVIDIA RTX 3050 |
- IDE : VS Code |

## Project Structure
    ├── main.py
    ├── auth.py
    ├── users.db
    ├── requirements.txt
    ├── dataset_loader.py
    ├── fine_tune.py
    ├── feature_extraction.py
    ├── similarity_utils.py
    ├── embeddings_20epochs/
    │ ├── train_embeddings.npy
    │ └── train_image_paths.npy
    ├── model/
    │ └── resnet50_finetuned_20epochs.pth
    └── data/
        └── caltech101/


