# AI-Powered Image Similarity Search & Recommendation System
A deep learning–based Content-Based Image Retrieval (CBIR) system that retrieves and recommends visually similar images using **ResNet50 embeddings** and **cosine similarity**.  
Built on the **Caltech-101 dataset** and powered by **PyTorch**.

## Project Demo Video
https://drive.google.com/file/d/10_ycUXxDlTay2mg9kGzNtilsIA71lJiq/view?usp=sharing

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
<img width="1420" height="711" alt="image" src="https://github.com/user-attachments/assets/9e4c5427-4ee8-4ce8-af51-cc1492da6a13" />


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

## Result and Observations
- Model Accuracy: 96% validation accuracy
- Early stopping after stable convergence
- Embedding Quality: 2048-dimensional feature vectors
- Captures high-level shapes, textures, patterns
- Similarity Retrieval: Top-5 most similar images retrieved
- Cosine similarity scores nearly 0.90 to 0.95
- Performance Efficiency
- Fast retrieval using pre-computed embeddings
- Stored as NumPy arrays for quick access
- Feature Space Behavior
- Similar-category images cluster closely
- Demonstrates strong semantic representation

## Sample Outputs
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7ec4b40f-19a3-4d77-8380-d8b7778203f0" />
<img width="1575" height="908" alt="image" src="https://github.com/user-attachments/assets/29523c22-26c5-4132-9a62-3d57c61f7622" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/7b3aa837-1222-4eb8-9b6f-59502442f421" />

