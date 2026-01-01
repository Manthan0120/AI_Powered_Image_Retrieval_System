import streamlit as st
import torch
import faiss
from PIL import Image
from torchvision import transforms
from src.model import ResNetTransferModel
from utils.display_utils import display_query_image, display_results
import numpy as np
import json

@st.cache_data
def load_faiss_index_and_paths(index_path='data/faiss_index.bin', paths_path='data/image_paths.json'):
    index = faiss.read_index(index_path)
    with open(paths_path, 'r') as f:
        image_paths = json.load(f)
    return index, image_paths

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_model(weights_path='weights/model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetTransferModel(feature_dim=512)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def extract_query_feature(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    embedding = embedding.cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize query features
    return embedding

def query_similar_images(query_feature, index, image_paths, top_k=5):
    distances, indices = index.search(query_feature, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((image_paths[idx], dist))
    return results

def main():
    st.title("AI-Powered Image Retrieval System")

    st.sidebar.title("Options")
    top_k = st.sidebar.slider(label="Number of Results", min_value=1, max_value=10, value=5)

    uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert('RGB')

        model, device = load_model()
        index, image_paths = load_faiss_index_and_paths()
        transform = get_transform()

        display_query_image(query_image)

        query_feature = extract_query_feature(query_image, model, transform, device)
        results = query_similar_images(query_feature, index, image_paths, top_k=top_k)

        display_results(results)
    else:
        st.write("Please upload an image to search.")

if __name__ == "__main__":
    main()
