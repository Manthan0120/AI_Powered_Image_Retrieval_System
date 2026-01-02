# AI-Powered Image Retrieval System

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io/)

An image similarity search system trained on the **Caltech101 dataset** using **ResNet50** with **triplet loss**. Extracts deep embeddings, builds **FAISS** index for fast similarity search, and provides an interactive **Streamlit** web interface for querying visually similar images.

## 📁 Project Structure

```
caltech101/
├── data/                    # Features, FAISS index (faiss_index.bin, features_path.json)
├── weights/                 # Trained model weights (model.pth)
├── src/                     # Model architecture
│   ├── __init__.py
│   └── model.py            # ResNet50 transfer learning with triplet loss
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data_utils.py       # Dataset loading & triplet dataset
│   ├── image_utils.py      # Image preprocessing
│   ├── faiss_utils.py      # FAISS indexing utilities
│   └── precompute_features.py # Feature extraction script
├── app.py                  # Streamlit web interface
├── precompute.sh           # Feature extraction & indexing script
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install torch torchvision streamlit faiss-cpu pillow matplotlib numpy
```

### 2. Train the Model
```bash
# Split dataset into train/val
python -c "from utils.data_utils import split_dataset; split_dataset('path/to/caltech101')"

# Train model (outputs weights/model.pth)
python -c "from src.model import ResNetTransferModel, train_model; from utils.data_utils import TripletDataset; # ... (training code)"
```

### 3. Generate FAISS Index
```bash
chmod +x precompute.sh
./precompute.sh
```
*This extracts features using trained model and builds FAISS index in `data/`*

### 4. Run Streamlit App
```bash
streamlit run app.py
```

## 🛠️ Workflow

1. **Training**: ResNet50 with triplet loss learns to embed similar images close together
2. **Feature Extraction**: Precompute embeddings for all dataset images → `data/features.npy`
3. **Indexing**: Build FAISS L2 index → `data/faiss_index.bin`
4. **Querying**: Upload image → extract embedding → find nearest neighbors in FAISS index

## 🔍 Key Features

- **Modular Architecture**: Clean separation of model, data, utils, and app
- **Efficient Search**: FAISS provides sub-linear similarity search
- **Interactive UI**: Streamlit app with upload and visual results
- **Production Ready**: Precompute pipeline with shell script

## 📊 Results

The system achieves fast retrieval of visually similar images using Euclidean distance on normalized embeddings (equivalent to cosine similarity).

## 📝 Usage in App

1. Upload query image
2. View top-k retrieved images with distance scores
3. Compare visual similarity instantly

## 🔧 Customization

- Adjust `top_k` in app for more/fewer results
- Modify image size in `display_results()` function
- Retrain model with different hyperparameters in `src/model.py`

## 📚 Dependencies

```
torch>=2.0
torchvision>=0.15
streamlit>=1.28
faiss-cpu>=1.7
pillow>=10
matplotlib>=3.7
numpy>=1.24
```

## 🎓 Learning Objectives

This project demonstrates:
- Transfer learning with ResNet50
- Triplet loss for embedding learning
- FAISS for efficient similarity search
- Streamlit for ML app deployment
- Modular Python package structure

---

**Built for Caltech101 image retrieval challenge** 🚀
