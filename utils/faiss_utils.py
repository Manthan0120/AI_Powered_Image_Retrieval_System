import faiss
import numpy as np
import os

import faiss
import numpy as np
import os


def create_faiss_index(features_path='data/features.npy', index_path='data/faiss_index.bin'):
    features = np.load(features_path)
    feature_dim = features.shape[1]
    index = faiss.IndexFlatL2(feature_dim)
    index.add(features)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to '{index_path}'")


def load_faiss_index(index_path='data/faiss_index.bin'):
    return faiss.read_index(index_path)
