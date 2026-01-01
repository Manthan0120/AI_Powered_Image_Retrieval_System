#!/bin/bash

echo "Starting feature extraction..."

python -m utils.precompute_features || { echo "Feature extraction failed"; exit 1; }

echo "Feature extraction completed."

echo "Building FAISS index..."

python -c "
import os
from utils.faiss_utils import create_faiss_index

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)
create_faiss_index(features_path='data/features.npy', index_path='data/faiss_index.bin')
print('FAISS index created at data/faiss_index.bin')
" || { echo "FAISS index creation failed"; exit 1; }

echo "FAISS index generation complete."

