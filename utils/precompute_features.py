import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

from src.model import ResNetTransferModel
from utils.image_utils import transform

class FeatureExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path
        except UnidentifiedImageError:
            print(f"Warning: Skipping unreadable image {img_path}")
            return None, None

def custom_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)

def extract_features(image_folder, model_path='weights/model.pth', batch_size=64, device='cpu'):
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, f))
    print(f"Total images found: {len(image_paths)}")

    dataset = FeatureExtractionDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate)

    model = ResNetTransferModel(feature_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    features = []
    valid_paths = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, paths = batch
            imgs = imgs.to(device)
            embeddings = model(imgs)
            features.append(embeddings.cpu().numpy())
            valid_paths.extend(paths)

    features_array = np.vstack(features)

    os.makedirs('data', exist_ok=True)
    np.save('data/features.npy', features_array)

    with open('data/image_paths.json', 'w') as f:
        json.dump(valid_paths, f)

    print(f"Extracted features shape: {features_array.shape}")
    print("Saved features to data/features.npy and image paths to data/image_paths.json")
