# utils/data_utils.py

import os
import random
import shutil
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()  # force actual loading to catch errors
        return img.convert('RGB')
    except UnidentifiedImageError:
        # Could not open image
        return None

class TripletDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        for cls in self.classes:
            cls_folder = os.path.join(folder, cls)
            for img_file in os.listdir(cls_folder):
                self.image_paths.append((os.path.join(cls_folder, img_file), self.class_to_idx[cls]))

    def __getitem__(self, index):
        while True:
            anchor_path, anchor_label = self.image_paths[index]
            positive_path = self.get_positive(anchor_label, anchor_path)
            negative_path = self.get_negative(anchor_label)

            anchor_img = pil_loader(anchor_path)
            positive_img = pil_loader(positive_path)
            negative_img = pil_loader(negative_path)

            if None in (anchor_img, positive_img, negative_img):
                # Skip if any image failed to load
                index = (index + 1) % len(self)
                continue

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return (anchor_img, positive_img, negative_img), []

    def __len__(self):
        return len(self.image_paths)

    def get_positive(self, anchor_label, anchor_path):
        positive_paths = [p for p, l in self.image_paths if l == anchor_label and p != anchor_path]
        if len(positive_paths) == 0:
            # fallback to anchor_path itself if none found
            return anchor_path
        return random.choice(positive_paths)

    def get_negative(self, anchor_label):
        negative_paths = [p for p, l in self.image_paths if l != anchor_label]
        if len(negative_paths) == 0:
            raise RuntimeError(f"No negative images found for label {anchor_label}")
        return random.choice(negative_paths)

def split_dataset(root_dir, train_ratio=0.8):
    classes = os.listdir(root_dir)
    classes = [cls for cls in classes if cls not in ['train', 'val']]
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]

        train_cls_dir = os.path.join(root_dir, 'train', cls)
        val_cls_dir = os.path.join(root_dir, 'val', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        for img in train_images:
            shutil.move(os.path.join(cls_path, img), os.path.join(train_cls_dir, img))
        for img in val_images:
            shutil.move(os.path.join(cls_path, img), os.path.join(val_cls_dir, img))

        # Remove empty class directory
        if not os.listdir(cls_path):
            os.rmdir(cls_path)
