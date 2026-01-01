import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

class ResNetTransferModel(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResNetTransferModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # New fully connected layer mapping to feature dimension
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # flatten features
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # normalize embeddings
        return x

def compute_triplet_accuracy(anchor_embed, positive_embed, negative_embed):
    ap_dist = torch.norm(anchor_embed - positive_embed, p=2, dim=1)
    an_dist = torch.norm(anchor_embed - negative_embed, p=2, dim=1)
    return torch.mean((ap_dist < an_dist).float()).item()

def train_model(model, train_loader, val_loader, device, num_epochs=15, save_path='weights/model.pth'):
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_accuracies = []

        for (anchor, positive, negative), _ in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)
            train_acc = compute_triplet_accuracy(anchor_embed, positive_embed, negative_embed)
            train_accuracies.append(train_acc)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc_epoch = sum(train_accuracies) / len(train_accuracies)

        model.eval()
        val_accuracies = []
        val_loss_total = 0.0

        with torch.no_grad():
            for (anchor, positive, negative), _ in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)
                loss = criterion(anchor_embed, positive_embed, negative_embed)
                val_loss_total += loss.item() * anchor.size(0)
                val_acc = compute_triplet_accuracy(anchor_embed, positive_embed, negative_embed)
                val_accuracies.append(val_acc)

        val_loss_epoch = val_loss_total / len(val_loader.dataset)
        val_acc_epoch = sum(val_accuracies) / len(val_accuracies)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_acc'].append(val_acc_epoch)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss_epoch:.4f} - Train Triplet Acc: {train_acc_epoch:.4f} - Val Triplet Acc: {val_acc_epoch:.4f}')

        # Save best model by validation accuracy
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved at epoch {epoch+1} with Val Acc: {val_acc_epoch:.4f}')

