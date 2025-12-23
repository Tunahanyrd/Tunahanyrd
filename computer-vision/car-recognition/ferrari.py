#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:51:30 2025

@author: tunahan
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pathlib
import random
datadir = pathlib.Path("./data/ferrari_dataset")
device = "cuda" if torch.cuda.is_available() else "cpu"
class FewShotDataset(Dataset):
    def __init__(self, subset, n_way=5, k_shot=1, q_queries=5, transform=None):
        self.dataset = subset
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        
        self.class2idx = {}
        for idx, (_, label) in enumerate(dataset):
            self.class2idx.setdefault(label, []).append(idx)
        
        self.available_classes = [cls for cls, idxs in self.class2idx.items() if len(idxs) >= (k_shot + q_queries)]
    
    def __len__(self):
        return float("inf")
    
    def __getitem__(self, index):
        selected_classes = random.sample(self.available_classes, self.n_way)
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []
        
        label_map = {cls: i for i, cls in enumerate(selected_classes)}
        
        for cls in selected_classes:
            indices = random.sample(self.class2idx[cls],
                                    self.k_shot + self.q_queries)
            suppor_idxs = indices[:self.k_shot]
            query_idxs = indices[self.k_shot:]
            
            for idx in suppor_idxs:
                image, _ = self.dataset[idx]
                if self.transform:
                    image = self.transform(image)
                support_imgs.append(image)
                support_labels.append(label_map[cls])
                
            for idx in query_idxs:
                image, _ = self.dataset[idx]
                if self.transform:
                    image = self.transform(image)
                query_imgs.append(image)
                query_labels.append(label_map[cls])

        return (
            torch.stack(support_imgs), torch.tensor(support_labels),
            torch.stack(query_imgs), torch.tensor(query_labels)
            )
    
class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.subset[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.subset)    
    
train_transform = transforms.Compose([transforms.RandomResizedCrop(320, scale=(0.6, 1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.2, 
                                                            contrast=0.2, 
                                                            saturation=0.2),
                                     transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=20),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.5]*3, std=[0.5]*3)
                                     ])
val_transform = transforms.Compose([transforms.Resize(320),
                                    transforms.CenterCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                                    ])


dataset = ImageFolder(datadir)

train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

train_dataset = CustomDataset(train_subset, transform=train_transform)
val_dataset = CustomDataset(val_subset, transform=val_transform)
test_dataset = CustomDataset(test_subset, transform=val_transform)

fewshot_train_dataset = FewShotDataset(train_subset, n_way=5, k_shot=1, q_queries=5, transform=train_transform)
episode_loader = DataLoader(fewshot_train_dataset, batch_size=1, shuffle=True)

class CNN(nn.Module):
    def __init__(self, out_size=13, emb_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), # [B, 320, 320, 3]
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2), # [B, 160, 160, 32]
            
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2), # [B, 80, 80, 64]
            
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2,2), # [B, 40, 40, 64]
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40*40*64, emb_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            )
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
def prototypical_loss(model, suppor_imgs, support_labels, query_imgs, query_labels):
    support_emb = model(suppor_imgs) # [N*K, 128]
    query_emb = model(query_imgs)  # [N*Q, 128]
    
    n_classes = support_labels.unique().size(0)
    prototypes = []
    for c in range(n_classes):
        class_mask = (support_labels == c)
        class_emb = support_emb[class_mask]
        proto = class_emb.mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes) # [N, 128]
    
    # (query - prototype)^2
    dists = ((query_emb.unsqueeze(1) - prototypes.unsqueeze(0))**2).sum(dim=2)  # [Q, N]
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)

    pred = log_p_y.argmax(dim=1)
    acc = (pred == query_labels).float().mean().item()

    return loss, acc

model = CNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

for episode in episode_loader:
    support_images, support_labels, query_images, query_labels = [x.squeeze(0).to(device) for x in episode]

    loss, acc = prototypical_loss(model, support_images, support_labels, query_images, query_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    torch.cuda.empty_cache()










    
    
    
    
    
    
    
    
    
    
    
    
    
    


