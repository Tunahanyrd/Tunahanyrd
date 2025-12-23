#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:04:05 2025

@author: tunahan
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import amp
from PIL import Image
import seaborn as sns
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []
        self.labels = []
        self.class_map = {
            "glioma": 0,
            "meningioma": 1,
            "pituitary": 2,
            "no_tumor": 3
        }
        
        for name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, name)
            if not os.path.isdir(label_dir): continue
            for img_file in os.listdir(label_dir):
                self.image_path.append(os.path.join(label_dir, img_file))
                self.labels.append(self.class_map[name])

    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

class EarlyStopping:
    def __init__(self,patience, delta=1e-3):
        self.delta = delta
        self.best_loss = float("inf")
        self.patience = patience
        self.counter = 0
    def __call__(self, last_loss):
        if last_loss + self.delta < self.best_loss:
            self.best_loss = last_loss 
            self.counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

train_t = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, std=[0.5]*3)
])

test_t = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, std=[0.5]*3)
])

train_dataset = CustomDataset(root_dir="./data/classification_task/train",transform=train_t)
test_dataset = CustomDataset(root_dir="./data/classification_task/test",transform=test_t)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

model_w = models.ConvNeXt_Tiny_Weights.DEFAULT
model = models.convnext_tiny(weights=model_w)

n_classes = 4 # glioma, meningioma, pituitary, no_tumor
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, n_classes)

for param in model.features.parameters():
    param.requires_grad = False

for param in model.parameters():
    param.requires_grad = True

model.to(device)

def train_model(model,dataloader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total
    
def val_model(model,dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
                
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / total, correct / total

def test_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []  
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs=30, ):
        model.train()
        scaler = amp.GradScaler(device="cuda")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = val_model(model, val_loader, criterion)
            print(f"[Epoch {epoch+1}]")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Lr: {optimizer.param_groups[0]['lr']}")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            scheduler.step(val_loss)

            if early_stopping(val_loss): 
                print(f"Early stopping triggered in {epoch}. epoch")
                break
            
            torch.cuda.empty_cache()
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                 patience=2, factor=0.5)
early_stopping = EarlyStopping(patience=2)

fit(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler, 
    early_stopping, epochs=5
)
model.load_state_dict(torch.load("best_model.pt"))

preds, labels, probs = test_model(model, test_loader)

print(classification_report(labels, preds, target_names=class_names))

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap="Blues", 
            annot=True, 
            xticklabels=class_names, 
            yticklabels=class_names, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Conf Matrix")
plt.tight_layout()
plt.show()

y_true_bin = label_binarize(labels, classes=[0,1,2,3])
y_score = np.array(probs)

fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
macro_roc_auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
print(f"Macro ROC-AUC: {macro_roc_auc:.4f}")

plt.figure(figsize=(8,6))
colors = ["red", "green", "blue", "purple"]
for i in range(4):
    plt.plot(fpr[i], tpr[i], colors[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Macro f1:", f1_score(labels, preds,average="macro"))

"""

              precision    recall  f1-score   support

      glioma       0.93      0.99      0.96       254
  meningioma       0.99      0.92      0.96       306
   pituitary       0.99      0.99      0.99       300
    no_tumor       0.98      1.00      0.99       140

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Macro ROC-AUC: 0.9988
Macro f1: 0.9720843793061107


"""


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
        
        def _register_hooks(self):
            def forward_hook(model, input, output):
                self.activations = output.detach()
                
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()
            
            self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        
        def generate(self, input_tensor, class_idx=None):
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            loss = output[0, class_idx]
        
        """loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        