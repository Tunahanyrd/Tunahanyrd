#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:16:16 2025

@author: tunahan
Pancreatic Cancer Biomedical Knowledge Graph
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import torch_geometric
from torch_geometric.nn import RGATConv
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data/pancreatic_cancer_kg_original.csv", nrows=3.5e4)

# Input:  (head, relation, tail) + sentence + attention score
# Output: cosine similarity
# Process: r-GAT -> Cosine (y_true) -> MLP Fusion Layer -> Cosine (y_true) 
#           |__________________________________________________|
#                             <-- backward <--

nodes = sorted(set(df["head"]).union(set(df["tail"])))
relations = sorted(df["relation"].unique())

node2id = {node: i for i, node in enumerate(nodes)}
rel2id = {rel: i for i, rel in enumerate(relations)}

df["head_id"] = df["head"].map(node2id)
df["tail_id"] = df["tail"].map(node2id)
df["rel_id"]  = df["relation"].map(rel2id)

# r-GAT is a directed graph
# (head → tail, relation)
# I will add (tail → head, relation_inverse) and self-loop

nrel = len(rel2id)
inv_rel_offset = nrel

edges = []
for h, r, t in zip(df["head_id"], df["rel_id"], df["tail_id"]):
    edges.append((h, r, t))                  # (h → t)
    edges.append((t, r + inv_rel_offset, h)) # (t → h) inverse relation

self_loop_rel = 2 * nrel
for node_id in range(len(node2id)):
    edges.append((node_id, self_loop_rel, node_id))

src_nodes = torch.tensor([e[0] for e in edges], dtype=torch.long)
rel_ids   = torch.tensor([e[1] for e in edges], dtype=torch.long)
tgt_nodes = torch.tensor([e[2] for e in edges], dtype=torch.long)

edge_idx = torch.stack([src_nodes, tgt_nodes], dim=0) # [2, num_edges]
edge_type  = rel_ids                                  # [num_edges]

# %%

sbert = SentenceTransformer("all-MiniLM-L6-v2")

sbert_embeddings = list(sbert.encode(
                    df["sentence"].tolist(),
                    convert_to_tensor=True,
                    show_progress_bar=True
))

df["sbert_embeddings"] = sbert_embeddings

torch.save(sbert_embeddings, "sbert_embeddings.pt")
def noise(tensor, p=0.5):
    mask = torch.bernoulli(torch.full(tensor.shape, p).to(tensor.device))
    return tensor * mask
# %%

class FusionModel(nn.Module):
    def __init__(self, num_nodes, num_relations, sbert_dim=384, in_dim=128, out_dim=128, hidden_dim=256):
        super().__init__()
        
        # 1. Node ID -> Embedding (random)
        self.node_emb = nn.Embedding(num_nodes, in_dim)
        
        # 2. RGAT Layer
        self.rgat = RGATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            num_relations=num_relations,
            heads=4,
            attention_mechanism="across-relation",
            attention_mode="additive-self-attention",
            concat=False
        )
        
        # 3. MLP: [head || tail || sentence] → fused vec
        fusion_input_dim = out_dim * 2 + sbert_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations)
        )
        
    def forward(self, head_ids, tail_ids, sbert_embeddings, precomp_gat):
        
        h_gat = precomp_gat[head_ids]     # [batch, out_dim]
        t_gat = precomp_gat[tail_ids]     # [batch, out_dim]
        
        if self.training:
            sbert_embeddings = noise(sbert_embeddings, p=0.3)
        
        fusion_input = torch.cat([h_gat, t_gat, sbert_embeddings], dim=1)
        fusion_input = F.dropout(fusion_input, p = 0.3)
        return self.classifier(fusion_input)

model = FusionModel(
    num_nodes=len(node2id),
    num_relations=len(rel2id)*2 + 1,
    sbert_dim=384, in_dim=128, out_dim=128, hidden_dim=256
).cpu()

with torch.no_grad():
    node_feats = model.node_emb.weight      # [num_nodes, in_dim]
    precomputed = model.rgat(node_feats, edge_idx, edge_type)
    precomputed = precomputed

precomputed = precomputed.to("cuda")
model = model.to("cuda")
for p in model.rgat.parameters():
    p.requires_grad = False
for p in model.node_emb.parameters():
    p.requires_grad = False

# %%
    
def train_model(model, dataloader, optimizer, criterion, precomputed):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    scaler = GradScaler()

    for batch in dataloader:
        optimizer.zero_grad()
        
        head_ids = batch["head_id"].to("cuda")
        tail_ids = batch["tail_id"].to("cuda")
        sbert_embeddings = batch["sent_emb"].to("cuda")
        true_rel = batch["rel_id"].to("cuda")
        
        with autocast():
            logits = model(head_ids, tail_ids, sbert_embeddings, precomputed)
            loss = criterion(logits, true_rel)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * head_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == true_rel).sum().item()
        total_samples += head_ids.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, precomputed):
    model.eval()
    total_loss,total_correct,total_samples=0,0,0
    with torch.no_grad():
        for batch in dataloader:
            h = batch["head_id"].to("cuda")
            t = batch["tail_id"].to("cuda")
            s = batch["sent_emb"].to("cuda")
            y = batch["rel_id"].to("cuda")

            logits = model(h, t, s, precomputed)
            loss = criterion(logits, y)

            total_loss   += loss.item()*h.size(0)
            preds        = logits.argmax(dim=1)
            total_correct+= (preds==y).sum().item()
            total_samples+= h.size(0)

    return total_loss/total_samples, total_correct/total_samples

def test_model(model, dataloader, criterion, precomputed):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            head_ids = batch["head_id"].to("cuda")
            tail_ids = batch["tail_id"].to("cuda")
            sent_emb = batch["sent_emb"].to("cuda")
            true_rel = batch["rel_id"].to("cuda")
            
            logits = model(head_ids, tail_ids, sent_emb, precomputed)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(true_rel.cpu().numpy())
    return all_preds, all_labels

# %%

class TripleDataset(Dataset):
    def __init__(self, df, sbert_emb, node2id, rel2id):
        self.df = df.reset_index(drop=True)
        self.sbert = sbert_emb
        self.node2id = node2id
        self.rel2id = rel2id
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "head_id": row["head_id"],
            "tail_id": row["tail_id"],
            "rel_id":  row["rel_id"],
            "sent_emb": self.sbert[idx].float()
        }
 
def collate_fn(batch):
    return {
        "head_id":   torch.tensor([b["head_id"] for b in batch], dtype=torch.long),
        "tail_id":   torch.tensor([b["tail_id"] for b in batch], dtype=torch.long),
        "rel_id":    torch.tensor([b["rel_id"]  for b in batch], dtype=torch.long),
        "sent_emb":  torch.stack([b["sent_emb"] for b in batch])
    }
       
dataset = TripleDataset(df, sbert_embeddings, node2id, rel2id)
train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=512, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=512, collate_fn=collate_fn)
test_loader  = DataLoader(test_set, batch_size=512, collate_fn=collate_fn)

# %%   

epochs = 100    
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-5)

best_val_loss = float("inf")
patience = 50
patience_counter = 0

for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, precomputed)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, precomputed)

    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_relation_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break
    torch.cuda.empty_cache()
    
model.load_state_dict(torch.load("best_relation_model.pt"))
preds, labels = test_model(model, test_loader, criterion, precomputed)
    
print(classification_report(labels, preds, target_names=sorted(rel2id, key=rel2id.get)))
  
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(rel2id, key=rel2id.get), yticklabels=sorted(rel2id, key=rel2id.get), cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()    
    
"""
                   precision    recall  f1-score   support

        activates       1.00      1.00      1.00       454
  associated_with       1.00      1.00      1.00       446
    biomarker_for       1.00      1.00      1.00       516
           causes       1.00      1.00      1.00       409
         inhibits       1.00      1.00      1.00       432
   interacts_with       1.00      1.00      1.00       407
       mutated_in       1.00      1.00      1.00       511
 overexpressed_in       1.00      1.00      1.00       544
       suppresses       1.00      1.00      1.00       381
           treats       1.00      1.00      1.00       489
underexpressed_in       1.00      1.00      1.00       411

         accuracy                           1.00      5000
        macro avg       1.00      1.00      1.00      5000
     weighted avg       1.00      1.00      1.00      5000
"""
# %%
column_names = pd.read_csv("./data/pancreatic_cancer_kg_original.csv", nrows=0).columns

df_new = pd.read_csv("./data/pancreatic_cancer_kg_original.csv", skiprows=int(5e4), nrows=int(1e4), names = column_names)

df_new["head_id"] = df_new["head"].map(node2id)
df_new["tail_id"] = df_new["tail"].map(node2id)
df_new["rel_id"]  = df_new["relation"].map(rel2id)

df_new = df_new.dropna(subset=["head_id", "tail_id", "rel_id"])
df_new = df_new.astype({"head_id": int, "tail_id": int, "rel_id": int})


new_sbert_emb = sbert.encode(df_new["sentence"].tolist(), convert_to_tensor=True, show_progress_bar=True)

new_sbert_emb = torch.zeros_like(new_sbert_emb)

new_dataset = TripleDataset(df_new, new_sbert_emb, node2id, rel2id)
new_loader = DataLoader(new_dataset, batch_size=4, collate_fn=collate_fn)

model.load_state_dict(torch.load("best_relation_model.pt"))
model = model.to("cuda")

preds, labels = test_model(model, new_loader, criterion, precomputed)

print(classification_report(labels, preds, target_names=sorted(rel2id, key=rel2id.get)))

"""
print(classification_report(labels, preds, target_names=sorted(rel2id, key=rel2id.get)))
                   precision    recall  f1-score   support

        activates       1.00      1.00      1.00       836
  associated_with       1.00      1.00      1.00       840
    biomarker_for       1.00      1.00      1.00       985
           causes       1.00      1.00      1.00       854
         inhibits       1.00      1.00      1.00       805
   interacts_with       1.00      1.00      1.00       871
       mutated_in       1.00      1.00      1.00      1065
 overexpressed_in       1.00      1.00      1.00      1003
       suppresses       1.00      1.00      1.00       835
           treats       1.00      1.00      1.00      1022
underexpressed_in       1.00      1.00      1.00       884

         accuracy                           1.00     10000
        macro avg       1.00      1.00      1.00     10000
     weighted avg       1.00      1.00      1.00     10000
"""
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(rel2id, key=rel2id.get), yticklabels=sorted(rel2id, key=rel2id.get), cmap='Blues')
plt.title("Confusion Matrix on New Unseen Data")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

    
    
    
    
    