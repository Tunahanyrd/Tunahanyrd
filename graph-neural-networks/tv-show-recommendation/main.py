#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:57:35 2025

@author: tunahan
"""
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_networkx
from pyvis.network import Network
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/home/tunahan/Masaüstü/kod/makine öğrenmesi/recommendation/data"
df = pd.read_csv(os.path.join(DATA_DIR, "shows.csv"))

def extract_root(title):
    return title.split(":")[0].split("–")[0].strip()

df["title_root"] = df["title"].apply(extract_root)
df = df.drop_duplicates(subset=["title_root"], keep="first").drop(columns=["title_root"])

print(df.isna().sum())

df = df.dropna(subset=["description", "genres"])

df["description"] = df.apply(
    lambda row: f"This series is directed by: {row['director']}. {row['description']}" 
    if pd.notna(row["director"]) else row["description"], 
axis=1)

df = df.drop_duplicates(subset=["title"])

print(df.info())

df["genres"] = df["genres"].apply(lambda x: [t.strip() for t in x.split(",")])

mlb = MultiLabelBinarizer()

genre_matrix = mlb.fit_transform(df["genres"])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

df = pd.concat([df.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)

del mlb

for a in ["rating", 'popularity','vote_count', 'vote_average']:
    df[a] = (df[a] - df[a].mean()) / df[a].std()

df["release_year"] = df["release_year"] / df["release_year"].max()

df = df.drop(columns=[
                      "director", "type", 
                      "date_added", "duration", 
                      ])

top_countries = df["country"].value_counts().head(10).index
top_langs = df["language"].value_counts().head(10).index

df["country"] = df["country"].apply(
    lambda x: next((c for c in top_countries if c in str(x)), "Other")
    )

df['language'] = df['language'].apply(
    lambda x: x if x in top_langs else 'Other'
)

df = pd.get_dummies(df, columns=["country", "language"], drop_first=True, prefix="ohe_")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

df["sbert_input"] = (
    "Title: " + df["title"] +
    ". Genre: " + df["genres"].apply(lambda x: ", ".join(x)) +
    ". Description: " + df["description"]
)

emb = model.encode(
    df["sbert_input"].tolist(),
    batch_size=128,
    show_progress_bar=True,
)

del model

sim = cosine_similarity(emb)

k = 10
min_sim = 0.5

edges = []
for i in range(len(df)):
    sims = list(enumerate(sim[i]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    
    top_k = [(j, score) for j, score in sims[1:] if score > min_sim][:k]
    
    for j, _ in top_k:
        edges.append((i, j))
        edges.append((j, i))
            
genre_cols = genre_df.columns.tolist()
ohe_cols = [col for col in df.columns if col.startswith("ohe_")]

cols = ["rating", 'popularity','vote_count', 'vote_average'] + genre_cols + ohe_cols

cols = df[cols].apply(pd.to_numeric, errors="raise").values.astype(np.float32)
extra_features = torch.tensor(cols, dtype=torch.float).to(device)

emb = torch.from_numpy(emb).float().to(device)
    
edge_idx = torch.tensor(edges).t().contiguous()

del genre_df, genre_matrix, genre_cols

class FeatureFusion(nn.Module):
    def __init__(self, emb_dim, feat_dim, out_dim=384):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, out_dim)
        self.linear2 = nn.Linear(feat_dim, out_dim)
        self.out = nn.Linear(out_dim * 2,out_dim)
    def forward(self, emb, feat):
        e1 = F.relu(self.linear1(emb))
        e2 = F.relu(self.linear2(feat))
        return self.out(torch.cat([e1, e2], dim=1))

fusion = FeatureFusion(emb.shape[1], extra_features.shape[1]).to(device)
x = fusion(emb, extra_features)
data = Data(x=x, edge_index=edge_idx).to(device)
      
class TripletDataset(Dataset):
    def __init__(self, sim_matrix, top_k=5, hard_negatives=False):
        self.sim = sim_matrix
        self.top_k = top_k
        self.num_items = sim_matrix.shape[0]
        self.hard_negatives = hard_negatives
    def __len__(self):
        return self.num_items
    def __getitem__(self, index):
        anchor = index
        
        sorted_sim = self.sim[anchor].argsort()[::-1]
        positives = [i for i in sorted_sim[1:self.top_k+1]]
        positive = random.choice(positives)
        
        if self.hard_negatives:
            negatives = [i for i in sorted_sim if self.sim[anchor][i] < 0.2]
        else:
            negatives = [i for i in range(self.num_items) 
                         if i != anchor and i not in positives]
        
        negative = random.choice(negatives)

        return anchor, positive, negative

triplet_dataset = TripletDataset(sim_matrix=sim, top_k=5)
triplet_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=64, shuffle=True)

def triplet_loss(anchor, positive, negative, margin=0.3):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, out_channels, heads=num_heads)
        self.gat2 = GATv2Conv(out_channels*num_heads, out_channels, heads=1)
        self.lin = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
            )

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_idx))
        x = self.gat2(x, edge_idx)
        return self.lin(x)
    
model = GAT(in_channels=x.shape[1], out_channels=64).to(device)

dataset_size = len(triplet_dataset)
train_size = int(0.8 * dataset_size)
test_size  = dataset_size - train_size
train_ds, test_ds = random_split(triplet_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

optimizer = optim.AdamW(model.parameters(), 
                        lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, 
                                                           T_mult=3, eta_min=1e-5)
num_epochs = 20
margin     = 0.3

for epoch in range(1, num_epochs+1):
    model.train()
    total_train_loss = 0.0
    with torch.no_grad():
        data.x = fusion(emb, extra_features)
        
    for anchor_ids, pos_ids, neg_ids in train_loader:

        embeddings = model(data)
        
        a = embeddings[anchor_ids]
        p = embeddings[pos_ids]
        n = embeddings[neg_ids]
        
        loss = triplet_loss(a, p, n, margin=margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        total_train_loss += loss.item() * anchor_ids.size(0)
        
    avg_train = total_train_loss / train_size

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        embeddings = model(data.to(device))
        for anchor_ids, pos_ids, neg_ids in test_loader:
            a = embeddings[anchor_ids]
            p = embeddings[pos_ids]
            n = embeddings[neg_ids]
            total_test_loss += triplet_loss(a, p, n, margin=margin).item() * anchor_ids.size(0)
    avg_test = total_test_loss / test_size

    print(f"Epoch {epoch:02d} — Train Loss: {avg_train:.4f} — Test Loss: {avg_test:.4f}")

def get_similar_items(target_item_id, data, top_n=10):
    model.eval()
    with torch.no_grad():
        output = model(data).cpu().numpy()

        target_embedding = output[target_item_id].reshape(1, -1)

        similarities = cosine_similarity(target_embedding, output) 
        
        recommended_items = similarities.argsort()[0][-top_n:][::-1]
        recommended_items = [i for i in recommended_items if i != target_item_id][:top_n]
        
        recommended_items_ids = df.iloc[recommended_items].index.values 
        
    return recommended_items_ids

def similarity(target, top_n=10):
    if target not in df['show_id'].values:
        print(f"Error: Show ID {target} not found.")
        return
    
    target_item_id = df[df['show_id'] == target].index[0]
    
    similar = get_similar_items(target_item_id, data, top_n)
    
    print(f"Input: {df.loc[target_item_id, 'title']} ")

    for movie_id in similar:
        title = df.loc[movie_id, "title"]
        genres = df.loc[movie_id, "genres"]
        print(f"- {title} (Genres: {genres})")

while True:
    inp = int(input("Enter id (0 for exit): "))
    if inp != 0:
        try:
            similarity(inp)
        except Exception as e:
            print(f"Error: {e}")
            continue
    else: break

G = to_networkx(data, to_undirected=True)

sub_nodes = list(range(1000))
subG = G.subgraph(sub_nodes)

plt.figure()
nx.draw(subG, with_labels=False, node_size=50),
plt.show()

net = Network(height="800px", width="100%", notebook=True)
net.force_atlas_2based()

titles = df["title"].to_dict()

limit=5
for i in range(limit):
    net.add_node(i, label=titles.get(df.index[i], f"[i]"), title=str(df.index[i]))
    
for src, dst in edges:
    if src < limit and dst < limit:
        net.add_edge(src, dst)
net.show("graph.html")



