#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tunahan
@dataset: amanandandrai/fake-dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from model.models import GAT, get_cosine_edge_weights, sGAT, MLP
from model.train import train_model, train_supervised_model, train_mlp_model
from model.triplet import get_triplets, triplet_loss
from utils.visualize import plot_umap
from utils.predict import predict_fake, predict_clf
from utils.compare import compare_bayes, compare_linearity, non_gat_train
from utils.mask import mask_tokens

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and prepare data and embedding them
df_t = pd.read_csv("./data/True.csv")
df_f = pd.read_csv("./data/Fake.csv")
df_t["fake"] = 0
df_f["fake"] = 1
df = pd.concat([df_t, df_f]).sample(frac=1).reset_index(drop=True)

del df_t, df_f

sbert = SentenceTransformer("all-mpnet-base-v2")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# train_df["title"] = train_df["title"].apply(mask_tokens)

train_embeddings = sbert.encode(train_df["title"].tolist(), 
                                show_progress_bar=True, 
                                normalize_embeddings=True)

test_embeddings  = sbert.encode(test_df["title"].tolist(),  
                                show_progress_bar=True,
                                normalize_embeddings=True)

del sbert 

edge_index, edge_attr = get_cosine_edge_weights(train_embeddings, threshold=0.7)
x_train = torch.tensor(train_embeddings, dtype=torch.float).to(device)
y_train = torch.tensor(train_df["fake"].values, dtype=torch.long).to(device)
data = Data(x=x_train, edge_index=edge_index, 
            edge_attr=edge_attr, y=y_train).to(device)

# Description of the model
model     = GAT(in_channels=x_train.shape[1], hidden_channel=128).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-5)

model = train_model(model, data, get_triplets,
                    triplet_loss, optimizer, scheduler,
                    epochs=60, patience=5)

# Evaulation
model.eval()
with torch.no_grad():
    x_test = torch.tensor(test_embeddings, dtype=torch.float).to(device)

    edge_index_t, edge_attr_t = get_cosine_edge_weights(test_embeddings, threshold=0.7)
    edge_index_t = edge_index_t.to(device)
    edge_attr_t  = edge_attr_t.to(device)

    final_train_embeddings = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()
    final_test_embeddings  = model(x_test, edge_index_t, edge_attr_t).cpu().numpy()

# Classifier description
clf = LogisticRegression(max_iter=1000)
clf.fit(final_train_embeddings, train_df["fake"])
y_pred = clf.predict(final_test_embeddings)
print("GAT Model Report:")
print(classification_report(test_df["fake"], y_pred))

# Visualization
plot_umap(final_train_embeddings, train_df["fake"], "Train data")
plot_umap(final_test_embeddings,  test_df["fake"], "Test data")

# Compare
y_test_bayes, y_pred_bayes = compare_bayes(df)
print("Naive Bayes Report:")
print(classification_report(y_test_bayes, y_pred_bayes))

# Testing Linearity
print("What if we used other machine learning algorithms instead of Logreg?")
compare_linearity(train_df, test_df, final_train_embeddings, final_test_embeddings)

# Testing non-gat results
print("What if we leaned the model from the sbert outputs?")
non_gat_train(train_df, test_df, train_embeddings, test_embeddings)

plot_umap(test_embeddings, test_df["fake"], desc="(SBERT only, Test data)")

# What if we trained a supervised gat model?
smodel = sGAT(in_channels=x_train.shape[1], hidden_channel=128).to(device)
optimizer = optim.AdamW(smodel.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()
smodel = train_supervised_model(smodel, data, criterion, optimizer, scheduler,
                    epochs=600, patience=50)

predict_clf(smodel, data)

# What if we trained a supervised mlp model?
mlp_model = MLP(input_dim=data.x.shape[1], hidden_dim=128).to(device)
optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

mlp_model = train_mlp_model(
    model=mlp_model,
    x=data.x,
    y=data.y,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=6000,
    patience=50
)

predict_clf(mlp_model, data)














