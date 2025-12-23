import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_triplets(embeddings, threshold_pos=0.85, threshold_neg=0.6):
    sim = cosine_similarity(embeddings)
    triplets = []

    for i in range(len(embeddings)):
        pos_indices = np.where(sim[i] > threshold_pos)[0]
        pos_indices = pos_indices[pos_indices != i]

        neg_indices = np.where(sim[i] < threshold_neg)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue  

        p = np.random.choice(pos_indices)
        n = np.random.choice(neg_indices)

        triplets.append((i, p, n))

    return triplets


def triplet_loss(emb, triplets, margin=1.0):
    if len(triplets) == 0:
        return torch.tensor(0.0, requires_grad=True).to(emb.device)

    loss = 0
    for a, p, n in triplets:
        anchor = emb[a]
        positive = emb[p]
        negative = emb[n]
        loss += F.triplet_margin_loss(anchor, positive, negative, margin=margin)    
    return loss / len(triplets)
