import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics.pairwise import cosine_similarity

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channel, num_classes=2):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channel, heads=4, edge_dim=1)
        self.dropout1 = nn.Dropout(0.2)

        self.gat2 = GATv2Conv(hidden_channel * 4, hidden_channel, heads=1, edge_dim=1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.dropout1(x)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.dropout2(x)
        return x

def get_cosine_edge_weights(embeddings, threshold=0.7):
    sim = cosine_similarity(embeddings)
    edge_index = []
    edge_attr = []
    for i in range(len(sim)):
        for j in range(len(sim)):
            if i != j and sim[i][j] > threshold:
                edge_index.append([i, j])
                edge_attr.append([sim[i][j]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

class sGAT(nn.Module):
    def __init__(self, in_channels, hidden_channel, num_classes=2):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channel, heads=4, edge_dim=1)
        self.dropout1 = nn.Dropout(0.2)

        self.gat2 = GATv2Conv(hidden_channel * 4, hidden_channel, heads=1, edge_dim=1)
        self.dropout2 = nn.Dropout(0.2)

        self.cf = nn.Sequential(
            nn.Linear(hidden_channel, 128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
            )
    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.dropout1(x)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.dropout2(x)
        x = self.cf(x)
        return x
    
    
class MLP(nn.Module):
   def __init__(self, input_dim=768, hidden_dim=128):
       super().__init__()
       self.model = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(hidden_dim, 2)
       )

   def forward(self, x):
       return self.model(x)
  
   
   
   
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    