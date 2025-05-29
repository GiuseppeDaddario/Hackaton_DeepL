## Data Loader Module
# This module is responsible for loading data from a specified file path.
# Includes also preprocessing steps.

import torch



def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

# source/transforms.py
import torch

class AddNodeFeatures:
    """
    Aggiunge feature ai nodi se non sono presenti.
    Inizializza i nodi con un vettore di uni.
    Puoi modificare la logica di inizializzazione (es. zeri, random, degree).
    """
    def __init__(self, node_feature_dim=1): # O un'altra dimensione iniziale
        self.node_feature_dim = node_feature_dim

    def __call__(self, data):
        if data.x is None:
            # Crea feature dei nodi come vettori di uni
            # La dimensione delle feature iniziali può essere 1,
            # e poi una Linear layer nel modello le proietterà a hidden_dim.
            data.x = torch.ones((data.num_nodes, self.node_feature_dim), dtype=torch.float)
        return data

# Esempio alternativo: usare i gradi dei nodi (non normalizzati)
from torch_geometric.utils import degree
class AddDegreeFeatures:
    def __call__(self, data):
        if data.x is None:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float).unsqueeze(1)
            data.x = deg
        return data