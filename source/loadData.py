
import gzip
import json
import torch
from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        # Carica tutti i dizionari dei grafi una volta
        with gzip.open(self.raw, "rt", encoding="utf-8") as f:
            self.graphs_dicts_list = json.load(f)
        self.num_graphs = len(self.graphs_dicts_list)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        graph_dict = self.graphs_dicts_list[idx]
        data_obj = dictToGraphObject(graph_dict)
        return {
            "edge_index": data_obj.edge_index,
            "edge_attr": data_obj.edge_attr,
            "num_nodes": data_obj.num_nodes,
            "y": data_obj.y,
            "original_idx": torch.tensor([idx], dtype=torch.long)
        }

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict.get("edge_attr") else None # Usa .get per sicurezza
    num_nodes = graph_dict["num_nodes"]
    # Assicurati che y sia gestito correttamente se è None o vuoto
    y_val = graph_dict.get("y")
    y = torch.tensor(y_val[0], dtype=torch.long) if y_val and len(y_val) > 0 else None # Gestisci y vuoto o None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)