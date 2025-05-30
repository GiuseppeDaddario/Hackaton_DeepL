import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from torch_geometric.utils import degree

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, add_zeros_transform=None):
        """
        Args:
            filename (str): Path al file .json.gz del dataset.
            transform (callable, optional): Trasformazione PyG standard da applicare on-the-fly.
            pre_transform (callable, optional): Trasformazione PyG standard da applicare una volta e salvare.
            add_zeros_transform (callable, optional): La tua trasformazione custom 'add_zeros'.
        """
        self.raw_filename = filename
        self.add_zeros_transform = add_zeros_transform
        self._load_and_process_data()

        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def _load_and_process_data(self):
        with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
            graphs_raw_list = json.load(f)

        self.processed_data_list = []
        all_labels = []
        self.inferred_num_edge_features = None
        self.graphs_dicts_list = graphs_raw_list

        for i, graph_dict in enumerate(graphs_raw_list):

            edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            edge_attr_raw = graph_dict.get("edge_attr")
            edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float) if edge_attr_raw is not None else None

            if self.inferred_num_edge_features is None and edge_attr is not None:
                if edge_attr.ndim == 2 :
                    self.inferred_num_edge_features = edge_attr.size(1)
                elif edge_attr.ndim == 1 and edge_attr.numel() > 0 :
                    pass

            num_nodes = graph_dict["num_nodes"]

            y_val_list = graph_dict.get("y")
            y_tensor = None
            if y_val_list and len(y_val_list) > 0:
                y_value = y_val_list[0]
                y_tensor = torch.tensor(y_value, dtype=torch.long)
                all_labels.append(y_value)
            else:
                all_labels.append(-1)

            true_original_idx_tensor = torch.tensor([i], dtype=torch.long)
            data_obj = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=y_tensor,
                original_idx=true_original_idx_tensor
            )

            if self.add_zeros_transform:
                data_obj = self.add_zeros_transform(data_obj)

            self.processed_data_list.append(data_obj)

        if all_labels:
            unique_labels = np.unique([label for label in all_labels if label != -1])
            self.inferred_num_classes = len(unique_labels) if len(unique_labels) > 0 else 1
        else:
            self.inferred_num_classes = 1

        if self.inferred_num_edge_features is None:
            self.inferred_num_edge_features = 0

    @property
    def num_classes(self):
        return self.inferred_num_classes

    @property
    def num_edge_features(self):
        return self.inferred_num_edge_features if self.inferred_num_edge_features is not None else 0


    def len(self):
        return len(self.processed_data_list)

    def get(self, idx):
        return self.processed_data_list[idx]





    def get_original_y_list_for_gcod(self):
        """Restituisce una lista delle etichette y nell'ordine originale del file, per GCOD."""
        y_list = []
        for graph_dict in self.graphs_dicts_list:
            y_val_list = graph_dict.get("y")
            if y_val_list and len(y_val_list) > 0:
                y_list.append(y_val_list[0])
            else:
                pass
        return y_list

class AddNodeFeatures:
    def __init__(self, node_feature_dim=1):
        self.node_feature_dim = node_feature_dim

    def __call__(self, data):
        if data.x is None:
            data.x = torch.ones((data.num_nodes, self.node_feature_dim), dtype=torch.float)
        return data


class AddDegreeFeatures:
    def __call__(self, data):
        if data.x is None:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float).unsqueeze(1)
            data.x = deg
        return data