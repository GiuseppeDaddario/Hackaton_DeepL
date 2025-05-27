# source/loadData.py
import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import numpy as np

# Assumi che add_zeros sia importabile da qualche parte se lo usi come transform
# Esempio: from .dataLoader import add_zeros # Se add_zeros è in source/dataLoader.py

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
        self.add_zeros_transform = add_zeros_transform # Salva la trasformazione add_zeros

        # Carica tutti i dizionari dei grafi e processali
        self._load_and_process_data()

        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def _load_and_process_data(self):
        with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
            graphs_raw_list = json.load(f)

        self.processed_data_list = []
        all_labels = []
        self.inferred_num_edge_features = None
        self.graphs_dicts_list = graphs_raw_list # Mantieni per gcod se necessario

        for i, graph_dict in enumerate(graphs_raw_list):
            # 1. Crea l'oggetto Data base
            edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

            edge_attr_raw = graph_dict.get("edge_attr")
            edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float) if edge_attr_raw is not None else None

            if self.inferred_num_edge_features is None and edge_attr is not None:
                if edge_attr.ndim == 2 : # Assicura che edge_attr sia 2D
                    self.inferred_num_edge_features = edge_attr.size(1)
                elif edge_attr.ndim == 1 and edge_attr.numel() > 0 : # Se è 1D ma non vuoto, potrebbe essere per un solo arco
                    # Questo caso è ambiguo per inferire num_features globali.
                    # Se hai sempre più archi, ndim==2 è più sicuro.
                    # Se può esserci un solo arco con N features, allora questa inferenza
                    # potrebbe fallire se il primo grafo ha 0 o 1 arco e edge_attr è 1D.
                    # Per ora, assumiamo che sia 2D o vuoto.
                    pass

            num_nodes = graph_dict["num_nodes"]

            y_val_list = graph_dict.get("y")
            y_tensor = None
            if y_val_list and len(y_val_list) > 0:
                y_value = y_val_list[0]
                y_tensor = torch.tensor(y_value, dtype=torch.long)
                all_labels.append(y_value)
            else:
                all_labels.append(-1) # Placeholder per grafi senza etichetta valida

            # 2. Aggiungi l'indice originale (fisso, non shufflato)
            # Questo è cruciale per gcodLoss
            true_original_idx_tensor = torch.tensor([i], dtype=torch.long)

            data_obj = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=y_tensor,
                original_idx=true_original_idx_tensor # Indice originale fisso
            )

            # 3. Applica la trasformazione add_zeros se fornita
            if self.add_zeros_transform:
                data_obj = self.add_zeros_transform(data_obj)

            self.processed_data_list.append(data_obj)

        # Inferisci il numero di classi
        if all_labels:
            unique_labels = np.unique([label for label in all_labels if label != -1]) # Escludi placeholder
            self.inferred_num_classes = len(unique_labels) if len(unique_labels) > 0 else 1
        else:
            self.inferred_num_classes = 1 # Default se non ci sono etichette

        if self.inferred_num_edge_features is None:
            self.inferred_num_edge_features = 0 # Default se nessun grafo ha edge_attr
            # O potresti voler sollevare un errore o usare un valore di default noto

    @property
    def num_classes(self):
        return self.inferred_num_classes

    @property
    def num_edge_features(self):
        # Restituisce il numero inferito. Potrebbe essere 0 se nessun grafo ha edge_attr.
        return self.inferred_num_edge_features if self.inferred_num_edge_features is not None else 0


    def len(self):
        return len(self.processed_data_list)

    def get(self, idx):
        # Restituisce l'oggetto Data già processato e trasformato (con add_zeros)
        # L'eventuale 'transform' di PyG (passato a __init__) viene applicato qui dal DataLoader
        return self.processed_data_list[idx]

    # Metodo per ottenere le etichette originali per GCOD, se il main.py lo richiede ancora
    # da graphs_dicts_list. Altrimenti, può essere rimosso se il main.py usa all_labels.
    def get_original_y_list_for_gcod(self):
        """Restituisce una lista delle etichette y nell'ordine originale del file, per GCOD."""
        y_list = []
        for graph_dict in self.graphs_dicts_list: # Usa la lista raw
            y_val_list = graph_dict.get("y")
            if y_val_list and len(y_val_list) > 0:
                y_list.append(y_val_list[0])
            else:
                # Coerenza con come GCOD si aspetta i dati (es. ignora o usa placeholder)
                # Se GCOD si aspetta un array numpy senza None/placeholder, filtra qui.
                pass # O aggiungi un placeholder se GCOD lo gestisce
        return y_list