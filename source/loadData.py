import gzip
import json
import os # Aggiunto per os.path.basename
import torch
from torch_geometric.data import Dataset, Data

# Funzione add_zeros spostata qui per coerenza
def add_zeros(data_obj, node_feature_dim=300): # Aggiunto parametro per la dimensione
    """
    Aggiunge feature di zeri (o uni, o degree) ai nodi se data.x non esiste.
    La dimensione delle feature deve corrispondere a quella attesa dal GNN.
    """
    if not hasattr(data_obj, 'x') or data_obj.x is None:
        if data_obj.num_nodes > 0 :
            data_obj.x = torch.zeros((data_obj.num_nodes, node_feature_dim), dtype=torch.float)
        else: # Nessun nodo, x vuoto con la dimensione corretta
            data_obj.x = torch.empty((0, node_feature_dim), dtype=torch.float)

    return data_obj


def dictToGraphObject(graph_dict, idx_in_file):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    edge_attr = None
    if "edge_attr" in graph_dict and graph_dict["edge_attr"] is not None and len(graph_dict["edge_attr"]) > 0:
        edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float)

    num_nodes = graph_dict.get("num_nodes", 0) # Default a 0 se non presente
    if not isinstance(num_nodes, int) or num_nodes < 0:
        print(f"Warning: num_nodes non valido ({num_nodes}) per grafo {idx_in_file}. Impostato a 0.")
        num_nodes = 0
        # Se num_nodes è 0, edge_index dovrebbe essere vuoto. Potrebbe essere necessario un controllo.
        if edge_index.numel() > 0:
            print(f"Warning: num_nodes è 0 ma edge_index non è vuoto per grafo {idx_in_file}.")
            edge_index = torch.empty((2,0), dtype=torch.long) # Forza edge_index vuoto
            if edge_attr is not None:
                edge_attr = torch.empty((0, edge_attr.size(1) if edge_attr.dim() > 1 else 0) , dtype=torch.float)


    y_tensor = None
    y_val = graph_dict.get("y")
    if y_val is not None and (isinstance(y_val, list) and len(y_val) > 0):
        try:
            y_tensor = torch.tensor(y_val[0], dtype=torch.long)
        except (ValueError, TypeError):
            print(f"Warning: y_val[0] non convertibile a tensore long per grafo {idx_in_file}. y_val: {y_val[0]}")
            # y_tensor rimane None
    elif y_val is not None and isinstance(y_val, (int, float)): # Se y è già uno scalare
        try:
            y_tensor = torch.tensor(y_val, dtype=torch.long)
        except (ValueError, TypeError):
            print(f"Warning: y_val scalare non convertibile a tensore long per grafo {idx_in_file}. y_val: {y_val}")

    # Le feature dei nodi (x) NON sono caricate qui. Devono essere aggiunte da 'transform=add_zeros'
    # o caricate esplicitamente se presenti nel JSON (es. graph_dict["node_feat"]).
    node_features_x = None
    if "node_feat" in graph_dict and graph_dict["node_feat"] is not None:
        try:
            node_features_x = torch.tensor(graph_dict["node_feat"], dtype=torch.float)
            if node_features_x.shape[0] != num_nodes:
                print(f"Warning: Mismatch tra num_nodes ({num_nodes}) e graph_dict['node_feat'].shape[0] ({node_features_x.shape[0]}) per grafo {idx_in_file}.")
                # Potresti decidere di scartare le feature o il grafo
                node_features_x = None # Scarta feature se Mismatch
        except Exception as e:
            print(f"Errore nel caricare node_feat per grafo {idx_in_file}: {e}")
            node_features_x = None


    data_obj = Data(
        x=node_features_x, # Aggiunto x qui, se presente nel JSON
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        y=y_tensor
    )
    data_obj.original_idx = torch.tensor([idx_in_file], dtype=torch.long)
    return data_obj

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, pre_filter=None):
        self.raw_filename = filename
        self.graphs_dicts_list = []

        try:
            with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
                self.graphs_dicts_list = json.load(f)
        except FileNotFoundError:
            print(f"Errore Dataset: File non trovato a {self.raw_filename}")
            raise
        except json.JSONDecodeError:
            print(f"Errore Dataset: File JSON non valido a {self.raw_filename}")
            raise
        except Exception as e:
            print(f"Errore Dataset: Errore durante il caricamento di {self.raw_filename}: {e}")
            raise

        self.num_graphs = len(self.graphs_dicts_list)
        if self.num_graphs == 0:
            print(f"Attenzione Dataset: Nessun grafo caricato da {self.raw_filename}")

        super().__init__(root=None, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.raw_filename)] # Usa os.path.basename

    @property
    def processed_file_names(self):
        return [] # O un nome file se usi la cache di processamento di PyG

    def len(self):
        return self.num_graphs

    def get(self, idx):
        if not (0 <= idx < self.num_graphs): # Controllo indice
            raise IndexError(f"Indice {idx} fuori range per dataset di lunghezza {self.num_graphs}")

        graph_dict = self.graphs_dicts_list[idx]
        data_obj = dictToGraphObject(graph_dict, idx)

        # Applica la trasformazione DOPO la creazione dell'oggetto base
        # Questo è importante perché add_zeros potrebbe dipendere da data_obj.num_nodes
        if self.transform:
            data_obj = self.transform(data_obj)
            if not hasattr(data_obj, 'x') or data_obj.x is None:
                if data_obj.num_nodes > 0 : # Stampa warning solo se ci sono nodi
                    print(f"Attenzione Dataset (get, idx {idx}): data.x è None dopo la trasformazione, ma num_nodes > 0. GNN potrebbe fallire.")

        return data_obj