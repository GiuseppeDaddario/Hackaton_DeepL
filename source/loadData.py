import gzip
import json
import torch
from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw_filename = filename
        with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
            self.graphs_dicts_list = json.load(f)
        self.num_graphs = len(self.graphs_dicts_list)
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        graph_dict = self.graphs_dicts_list[idx]
        data_obj = dictToGraphObject(graph_dict)

        # Nome attributo corretto e coerente
        data_obj.original_index = torch.tensor([idx], dtype=torch.long)

        # DEBUG PRINT (puoi rimuoverlo dopo aver confermato che funziona)
        print(f"DEBUG GraphDataset.get(idx={idx}): hasattr(data_obj, 'original_index') = {hasattr(data_obj, 'original_index')}")
        if hasattr(data_obj, 'original_index'):
            print(f"DEBUG GraphDataset.get(idx={idx}): data_obj.original_index = {data_obj.original_index}")

        return data_obj

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    edge_attr_raw = graph_dict.get("edge_attr")
    edge_attr = None
    if edge_attr_raw is not None and isinstance(edge_attr_raw, list) and len(edge_attr_raw) > 0:
        try:
            edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)
        except Exception as e:
            print(f"Warning: Errore conversione edge_attr: {e}. Dati: {edge_attr_raw}. edge_attr sarà None.")
            edge_attr = None

    num_nodes = graph_dict["num_nodes"]

    y_val_raw = graph_dict.get("y")
    y = None
    if y_val_raw is not None:
        if isinstance(y_val_raw, list) and len(y_val_raw) > 0:
            try:
                y = torch.tensor([y_val_raw[0]], dtype=torch.long)
            except Exception as e:
                print(f"Warning: Errore conversione y: {e}. Dati: {y_val_raw}. y sarà None.")
        elif isinstance(y_val_raw, (int, float)):
            y = torch.tensor([y_val_raw], dtype=torch.long)
        # else: y rimane None o gestisci altri formati qui

    data_obj = Data(edge_index=edge_index, num_nodes=num_nodes)
    if edge_attr is not None:
        data_obj.edge_attr = edge_attr
    if y is not None:
        data_obj.y = y
    else:
        # Se y è None, PyG non lo aggiungerà, il che è ok se il modello non lo usa sempre.
        # Ma il tuo codice di training probabilmente lo usa per calcolare la loss e l'accuracy.
        # Potrebbe essere necessario gestire diversamente i campioni senza y.
        # print(f"Warning: y è None per un campione. Chiavi dict: {list(graph_dict.keys())}")
        pass

    # Aggiungi altri attributi obbligatori se necessario (es. 'x')
    # if "x" in graph_dict and graph_dict["x"] is not None:
    #    data_obj.x = torch.tensor(graph_dict["x"], dtype=torch.float)
    # else:
    #    # Gestisci il caso in cui x manchi, se è un problema
    #    print(f"Warning: 'x' mancante o None per un campione. Chiavi dict: {list(graph_dict.keys())}")
    #    # Potresti dover impostare un valore predefinito o lanciare un errore
    #    # data_obj.x = torch.zeros((num_nodes, feature_dim)) # Esempio di placeholder

    return data_obj