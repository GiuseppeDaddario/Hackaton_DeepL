import gzip
import json
import torch
from torch_geometric.data import Dataset, Data # Data deve essere importato da torch_geometric.data

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw_filename = filename # Meglio usare un nome diverso da self.raw se Dataset ha già un self.raw

        # Carica tutti i dizionari dei grafi una volta
        with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
            self.graphs_dicts_list = json.load(f)

        self.num_graphs = len(self.graphs_dicts_list)

        # Chiamare super().__init__ è importante.
        # Passa None per root se non stai usando le funzionalità di download/processo di PyG
        # che salvano in una directory root.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def len(self):
        return self.num_graphs

    def get(self, idx): # idx qui è l'indice fornito dal DataLoader (dopo eventuale shuffle)
        graph_dict = self.graphs_dicts_list[idx]

        # Converti il dizionario in un oggetto Data
        data_obj = dictToGraphObject(graph_dict) # La funzione dictToGraphObject crea e restituisce l'oggetto Data

        # Aggiungi l'indice 'idx' come attributo 'original_index' all'oggetto Data.
        # PyG si occuperà di collazionare questo in batch.original_index.
        # Il nome "original_index" qui si riferisce all'indice *all'interno del dataset*
        # come è stato caricato e indicizzato da DataLoader. Se il DataLoader usa shuffle=True,
        # questo 'idx' sarà un indice shufflato dell'elenco self.graphs_dicts_list.
        # Questo è corretto perché self.u e self.prev_gnn_embeddings sono indicizzati
        # con questi indici (0 a num_examp-1).
        data_obj.original_index = torch.tensor([idx], dtype=torch.long)

        return data_obj

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    # Gestione più robusta per edge_attr
    edge_attr_raw = graph_dict.get("edge_attr")
    if edge_attr_raw is not None and len(edge_attr_raw) > 0: # Controlla anche se è una lista vuota
        try:
            edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)
        except Exception as e:
            print(f"Warning: Errore nella conversione di edge_attr: {e}. edge_attr sarà None. Dati: {edge_attr_raw}")
            edge_attr = None
    else:
        edge_attr = None

    num_nodes = graph_dict["num_nodes"]

    # Gestione più robusta per y
    y_val_raw = graph_dict.get("y")
    y = None # Default
    if y_val_raw is not None:
        if isinstance(y_val_raw, list) and len(y_val_raw) > 0:
            try:
                # Assumiamo che y sia un singolo intero per la classificazione
                y = torch.tensor([y_val_raw[0]], dtype=torch.long) # Metti [y_val_raw[0]] per renderlo (1,)
            except Exception as e:
                print(f"Warning: Errore nella conversione di y: {e}. y sarà None. Dati: {y_val_raw}")
        elif isinstance(y_val_raw, (int, float)): # Se y è già uno scalare
            y = torch.tensor([y_val_raw], dtype=torch.long)
        # else: y rimane None o gestisci altri formati

    # È buona pratica assicurarsi che tutti gli attributi attesi da Data siano presenti,
    # anche se sono None.
    data_obj = Data(edge_index=edge_index, num_nodes=num_nodes)
    if edge_attr is not None:
        data_obj.edge_attr = edge_attr
    if y is not None:
        data_obj.y = y
    else:
        # Se y è cruciale e non può essere None, potresti voler lanciare un errore
        # o gestire questo caso in modo specifico.
        # Per ora, se y è None, non verrà aggiunto all'oggetto Data.
        # Tuttavia, il tuo codice di training si aspetta graphs.y
        print(f"Warning: y è None per graph_dict: {list(graph_dict.keys())}")


    # Aggiungi altri attributi se presenti nel graph_dict e necessari
    # Esempio:
    # if "x" in graph_dict:
    #     data_obj.x = torch.tensor(graph_dict["x"], dtype=torch.float)

    return data_obj