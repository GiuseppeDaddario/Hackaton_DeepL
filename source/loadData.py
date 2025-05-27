import gzip
import json
import os
import torch
from torch_geometric.data import Dataset, Data

# Assicurati che node_feature_dim sia passato correttamente a questa funzione
def add_zeros(data_obj, node_feature_dim):
    # 1. Gestione di data.x (feature dei nodi)
    if not hasattr(data_obj, 'x') or data_obj.x is None:
        if data_obj.num_nodes is not None and data_obj.num_nodes > 0:
            data_obj.x = torch.zeros((data_obj.num_nodes, node_feature_dim), dtype=torch.float)
        elif data_obj.num_nodes is not None and data_obj.num_nodes == 0:
            data_obj.x = torch.empty((0, node_feature_dim), dtype=torch.float)
        else: # num_nodes è None o negativo, problematico
            # Potresti voler sollevare un errore qui o impostare num_nodes a 0 e creare x vuoto
            # print(f"Warning (add_zeros): num_nodes problematico: {data_obj.num_nodes}. Imposto x vuoto.")
            data_obj.num_nodes = 0 # Correggi num_nodes se possibile
            data_obj.x = torch.empty((0, node_feature_dim), dtype=torch.float)

    # 2. Gestione di data.edge_index
    if not hasattr(data_obj, 'edge_index') or data_obj.edge_index is None:
        data_obj.edge_index = torch.empty((2, 0), dtype=torch.long)
        # Se edge_index è vuoto, anche edge_attr dovrebbe esserlo
        if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
            # Calcola la dimensione delle feature degli archi dal primo attributo edge_attr non vuoto, se esiste,
            # altrimenti assumi 0 o una dimensione di default se la conosci.
            # Questo è complicato se edge_attr è eterogeneo.
            # Per semplicità, se edge_index è vuoto, rendi edge_attr vuoto con 0 feature se non è già None.
            dim1_ea = 0
            if data_obj.edge_attr.dim() > 1 and data_obj.edge_attr.size(0) > 0 : # se ha una forma tipo [num_edges, num_edge_features]
                dim1_ea = data_obj.edge_attr.size(1)

            data_obj.edge_attr = torch.empty((0, dim1_ea), dtype=torch.float)


    # 3. Assicurati che data.y esista (può essere None, PyG lo gestisce)
    if not hasattr(data_obj, 'y'):
        data_obj.y = None

    # 4. Assicurati che original_idx esista (come tensore)
    if not hasattr(data_obj, 'original_idx') or data_obj.original_idx is None:
        # Questo non dovrebbe accadere se dictToGraphObject lo imposta sempre.
        # print("Warning (add_zeros): original_idx mancante, impostazione a placeholder -1.")
        data_obj.original_idx = torch.tensor([-1], dtype=torch.long) # Placeholder problematico

    return data_obj


def dictToGraphObject(graph_dict, idx_in_file, node_feature_dim_for_json_load=None): # Opzionale
    # Edge Index (obbligatorio)
    edge_index_list = graph_dict.get("edge_index")
    if edge_index_list is None: # Deve esistere
        # print(f"Errore: 'edge_index' mancante per grafo {idx_in_file}.")
        edge_index = torch.empty((2,0), dtype=torch.long) # Fallback problematico
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            # print(f"Errore: 'edge_index' malformato per grafo {idx_in_file}. Shape: {edge_index.shape}")
            edge_index = torch.empty((2,0), dtype=torch.long) # Fallback

    # Num Nodes (obbligatorio)
    num_nodes = graph_dict.get("num_nodes")
    if num_nodes is None or not isinstance(num_nodes, int) or num_nodes < 0:
        # print(f"Errore: 'num_nodes' mancante o non valido per grafo {idx_in_file}: {num_nodes}. Provo a dedurlo.")
        if edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else 0
        else:
            num_nodes = 0 # Se non ci sono archi
        # print(f"Num nodes dedotto/corretto a: {num_nodes}")


    # Node Features (x) - se presenti nel JSON
    node_features_x = None
    if "node_feat" in graph_dict and graph_dict["node_feat"] is not None:
        try:
            node_features_x = torch.tensor(graph_dict["node_feat"], dtype=torch.float)
            if node_features_x.dim() == 1 and num_nodes > 0 and node_features_x.size(0) == num_nodes * node_feature_dim_for_json_load : # Se è flat e la dim è nota
                node_features_x = node_features_x.view(num_nodes, node_feature_dim_for_json_load)
            if node_features_x.shape[0] != num_nodes and num_nodes > 0 : # num_nodes deve essere >0 per avere features
                # print(f"Warning: Mismatch tra num_nodes ({num_nodes}) e node_feat.shape[0] ({node_features_x.shape[0]}) per grafo {idx_in_file}. Ignoro node_feat.")
                node_features_x = None
            elif num_nodes == 0 and node_features_x.numel() > 0: # Nodi zero ma features presenti
                # print(f"Warning: num_nodes è 0 ma node_feat presenti per grafo {idx_in_file}. Ignoro node_feat.")
                node_features_x = None
        except Exception as e:
            # print(f"Errore nel caricare node_feat per grafo {idx_in_file}: {e}. Ignoro node_feat.")
            node_features_x = None

    # Edge Features (edge_attr) - opzionale
    edge_attr = None
    if "edge_attr" in graph_dict and graph_dict["edge_attr"] is not None:
        try:
            edge_attr_list = graph_dict["edge_attr"]
            if isinstance(edge_attr_list, list) and len(edge_attr_list) > 0:
                # Verifica se il numero di archi corrisponde
                num_edges_from_idx = edge_index.size(1)
                # Questo può essere complesso se edge_attr è una lista di liste o flat.
                # Assumiamo sia una lista di liste [num_edges, num_edge_features] o una lista flat.
                # Per ora, lo carichiamo così com'è se non vuoto.
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
                if edge_attr.size(0) != num_edges_from_idx and num_edges_from_idx > 0 :
                    # print(f"Warning: Mismatch tra num_edges ({num_edges_from_idx}) e edge_attr.size(0) ({edge_attr.size(0)}) per grafo {idx_in_file}. Ignoro edge_attr.")
                    edge_attr = None
                elif num_edges_from_idx == 0 and edge_attr.numel() > 0:
                    # print(f"Warning: num_edges è 0 ma edge_attr presenti per grafo {idx_in_file}. Ignoro edge_attr.")
                    edge_attr = None

        except Exception as e:
            # print(f"Errore nel caricare edge_attr per grafo {idx_in_file}: {e}. Ignoro edge_attr.")
            edge_attr = None

    # Labels (y) - opzionale
    y_tensor = None
    y_val = graph_dict.get("y")
    if y_val is not None:
        try:
            if isinstance(y_val, list) and len(y_val) > 0:
                y_tensor = torch.tensor(y_val[0], dtype=torch.long) # Assumendo che il primo elemento sia l'etichetta
            elif isinstance(y_val, (int, float)):
                y_tensor = torch.tensor(y_val, dtype=torch.long)
            # else: y_val ha un formato inatteso, y_tensor rimane None
        except Exception as e:
            # print(f"Errore nel processare y_val per grafo {idx_in_file}: {y_val}, errore: {e}. y rimane None.")
            y_tensor = None # Fallback a None

    data_obj = Data(
        x=node_features_x,      # Sarà creato da add_zeros se None e num_nodes > 0
        edge_index=edge_index,
        edge_attr=edge_attr,    # Può essere None
        num_nodes=num_nodes,
        y=y_tensor,             # Può essere None
        original_idx=torch.tensor([idx_in_file], dtype=torch.long) # Sempre presente
    )
    return data_obj


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, pre_filter=None, node_feature_dim_for_json_load=None): # Aggiunto per dictToGraphObject
        self.raw_filename = filename
        self.graphs_dicts_list = []
        self.transform_func = transform # Salva la funzione di trasformazione
        self.node_feature_dim_for_json_load = node_feature_dim_for_json_load

        try:
            with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
                self.graphs_dicts_list = json.load(f)
        except Exception as e:
            print(f"ERRORE FATALE: Impossibile caricare o fare il parsing del dataset {self.raw_filename}: {e}")
            raise

        self.num_graphs = len(self.graphs_dicts_list)
        if self.num_graphs == 0:
            print(f"ATTENZIONE: Nessun grafo caricato da {self.raw_filename}. Il training fallirà.")

        # root=None se non usi il sistema di caching di PyG per i file processati.
        super().__init__(root=None, transform=None, pre_transform=pre_transform, pre_filter=pre_filter)
        # La transform viene applicata in get() manualmente per avere più controllo.

    @property
    def raw_file_names(self):
        return [os.path.basename(self.raw_filename)]

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return self.num_graphs

    def get(self, idx):
        if not (0 <= idx < self.num_graphs):
            raise IndexError(f"Indice dataset {idx} fuori range per dataset di lunghezza {self.num_graphs}")

        graph_dict = self.graphs_dicts_list[idx]
        # Passa node_feature_dim_for_json_load se serve a dictToGraphObject per interpretare 'node_feat' flat
        data_obj = dictToGraphObject(graph_dict, idx, self.node_feature_dim_for_json_load)

        # Applica la trasformazione (es. add_zeros) QUI.
        # add_zeros deve essere robusta e assicurare che data.x, data.edge_index, ecc. siano consistenti.
        if self.transform_func:
            data_obj = self.transform_func(data_obj)

        # Controlli di sanità finali sull'oggetto Data prima che vada al collate_fn
        if data_obj.num_nodes is None or data_obj.num_nodes < 0:
            raise ValueError(f"Data object per idx {idx} ha num_nodes invalido: {data_obj.num_nodes}")
        if not hasattr(data_obj, 'x') or data_obj.x is None:
            if data_obj.num_nodes > 0:
                raise ValueError(f"Data object per idx {idx} (num_nodes={data_obj.num_nodes}) non ha attributo 'x' o è None dopo la trasformazione.")
            elif data_obj.num_nodes == 0: # Se 0 nodi, x deve essere (0, dim)
                data_obj.x = torch.empty((0, self.node_feature_dim_for_json_load if self.node_feature_dim_for_json_load else 0), dtype=torch.float) # Assumendo che add_zeros lo abbia già fatto
        if not hasattr(data_obj, 'edge_index') or data_obj.edge_index is None:
            raise ValueError(f"Data object per idx {idx} non ha attributo 'edge_index' o è None dopo la trasformazione.")
        if not hasattr(data_obj, 'original_idx') or data_obj.original_idx is None:
            raise ValueError(f"Data object per idx {idx} non ha attributo 'original_idx' o è None.")

        return data_obj