import gzip
import json
import os
import torch
from torch_geometric.data import Dataset, Data

# ==============================================================================
# 1. Funzione dictToGraphObject
# ==============================================================================
# Questo parametro `node_feature_dim_for_json_load` serve solo se nel tuo JSON
# le 'node_feat' sono una lista flat e hai bisogno della dimensione per fare il reshape.
# Se 'node_feat' è già una lista di liste (num_nodes, feature_dim) o se non hai 'node_feat'
# nel JSON (e ti affidi a add_zeros), puoi anche non usarlo.
def dictToGraphObject(graph_dict, idx_in_file, node_feature_dim_for_json_load=None):
    """
    Converte un dizionario da JSON in un oggetto Data "grezzo".
    Gestisce attributi mancanti o malformati impostandoli a None o a valori di default sicuri.
    La funzione 'add_zeros' (transform) si occuperà poi di finalizzare l'oggetto Data.
    """

    # --- Gestione num_nodes (numero di nodi) ---
    num_nodes = graph_dict.get("num_nodes")
    if not isinstance(num_nodes, int) or num_nodes < 0:
        # print(f"Warning (dictToGraph, idx {idx_in_file}): num_nodes mancante o invalido ({num_nodes}). Provo a dedurlo o imposto a 0.")
        # Prova a dedurlo da edge_index se presente e valido, altrimenti 0.
        # Questa logica di deduzione può essere omessa se num_nodes è sempre affidabile nel JSON.
        temp_edge_index_list = graph_dict.get("edge_index")
        if isinstance(temp_edge_index_list, list) and len(temp_edge_index_list) > 0 and \
                all(isinstance(sublist, list) and len(sublist) > 0 for sublist in temp_edge_index_list if isinstance(temp_edge_index_list[0], list)) : # Se è lista di liste
            # Questa è una logica di deduzione base, potrebbe non essere perfetta per tutti i formati di edge_index
            try:
                max_node_idx = 0
                for sublist in temp_edge_index_list: # [[n1,n2,n3..], [m1,m2,m3..]]
                    if sublist: max_node_idx = max(max_node_idx, max(sublist))
                num_nodes = max_node_idx + 1 if max_node_idx >=0 else 0
            except: num_nodes = 0 # Fallback
        elif isinstance(temp_edge_index_list, list) and len(temp_edge_index_list) == 2 and \
                all(isinstance(sublist, list) for sublist in temp_edge_index_list) : # Formato [[row_indices],[col_indices]]
            try:
                flat_indices = [item for sublist in temp_edge_index_list for item in sublist]
                num_nodes = max(flat_indices) + 1 if flat_indices else 0
            except: num_nodes = 0
        else:
            num_nodes = 0
        # print(f"Warning (dictToGraph, idx {idx_in_file}): num_nodes dedotto/impostato a {num_nodes}.")

    # --- Gestione edge_index (connettività del grafo) ---
    edge_index_list = graph_dict.get("edge_index")
    edge_index = None # Inizializza a None
    if isinstance(edge_index_list, list):
        try:
            # PyG si aspetta edge_index come [2, num_edges]
            # Se il tuo JSON ha un formato diverso (es. lista di coppie), devi trasformarlo.
            # Assumendo che il tuo JSON sia già [ [sources...], [targets...] ] o possa esserlo.
            # Se è lista di coppie [[s1,t1], [s2,t2], ...], allora:
            # if edge_index_list and isinstance(edge_index_list[0], list) and len(edge_index_list[0]) == 2:
            #    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            # else: # Assumendo formato [2, num_edges] o che torch.tensor lo gestisca
            edge_index_tensor_try = torch.tensor(edge_index_list, dtype=torch.long)
            if edge_index_tensor_try.dim() == 2 and edge_index_tensor_try.size(0) == 2:
                edge_index = edge_index_tensor_try
            elif edge_index_tensor_try.dim() == 2 and edge_index_tensor_try.size(1) == 2: # Potrebbe essere [num_edges, 2]
                edge_index = edge_index_tensor_try.t().contiguous()
            # else: edge_index rimane None se il formato non è riconosciuto o vuoto

        except Exception: # e:
            # print(f"Warning (dictToGraph, idx {idx_in_file}): Errore nel processare edge_index. Rimane None. Errore: {e}")
            edge_index = None # Fallback a None

    # Se edge_index è ancora None o vuoto e ci sono nodi, crea un edge_index vuoto.
    # add_zeros si assicurerà che esista sempre.
    # if edge_index is None:
    #     edge_index = torch.empty((2,0), dtype=torch.long)


    # --- Gestione node_features_x (feature dei nodi, se presenti nel JSON) ---
    # `add_zeros` si occuperà di creare `data.x` se `node_features_x` è `None`
    node_features_x = None
    if "node_feat" in graph_dict and graph_dict["node_feat"] is not None:
        try:
            feat_data = graph_dict["node_feat"]
            if isinstance(feat_data, list) and len(feat_data) > 0:
                node_features_x_try = torch.tensor(feat_data, dtype=torch.float)
                # Se è flat (1D) e conosci num_nodes e node_feature_dim_for_json_load, fai reshape
                if node_features_x_try.dim() == 1 and \
                        node_feature_dim_for_json_load is not None and \
                        num_nodes > 0 and \
                        node_features_x_try.size(0) == num_nodes * node_feature_dim_for_json_load:
                    node_features_x = node_features_x_try.view(num_nodes, node_feature_dim_for_json_load)
                # Se è già 2D e la prima dimensione corrisponde a num_nodes
                elif node_features_x_try.dim() == 2 and node_features_x_try.size(0) == num_nodes:
                    node_features_x = node_features_x_try
                # else:
                # print(f"Warning (dictToGraph, idx {idx_in_file}): node_feat ha forma/tipo inatteso o mismatch con num_nodes. x sarà None.")
        except Exception: # e:
            # print(f"Warning (dictToGraph, idx {idx_in_file}): Errore nel caricare node_feat. x sarà None. Errore: {e}")
            node_features_x = None # Fallback a None


    # --- Gestione edge_attr (feature degli archi, opzionale) ---
    edge_attr = None
    if "edge_attr" in graph_dict and graph_dict["edge_attr"] is not None:
        try:
            attr_data = graph_dict["edge_attr"]
            if isinstance(attr_data, list) and len(attr_data) > 0:
                edge_attr_try = torch.tensor(attr_data, dtype=torch.float)
                # Verifica che il numero di archi (prima dimensione) corrisponda a edge_index.size(1)
                num_edges_from_idx = edge_index.size(1) if edge_index is not None else 0
                if edge_attr_try.dim() >=1 and edge_attr_try.size(0) == num_edges_from_idx:
                    edge_attr = edge_attr_try
                # else:
                # print(f"Warning (dictToGraph, idx {idx_in_file}): edge_attr ha forma/tipo inatteso o mismatch con num_edges. edge_attr sarà None.")
        except Exception: # e:
            # print(f"Warning (dictToGraph, idx {idx_in_file}): Errore caricamento edge_attr. Sarà None. Errore: {e}")
            edge_attr = None # Fallback a None


    # --- Gestione y (etichette, opzionale) ---
    y_tensor = None
    y_val = graph_dict.get("y")
    if y_val is not None:
        try:
            if isinstance(y_val, list) and y_val: # Se è una lista non vuota
                y_tensor = torch.tensor([y_val[0]], dtype=torch.long) # Prendi il primo elemento, wrappalo in lista per tensor
            elif isinstance(y_val, (int, float)): # Se è già uno scalare
                y_tensor = torch.tensor([y_val], dtype=torch.long) # Wrappalo in lista per tensor
            # else: y_tensor rimane None
        except Exception: # e:
            # print(f"Warning (dictToGraph, idx {idx_in_file}): Errore processamento y. Sarà None. Val: {y_val}. Errore: {e}")
            y_tensor = None # Fallback a None

    # --- Creazione dell'oggetto Data ---
    # Includi solo gli attributi che hanno un valore (non None) o quelli obbligatori
    # `add_zeros` si occuperà di popolare `x` se è `None` e `num_nodes > 0`.
    # `add_zeros` si assicurerà anche che `edge_index` esista.

    # Inizializza il dizionario degli attributi per Data
    data_kwargs = {'num_nodes': num_nodes} # num_nodes è sempre presente

    if node_features_x is not None:
        data_kwargs['x'] = node_features_x
    if edge_index is not None:
        data_kwargs['edge_index'] = edge_index
    if edge_attr is not None:
        data_kwargs['edge_attr'] = edge_attr
    if y_tensor is not None:
        data_kwargs['y'] = y_tensor

    # original_idx è sempre presente
    data_kwargs['original_idx'] = torch.tensor([idx_in_file], dtype=torch.long)

    data_obj = Data(**data_kwargs)

    return data_obj

# ==============================================================================
# 2. Funzione add_zeros (transform)
# ==============================================================================
def add_zeros(data_obj, node_feature_dim):
    """
    Finalizza l'oggetto Data:
    1. Crea data.x (feature dei nodi) se non esiste, usando `node_feature_dim`.
    2. Assicura che data.edge_index esista.
    3. Assicura che data.y e data.original_idx esistano (y può essere None).
    """
    current_num_nodes = data_obj.num_nodes if hasattr(data_obj, 'num_nodes') and data_obj.num_nodes is not None else 0

    # --- Finalizzazione di data.x ---
    if not hasattr(data_obj, 'x') or data_obj.x is None:
        if current_num_nodes > 0:
            data_obj.x = torch.zeros((current_num_nodes, node_feature_dim), dtype=torch.float)
        else: # num_nodes è 0
            data_obj.x = torch.empty((0, node_feature_dim), dtype=torch.float)
    elif torch.is_tensor(data_obj.x) and data_obj.x.shape[0] != current_num_nodes :
        # print(f"Warning (add_zeros): Mismatch tra data.x.shape[0] ({data_obj.x.shape[0]}) e num_nodes ({current_num_nodes}). Ricreo x.")
        if current_num_nodes > 0:
            data_obj.x = torch.zeros((current_num_nodes, node_feature_dim), dtype=torch.float)
        else:
            data_obj.x = torch.empty((0, node_feature_dim), dtype=torch.float)


    # --- Finalizzazione di data.edge_index ---
    if not hasattr(data_obj, 'edge_index') or data_obj.edge_index is None:
        data_obj.edge_index = torch.empty((2, 0), dtype=torch.long)

    # Se edge_index è vuoto, anche edge_attr (se esiste) dovrebbe essere vuoto o None
    if data_obj.edge_index.size(1) == 0:
        if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
            if data_obj.edge_attr.numel() > 0 : # Se ha elementi ma non ci sono archi
                # print(f"Warning (add_zeros): edge_index è vuoto ma edge_attr ha elementi. Svuoto edge_attr.")
                # Determina la dimensione delle feature degli archi da edge_attr esistente se possibile
                dim1_ea = data_obj.edge_attr.size(1) if data_obj.edge_attr.dim() > 1 else 0
                data_obj.edge_attr = torch.empty((0, dim1_ea), dtype=torch.float)
    # else: # Ci sono archi
    # Se edge_attr esiste, verifica la consistenza con il numero di archi
    # if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
    #    if data_obj.edge_attr.size(0) != data_obj.edge_index.size(1):
    #        print(f"Warning (add_zeros): Mismatch tra edge_attr e edge_index. Imposto edge_attr a None.")
    #        data_obj.edge_attr = None


    # --- Assicura che y e original_idx esistano (y può essere None) ---
    if not hasattr(data_obj, 'y'):
        data_obj.y = None # y può legittimamente essere None

    if not hasattr(data_obj, 'original_idx') or data_obj.original_idx is None:
        # Questo è un problema se dictToGraphObject non l'ha impostato.
        # print(f"ERRORE (add_zeros): original_idx non trovato sull'oggetto Data. Questo non dovrebbe accadere.")
        # Fallback problematico:
        data_obj.original_idx = torch.tensor([-1], dtype=torch.long)

    return data_obj


# ==============================================================================
# 3. Classe GraphDataset
# ==============================================================================
class GraphDataset(Dataset):
    def __init__(self, filename, transform_lambda=None, # Rinomina transform in transform_lambda per chiarezza
                 node_feature_dim_for_json_load=None, # Passato a dictToGraphObject
                 node_feature_dim_for_add_zeros=None): # Passato a add_zeros dentro la lambda
        self.raw_filename = filename
        self.graphs_dicts_list = []

        # Salva i parametri che verranno usati dalla lambda della transform
        self.node_feature_dim_for_add_zeros = node_feature_dim_for_add_zeros

        # Crea la funzione di trasformazione completa che verrà chiamata
        if transform_lambda is not None:
            self.effective_transform = transform_lambda # La lambda dovrebbe già avere node_feature_dim
        else:
            self.effective_transform = None

        self.node_feature_dim_for_json_load_internal = node_feature_dim_for_json_load

        try:
            with gzip.open(self.raw_filename, "rt", encoding="utf-8") as f:
                self.graphs_dicts_list = json.load(f)
        except Exception as e:
            print(f"ERRORE FATALE: Impossibile caricare o fare il parsing del dataset {self.raw_filename}: {e}")
            raise

        self.num_graphs = len(self.graphs_dicts_list)
        if self.num_graphs == 0:
            print(f"ATTENZIONE: Nessun grafo caricato da {self.raw_filename}.")

        # Non passare self.effective_transform al costruttore super, lo applichiamo manualmente in get().
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

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
            raise IndexError(f"Indice dataset {idx} fuori range per dataset di {self.num_graphs} grafi.")

        graph_dict = self.graphs_dicts_list[idx]
        data_obj = dictToGraphObject(graph_dict, idx, self.node_feature_dim_for_json_load_internal)

        if self.effective_transform:
            # La lambda passata come transform_lambda dovrebbe chiamare add_zeros
            # con il corretto node_feature_dim.
            # Esempio di chiamata nel main:
            # transform=lambda data: add_zeros(data, node_feature_dim=args.emb_dim)
            data_obj = self.effective_transform(data_obj)

        # Controlli di sanità finali (opzionali ma utili per il debug)
        if data_obj.num_nodes is None or data_obj.num_nodes < 0:
            raise ValueError(f"Data obj idx {idx} ha num_nodes invalido: {data_obj.num_nodes} dopo transform.")
        if not hasattr(data_obj, 'x') or data_obj.x is None:
            if data_obj.num_nodes > 0:
                raise ValueError(f"Data obj idx {idx} (nodi={data_obj.num_nodes}) non ha 'x' dopo transform.")
            # Se num_nodes è 0, x dovrebbe essere (0, dim), gestito da add_zeros
        if not hasattr(data_obj, 'edge_index') or data_obj.edge_index is None:
            raise ValueError(f"Data obj idx {idx} non ha 'edge_index' dopo transform.")
        if not hasattr(data_obj, 'original_idx') or data_obj.original_idx is None:
            raise ValueError(f"Data obj idx {idx} non ha 'original_idx' dopo transform.")

        # Debug print che hai usato prima, se serve ancora:
        # print(f"DEBUG Dataset get(idx={idx}):")
        # print(f"  num_nodes: {data_obj.num_nodes}")
        # for attr_key in ['x', 'edge_index', 'edge_attr', 'y', 'original_idx']:
        #     if hasattr(data_obj, attr_key):
        #         val = getattr(data_obj, attr_key)
        #         if torch.is_tensor(val): print(f"  {attr_key}: Tensor, shape={val.shape}, dtype={val.dtype}")
        #         else: print(f"  {attr_key}: {type(val)}, value={val}")
        #     else: print(f"  {attr_key}: NOT PRESENT")
        # print("-" * 20)

        return data_obj