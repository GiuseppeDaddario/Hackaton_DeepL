# Salva questo file, ad esempio, come: GNNPlus/dataset/my_custom_graph_dataset.py

import gzip
import json
import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os
import os.path as osp

# Se 'add_zeros_transform' è una classe custom, importala.
# Esempio: from GNNPlus.transforms import AddZerosTransform
# Altrimenti, se è solo una funzione, la passerai come pre_transform.

class MyPPA6Classes(InMemoryDataset):
    def __init__(self, root, raw_filename="data.json.gz", transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Directory root dove GraphGym gestirà il dataset.
                        GraphGym la imposterà da cfg.dataset.dir e cfg.dataset.name
                        (es. 'data/MyPPA').
            raw_filename (str): Nome del file .json.gz da cercare in root/raw/.
                                Lo specificherai nel file YAML.
            transform (callable, optional): Trasformazione PyG da applicare on-the-fly.
            pre_transform (callable, optional): Trasformazione PyG da applicare prima di salvare
                                                i dati processati. La tua 'add_zeros_transform'
                                                dovrebbe essere passata qui.
            pre_filter (callable, optional): Filtro PyG.
        """
        self.source_raw_filename = raw_filename # Memorizza il nome del file sorgente
        # Se add_zeros_transform è una pre_transform speciale, la logica
        # per la sua applicazione è nel metodo 'process'.
        # Qui, pre_transform è lo standard di PyG.
        super().__init__(root, transform, pre_transform, pre_filter)
        # InMemoryDataset caricherà automaticamente da self.processed_paths[0]
        self.data, self.slices = torch.load(self.processed_paths[0])
        # Le proprietà come num_classes, num_node_features, ecc., saranno
        # inferite da GraphGym dall'oggetto self.data.

    @property
    def raw_dir(self):
        # GraphGym si aspetta che i file raw siano in questa sottocartella
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        # GraphGym si aspetta che i file processati siano in questa sottocartella
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # Il file che GraphGym cercherà in self.raw_dir.
        # Deve corrispondere a quello che hai messo lì.
        return [self.source_raw_filename]

    @property
    def processed_file_names(self):
        # Il nome del file che conterrà l'intero dataset processato.
        return ['data.pt']

    def download(self):
        # Questa funzione viene chiamata da PyG se i file in raw_file_names non esistono.
        # Siccome il tuo dataset è locale, qui non c'è nulla da scaricare.
        # L'utente deve assicurarsi che il file specificato da 'raw_filename'
        # sia copiato manualmente in self.raw_dir (es. data/MyPPA/raw/mio_dataset.json.gz)
        # primadi eseguire lo script per la prima volta.
        if not osp.exists(osp.join(self.raw_dir, self.source_raw_filename)):
            raise FileNotFoundError(
                f"File raw '{self.source_raw_filename}' non trovato in '{self.raw_dir}'. "
                f"Per favore, copialo lì prima di eseguire il training."
            )

    def process(self):
        # Questa è la tua logica di caricamento e processamento originale, adattata.
        full_raw_path = osp.join(self.raw_dir, self.source_raw_filename)

        with gzip.open(full_raw_path, "rt", encoding="utf-8") as f:
            graphs_raw_list = json.load(f)

        processed_data_list = []
        # self.graphs_dicts_list = graphs_raw_list # Lo salvi se serve ancora per GCOD

        for i, graph_dict in enumerate(graphs_raw_list):
            edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            num_nodes = graph_dict["num_nodes"]

            # --- GESTIONE FEATURE DEI NODI (x) ---
            # La tua classe originale non aveva 'x'. I GNN di solito ne hanno bisogno.
            # Devi decidere come gestirle.
            # Esempio 1: Se sono nel json_dict come "node_features"
            node_features_raw = graph_dict.get("node_features")
            if node_features_raw is not None:
                x = torch.tensor(node_features_raw, dtype=torch.float)
            else:
                # Esempio 2: Placeholder se non ci sono (es. feature costanti o degree)
                # Potresti anche creare degree features qui se necessario.
                x = torch.ones((num_nodes, 1), dtype=torch.float) # Feature placeholder
                if num_nodes == 0: # Gestisci grafi vuoti se possibile
                    x = torch.empty((0,1), dtype=torch.float)


            edge_attr_raw = graph_dict.get("edge_attr")
            edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float) if edge_attr_raw is not None else None
            if edge_attr is not None and num_nodes == 0 and edge_attr.numel() > 0: # Coerenza per grafi vuoti
                edge_attr = torch.empty((0, edge_attr.size(1) if edge_attr.ndim == 2 else 0) , dtype=torch.float)


            y_val_list = graph_dict.get("y")
            y_tensor = None
            if y_val_list and len(y_val_list) > 0: # Assumendo classificazione di grafo
                y_value = y_val_list[0]
                y_tensor = torch.tensor([y_value], dtype=torch.long) # Per class. grafo, y è [batch_size, 1] o [batch_size]
            # else: Lascia y_tensor a None se non c'è etichetta per questo grafo.

            data_obj = Data(
                x=x, # FEATURE DEI NODI SONO NECESSARIE
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=y_tensor
                # original_idx=torch.tensor([i], dtype=torch.long) # Se ti serve ancora per gcodLoss
            )
            processed_data_list.append(data_obj)

        # --- APPLICAZIONE DI PRE_TRANSFORM / PRE_FILTER (Standard PyG) ---
        # Se hai passato `add_zeros_transform` come `pre_transform` al costruttore,
        # PyG lo applicherà automaticamente qui se lo includi nella chiamata a `super().process()`.
        # O lo applichi manualmente se la logica è complessa:
        if self.pre_filter is not None:
            processed_data_list = [data for data in processed_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            # Qui 'self.pre_transform' è la trasformazione passata a __init__,
            # che potrebbe essere la tua 'add_zeros_transform' o un'altra.
            processed_data_list = [self.pre_transform(data) for data in processed_data_list]

        # Salva la lista di oggetti Data processati in un singolo file .pt
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices), self.processed_paths[0])

    # Non hai più bisogno di definire len(), get(), num_classes(), ecc.
    # InMemoryDataset li gestisce per te, inferendoli da self.data e self.slices.
    # GraphGym inferirà dim_in, dim_out, ecc. dal dataset caricato.