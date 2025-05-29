import logging  # Aggiunto per i warning interni

# In source/models.py (questa è la versione che si aspetta 'in_channels', etc.)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool, BatchNorm
from torch_geometric.nn import TransformerConv, \
    GlobalAttention


# Nuovo Blocco Transformer Convoluzionale
class TransformerConvBlock(torch.nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout_ratio=0.1, concat=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if concat:
            assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads if concat is True"
            self.att_out_channels = emb_dim // num_heads
        else:
            self.att_out_channels = emb_dim


        self.transformer_conv = TransformerConv(
            in_channels=emb_dim,
            out_channels=self.att_out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout_ratio,
            edge_dim=emb_dim
        )
        self.norm1 = torch.nn.LayerNorm(emb_dim)
        self.norm2 = torch.nn.LayerNorm(emb_dim)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_ratio),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.dropout = torch.nn.Dropout(dropout_ratio)


    def forward(self, x, edge_index, edge_embedding):
        # Self-Attention part
        # x (N, emb_dim), edge_index (2, E), edge_embedding (E, emb_dim)
        attended_x = self.transformer_conv(x, edge_index, edge_attr=edge_embedding)
        attended_x = self.dropout(attended_x) # Dropout dopo l'attention
        x = self.norm1(x + attended_x)  # Connessione residuale e LayerNorm

        # Feed-Forward part
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=True, gnn_type='transformer', num_edge_features=7, transformer_heads=4):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.gnn_type = gnn_type.lower()

        if self.num_layer < 1: # Può essere anche 1 se JK="last" ha senso
            raise ValueError("Number of GNN layers must be at least 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(num_edge_features, emb_dim) # Encoder per edge features

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList() # Rinominato per chiarezza

        for layer in range(num_layer):
            if self.gnn_type == 'transformer':
                self.convs.append(TransformerConvBlock(emb_dim, num_heads=transformer_heads, dropout_ratio=drop_ratio))
            elif self.gnn_type == 'gine':
                # GINEConv richiede un MLP per il suo argomento 'nn'
                # Questo MLP trasforma le feature aggregate (incluse quelle del nodo stesso)
                gine_nn = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim * 2), # Espande la dimensionalità
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_dim * 2, emb_dim)  # Riporta alla dimensionalità originale
                )
                self.convs.append(GINEConv(nn=gine_nn, eps=0., train_eps=True, edge_dim=emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.layer_norms.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = x.long().clamp(min=0, max=self.node_encoder.num_embeddings - 1)
        x = x.to(self.node_encoder.weight.device)

        initial_node_emb = self.node_encoder(x)
        if initial_node_emb.ndim == 3 and initial_node_emb.size(1) == 1:
            initial_node_emb = initial_node_emb.squeeze(1)
        h_list = [initial_node_emb]

        edge_embedding = self.edge_encoder(edge_attr)

        for layer in range(self.num_layer):
            h_input_layer = h_list[layer]

            if self.gnn_type == 'transformer':
                h_output_conv = self.convs[layer](h_input_layer, edge_index, edge_embedding)
            elif self.gnn_type == 'gine':
                h_output_conv = self.convs[layer](h_input_layer, edge_index, edge_attr=edge_embedding)
            else:
                h_output_conv = h_input_layer

            h_after_norm = self.layer_norms[layer](h_output_conv)

            if layer < self.num_layer - 1:
                h_activated = F.leaky_relu(h_after_norm)
                h_after_dropout = F.dropout(h_activated, self.drop_ratio, training=self.training)
            else:
                h_after_dropout = F.dropout(h_after_norm, self.drop_ratio, training=self.training)

            current_h = h_after_dropout

            if self.residual:
                current_h = current_h + h_input_layer

            h_list.append(current_h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.stack(h_list).sum(dim=0)
        elif self.JK == "mean":
            node_representation = torch.stack(h_list).mean(dim=0)
        else:
            node_representation = h_list[-1]

        return node_representation
class GNN(torch.nn.Module):
    def __init__(self, num_class, num_layer=5, emb_dim=300,
                 gnn_type='transformer', residual=True, drop_ratio=0.5, JK="last", graph_pooling="attention",
                 num_edge_features=7, transformer_heads=4):
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        self.gnn_node = GNN_node(
            num_layer=num_layer,
            emb_dim=emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            num_edge_features=num_edge_features,
            transformer_heads=transformer_heads
        )

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            gate_nn_in_dim = emb_dim # Se JK non è concat
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(gate_nn_in_dim, 2 * gate_nn_in_dim),
                torch.nn.LayerNorm(2 * gate_nn_in_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * gate_nn_in_dim, 1)
            ))
        else:
            raise ValueError("Invalid graph pooling type.")

        linear_in_dim = 2 * self.emb_dim if graph_pooling == "set2set" else self.emb_dim

        self.graph_pred_linear1 = torch.nn.Linear(linear_in_dim, self.emb_dim)
        self.graph_pred_linear_bn = torch.nn.LayerNorm(self.emb_dim)
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        # Elaborazione dell'embedding del grafo
        graph_emb = F.leaky_relu(self.graph_pred_linear1(h_graph))
        graph_emb = self.graph_pred_linear_bn(graph_emb)
        graph_emb = F.dropout(graph_emb, p=self.drop_ratio, training=self.training)

        final_logits = self.graph_pred_linear(graph_emb)

        return final_logits, graph_emb, h_node



















class GINEncoderBlock(nn.Module): # Assicurati che questa sia definita
    def __init__(self, hidden_dim, dropout_rate, edge_dim_in_gine):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # train_eps era nel GINEConv ma non è un parametro di GINEncoderBlock
        self.conv = GINEConv(mlp, train_eps=True, edge_dim=edge_dim_in_gine)
        self.norm = BatchNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        x_residual = x
        x_conv = self.conv(x, edge_index, edge_attr=edge_attr)
        x_norm = self.norm(x_conv)
        x_sum = x_norm + x_residual # Connessione residuale
        x_relu = self.relu(x_sum)
        x_drop = self.dropout(x_relu)
        return x_drop

class GINENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 edge_dim=0, # train_eps non è un param di GINENet, ma di GINEConv
                 dropout_rate=0.5, graph_pooling="mean"): # graph_pooling invece di pool_type
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.graph_pooling_type = graph_pooling # Salva il tipo di pooling

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        if edge_dim is not None and edge_dim > 0: # Controllo aggiuntivo per edge_dim
            self.edge_encoder = nn.Linear(edge_dim, hidden_channels)
            self.gine_edge_dim_for_conv = hidden_channels
        else:
            self.edge_encoder = None
            self.gine_edge_dim_for_conv = None # GINEConv gestirà edge_attr=None o se edge_dim=0 o None

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEncoderBlock(hidden_channels, dropout_rate, self.gine_edge_dim_for_conv))

        if self.graph_pooling_type == "sum" or self.graph_pooling_type == "add":
            self.pool = global_add_pool
        elif self.graph_pooling_type == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling_type == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported graph pooling type: {self.graph_pooling_type}")

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None or x.numel() == 0:
            logging.warning(f"GINENet: Input node features 'x' is None or empty. original_idx: {data.original_idx if hasattr(data, 'original_idx') else 'N/A'}")
            num_graphs_in_batch = data.num_graphs if hasattr(data, 'num_graphs') else 1
            return torch.zeros((num_graphs_in_batch, self.output_mlp[-1].out_features), device=edge_index.device if edge_index is not None else 'cpu')

        x = self.node_encoder(x)
        x = F.relu(x) # Attivazione dopo l'encoder iniziale
        x = F.dropout(x, p=self.dropout_rate, training=self.training)


        current_edge_attr = edge_attr # Usa una variabile locale
        if self.edge_encoder and current_edge_attr is not None:
            if current_edge_attr.numel() > 0: # Solo se ci sono edge_attr
                current_edge_attr = self.edge_encoder(current_edge_attr)
            else: # edge_attr è un tensore vuoto, l'encoder non può processarlo, lascialo vuoto
                pass # o current_edge_attr = None se GINEConv lo preferisce così


        for conv_layer in self.convs:
            x = conv_layer(x, edge_index, current_edge_attr)

        current_batch_assignment = batch
        if current_batch_assignment is None:
            if x.numel() > 0 :
                current_batch_assignment = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                pass

        if x.numel() > 0 and current_batch_assignment is not None :
            x_pooled = self.pool(x, current_batch_assignment)
        elif x.numel() > 0 and current_batch_assignment is None:
            x_pooled = x.mean(dim=0, keepdim=True)
        else:
            num_graphs_in_batch = data.num_graphs if hasattr(data, 'num_graphs') else 1
            return torch.zeros((num_graphs_in_batch, self.output_mlp[-1].out_features), device=edge_index.device if edge_index is not None else 'cpu')

        out_logits = self.output_mlp(x_pooled)
        return out_logits