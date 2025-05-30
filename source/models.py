# In source/models.py (questa è la versione che si aspetta 'in_channels', etc.)
import torch
from torch.nn import Sequential, Linear, Dropout, LayerNorm, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import MessagePassing, TransformerConv


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

















import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool, BatchNorm
from torch_geometric.utils import to_dense_batch # Utile per il Transformer per-grafo
import logging

class GINEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, no_residual, edge_dim_in_gine):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.conv = GINEConv(mlp, train_eps=True, edge_dim=edge_dim_in_gine)
        self.norm = BatchNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = not no_residual

    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        out = self.dropout(out)
        if self.residual:
            logging.debug("GINEncoderBlock: Residual connection applied.")
            out = out + residual
        out = self.relu(out)

        return out

class GINENetWithTransformer(nn.Module): # Rinominata per chiarezza
    def __init__(self, in_channels, hidden_channels, out_channels, num_gin_layers, no_residual=False, jk="last",
                 edge_dim=0, dropout_rate=0.5, graph_pooling="mean",
                 # Parametri per il Transformer
                 num_transformer_layers=0, # Metti > 0 per attivare il Transformer
                 transformer_nhead=4,
                 transformer_dim_feedforward=None # Solitamente 2*hidden_channels o 4*hidden_channels
                 ):
        super().__init__()
        self.num_gin_layers = num_gin_layers
        self.dropout_rate = dropout_rate
        self.graph_pooling_type = graph_pooling
        self.hidden_channels = hidden_channels
        self.jk = jk

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        if edge_dim is not None and edge_dim > 0:
            self.edge_encoder = nn.Linear(edge_dim, hidden_channels)
            self.gine_edge_dim_for_conv = hidden_channels
        else:
            self.edge_encoder = None
            self.gine_edge_dim_for_conv = None

        self.gin_convs = nn.ModuleList() # Rinominato per chiarezza
        for _ in range(num_gin_layers):
            self.gin_convs.append(GINEncoderBlock(hidden_channels, dropout_rate, no_residual, self.gine_edge_dim_for_conv))

        # --- Transformer Encoder ---
        self.num_transformer_layers = num_transformer_layers
        if self.num_transformer_layers > 0:
            if transformer_dim_feedforward is None:
                transformer_dim_feedforward = hidden_channels * 2 # Default

            # Layer singolo del Transformer Encoder
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,      # Dimensione dell'input (hidden_channels)
                nhead=transformer_nhead,      # Numero di attention heads
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout_rate,
                activation='relu',
                batch_first=True              # Input: (N, S, E) -> (batch_size, seq_len, features)
                # PyG batching è diverso, lo gestiremo
            )
            # Stack di layer Transformer Encoder
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer,
                num_layers=num_transformer_layers
            )
            # LayerNorm dopo il Transformer (opzionale, ma comune)
            self.transformer_norm = nn.LayerNorm(hidden_channels)
            logging.info(f"GINENet: Transformer Encoder attivato con {num_transformer_layers} layer(s).")
        else:
            self.transformer_encoder = None
            self.transformer_norm = None
            logging.info("GINENet: Transformer Encoder non attivo.")

        # --- Graph Pooling ---
        if self.graph_pooling_type == "sum" or self.graph_pooling_type == "add":
            self.pool = global_add_pool
        elif self.graph_pooling_type == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling_type == "max":
            self.pool = global_max_pool
        elif self.graph_pooling_type == "attention":
            if self.graph_pooling_type == "attention":
                gate_nn_in_dim = hidden_channels  # default

                if jk == "concat":
                    gate_nn_in_dim = hidden_channels * num_gin_layers

                self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(gate_nn_in_dim, 2 * gate_nn_in_dim),
                    torch.nn.LayerNorm(2 * gate_nn_in_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(2 * gate_nn_in_dim, 1)
                ))
        else:
            raise ValueError("Invalid graph pooling type.")

        out_mlp_in_dim = hidden_channels * num_gin_layers if jk == "concat" else hidden_channels
        self.output_mlp = nn.Sequential(
            nn.Linear(out_mlp_in_dim, out_mlp_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_mlp_in_dim // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        output_device = x.device if x is not None else (edge_index.device if edge_index is not None else 'cpu')


        if x is None or x.numel() == 0:
            logging.warning(f"GINENet: Input 'x' is None or empty. original_idx: {data.original_idx if hasattr(data, 'original_idx') else 'N/A'}")
            num_graphs_in_batch = data.num_graphs if hasattr(data, 'num_graphs') else 1
            dummy_logits = torch.zeros((num_graphs_in_batch, self.output_mlp[-1].out_features), device=output_device)
            dummy_embeddings = torch.zeros((num_graphs_in_batch, self.hidden_channels), device=output_device)
            return dummy_logits, dummy_embeddings

        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.edge_encoder and edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_encoder(edge_attr)

        layer_outputs = []
        for conv_layer in self.gin_convs:
            x = conv_layer(x, edge_index, edge_attr)
            layer_outputs.append(x)

            # --- Apply JK ---
            if self.jk == "last":
                x = layer_outputs[-1]
            elif self.jk == "concat":
                x = torch.cat(layer_outputs, dim=-1)
            elif self.jk == "sum":
                x = torch.stack(layer_outputs, dim=0).sum(dim=0)
            elif self.jk == "max":
                x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
            else:
                raise ValueError(f"Invalid JK mode: {self.jk}")

        # --- Transformer Encoder ---
        if self.transformer_encoder:
            if batch is not None:
                x_dense, node_mask = to_dense_batch(x, batch, fill_value=0)
                if x_dense.numel() > 0:
                    src_key_padding_mask = ~node_mask
                    x = self.transformer_encoder(x_dense, src_key_padding_mask=src_key_padding_mask)
                    if self.transformer_norm:
                        x = self.transformer_norm(x)
                    x = x[node_mask]
            else:
                if x.numel() > 0:
                    x = self.transformer_encoder(x.unsqueeze(0))
                    if self.transformer_norm:
                        x = self.transformer_norm(x)
                    x = x.squeeze(0)

        # --- Pooling ---
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if x.numel() > 0:
            x_pooled = self.pool(x, batch)
        else:
            logging.error("GINENet: x_pooled non calcolabile.")
            num_graphs_in_batch = data.num_graphs if hasattr(data, 'num_graphs') else 1
            dummy_logits = torch.zeros((num_graphs_in_batch, self.output_mlp[-1].out_features), device=output_device)
            dummy_embeddings = torch.zeros((num_graphs_in_batch, self.hidden_channels), device=output_device)
            return dummy_logits, dummy_embeddings

        out_logits = self.output_mlp(x_pooled)
        return out_logits, x_pooled













class GINConvE(MessagePassing):
    def __init__(self, emb_dim, edge_input_dim=7, dropout_edge_mlp=0.2, dropout_out=0.2):
        super().__init__(aggr="add")

        self.mlp = Sequential(
            Linear(emb_dim, 2 * emb_dim),
            LayerNorm(2 * emb_dim),
            LeakyReLU(0.15),
            Dropout(dropout_out),
            Linear(2 * emb_dim, emb_dim),
            LayerNorm(emb_dim)
        )

        self.edge_encoder = Sequential(
            Linear(edge_input_dim, emb_dim),
            LayerNorm(emb_dim),
            LeakyReLU(0.15),
            Dropout(dropout_edge_mlp),
            Linear(emb_dim, emb_dim),
            LeakyReLU(0.15),
            Linear(emb_dim, emb_dim)
        )

        self.eps = torch.nn.Parameter(torch.Tensor([1e-6])) # Valore piccolo non nullo

        self.dropout = Dropout(dropout_out)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        out = self.mlp((1 + self.eps) * x + out) # Somma pesata del nodo centrale e messaggi aggregati + MLP
        # Residual connection + dropout
        return x + self.dropout(out) # Residuale sull'input originale x

    def message(self, x_j, edge_attr):
        # Gated fusion tra x_j ed edge_attr
        gate = torch.sigmoid(edge_attr)
        return gate * x_j + (1 - gate) * edge_attr

    def update(self, aggr_out):
        return aggr_out

class TransformerConvE(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, heads=1, concat=True,
                 dropout_transformer=0.2, edge_input_dim=7, dropout_edge_mlp=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim # Questo è l'out_channels per il TransformerConv o per il final_proj
        self.heads = heads
        self.concat = concat

        # L'output di edge_encoder deve essere edge_dim per TransformerConv
        # Se edge_dim = emb_dim, allora l'output di edge_encoder deve essere emb_dim
        self.edge_encoder_output_dim = emb_dim # Assumiamo che edge_dim per TransformerConv sia emb_dim

        edge_encoder_layers = [
            Linear(edge_input_dim, self.edge_encoder_output_dim),
            LeakyReLU(0.15),
            Linear(self.edge_encoder_output_dim, self.edge_encoder_output_dim)
        ]
        if dropout_edge_mlp > 0:
            edge_encoder_layers.append(torch.nn.Dropout(dropout_edge_mlp))
        self.edge_encoder = Sequential(*edge_encoder_layers)


        if concat:
            out_channels_per_head = emb_dim // heads # out_channels per TransformerConv
            transformer_conv_output_dim = heads * out_channels_per_head
        else:
            out_channels_per_head = emb_dim
            transformer_conv_output_dim = emb_dim

        self.transformer_conv = TransformerConv(
            in_channels=in_channels, # Dimensione dell'input dei nodi
            out_channels=out_channels_per_head, # Dimensione dell'output per testa
            heads=heads,
            concat=concat,
            dropout=dropout_transformer,
            edge_dim=self.edge_encoder_output_dim, # Dimensione dell'output dell'edge_encoder
            root_weight=True,
            beta=True # Abilita il gating beta per le feature degli archi
        )

        # Proiezione finale se l'output concatenato del transformer non corrisponde a emb_dim desiderato
        if concat and transformer_conv_output_dim != emb_dim:
            self.final_proj = Linear(transformer_conv_output_dim, emb_dim)
        elif not concat and out_channels_per_head != emb_dim: # Anche se non concat, l'output potrebbe essere diverso
            self.final_proj = Linear(out_channels_per_head, emb_dim)
        else:
            self.final_proj = torch.nn.Identity()


    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out_transformer = self.transformer_conv(x, edge_index, edge_attr=edge_embedding)
        out_projected = self.final_proj(out_transformer)
        return out_projected


class newModel(torch.nn.Module):
    def __init__(self, emb_dim, edge_input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_input_dim = edge_input_dim

        self.input_proj = Linear(1, emb_dim) # Assume feature nodi iniziali di dim 1

        self.gin_block1 = torch.nn.ModuleList([
            GINConvE(emb_dim, self.edge_input_dim, dropout_edge_mlp=dropout, dropout_out=dropout)
            for _ in range(2)
        ])
        self.bn1 = BatchNorm1d(emb_dim)

        self.transformer1 = TransformerConvE(emb_dim, emb_dim, heads=2, concat=True,
                                             dropout_transformer=dropout, edge_input_dim=edge_input_dim, dropout_edge_mlp=dropout)
        self.bn_transformer1 = BatchNorm1d(emb_dim)

        self.gin_block2 = torch.nn.ModuleList([
            GINConvE(emb_dim, edge_input_dim, dropout_edge_mlp=dropout, dropout_out=dropout)
            for _ in range(2)
        ])
        self.bn2 = BatchNorm1d(emb_dim)

        self.pool = global_mean_pool
        self.classifier = Sequential(
            Linear(emb_dim, emb_dim // 2),
            LeakyReLU(0.15),
            Dropout(dropout),
            Linear(emb_dim // 2, num_classes)
        )
        self.dropout_final_nodes = Dropout(dropout) # Rinominato per chiarezza rispetto a self.dropout in GINConvE

    # MODIFICA QUI: Cambia la firma di forward per accettare 'data' e restituisci (logits, embeddings)
    def forward(self, data): # Accetta l'oggetto Data o Batch di PyG
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None:
            # Usa data.num_nodes per ottenere il numero di nodi nel batch corrente
            num_nodes = data.num_nodes
            x = torch.zeros((num_nodes, 1), device=edge_index.device if edge_index is not None else (batch.device if batch is not None else 'cpu'))

        x = self.input_proj(x)

        for conv_layer in self.gin_block1:
            x = conv_layer(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout_final_nodes(x) # Usa il dropout del modello principale

        residual_pre_transformer = x
        x = self.transformer1(x, edge_index, edge_attr)
        x = self.bn_transformer1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout_final_nodes(x) # Usa il dropout del modello principale
        x = x + residual_pre_transformer

        for conv_layer in self.gin_block2:
            x = conv_layer(x, edge_index, edge_attr)
        x = self.bn2(x)
        x_node_embeddings_final = F.leaky_relu(x, negative_slope=0.15) # Applica ReLU prima del dropout finale
        x_node_embeddings_final = self.dropout_final_nodes(x_node_embeddings_final) # Applica dropout ai final node embeddings

        # --- MODIFICA QUI: Cattura gli embedding del grafo ---
        graph_embeddings = self.pool(x_node_embeddings_final, batch)
        logits = self.classifier(graph_embeddings)

        return logits, graph_embeddings # Restituisci logits e graph_embeddings