import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Dropout, LayerNorm, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


#Used for datasets C and D
class TransBlock(torch.nn.Module):
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
        attended_x = self.transformer_conv(x, edge_index, edge_attr=edge_embedding)
        attended_x = self.dropout(attended_x) # Dropout dopo l'attention
        x = self.norm1(x + attended_x)  # Connessione residuale e LayerNorm

        # Feed-Forward
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

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(num_edge_features, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if self.gnn_type == 'transformer':
                self.convs.append(TransBlock(emb_dim, num_heads=transformer_heads, dropout_ratio=drop_ratio))
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

        linear_in_dim = self.emb_dim

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


#Used for datasets A and B
class EdgeGatedGIN(MessagePassing):
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

class EdgeTrans(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, heads=1, concat=True,
                 dropout_transformer=0.2, edge_input_dim=7, dropout_edge_mlp=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.heads = heads
        self.concat = concat

        # Encoder per gli attributi degli edge
        self.edge_encoder = Sequential(
            Linear(edge_input_dim, emb_dim),
            LeakyReLU(0.15),
            Linear(emb_dim, emb_dim),
            Dropout(dropout_edge_mlp) if dropout_edge_mlp > 0 else torch.nn.Identity()
        )

        # Calcolo dimensioni di output del TransformerConv
        if concat:
            out_channels_per_head = emb_dim // heads
            transformer_output_dim = heads * out_channels_per_head
        else:
            out_channels_per_head = emb_dim
            transformer_output_dim = emb_dim

        self.transformer_conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels_per_head,
            heads=heads,
            concat=concat,
            dropout=dropout_transformer,
            edge_dim=emb_dim,
            root_weight=True,
            beta=True
        )

        # Proiezione finale (identità se non necessaria)
        if transformer_output_dim != emb_dim:
            self.final_proj = Linear(transformer_output_dim, emb_dim)
        else:
            self.final_proj = torch.nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        x = self.transformer_conv(x, edge_index, edge_attr=edge_emb)
        x = self.final_proj(x)
        return x


class GINtrans(torch.nn.Module):
    def __init__(self, emb_dim, edge_input_dim, num_classes, dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim
        self.edge_input_dim = edge_input_dim
        self.input_proj = Linear(in_features=1, out_features=emb_dim)

        # First GIN (2 layer)
        self.gin_1 = torch.nn.ModuleList()
        for _ in range(2):
            self.gin_1.append(
                EdgeGatedGIN(
                    emb_dim=emb_dim,
                    edge_input_dim=self.edge_input_dim,
                    dropout_edge_mlp=dropout,
                    dropout_out=dropout
                )
            )
        self.bn1 = BatchNorm1d(num_features=emb_dim)

        # Transformer
        self.transformer1 = EdgeTrans(
            in_channels=emb_dim,
            emb_dim=emb_dim,
            heads=2,
            concat=True,
            dropout_transformer=dropout,
            edge_input_dim=edge_input_dim,
            dropout_edge_mlp=dropout
        )
        self.bn_transformer1 = BatchNorm1d(num_features=emb_dim)

        # Second GIN (2 layer)
        self.gin_2 = torch.nn.ModuleList([
            EdgeGatedGIN(
                emb_dim=emb_dim,
                edge_input_dim=edge_input_dim,
                dropout_edge_mlp=dropout,
                dropout_out=dropout
            )
            for _ in range(2)
        ])
        self.bn2 = BatchNorm1d(num_features=emb_dim)

        # Pooling
        self.pool = global_mean_pool

        # MLP
        self.MLPcla = Sequential(
            Linear(emb_dim, emb_dim // 2),
            LeakyReLU(negative_slope=0.15),
            Dropout(p=dropout),
            Linear(emb_dim // 2, num_classes)
        )

        # Dropout
        self.dropout_final_nodes = Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Init of x
        if x is None:
            device = edge_index.device if edge_index is not None else (batch.device if batch is not None else 'cpu')
            x = torch.zeros((data.num_nodes, 1), device=device)

        x = self.input_proj(x)

        # First GIN
        for conv in self.gin_1:
            x = conv(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout_final_nodes(x)

        # Transformer
        residual = x
        x = self.transformer1(x, edge_index, edge_attr)
        x = self.bn_transformer1(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout_final_nodes(x)
        x = x + residual

        # Second GIN
        for conv in self.gin_2:
            x = conv(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.15)
        x = self.dropout_final_nodes(x)

        # Pooling and MLP
        graph_embeddings = self.pool(x, batch)
        logits = self.MLPcla(graph_embeddings)

        return logits, graph_embeddings

