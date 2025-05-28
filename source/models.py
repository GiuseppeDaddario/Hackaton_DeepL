import torch_geometric.graphgym.models.head  # noqa, register module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import TransformerConv, \
    global_add_pool, GlobalAttention, global_max_pool, \
    global_mean_pool, Linear


from source.conv import GatedGCNLayer


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
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.layer_norms.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = x.long().clamp(min=0, max=self.node_encoder.num_embeddings - 1)
        x = x.to(self.node_encoder.weight.device)
        h_list = [self.node_encoder(x)]

        edge_embedding = self.edge_encoder(edge_attr)

        for layer in range(self.num_layer):
            h_prev_layer = h_list[layer]

            if self.gnn_type == 'transformer':
                h = self.convs[layer](h_prev_layer, edge_index, edge_embedding)
            else:
                h = h_prev_layer

            h = self.layer_norms[layer](h) # LayerNorm dopo il blocco convoluzionale

            if layer < self.num_layer - 1: # Applica LeakyReLU e Dropout tranne all'ultimo layer
                h = F.leaky_relu(h) # Spostato LeakyReLU dopo LayerNorm
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else: # Dropout per l'ultimo layer (senza LeakyReLU)
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual and layer < self.num_layer : # La connessione residuale si somma all'input del layer corrente
                h = h + h_list[layer] # Aggiungi l'output del layer precedente (prima di passare per la conv corrente)

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.stack(h_list).sum(dim=0)
        elif self.JK == "mean":
            node_representation = torch.stack(h_list).mean(dim=0)
        else:
            node_representation = h_list[-1] # Default a last

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

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    def __init__(self, dim_node_feat_raw, dim_out, cfg):
        super().__init__()
        self.cfg = cfg

        current_node_dim = dim_node_feat_raw

        if self.cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                current_node_dim, self.cfg.gnn.dim_inner, self.cfg.gnn.layers_pre_mp)
            current_node_dim = self.cfg.gnn.dim_inner
            self.input_proj = nn.Identity()
        else:
            self.pre_mp = nn.Identity()
            if current_node_dim != self.cfg.gnn.dim_inner:
                self.input_proj = Linear(current_node_dim, self.cfg.gnn.dim_inner)
                current_node_dim = self.cfg.gnn.dim_inner
            else:
                self.input_proj = nn.Identity()

        assert self.cfg.gnn.dim_inner == current_node_dim, \
            "The inner and hidden dims must match after node feature projection."

        dim_edge_feat_raw_hardcoded = 7
        target_edge_dim = self.cfg.gnn.dim_inner

        if dim_edge_feat_raw_hardcoded > 0:
            if dim_edge_feat_raw_hardcoded != target_edge_dim:
                self.edge_input_proj = Linear(dim_edge_feat_raw_hardcoded, target_edge_dim)
            else:
                self.edge_input_proj = nn.Identity()
        else:
            self.edge_input_proj = nn.Identity()

        conv_class = self.build_conv_model(self.cfg.gnn.layer_type)

        gnn_layers_list = []
        dim_for_gnn_conv = self.cfg.gnn.dim_inner

        for _ in range(self.cfg.gnn.layers_mp):
            gnn_layers_list.append(conv_class(dim_for_gnn_conv,
                                              dim_for_gnn_conv,
                                              dropout=self.cfg.gnn.dropout,
                                              residual=self.cfg.gnn.residual,
                                              ffn=self.cfg.gnn.ffn,
                                              act=self.cfg.gnn.act,
                                              batchnorm=self.cfg.gnn.batchnorm
                                              ))
        self.gnn_layers = nn.Sequential(*gnn_layers_list)

        GNNHead_class = register.head_dict[self.cfg.gnn.head]
        self.post_mp = GNNHead_class(dim_in=self.cfg.gnn.dim_inner, dim_out=dim_out)
    def build_conv_model(self, model_type):
        return GatedGCNLayer

    def forward(self, batch):
        batch = self.pre_mp(batch)

        if not isinstance(self.input_proj, nn.Identity):
            batch.x = self.input_proj(batch.x)

        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            if not isinstance(self.edge_input_proj, nn.Identity):
                batch.edge_attr = self.edge_input_proj(batch.edge_attr)

        batch = self.gnn_layers(batch)
        out_head = self.post_mp(batch)

        if isinstance(out_head, tuple):
            logits = out_head[0]
            graph_emb = out_head[1] if len(out_head) > 1 else None
        else:
            logits = out_head
            graph_emb = None


        return logits, graph_emb, None