import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from source.conv import GNN_node, GNN_node_Virtualnode, GatedGCNLayer


class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300,
                 gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data,train=False):
        h_node = self.gnn_node(batched_data,train)

        h_graph = self.pool(h_node, batched_data.batch)
        graph = F.relu(h_graph)
        graph=self.graph_pred_linear1(graph)
        graph = F.relu(graph)
        return self.graph_pred_linear(graph),graph,h_node


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    def __init__(self, dim_node_feat_raw, dim_out, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = FeatureEncoder(dim_node_feat_raw)
        dim_in = self.encoder.dim_in

        if self.cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, self.cfg.gnn.dim_inner, self.cfg.gnn.layers_pre_mp)
            dim_in = self.cfg.gnn.dim_inner

        assert self.cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(self.cfg.gnn.layer_type)
        layers = []
        for _ in range(self.cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=self.cfg.gnn.dropout,
                                     residual=self.cfg.gnn.residual,
                                     ffn=self.cfg.gnn.ffn))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[self.cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=self.cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        return GatedGCNLayer

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
