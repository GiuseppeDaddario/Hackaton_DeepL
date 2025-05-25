
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import warnings

#
# def dirichlet_energy(x, edge_index,split):

#     # adj = to_dense_adj(edge_index)[0]  # Convert edge_index to adjacency matrix
#     _,count = torch.unique(edge_index[0], return_counts=True)
#     # count = {key.item(): value.item() for key, value in zip(_, count)}
#     di = count[edge_index[0]]
#     # di =  torch.tensor([count[key.item()] for key in edge_index[0]],device = x.device)
#     dj = count[edge_index[1]]
#     # dj = torch.tensor([count[key.item()] for key in edge_index[1]],device = x.device)
#     diff = x[edge_index[0]]/torch.sqrt(di.view(-1,1)) - x[edge_index[1]]/torch.sqrt(dj.view(-1,1))
#     edge_sum = torch.sum(diff.pow(2),dim=1)
#     split = split.tolist()
#     split_pionts = [split[i+1] - split[i] for i in range(len(split) - 1)]
#     graph_sum = [(1/2)*torch.sum(x) for x in torch.split(edge_sum, split_pionts)]
#     return graph_sum


#this One is added to perform general calculation for every dataset otherwise the
#top one is working fine

def dirichlet_energy(x, edge_index,split):
    values,counts = edge_index[0].unique(return_counts=True)
    count = dict(zip(values.tolist(), counts.tolist()))
    di =  torch.tensor([count.get(key.item(),None) for key in edge_index[0]],device = x.device)
    dj = torch.tensor([count.get(key.item(),None) for key in edge_index[1]],device = x.device)
    diff = x[edge_index[0]]/torch.sqrt(di.view(-1,1)) - x[edge_index[1]]/torch.sqrt(dj.view(-1,1))
    edge_sum = torch.sum(diff.pow(2),dim=1)
    split = split.tolist()
    split_pionts = [split[i+1] - split[i] for i in range(len(split) - 1)]
    graph_sum = [(1/2)*torch.sum(x) for x in torch.split(edge_sum, split_pionts)]
    return graph_sum

