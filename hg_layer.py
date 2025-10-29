import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class HyperbolicGraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, manifold: PoincareBall, c=1.0, aggr='mean'):
        
        super(HyperbolicGraphLayer, self).__init__(aggr=aggr)
        self.manifold = manifold
        self.c = c

        self.linear = nn.Linear(in_channels, out_channels)
        self.bias = ManifoldParameter(torch.randn(out_channels), manifold=manifold)

    def forward(self, x, edge_index):
        x_tan = self.manifold.logmap0(x) # [num_nodes, dim]

        h_tan = self.linear(x_tan)
        h_tan = h_tan + self.manifold.transp0(self.bias, h_tan) 

        h_hyp = self.manifold.expmap0(h_tan) # [num_nodes, dim]

        return self.propagate(edge_index, x=h_hyp)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index):
        inputs_tan = self.manifold.logmap0(inputs)
        aggregated_tan = super().aggregate(inputs_tan, index) 
        aggregated_hyp = self.manifold.expmap0(aggregated_tan)
        return aggregated_hyp

    def update(self, aggr_out):
        aggr_out_tan = self.manifold.logmap0(aggr_out)
        aggr_out_tan = F.relu(aggr_out_tan)
        aggr_out_hyp = self.manifold.expmap0(aggr_out_tan)
        return aggr_out_hyp