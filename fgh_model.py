import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from geoopt.manifolds import PoincareBall
from hg_layer import HyperbolicGraphLayer

class FGH(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, c=1.0):
        super(FGH, self).__init__()
        self.manifold = PoincareBall(c=c)
        self.num_layers = num_layers
        self.c = c

        self.to_hyperbolic = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HyperbolicGraphLayer(hidden_dim, hidden_dim, self.manifold, c=c))

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        
        x_tan = self.to_hyperbolic(x) 
        x_hyp = self.manifold.expmap0(x_tan)

        for conv in self.convs:
            x_hyp = conv(x_hyp, edge_index)

        x_tan_readout = self.manifold.logmap0(x_hyp)
        graph_embedding_tan = global_mean_pool(x_tan_readout, batch)
        graph_embedding_hyp = self.manifold.expmap0(graph_embedding_tan)

        graph_embedding_final_tan = self.manifold.logmap0(graph_embedding_hyp)
        out = self.classifier(graph_embedding_final_tan)
        return torch.sigmoid(out) 