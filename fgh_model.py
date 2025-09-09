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

        # 将初始的欧几里得特征映射到双曲空间（公式2）
        self.to_hyperbolic = nn.Linear(input_dim, hidden_dim)

        # 堆叠多个双曲图卷积层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HyperbolicGraphLayer(hidden_dim, hidden_dim, self.manifold, c=c))

        # 最终的分类层（在切空间进行）
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # x: 初始节点特征 [num_nodes, input_dim] (来自GraphCodeBERT的欧几里得特征)
        # edge_index: 图的边索引 [2, num_edges]
        # batch: 指示每个节点属于哪个图的索引 [num_nodes]

        # 1. 初始映射到双曲空间
        x_tan = self.to_hyperbolic(x) # 先在欧几里得空间做线性变换
        x_hyp = self.manifold.expmap0(x_tan) # 映射到双曲空间

        # 2. 通过双曲图卷积层
        for conv in self.convs:
            x_hyp = conv(x_hyp, edge_index)

        # 3. 图级读出（READOUT）（公式10）
        # 将所有节点映射到切空间，求和，再映射回双曲空间
        x_tan_readout = self.manifold.logmap0(x_hyp)
        graph_embedding_tan = global_mean_pool(x_tan_readout, batch) # 在切空间进行全局平均池化
        graph_embedding_hyp = self.manifold.expmap0(graph_embedding_tan)

        # 4. 分类（在切空间进行）
        graph_embedding_final_tan = self.manifold.logmap0(graph_embedding_hyp)
        out = self.classifier(graph_embedding_final_tan)
        return torch.sigmoid(out) # 二分类输出