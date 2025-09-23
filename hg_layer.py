import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class HyperbolicGraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, manifold: PoincareBall, c=1.0, aggr='mean'):
        """
        双曲图卷积层
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            manifold: 双曲流形（Poincaré球）
            c: 曲率
            aggr: 聚合方式
        """
        super(HyperbolicGraphLayer, self).__init__(aggr=aggr)
        self.manifold = manifold
        self.c = c

        # 在切空间（欧几里得空间）中定义权重和偏置
        self.linear = nn.Linear(in_channels, out_channels)
        self.bias = ManifoldParameter(torch.randn(out_channels), manifold=manifold)

    def forward(self, x, edge_index):
        # x: 双曲空间中的节点特征 [num_nodes, dim]
        # 1. 将双曲特征映射到切空间（对数映射）
        x_tan = self.manifold.logmap0(x) # [num_nodes, dim]

        # 2. 在切空间中进行特征变换 (公式 3)
        h_tan = self.linear(x_tan) # 欧几里得变换
        # 将偏置通过并行传输加到切空间向量上（公式5的近似简化）
        h_tan = h_tan + self.manifold.transp0(self.bias, h_tan) # 注意：这里是一个简化

        # 3. 将变换后的特征映射回双曲空间（指数映射）
        h_hyp = self.manifold.expmap0(h_tan) # [num_nodes, dim]

        # 4. 消息传递和聚合（将在message和aggregate方法中处理）
        # 注意：双曲空间的聚合需要在切空间进行
        return self.propagate(edge_index, x=h_hyp)

    def message(self, x_j):
        # x_j: 邻居节点的特征
        return x_j

    def aggregate(self, inputs, index):
        #  inputs: 来自message函数的输出
        #  index: 目标节点的索引，指示哪些消息应该聚合到哪个节点

        # 将双曲空间的特征投影到切空间进行聚合（公式8）
        inputs_tan = self.manifold.logmap0(inputs)
        aggregated_tan = super().aggregate(inputs_tan, index) # 在切空间进行均值聚合
        aggregated_hyp = self.manifold.expmap0(aggregated_tan)
        return aggregated_hyp

    def update(self, aggr_out):
        # aggr_out: 聚合后的消息
        # 这里可以添加非线性激活等操作（公式9）
        # 双曲非线性激活：映射到切空间->激活->映射回双曲空间
        aggr_out_tan = self.manifold.logmap0(aggr_out)
        aggr_out_tan = F.relu(aggr_out_tan)
        aggr_out_hyp = self.manifold.expmap0(aggr_out_tan)
        return aggr_out_hyp