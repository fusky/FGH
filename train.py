import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from fgh_model import FGH

# 假设我们已经有一个数据集 `dataset`，是Data对象的列表
# 每个Data对象有: x, edge_index, y (图标签)
# 这里提供一个可运行的合成数据集示例
def build_synthetic_dataset(num_graphs=100, num_nodes=10, input_dim=768):
    dataset = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes, input_dim)
        # 构造一条简单链式边
        src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        y = torch.randint(0, 2, (1,), dtype=torch.long)  # 图级标签，二分类
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset

dataset = build_synthetic_dataset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FGH(input_dim=768, hidden_dim=256, output_dim=1, num_layers=3, c=3.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 使用geoopt提供的优化器来优化流形参数（如bias）
# optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=0.001)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model.train()
for epoch in range(500):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')