import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fgh_model import FGH
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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
criterion = nn.BCELoss()

# 使用geoopt提供的优化器来优化流形参数（如bias）
# optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=0.001)

def train_one_setting(c_value: float, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, lr: float, epochs: int, batch_size: int):
    model = FGH(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, c=c_value).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    best_loss = float('inf')
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f'c={c_value} | Epoch {epoch} | Loss: {avg_loss:.6f}')
    return best_loss, history

def parse_c_list(c_list_str: str):
    items = [s.strip() for s in c_list_str.split(',') if s.strip()]
    return [float(x) for x in items]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--scan_c', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scan_c:
        c_values = parse_c_list(args.scan_c)
    else:
        c_values = [3.5]

    results = []
    for c_value in c_values:
        best_loss, history = train_one_setting(
            c_value=c_value,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.layers,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append({'c': c_value, 'best_train_loss': best_loss})

        # 保存单条曲线
        if plt is not None:
            try:
                plt.figure()
                plt.plot(history)
                plt.xlabel('Epoch')
                plt.ylabel('Train Loss')
                plt.title(f'Train Loss vs Epoch (c={c_value})')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'loss_curve_c_{c_value}.png'))
                plt.close()
            except Exception:
                pass

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, 'scan_c_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['c', 'best_train_loss'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # 汇总图
    if plt is not None and len(results) > 1:
        try:
            plt.figure()
            xs = [r['c'] for r in results]
            ys = [r['best_train_loss'] for r in results]
            plt.plot(xs, ys, marker='o')
            plt.xlabel('Curvature c')
            plt.ylabel('Best Train Loss')
            plt.title('Best Training Loss vs Curvature c')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'scan_c_summary.png'))
            plt.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()