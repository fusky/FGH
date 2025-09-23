import os
import argparse
import csv
import random
from typing import Tuple
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
try:
    from sklearn import metrics as skm
except Exception:
    skm = None

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

def _split_dataset(ds, val_ratio=0.2, seed=42):
    rnd = random.Random(seed)
    indices = list(range(len(ds)))
    rnd.shuffle(indices)
    val_size = max(1, int(len(ds) * val_ratio))
    val_idx = set(indices[:val_size])
    train_set, val_set = [], []
    for i, d in enumerate(ds):
        (val_set if i in val_idx else train_set).append(d)
    return train_set, val_set

def _compute_metrics(logits, labels) -> Tuple[float, float, float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits).detach().cpu().view(-1)
        preds = (probs >= 0.5).long()
        labels = labels.detach().cpu().view(-1).long()
        correct = (preds == labels).sum().item()
        acc = correct / max(1, labels.numel())
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, (precision + recall))
        # ROC AUC（若可用）
        try:
            if skm is not None and labels.unique().numel() > 1:
                auc = float(skm.roc_auc_score(labels.numpy(), probs.numpy()))
            else:
                auc = 0.0
        except Exception:
            auc = 0.0
    return acc, f1, auc

# 使用geoopt提供的优化器来优化流形参数（如bias）
# optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=0.001)

def train_one_setting(c_value: float, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, lr: float, epochs: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    train_set, val_set = _split_dataset(dataset, val_ratio=0.2, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = FGH(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, c=c_value).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y.float().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(1, len(train_loader))

        # 验证
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                logits = model(data.x, data.edge_index, data.batch).view(-1)
                loss_v = criterion(logits, data.y.float().view(-1))
                val_loss += loss_v.item()
                all_logits.append(logits)
                all_labels.append(data.y)
        avg_val_loss = val_loss / max(1, len(val_loader))
        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            val_acc, val_f1, val_auc = _compute_metrics(logits_cat, labels_cat)
        else:
            val_acc, val_f1, val_auc = 0.0, 0.0, 0.0

        history.append(avg_train_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
        print(f'c={c_value} | seed={seed} | Epoch {epoch} | train_loss: {avg_train_loss:.6f} | val_loss: {avg_val_loss:.6f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_auc: {val_auc:.4f}')

    return best_val_loss, best_val_acc, best_val_f1, best_val_auc, history

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
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=2024)
    parser.add_argument('--method_name', type=str, default='FGH')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scan_c:
        c_values = parse_c_list(args.scan_c)
    else:
        c_values = [3.5]

    results = []
    for c_value in c_values:
        for run in range(args.runs):
            seed = args.seed_base + run
            best_val_loss, best_val_acc, best_val_f1, best_val_auc, history = train_one_setting(
                c_value=c_value,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=1,
                num_layers=args.layers,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
            )
            results.append({
                'method': args.method_name,
                'c': c_value,
                'run': run,
                'seed': seed,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'best_val_auc': best_val_auc,
            })

            # 保存单条曲线
            if plt is not None:
                try:
                    plt.figure()
                    plt.plot(history)
                    plt.xlabel('Epoch')
                    plt.ylabel('Train Loss')
                    plt.title(f'Train Loss vs Epoch (c={c_value}, run={run})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f'loss_curve_c_{c_value}_run_{run}.png'))
                    plt.close()
                except Exception:
                    pass

    # 保存 CSV（包含多次重复与验证集指标）
    csv_path = os.path.join(args.output_dir, 'scan_c_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'c', 'run', 'seed', 'best_val_loss', 'best_val_acc', 'best_val_f1', 'best_val_auc'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # 按 c 汇总图（以 best_val_loss 为纵轴）
    if plt is not None and len(results) > 0:
        try:
            # 取每个 c 的均值
            from collections import defaultdict
            acc_by_c = defaultdict(list)
            loss_by_c = defaultdict(list)
            for r in results:
                acc_by_c[r['c']].append(r['best_val_acc'])
                loss_by_c[r['c']].append(r['best_val_loss'])
            cs = sorted(loss_by_c.keys())
            mean_losses = [sum(loss_by_c[c])/len(loss_by_c[c]) for c in cs]
            plt.figure()
            plt.plot(cs, mean_losses, marker='o')
            plt.xlabel('Curvature c')
            plt.ylabel('Mean Best Val Loss')
            plt.title('Mean Best Val Loss vs c')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'scan_c_summary.png'))
            plt.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()