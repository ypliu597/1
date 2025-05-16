import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch.utils.data import DataLoader
from tqdm import tqdm

# === 模型模块 ===
class EquiAttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, h, x, batch):
        rel_x = x.unsqueeze(1) - x.unsqueeze(0)
        edge_feat = self.edge_mlp(rel_x)

        q = self.to_q(h).unsqueeze(1)
        k = self.to_k(h).unsqueeze(0)
        attn = (q * k).sum(-1) / h.size(-1)**0.5
        attn = torch.softmax(attn, dim=-1)

        v = self.to_v(h)
        agg_h = torch.matmul(attn, v)

        delta_x = self.coord_mlp(edge_feat) * rel_x
        delta_x = torch.sum(attn.unsqueeze(-1) * delta_x, dim=1)

        return h + agg_h, x + delta_x


class EquiStructureRegressor(nn.Module):
    def __init__(self, node_attr_dim, global_attr_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.coord_encoder = nn.Linear(2, hidden_dim)
        self.attr_encoder = nn.Linear(node_attr_dim, hidden_dim)
        self.global_encoder = nn.Linear(global_attr_dim, hidden_dim)

        self.blocks = nn.ModuleList([EquiAttentionBlock(hidden_dim) for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, pos, batch, graph_attr = data.hole_attr, data.hole_pos, data.batch, data.global_attr

        h = self.coord_encoder(pos) + self.attr_encoder(x) + self.global_encoder(graph_attr)[batch]

        for block in self.blocks:
            h, pos = block(h, pos, batch)

        graph_h = scatter_mean(h, batch, dim=0)
        out = self.readout(graph_h)
        return out


# === 训练器模块 ===
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()
        return total_loss / len(dataloader)


# === 主程序入口 ===
if __name__ == '__main__':
    from core.datasets.forward_dataset import FiberForwardDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FiberForwardDataset(root='dataset/')
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = EquiStructureRegressor(
        node_attr_dim=5,
        global_attr_dim=4,
        output_dim=5,
        hidden_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()

    trainer = Trainer(model, optimizer, criterion, device)
    for epoch in range(1, 51):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.eval_epoch(val_loader)
        print(f"[Epoch {epoch}] Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    torch.save(model.state_dict(), 'saved_models_equi/forward_best_model.pth')
