import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from core.models.condition_encoder import ConditionEncoder


class EquiAttentionBlock(nn.Module):
    """SE(2)-等变注意力模块（2D版 MolCRAFT）"""

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
        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim + 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        self.coord_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, h, x, batch):
        # h: [N, H], x: [N, 2]
        rel_x = x.unsqueeze(1) - x.unsqueeze(0)     # [N, N, 2]
        rel_h = h.unsqueeze(1) - h.unsqueeze(0)     # [N, N, H]

        edge_input = rel_x                          # 相对坐标作为边输入
        # edge_input = torch.cat([rel_x, rel_h], dim=-1)
        edge_feat = self.edge_mlp(edge_input)       # [N, N, H]

        q = self.to_q(h).unsqueeze(1)               # [N, 1, H]
        k = self.to_k(h).unsqueeze(0)               # [1, N, H]
        attn_score = (q * k).sum(-1) / h.size(-1)**0.5   # [N, N]
        attn_weight = torch.softmax(attn_score, dim=-1)  # [N, N]

        v = self.to_v(h)                            # [N, H]
        agg_h = torch.matmul(attn_weight, v)        # [N, H]

        delta_x = self.coord_mlp(edge_feat) * rel_x  # [N, N, 2]
        delta_x = torch.sum(attn_weight.unsqueeze(-1) * delta_x, dim=1)

        return h + agg_h, x + delta_x


class EquiStructureDecoder(nn.Module):
    """
    SE(2)-等变结构生成器：
        - 输入：θ + time + condition
        - 输出：坐标、属性、global_attr
    """

    def __init__(self,
                 node_attr_dim,
                 global_dim,
                 input_condition_dim,
                 hidden_dim=128,
                 encoder_type='transformer'
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 条件编码器（支持四类输入）
        self.condition_encoder = ConditionEncoder(
            input_dim=input_condition_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type
        )

        # 初始投影
        self.coord_encoder = nn.Linear(2, hidden_dim)
        self.attr_encoder = nn.Linear(node_attr_dim, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)

        # 多层等变 block
        self.blocks = nn.ModuleList([
            EquiAttentionBlock(hidden_dim) for _ in range(3)
        ])

        # 输出预测头
        self.coord_out = nn.Linear(hidden_dim, 2)
        self.attr_out = nn.Linear(hidden_dim, node_attr_dim)
        self.global_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim)
        )

    def forward(self, theta_coord, theta_attr, theta_global, t, condition, batch):
        if torch.isnan(condition).any():
            print("[❌ NaN] condition (cond_encoded) contains NaN")
            raise ValueError("NaN in condition")

        if torch.isnan(t).any():
            print("[❌ NaN] time t contains NaN")
            raise ValueError("NaN in time input")

        t_encoded = self.time_proj(t)
        if torch.isnan(t_encoded).any():
            print("[❌ NaN] After time_proj(t)")
            print(t_encoded)
            raise ValueError("NaN detected in time projection")

        # ==== 信息打印 ====
        # print("[DEBUG] === Init ===")
        # print("theta_coord:", theta_coord.min().item(), theta_coord.max().item())
        # print("theta_attr:", theta_attr.min().item(), theta_attr.max().item())
        # print("cond_encoded:", condition.min().item(), condition.max().item())
        # print("t_encoded:", t_encoded.min().item(), t_encoded.max().item())

        # 初始融合
        h = self.coord_encoder(theta_coord) + self.attr_encoder(theta_attr) + condition + t_encoded
        x = theta_coord

        # print("h init:", h.min().item(), h.max().item())
        # print("x init:", x.min().item(), x.max().item())

        for i, block in enumerate(self.blocks):
            h, x = block(h, x, batch)
            # print(f"[DEBUG] --- After Block {i} ---")
            # print("h min=", h.min().item(), "max=", h.max().item(), "mean=", h.mean().item())
            # print("x min=", x.min().item(), "max=", x.max().item(), "mean=", x.mean().item())

            if torch.isnan(h).any() or torch.isnan(x).any():
                print(f"[❌ NaN] Detected in block {i}")
                break

        coord_pred = self.coord_out(h)
        attr_pred = self.attr_out(h)
        h_graph = scatter_mean(h, batch, dim=0)
        global_pred = self.global_out(h_graph)

        # === 最后再做一轮检查 ===
        if torch.isnan(coord_pred).any():
            print("[❌ NaN] in coord_pred")
        if torch.isnan(attr_pred).any():
            print("[❌ NaN] in attr_pred")
        if torch.isnan(global_pred).any():
            print("[❌ NaN] in global_pred")

        # print("[DEBUG] === Output ===")
        # print("coord_pred:", coord_pred.min().item(), coord_pred.max().item())
        # print("attr_pred:", attr_pred.min().item(), attr_pred.max().item())
        # print("global_pred:", global_pred.min().item(), global_pred.max().item())

        return coord_pred, attr_pred, global_pred

