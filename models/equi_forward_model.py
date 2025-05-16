# core/models/equi_forward_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_add


# --- InvariantGNNBlock (方案一的核心GNN层) ---
class InvariantGNNBlock(nn.Module):
    """
    GNN块，其消息和注意力机制主要基于SE(2)不变特征（如距离）和节点固有特征。
    """

    def __init__(self, hidden_dim, edge_feat_input_dim=1):  # edge_feat_input_dim 现在是1 (距离)
        super().__init__()
        self.hidden_dim = hidden_dim

        self.to_q_node = nn.Linear(hidden_dim, hidden_dim)
        self.to_k_node = nn.Linear(hidden_dim, hidden_dim)
        self.to_v_node = nn.Linear(hidden_dim, hidden_dim)

        # MLP处理传入的边特征 (现在只有距离)
        self.edge_feature_processor = nn.Sequential(
            nn.Linear(edge_feat_input_dim, hidden_dim),  # 输入维度为1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 输出的距离嵌入维度为 hidden_dim
        )

        # MLP 用于计算注意力分数
        # 输入: q_i (H), k_j (H), processed_distance_feature (H)
        self.attention_calculator_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # MLP 用于构建消息内容
        # 输入: v_j (H), processed_distance_feature (H)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos, edge_index, edge_attr, batch=None):  # pos 在此块中不直接使用，但保持签名
        num_nodes = h.size(0)
        if edge_index.numel() == 0: return h, pos  # 没有边则不更新

        row, col = edge_index

        q_i_all = self.to_q_node(h)
        k_j_all = self.to_k_node(h)
        v_j_all = self.to_v_node(h)

        # edge_attr 现在是 [E, 1] (距离)
        edge_dist_processed = self.edge_feature_processor(edge_attr)  # [E, H]

        q_i_for_attn = q_i_all[col]
        k_j_for_attn = k_j_all[row]

        attention_input = torch.cat([q_i_for_attn, k_j_for_attn, edge_dist_processed], dim=-1)
        edge_attention_scores_raw = self.attention_calculator_mlp(attention_input)
        edge_attention_weights = softmax(edge_attention_scores_raw, index=col, num_nodes=num_nodes)

        v_j_for_message = v_j_all[row]
        message_generation_input = torch.cat([v_j_for_message, edge_dist_processed], dim=-1)
        message_content = self.message_mlp(message_generation_input)

        weighted_messages = message_content * edge_attention_weights
        aggregated_messages_i = scatter_add(weighted_messages, col, dim=0, dim_size=num_nodes)

        h_updated = h + aggregated_messages_i

        return h_updated, pos
    # --- 结束 InvariantGNNBlock 定义 ---


class EquiForwardModel(nn.Module):
    def __init__(self,
                 node_feat_dim=5,  # <--- 不包含径向距离 (例如 2半径+3形状)
                 coord_dim=2,
                 graph_feat_dim=5,
                 output_dim=5,  # 预测5个参数
                 hidden_dim=256,
                 num_layers=5,
                 pooling_method='mean',
                 edge_input_dim=1  # <--- 修改: edge_attr 只包含距离，所以是1
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim  # 保存以便传递给 InvariantGNNBlock

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.graph_feature_encoder = nn.Linear(graph_feat_dim, hidden_dim)

        # 使用新的 InvariantGNNBlock
        self.blocks = nn.ModuleList([
            InvariantGNNBlock(hidden_dim, edge_feat_input_dim=self.edge_input_dim)
            for _ in range(num_layers)
        ])

        if pooling_method.lower() == 'mean':
            self.pooling = global_mean_pool
        elif pooling_method.lower() == 'add':
            self.pooling = global_add_pool
        else:
            raise ValueError(f"未知的池化方法: {pooling_method}. 请选择 'mean' 或 'add'.")

        # Multi-Head Prediction Architecture (保持不变)
        self.param_keys_for_heads = ["neff", "Aeff", "NL", "Disp", "GVD"]
        if output_dim != len(self.param_keys_for_heads):
            raise ValueError(f"output_dim ({output_dim}) 与头的数量不符。")
        self.prediction_heads = nn.ModuleDict()

        def default_head_structure(h_dim, out_size=1):
            return nn.Sequential(
                nn.Linear(h_dim, h_dim // 2), nn.SiLU(),
                nn.Linear(h_dim // 2, h_dim // 4), nn.SiLU(),
                nn.Linear(h_dim // 4, out_size))

        def complex_head_structure(h_dim, out_size=1):
            return nn.Sequential(
                nn.Linear(h_dim, h_dim), nn.SiLU(),
                nn.Linear(h_dim, h_dim // 2), nn.SiLU(),
                nn.Linear(h_dim // 2, h_dim // 4), nn.SiLU(),
                nn.Linear(h_dim // 4, out_size))

        for head_name in self.param_keys_for_heads:
            if head_name == "Aeff" or head_name == "NL":
                self.prediction_heads[head_name] = complex_head_structure(self.hidden_dim)
            else:
                self.prediction_heads[head_name] = default_head_structure(self.hidden_dim)

    def forward(self, data):
        x, pos, graph_features, batch = data.x, data.pos, data.graph_features, data.batch
        edge_index, edge_attr = data.edge_index, data.edge_attr  # edge_attr 现在是距离

        h = self.node_encoder(x)  # x 不含径向距离

        graph_emb = self.graph_feature_encoder(graph_features)
        if batch is None:  # (处理单图和空图的逻辑与您上一版一致)
            if x is not None and x.size(0) > 0:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            elif pos is not None and pos.size(0) > 0:
                batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
            else:
                if graph_emb.size(0) == 1:
                    h_graph = torch.zeros((graph_emb.size(0), self.hidden_dim), device=graph_emb.device)
                    predictions_list = [self.prediction_heads[name](h_graph) for name in self.param_keys_for_heads]
                    return torch.cat(predictions_list, dim=-1)
                else:
                    raise ValueError("Batch is None, x/pos empty, but multiple graph_features exist.")

        graph_emb_nodes = graph_emb[batch]
        h = h + graph_emb_nodes

        current_pos = pos
        for block in self.blocks:
            h_updated, _ = block(h, current_pos, edge_index, edge_attr, batch)
            h = h_updated

        if h.size(0) == 0:
            num_graphs_in_batch = graph_emb.size(0) if graph_emb is not None else (
                batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1)
            h_graph = torch.zeros((num_graphs_in_batch, self.hidden_dim),
                                  device=h.device if h.numel() > 0 else graph_emb.device)
        else:
            h_graph = self.pooling(h, batch)

        predictions_list = []
        for head_name in self.param_keys_for_heads:
            predictions_list.append(self.prediction_heads[head_name](h_graph))
        predictions = torch.cat(predictions_list, dim=-1)

        return predictions