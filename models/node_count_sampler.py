import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeCountSampler(nn.Module):
    """
    条件下的节点数量分布建模：p(n | condition)
    作用：
        - 输入：condition [B, d]
        - 输出：PMF over possible node numbers [B, num_classes]
        - 可在采样时：从 PMF 中抽样节点数 n
    """

    def __init__(self, condition_dim, num_classes=4):
        super().__init__()
        self.condition_dim = condition_dim
        self.num_classes = num_classes  # 类别数，比如 18/36/60/90 对应 num_classes=4

        self.class_to_nodes = torch.tensor([18, 36, 60, 90])  # 可自定义

        self.net = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, condition, sample=True):
        """
        condition: [B, d]
        return:
            if sample=True → 采样出的节点数张量 [B]
            if sample=False → 概率分布 PMF [B, num_classes]
        """
        logits = self.net(condition)  # [B, num_classes]
        pmf = F.softmax(logits, dim=-1)  # [B, num_classes]

        if sample:
            sampled_idx = torch.multinomial(pmf, num_samples=1).squeeze(1)  # [B]
            node_counts = self.class_to_nodes[sampled_idx]
            return node_counts
        else:
            return pmf