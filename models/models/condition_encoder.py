import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConditionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_type="transformer"):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self._reset_parameters()

        if self.encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif self.encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        if torch.isnan(x).any():
            print("[❌ NaN] in input x")
            print(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, C]

        x = self.input_proj(x)

        if torch.isnan(self.input_proj.weight).any() or torch.isnan(self.input_proj.bias).any():
            print("[❌ NaN] in input_proj weights or bias")
            raise ValueError("NaN in ConditionEncoder input_proj weights")

        if torch.isnan(x).any():
            print("[❌ NaN] After input_proj")
            raise ValueError("NaN detected after input_proj")

        x = self.encoder(x)
        x = x.mean(dim=1)
        return x
