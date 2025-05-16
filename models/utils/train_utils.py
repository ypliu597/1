import torch
import torch.nn as nn

def check_and_fix_nan_params(model):
    """
    遍历模型中的所有参数，检测是否存在 NaN 或 Inf，
    如果在 Linear 或 Embedding 层中发现异常，则重新初始化。
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                if torch.isnan(weight).any() or torch.isinf(weight).any():
                    print(f"[⚠️ NaN Fix] Detected NaN/Inf in {name}. Reinitializing weight...")
                    nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.data
                if torch.isnan(bias).any() or torch.isinf(bias).any():
                    print(f"[⚠️ NaN Fix] Detected NaN/Inf in {name} bias. Reinitializing bias...")
                    nn.init.zeros_(module.bias)