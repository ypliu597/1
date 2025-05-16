import torch
import numpy as np
from core.evaluation.forward_model_interface import model_forward_func
from core.evaluation.metrics import (
    coordinate_distribution_score,
    radius_distribution_score,
    coverage_score,
    matching_score
)
from core.models.bfn_fiber import BFN4FiberDesign
from core.models.gcn_forward import GCNModel

def evaluate_structure(coords, attr, global_attr, target_param, model_forward_func, forward_model, gt_coords=None, gt_attr=None):
    results = {}
    results["mean_radius"] = attr[:, 0].mean().item()
    results["std_radius"] = attr[:, 0].std().item()

    pred_param = model_forward_func(coords, attr, global_attr, model=forward_model)
    results["param_mse"] = torch.mean((pred_param - target_param) ** 2).item()

    if gt_coords is not None:
        results["coord_score"] = coordinate_distribution_score(gt_coords.numpy(), coords.numpy())
        results["radius_score"] = radius_distribution_score(gt_attr[:, 0].numpy(), attr[:, 0].numpy())
        results["coverage"] = coverage_score(gt_coords.numpy(), coords.numpy())
        results["mmd"] = matching_score(gt_coords.numpy(), coords.numpy())

    return results

if __name__ == "__main__":
    torch.manual_seed(42)

    target_param = torch.tensor([0.7, 1.3, 0.08, -25.0, 35.0, 1.55])
    net_config = {
        "node_attr_dim": 8,
        "global_attr_dim": 4,
        "condition_dim": 6,
        "hidden_dim": 128,
    }
    model = BFN4FiberDesign(net_config)
    model.eval()

    coords, attr, global_attr = model.sample(target_param.unsqueeze(0))  # 自动节点数

    forward_model = GCNModel(num_node_features=9, num_graph_attributes=5, output_dim=5)
    forward_model.load_state_dict(torch.load("saved_models_gcn/best_model.pth", map_location="cpu"))
    forward_model.eval()

    results = evaluate_structure(
        coords, attr, global_attr, target_param,
        model_forward_func=model_forward_func,
        forward_model=forward_model,
        gt_coords=coords + 0.01 * torch.randn_like(coords),  # 可替换为真实结构
        gt_attr=attr
    )

    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")