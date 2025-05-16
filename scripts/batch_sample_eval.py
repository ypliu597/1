import torch
import pandas as pd
from core.models.bfn_fiber import BFN4FiberDesign
from core.models.gcn_forward import GCNModel
from core.evaluation.forward_model_interface import model_forward_func
from core.evaluation.metrics import (
    coordinate_distribution_score,
    radius_distribution_score,
    coverage_score,
    matching_score
)

target_params = [
    [0.75, 1.25, 0.06, -35.0, 25.0, 1.50],
    [0.82, 1.18, 0.05, -30.0, 30.0, 1.55],
    [0.68, 1.40, 0.07, -28.0, 33.0, 1.60],
]

def load_bfn():
    net_config = {
        "node_attr_dim": 8,
        "global_attr_dim": 4,
        "condition_dim": 6,
        "hidden_dim": 128
    }
    model = BFN4FiberDesign(net_config)
    model.eval()
    return model

def load_forward_model():
    model = GCNModel(num_node_features=9, num_graph_attributes=5, output_dim=5)
    model.load_state_dict(torch.load("saved_models_gcn/best_model.pth", map_location="cpu"))
    model.eval()
    return model

def evaluate_batch(target_list, out_csv="batch_eval_results.csv"):
    bfn = load_bfn()
    forward_model = load_forward_model()
    results = []

    for i, param in enumerate(target_list):
        print(f"Sampling {i+1}/{len(target_list)}...")
        target_param = torch.tensor(param)
        coords, attr, global_attr = bfn.sample(target_param.unsqueeze(0))  # 自动节点数

        pred_param = model_forward_func(coords, attr, global_attr, model=forward_model)
        param_mse = torch.mean((pred_param - target_param) ** 2).item()

        gt_coords = coords + 0.01 * torch.randn_like(coords)
        gt_attr = attr.clone()

        results.append({
            "index": i,
            "target_param": param,
            "pred_param": pred_param.detach().cpu().numpy(),
            "param_mse": param_mse,
            "coord_score": coordinate_distribution_score(gt_coords.numpy(), coords.numpy()),
            "radius_score": radius_distribution_score(gt_attr[:, 0].numpy(), attr[:, 0].numpy()),
            "coverage": coverage_score(gt_coords.numpy(), coords.numpy()),
            "mmd": matching_score(gt_coords.numpy(), coords.numpy())
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

if __name__ == "__main__":
    evaluate_batch(target_params)