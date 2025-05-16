import torch
import os
import pandas as pd
from core.models.bfn_fiber import BFN4FiberDesign
from core.models.equi_structure_decoder import EquiStructureDecoder

# === 模型配置 ===
net_config = {
    "node_attr_dim": 8,
    "global_attr_dim": 4,
    "condition_dim": 6,
    "hidden_dim": 128,
    "encoder_type": "transformer"
}

def load_model(ckpt_path=None):
    model = BFN4FiberDesign(
        net_config=net_config,
        sigma1_coord=0.03,
        beta1=1.5,
        use_discrete_t=True,
        discrete_steps=1000
    )
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    return model

def sample_structure(model, target_param, num_nodes=60):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        condition = target_param.unsqueeze(0).to(device)
        theta_coord, theta_attr, theta_global = model.sample(condition, num_nodes)
    return theta_coord.cpu(), theta_attr.cpu(), theta_global.cpu()

def save_to_csv(coords, attr, global_attr, output_path):
    """
    保存结构为 CSV 文件
    每行表示一个孔：x, y, r1, r2, shape1, shape2, ...
    最后一行写 global 属性
    """
    coords = coords.numpy()
    attr = attr.numpy()
    df_nodes = pd.DataFrame(
        data = np.concatenate([coords, attr], axis=1),
        columns = ["x", "y"] + [f"attr{i}" for i in range(attr.shape[1])]
    )
    df_global = pd.DataFrame([global_attr.numpy()], columns = [f"global{i}" for i in range(global_attr.shape[0])])
    df_all = pd.concat([df_nodes, df_global], axis=0, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"Saved structure to {output_path}")

if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)

    target_param = torch.tensor([0.7, 1.3, 0.08, -25.0, 35.0, 1.55])  # 可修改
    model = load_model(ckpt_path=None)
    coords, attr, global_attr = sample_structure(model, target_param, num_nodes=60)

    save_to_csv(coords, attr, global_attr, output_path="outputs/sample_structure.csv")