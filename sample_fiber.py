import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from core.models.bfn_fiber import BFN4FiberDesign
from core.utils.visualize import plot_fiber_structure
import yaml

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# 加载模型
model = BFN4FiberDesign(
    net_config={
        "node_attr_dim": config['model']['node_attr_dim'],
        "global_attr_dim": config['model']['global_attr_dim'],
        "condition_dim": config['model']['condition_dim'],
        "hidden_dim": config['model']['hidden_dim'],
        "encoder_type": config['model']['encoder_type']
    },
    sigma1_coord=config['train']['sigma1_coord'],
    beta1=config['train']['beta1'],
    use_discrete_t=True,
    discrete_steps=config['train']['discrete_steps']
)
model.eval()

# 示例目标参数（可换）
target_param = torch.tensor([400.2769142,1.468846752,9.450630532,44.67975884,-1128.64682,95.93520891])
with torch.no_grad():
    coords, attr, global_attr = model.sample(target_param.unsqueeze(0))  # 自动推断节点数

# 可视化
plot_fiber_structure(coords, attr, global_attr, title="Sampled Fiber Structure")