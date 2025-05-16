import torch
from torch_geometric.data import Data, Batch

def model_forward_func(coords, attr, global_attr, model):
    """
    将生成结构输入正向预测模型，返回预测参数（如 n_eff, A_eff, D, GVD, gamma）
    参数：
        coords: [N, 2] 节点坐标
        attr: [N, d] 节点属性（半径、形状）
        global_attr: [d_g] 光纤材料、包层半径等
        model: 已加载权重的 GNN 正向模型（如 GCNModel）

    返回：
        pred_param: [D] 参数预测结果
    """
    model.eval()
    device = next(model.parameters()).device
    coords, attr, global_attr = coords.to(device), attr.to(device), global_attr.to(device)

    x = torch.cat([coords, attr], dim=1)
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)  # 点云无边图

    graph = Data(
        x=x,
        edge_index=edge_index,
        graph_attr=global_attr.unsqueeze(0)
    )
    batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        pred = model(batch)  # [1, D]
    return pred.squeeze(0)