# MolCRAFT/core/models/bfn_fiber.py
import sys
import os
import warnings  # 引入 warnings

# 添加项目根目录到 sys.path，确保 core 可以被找到
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 通常不需要如果项目结构良好或使用相对导入
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch_geometric.utils import degree  # 引入 degree 用于获取每个图的节点数

# 确保从更新后的 bfn_base 导入
from core.models.bfn_base import BFNBase
from core.models.equi_structure_decoder import EquiStructureDecoder
from core.models.condition_encoder import ConditionEncoder
from core.models.node_count_sampler import NodeCountSampler


# MolCRAFT/core/models/bfn_fiber.py

# ... (imports) ...

class BFN4FiberDesign(BFNBase):
    def __init__(self, net_config, sigma1_coord=0.03, beta1=1.5, use_discrete_t=True, discrete_steps=1000):
        # super().__init__(net_config=net_config) # BFNBase 没有 __init__ 参数，不需要传递
        super().__init__()  # 直接调用基类构造函数

        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)
        self.use_discrete_t = use_discrete_t
        self.discrete_steps = discrete_steps

        # --- 存储原始配置维度 ---
        self.node_attr_dim = net_config['node_attr_dim']
        self.global_dim = net_config['global_attr_dim']
        self.input_condition_dim = net_config['condition_dim']
        self.hidden_dim = net_config['hidden_dim']
        self.coord_dim = 2

        # --- 定义和存储连续/离散部分的维度 ---
        # 假设 node_attr = [radius1, radius2, shape_onehot...]
        self.D_radius_node = 2
        self.K_shape = self.node_attr_dim - self.D_radius_node
        if self.K_shape <= 0:
            raise ValueError(
                f"node_attr_dim ({self.node_attr_dim}) must be greater than D_radius_node ({self.D_radius_node})")

        # 假设 global_attr = [fiber_radius, material_onehot...]
        self.D_radius_global = 1
        self.K_material = self.global_dim - self.D_radius_global
        if self.K_material <= 0:
            raise ValueError(
                f"global_dim ({self.global_dim}) must be greater than D_radius_global ({self.D_radius_global})")
        # --- 结束维度定义 ---

        self.condition_encoder = ConditionEncoder(
            input_dim=self.input_condition_dim,
            hidden_dim=self.hidden_dim,
            encoder_type=net_config.get('encoder_type', 'transformer')
        )

        # NodeCountSampler 初始化
        node_classes_list = net_config.get('node_classes', [18, 36, 60, 90])
        self.num_node_classes = len(node_classes_list)
        self.node_sampler = NodeCountSampler(
            condition_dim=self.hidden_dim,
            num_classes=self.num_node_classes
        )
        if not hasattr(self.node_sampler, 'class_to_nodes'):
            self.node_sampler.class_to_nodes = torch.tensor(node_classes_list)
        self.node_count_to_class_idx = {count: idx for idx, count in enumerate(node_classes_list)}

        # EquiStructureDecoder 初始化
        self.model = EquiStructureDecoder(
            node_attr_dim=self.node_attr_dim,
            global_dim=self.global_dim,
            input_condition_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            encoder_type=net_config.get('encoder_type', 'transformer'),
        )

        # Buffer 注册
        self.register_buffer('sigma1_coord_buffer', self.sigma1_coord)
        self.register_buffer('beta1_buffer', self.beta1)

    def map_num_nodes_to_class(self, num_nodes_tensor):
        # ... (代码不变) ...
        # 将 class_to_nodes 移到正确的设备
        class_nodes = self.node_sampler.class_to_nodes.to(num_nodes_tensor.device)
        diff = torch.abs(num_nodes_tensor.unsqueeze(1) - class_nodes.unsqueeze(0))
        closest_class_idx = torch.argmin(diff, dim=1)
        return closest_class_idx

    def interdependency_modeling(self, theta_coord, theta_attr, theta_global, t, condition, batch):
        # ... (代码不变) ...
        return self.model(theta_coord, theta_attr, theta_global, t, condition, batch)

    def loss_one_step(self, data, t_per_graph):  # 输入参数已改为 t_per_graph
        # ... (Assertions, Batch Info) ...

        # --- 移除局部的维度定义 ---
        # D_radius_node = 2 (使用 self.D_radius_node)
        # K_shape = 3 (使用 self.K_shape)
        # D_radius_global = 1 (使用 self.D_radius_global)
        # K_material = 3 (使用 self.K_material)

        # --- Condition Encoding ---
        raw_condition = data.input_condition.view(data.num_graphs, -1)
        encoded_condition = self.condition_encoder(raw_condition)

        # --- Node Count Sampler Loss ---
        nodes_per_graph_list = []
        for i in range(data.num_graphs):
            nodes_per_graph_list.append((data.batch == i).sum())
        true_num_nodes_per_graph = torch.stack(nodes_per_graph_list).to(encoded_condition.device)
        node_count_logits = self.node_sampler.net(encoded_condition)
        target_node_class_idx = self.map_num_nodes_to_class(true_num_nodes_per_graph)
        loss_node_count = F.cross_entropy(node_count_logits, target_node_class_idx, reduction='mean')

        # --- Prepare Time Variables (修正后的逻辑) ---
        batch_idx = data.batch
        num_graphs = data.num_graphs
        num_total_nodes = data.num_nodes
        if t_per_graph.shape[0] != num_graphs:
            raise ValueError(
                f"Input time t shape {t_per_graph.shape} does not match num_graphs {num_graphs}. Expected per-graph time.")
        t_node = t_per_graph[batch_idx]
        t_graph = t_per_graph
        encoded_condition_node = encoded_condition[batch_idx]
        # --- End Time Prep ---

        # --- Bayesian Flow (Forward noising) ---
        mu_coord_t, gamma_coord_t = self.continuous_var_bayesian_update(
            t_node, self.sigma1_coord_buffer, data.hole_pos
        )
        # 假设 discrete_var_bayesian_update 应用于整个块
        theta_attr_t = self.discrete_var_bayesian_update(
            t_node, self.beta1_buffer, data.hole_attr, K=self.node_attr_dim  # K是总维度
        )
        theta_global_t = self.discrete_var_bayesian_update(
            t_graph, self.beta1_buffer, data.global_attr, K=self.global_dim  # K是总维度
        )

        # --- Model Prediction ---
        coord_pred_x0, attr_pred_x0, global_pred_x0 = self.interdependency_modeling(
            mu_coord_t, theta_attr_t, theta_global_t, t_node,
            encoded_condition_node, batch_idx
        )

        # --- Loss Calculation ---
        # Coordinate Loss (closs)
        # ... (closs calculation logic remains the same, use self.sigma1_coord_buffer) ...
        if self.use_discrete_t:
            i_node = (t_node * self.discrete_steps).int() + 1
            i_node = torch.clamp(i_node, min=1, max=self.discrete_steps)
            closs_elements = self.dtime4continuous_loss(
                i_node, self.discrete_steps, self.sigma1_coord_buffer,
                x_pred=coord_pred_x0, x=data.hole_pos, segment_ids=None
            )
        else:
            closs_elements = self.ctime4continuous_loss(
                t_node, self.sigma1_coord_buffer, x_pred=coord_pred_x0, x=data.hole_pos, segment_ids=None
            )
        if torch.isnan(closs_elements).any():
            warnings.warn("NaN detected in coordinate loss elements.", RuntimeWarning)
            closs = torch.tensor(0.0, device=closs_elements.device)
        else:
            closs = closs_elements.mean()

        # Node Attribute Loss (dloss_attr)
        target_node_radii = data.hole_attr[:, :self.D_radius_node]
        target_node_shape_onehot = data.hole_attr[:, self.D_radius_node:]  # Shape [N_nodes, K_shape]
        pred_node_radii = attr_pred_x0[:, :self.D_radius_node]
        pred_node_shape_logits = attr_pred_x0[:, self.D_radius_node:]  # Shape [N_nodes, K_shape]

        loss_node_radii = F.mse_loss(pred_node_radii, target_node_radii)

        if self.use_discrete_t:
            pred_node_shape_probs = F.softmax(pred_node_shape_logits, dim=-1)
            i_node_for_attr = (t_node * self.discrete_steps).int() + 1
            i_node_for_attr = torch.clamp(i_node_for_attr, min=1, max=self.discrete_steps)
            loss_node_shape_elements = self.dtime4discrete_loss_prob(
                i_node_for_attr, self.discrete_steps, self.beta1_buffer,
                target_node_shape_onehot, pred_node_shape_probs, self.K_shape,  # Use self.K_shape
                n_samples=10, segment_ids=None
            )
        else:
            pred_node_shape_probs = F.softmax(pred_node_shape_logits, dim=-1)
            loss_node_shape_elements = self.ctime4discrete_loss(
                t_node, self.beta1_buffer, target_node_shape_onehot,
                pred_node_shape_probs, self.K_shape, segment_ids=None  # Use self.K_shape
            )

        if torch.isnan(loss_node_radii).any() or torch.isnan(loss_node_shape_elements).any():
            warnings.warn(
                f"NaN detected in node attribute loss components (radius:{torch.isnan(loss_node_radii).any()}, shape:{torch.isnan(loss_node_shape_elements).any()}).",
                RuntimeWarning)
            dloss_attr = torch.tensor(0.0, device=loss_node_radii.device)
        else:
            loss_node_shape = loss_node_shape_elements.mean()
            dloss_attr = loss_node_radii + loss_node_shape

        # Global Attribute Loss (dloss_global)
        target_global_radius = data.global_attr[:, :self.D_radius_global]
        target_global_material_onehot = data.global_attr[:, self.D_radius_global:]  # Shape [num_graphs, K_material]
        pred_global_radius = global_pred_x0[:, :self.D_radius_global]
        pred_global_material_logits = global_pred_x0[:, self.D_radius_global:]  # Shape [num_graphs, K_material]

        loss_global_radius = F.mse_loss(pred_global_radius, target_global_radius)

        if self.use_discrete_t:
            pred_global_material_probs = F.softmax(pred_global_material_logits, dim=-1)
            i_graph = (t_graph * self.discrete_steps).int() + 1
            i_graph = torch.clamp(i_graph, min=1, max=self.discrete_steps)
            loss_global_material_elements = self.dtime4discrete_loss_prob(
                i_graph, self.discrete_steps, self.beta1_buffer,
                target_global_material_onehot, pred_global_material_probs, self.K_material,  # Use self.K_material
                n_samples=10, segment_ids=None
            )
        else:
            pred_global_material_probs = F.softmax(pred_global_material_logits, dim=-1)
            loss_global_material_elements = self.ctime4discrete_loss(
                t_graph, self.beta1_buffer, target_global_material_onehot,
                pred_global_material_probs, self.K_material, segment_ids=None  # Use self.K_material
            )

        if torch.isnan(loss_global_radius).any() or torch.isnan(loss_global_material_elements).any():
            warnings.warn(
                f"NaN detected in global attribute loss components (radius:{torch.isnan(loss_global_radius).any()}, material:{torch.isnan(loss_global_material_elements).any()}).",
                RuntimeWarning)
            dloss_global = torch.tensor(0.0, device=loss_global_radius.device)
        else:
            loss_global_material = loss_global_material_elements.mean()
            dloss_global = loss_global_radius + loss_global_material

        if torch.isnan(loss_node_count).any():
            warnings.warn("NaN detected in node count loss.", RuntimeWarning)
            loss_node_count = torch.tensor(0.0, device=loss_node_count.device)

        # --- Return all loss components ---
        return closs, dloss_attr, dloss_global, loss_node_count

    @torch.no_grad()
    def sample(self, raw_condition, num_nodes=None, sample_steps=None, T_min=1e-4):
        # ... (Initialization, node count sampling logic - no changes needed here) ...
        self.eval()
        device = next(self.parameters()).device
        raw_condition = raw_condition.to(device)
        encoded_condition = self.condition_encoder(raw_condition)

        if num_nodes is None:
            if not hasattr(self, 'node_sampler'):
                raise AttributeError("NodeCountSampler 'node_sampler' not found, but num_nodes was not provided.")
            node_count_logits = self.node_sampler.net(encoded_condition)
            pmf = F.softmax(node_count_logits, dim=-1)
            sampled_idx = torch.multinomial(pmf, num_samples=1).squeeze(1).item()
            num_nodes = int(self.node_sampler.class_to_nodes[sampled_idx].item())
            print(f"Sampled num_nodes: {num_nodes}")

        t_steps = sample_steps if sample_steps is not None else self.discrete_steps
        mu_pos_t = torch.randn(num_nodes, self.coord_dim, device=device)
        theta_attr_t = torch.ones(num_nodes, self.node_attr_dim, device=device) / self.node_attr_dim
        theta_global_t = torch.ones(1, self.global_dim, device=device) / self.global_dim
        encoded_condition_node = encoded_condition.expand(num_nodes, -1)
        batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        # --- End Initialization ---

        for i in range(t_steps, 0, -1):
            t_curr_val = i / t_steps
            t_prev_val = (i - 1) / t_steps
            t_curr_node = torch.full((num_nodes, 1), t_curr_val, dtype=torch.float32, device=device)
            t_curr_node_clipped = torch.clamp(t_curr_node, min=T_min)
            t_prev_node = torch.full((num_nodes, 1), t_prev_val, dtype=torch.float32, device=device)

            # --- Model Prediction ---
            coord_pred_x0, attr_pred_x0, global_pred_x0 = self.interdependency_modeling(
                mu_pos_t, theta_attr_t, theta_global_t,
                t_curr_node_clipped,
                encoded_condition_node, batch_idx
            )

            # --- Check Finite ---
            if not torch.isfinite(coord_pred_x0).all() or \
                    not torch.isfinite(attr_pred_x0).all() or \
                    not torch.isfinite(global_pred_x0).all():
                warnings.warn(f"Non-finite prediction detected at step {i}. Stopping sampling.", RuntimeWarning)
                return mu_pos_t, theta_attr_t, theta_global_t

            # --- Intermediate Noise (for coords) ---
            noise_std = (1 / t_steps) ** 0.5 if t_steps > 0 else 0.0
            y_coord = coord_pred_x0 + torch.randn_like(coord_pred_x0) * noise_std

            # --- Update State to t_prev ---
            if i > 1:
                # Coordinates
                mu_pos_t_prev, _ = self.continuous_var_bayesian_update(
                    t_prev_node, self.sigma1_coord_buffer, y_coord
                )

                # Node Attributes (Separated)
                pred_node_radii_x0 = attr_pred_x0[:, :self.D_radius_node]  # Use self.D_radius_node
                pred_node_shape_logits_x0 = attr_pred_x0[:, self.D_radius_node:]  # Use self.D_radius_node
                pred_node_shape_probs_x0 = F.softmax(pred_node_shape_logits_x0, dim=-1)

                theta_attr_t_discrete_prev = self.discrete_var_bayesian_update(
                    t_prev_node, self.beta1_buffer, pred_node_shape_probs_x0, K=self.K_shape  # Use self.K_shape
                )
                mu_node_radii_t_prev, _ = self.continuous_var_bayesian_update(
                    t_prev_node, self.sigma1_coord_buffer, pred_node_radii_x0
                )
                theta_attr_t_prev = torch.cat([mu_node_radii_t_prev, theta_attr_t_discrete_prev], dim=-1)

                # Global Attributes (Separated)
                pred_global_radius_x0 = global_pred_x0[:, :self.D_radius_global]  # Use self.D_radius_global
                pred_global_material_logits_x0 = global_pred_x0[:, self.D_radius_global:]  # Use self.D_radius_global
                pred_global_material_probs_x0 = F.softmax(pred_global_material_logits_x0, dim=-1)

                t_prev_graph = torch.full((1, 1), t_prev_val, dtype=torch.float32, device=device)

                theta_global_t_material_prev = self.discrete_var_bayesian_update(
                    t_prev_graph, self.beta1_buffer, pred_global_material_probs_x0, K=self.K_material
                    # Use self.K_material
                )
                mu_global_radius_t_prev, _ = self.continuous_var_bayesian_update(
                    t_prev_graph, self.sigma1_coord_buffer, pred_global_radius_x0
                )
                theta_global_t_prev = torch.cat([mu_global_radius_t_prev, theta_global_t_material_prev], dim=-1)

                # Update state
                mu_pos_t = mu_pos_t_prev
                theta_attr_t = theta_attr_t_prev
                theta_global_t = theta_global_t_prev

            else:  # i == 1, final step -> use predicted x0
                mu_pos_t = coord_pred_x0
                # Node Attr
                pred_node_radii_x0 = attr_pred_x0[:, :self.D_radius_node]
                pred_node_shape_logits_x0 = attr_pred_x0[:, self.D_radius_node:]
                pred_node_shape_probs_x0 = F.softmax(pred_node_shape_logits_x0, dim=-1)
                theta_attr_t = torch.cat([pred_node_radii_x0, pred_node_shape_probs_x0], dim=-1)
                # Global Attr
                pred_global_radius_x0 = global_pred_x0[:, :self.D_radius_global]
                pred_global_material_logits_x0 = global_pred_x0[:, self.D_radius_global:]
                pred_global_material_probs_x0 = F.softmax(pred_global_material_logits_x0, dim=-1)
                theta_global_t = torch.cat([pred_global_radius_x0, pred_global_material_probs_x0], dim=-1)
                break  # End loop

        return mu_pos_t, theta_attr_t, theta_global_t

    # map_num_nodes_to_class method remains the same