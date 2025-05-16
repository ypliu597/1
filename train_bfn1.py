import sys
import os
import glob
import yaml
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from core.utils.train_utils import check_and_fix_nan_params
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from core.datasets.fiber_dataset import FiberInverseDataset
from core.models.bfn_fiber import BFN4FiberDesign

# ========== 配置加载 ==========

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# ========== Wandb Logger ==========

wandb_logger = WandbLogger(
    project="FiberInverseDesign",
    name="BFN_Run",
    log_model=True,
    save_dir=config['train']['log_dir']
)

# ========== Resume Checkpoint 检测 ==========

ckpt_dir = config['train']['ckpt_dir']
os.makedirs(ckpt_dir, exist_ok=True)
latest_checkpoint = None
checkpoint_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
# if checkpoint_files:
#     latest_checkpoint = checkpoint_files[-1]
#     print(f"[INFO] Found checkpoint: {latest_checkpoint}")
# else:
#     print("[INFO] No checkpoint found, training from scratch.")

# ========== 数据加载 ==========

def load_data(root, test_size=0.2):
    dataset = FiberInverseDataset(root=root)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

train_dataset, val_dataset = load_data(root="core/datasets/", test_size=0.2)



train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], num_workers=8)

model = BFN4FiberDesign(
    net_config={
        "node_attr_dim": config['model']['node_attr_dim'],
        "global_attr_dim": config['model']['global_attr_dim'],
        "condition_dim": config['model']['condition_dim'],
        "hidden_dim": config['model']['hidden_dim'],
        "encoder_type": config['model']['encoder_type'],
    },
    sigma1_coord=config['train']['sigma1_coord'],
    beta1=config['train']['beta1'],
    use_discrete_t=True,
    discrete_steps=config['train']['discrete_steps']
)
check_and_fix_nan_params(model)

# ========== Lightning Module ==========

class LitWrapper(pl.LightningModule):
    def __init__(self, model, node_count_loss_weight=1.0):  # 添加权重参数
        super().__init__()
        self.model = model
        self.node_count_loss_weight = node_count_loss_weight
        # 不再需要在 LitWrapper 中存储 node_sampler 的信息或映射函数
        # self.register_buffer("node_sampler_class_nodes", ...) # 移除
        # self.node_sampler_map = ... # 移除

    # map_num_nodes_to_class_lit 函数也不再需要，这个逻辑在 BFN4FiberDesign.loss_one_step 中处理
    # def map_num_nodes_to_class_lit(self, num_nodes_tensor): # 移除
    #    ...

    def training_step(self, batch, batch_idx):
        # --- 修改时间采样：每个图采样一个 t ---
        num_graphs_in_batch = batch.num_graphs
        # 确保在正确的设备上采样
        t_per_graph = torch.rand(num_graphs_in_batch, 1, device=self.device) * 0.99 + 0.01  # Shape: [num_graphs, 1]
        # --- 结束时间采样修改 ---

        # --- 调用 BFN4FiberDesign 的 loss_one_step ---
        # 将每个图的时间 t_per_graph 传递给 loss_one_step
        # 假设 self.model.loss_one_step 已被修改为接收和处理 t_per_graph
        closs, dloss_attr, dloss_global, loss_node_count = self.model.loss_one_step(batch, t_per_graph)

        # 检查 NaN
        if any(torch.isnan(x) for x in [closs, dloss_attr, dloss_global, loss_node_count]):
            self.print(
                f"[❌ NaN @ Train Step {self.global_step}] Losses: coord={closs}, attr={dloss_attr}, global={dloss_global}, node_count={loss_node_count}"
            )
            # 替换 NaN 损失为 0 并继续训练 (可能掩盖问题，但避免崩溃)
            closs = torch.nan_to_num(closs, nan=0.0)
            dloss_attr = torch.nan_to_num(dloss_attr, nan=0.0)
            dloss_global = torch.nan_to_num(dloss_global, nan=0.0)
            loss_node_count = torch.nan_to_num(loss_node_count, nan=0.0)
            # 如果总是有 NaN，考虑 raise ValueError("NaN detected, stopping training.")

        # --- 计算总损失 ---
        total_loss = closs + dloss_attr + dloss_global + self.node_count_loss_weight * loss_node_count

        # --- 日志记录 ---
        # 使用 sync_dist=True 确保在分布式训练时正确聚合日志
        self.log("train/coord_loss", closs, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/attr_loss", dloss_attr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/global_loss", dloss_global, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("train/node_count_loss", loss_node_count, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # --- 修改时间采样：每个图采样一个 t ---
        num_graphs_in_batch = batch.num_graphs
        # 确保在正确的设备上采样
        t_per_graph = torch.rand(num_graphs_in_batch, 1, device=self.device) * 0.99 + 0.01  # Shape: [num_graphs, 1]
        # --- 结束时间采样修改 ---

        # --- 调用 BFN4FiberDesign 的 loss_one_step ---
        closs, dloss_attr, dloss_global, loss_node_count = self.model.loss_one_step(batch,
                                                                                    t_per_graph)  # 传递 t_per_graph

        # 检查 NaN
        if any(torch.isnan(x) for x in [closs, dloss_attr, dloss_global, loss_node_count]):
            self.print(
                f"[❌ NaN @ Validation] Losses: coord={closs}, attr={dloss_attr}, global={dloss_global}, node_count={loss_node_count}"
            )
            closs = torch.nan_to_num(closs, nan=0.0)
            dloss_attr = torch.nan_to_num(dloss_attr, nan=0.0)
            dloss_global = torch.nan_to_num(dloss_global, nan=0.0)
            loss_node_count = torch.nan_to_num(loss_node_count, nan=0.0)

        # --- 计算总损失 ---
        total_loss = closs + dloss_attr + dloss_global + self.node_count_loss_weight * loss_node_count

        # --- 日志记录 ---
        # on_step=False (默认), on_epoch=True 用于验证步骤
        self.log("val/coord_loss", closs, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/attr_loss", dloss_attr, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/global_loss", dloss_global, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/node_count_loss", loss_node_count, prog_bar=True, logger=True, sync_dist=True)  # 可以在进度条显示
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)  # 监控指标

        # PTL 2.0+ validation_step 不返回 loss

    def configure_optimizers(self):
        # --- 优化器和调度器配置 ---
        # 确保 self.parameters() 包含模型的所有可训练参数
        # 如果 BFN4FiberDesign 正确继承 nn.Module 并注册了所有子模块，这应该是自动的
        optimizer = torch.optim.Adam(self.parameters(), lr=config['train']['lr'])

        # 学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',  # 监控验证集的总损失
                'interval': 'epoch',  # 每个 epoch 结束时检查
                'frequency': 1,  # 每个检查点检查一次
            }
        }

# ========== Trainer ==========

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True) # 保持 anomaly detection

    # --- 实例化 LitWrapper ---
    # 您可以在这里传递 node_count_loss_weight
    lit_model = LitWrapper(model, node_count_loss_weight=0.1) # 例如，给节点数损失一个较小的初始权重

    # --- Callbacks ---
    # ... (checkpoint_callback, early_stop_callback, lr_monitor 定义) ...
    # 根据需要调整 monitor 的 key
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="val/total_loss", # 可以监控总损失或某个关键损失
        mode="min",
        filename="best_model-{epoch}-{val/total_loss:.4f}" # 包含epoch和指标的文件名
    )
    early_stop_callback = EarlyStopping(
        monitor="val/total_loss", # 监控验证集总损失
        patience=20, # 增加早停耐心
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1, # 假设单卡训练
        # precision='16-mixed' if torch.cuda.is_available() else None, # 尝试混合精度加速
        gradient_clip_val=1.0, # 保持梯度裁剪
        # check_val_every_n_epoch=5, # 可以调整验证频率
        # num_sanity_val_steps=0 # 跳过初始的 sanity check 加快启动
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=latest_checkpoint) # 使用修改后的 lit_model