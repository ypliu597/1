# train_bfn1.py (修改版)

import sys
import os
import glob
import yaml
import joblib # <--- 新增: 用于加载 scaler/encoder
import argparse # <--- 新增: 用于命令行参数
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar # <--- 添加 RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger # <--- 可以选择 Logger
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 确保能找到 core 包 ---
# (根据您的项目结构，可能需要调整路径)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # 假设脚本在项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)
# --- --- --- --- --- --- --

# --- 从 core 包导入 ---
from core.datasets.inverse_dataset import FiberInverseDataset # <--- 需要修改这个类以接收共享对象!
from core.models.bfn_fiber import BFN4FiberDesign
from core.utils.train_utils import check_and_fix_nan_params # 假设这个工具函数存在
# --- --- --- --- --- ---

# ========== 配置加载 ==========
def load_config(config_path="configs/default.yaml"):
    """加载 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_path} 未找到，将使用默认参数或命令行参数。")
        return {}

# ========== 加载共享对象函数 (与 train_forward_model.py 中类似) ==========
def load_shared_objects(path):
    """加载共享的 Scaler 和 Encoder"""
    try:
        shared_objects = joblib.load(path)
        print(f"成功从 {path} 加载共享 Scalers 和 Encoders。")
        # 基本检查 (确保加载了正确的东西)
        if 'scalers' not in shared_objects or 'encoders' not in shared_objects:
             raise ValueError("加载的对象缺少 'scalers' 或 'encoders' 键。")
        # 可以根据 FiberInverseDataset 的需要添加更详细的键检查
        if not all(k in shared_objects['scalers'] for k in ['node_coord', 'hole_radius', 'fiber_radius', 'wavelength', 'target']):
             print("警告：加载的 'scalers' 可能缺少 InverseDataset 需要的键（如 'fiber_radius'）。") # 修改为警告，因为Inverse可能不需要target scaler
        if not all(k in shared_objects['encoders'] for k in ['shape', 'material']):
             raise ValueError("加载的 'encoders' 缺少必要的键 ('shape', 'material')。")
        return shared_objects['scalers'], shared_objects['encoders']
    except FileNotFoundError:
        print(f"错误: 找不到共享 Scaler/Encoder 文件: {path}")
        print("请先运行 preprocess_scalers_encoders.py 脚本生成共享对象。")
        sys.exit(1)
    except Exception as e:
        print(f"加载共享 Scaler/Encoder 文件时出错 ({path}): {e}")
        sys.exit(1)

# ========== 修改后的数据加载函数 ==========
def load_data(root, shared_scalers, shared_encoders, test_size=0.2, seed=42):
    """
    加载数据，实例化修改后的 FiberInverseDataset 并进行划分。

    Args:
        root (str): 数据集根目录。
        shared_scalers (dict): 预先拟合好的共享 Scaler。
        shared_encoders (dict): 预先拟合好的共享 Encoder。
        test_size (float): 验证集划分比例。
        seed (int): 随机种子。

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # !!! 关键: 实例化 *修改后* 的 FiberInverseDataset，传入共享对象 !!!
    # !!! 您需要先修改 FiberInverseDataset 类使其接受这些参数 !!!
    try:
        dataset = FiberInverseDataset(
            root=root,
            shared_scalers=shared_scalers, # <--- 传入
            shared_encoders=shared_encoders  # <--- 传入
            # use_column_names 参数根据需要设置
        )
    except TypeError as e:
         print("\n错误：实例化 FiberInverseDataset 时出错。")
         print("请确保您已经修改了 'core/datasets/inverse_dataset.py' 中的 FiberInverseDataset 类，")
         print("使其 __init__ 方法能够接收 'shared_scalers' 和 'shared_encoders' 参数，")
         print("并且移除了内部的 _precompute_normalization 调用。")
         print(f"原始错误: {e}")
         sys.exit(1)

    if len(dataset) == 0:
        print("错误：数据集为空，请检查数据路径和文件。")
        sys.exit(1)

    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=seed
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

# ========== Lightning Module (基本保持不变) ==========
# LitWrapper 类定义基本保持不变，因为它操作的是模型和批次数据，
# 而数据格式对齐是在 Dataset 类中完成的。
# (这里省略 LitWrapper 的代码，与您提供的版本相同)
class LitWrapper(pl.LightningModule):
    def __init__(self, model, train_hparams): # 传入训练超参数字典
        super().__init__()
        self.model = model
        self.train_hparams = train_hparams
        self.learning_rate = self.train_hparams.get('lr', 1e-3) # 从字典获取 lr
        self.node_count_loss_weight = self.train_hparams.get('node_count_loss_weight', 0.1) # 获取权重

        # (移除 LitWrapper 内部关于 node_sampler 的逻辑)

    def training_step(self, batch, batch_idx):
        num_graphs_in_batch = batch.num_graphs
        # 使用 torch.rand 创建在模型所在设备上的张量
        t_per_graph = torch.rand(num_graphs_in_batch, 1, device=self.device) * 0.99 + 0.01
        batch_size_log = batch.num_graphs # 用于日志记录的 batch size

        try:
            closs, dloss_attr, dloss_global, loss_node_count = self.model.loss_one_step(batch, t_per_graph)

            # 检查 NaN
            has_nan = False
            if torch.isnan(closs): has_nan=True; closs = torch.tensor(0.0, device=self.device, requires_grad=True) # 赋予梯度
            if torch.isnan(dloss_attr): has_nan=True; dloss_attr = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(dloss_global): has_nan=True; dloss_global = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(loss_node_count): has_nan=True; loss_node_count = torch.tensor(0.0, device=self.device, requires_grad=True)

            if has_nan:
                self.print(f"[❌ NaN @ Train Step {self.global_step}] Replaced NaN losses with 0.")

            total_loss = closs + dloss_attr + dloss_global + self.node_count_loss_weight * loss_node_count

            # 日志记录
            self.log("train/coord_loss", closs, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("train/attr_loss", dloss_attr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("train/global_loss", dloss_global, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("train/node_count_loss", loss_node_count, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)

            return total_loss

        except Exception as e:
             print(f"\nError during training step {self.global_step}: {e}")
             # 可以选择跳过这个 batch 或停止训练
             # return None # 跳过 batch 可能导致问题，最好是修复错误
             raise e # 重新抛出异常，停止训练

    def validation_step(self, batch, batch_idx):
        num_graphs_in_batch = batch.num_graphs
        t_per_graph = torch.rand(num_graphs_in_batch, 1, device=self.device) * 0.99 + 0.01
        batch_size_log = batch.num_graphs

        try:
            closs, dloss_attr, dloss_global, loss_node_count = self.model.loss_one_step(batch, t_per_graph)

             # 检查 NaN
            has_nan = False
            if torch.isnan(closs): has_nan=True; closs = torch.tensor(0.0, device=self.device)
            if torch.isnan(dloss_attr): has_nan=True; dloss_attr = torch.tensor(0.0, device=self.device)
            if torch.isnan(dloss_global): has_nan=True; dloss_global = torch.tensor(0.0, device=self.device)
            if torch.isnan(loss_node_count): has_nan=True; loss_node_count = torch.tensor(0.0, device=self.device)

            if has_nan:
                self.print(f"[❌ NaN @ Validation Step {self.global_step}] Replaced NaN losses with 0.")


            total_loss = closs + dloss_attr + dloss_global + self.node_count_loss_weight * loss_node_count

            self.log("val/coord_loss", closs, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("val/attr_loss", dloss_attr, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("val/global_loss", dloss_global, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("val/node_count_loss", loss_node_count, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
            self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log) # 监控指标

        except Exception as e:
             print(f"\nError during validation step {self.global_step}: {e}")
             # 记录一个非常大的损失值，以便监控和早停能正确反应
             self.log("val/total_loss", torch.tensor(float('inf')), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size_log)
             # raise e # 或者直接停止


    def configure_optimizers(self):
        optimizer_name = self.train_hparams.get('optimizer', 'Adam')
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
             raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 可选的学习率调度器 (与 forward 训练脚本一致)
        scheduler_config = self.train_hparams.get('lr_scheduler')
        if scheduler_config and scheduler_config.get('enabled', False):
             if scheduler_config.get('name') == 'ReduceLROnPlateau':
                 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer,
                     mode='min',
                     factor=scheduler_config.get('factor', 0.1),
                     patience=scheduler_config.get('patience', 10), # ReduceLROnPlateau 的 patience
                     verbose=True
                 )
                 return {
                     'optimizer': optimizer,
                     'lr_scheduler': {
                         'scheduler': scheduler,
                         'monitor': 'val/total_loss',
                         'interval': 'epoch',
                         'frequency': 1,
                     }
                 }
             else:
                 print(f"警告: 不支持的学习率调度器名称 '{scheduler_config.get('name')}'")
                 return optimizer
        else:
             return optimizer


# ========== 主执行逻辑 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练逆向光纤设计模型 (BFN)")
    parser.add_argument('--config_file', type=str, default='configs/default.yaml', help='主配置文件路径')
    # --- 添加命令行参数以覆盖配置 ---
    parser.add_argument('--data_root', type=str, default=None, help='数据集根目录 (覆盖配置)')
    parser.add_argument('--scaler_encoder_path', type=str, default=None, help='共享 Scaler/Encoder 文件路径 (覆盖配置, 必需)')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='检查点保存目录 (覆盖配置)')
    parser.add_argument('--log_dir', type=str, default=None, help='日志保存目录 (覆盖配置)')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小 (覆盖配置)')
    parser.add_argument('--epochs', type=int, default=None, help='最大训练轮数 (覆盖配置)')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (覆盖配置)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载 worker 数量 (覆盖配置)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (覆盖配置)')
    parser.add_argument('--test_size', type=float, default=None, help='验证集比例 (覆盖配置)')
    parser.add_argument('--node_count_loss_weight', type=float, default=None, help='节点数量损失权重 (覆盖配置)')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值 (覆盖配置)')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定的检查点恢复训练')

    args = parser.parse_args()

    # --- 加载和整合配置 ---
    config = load_config(args.config_file)
    model_config_base = config.get('model', {})
    train_config_base = config.get('train', {})
    data_config_base = config.get('data', {})

    # 使用命令行参数或配置文件值，命令行优先
    data_root = args.data_root or data_config_base.get('dataset_path', 'core/datasets/') # 修改默认路径
    scaler_encoder_path = args.scaler_encoder_path or data_config_base.get('scaler_encoder_path') # 应在配置文件或命令行提供
    if not scaler_encoder_path:
        print("错误：必须通过 --scaler_encoder_path 或在配置文件的 data.scaler_encoder_path 中指定共享 scaler/encoder 文件路径。")
        sys.exit(1)

    ckpt_dir = args.ckpt_dir or train_config_base.get('ckpt_dir', 'checkpoints/')
    log_dir = args.log_dir or train_config_base.get('log_dir', 'logs/')
    batch_size = args.batch_size or train_config_base.get('batch_size', 16)
    epochs = args.epochs or train_config_base.get('epochs', 1000)
    lr = args.lr or train_config_base.get('lr', 0.001)
    num_workers = args.num_workers or train_config_base.get('num_workers', 8) # 保持默认8?
    seed = args.seed or train_config_base.get('seed', 42)
    test_size = args.test_size or train_config_base.get('test_size', 0.2)
    node_count_loss_weight = args.node_count_loss_weight or train_config_base.get('node_count_loss_weight', 0.1) # 默认0.1
    patience = args.patience or train_config_base.get('patience', 20) # 早停耐心

    # 将训练相关超参数整合到字典中，传递给 LitWrapper
    train_hparams = {
        'lr': lr,
        'optimizer': train_config_base.get('optimizer', 'Adam'),
        'lr_scheduler': train_config_base.get('lr_scheduler'), # 从配置加载调度器设置
        'node_count_loss_weight': node_count_loss_weight,
        # 可以添加其他需要记录或使用的训练参数
    }


    pl.seed_everything(seed)
    torch.autograd.set_detect_anomaly(True) # 保持异常检测
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- 加载共享对象 ---
    shared_scalers, shared_encoders = load_shared_objects(scaler_encoder_path)

    # --- 数据准备 (使用修改后的加载函数) ---
    print("准备数据加载器...")
    train_dataset, val_dataset = load_data(
        root=data_root,
        shared_scalers=shared_scalers,
        shared_encoders=shared_encoders,
        test_size=test_size,
        seed=seed
    )
    print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    # --- 模型准备 ---
    print("构建逆向模型...")
    # 传递来自配置文件的模型结构参数
    model = BFN4FiberDesign(
        net_config={
            "node_attr_dim": model_config_base.get('node_attr_dim', 5), # 确保与 FiberInverseDataset 输出一致
            "global_attr_dim": model_config_base.get('global_attr_dim', 4), # 确保与 FiberInverseDataset 输出一致
            "condition_dim": model_config_base.get('condition_dim', 6), # 确保与 FiberInverseDataset 输出一致
            "hidden_dim": model_config_base.get('hidden_dim', 128),
            "encoder_type": model_config_base.get('encoder_type', 'transformer'),
            # "node_classes": model_config_base.get('node_classes', [18, 36, 60, 90]) # 如果 node_classes 在 model 配置下
        },
        # 传递来自配置文件的 BFN 训练参数
        sigma1_coord=train_config_base.get('sigma1_coord', 0.03),
        beta1=train_config_base.get('beta1', 1.5),
        use_discrete_t=train_config_base.get('use_discrete_t', True), # 假设配置中有
        discrete_steps=train_config_base.get('discrete_steps', 1000)
    )
    # check_and_fix_nan_params(model) # 按需保留

    # --- Lightning 封装 ---
    lit_model = LitWrapper(model, train_hparams) # 传递训练超参数字典

    # --- Callbacks ---
    print("设置 Callbacks 和 Logger...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="val/total_loss", # 监控验证集总损失
        mode="min",
        filename="bfn-best-{epoch}-{val/total_loss:.4f}" # 更好的文件名
    )
    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        patience=patience, # 使用配置的耐心值
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = RichProgressBar()

    # --- Logger ---
    use_wandb = train_config_base.get('use_wandb', True) # 假设默认使用 Wandb
    if use_wandb:
        logger = WandbLogger(
            project=train_config_base.get('wandb_project', "FiberInverseDesign"),
            name=train_config_base.get('wandb_run_name', "BFN_AlignedData_Run"), # 更新名称
            save_dir=log_dir,
            config={**model_config_base, **train_hparams} # 记录超参数
        )
    else:
        logger = TensorBoardLogger(save_dir=log_dir, name="bfn_inverse_model")


    # --- Trainer ---
    print("配置 Trainer...")
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=train_config_base.get('log_steps', 10),
        accelerator="auto",
        devices="auto", # 使用自动设备选择
        # precision=train_config_base.get('precision', '32-true'), # 可以配置精度
        gradient_clip_val=train_config_base.get('grad_clip', 1.0),
        # check_val_every_n_epoch=5, # 按需配置
        # num_sanity_val_steps=0
    )

    # --- 检查恢复点 ---
    resume_checkpoint_path = args.resume_from # 从命令行获取恢复路径
    if not resume_checkpoint_path and train_config_base.get('auto_resume', True): # 尝试自动恢复
         checkpoint_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
         if checkpoint_files:
             resume_checkpoint_path = checkpoint_files[-1]
             print(f"[INFO] 自动从最近的检查点恢复: {resume_checkpoint_path}")

    # --- 开始训练 ---
    print("开始训练逆向模型...")
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_checkpoint_path) # ckpt_path 用于恢复
    print("训练完成！")
    if checkpoint_callback.best_model_path:
        print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")