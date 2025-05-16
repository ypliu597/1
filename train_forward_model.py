# 建议文件名: train_forward_model.py

import os
import sys
import glob
import yaml
import joblib
import argparse

import torch
import torch.nn as nn # 导入 nn
import torch.nn.functional as F # 导入 F 用于 SmoothL1Loss
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

# --- 确保能找到 core 包 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# --- --- --- --- --- --- --

from core.models.equi_forward_model import EquiForwardModel
from core.datasets.forward_dataset import FiberForwardDataset

# 定义目标参数名称 (保持不变)
TARGET_PARAM_NAMES = [
    'Effective Index (neff_real)',
    'Effective mode area(um^2)',
    'Nonlinear coefficient(1/W/km)',
    'Dispersion (ps/nm/km)',
    'GVD(ps^2/km)'
]
PARAM_KEYS_FOR_LOG = ['neff', 'Aeff', 'NL', 'Disp', 'GVD']


def load_config(config_path="configs/default.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_path} 未找到，将使用默认参数。")
        return {}
    except Exception as e:
        print(f"加载配置文件 {config_path} 时出错: {e}")
        return {}


def load_shared_objects(path):
    try:
        shared_objects = joblib.load(path)
        print(f"成功从 {path} 加载共享 Scalers 和 Encoders。")
        if 'scalers' not in shared_objects or 'encoders' not in shared_objects:
             raise ValueError("加载的对象缺少 'scalers' 或 'encoders' 键。")
        return shared_objects['scalers'], shared_objects['encoders']
    except FileNotFoundError:
        print(f"错误: 找不到共享 Scaler/Encoder 文件: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载共享 Scaler/Encoder 文件时出错 ({path}): {e}")
        sys.exit(1)

class LitForwardWrapper(pl.LightningModule):
    def __init__(self, model_instance, train_config, num_outputs=5):
        super().__init__()
        self.model = model_instance
        self.train_config = train_config
        self.learning_rate = train_config.get('lr', 1e-3)
        self.num_outputs = num_outputs

        loss_func_name = train_config.get('loss_func', 'SmoothL1Loss') # 默认为 SmoothL1Loss
        self.loss_fn_name = loss_func_name # 保存损失函数名称，用于日志

        if loss_func_name == 'SmoothL1Loss':
            # reduction='none'以便我们可以独立处理每个参数的损失
            self.loss_calculator_internal = nn.SmoothL1Loss(reduction='none', beta=train_config.get('smooth_l1_beta', 1.0))
        elif loss_func_name == 'MSELoss':
            self.loss_calculator_internal = nn.MSELoss(reduction='none')
        elif loss_func_name == 'L1Loss':
            self.loss_calculator_internal = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的损失函数: {loss_func_name}. 请选择 'SmoothL1Loss', 'MSELoss', 或 'L1Loss'.")

        # 定义损失权重
        # 您指定的权重: [1.0, 2.0, 2.0, 1.0, 1.0]
        # 这个权重现在可以从配置文件读取，或者硬编码一个默认值

        default_weights = [1.0] # 您指定的权重

        if self.num_outputs != 5 and train_config.get('loss_weights') is None: # 如果输出不是5且配置中没有，则用等权重
            print(f"警告: 模型输出维度为 {self.num_outputs}，与默认权重长度5不符，且配置中未提供 loss_weights。将使用等权重。")
            default_weights = [1.0] * self.num_outputs

        self.loss_weights = torch.tensor(
            train_config.get('loss_weights', default_weights), # 从配置读取，否则用默认
        ) # 设备将在 _calculate_loss 中设置

        if len(self.loss_weights) != self.num_outputs:
            print(f"警告: 损失权重数量 ({len(self.loss_weights)})与输出数量 ({self.num_outputs}) 不匹配。将使用等权重。")
            self.loss_weights = torch.ones(self.num_outputs)


        hparams_to_save = {**train_config}
        if hasattr(model_instance, 'hparams') and model_instance.hparams:
             hparams_to_save.update(model_instance.hparams)
        self.save_hyperparameters(hparams_to_save, ignore=['model_instance'])


    def forward(self, data):
        return self.model(data)

    def _calculate_loss(self, predictions, targets):
        """计算损失的辅助函数"""
        # predictions: [batch_size, num_outputs]
        # targets: [batch_size, num_outputs]

        # 1. 计算每个元素的基础损失 (SmoothL1Loss, MSELoss, or L1Loss)
        # loss_elements 的形状将是 [batch_size, num_outputs]
        loss_elements = self.loss_calculator_internal(predictions, targets)

        # 2. 对 batch_size 维度求平均，得到每个参数的平均损失
        # loss_per_param 的形状将是 [num_outputs]
        loss_per_param = torch.mean(loss_elements, dim=0)

        # 3. 应用权重
        # 确保 self.loss_weights 在正确的设备上
        if self.loss_weights.device != loss_per_param.device:
            self.loss_weights = self.loss_weights.to(loss_per_param.device)

        weighted_loss = (loss_per_param * self.loss_weights).sum()
        return weighted_loss, loss_per_param # 返回总损失和每个参数的损失 (未加权)

    def _common_step(self, batch, batch_idx, stage_prefix):
        predictions = self(batch)
        targets = batch.y
        weighted_loss, loss_per_param_unweighted = self._calculate_loss(predictions, targets)

        # 当记录训练阶段的epoch级别指标时，添加 sync_dist=True
        is_train_stage_epoch_log = (stage_prefix == 'train')

        self.log(f'{stage_prefix}_loss',
                   weighted_loss,
                   on_step=(stage_prefix == 'train'),
                   on_epoch=True,
                   prog_bar=True,
                   logger=True,
                   batch_size=batch.num_graphs,
                   sync_dist=is_train_stage_epoch_log) # 应用 sync_dist

        log_loss_type_name = ""
        if self.loss_fn_name == 'SmoothL1Loss':
            log_loss_type_name = 'smoothl1loss'
        elif self.loss_fn_name == 'MSELoss':
            log_loss_type_name = 'mse'
        elif self.loss_fn_name == 'L1Loss':
            log_loss_type_name = 'l1loss'

        if log_loss_type_name:
            for i in range(self.num_outputs):
                param_key = PARAM_KEYS_FOR_LOG[i] if i < len(PARAM_KEYS_FOR_LOG) else f"param_{i}"
                self.log(f'{stage_prefix}_{log_loss_type_name}_{param_key}',
                           loss_per_param_unweighted[i],
                           on_step=False, # 通常每个参数的损失不在step级别记录
                           on_epoch=True,
                           logger=True,
                           batch_size=batch.num_graphs,
                           sync_dist=is_train_stage_epoch_log) # 应用 sync_dist
        return weighted_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def on_train_epoch_end(self):
        current_epoch_train_loss = self.trainer.logged_metrics.get('train_loss_epoch')
        if current_epoch_train_loss is not None:
            print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} - Train Loss: {current_epoch_train_loss:.6f}", end='')

    def on_validation_epoch_end(self):
        current_epoch_val_loss = self.trainer.logged_metrics.get('val_loss_epoch')
        if current_epoch_val_loss is not None:
            print(f" - Val Loss: {current_epoch_val_loss:.6f}")
        else:
            print()

    def configure_optimizers(self):
        optimizer_name = self.train_config.get('optimizer', 'Adam')
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'AdamW':
             optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        scheduler_config = self.train_config.get('lr_scheduler')
        if scheduler_config and scheduler_config.get('enabled', False):
            if scheduler_config.get('name') == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.1),
                    patience=scheduler_config.get('patience', 10),
                    verbose=True
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1,
                    }
                }
            # 可以添加其他调度器，例如 CosineAnnealingLR
            # elif scheduler_config.get('name') == 'CosineAnnealingLR':
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer,
            #         T_max=scheduler_config.get('T_max', self.trainer.max_epochs), # 通常是总epochs
            #         eta_min=scheduler_config.get('eta_min', 0)
            #     )
            #     return [optimizer], [scheduler] # 新版PL返回列表
            else:
                 print(f"警告: 不支持的学习率调度器名称 '{scheduler_config.get('name')}'")
                 return optimizer
        else:
            return optimizer


def main(args):
    # --- 加载配置 ---
    config = load_config(args.config_file)
    if not config: # 如果加载失败或为空
        print("错误：无法加载配置文件或配置文件为空。退出。")
        sys.exit(1)

    model_config = config.get('forward_model', {})
    train_config = config.get('forward_train', {})

    # --- 更新/覆盖配置 (通过命令行参数) ---
    data_root = args.data_root or train_config.get('data_root', 'core/datasets/')
    scaler_encoder_path = args.scaler_encoder_path or train_config.get('scaler_encoder_path', 'shared_scalers_encoders.pkl')
    ckpt_dir = args.ckpt_dir or train_config.get('ckpt_dir', 'checkpoints_forward/')
    log_dir = args.log_dir or train_config.get('log_dir', 'logs_forward/')
    batch_size = args.batch_size or train_config.get('batch_size', 32)
    epochs = args.epochs or train_config.get('epochs', 200)
    lr = args.lr or train_config.get('lr', 1e-3)
    num_workers = args.num_workers or train_config.get('num_workers', 4)
    seed = args.seed or train_config.get('seed', 42)
    # build_edges 和 knn_k 的读取，确保从 model_config 获取
    build_edges = args.build_edges if args.build_edges is not None else model_config.get('build_edges', True) # 默认True
    knn_k = args.knn_k or model_config.get('knn_k', 6) # 默认6 (与上面yaml示例一致)
    num_outputs = model_config.get('output_dim', 1)
    edge_input_dimension = model_config.get('edge_input_dim', 3) # <--- 从 model_config 读取


    # 更新 train_config 以便传递给 LightningModule
    train_config['lr'] = lr
    # 确保 loss_func 和 loss_weights 也在 train_config 中，如果它们是从命令行或此函数中设置的话
    train_config['loss_func'] = train_config.get('loss_func', 'SmoothL1Loss') # 确保存在
    if 'loss_weights' not in train_config: # 如果配置文件中没有，则使用您提供的默认值
        default_weights = [1.0, 2.0, 2.0, 1.0, 1.0]
        if num_outputs != 5:
             default_weights = [1.0] * num_outputs
        train_config['loss_weights'] = default_weights
    elif len(train_config['loss_weights']) != num_outputs:
        print(f"警告: 配置文件中的 loss_weights 长度与模型输出维度 ({num_outputs}) 不符。将使用等权重。")
        train_config['loss_weights'] = [1.0] * num_outputs

    # (SmoothL1Loss 的 beta 参数)
    train_config['smooth_l1_beta'] = train_config.get('smooth_l1_beta', 1.0)


    pl.seed_everything(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    shared_scalers, shared_encoders = load_shared_objects(scaler_encoder_path)
    full_dataset = FiberForwardDataset(
        root=data_root,
        shared_scalers=shared_scalers,
        shared_encoders=shared_encoders,
        build_edges=build_edges, # <--- 传递
        knn_k=knn_k              # <--- 传递
    )
    if len(full_dataset) == 0:
        print("错误：数据集为空。"); return

    # 数据划分 (与上一版保持一致)
    train_val_ratio_from_full = train_config.get('train_val_split_ratio_from_full', 0.9) # 例如 90% 用于训练+验证, 10% 测试
    test_split_ratio_actual = 1.0 - train_val_ratio_from_full

    num_total = len(full_dataset)
    num_test = int(test_split_ratio_actual * num_total)
    num_train_val = num_total - num_test

    if num_train_val <= 0 or num_test < 0: # num_test可以为0，如果不想要测试集
        print("错误：数据集太小或划分比例不当。")
        if num_test == 0: print("提示：当前配置下测试集大小为0。")
        # return # 如果不希望在测试集大小为0时退出，可以注释掉这行

    if num_test > 0:
        train_val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [num_train_val, num_test],
            generator=torch.Generator().manual_seed(seed)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0
        )
        print(f"测试集大小: {len(test_dataset)}")
    else: # 如果不划分测试集
        train_val_dataset = full_dataset
        test_loader = None # 没有测试集
        print("警告: 未划分测试集。")


    # 从 train_val_dataset 中再划分出训练集和验证集
    # split_ratio_train_in_tv 指的是训练集在 (训练集+验证集) 中的比例
    split_ratio_train_in_tv = train_config.get('split_ratio_train_in_tv', 0.8) # 例如 80% 的 train_val 用于训练
    num_train_in_tv = int(split_ratio_train_in_tv * len(train_val_dataset))
    num_val_in_tv = len(train_val_dataset) - num_train_in_tv

    if num_train_in_tv <= 0 or num_val_in_tv <= 0:
        print(f"错误：(训练+验证) 数据集 ({len(train_val_dataset)}) 太小或划分比例不当。")
        return

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [num_train_in_tv, num_val_in_tv],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"数据集大小: 总计={len(full_dataset)}, 训练={len(train_dataset)}, 验证={len(val_dataset)}")


    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    model = EquiForwardModel(
        node_feat_dim=model_config.get('num_node_features', 6),
        graph_feat_dim=model_config.get('num_graph_attributes', 5),
        output_dim=num_outputs,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 3),
        pooling_method=model_config.get('pooling', 'mean'),
        edge_input_dim = edge_input_dimension # <--- 从配置获取或用计算值
    )

    lit_model = LitForwardWrapper(model, train_config, num_outputs=num_outputs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='forward-best-{epoch}-{val_loss_epoch:.4f}', # 使用 val_loss_epoch
        save_top_k=1,
        monitor="val_loss", # 监控聚合后的验证损失
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_patience = train_config.get('early_stopping_patience', 30)
    early_stopping = EarlyStopping(
        monitor="val_loss", # 监控聚合后的验证损失
        patience=early_stopping_patience,
        mode="min",
        verbose=True
    )
    progress_bar = RichProgressBar()

    use_wandb = train_config.get('use_wandb', True)
    if use_wandb:
        # 确保 config 是一个扁平的字典或 wandb 可以接受的格式
        flat_config_for_wandb = {}
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict): # 修改: isinstance(v, MutableMapping) -> isinstance(v, dict)
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        try:
            flat_config_for_wandb = flatten_dict(config)
        except Exception as e_flat:
            print(f"将配置扁平化以用于wandb时出错: {e_flat}. 将尝试使用原始配置。")
            flat_config_for_wandb = config

        logger_tool = WandbLogger(
            project=train_config.get('wandb_project', "ForwardFiberPredictor"),
            name=train_config.get('wandb_run_name', "equi_forward_run"),
            save_dir=log_dir,
            config=flat_config_for_wandb
        )
        if hasattr(logger_tool, 'watch') and train_config.get('wandb_watch_model', False):
            logger_tool.watch(model, log='all', log_freq=train_config.get('wandb_watch_log_freq', 100))
    else:
        logger_tool = TensorBoardLogger(save_dir=log_dir, name="forward_model")


    print("配置 Trainer 以强制单GPU运行...")
    # --- 强制单GPU运行的修改 ---
    final_accelerator = 'gpu' # 直接指定 'gpu'
    final_devices = [0]       # 直接指定使用0号GPU（或者用 1 表示使用第一个可用的GPU）

    # 检查是否有可用的GPU，如果没有，则退回到CPU
    if not torch.cuda.is_available():
        print("警告: 未检测到可用CUDA设备，将强制在CPU上运行。")
        final_accelerator = 'cpu'
        final_devices = 1 # 或者 'auto' 对于CPU

    print(f"最终配置 Trainer: accelerator='{final_accelerator}', devices={final_devices}")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=final_accelerator, # 使用强制的加速器设置
        devices=final_devices,         # 使用强制的设备设置
        callbacks=[checkpoint_callback, lr_monitor, early_stopping, progress_bar],
        logger=logger_tool,
        log_every_n_steps=train_config.get('log_steps', 50),
        precision=train_config.get('precision', '32-true'),
        gradient_clip_val=train_config.get('grad_clip', 1.0),
    )
    # --- 结束 Trainer 配置修改 ---

    print("开始训练正向模型...")
    trainer.fit(lit_model, train_loader, val_loader)
    print("训练完成！")
    best_model_path = checkpoint_callback.best_model_path
    print(f"最佳模型保存在: {best_model_path}")

    if test_loader: # 仅当有测试集时才进行测试
        if best_model_path and os.path.exists(best_model_path):
            print("\n开始在测试集上评估最佳模型...")
            # 使用最佳checkpoint进行测试
            test_results = trainer.test(dataloaders=test_loader, ckpt_path=best_model_path)
            # 如果你想用fit之后的模型状态（非最佳checkpoint），则用：
            # test_results = trainer.test(lit_model, dataloaders=test_loader)

            if test_results:
                print("--- 测试集评估结果 ---")
                # 获取加权总测试损失
                test_loss_weighted = test_results[0].get('test_loss') # test_step中记录为 'test_loss'
                if test_loss_weighted is not None:
                    print(f"  加权总测试损失 (Weighted Total Test Loss): {test_loss_weighted:.6f}")

                # 打印每个参数未经加权的测试损失
                log_loss_name = ""
                if lit_model.loss_fn_name == (''
                                              ''): log_loss_name = 'smoothl1loss'
                elif lit_model.loss_fn_name == 'MSELoss': log_loss_name = 'mse'
                elif lit_model.loss_fn_name == 'L1Loss': log_loss_name = 'l1loss'

                if log_loss_name:
                    for i in range(num_outputs):
                        param_key = PARAM_KEYS_FOR_LOG[i] if i < len(PARAM_KEYS_FOR_LOG) else f"param_{i}"
                        # PL 2.0+ 在 test() 的结果中，log的key通常不会自动加 _epoch
                        param_loss_key = f'test_{log_loss_name}_{param_key}'
                        param_loss_value = test_results[0].get(param_loss_key)
                        if param_loss_value is not None:
                            print(f"  测试集 {lit_model.loss_fn_name} for {TARGET_PARAM_NAMES[i] if i < len(TARGET_PARAM_NAMES) else param_key}: {param_loss_value:.6f}")
                print("----------------------")
            else:
                print("未能获取测试结果。")
        else:
            print("未找到最佳模型检查点或测试加载器为空，跳过测试集评估。")
    else:
        print("没有测试集，跳过测试集评估。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练正向光纤属性预测模型")
    parser.add_argument('--config_file', type=str, default='configs/default.yaml', help='主配置文件路径')
    parser.add_argument('--data_root', type=str, default=None, help='数据集根目录 (覆盖配置)')
    parser.add_argument('--scaler_encoder_path', type=str, default=None, help='共享 Scaler/Encoder 文件路径 (覆盖配置)')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='检查点保存目录 (覆盖配置)')
    parser.add_argument('--log_dir', type=str, default=None, help='日志保存目录 (覆盖配置)')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小 (覆盖配置)')
    parser.add_argument('--epochs', type=int, default=None, help='最大训练轮数 (覆盖配置)')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (覆盖配置)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载 worker 数量 (覆盖配置)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (覆盖配置)')
    parser.add_argument('--build_edges', action=argparse.BooleanOptionalAction, default=None, help='是否在数据集中构建边 (覆盖配置)')
    parser.add_argument('--knn_k', type=int, default=None, help='如果构建边，KNN 的 k 值 (覆盖配置)')
    parser.add_argument('--accelerator', type=str, default=None, help='训练加速器 (覆盖配置: "cpu", "gpu", "auto")')
    parser.add_argument('--devices', default=None, help='使用的设备 (覆盖配置: "auto", 1, [0,1], "0,1", or number of devices as string like "2")')

    args = parser.parse_args()
    main(args)