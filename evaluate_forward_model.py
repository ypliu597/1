# Suggested filename: evaluate_forward_model.py

import os
import sys
import glob
import yaml
import joblib
import argparse
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns  # 可选，用于更好看的图
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split  # 用于获取测试集
import pytorch_lightning as pl

# --- 确保能找到 core 包 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# --- --- --- --- --- --- --

from core.models.equi_forward_model import EquiForwardModel  # 导入您的正向模型
from core.datasets.forward_dataset import FiberForwardDataset  # 导入对齐后的数据集
from train_forward_model import load_shared_objects, LitForwardWrapper, load_config  # 从训练脚本导入辅助函数和包装器

TARGET_PARAM_NAMES = [
    'Effective Index (neff_real)',  # 确保与您的 CSV 和预处理脚本中的定义匹配
    'Effective mode area(um^2)',  # 确保与您的 CSV 和预处理脚本中的定义匹配
    'Nonlinear coefficient(1/W/km)',  # 确保与您的 CSV 和预处理脚本中的定义匹配
    'Dispersion (ps/nm/km)',  # 确保与您的 CSV 和预处理脚本中的定义匹配
    'GVD(ps^2/km)'  # 确保与您的 CSV 和预处理脚本中的定义匹配
]


def denormalize_predictions(predictions_norm, targets_norm, target_scaler):
    """使用共享的 target_scaler 反归一化预测值和目标值"""
    if target_scaler is None or not hasattr(target_scaler, 'data_max_') or target_scaler.data_max_ is None:
        print("警告: Target scaler 未提供或未拟合，无法进行反归一化。返回原始值。")
        return predictions_norm, targets_norm

    pred_shape_orig = predictions_norm.shape
    target_shape_orig = targets_norm.shape

    if predictions_norm.ndim == 1: predictions_norm = predictions_norm.reshape(-1, 1)
    if targets_norm.ndim == 1: targets_norm = targets_norm.reshape(-1, 1)

    if predictions_norm.shape[1] != target_scaler.n_features_in_:
        print(
            f"警告: 预测值特征数 ({predictions_norm.shape[1]}) 与 scaler 期望特征数 ({target_scaler.n_features_in_}) 不符。尝试使用前 N 列。")
        predictions_denorm = target_scaler.inverse_transform(predictions_norm[:, :target_scaler.n_features_in_])
    else:
        predictions_denorm = target_scaler.inverse_transform(predictions_norm)

    if targets_norm.shape[1] != target_scaler.n_features_in_:
        print(
            f"警告: 目标值特征数 ({targets_norm.shape[1]}) 与 scaler 期望特征数 ({target_scaler.n_features_in_}) 不符。尝试使用前 N 列。")
        targets_denorm = target_scaler.inverse_transform(targets_norm[:, :target_scaler.n_features_in_])
    else:
        targets_denorm = target_scaler.inverse_transform(targets_norm)

    return predictions_denorm.reshape(pred_shape_orig), targets_denorm.reshape(target_shape_orig)


def perform_evaluation_on_subset(
        model,
        data_loader,
        dataset_subset,  # torch.utils.data.Subset
        dataset_name,  # "train" or "eval"
        shared_scalers,
        output_param_names,
        output_dir_subset,
        device,
        config  # for model_config, TARGET_PARAM_NAMES fallback
):
    """
    对给定的数据集子集执行评估、可视化和数据保存。
    """
    model_config = config.get('forward_model', {})
    target_scaler = shared_scalers.get('target')

    print(f"\n--- 开始在 {dataset_name} 集上进行评估和可视化 ---")
    os.makedirs(output_dir_subset, exist_ok=True)

    # --- 进行预测 ---
    all_predictions_norm = []
    all_targets_norm = []

    print(f"开始在 {dataset_name} 集上进行预测...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Predicting on {dataset_name} set"):
            batch = batch.to(device)
            predictions_norm_batch = model(batch)
            all_predictions_norm.append(predictions_norm_batch.cpu().numpy())
            all_targets_norm.append(batch.y.cpu().numpy())

    all_predictions_norm = np.concatenate(all_predictions_norm, axis=0)
    all_targets_norm = np.concatenate(all_targets_norm, axis=0)

    # --- 反归一化 ---
    print(f"对 {dataset_name} 集结果进行反归一化...")
    all_predictions_denorm, all_targets_denorm = denormalize_predictions(
        all_predictions_norm, all_targets_norm, target_scaler
    )
    if target_scaler is None or not hasattr(target_scaler, 'data_max_') or target_scaler.data_max_ is None:
        print(f"警告: Target Scaler 不可用或未拟合，{dataset_name} 集结果将是归一化后的值。")

    # --- 计算评估指标 ---
    current_output_param_names = model_config.get('output_param_names',
                                                  output_param_names)  # Use specific from config if available
    if len(current_output_param_names) != all_predictions_denorm.shape[1]:
        print(
            f"警告: output_param_names 长度 ({len(current_output_param_names)}) 与预测维度 ({all_predictions_denorm.shape[1]}) 不符。将使用通用名称。")
        current_output_param_names = [f"Param_{i + 1}" for i in range(all_predictions_denorm.shape[1])]

    print(f"\n--- {dataset_name.capitalize()} Set 评估结果 (反归一化后) ---")
    metrics_summary = {}
    for i in range(all_predictions_denorm.shape[1]):
        param_name = current_output_param_names[i]
        pred_col = all_predictions_denorm[:, i]
        target_col = all_targets_denorm[:, i]

        mse = mean_squared_error(target_col, pred_col)
        mae = mean_absolute_error(target_col, pred_col)
        r2 = r2_score(target_col, pred_col)
        metrics_summary[param_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"参数 ({dataset_name}): {param_name}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R² : {r2:.6f}")
        print("-" * 30)

    metrics_df = pd.DataFrame.from_dict(metrics_summary, orient='index')
    metrics_file_path = os.path.join(output_dir_subset, f"{dataset_name}_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_file_path)
    print(f"{dataset_name.capitalize()} 集评估指标已保存到: {metrics_file_path}")

    # --- 可视化 (散点图和残差图) ---
    print(f"\n开始为 {dataset_name} 集绘制可视化图表...")
    sns.set_theme(style="whitegrid")

    for i in range(all_predictions_denorm.shape[1]):
        param_name = current_output_param_names[i]
        plt.figure(figsize=(12, 5))  # Adjusted for two subplots

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=all_targets_denorm[:, i], y=all_predictions_denorm[:, i], alpha=0.6,
                        s=30)  # s for marker size
        min_val = min(all_targets_denorm[:, i].min(), all_predictions_denorm[:, i].min())
        max_val = max(all_targets_denorm[:, i].max(), all_predictions_denorm[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x (Ideal)')
        plt.xlabel(f"真实值 - {param_name}")
        plt.ylabel(f"预测值 - {param_name}")
        plt.title(f"{dataset_name.capitalize()} Set: {param_name}\n真实值 vs. 预测值")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        residuals = all_targets_denorm[:, i] - all_predictions_denorm[:, i]
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel("残差 (真实值 - 预测值)")
        plt.ylabel("频数")
        plt.title(f"{dataset_name.capitalize()} Set: {param_name}\n残差分布")
        plt.axvline(0, color='r', linestyle='--', lw=2)
        plt.grid(True)

        plt.tight_layout()
        fig_path = os.path.join(output_dir_subset,
                                f"{dataset_name}_scatter_residual_{param_name.replace('/', '_').replace(' ', '_')}.png")
        plt.savefig(fig_path)
        plt.close()
    print(f"{dataset_name.capitalize()} 集散点图和残差图已保存到目录: {output_dir_subset}")

    # --- 参数 vs 波长 对比图 和 CSV 数据保存 ---
    print(f"\n开始为 {dataset_name} 集的单个样本绘制 参数 vs 波长 对比图...")
    if not dataset_subset.indices:
        print(f"{dataset_name.capitalize()} 集为空，无法绘制 参数 vs 波长 图。")
        return

    # 选择第一个样本进行可视化
    first_sample_master_idx = dataset_subset.indices[0]
    # .dataset gives access to the original FiberForwardDataset
    example_file_path, _ = dataset_subset.dataset.samples[first_sample_master_idx]
    example_file_basename = os.path.basename(example_file_path)
    print(f"选择 {dataset_name} 集样本文件进行可视化: {example_file_basename}")

    example_indices_in_subset_results = []  # Indices in all_predictions_denorm for this file
    example_master_indices_for_wl = []  # Master indices in full_dataset for this file

    for i in range(len(dataset_subset)):  # i is index within the subset
        master_idx = dataset_subset.indices[i]  # master_idx is index within the full_dataset
        fp, _ = dataset_subset.dataset.samples[master_idx]
        if fp == example_file_path:
            example_indices_in_subset_results.append(i)
            example_master_indices_for_wl.append(master_idx)

    if not example_indices_in_subset_results:
        print(f"错误：在 {dataset_name} 集结果中未能找到文件 {example_file_basename} 的任何数据点。")
        return

    print(f"找到该文件在 {dataset_name} 集中的 {len(example_indices_in_subset_results)} 个数据点 (不同波长)。")

    example_preds_denorm = all_predictions_denorm[example_indices_in_subset_results]
    example_targets_denorm = all_targets_denorm[example_indices_in_subset_results]

    example_wavelengths_norm_list = []
    for master_idx in example_master_indices_for_wl:
        data_item = dataset_subset.dataset[master_idx]  # Access item from the original full dataset
        norm_wl = data_item.graph_features[0, 0].item()
        example_wavelengths_norm_list.append(norm_wl)
    example_wavelengths_norm_np = np.array(example_wavelengths_norm_list).reshape(-1, 1)

    wavelength_scaler = shared_scalers.get('wavelength')
    if wavelength_scaler is not None and hasattr(wavelength_scaler,
                                                 'data_max_') and wavelength_scaler.data_max_ is not None:
        example_wavelengths_denorm = wavelength_scaler.inverse_transform(example_wavelengths_norm_np).flatten()
    else:
        print(f"警告: Wavelength scaler 不可用或未拟合，{dataset_name} 集 X轴将使用归一化后的波长。")
        example_wavelengths_denorm = example_wavelengths_norm_np.flatten()

    sort_indices = np.argsort(example_wavelengths_denorm)
    sorted_wavelengths = example_wavelengths_denorm[sort_indices]
    sorted_preds = example_preds_denorm[sort_indices]
    sorted_targets = example_targets_denorm[sort_indices]

    # --- 保存绘图数据到 CSV ---
    plot_data_df_dict = {'Wavelength (nm)': sorted_wavelengths}
    for i in range(sorted_preds.shape[1]):
        param_name = current_output_param_names[i]
        clean_param_name_col = param_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(
            '^', '').replace('*', '')
        plot_data_df_dict[f'True - {clean_param_name_col}'] = sorted_targets[:, i]
        plot_data_df_dict[f'Predicted - {clean_param_name_col}'] = sorted_preds[:, i]

    plot_data_df = pd.DataFrame(plot_data_df_dict)
    csv_data_filename = f"{dataset_name}_param_vs_wavelength_data_{example_file_basename.replace('.csv', '')}.csv"
    csv_data_path = os.path.join(output_dir_subset, csv_data_filename)
    plot_data_df.to_csv(csv_data_path, index=False)
    print(f"参数 vs 波长绘图数据已保存到: {csv_data_path}")

    # --- 绘图 (每个参数一张图) ---
    num_params_to_plot = sorted_preds.shape[1]
    plot_filename_prefix = f"{dataset_name}_param_vs_wavelength_{example_file_basename.replace('.csv', '')}"

    for i in range(num_params_to_plot):
        param_name = current_output_param_names[i]
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_wavelengths, sorted_targets[:, i], 'bo-', label='真实值 (True)', linewidth=2, markersize=5)
        plt.plot(sorted_wavelengths, sorted_preds[:, i], 'rx--', label='预测值 (Predicted)', linewidth=2, markersize=5)
        plt.xlabel(f"波长 (nm)")
        plt.ylabel(f"{param_name}")
        plt.title(f"{param_name} vs. 波长 ({dataset_name.capitalize()} Set)\n(样本: {example_file_basename})")
        plt.legend()
        plt.grid(True)
        clean_param_name_file = param_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')',
                                                                                                        '').replace('^',
                                                                                                                    '').replace(
            '*', '')
        fig_path = os.path.join(output_dir_subset, f"{plot_filename_prefix}_param_{i + 1}_{clean_param_name_file}.png")
        plt.savefig(fig_path)
        plt.close()
    print(f"{dataset_name.capitalize()} 集参数 vs 波长对比图已保存到目录: {output_dir_subset}")


def evaluate_and_visualize_main(args):
    print("开始评估和可视化正向模型...")

    config = load_config(args.config_file)
    model_config = config.get('forward_model', {})
    train_config = config.get('forward_train', {})
    data_config = config.get('data', {})

    data_root = args.data_root or train_config.get('data_root', 'core/datasets/')
    scaler_encoder_path = args.scaler_encoder_path or data_config.get('scaler_encoder_path')
    if not scaler_encoder_path:
        print(
            "错误: 必须提供共享 scaler/encoder 文件路径 (--scaler_encoder_path 或在配置文件中 data.scaler_encoder_path)。")
        sys.exit(1)

    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size or train_config.get('batch_size', 32)
    num_workers = args.num_workers or train_config.get('num_workers', 0)
    seed = args.seed or train_config.get('seed', 42)
    build_edges = args.build_edges if args.build_edges is not None else model_config.get('build_edges', False)
    knn_k = args.knn_k or model_config.get('knn_k', 8)

    base_output_dir = args.output_dir  # Main output directory for all results
    # os.makedirs(base_output_dir, exist_ok=True) # Will be created by sub-functions if needed

    pl.seed_everything(seed)

    print(f"加载共享对象从: {scaler_encoder_path}")
    shared_scalers, shared_encoders = load_shared_objects(scaler_encoder_path)

    # --- Global TARGET_PARAM_NAMES from config or default ---
    global_output_param_names = model_config.get('output_param_names', TARGET_PARAM_NAMES)

    print(f"加载数据集从: {data_root}")
    full_dataset = FiberForwardDataset(
        root=data_root,
        shared_scalers=shared_scalers,
        shared_encoders=shared_encoders,
        build_edges=build_edges,
        knn_k=knn_k
    )
    if len(full_dataset) == 0:
        print("错误：数据集为空。")
        return

    # Split into train and validation sets
    train_val_ratio = train_config.get('split_ratio', 0.8)
    num_total = len(full_dataset)
    num_train = int(train_val_ratio * num_total)

    # Ensure consistent split for train and validation
    all_indices = list(range(num_total))
    train_dataset_indices, val_dataset_indices = train_test_split(
        all_indices,
        train_size=num_train,  # or train_val_ratio
        random_state=seed,
        shuffle=True  # Shuffle before splitting
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_dataset_indices)
    eval_dataset = torch.utils.data.Subset(full_dataset, val_dataset_indices)  # This is the validation set

    print(f"训练集大小: {len(train_dataset)}")
    print(f"评估集 (验证集) 大小: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"加载模型检查点从: {checkpoint_path}")
    trained_model = EquiForwardModel(
        node_feat_dim=model_config.get('num_node_features', 5),
        graph_feat_dim=model_config.get('num_graph_attributes', 5),
        output_dim=model_config.get('output_dim', 1),  # Should match len(TARGET_PARAM_NAMES)
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 3),
        pooling_method=model_config.get('pooling', 'mean')
    )
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items() if
                        k.startswith("model.")}
    trained_model.load_state_dict(model_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    # --- Evaluate on Training Set ---
    train_output_dir = os.path.join(base_output_dir, "train_set_results")
    perform_evaluation_on_subset(
        model=trained_model,
        data_loader=train_loader,
        dataset_subset=train_dataset,
        dataset_name="train",
        shared_scalers=shared_scalers,
        output_param_names=global_output_param_names,
        output_dir_subset=train_output_dir,
        device=device,
        config=config
    )

    # --- Evaluate on Evaluation (Validation) Set ---
    eval_output_dir = os.path.join(base_output_dir, "eval_set_results")
    perform_evaluation_on_subset(
        model=trained_model,
        data_loader=eval_loader,
        dataset_subset=eval_dataset,
        dataset_name="eval",  # "eval" or "validation"
        shared_scalers=shared_scalers,
        output_param_names=global_output_param_names,
        output_dir_subset=eval_output_dir,
        device=device,
        config=config
    )
    print(f"\n所有评估结果和图表已保存到基础目录: {base_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估和可视化训练好的正向模型")
    parser.add_argument('--config_file', type=str, default='configs/default.yaml', help='主配置文件路径')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints_forward/forward-best-epoch=96-val_loss_epoch=0.0000.ckpt',
                        help='训练好的正向模型检查点 (.ckpt) 文件路径。')
    parser.add_argument('--data_root', type=str, default=None, help='数据集根目录 (覆盖配置)。')
    parser.add_argument('--scaler_encoder_path', type=str, default=None,
                        help='共享 Scaler/Encoder 文件路径 (覆盖配置)。')
    parser.add_argument('--batch_size', type=int, default=None, help='评估时的批处理大小 (覆盖配置)。')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载 worker 数量 (覆盖配置)。')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (覆盖配置)。')
    parser.add_argument('--build_edges', action=argparse.BooleanOptionalAction, default=None,
                        help='是否在数据集中构建边 (覆盖配置)。')
    parser.add_argument('--knn_k', type=int, default=None, help='如果构建边，KNN 的 k 值 (覆盖配置)。')
    parser.add_argument('--output_dir', type=str, default='evaluation_forward_all_sets/',
                        help='保存所有评估结果和图表的基目录。')

    args = parser.parse_args()
    evaluate_and_visualize_main(args)