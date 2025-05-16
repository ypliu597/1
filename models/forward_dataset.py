import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
import os
import glob
import numpy as np
from torch_geometric.nn import knn_graph

# --- 定义CSV列名常量 ---
COL_X_COORD = 'X (um)'
COL_Y_COORD = 'Y (um)'
# COL_RADIAL_DIST = 'R (um)' # This parameter will not be used as a node feature
COL_RADIUS_1 = 'Air Hole Radius (um)'
COL_RADIUS_2 = 'Air Hole Radius2 (um)'
COL_SHAPE = 'air_hole_shape'
COL_FIBER_RADIUS = 'Fiber Radius (um)'
COL_MATERIAL = 'Fiber Material'
COL_WAVELENGTH = 'Wavelength (nm)'  # 与 preprocess 一致，使用 COL_WAVELENGTH_NM
TARGET_PARAM_NAMES = [
    'Effective Index (neff_real)',
    'Effective mode area(um^2)',
    'Nonlinear coefficient(1/W/km)',
    'Dispersion (ps/nm/km)',
    'GVD(ps^2/km)'
]
# --- 结束常量定义 ---

# 定义 NaN 占位符 (与 preprocess_scalers_encoders.py 中一致)
UNKNOWN_MATERIAL_PLACEHOLDER = 'UNKNOWN_MATERIAL'
UNKNOWN_SHAPE_PLACEHOLDER = 'UNKNOWN_SHAPE'


def detect_structure_rows_by_nan(df):
    try:
        x_col = df.iloc[:, 0]
        y_col = df.iloc[:, 1]
    except IndexError:
        return 0
    for i in range(len(x_col)):
        if pd.isna(x_col[i]) or pd.isna(y_col[i]):
            return i
    return len(x_col)


def create_no_edge_graph(num_nodes):
    return torch.empty((2, 0), dtype=torch.long)


class FiberForwardDataset(Dataset):
    def __init__(self, root, shared_scalers, shared_encoders, transform=None, pre_transform=None, use_column_names=True,
                 build_edges=True, knn_k=6):
        super().__init__(root, transform, pre_transform)
        self.file_list = sorted(glob.glob(os.path.join(self.root, "*.csv")))
        self.use_column_names = use_column_names
        self.build_edges = build_edges
        self.knn_k = knn_k

        if not all(k in shared_scalers for k in ['node_coord', 'hole_radius', 'fiber_radius', 'wavelength', 'target']):
            raise ValueError("shared_scalers dictionary is missing required keys or keys are None.")
        if not all(k in shared_encoders for k in ['shape', 'material']):
            raise ValueError("shared_encoders dictionary is missing required keys or keys are None.")

        self.scalers = shared_scalers
        self.encoders = shared_encoders

        self.samples = []
        for file_path in self.file_list:
            try:
                df_full_for_rows = pd.read_csv(file_path)
                if df_full_for_rows.empty:
                    print(f"警告: 文件 {file_path} 为空。跳过。")
                    continue

                required_cols_check_ds = [COL_X_COORD, COL_Y_COORD, COL_RADIUS_1, COL_SHAPE,
                                          COL_MATERIAL, COL_FIBER_RADIUS, COL_WAVELENGTH] + TARGET_PARAM_NAMES
                if not all(col in df_full_for_rows.columns for col in required_cols_check_ds):
                    missing_cols_ds = [col for col in required_cols_check_ds if col not in df_full_for_rows.columns]
                    print(f"警告: 文件 {file_path} 缺少必要的列: {missing_cols_ds}。跳过此文件用于样本生成。")
                    continue

                num_structure_rows = detect_structure_rows_by_nan(df_full_for_rows)
                num_total_rows_in_file = len(df_full_for_rows)
                num_condition_rows_in_file = num_total_rows_in_file - num_structure_rows

                if num_condition_rows_in_file <= 0:
                    continue
            except pd.errors.EmptyDataError:
                continue
            except Exception as e:
                continue

            for cond_idx_offset in range(num_condition_rows_in_file):
                actual_cond_idx_in_df = num_structure_rows + cond_idx_offset
                if actual_cond_idx_in_df < num_total_rows_in_file:
                    self.samples.append((file_path, actual_cond_idx_in_df))

        if not self.file_list: print("警告: 在指定的根目录中没有找到CSV文件 (ForwardDataset)。")
        if not self.samples: print("警告: 没有加载任何样本 (ForwardDataset)。请检查数据集路径和CSV文件。")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        file_path, cond_idx = self.samples[idx]
        try:
            df = pd.read_csv(file_path)
            if df.empty: raise ValueError(f"ForwardDataset: CSV文件为空: {file_path}")
        except Exception as e:
            print(f"ForwardDataset: 读取文件 {file_path} (索引 {idx}) 时出错。错误: {e}")
            # node_feat_dim now accounts for 2 radius features + shape features
            node_feat_dim = 2 + (self.encoders['shape'].categories_[0].shape[0] if 'shape' in self.encoders and hasattr(
                self.encoders['shape'], 'categories_') else 3) # Default to 3 shape features if encoder not detailed
            graph_feat_dim = 1 + self.encoders['material'].categories_[0].shape[
                0] + 1 if 'material' in self.encoders and hasattr(self.encoders['material'], 'categories_') else 5
            target_dim = len(TARGET_PARAM_NAMES) if TARGET_PARAM_NAMES else 1

            return Data(x=torch.empty(0, node_feat_dim), pos=torch.empty(0, 2),
                        edge_index=create_no_edge_graph(0), edge_attr=torch.empty(0, 3),
                        y=torch.empty(0, target_dim), graph_features=torch.empty(0, graph_feat_dim),
                        num_nodes=0)

        structure_rows = detect_structure_rows_by_nan(df)

        # --- 提取孔洞特征 (x) 和位置 (pos) ---
        if structure_rows > 0:
            hole_pos_raw_np = df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values.astype(np.float32)
            if hasattr(self.scalers['node_coord'], 'data_max_') and self.scalers['node_coord'].data_max_ is not None:
                pos = torch.tensor(self.scalers['node_coord'].transform(hole_pos_raw_np), dtype=torch.float32)
            else:
                print(f"警告 FD get: 共享 node_coord_scaler 未拟合。对 {file_path} 使用原始坐标。")
                pos = torch.tensor(hole_pos_raw_np, dtype=torch.float32)

            radius_cols_to_use = [COL_RADIUS_1]
            if COL_RADIUS_2 in df.columns and not df.loc[:structure_rows - 1, COL_RADIUS_2].isna().all():
                radius_cols_to_use.append(COL_RADIUS_2)
            hole_radius_attr_raw_np = df.loc[:structure_rows - 1, radius_cols_to_use].values.astype(np.float32)
            if hole_radius_attr_raw_np.shape[1] == 1:
                hole_radius_attr_raw_np = np.hstack([hole_radius_attr_raw_np, hole_radius_attr_raw_np])
            hole_radius_attr_raw_np = np.nan_to_num(hole_radius_attr_raw_np, nan=0.0)

            if hasattr(self.scalers['hole_radius'], 'data_max_') and self.scalers['hole_radius'].data_max_ is not None:
                hole_radius_attr_scaled_np = self.scalers['hole_radius'].transform(hole_radius_attr_raw_np)
            else:
                print(f"警告 FD get: 共享 hole_radius_scaler 未拟合。对 {file_path} 使用原始半径。")
                hole_radius_attr_scaled_np = hole_radius_attr_raw_np
            hole_radius_attr_scaled = torch.tensor(hole_radius_attr_scaled_np, dtype=torch.float32)

            shape_attr_list = []
            for i in range(structure_rows):
                shape_val_raw = df.loc[i, COL_SHAPE]
                if pd.isna(shape_val_raw):
                    shape_val_for_transform = UNKNOWN_SHAPE_PLACEHOLDER
                else:
                    shape_val_for_transform = str(shape_val_raw).strip()
                transformed_shape = self.encoders['shape'].transform([[shape_val_for_transform]])
                shape_attr_list.append(transformed_shape[0])
            shape_attr_encoded = torch.tensor(np.array(shape_attr_list), dtype=torch.float32)

            # Node features are concatenation of hole radius (2 features) and shape (one-hot encoded)
            # COL_RADIAL_DIST is no longer used here.
            node_features_x = torch.cat([hole_radius_attr_scaled, shape_attr_encoded], dim=-1)

        else: # No structure rows
            pos = torch.empty((0, 2), dtype=torch.float32)
            # node_feat_dim_actual calculates 2 (for radii) + num_shape_categories
            node_feat_dim_actual = 2 + (
                self.encoders['shape'].categories_[0].shape[0] if 'shape' in self.encoders and hasattr(
                    self.encoders['shape'], 'categories_') else 3) # Default to 3 shape features if encoder not detailed
            node_features_x = torch.empty((0, node_feat_dim_actual), dtype=torch.float32)

        # --- 提取全局/图级别特征 (graph_features) ---
        material_ref_row = 0
        if structure_rows == 0 and len(df) > 0:
            pass
        elif structure_rows > 0:
            material_ref_row = 0
        elif len(df) == 0:
            print(f"错误: 文件 {file_path} 为空，无法提取材料。")
            material_attr_encoded = torch.zeros(
                self.encoders['material'].categories_[0].shape[0] if 'material' in self.encoders and hasattr(
                    self.encoders['material'], 'categories_') else 3, dtype=torch.float32)

        if material_ref_row < len(df):
            material_val_raw = df.loc[material_ref_row, COL_MATERIAL]
            if pd.isna(material_val_raw):
                material_val_for_transform = UNKNOWN_MATERIAL_PLACEHOLDER
            else:
                material_val_for_transform = str(material_val_raw).strip()
            transformed_material = self.encoders['material'].transform([[material_val_for_transform]])
            material_attr_encoded = torch.tensor(transformed_material[0], dtype=torch.float32)
        else:
            print(f"警告: 文件 {file_path} 行数不足以在索引 {material_ref_row} 处提取材料。使用占位符。")
            transformed_material = self.encoders['material'].transform([[UNKNOWN_MATERIAL_PLACEHOLDER]])
            material_attr_encoded = torch.tensor(transformed_material[0], dtype=torch.float32)

        wavelength_raw_val = df.loc[cond_idx, COL_WAVELENGTH]
        wavelength_raw_np = np.array([[wavelength_raw_val]]).astype(np.float32)
        if hasattr(self.scalers['wavelength'], 'data_max_') and self.scalers['wavelength'].data_max_ is not None:
            wavelength_scaled_np = self.scalers['wavelength'].transform(wavelength_raw_np)[0]
        else:
            print(f"警告 FD get: 共享 wavelength_scaler 未拟合。对 {file_path} 使用原始波长。")
            wavelength_scaled_np = wavelength_raw_np.flatten()
        wavelength_scaled = torch.tensor(wavelength_scaled_np, dtype=torch.float32)

        fiber_radius_ref_row = 0
        if structure_rows == 0 and len(df) > 0:
            pass
        elif structure_rows > 0:
            fiber_radius_ref_row = 0
        elif len(df) == 0:
            print(f"错误: 文件 {file_path} 为空，无法提取光纤半径。")
            fiber_radius_scaled = torch.tensor([0.0], dtype=torch.float32)

        if fiber_radius_ref_row < len(df):
            fiber_radius_raw_val = df.loc[fiber_radius_ref_row, COL_FIBER_RADIUS]
            fiber_radius_raw_np = np.array(
                [[float(fiber_radius_raw_val) if pd.notna(fiber_radius_raw_val) else 0.0]]).astype(np.float32)
            if hasattr(self.scalers['fiber_radius'], 'data_max_') and self.scalers[
                'fiber_radius'].data_max_ is not None:
                fiber_radius_scaled_np = self.scalers['fiber_radius'].transform(fiber_radius_raw_np)[0]
            else:
                print(f"警告 FD get: 共享 fiber_radius_scaler 未拟合。对 {file_path} 使用原始光纤半径。")
                fiber_radius_scaled_np = fiber_radius_raw_np.flatten()
            fiber_radius_scaled = torch.tensor(fiber_radius_scaled_np, dtype=torch.float32)
        else:
            print(f"警告: 文件 {file_path} 行数不足以在索引 {fiber_radius_ref_row} 处提取光纤半径。使用0。")
            fiber_radius_scaled = torch.tensor([0.0], dtype=torch.float32)

        graph_features = torch.cat([
            wavelength_scaled.view(1),
            material_attr_encoded,
            fiber_radius_scaled.view(1)
        ], dim=0).unsqueeze(0)

        # --- 提取目标输出 (y) ---
        if not TARGET_PARAM_NAMES:
            print(f"错误: TARGET_PARAM_NAMES 为空 (Dataset)。文件 {file_path}, 条件索引 {cond_idx}")
            target_y = torch.empty((1, 0), dtype=torch.float32)
        else:
            target_raw_np_list = []
            for target_col_name in TARGET_PARAM_NAMES:
                if target_col_name in df.columns and cond_idx < len(df):
                    val = df.loc[cond_idx, target_col_name]
                    target_raw_np_list.append(float(val) if pd.notna(val) else 0.0)
                else:
                    print(f"警告: 目标列 '{target_col_name}' 或索引 {cond_idx} 在 {file_path} 中无效。使用0填充。")
                    target_raw_np_list.append(0.0)
            target_raw_np = np.array(target_raw_np_list).astype(np.float32)

            if hasattr(self.scalers['target'], 'data_max_') and self.scalers['target'].data_max_ is not None:
                target_scaled_np = self.scalers['target'].transform(target_raw_np.reshape(1, -1))[0]
            else:
                print(f"警告 FD get: 共享 target_scaler 未拟合。对 {file_path} 使用原始目标值。")
                target_scaled_np = target_raw_np
            target_y = torch.tensor(target_scaled_np, dtype=torch.float32).unsqueeze(0)

        # --- 构建边信息 (edge_index 和 edge_attr) ---
        edge_index = create_no_edge_graph(pos.size(0))
        # edge_attr = torch.empty((0, 3), dtype=torch.float32)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        if self.build_edges and pos.size(0) > 1:
            actual_k = min(self.knn_k, pos.size(0) - 1)
            if actual_k > 0 :
                 edge_index = knn_graph(pos, k=actual_k, batch=None, loop=False)
                 if edge_index.numel() > 0:
                     row, col = edge_index
                     # 计算相对位移用于计算距离，但不直接作为边属性
                     rel_pos = pos[col] - pos[row] # x_i - x_j
                     dist = torch.norm(rel_pos, p=2, dim=-1, keepdim=True) # [E, 1]
                     edge_attr = dist # <--- 只使用距离作为边属性
                 # else: edge_attr 保持为空 [0,1]
            # else: k<=0, edge_attr 保持为空 [0,1]


        data = Data(
            x=node_features_x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target_y,
            graph_features=graph_features,
            num_nodes=pos.size(0)
        )

        if self.transform:
            data = self.transform(data)
        return data