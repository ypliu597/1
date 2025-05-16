import torch
import re
import pandas as pd
from torch_geometric.data import Dataset, Data
import os
import glob
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

# 定义CSV列名常量 (请根据您的实际CSV表头修改这些常量)
# 如果CSV没有表头，您可能仍然希望用常量来表示数字索引以提高可读性
# 例如：COL_X_COORD = 0, COL_Y_COORD = 1, 等。
# 这里我们假设有表头，并且使用表头名。
# 如果您的CSV文件确实没有表头，请将下面的df.loc[...]替换回df.iloc[...]，并确保数字索引正确。

# 结构部分
COL_X_COORD = 'X (um)'  # 第1列 (索引0)
COL_Y_COORD = 'Y (um)'  # 第2列 (索引1)
# COL_UNUSED_1 = 'R (um)'    # 第3列 (索引2) - 假设未使用
# COL_UNUSED_2 = 'Theta (rad)'    # 第4列 (索引3) - 假设未使用
COL_RADIUS_1 = 'Air Hole Radius (um)'  # 第5列 (索引4)
COL_RADIUS_2 = 'Air Hole Radius2 (um)'  # 第6列 (索引5)
COL_SHAPE = 'air_hole_shape'  # 第7列 (索引6)
COL_FIBER_RADIUS = 'Fiber Radius (um)'  # 第8列 (索引7)
COL_MATERIAL = 'Fiber Material'  # 第9列 (索引8)
# COL_UNUSED_3 = 'Distance Between Holes (um)'    # 第10列 (索引9) - 假设未使用
# COL_UNUSED_4 = 'fiber_core_radius(um)'    # 第11列 (索引10) - 假设未使用

# 条件参数部分
COL_WAVELENGTH = 'Wavelength (nm)'  # 第12列 (索引11)
# 目标参数从第13列到第17列 (索引12到16)
TARGET_PARAM_COLS = [f'target_param_{i}' for i in range(1, 6)]  # 假设列名为 target_param_1, ..., target_param_5


# 您需要根据您CSV文件的实际列名来调整上面的常量
# 例如，如果您的CSV没有表头，并且原始代码的数字索引是正确的：
# COL_X_COORD_IDX = 0
# COL_Y_COORD_IDX = 1
# COL_RADIUS_1_IDX = 4
# COL_RADIUS_2_IDX = 5
# COL_SHAPE_IDX = 6
# COL_FIBER_RADIUS_IDX = 7
# COL_MATERIAL_IDX = 8
# COL_WAVELENGTH_IDX = 11
# TARGET_PARAM_START_IDX = 12
# TARGET_PARAM_END_IDX = 17


def detect_structure_rows_by_nan(df):
    """
    检测每张图的节点数。
    根据前两列（默认为x, y坐标）出现NaN的位置来判断节点区域结束。
    注意：如果使用列名，需要确保这里访问的是正确的列。
    为了保持原函数的行为，如果CSV没有header，前两列是索引0和1。
    如果CSV有header，需要传递列名给这个函数或修改它以使用列名常量。
    此处暂时保留原实现，假设访问的是坐标列。
    """
    # 如果CSV有header, 确保这两列是坐标列
    # x_col = df[COL_X_COORD]
    # y_col = df[COL_Y_COORD]
    # 如果CSV无header，且坐标在前两列:
    x_col = df.iloc[:, 0]
    y_col = df.iloc[:, 1]

    for i in range(len(x_col)):
        if pd.isna(x_col[i]) or pd.isna(y_col[i]):
            return i
    return len(x_col)  # 如果没有遇到NaN，说明全部是节点


class FiberInverseDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, use_column_names=False):  # 新增 use_column_names
        super().__init__(root, transform, pre_transform)
        self.file_list = sorted(glob.glob(os.path.join(self.root, "*.csv")))
        self.use_column_names = use_column_names  # 如果为True，则使用定义的列名常量

        # OneHot编码器
        self.shape_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.shape_encoder.fit([['circle'], ['square'], ['ellipse']])

        self.material_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.material_encoder.fit([['GeO2'], ['ZBLAN'], ['SiO2']])

        # 归一化器
        self.node_scaler = MinMaxScaler()
        self.radius_scaler = MinMaxScaler()
        self.wavelength_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self._precompute_normalization()

        self.samples = []  # 每条数据：(文件路径, 波长索引)

        for file_path in self.file_list:
            try:
                # 预读行数，避免完整加载大型文件来确定行数
                # 这部分对于min(100, df.shape[0])来说可能不是必须的，因为pandas会处理
                df_peek = pd.read_csv(file_path, nrows=105)  # 读取稍多一点以获取df.shape[0]
                num_total_rows = len(pd.read_csv(file_path, usecols=[0]))  # 获取真实总行数的一种方式
                num_conditions = min(100, num_total_rows)  # 只取前100行条件或实际行数（较小者）
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty or invalid CSV file: {file_path}")
                continue
            except Exception as e:
                print(f"Warning: Could not read shape for {file_path}. Error: {e}. Skipping.")
                continue

            for cond_idx in range(num_conditions):
                self.samples.append((file_path, cond_idx))

        if not self.file_list:
            print("Warning: No CSV files found in the specified root directory.")
        if not self.samples:
            print("Warning: No samples were loaded. Check dataset path and CSV files.")

    def _precompute_normalization(self):
        """预先拟合归一化器"""
        all_hole_pos, all_radius, all_wavelengths, all_targets = [], [], [], []

        for file_path in self.file_list:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Warning: CSV file is empty, skipping for normalization: {file_path}")
                    continue
            except pd.errors.EmptyDataError:
                print(f"Warning: CSV file is empty or invalid, skipping for normalization: {file_path}")
                continue
            except Exception as e:
                print(f"Warning: Could not read {file_path} for normalization. Error: {e}. Skipping.")
                continue

            structure_rows = detect_structure_rows_by_nan(df)  # 假设这个函数仍然基于前两列的NaN

            if structure_rows > 0:
                if self.use_column_names:
                    all_hole_pos.append(df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values)
                    all_radius.append(df.loc[:structure_rows - 1, [COL_RADIUS_1, COL_RADIUS_2]].values)
                else:
                    all_hole_pos.append(df.iloc[:structure_rows, [0, 1]].values)  # 使用索引
                    all_radius.append(df.iloc[:structure_rows, [4, 5]].values)  # 使用索引

            # 条件参数部分
            num_conditions = min(100, df.shape[0])
            if num_conditions > 0:
                if self.use_column_names:
                    all_wavelengths.append(df.loc[:num_conditions - 1, COL_WAVELENGTH].values.reshape(-1, 1))
                    # 确保TARGET_PARAM_COLS中的列名存在于df.columns中
                    valid_target_cols = [col for col in TARGET_PARAM_COLS if col in df.columns]
                    if len(valid_target_cols) == len(TARGET_PARAM_COLS):
                        all_targets.append(
                            df.loc[:num_conditions - 1, TARGET_PARAM_COLS].astype(float).fillna(0).values)
                    else:
                        print(
                            f"Warning: Not all target parameter columns found in {file_path}. Using iloc fallback or skipping.")
                        # Fallback to iloc if column names are missing, assuming original indexing
                        all_targets.append(df.iloc[:num_conditions, 12:17].astype(float).fillna(0).values)
                else:
                    all_wavelengths.append(df.iloc[:num_conditions, 11].values.reshape(-1, 1))  # 使用索引
                    all_targets.append(df.iloc[:num_conditions, 12:17].astype(float).fillna(0).values)  # 使用索引

        if all_hole_pos:
            self.node_scaler.fit(np.vstack(all_hole_pos))
        else:
            print("Warning: No hole_pos data found to fit node_scaler.")

        if all_radius:
            self.radius_scaler.fit(np.vstack(all_radius))
        else:
            print("Warning: No radius data found to fit radius_scaler.")

        if all_wavelengths:
            self.wavelength_scaler.fit(np.vstack(all_wavelengths))
        else:
            print("Warning: No wavelength data found to fit wavelength_scaler.")

        if all_targets:
            self.target_scaler.fit(np.vstack(all_targets))
        else:
            print("Warning: No target parameter data found to fit target_scaler.")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        file_path, cond_idx = self.samples[idx]
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError(f"CSV file is empty: {file_path}")
        except Exception as e:
            print(f"Error reading or processing file {file_path} at index {idx}. Error: {e}")
            # 返回一个空的或占位符Data对象，或者根据策略处理错误
            return Data(input_condition=torch.empty(0), hole_pos=torch.empty(0, 2), hole_attr=torch.empty(0, 1),
                        global_attr=torch.empty(0, 1), num_nodes=0)

        structure_rows = detect_structure_rows_by_nan(df)

        if structure_rows == 0:
            # print(f"Warning: No structure rows detected for {file_path}. Returning empty data.")
            hole_pos_np = np.empty((0, 2))
            radius_attr_np = np.empty((0, 2))
            shape_value_np = np.empty((0, 1))
        else:
            if self.use_column_names:
                hole_pos_np = df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values
                radius_attr_np = df.loc[:structure_rows - 1, [COL_RADIUS_1, COL_RADIUS_2]].values
                shape_value_np = df.loc[:structure_rows - 1, COL_SHAPE].values.reshape(-1, 1)
            else:  # 使用索引
                hole_pos_np = df.iloc[:structure_rows, [0, 1]].values
                radius_attr_np = df.iloc[:structure_rows, [4, 5]].values
                shape_value_np = df.iloc[:structure_rows, 6].values.reshape(-1, 1)

        hole_pos = torch.tensor(hole_pos_np, dtype=torch.float32)
        radius_attr = torch.tensor(radius_attr_np, dtype=torch.float32)

        if shape_value_np.size > 0:
            shape_attr = torch.tensor(self.shape_encoder.transform(shape_value_np), dtype=torch.float32)
        else:
            shape_attr = torch.empty((0, len(self.shape_encoder.categories_[0])), dtype=torch.float32)

        if self.use_column_names:
            fiber_radius_val = df.loc[0, COL_FIBER_RADIUS]
            material_value_val = df.loc[0, COL_MATERIAL]
        else:  # 使用索引
            fiber_radius_val = df.iloc[0, 7]
            material_value_val = df.iloc[0, 8]

        fiber_radius = torch.tensor(fiber_radius_val, dtype=torch.float32).view(1)
        material_attr = torch.tensor(self.material_encoder.transform([[material_value_val]])).float().squeeze(0)
        global_attr = torch.cat([fiber_radius, material_attr], dim=0).unsqueeze(0)

        # 归一化
        if self.radius_scaler.data_max_ is not None and radius_attr.numel() > 0:
            radius_attr_normalized_np = self.radius_scaler.transform(radius_attr.numpy())
        elif radius_attr.numel() > 0:
            print(
                f"Warning: radius_scaler has not been fitted or no data to transform for file {file_path}. Using raw radius_attr.")
            radius_attr_normalized_np = radius_attr.numpy()
        else:  # radius_attr is empty
            radius_attr_normalized_np = np.empty((0, 2))  # 保持与radius_attr相同的列数
        radius_attr_normalized = torch.tensor(radius_attr_normalized_np, dtype=torch.float32)

        if self.node_scaler.data_max_ is not None and hole_pos.numel() > 0:
            hole_pos_normalized_np = self.node_scaler.transform(hole_pos.numpy())
        elif hole_pos.numel() > 0:
            print(
                f"Warning: node_scaler has not been fitted or no data to transform for file {file_path}. Using raw hole_pos.")
            hole_pos_normalized_np = hole_pos.numpy()
        else:  # hole_pos is empty
            hole_pos_normalized_np = np.empty((0, 2))
        hole_pos_normalized = torch.tensor(hole_pos_normalized_np, dtype=torch.float32)

        if radius_attr_normalized.numel() > 0 or shape_attr.numel() > 0:
            hole_attr = torch.cat([radius_attr_normalized, shape_attr], dim=-1)
        # 处理其中一个为空，但另一个不为空的情况
        elif radius_attr_normalized.numel() > 0:  # shape_attr is empty
            # 确定shape_attr应该有的维度 (num_features_shape)
            num_shape_features = len(self.shape_encoder.categories_[0])
            empty_shape_attr = torch.zeros((radius_attr_normalized.shape[0], num_shape_features), dtype=torch.float32)
            hole_attr = torch.cat([radius_attr_normalized, empty_shape_attr], dim=-1)
        elif shape_attr.numel() > 0:  # radius_attr_normalized is empty
            # 确定radius_attr应该有的维度 (num_features_radius, e.g., 2)
            num_radius_features = 2  # 假设半径有2个特征
            empty_radius_attr = torch.zeros((shape_attr.shape[0], num_radius_features), dtype=torch.float32)
            hole_attr = torch.cat([empty_radius_attr, shape_attr], dim=-1)
        else:  # both are empty
            hole_attr = torch.empty((0, 2 + len(self.shape_encoder.categories_[0])), dtype=torch.float32)

        # --- 当前条件部分（input_condition） ---
        if self.use_column_names:
            wavelength_raw = df.loc[cond_idx, COL_WAVELENGTH]
            valid_target_cols = [col for col in TARGET_PARAM_COLS if col in df.columns]
            if len(valid_target_cols) == len(TARGET_PARAM_COLS):
                target_raw = df.loc[cond_idx, TARGET_PARAM_COLS].astype(float).values
            else:  # Fallback
                target_raw = df.iloc[cond_idx, 12:17].astype(float).values

        else:  # 使用索引
            wavelength_raw = df.iloc[cond_idx, 11]
            target_raw = df.iloc[cond_idx, 12:17].astype(float).values

        # 归一化
        wavelength_np = np.array([[wavelength_raw]])
        if self.wavelength_scaler.data_max_ is not None:
            wavelength_scaled_np = self.wavelength_scaler.transform(wavelength_np)[0]
        else:
            print(f"Warning: wavelength_scaler has not been fitted for file {file_path}. Using raw wavelength.")
            wavelength_scaled_np = wavelength_np.flatten()
        wavelength = torch.tensor(wavelength_scaled_np, dtype=torch.float32)

        if self.target_scaler.data_max_ is not None:
            target_scaled_np = self.target_scaler.transform(target_raw.reshape(1, -1))[0]
        else:
            print(f"Warning: target_scaler has not been fitted for file {file_path}. Using raw target_param.")
            target_scaled_np = target_raw
        target_param = torch.tensor(target_scaled_np, dtype=torch.float32)

        input_condition = torch.cat([wavelength.view(1), target_param], dim=0)  # 确保wavelength也是1D再拼接

        data = Data(
            input_condition=input_condition,
            hole_pos=hole_pos_normalized,  # 使用归一化后的坐标
            hole_attr=hole_attr,
            global_attr=global_attr,
            num_nodes=hole_pos_normalized.size(0)
        )

        if self.transform:
            data = self.transform(data)

        return data


# 备注: FiberForwardDatasetModified 类没有在这个请求中提供，所以这里只修改了 FiberInverseDataset。
# 如果需要，可以按照类似的逻辑修改 FiberForwardDatasetModified。
# 同样，如果您的CSV文件没有表头，请将 use_column_names 设置为 False (或移除该参数并固定使用iloc)。
# 确保COL_XXX常量定义与您的数据格式（有无表头，列的实际名称或索引）完全匹配。


import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
import os
import glob
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

# --- 定义CSV列名常量 (根据您的CSV表头进行调整) ---
# 结构部分
COL_X_COORD = 'X (um)'
COL_Y_COORD = 'Y (um)'
# COL_R_POLAR = 'R (um)' # 极坐标半径，如果需要可以取消注释
# COL_THETA_POLAR = 'Theta (rad)' # 极坐标角度，如果需要可以取消注释
COL_RADIUS_1 = 'Air Hole Radius (um)'
COL_RADIUS_2 = 'Air Hole Radius2 (um)'  # 如果您的椭圆/矩形孔洞有两个半径参数
COL_SHAPE = 'air_hole_shape'
COL_FIBER_RADIUS = 'Fiber Radius (um)'
COL_MATERIAL = 'Fiber Material'

# 条件参数部分 (这些也是 FiberForwardDataset 的目标输出 'y' 和部分输入特征)
COL_WAVELENGTH = 'Wavelength (um)'  # 假设波长列名是这个，如果和逆向设计一样，请统一

# 目标光学参数列 (对应 FiberForwardDataset 的 'y')
# 假设与 FiberInverseDataset 中的 target_param_cols 相同
# 这些通常是模型的输出，例如：有效折射率(neff)、有效面积(Aeff)、色散(D)、色散斜率(Ds)、非线性系数(gamma)等
# 假设有5个目标参数，列名从 'Effective Mode Index' 开始
TARGET_PARAM_NAMES = [
    'Effective Mode Index',  # 示例，请替换为实际列名
    'Effective Area (um^2)',  # 示例
    'Dispersion (ps/(nm*km))',  # 示例
    'Dispersion Slope (ps/(nm^2*km))',  # 示例
    'Nonlinear Coefficient (1/(W*km))'  # 示例
]


# --- 结束列名常量定义 ---


def detect_structure_rows_by_nan(df):
    """
    检测每张图的节点数。
    根据前两列（默认为X, Y坐标）出现NaN的位置来判断节点区域结束。
    """
    # 假设坐标列是 COL_X_COORD 和 COL_Y_COORD
    # 如果CSV无header，则使用iloc更为直接
    # x_col = df[COL_X_COORD]
    # y_col = df[COL_Y_COORD]
    # 为了与之前的FiberInverseDataset中的detect_structure_rows_by_nan行为一致（基于iloc）
    # 如果确定CSV有header且列名正确，可以切换到使用列名
    try:
        x_col = df.iloc[:, 0]  # 通常是X坐标
        y_col = df.iloc[:, 1]  # 通常是Y坐标
    except IndexError:
        # 如果列数不足，则认为没有有效结构行
        return 0

    for i in range(len(x_col)):
        if pd.isna(x_col[i]) or pd.isna(y_col[i]):
            return i
    return len(x_col)


# 假设这是一个用于创建空图或无边图的辅助函数
def create_no_edge_graph(num_nodes):
    return torch.empty((2, 0), dtype=torch.long)


class FiberForwardDatasetModified(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, use_column_names=True):  # 假设默认使用列名
        super().__init__(root, transform, pre_transform)
        self.file_list = sorted(glob.glob(os.path.join(self.root, "*.csv")))
        self.use_column_names = use_column_names

        # OneHot编码器
        self.shape_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.shape_encoder.fit([['circle'], ['square'], ['ellipse']])  # 根据您的数据调整

        self.material_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.material_encoder.fit([['GeO2'], ['ZBLAN'], ['SiO2']])  # 根据您的数据调整

        # 归一化器
        self.node_coord_scaler = MinMaxScaler()  # 用于孔洞坐标 (x, y)
        self.hole_radius_scaler = MinMaxScaler()  # 用于孔洞半径属性 (r1, r2)
        self.fiber_radius_scaler = MinMaxScaler()  # 用于光纤包层半径 (全局属性)
        self.wavelength_scaler = MinMaxScaler()  # 用于波长 (既是输入也是条件)
        self.target_scaler = MinMaxScaler()  # 用于目标光学参数 (模型的输出y)

        self._precompute_normalization()

        self.samples = []  # 每条数据：(文件路径, 条件索引)
        for file_path in self.file_list:
            try:
                df_peek = pd.read_csv(file_path, nrows=105)  # 尝试读取一些行来获取行数
                num_total_rows = len(pd.read_csv(file_path, usecols=[TARGET_PARAM_NAMES[0]]))  # 获取真实条件行数
                num_conditions = min(100, num_total_rows)  # 与FiberInverseDataset一致，限制条件行数
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty or invalid CSV file for sample generation: {file_path}")
                continue
            except Exception as e:
                print(f"Warning: Could not read shape for {file_path} during sample generation. Error: {e}. Skipping.")
                continue

            # 对于ForwardDataset，每个(结构+波长)组合都是一个样本，目标是该组合下的光学参数
            for cond_idx in range(num_conditions):
                self.samples.append((file_path, cond_idx))

        if not self.file_list:
            print("Warning: No CSV files found in the specified root directory for ForwardDataset.")
        if not self.samples:
            print("Warning: No samples were loaded for ForwardDataset. Check dataset path and CSV files.")

    def _precompute_normalization(self):
        """预先拟合归一化器"""
        all_hole_coords, all_hole_radius_attrs, all_fiber_radius, all_wavelengths, all_targets = [], [], [], [], []

        for file_path in self.file_list:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Warning: CSV file is empty, skipping for ForwardDataset normalization: {file_path}")
                    continue
            except pd.errors.EmptyDataError:
                print(f"Warning: CSV file is empty or invalid, skipping for ForwardDataset normalization: {file_path}")
                continue
            except Exception as e:
                print(f"Warning: Could not read {file_path} for ForwardDataset normalization. Error: {e}. Skipping.")
                continue

            structure_rows = detect_structure_rows_by_nan(df)

            # 孔洞坐标 (x, y) - 来自结构部分
            if structure_rows > 0:
                hole_coords_raw = df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values
                all_hole_coords.append(hole_coords_raw)

                # 孔洞半径属性 (r1, r2) - 来自结构部分
                # 假设 COL_RADIUS_2 可能不存在 (例如圆形孔只有一个半径)
                radius_cols_to_use = [COL_RADIUS_1]
                if COL_RADIUS_2 in df.columns:
                    radius_cols_to_use.append(COL_RADIUS_2)

                hole_radius_raw = df.loc[:structure_rows - 1, radius_cols_to_use].values
                # 如果只有一个半径列，确保它仍然是2D的，MinMaxScaler期望2D输入
                if hole_radius_raw.ndim == 1:
                    hole_radius_raw = hole_radius_raw.reshape(-1, 1)
                # 如果有两个半径列，但第二列全是NaN (例如对圆形孔)，也可能需要处理
                # 简单处理：如果第二列存在但值是NaN，可以先用0填充或用第一列的值填充
                if len(radius_cols_to_use) == 2:  # 检查是否有两列
                    # 检查第二列是否全为NaN或者部分为NaN
                    col2_is_nan = pd.isna(df.loc[:structure_rows - 1, COL_RADIUS_2].values)
                    if np.all(col2_is_nan):  # 如果全为NaN，则复制第一列
                        hole_radius_raw[:, 1] = df.loc[:structure_rows - 1, COL_RADIUS_1].values
                    elif np.any(col2_is_nan):  # 如果部分为NaN，则用0填充
                        hole_radius_raw[np.isnan(hole_radius_raw[:, 1]), 1] = 0.0

                all_hole_radius_attrs.append(hole_radius_raw)

            # 光纤包层半径 (全局属性) - 通常在文件的第一行定义，且对该文件所有条件行都一样
            # MinMaxScaler期望2D输入，所以reshape(-1,1)
            if not df.empty:
                current_fiber_radius = df.loc[0, COL_FIBER_RADIUS]
                all_fiber_radius.append(np.array([[current_fiber_radius]]))

            # 波长和目标参数 - 来自每个条件行
            num_conditions = min(100, df.shape[0])
            if num_conditions > 0:
                wavelengths_raw = df.loc[:num_conditions - 1, COL_WAVELENGTH].values.reshape(-1, 1)
                all_wavelengths.append(wavelengths_raw)

                # 确保目标参数列存在
                valid_target_cols = [col for col in TARGET_PARAM_NAMES if col in df.columns]
                if len(valid_target_cols) == len(TARGET_PARAM_NAMES):
                    targets_raw = df.loc[:num_conditions - 1, valid_target_cols].astype(float).fillna(0).values
                    all_targets.append(targets_raw)
                else:
                    print(
                        f"Warning: Not all target parameter columns for 'y' found in {file_path} during precompute. Check TARGET_PARAM_NAMES.")

        if all_hole_coords:
            self.node_coord_scaler.fit(np.vstack(all_hole_coords))
        else:
            print("Warning: No hole_pos data to fit node_coord_scaler for ForwardDataset.")

        if all_hole_radius_attrs:
            # 在vstack之前，确保所有数组都有相同的列数 (例如，都补齐到2列)
            processed_radius_attrs = []
            for r_attr in all_hole_radius_attrs:
                if r_attr.shape[1] == 1:  # 如果只有一个半径列
                    processed_radius_attrs.append(np.hstack([r_attr, r_attr.copy()]))  # 复制第一列作为第二列
                elif r_attr.shape[1] == 2:
                    processed_radius_attrs.append(r_attr)
                # else: 忽略形状不符的，或者抛出错误
            if processed_radius_attrs:
                self.hole_radius_scaler.fit(np.vstack(processed_radius_attrs))
            else:
                print("Warning: No valid radius data to fit hole_radius_scaler for ForwardDataset.")

        else:
            print("Warning: No radius data to fit hole_radius_scaler for ForwardDataset.")

        if all_fiber_radius:
            self.fiber_radius_scaler.fit(np.vstack(all_fiber_radius))
        else:
            print("Warning: No fiber_radius data to fit fiber_radius_scaler for ForwardDataset.")

        if all_wavelengths:
            self.wavelength_scaler.fit(np.vstack(all_wavelengths))
        else:
            print("Warning: No wavelength data to fit wavelength_scaler for ForwardDataset.")

        if all_targets:
            self.target_scaler.fit(np.vstack(all_targets))
        else:
            print("Warning: No target (y) data to fit target_scaler for ForwardDataset.")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        file_path, cond_idx = self.samples[idx]
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError(f"ForwardDataset: CSV file is empty: {file_path}")
        except Exception as e:
            print(f"ForwardDataset: Error reading file {file_path} at index {idx}. Error: {e}")
            return Data(x=torch.empty(0, 1), pos=torch.empty(0, 2), edge_index=create_no_edge_graph(0),
                        y=torch.empty(0), graph_features=torch.empty(0), num_nodes=0)

        structure_rows = detect_structure_rows_by_nan(df)

        # --- 提取孔洞特征 (x) 和位置 (pos) ---
        if structure_rows > 0:
            hole_pos_raw_np = df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values

            radius_cols_to_use = [COL_RADIUS_1]
            if COL_RADIUS_2 in df.columns:
                radius_cols_to_use.append(COL_RADIUS_2)
            hole_radius_attr_raw_np = df.loc[:structure_rows - 1, radius_cols_to_use].values
            if hole_radius_attr_raw_np.ndim == 1:  # 单半径列
                hole_radius_attr_raw_np = hole_radius_attr_raw_np.reshape(-1, 1)
                # 补齐到两列，复制第一列 (MinMaxScaler期望一致的列数)
                hole_radius_attr_raw_np = np.hstack([hole_radius_attr_raw_np, hole_radius_attr_raw_np.copy()])
            elif len(radius_cols_to_use) == 2:  # 双半径列
                col2_is_nan = pd.isna(df.loc[:structure_rows - 1, COL_RADIUS_2].values)
                if np.all(col2_is_nan):
                    hole_radius_attr_raw_np[:, 1] = df.loc[:structure_rows - 1, COL_RADIUS_1].values
                elif np.any(col2_is_nan):
                    hole_radius_attr_raw_np[np.isnan(hole_radius_attr_raw_np[:, 1]), 1] = 0.0

            shape_value_raw_np = df.loc[:structure_rows - 1, COL_SHAPE].values.reshape(-1, 1)

            # 归一化孔洞坐标
            if self.node_coord_scaler.data_max_ is not None:
                hole_pos_scaled_np = self.node_coord_scaler.transform(hole_pos_raw_np)
            else:
                print(f"Warning FD get: node_coord_scaler not fit. Using raw for {file_path}");
                hole_pos_scaled_np = hole_pos_raw_np
            hole_pos_scaled = torch.tensor(hole_pos_scaled_np, dtype=torch.float32)

            # 归一化孔洞半径
            if self.hole_radius_scaler.data_max_ is not None:
                hole_radius_attr_scaled_np = self.hole_radius_scaler.transform(hole_radius_attr_raw_np)
            else:
                print(f"Warning FD get: hole_radius_scaler not fit. Using raw for {file_path}");
                hole_radius_attr_scaled_np = hole_radius_attr_raw_np
            hole_radius_attr_scaled = torch.tensor(hole_radius_attr_scaled_np, dtype=torch.float32)

            # One-hot 编码形状
            shape_attr_encoded = torch.tensor(self.shape_encoder.transform(shape_value_raw_np), dtype=torch.float32)

            # 拼接成节点特征 x (归一化半径 + 形状编码)
            node_features_x = torch.cat([hole_radius_attr_scaled, shape_attr_encoded], dim=-1)

        else:  # 没有孔洞结构
            hole_pos_scaled = torch.empty((0, 2), dtype=torch.float32)
            # 确定半径和形状的特征维度
            num_radius_features = 2  # 假设归一化后总是两列
            num_shape_features = len(self.shape_encoder.categories_[0])
            node_features_x = torch.empty((0, num_radius_features + num_shape_features), dtype=torch.float32)

        # --- 提取全局/图级别特征 (graph_features) ---
        # 包括：归一化波长, One-hot材料编码, 归一化光纤包层半径

        # 波长 (来自当前条件行 cond_idx)
        wavelength_raw_val = df.loc[cond_idx, COL_WAVELENGTH]
        wavelength_raw_np = np.array([[wavelength_raw_val]])
        if self.wavelength_scaler.data_max_ is not None:
            wavelength_scaled_np = self.wavelength_scaler.transform(wavelength_raw_np)[0]
        else:
            print(f"Warning FD get: wavelength_scaler not fit. Using raw for {file_path}");
            wavelength_scaled_np = wavelength_raw_np.flatten()
        wavelength_scaled = torch.tensor(wavelength_scaled_np, dtype=torch.float32)  # 应该是 scalar or [1]

        # 材料 (通常来自文件第一行，对该文件所有条件行都一样)
        material_value_val = df.loc[0, COL_MATERIAL]
        material_attr_encoded = torch.tensor(self.material_encoder.transform([[material_value_val]])[0],
                                             dtype=torch.float32)  # [num_mat_classes]

        # 光纤包层半径 (通常来自文件第一行)
        fiber_radius_raw_val = df.loc[0, COL_FIBER_RADIUS]
        fiber_radius_raw_np = np.array([[fiber_radius_raw_val]])
        if self.fiber_radius_scaler.data_max_ is not None:
            fiber_radius_scaled_np = self.fiber_radius_scaler.transform(fiber_radius_raw_np)[0]
        else:
            print(f"Warning FD get: fiber_radius_scaler not fit. Using raw for {file_path}");
            fiber_radius_scaled_np = fiber_radius_raw_np.flatten()
        fiber_radius_scaled = torch.tensor(fiber_radius_scaled_np, dtype=torch.float32)  # 应该是 scalar or [1]

        # 拼接 graph_features: 顺序可能影响模型，保持一致性
        # 示例顺序: 波长, 材料one-hot, 光纤包层半径
        graph_features = torch.cat([
            wavelength_scaled.view(1),  # 确保是1D
            material_attr_encoded,  # 已经是1D
            fiber_radius_scaled.view(1)  # 确保是1D
        ], dim=0)  # Shape: [1 + num_mat_classes + 1]

        # --- 提取目标输出 (y) ---
        # 光学参数 (来自当前条件行 cond_idx)
        valid_target_cols = [col for col in TARGET_PARAM_NAMES if col in df.columns]
        if len(valid_target_cols) == len(TARGET_PARAM_NAMES):
            target_raw_np = df.loc[cond_idx, valid_target_cols].astype(float).fillna(0).values
        else:  # Fallback or error
            print(
                f"Error: Not all target parameter columns for 'y' found in {file_path} at cond_idx {cond_idx}. Check TARGET_PARAM_NAMES.")
            # 你可能需要决定如何处理这种情况，例如返回一个空y或特定的错误标记
            target_raw_np = np.zeros(len(TARGET_PARAM_NAMES))  # 作为一个占位符

        if self.target_scaler.data_max_ is not None:
            target_scaled_np = self.target_scaler.transform(target_raw_np.reshape(1, -1))[0]
        else:
            print(f"Warning FD get: target_scaler not fit. Using raw for {file_path}");
            target_scaled_np = target_raw_np

        target_y = torch.tensor(target_scaled_np, dtype=torch.float32)  # Shape: [num_target_params]

        data = Data(
            x=node_features_x,  # 节点特征 (孔洞属性: 归一化半径r1,r2 + 形状one-hot)
            pos=hole_pos_scaled,  # 节点位置 (孔洞坐标: 归一化x,y)
            edge_index=create_no_edge_graph(hole_pos_scaled.size(0)),  # 通常正向模型中边信息很重要，这里是无边图
            y=target_y,  # 目标输出 (归一化后的光学参数)
            graph_features=graph_features,  # 全局/条件图特征 (波长, 材料, 光纤半径)
            num_nodes=hole_pos_scaled.size(0)
        )

        if self.transform:
            data = self.transform(data)

        return data
