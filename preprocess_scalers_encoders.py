import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from tqdm import tqdm

# --- 定义所有可能用到的列名常量 ---
COL_X_COORD = 'X (um)'
COL_Y_COORD = 'Y (um)'
COL_RADIUS_1 = 'Air Hole Radius (um)'
COL_RADIUS_2 = 'Air Hole Radius2 (um)'
COL_SHAPE = 'air_hole_shape'
COL_FIBER_RADIUS = 'Fiber Radius (um)'
COL_MATERIAL = 'Fiber Material'
COL_WAVELENGTH_NM = 'Wavelength (nm)'
# COL_RADIAL_DIST = 'R (um)'

TARGET_PARAM_NAMES = [
    'Effective Index (neff_real)',
    'Effective mode area(um^2)',
    'Nonlinear coefficient(1/W/km)',
    'Dispersion (ps/nm/km)',
    'GVD(ps^2/km)'
]
# --- 结束常量定义 ---

# 定义 NaN 占位符
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


def main(args):
    data_root = args.data_root
    output_file = args.output_file
    use_column_names = not args.no_header

    all_files = sorted(glob.glob(os.path.join(data_root, "graph_*.csv")))
    if not all_files:
        print(f"错误: 在目录 '{data_root}' 中没有找到 'graph_*.csv' 文件。")
        return

    print(f"找到 {len(all_files)} 个 CSV 文件，开始收集数据...")

    # --- 初始化列表 ---
    all_coords, all_hole_radii, all_fiber_radii, all_wavelengths_nm, all_targets = [], [], [], [], []
    # all_radial_distances = [] # Removed: No longer collecting radial distances
    all_shapes_for_fitting = []
    all_materials_for_fitting = []
    # --- --- --- --- ---

    # --- 遍历所有文件收集数据 ---
    for file_path in tqdm(all_files, desc="Processing CSVs"):
        try:
            if not use_column_names:
                print(f"\n错误: 当前脚本要求 CSV 文件包含表头。跳过 --no_header 模式。")
                continue

            df = pd.read_csv(file_path)
            if df.empty: continue

            # 基本列检查 (COL_RADIAL_DIST removed)
            required_cols_check = [COL_X_COORD, COL_Y_COORD, COL_RADIUS_1, COL_SHAPE,
                                   COL_FIBER_RADIUS, COL_MATERIAL, COL_WAVELENGTH_NM
                                   ] + TARGET_PARAM_NAMES # COL_RADIAL_DIST removed
            if not all(col in df.columns for col in required_cols_check):
                missing_cols = [col for col in required_cols_check if col not in df.columns]
                # Also check if COL_RADIAL_DIST was present for a more informative warning if files still contain it but it's not required
                if 'R (um)' in df.columns and 'R (um)' not in required_cols_check: # 'R (um)' is the typical name for COL_RADIAL_DIST
                     pass # It's okay if it's present but not required
                # Print warning only if truly required columns are missing
                if any(mc for mc in missing_cols if mc != 'R (um)' and mc not in TARGET_PARAM_NAMES and mc not in [COL_X_COORD, COL_Y_COORD, COL_RADIUS_1, COL_SHAPE, COL_FIBER_RADIUS, COL_MATERIAL, COL_WAVELENGTH_NM]): # check primary required
                     print(f"\n警告: 文件 {os.path.basename(file_path)} 缺少必要的列: {missing_cols}。跳过此文件。")
                     continue
                # Check missing target params separately if needed, or handle as NaN later
                # For now, if any of the primary required are missing, skip.

            structure_rows = detect_structure_rows_by_nan(df)

            # --- 材料收集 ---
            material_val_raw = df.loc[0, COL_MATERIAL]
            if pd.isna(material_val_raw):
                all_materials_for_fitting.append([UNKNOWN_MATERIAL_PLACEHOLDER])
            else:
                all_materials_for_fitting.append([str(material_val_raw).strip()])

            # --- 形状收集 ---
            if structure_rows > 0:
                unique_shapes_in_file = df.loc[:structure_rows - 1, COL_SHAPE].unique()
                for s_val in unique_shapes_in_file:
                    if pd.isna(s_val):
                        all_shapes_for_fitting.append([UNKNOWN_SHAPE_PLACEHOLDER])
                    else:
                        all_shapes_for_fitting.append([str(s_val).strip()])
            elif COL_SHAPE in df.columns and not df[COL_SHAPE].empty:
                s_val = df.loc[0, COL_SHAPE]
                if pd.isna(s_val):
                    all_shapes_for_fitting.append([UNKNOWN_SHAPE_PLACEHOLDER])
                else:
                    all_shapes_for_fitting.append([str(s_val).strip()])

            # --- 提取结构数据 (坐标、半径) ---
            if structure_rows > 0:
                coords = df.loc[:structure_rows - 1, [COL_X_COORD, COL_Y_COORD]].values.astype(np.float32)
                all_coords.append(coords)

                radius_cols_to_use = [COL_RADIUS_1]
                if COL_RADIUS_2 in df.columns and not df.loc[:structure_rows - 1, COL_RADIUS_2].isna().all():
                    radius_cols_to_use.append(COL_RADIUS_2)
                hole_radius_raw = df.loc[:structure_rows - 1, radius_cols_to_use].values.astype(np.float32)
                if hole_radius_raw.shape[1] == 1:
                    hole_radius_raw = np.hstack([hole_radius_raw, hole_radius_raw])
                hole_radius_raw = np.nan_to_num(hole_radius_raw, nan=0.0)
                all_hole_radii.append(hole_radius_raw)

                # --- COL_RADIAL_DIST extraction removed ---

            # --- 提取全局和条件数据 ---
            num_total_rows = len(df)
            num_condition_rows = num_total_rows - structure_rows
            condition_start_idx = structure_rows
            num_condition_rows_to_process = num_condition_rows

            if num_condition_rows <= 0:
                if structure_rows > 0 and structure_rows < num_total_rows:
                    num_condition_rows_to_process = 1
                elif structure_rows == 0 and num_total_rows > 0:
                    num_condition_rows_to_process = num_total_rows
                    condition_start_idx = 0
                else:
                    num_condition_rows_to_process = 0

            ref_row_idx_global = 0
            if COL_FIBER_RADIUS in df.columns and len(df) > ref_row_idx_global:
                fiber_radius_val = df.loc[ref_row_idx_global, COL_FIBER_RADIUS]
                if pd.notna(fiber_radius_val): all_fiber_radii.append(float(fiber_radius_val))

            for i in range(num_condition_rows_to_process):
                current_cond_idx = condition_start_idx + i
                if current_cond_idx >= len(df): continue

                # Check if all TARGET_PARAM_NAMES and COL_WAVELENGTH_NM exist at this row before trying to access
                # This is a more robust check for condition rows
                condition_row_cols = [COL_WAVELENGTH_NM] + TARGET_PARAM_NAMES
                if not all(col in df.columns for col in condition_row_cols):
                    # print(f"\n警告: 文件 {os.path.basename(file_path)}, 行 {current_cond_idx} 缺少条件列。跳过此行。")
                    continue


                wl = df.loc[current_cond_idx, COL_WAVELENGTH_NM]
                if pd.notna(wl):
                    all_wavelengths_nm.append(float(wl))
                else:
                    # print(f"\n警告: 文件 {os.path.basename(file_path)}, 行 {current_cond_idx} 波长为NaN。跳过此条件。")
                    continue # Skip if wavelength is NaN as it's a key condition

                # Check if all target columns exist before trying to access df.loc with a list of them
                valid_target_cols = [tc for tc in TARGET_PARAM_NAMES if tc in df.columns]
                if len(valid_target_cols) != len(TARGET_PARAM_NAMES):
                    # print(f"\n警告: 文件 {os.path.basename(file_path)}, 行 {current_cond_idx} 缺少部分目标列。跳过此条件。")
                    continue

                target_vals_series = df.loc[current_cond_idx, TARGET_PARAM_NAMES]
                if target_vals_series.isna().any(): # Check for NaNs in target values
                    # print(f"\n警告: 文件 {os.path.basename(file_path)}, 行 {current_cond_idx} 目标参数包含NaN。跳过此条件。")
                    continue
                all_targets.append(target_vals_series.astype(float).values)


        except Exception as e:
            print(f"\n处理文件 {os.path.basename(file_path)} 时出错: {e}")
            continue

    print("\n数据收集完成，开始拟合 Scaler 和 Encoder...")
    scalers = {}
    encoders = {}

    # --- 拟合 Scaler ---
    if all_coords:
        scalers['node_coord'] = MinMaxScaler().fit(np.vstack(all_coords)); print("拟合: node_coord scaler")
    else:
        scalers['node_coord'] = MinMaxScaler(); print("警告: 未收集到 node_coord 数据, 创建空 scaler")

    if all_hole_radii:
        scalers['hole_radius'] = MinMaxScaler().fit(np.vstack(all_hole_radii)); print("拟合: hole_radius scaler")
    else:
        scalers['hole_radius'] = MinMaxScaler(); print("警告: 未收集到 hole_radius 数据, 创建空 scaler")

    # --- COL_RADIAL_DIST scaler fitting removed ---

    if all_fiber_radii:
        scalers['fiber_radius'] = MinMaxScaler().fit(np.array(all_fiber_radii).reshape(-1, 1)); print(
            "拟合: fiber_radius scaler")
    else:
        scalers['fiber_radius'] = MinMaxScaler(); print("警告: 未收集到 fiber_radius 数据, 创建空 scaler")

    if all_wavelengths_nm:
        scalers['wavelength'] = MinMaxScaler().fit(np.array(all_wavelengths_nm).reshape(-1, 1)); print(
            "拟合: wavelength scaler (nm)")
    else:
        scalers['wavelength'] = MinMaxScaler(); print("警告: 未收集到 wavelength 数据, 创建空 scaler")

    if all_targets:
        scalers['target'] = MinMaxScaler().fit(np.vstack(all_targets)); print(
            f"拟合: target scaler ({len(TARGET_PARAM_NAMES)} 个目标)")
    else:
        scalers['target'] = MinMaxScaler(); print("警告: 未收集到 target 数据, 创建空 scaler")

    # --- 清洗并拟合 Encoder ---
    if all_shapes_for_fitting:
        unique_shapes = sorted(list(set(s[0] for s in all_shapes_for_fitting if s[0] is not None))) # ensure no None
        if not unique_shapes or (len(unique_shapes) == 1 and unique_shapes[0] == UNKNOWN_SHAPE_PLACEHOLDER and not any(s[0] != UNKNOWN_SHAPE_PLACEHOLDER for s in all_shapes_for_fitting if s[0] is not None) ):
             print(f"警告: 仅找到占位符形状 '{UNKNOWN_SHAPE_PLACEHOLDER}' 或无有效形状数据。Encoder 将主要基于此。")
             if UNKNOWN_SHAPE_PLACEHOLDER not in unique_shapes and all_shapes_for_fitting : unique_shapes.append(UNKNOWN_SHAPE_PLACEHOLDER) # Ensure placeholder
             if not unique_shapes: unique_shapes = [UNKNOWN_SHAPE_PLACEHOLDER] # Handle empty case

        print(f"用于拟合 Shape Encoder 的唯一类别: {unique_shapes}")
        encoders['shape'] = OneHotEncoder(categories=[unique_shapes], sparse_output=False, handle_unknown='ignore')
        # Filter out any potential [None] that might have slipped through if not handled above
        valid_shapes_for_fit = [s for s in all_shapes_for_fitting if s[0] is not None]
        if not valid_shapes_for_fit: valid_shapes_for_fit = [[UNKNOWN_SHAPE_PLACEHOLDER]] # fit with placeholder if empty

        encoders['shape'].fit(valid_shapes_for_fit)
        print(f"拟合: shape encoder (学习到的类别: {encoders['shape'].categories_[0]})")
    else:
        print("警告: 未收集到 shape 数据, 创建含占位符的 encoder")
        encoders['shape'] = OneHotEncoder(categories=[[UNKNOWN_SHAPE_PLACEHOLDER]], sparse_output=False,
                                          handle_unknown='ignore')
        encoders['shape'].fit([[UNKNOWN_SHAPE_PLACEHOLDER]])

    if all_materials_for_fitting:
        unique_materials = sorted(list(set(m[0] for m in all_materials_for_fitting if m[0] is not None)))
        if not unique_materials or (len(unique_materials) == 1 and unique_materials[0] == UNKNOWN_MATERIAL_PLACEHOLDER and not any(m[0] != UNKNOWN_MATERIAL_PLACEHOLDER for m in all_materials_for_fitting if m[0] is not None)):
            print(f"警告: 仅找到占位符材料 '{UNKNOWN_MATERIAL_PLACEHOLDER}' 或无有效材料数据。Encoder 将主要基于此。")
            if UNKNOWN_MATERIAL_PLACEHOLDER not in unique_materials and all_materials_for_fitting: unique_materials.append(UNKNOWN_MATERIAL_PLACEHOLDER)
            if not unique_materials: unique_materials = [UNKNOWN_MATERIAL_PLACEHOLDER]


        print(f"用于拟合 Material Encoder 的唯一类别: {unique_materials}")
        encoders['material'] = OneHotEncoder(categories=[unique_materials], sparse_output=False,
                                             handle_unknown='ignore')
        valid_materials_for_fit = [m for m in all_materials_for_fitting if m[0] is not None]
        if not valid_materials_for_fit: valid_materials_for_fit = [[UNKNOWN_MATERIAL_PLACEHOLDER]]

        encoders['material'].fit(valid_materials_for_fit)
        print(f"拟合: material encoder (学习到的类别: {encoders['material'].categories_[0]})")
    else:
        print("警告: 未收集到 material 数据, 创建含占位符的 encoder")
        encoders['material'] = OneHotEncoder(categories=[[UNKNOWN_MATERIAL_PLACEHOLDER]], sparse_output=False,
                                             handle_unknown='ignore')
        encoders['material'].fit([[UNKNOWN_MATERIAL_PLACEHOLDER]])


    shared_objects = {'scalers': scalers, 'encoders': encoders}
    # 更新必需的 scaler 键 (COL_RADIAL_DIST removed)
    required_scaler_keys = ['node_coord', 'hole_radius', 'fiber_radius', 'wavelength', 'target']
    required_encoder_keys = ['shape', 'material']

    if not all(k in scalers for k in required_scaler_keys):
        print(f"错误：未能创建所有必需的 Scalers。缺失: {[k for k in required_scaler_keys if k not in scalers]}")
        return
    if not all(k in encoders for k in required_encoder_keys):
        print(f"错误：未能创建所有必需的 Encoders。缺失: {[k for k in required_encoder_keys if k not in encoders]}")
        return

    try:
        joblib.dump(shared_objects, output_file)
        print(f"\n成功将共享 Scaler 和 Encoder 对象保存到: {output_file}")
    except Exception as e:
        print(f"\n保存文件 {output_file} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为光纤模型准备共享的 Scaler 和 Encoder")
    parser.add_argument('--data_root', type=str, default='core/datasets',
                        help='包含所有 graph_*.csv 文件的数据集根目录。')
    parser.add_argument('--output_file', type=str, default='shared_scalers_encoders.pkl',
                        help='保存拟合好的 Scaler 和 Encoder 的输出文件路径。')
    parser.add_argument('--no_header', action='store_true', help='指定 CSV 文件没有表头 (不推荐，当前脚本主要依赖表头)。')
    args = parser.parse_args()
    main(args)