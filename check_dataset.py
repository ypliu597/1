import pandas as pd
import glob
import os
import numpy as np

# --- 定义您CSV文件中实际的列名 ---
COL_MATERIAL = 'Fiber Material'
COL_SHAPE = 'air_hole_shape'


# --- ---

def check_data_source_for_nan_and_types(data_root_path):  # 重命名函数以反映其功能
    """
    检查指定目录下所有 graph_*.csv 文件中材料和形状列，
    打印包含 NaN 的文件名，并总结唯一值及类型。
    """
    all_csv_files = sorted(glob.glob(os.path.join(data_root_path, "graph_*.csv")))

    if not all_csv_files:
        print(f"错误: 在目录 '{data_root_path}' 中没有找到 'graph_*.csv' 文件。")
        return

    print(f"找到 {len(all_csv_files)} 个 CSV 文件，开始检查材料和形状列中的 NaN 及类型...\n")

    all_materials_with_types = {}
    all_shapes_with_types = {}
    files_with_nan_material = set()
    files_with_nan_shape = set()

    for file_path in all_csv_files:
        # print(f"--- 正在处理文件: {os.path.basename(file_path)} ---") # 可以取消注释以查看每个文件处理情况
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                # print("  文件为空，跳过。")
                continue

            # 检查材料列
            if COL_MATERIAL in df.columns:
                # 检查该列是否有任何 NaN 值
                if df[COL_MATERIAL].isnull().any():  # pd.isnull() 可以检测 None 和 np.nan
                    files_with_nan_material.add(os.path.basename(file_path))

                unique_materials_in_file = df[COL_MATERIAL].unique()
                for val in unique_materials_in_file:
                    all_materials_with_types[val] = type(val)
            # else:
            # print(f"  警告: 文件中缺少 '{COL_MATERIAL}' 列。")

            # 检查形状列
            if COL_SHAPE in df.columns:
                # 检查该列是否有任何 NaN 值 (形状通常在每个结构行)
                # 为了简单起见，我们检查整个列，如果您的NaN只可能出现在特定行，可以调整
                if df[COL_SHAPE].isnull().any():
                    files_with_nan_shape.add(os.path.basename(file_path))

                unique_shapes_in_file = df[COL_SHAPE].unique()
                for val in unique_shapes_in_file:
                    all_shapes_with_types[val] = type(val)
            # else:
            # print(f"  警告: 文件中缺少 '{COL_SHAPE}' 列。")

        except pd.errors.EmptyDataError:
            # print(f"  错误: 文件为空或无法解析，跳过。")
            pass
        except Exception as e:
            print(f"  处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
        # print("-" * (len(os.path.basename(file_path)) + 28))

    print("\n--- NaN 值检查结果 ---")
    if files_with_nan_material:
        print(f"在以下文件中 '{COL_MATERIAL}' 列检测到 NaN 值:")
        for fname in sorted(list(files_with_nan_material)):
            print(f"  - {fname}")
    else:
        print(f"在所有已检查文件的 '{COL_MATERIAL}' 列中未检测到 NaN 值。")

    if files_with_nan_shape:
        print(f"\n在以下文件中 '{COL_SHAPE}' 列检测到 NaN 值:")
        for fname in sorted(list(files_with_nan_shape)):
            print(f"  - {fname}")
    else:
        print(f"在所有已检查文件的 '{COL_SHAPE}' 列中未检测到 NaN 值。")

    print("\n--- 唯一值及类型总结 ---")
    material_print_list = []
    for val, val_type in all_materials_with_types.items():
        material_print_list.append((str(val), (val, val_type)))
    material_print_list.sort(key=lambda x: x[0])
    print(f"所有文件中找到的唯一材料值 ({len(all_materials_with_types)} 个):")
    for str_val, (original_val, val_type) in material_print_list:
        print(f"  - '{original_val}' (类型: {val_type})")
    material_value_types_set = set(all_materials_with_types.values())
    print(f"材料值中出现的数据类型: {material_value_types_set}")

    shape_print_list = []
    for val, val_type in all_shapes_with_types.items():
        shape_print_list.append((str(val), (val, val_type)))
    shape_print_list.sort(key=lambda x: x[0])
    print(f"\n所有文件中找到的唯一形状值 ({len(all_shapes_with_types)} 个):")
    for str_val, (original_val, val_type) in shape_print_list:
        print(f"  - '{original_val}' (类型: {val_type})")
    shape_value_types_set = set(all_shapes_with_types.values())
    print(f"形状值中出现的数据类型: {shape_value_types_set}")

    # ... (底部的总结性建议保持不变) ...
    found_nan_float_material = any(isinstance(m, float) and np.isnan(m) for m in all_materials_with_types.keys())
    found_nan_float_shape = any(isinstance(s, float) and np.isnan(s) for s in all_shapes_with_types.keys())

    if found_nan_float_material or found_nan_float_shape:
        print(
            "\n重要：检测到数据类型问题（包含浮点型NaN）！强烈建议您修改 `preprocess_scalers_encoders.py` 和 `FiberForwardDataset.py`：")
        print("1. 在这两个脚本中，当读取材料和形状值时，使用 `pd.isna(value)` 来判断是否为缺失值。")
        print("2. 如果是缺失值，将其替换为一个固定的字符串占位符 (例如 'UNKNOWN_MATERIAL', 'UNKNOWN_SHAPE')。")
        print("3. 对于非缺失值，使用 `str(value).strip()` 将其强制转换为字符串并去除首尾空格。")
        print("   确保在预处理（拟合Encoder）和数据加载（transform）时使用完全相同的逻辑和占位符。")


# --- 使用示例 ---
if __name__ == "__main__":
    your_dataset_root_directory = 'core/datasets'

    data_root_env = os.getenv('MY_PROJECT_DATA_ROOT')
    if data_root_env:
        your_dataset_root_directory = data_root_env
        print(f"使用环境变量 MY_PROJECT_DATA_ROOT 指定的数据目录: {your_dataset_root_directory}")
    elif not os.path.isdir(your_dataset_root_directory):
        print(f"错误: 默认数据目录 '{your_dataset_root_directory}' 不存在。")
        print("请修改脚本中的 'your_dataset_root_directory' 为您的CSV文件所在目录，")
        print("或者设置环境变量 MY_PROJECT_DATA_ROOT 指向该目录。")
        exit()

    print(f"将检查以下目录中的CSV文件: {os.path.abspath(your_dataset_root_directory)}")
    check_data_source_for_nan_and_types(your_dataset_root_directory)