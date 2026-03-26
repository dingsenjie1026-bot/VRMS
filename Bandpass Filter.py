import numpy as np
import scipy.io as sio
from scipy import signal
import mne
import os
import h5py  # 需要导入 h5py

# --- 配置 ---
# 定义通道名称和类型
ch_names = ['Fz', 'Cz', 'Pz', 'CP5', 'CP6', 'P3', 'P4']
ch_types = ['eeg'] * 7

# 定义频带范围
freq_bands = {
    'Delta': (1, 4),
    'Theta': (4, 7),
    'Alpha': (7, 13),
    'Beta': (13, 30),
    'All': (1, 30)  # 全频带
}

# 采样频率
SAMPLING_FREQ = 250.0

# --- 请根据您的实际情况修改以下路径 ---

# 您存放 B1, B2, G, R 文件夹的根目录
DATA_ROOT_DIR = r"F:\软件\pycharm\untitled6"

# 指定输出基础路径 (请修改为您希望保存结果的目录)
OUTPUT_BASE_PATH = r"F:\软件\pycharm\untitled6\Processed_Frequency_Bands"

# .mat 文件中EEG数据的变量名 (请确认!)
MAT_VARIABLE_NAME = 'data'  # <--- 重要：请检查您的 .mat 文件

# --- 函数定义 ---

def load_eeg_data_h5py(mat_file_path, variable_name):
    """
    使用 h5py 加载 v7.3 或更高版本的 .mat 文件中的数据。
    """
    try:
        with h5py.File(mat_file_path, 'r') as f:
            if variable_name not in f:
                raise KeyError(f"变量 '{variable_name}' 未在文件 '{mat_file_path}' 中找到。")
            data = f[variable_name][:]

            # 移除大小为1的维度
            data = np.squeeze(data)

            # --- 统一维度校验与转置 ---
            # h5py 读取的数据可能需要转置，判断如果行数是7（通道数），则转置为 (samples, 7)
            if len(data.shape) == 2:
                if data.shape[0] == 7 and data.shape[1] > 7:
                    data = data.T
            else:
                print(f"  警告: 从 '{mat_file_path}' 读取的数据不是二维数组，形状为 {data.shape}。")
            
            return data
    except Exception as e:
        print(f"  错误: 使用 h5py 读取文件 '{mat_file_path}' 时失败: {e}")
        raise  # 重新抛出异常，让调用者处理

def load_and_filter_eeg_data(mat_file_path, sfreq=250.0):
    """
    加载MAT文件并提取各频带数据
    """
    try:
        # 尝试使用 scipy.io.loadmat (适用于旧版 .mat 文件)
        mat_data = sio.loadmat(mat_file_path)
        eeg_data = mat_data[MAT_VARIABLE_NAME]
        print(f"  使用 scipy.io.loadmat 成功加载 '{mat_file_path}'")

    except NotImplementedError as e:
        # 如果失败，是因为是 v7.3 格式，则使用 h5py
        if "Please use HDF reader" in str(e):
            print(f"  检测到 v7.3 格式，切换到 h5py 读取 '{mat_file_path}'")
            eeg_data = load_eeg_data_h5py(mat_file_path, MAT_VARIABLE_NAME)
        else:
            raise e
    except KeyError as e:
        print(f"  错误: 在 '{mat_file_path}' 中未找到变量 '{MAT_VARIABLE_NAME}': {e}")
        return None
    except Exception as e:
        print(f"  错误: 加载文件 '{mat_file_path}' 时发生未知错误: {e}")
        return None

    # --- 【修复1】严格验证并修正数据形状 ---
    eeg_data = np.squeeze(eeg_data) # 去除多余的维度
    
    if len(eeg_data.shape) != 2:
        print(f"  跳过: 数据维度不符合要求，期望2维，实际形状为 {eeg_data.shape}。")
        return None
        
    # 如果读出来的形状是 (7, samples)，主动将其转置为 (samples, 7)
    if eeg_data.shape[0] == 7 and eeg_data.shape[1] > 7:
        eeg_data = eeg_data.T
        
    # 强制校验通道数是否为7（MNE强依赖这个匹配）
    if eeg_data.shape[1] != 7:
        print(f"  致命错误: 数据通道数不为 7，当前形状为 {eeg_data.shape}。跳过此文件。")
        return None
        
    # 提示样本点数量异常（但不中断程序）
    if eeg_data.shape[0] != 5000:
        print(f"  注意: 文件样本点数为 {eeg_data.shape[0]}，而不是标准的 5000。")

    # 创建MNE Info对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # --- 【修复2】关闭 MNE 构建 Raw 对象时的默认冗余输出 ---
    # MNE 期望的输入是 (n_channels, n_times)，所以这里用 eeg_data.T
    raw = mne.io.RawArray(eeg_data.T, info, verbose='ERROR')

    # 提取各频带数据
    filtered_data = {}
    for band, (l_freq, h_freq) in freq_bands.items():
        raw_band = raw.copy()
        
        # --- 【修复2】关闭 MNE 滤波时的 FIR 滤波器设计冗余输出 ---
        raw_band.filter(l_freq, h_freq, fir_design='firwin', verbose='ERROR')
        
        # 获取滤波后的数据并转置回 (samples, channels) 形状，方便后续MATLAB或Python处理
        filtered_data[band] = raw_band.get_data().T

    return filtered_data

def process_all_mat_files(data_root_dir, output_base_path):
    """
    处理所有文件夹中的MAT文件
    """
    conditions = ['B1', 'B2', 'G', 'R']
    os.makedirs(output_base_path, exist_ok=True)
    print(f"输出基础路径: {output_base_path}")

    for condition in conditions:
        condition_path = os.path.join(data_root_dir, condition)

        if not os.path.exists(condition_path):
            print(f"警告: 条件文件夹 '{condition_path}' 未找到，跳过。")
            continue

        print(f"\n--- 正在处理条件: {condition} ---")
        mat_files = [f for f in os.listdir(condition_path) if f.lower().endswith('.mat')]

        if not mat_files:
            print(f"  在 '{condition_path}' 中未找到 .mat 文件。")
            continue
        else:
            print(f"  找到 {len(mat_files)} 个 .mat 文件。")

        for mat_file in mat_files:
            print(f"  正在处理: {mat_file}")
            mat_file_path = os.path.join(condition_path, mat_file)

            # 提取各频带数据
            filtered_data = load_and_filter_eeg_data(mat_file_path, sfreq=SAMPLING_FREQ)

            if filtered_data is None:
                print(f"    跳过保存 '{mat_file}' (处理失败)。")
                continue

            for band in freq_bands.keys():
                band_folder = os.path.join(output_base_path, condition, band)
                os.makedirs(band_folder, exist_ok=True)
                
                output_file_path = os.path.join(band_folder, mat_file)
                variable_name = f'eeg_{band.lower()}'
                
                try:
                    sio.savemat(output_file_path, {variable_name: filtered_data[band]})
                    # print(f"    已保存 {band} 频段数据到: {output_file_path}") # 注释掉这句可以让控制台更清爽
                except Exception as e:
                    print(f"    错误: 保存文件 '{output_file_path}' 时失败: {e}")

if __name__ == "__main__":
    print("开始处理EEG数据...")
    if not os.path.exists(DATA_ROOT_DIR):
        print(f"错误: 数据根目录 '{DATA_ROOT_DIR}' 不存在。请检查路径。")
    else:
        process_all_mat_files(DATA_ROOT_DIR, OUTPUT_BASE_PATH)
        print("\n--- 所有文件处理完成 ---")