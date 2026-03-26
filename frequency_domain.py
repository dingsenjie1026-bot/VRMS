import numpy as np
import scipy.io as sio
import os
from scipy import signal as scipy_signal
import warnings

warnings.filterwarnings('ignore')

# --- 频域特征提取函数 ---

def compute_psd(eeg_signal, fs=250):
    """使用Welch方法计算功率谱密度 (PSD)"""
    freqs, psd = scipy_signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)))
    return freqs, psd

def compute_spectral_centroid(freqs, psd_norm):
    """计算谱质心 (已接收归一化后的PSD)"""
    centroid = np.sum(freqs * psd_norm)
    return centroid

def compute_spectral_skewness(freqs, psd_norm, centroid):
    """
    计算谱偏度 (Spectral Skewness)
    注意：输入必须是概率归一化后的 psd_norm
    """
    # 计算二阶中心矩 (方差)
    variance = np.sum(((freqs - centroid) ** 2) * psd_norm)
    if variance == 0:
        return 0
        
    # 计算三阶中心矩
    moment_3 = np.sum(((freqs - centroid) ** 3) * psd_norm)
    
    # 偏度 = 三阶矩 / 标准差的立方
    skewness = moment_3 / (variance ** 1.5)
    return skewness

def compute_spectral_entropy(psd_norm):
    """计算谱熵 (Spectral Entropy)"""
    # 避免log(0)的情况
    psd_norm_safe = np.clip(psd_norm, 1e-10, 1)
    spec_entropy = -np.sum(psd_norm_safe * np.log(psd_norm_safe))
    return spec_entropy

def compute_power_ratio(freqs, psd, band1, band2):
    """计算两个频带的绝对功率比"""
    idx_band1 = np.where((freqs >= band1[0]) & (freqs <= band1[1]))[0]
    idx_band2 = np.where((freqs >= band2[0]) & (freqs <= band2[1]))[0]
    
    if len(idx_band1) == 0 or len(idx_band2) == 0:
        return 0
        
    power_band1 = np.sum(psd[idx_band1])
    power_band2 = np.sum(psd[idx_band2])
    
    if power_band2 == 0:
        return 0
    return power_band1 / power_band2

def extract_frequency_domain_features(eeg_signal, band_name, fs=250):
    """
    提取频域特征
    加入 band_name 参数，确保只在 'All' 宽带信号上计算跨频段比率
    """
    features = {}
    freqs, psd = compute_psd(eeg_signal, fs=fs)
    
    # 核心修复：预先将PSD归一化为概率分布，供形状特征使用
    sum_psd = np.sum(psd)
    if sum_psd == 0:
        psd_norm = psd  # 避免全零除息报错
    else:
        psd_norm = psd / sum_psd
        
    # 提取基本形状特征
    centroid = compute_spectral_centroid(freqs, psd_norm)
    features['spectral_centroid'] = centroid
    features['spectral_skewness'] = compute_spectral_skewness(freqs, psd_norm, centroid)
    features['spectral_entropy'] = compute_spectral_entropy(psd_norm)
    
    # 定义EEG频带范围
    eeg_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    
    # 只有当处理的是全频带 (All) 信号时，才计算跨频带比例
    # 否则在已滤波的信号上算比例会得到无意义的噪声值
    if band_name == 'All':
        features['delta_theta_ratio'] = compute_power_ratio(freqs, psd, eeg_bands['delta'], eeg_bands['theta'])
        features['delta_alpha_ratio'] = compute_power_ratio(freqs, psd, eeg_bands['delta'], eeg_bands['alpha'])
        features['delta_beta_ratio']  = compute_power_ratio(freqs, psd, eeg_bands['delta'], eeg_bands['beta'])
        features['theta_alpha_ratio'] = compute_power_ratio(freqs, psd, eeg_bands['theta'], eeg_bands['alpha'])
        features['theta_beta_ratio']  = compute_power_ratio(freqs, psd, eeg_bands['theta'], eeg_bands['beta'])
        features['alpha_beta_ratio']  = compute_power_ratio(freqs, psd, eeg_bands['alpha'], eeg_bands['beta'])
        
    return features

# --- 文件与批处理逻辑 ---

def get_variable_name_for_band(band_name):
    """根据频段文件夹名称获取对应的MATLAB变量名"""
    mapping = {
        'Delta': 'eeg_delta',
        'Theta': 'eeg_theta',
        'Alpha': 'eeg_alpha',
        'Beta':  'eeg_beta',
        'All':   'eeg_all'
    }
    return mapping.get(band_name, 'eeg_all')

def process_frequency_band_for_freq_domain(folder_path, band_name, output_folder, fs=250):
    """处理单个频段文件夹中的所有MAT文件"""
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    var_name_to_load = get_variable_name_for_band(band_name)
    
    for mat_file in mat_files:
        print(f"处理文件: {mat_file} (频段: {band_name}, 变量: {var_name_to_load})")
        mat_data = sio.loadmat(os.path.join(folder_path, mat_file))
        
        if var_name_to_load in mat_data:
            eeg_data = mat_data[var_name_to_load]
        else:
            print(f"  警告: 在文件 {mat_file} 中未找到变量 '{var_name_to_load}'。跳过。")
            continue
            
        if eeg_data.ndim != 2 or eeg_data.shape[1] != 7:
            print(f"  警告: '{var_name_to_load}' 的形状 {eeg_data.shape} 不符合预期 (N, 7)。跳过。")
            continue
            
        all_features = {}
        channel_names = ['Fz', 'Cz', 'Pz', 'CP5', 'CP6', 'P3', 'P4']
        
        for ch_idx, ch_name in enumerate(channel_names):
            channel_signal = eeg_data[:, ch_idx]
            try:
                # 传入 band_name 以判断是否计算频段比率
                features = extract_frequency_domain_features(channel_signal, band_name, fs=fs)
                for feat_name, feat_value in features.items():
                    all_features[f'{ch_name}_{feat_name}'] = feat_value
            except Exception as e:
                print(f"  处理通道 {ch_name} 时出错: {e}")
                continue
                
        file_name, file_ext = os.path.splitext(mat_file)
        output_file_name = f"{file_name}_{band_name}_freq_domain{file_ext}"
        output_file = os.path.join(output_folder, output_file_name)
        
        sio.savemat(output_file, all_features)
        print(f"  已保存频域特征到: {output_file}")

def main():
    base_path = r"F:\软件\pycharm\untitled6\Processed_Frequency_Bands\R"
    output_base = os.path.join(base_path, "Frequency-domain features")
    os.makedirs(output_base, exist_ok=True)
    
    fs = 250
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'All']
    
    for band in frequency_bands:
        print(f"\n{'=' * 20} 处理频段: {band} {'=' * 20}")
        band_folder = os.path.join(base_path, band)
        
        if not os.path.isdir(band_folder):
            print(f"  警告: 频段文件夹 '{band_folder}' 不存在。跳过。")
            continue
            
        process_frequency_band_for_freq_domain(band_folder, band, output_base, fs=fs)

if __name__ == "__main__":
    main()