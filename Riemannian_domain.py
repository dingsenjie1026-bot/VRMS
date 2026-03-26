import numpy as np
import scipy.io as sio
import os
from scipy.linalg import sqrtm, logm
import scipy.linalg as splalg
import warnings

warnings.filterwarnings('ignore')

# 定义通道名称和数量
ch_names = ['Fz', 'Cz', 'Pz', 'CP5', 'CP6', 'P3', 'P4']
n_channels = len(ch_names)

# ==========================================
# 1. 黎曼几何度量与矩阵运算函数 (数值稳定版)
# ==========================================

def log_euclidean_distance(cov1, cov2):
    """计算Log-Euclidean距离"""
    try:
        log_cov1 = logm(cov1)
        log_cov2 = logm(cov2)
        diff = log_cov1 - log_cov2
        # 使用 Frobenius 范数，取实部防止复数残留
        distance = np.linalg.norm(diff, ord='fro')
        return np.real(distance)
    except Exception as e:
        print(f"    Log-Euclidean 计算失败: {e}")
        return np.nan

def affine_invariant_distance(cov1, cov2):
    """
    计算Affine-invariant距离 (AIRM)
    优化: 使用广义特征值问题代替复杂的求逆和矩阵乘法，数值极其稳定
    """
    try:
        # 求解广义特征值 evals, 满足 cov2 @ v = evals * cov1 @ v
        evals = splalg.eigvals(cov2, cov1)
        # 舍弃因浮点误差产生的极小虚部，并过滤掉极小的非正特征值
        evals = np.real(evals)
        evals = np.clip(evals, 1e-10, np.inf)
        
        distance = np.sqrt(np.sum(np.log(evals) ** 2))
        return distance
    except Exception as e:
        print(f"    Affine-invariant 计算失败: {e}")
        return np.nan

def kullback_leibler_distance(cov1, cov2):
    """计算Kullback-Leibler距离"""
    try:
        cov2_inv = np.linalg.inv(cov2)
        term1 = np.trace(cov2_inv @ cov1)
        
        # 优化: 使用 slogdet 解决行列式下溢问题
        sign1, logdet1 = np.linalg.slogdet(cov1)
        sign2, logdet2 = np.linalg.slogdet(cov2)
        
        if sign1 <= 0 or sign2 <= 0:
            raise ValueError("协方差矩阵非正定，行列式异常")
            
        term2 = logdet2 - logdet1
        distance = 0.5 * (term1 + term2 - n_channels)
        return np.real(distance)
    except Exception as e:
        print(f"    KL距离 计算失败: {e}")
        return np.nan

def jeffreys_distance(cov1, cov2):
    """计算Jeffreys距离 (对称KL距离)"""
    kl1 = kullback_leibler_distance(cov1, cov2)
    kl2 = kullback_leibler_distance(cov2, cov1)
    if np.isnan(kl1) or np.isnan(kl2):
        return np.nan
    return 0.5 * (kl1 + kl2)

def wasserstein_distance(cov1, cov2):
    """计算Wasserstein距离 (Bures metric)"""
    try:
        cov1_sqrt = sqrtm(cov1)
        middle_matrix = cov1_sqrt @ cov2 @ cov1_sqrt
        middle_sqrt = sqrtm(middle_matrix)
        
        # 计算平方距离
        d_sq = np.trace(cov1 + cov2 - 2 * middle_sqrt)
        d_sq = np.real(d_sq)  # 必须强制转实数
        
        # 防止因浮点精度导致的微小负数
        distance = np.sqrt(np.clip(d_sq, 0, None))
        return distance
    except Exception as e:
        print(f"    Wasserstein 计算失败: {e}")
        return np.nan

def compute_covariance_matrix(eeg_data):
    """计算并正则化EEG数据的协方差矩阵"""
    data = eeg_data.T
    cov_matrix = np.cov(data)
    # 增加的正则化项保证矩阵在黎曼流形上是严格正定的 (SPD)
    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
    return cov_matrix

# ==========================================
# 2. 特征提取逻辑
# ==========================================

def extract_riemannian_features(eeg_data):
    features = {}
    cov_matrix = compute_covariance_matrix(eeg_data)
    identity_matrix = np.eye(n_channels)
    
    features['log_euclidean_dist'] = log_euclidean_distance(cov_matrix, identity_matrix)
    features['affine_invariant_dist'] = affine_invariant_distance(cov_matrix, identity_matrix)
    features['kullback_leibler_dist'] = kullback_leibler_distance(cov_matrix, identity_matrix)
    features['jeffreys_dist'] = jeffreys_distance(cov_matrix, identity_matrix)
    features['wasserstein_dist'] = wasserstein_distance(cov_matrix, identity_matrix)
    
    return features

# ==========================================
# 3. 批处理与主函数
# ==========================================

def get_variable_name_for_band(band_name):
    mapping = {
        'Delta': 'eeg_delta',
        'Theta': 'eeg_theta',
        'Alpha': 'eeg_alpha',
        'Beta': 'eeg_beta',
        'All': 'eeg_all'
    }
    return mapping.get(band_name, 'eeg_all')

def process_frequency_band_for_riemannian(folder_path, band_name, output_folder):
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
            print(f"  警告: 形状 {eeg_data.shape} 不符合预期 (N, 7)。跳过。")
            continue
            
        all_features = {}
        try:
            features = extract_riemannian_features(eeg_data)
            for feat_name, feat_value in features.items():
                all_features[feat_name] = feat_value
        except Exception as e:
            print(f"  处理文件 {mat_file} 时出错: {e}")
            continue
            
        file_name, file_ext = os.path.splitext(mat_file)
        output_file_name = f"{file_name}_{band_name}_riemannian{file_ext}"
        output_file = os.path.join(output_folder, output_file_name)
        
        sio.savemat(output_file, all_features)
        print(f"  已保存黎曼特征到: {output_file}")

def main():
    base_path = r"F:\软件\pycharm\untitled6\Processed_Frequency_Bands\R"
    output_base = os.path.join(base_path, "Riemannian-features")
    os.makedirs(output_base, exist_ok=True)
    
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'All']
    
    for band in frequency_bands:
        print(f"\n{'=' * 20} 处理频段: {band} {'=' * 20}")
        band_folder = os.path.join(base_path, band)
        
        if not os.path.isdir(band_folder):
            print(f"  警告: 频段文件夹 '{band_folder}' 不存在。跳过。")
            continue
            
        process_frequency_band_for_riemannian(band_folder, band, output_base)

if __name__ == "__main__":
    main()