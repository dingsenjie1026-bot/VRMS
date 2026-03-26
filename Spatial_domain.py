import numpy as np
import scipy.io as sio
import os
from scipy import signal
from scipy.stats import pearsonr
import warnings
import networkx as nx
from scipy.signal import hilbert
import itertools
from math import log, floor

warnings.filterwarnings('ignore')

# 定义通道名称和数量
ch_names = ['Fz', 'Cz', 'Pz', 'CP5', 'CP6', 'P3', 'P4']
n_channels = len(ch_names)

# 兼容不同版本的 networkx
def create_graph_from_matrix(matrix):
    """从矩阵创建无向图，自动移除自环，兼容不同版本的 networkx"""
    try:
        G = nx.from_numpy_array(matrix)
    except AttributeError:
        try:
            G = nx.from_numpy_matrix(matrix)
        except AttributeError:
            G = nx.Graph()
            n = matrix.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i, j] != 0:
                        G.add_edge(i, j, weight=matrix[i, j])
    
    # 清理可能存在的自环 (Self-loops) 以防万一
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ==========================================
# 1. 功能连接指标计算函数
# ==========================================

def calculate_mutual_information(x, y, bins=10):
    """计算两个信号之间的互信息 (Mutual Information)"""
    c_xy = np.histogram2d(x, y, bins)[0]
    c_x = np.histogram(x, bins)[0]
    c_y = np.histogram(y, bins)[0]
    
    p_xy = c_xy / np.sum(c_xy)
    p_x = c_x / np.sum(c_x)
    p_y = c_y / np.sum(c_y)
    
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

def calculate_correlation(x, y):
    """计算两个信号之间的相关系数"""
    corr, _ = pearsonr(x, y)
    return corr

def calculate_phase_locking_value(x, y):
    """计算相位锁定值 (PLV)"""
    analytic_x = hilbert(x)
    analytic_y = hilbert(y)
    phase_x = np.angle(analytic_x)
    phase_y = np.angle(analytic_y)
    phase_diff = phase_x - phase_y
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def calculate_coherence(x, y, fs=250):
    """计算相干性 (Coherence)"""
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=min(256, len(x)))
    return np.mean(Cxy)

def calculate_phase_lag_index(x, y):
    """计算相位滞后指数 (PLI)"""
    analytic_x = hilbert(x)
    analytic_y = hilbert(y)
    phase_x = np.angle(analytic_x)
    phase_y = np.angle(analytic_y)
    phase_diff = phase_x - phase_y
    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    return pli

def calculate_h_index(x):
    """计算信号的H指数 (反映信号不规则性/复杂度)"""
    diff_x = np.diff(x)
    h = np.mean(diff_x ** 2) / (np.mean(np.abs(diff_x)) ** 2 + 1e-10)
    return h

# ==========================================
# 2. 脑网络指标计算函数
# ==========================================

def calculate_clustering_coefficient(connectivity_matrix, threshold=None):
    """计算聚类系数"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
    G = create_graph_from_matrix(connectivity_matrix)
    clustering_coeffs = nx.clustering(G)
    return np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0

def calculate_global_efficiency(connectivity_matrix, threshold=None):
    """计算全局效率"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
    G = create_graph_from_matrix(connectivity_matrix)
    n = len(G)
    if n < 2:
        return 0
    return nx.global_efficiency(G)

def calculate_local_efficiency(connectivity_matrix, threshold=None):
    """计算局部效率"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
    G = create_graph_from_matrix(connectivity_matrix)
    return nx.local_efficiency(G)

def calculate_nodal_vulnerability(connectivity_matrix, threshold=None):
    """计算节点脆弱性"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
        
    global_eff = calculate_global_efficiency(connectivity_matrix)
    nodal_vuln = []
    n = connectivity_matrix.shape[0]
    
    for i in range(n):
        reduced_matrix = np.delete(np.delete(connectivity_matrix, i, axis=0), i, axis=1)
        reduced_global_eff = calculate_global_efficiency(reduced_matrix)
        vuln = (global_eff - reduced_global_eff) / global_eff if global_eff != 0 else 0
        nodal_vuln.append(vuln)
        
    return np.mean(nodal_vuln)

def calculate_rich_club_coefficient(connectivity_matrix, threshold=None):
    """计算 Rich Club 系数"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
    G = create_graph_from_matrix(connectivity_matrix)
    degrees = np.array([d for n, d in G.degree()])
    if len(degrees) == 0:
        return 0
        
    mean_degree = np.mean(degrees)
    rich_nodes = [i for i, d in enumerate(degrees) if d > mean_degree]
    if len(rich_nodes) < 2:
        return 0
        
    rich_subgraph = G.subgraph(rich_nodes)
    rich_edges = rich_subgraph.number_of_edges()
    possible_edges = len(rich_nodes) * (len(rich_nodes) - 1) / 2
    
    return rich_edges / possible_edges if possible_edges > 0 else 0

def calculate_edge_betweenness(connectivity_matrix, threshold=None):
    """计算平均边介数 (使用优化的自带函数)"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
    G = create_graph_from_matrix(connectivity_matrix)
    if G.number_of_edges() == 0:
        return 0
    
    edge_betweenness = nx.edge_betweenness_centrality(G)
    return np.mean(list(edge_betweenness.values()))

def calculate_small_world_propensity(connectivity_matrix, threshold=None):
    """计算小世界属性 Sigma"""
    if threshold is not None:
        connectivity_matrix = (connectivity_matrix > threshold).astype(float)
        
    G = create_graph_from_matrix(connectivity_matrix)
    n = len(G)
    if n < 2 or G.number_of_edges() == 0:
        return 0

    C_real = calculate_clustering_coefficient(connectivity_matrix)
    
    # 修复断连图的路径长度计算错误
    try:
        if nx.is_connected(G):
            L_real = nx.average_shortest_path_length(G)
        else:
            # 提取最大连通子图计算路径
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            L_real = nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0
    except Exception:
        L_real = 0
        
    # 随机网络近似 (保证避免除以0)
    C_rand = np.mean(connectivity_matrix) if np.mean(connectivity_matrix) > 0 else 0.1
    avg_degree = np.mean([d for n, d in G.degree()])
    L_rand = np.log(n) / np.log(avg_degree + 1e-10) if avg_degree > 1 else 0
    
    if C_rand > 0 and L_rand > 0 and L_real > 0:
        sigma = (C_real / C_rand) / (L_real / L_rand)
        return sigma
    return 0

# ==========================================
# 3. 提取所有空域特征
# ==========================================

def extract_spatial_domain_features(eeg_data, fs=250, conn_threshold=0.5):
    features = {}
    n_channels = eeg_data.shape[1]
    
    mi_matrix = np.zeros((n_channels, n_channels))
    corr_matrix = np.zeros((n_channels, n_channels))
    plv_matrix = np.zeros((n_channels, n_channels))
    coh_matrix = np.zeros((n_channels, n_channels))
    pli_matrix = np.zeros((n_channels, n_channels))
    h_values = np.zeros(n_channels)
    
    for i in range(n_channels):
        h_values[i] = calculate_h_index(eeg_data[:, i])
        features[f'{ch_names[i]}_h_index'] = h_values[i]
        
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            x = eeg_data[:, i]
            y = eeg_data[:, j]
            
            mi_matrix[i, j] = mi_matrix[j, i] = calculate_mutual_information(x, y)
            corr_matrix[i, j] = corr_matrix[j, i] = calculate_correlation(x, y)
            plv_matrix[i, j] = plv_matrix[j, i] = calculate_phase_locking_value(x, y)
            coh_matrix[i, j] = coh_matrix[j, i] = calculate_coherence(x, y, fs=fs)
            pli_matrix[i, j] = pli_matrix[j, i] = calculate_phase_lag_index(x, y)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            features[f'{ch_names[i]}_{ch_names[j]}_mi'] = mi_matrix[i, j]
            features[f'{ch_names[i]}_{ch_names[j]}_corr'] = corr_matrix[i, j]
            features[f'{ch_names[i]}_{ch_names[j]}_plv'] = plv_matrix[i, j]
            features[f'{ch_names[i]}_{ch_names[j]}_coh'] = coh_matrix[i, j]
            features[f'{ch_names[i]}_{ch_names[j]}_pli'] = pli_matrix[i, j]
            
    # 【核心修复】计算图论特征前，强制将对角线(自环)设为0
    corr_matrix_graph = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_graph, 0)
    
    features['clustering_coefficient'] = calculate_clustering_coefficient(corr_matrix_graph, conn_threshold)
    features['global_efficiency'] = calculate_global_efficiency(corr_matrix_graph, conn_threshold)
    features['local_efficiency'] = calculate_local_efficiency(corr_matrix_graph, conn_threshold)
    features['nodal_vulnerability'] = calculate_nodal_vulnerability(corr_matrix_graph, conn_threshold)
    features['rich_club_coefficient'] = calculate_rich_club_coefficient(corr_matrix_graph, conn_threshold)
    features['edge_betweenness'] = calculate_edge_betweenness(corr_matrix_graph, conn_threshold)
    features['small_world_propensity'] = calculate_small_world_propensity(corr_matrix_graph, conn_threshold)
    
    return features

# ==========================================
# 4 & 5. 批处理与主函数
# ==========================================

def process_frequency_band_for_spatial_domain(folder_path, band_name, output_folder, fs=250):
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    var_name_to_load = f'eeg_{band_name.lower()}'
    
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
            
        try:
            features = extract_spatial_domain_features(eeg_data, fs=fs)
        except Exception as e:
            print(f"  处理文件 {mat_file} 时出错: {e}")
            continue
            
        file_name, file_ext = os.path.splitext(mat_file)
        output_file_name = f"{file_name}_{band_name}_spatial_domain{file_ext}"
        output_file = os.path.join(output_folder, output_file_name)
        
        sio.savemat(output_file, features)
        print(f"  已保存空域特征到: {output_file}")

def main():
    base_path = r"F:\软件\pycharm\untitled6\Processed_Frequency_Bands\B2"
    output_base = os.path.join(base_path, "Spatial-domain features")
    os.makedirs(output_base, exist_ok=True)
    fs = 250
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'All']
    
    for band in frequency_bands:
        print(f"\n{'=' * 20} 处理频段: {band} {'=' * 20}")
        band_folder = os.path.join(base_path, band)
        
        if not os.path.isdir(band_folder):
            print(f"  警告: 频段文件夹 '{band_folder}' 不存在。跳过。")
            continue
            
        process_frequency_band_for_spatial_domain(band_folder, band, output_base, fs=fs)

if __name__ == "__main__":
    main()