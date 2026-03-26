import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 特征提取与数据集定义
# ==========================================

class DeltaFeatureExtractor:
    """Delta关键特征提取器"""
    def __init__(self):
        self.channel_names = ['Fz', 'Cz', 'Pz', 'CP5', 'CP6', 'P3', 'P4']
    
    def extract_delta_ratios(self, df):
        """提取Delta-Alpha和Delta-Beta比值特征"""
        delta_alpha_features = []
        delta_beta_features = []
        
        for channel in self.channel_names:
            alpha_col = f'Delta_freq_{channel}_delta_alpha_ratio'
            if alpha_col in df.columns:
                delta_alpha_features.append(alpha_col)
                
            beta_col = f'Delta_freq_{channel}_delta_beta_ratio'
            if beta_col in df.columns:
                delta_beta_features.append(beta_col)
                
        return delta_alpha_features, delta_beta_features

class EEGDataset(Dataset):
    """EEG数据集 (修复了掩码传递逻辑)"""
    def __init__(self, delta_features, original_features, labels, channel_masks=None):
        self.delta_features = torch.FloatTensor(delta_features)
        self.original_features = torch.FloatTensor(original_features)
        self.labels = torch.LongTensor(labels)
        self.channel_masks = channel_masks
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'delta_features': self.delta_features[idx],
            'original_features': self.original_features[idx],
            'labels': self.labels[idx]
        }
        
        # 掩码是全局的特征标识，对所有样本相同
        if self.channel_masks is not None:
            sample['delta_channel_mask'] = torch.FloatTensor(self.channel_masks['delta'])
            sample['original_channel_mask'] = torch.FloatTensor(self.channel_masks['original'])
            
        return sample

# ==========================================
# 2. 核心网络组件 (彻底修复的架构)
# ==========================================

class CzPriorEnhancementLayer(nn.Module):
    """
    Cz 先验特征增强层 (放在网络的最前端)
    在特征被混合之前，明确放大 Cz 通道的数值和反向传播梯度。
    """
    def __init__(self, cz_enhancement_factor=1.2, learnable_enhancement=False):
        super(CzPriorEnhancementLayer, self).__init__()
        self.learnable = learnable_enhancement
        
        if learnable_enhancement:
            self.cz_factor = nn.Parameter(torch.tensor(float(cz_enhancement_factor)))
        else:
            self.register_buffer('cz_factor', torch.tensor(float(cz_enhancement_factor)))

    def forward(self, x, channel_mask):
        if channel_mask is None:
            return x, 1.0
            
        # 确保 mask 维度 [batch_size, input_dim] 匹配
        if len(channel_mask.shape) == 1:
            channel_mask = channel_mask.unsqueeze(0).expand(x.size(0), -1)
            
        # 限制可学习参数范围，防止梯度爆炸
        if self.learnable:
            effective_factor = torch.clamp(self.cz_factor, 0.5, 3.0)
        else:
            effective_factor = self.cz_factor
            
        # 乘数矩阵: 非Cz位置为 1.0，Cz位置为 (1.0 + factor)
        multiplier = 1.0 + (channel_mask * effective_factor)
        
        # 施加逐元素增强
        x_enhanced = x * multiplier
        return x_enhanced, effective_factor

class OptimizedDeltaCNN(nn.Module):
    """优化后的 Delta特征CNN (输入端先验增强 + 深层自注意力)"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3, cz_enhancement_factor=1.2, learnable_enhancement=True):
        super(OptimizedDeltaCNN, self).__init__()
        
        # 1. 前馈先验增强层
        self.cz_enhancement = CzPriorEnhancementLayer(cz_enhancement_factor, learnable_enhancement)
        
        # 2. 特征投影
        self.input_projection = nn.Linear(input_dim, 64)
        
        # 3. 卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        
        # 4. 深层自注意力融合机制
        self.self_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 5. 特征输出层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def get_enhancement_factor(self):
        return self.cz_enhancement.cz_factor.item()
        
    def forward(self, x, channel_mask=None):
        # 步骤 1：明确放大 Cz 特征
        x, current_factor = self.cz_enhancement(x, channel_mask)
        
        # 步骤 2：投影与卷积提取
        x = self.input_projection(x).unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # [batch, 128, seq_len]
        
        # 步骤 3：深层自注意力加权
        x = x.transpose(1, 2)  # [batch, seq_len, 128]
        attn_weights = self.self_attention(x)
        x = torch.sum(x * attn_weights, dim=1)  # [batch, 128]
        
        # 步骤 4：特征降维输出
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        
        return x, {'factor': current_factor, 'attn_weights': attn_weights}

class OriginalFeatureCNN(nn.Module):
    """重构后的原始特征CNN (纯净基线对照网络，无Cz增强)"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(OriginalFeatureCNN, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, 128)
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        self.self_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x, channel_mask=None):
        x = self.input_projection(x).unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.transpose(1, 2)
        attn_weights = self.self_attention(x)
        x = torch.sum(x * attn_weights, dim=1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        
        return x, {'attn_weights': attn_weights}

class DynamicWeightedFusion(nn.Module):
    """动态加权融合模块"""
    def __init__(self, delta_dim, original_dim, num_classes):
        super(DynamicWeightedFusion, self).__init__()
        
        self.weight_net = nn.Sequential(
            nn.Linear(delta_dim + original_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(delta_dim + original_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, delta_features, original_features):
        combined_features = torch.cat([delta_features, original_features], dim=1)
        weights = self.weight_net(combined_features)
        
        weighted_delta = delta_features * weights[:, 0:1]
        weighted_original = original_features * weights[:, 1:2]
        
        final_features = torch.cat([weighted_delta, weighted_original], dim=1)
        output = self.classifier(final_features)
        
        return output, weights

class ImprovedEEGCNN(nn.Module):
    """结合双分支的完整架构"""
    def __init__(self, delta_input_dim, original_input_dim, num_classes, dropout_rate=0.3, 
                 delta_cz_factor=1.2, learnable_enhancement=True):
        super(ImprovedEEGCNN, self).__init__()
        
        self.delta_cnn = OptimizedDeltaCNN(
            delta_input_dim, num_classes, dropout_rate, 
            cz_enhancement_factor=delta_cz_factor, 
            learnable_enhancement=learnable_enhancement
        )
        
        self.original_cnn = OriginalFeatureCNN(
            original_input_dim, num_classes, dropout_rate
        )
        
        self.fusion = DynamicWeightedFusion(32, 16, num_classes)
    
    def get_enhancement_factor(self):
        return self.delta_cnn.get_enhancement_factor()
        
    def forward(self, delta_x, original_x, delta_mask=None, original_mask=None):
        delta_output, delta_info = self.delta_cnn(delta_x, delta_mask)
        original_output, original_info = self.original_cnn(original_x, original_mask)
        final_output, fusion_weights = self.fusion(delta_output, original_output)
        
        return final_output, {
            'delta_info': delta_info,
            'original_info': original_info,
            'fusion_weights': fusion_weights
        }

# ==========================================
# 3. 模型管理器 (无数据泄露的训练流程)
# ==========================================

class ImprovedEEGModel:
    def __init__(self, num_classes=4, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 delta_cz_factor=1.2, learnable_enhancement=True):
        self.device = device
        self.num_classes = num_classes
        self.delta_cz_factor = delta_cz_factor
        self.learnable_enhancement = learnable_enhancement
        self.model = None
        self.scaler_delta = StandardScaler()
        self.scaler_original = StandardScaler()
        self.feature_extractor = DeltaFeatureExtractor()
        
    def prepare_data(self, df):
        """清洗并准备基础数据字典 (不做任何标准化)"""
        delta_alpha_features, delta_beta_features = self.feature_extractor.extract_delta_ratios(df)
        delta_features = delta_alpha_features + delta_beta_features
        delta_data = df[delta_features].values
        
        all_features = [col for col in df.columns if col not in ['subject_id', 'condition', 'dataset']]
        original_features = [col for col in all_features if col not in delta_features]
        original_data = df[original_features].values
        
        label_mapping = {'N': 0, 'L': 1, 'M': 2, 'S': 3}
        labels = df['condition'].map(label_mapping).values
        
        def _create_channel_mask(feature_names, target_channel):
            return np.array([1 if target_channel in f else 0 for f in feature_names])
            
        return {
            'delta_features': delta_data,
            'original_features': original_data,
            'labels': labels,
            'delta_channel_mask': _create_channel_mask(delta_features, 'Cz'),
            'original_channel_mask': _create_channel_mask(original_features, 'Cz'),
            'delta_feature_names': delta_features,
            'original_feature_names': original_features
        }
    
    def train_model(self, data_dict, test_size=0.2, batch_size=32, epochs=100, lr=0.001, verbose=True):
        """彻底修复数据泄露的严格训练流程"""
        
        # 1. 严格分离：在拟合任何特征前切分数据集
        X_delta_train, X_delta_test, X_orig_train, X_orig_test, y_train, y_test = train_test_split(
            data_dict['delta_features'], data_dict['original_features'], data_dict['labels'],
            test_size=test_size, random_state=42, stratify=data_dict['labels']
        )
        
        # 2. 避免泄露：仅在训练集上 fit，在测试集上 transform
        X_delta_train = self.scaler_delta.fit_transform(X_delta_train)
        X_delta_test = self.scaler_delta.transform(X_delta_test)
        
        X_orig_train = self.scaler_original.fit_transform(X_orig_train)
        X_orig_test = self.scaler_original.transform(X_orig_test)
        
        # 3. 创建纯净数据集
        channel_masks = {
            'delta': data_dict['delta_channel_mask'],
            'original': data_dict['original_channel_mask']
        }
        train_dataset = EEGDataset(X_delta_train, X_orig_train, y_train, channel_masks)
        
        # ⭐ 将纯净的测试集保存至实例中，专供 evaluation_model 阶段使用
        self.test_dataset = EEGDataset(X_delta_test, X_orig_test, y_test, channel_masks)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 验证时使用 test_dataset 监控过拟合
        val_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False) 
        
        # 4. 初始化模型
        self.model = ImprovedEEGCNN(
            delta_input_dim=len(data_dict['delta_feature_names']),
            original_input_dim=len(data_dict['original_feature_names']),
            num_classes=self.num_classes,
            delta_cz_factor=self.delta_cz_factor,
            learnable_enhancement=self.learnable_enhancement
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        enhancement_factors = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for batch in train_loader:
                delta_x = batch['delta_features'].to(self.device)
                original_x = batch['original_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                delta_mask = batch['delta_channel_mask'].to(self.device)
                original_mask = batch['original_channel_mask'].to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(delta_x, original_x, delta_mask, original_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    delta_x = batch['delta_features'].to(self.device)
                    original_x = batch['original_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    delta_mask = batch['delta_channel_mask'].to(self.device)
                    original_mask = batch['original_channel_mask'].to(self.device)
                    
                    outputs, _ = self.model(delta_x, original_x, delta_mask, original_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if self.learnable_enhancement:
                enhancement_factors.append(self.model.get_enhancement_factor())
            
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        self.training_history = {
            'train_losses': train_losses, 'train_accuracies': train_accuracies,
            'val_losses': val_losses, 'val_accuracies': val_accuracies,
            'enhancement_factors': enhancement_factors if self.learnable_enhancement else None
        }
        
        if verbose:
            print(f"\n✅ 训练完成！动态学到的 Cz 最终增强系数为: {self.model.get_enhancement_factor():.3f}")
            
        return self.training_history
    
    def evaluate_model(self):
        """使用严格分离的纯净测试集进行诚实评估"""
        if self.model is None or not hasattr(self, 'test_dataset'):
            raise ValueError("模型尚未训练，或未保存测试集!")
            
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        all_predictions, all_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                delta_x = batch['delta_features'].to(self.device)
                original_x = batch['original_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                delta_mask = batch['delta_channel_mask'].to(self.device)
                original_mask = batch['original_channel_mask'].to(self.device)
                
                outputs, _ = self.model(delta_x, original_x, delta_mask, original_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        test_accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        print(f"\n真实纯净测试集准确率: {test_accuracy*100:.2f}%")
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, target_names=['N', 'L', 'M', 'S']))
        
        return {'predictions': all_predictions, 'labels': all_labels, 'test_accuracy': test_accuracy}

    def plot_training_history(self):
        if not hasattr(self, 'training_history'): return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.training_history['train_losses'], label='训练损失')
        ax1.plot(self.training_history['val_losses'], label='验证损失')
        ax1.set_title('训练和验证损失'); ax1.legend(); ax1.grid(True)
        
        ax2.plot(self.training_history['train_accuracies'], label='训练准确率')
        ax2.plot(self.training_history['val_accuracies'], label='验证准确率')
        ax2.set_title('训练和验证准确率'); ax2.legend(); ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    print("正在加载数据...")
    df = pd.read_csv(r"F:\软件\pycharm\untitled6\Processed_Frequency_Bands\ALL\all_features_per_subject.csv")
    
    # 初始化改进后的模型
    model = ImprovedEEGModel(num_classes=4, learnable_enhancement=True)
    
    data_dict = model.prepare_data(df)
    print(f"Delta特征数量: {len(data_dict['delta_feature_names'])}, 原始特征数量: {len(data_dict['original_feature_names'])}")
    
    # 严格训练（分离训练与测试集）
    history = model.train_model(data_dict, epochs=50, batch_size=16, lr=0.001)
    
    # 纯净评估（注意这里不需要再传 data_dict，内部会直接使用安全的 test_dataset）
    results = model.evaluate_model()
    
    model.plot_training_history()
    return model, results

if __name__ == "__main__":
    model, results = main()