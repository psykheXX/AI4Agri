import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 0. 辅助函数：快速获取所有文件路径并混合
def get_all_file_paths(root_dir):
    # 假设你的目录结构是 root_dir/ZS31/*.npy, root_dir/ZS47/*.npy ...
    # 使用 glob 获取所有子文件夹下的 .npy 文件
    search_path = os.path.join(root_dir, '*', '*.npy')
    all_files = glob.glob(search_path)
    
    # 策略3：数据随机混合
    # 直接打乱文件路径列表，这样不同生育期的数据就混合了
    np.random.shuffle(all_files)
    return all_files

# --- 辅助函数：反归一化 (用于将模型预测结果还原为物理值) ---
def denormalize_preds(preds_norm, stats_dict, device='cpu'):
    """
    preds_norm: 模型输出的归一化后的预测值 [Batch, 3]
    stats_dict: 统计量字典
    """
    # 提取 min 和 max
    mins = torch.tensor([stats_dict['label_min'][k] for k in ['LAI', 'Cab', 'Cp']]).to(device)
    maxs = torch.tensor([stats_dict['label_max'][k] for k in ['LAI', 'Cab', 'Cp']]).to(device)
    
    # 反归一化公式: x_norm * (max - min) + min
    return preds_norm * (maxs - mins) + mins
    
# --- 辅助函数：反归一化 (用于将模型预测结果还原为物理值) ---
def denormalize_preds(preds_norm, stats_dict, device='cpu'):
    """
    preds_norm: 模型输出的归一化后的预测值 [Batch, 3]
    stats_dict: 统计量字典
    """
    # 提取 min 和 max
    mins = torch.tensor([stats_dict['label_min'][k] for k in ['LAI', 'Cab', 'Cp']]).to(device)
    maxs = torch.tensor([stats_dict['label_max'][k] for k in ['LAI', 'Cab', 'Cp']]).to(device)
    
    # 反归一化公式: x_norm * (max - min) + min
    return preds_norm * (maxs - mins) + mins

class HyperspectralDataset(Dataset):
    def __init__(self, file_paths, stats, mode='train', noise_level=0.01):
        """
        Args:
            file_paths: 包含所有 .npy 文件绝对路径的列表
            stats: 包含 spectra_mean, spectra_std, label_min, label_max 的字典
            mode: 'train' 或 'test'
            noise_level: 高斯噪声的标准差系数 (例如 0.01 代表 1% 的噪声)
        """
        self.file_paths = file_paths
        self.stats = stats
        self.mode = mode
        self.noise_level = noise_level
        
        # 预处理统计量转为 Tensor 加速计算
        self.spec_mean = torch.FloatTensor(stats['spectra_mean'])
        self.spec_std = torch.FloatTensor(stats['spectra_std']) + 1e-6 # 防止除零
        
        # 标签归一化参数
        self.label_keys = ['LAI', 'Cab', 'Cp']
        self.label_min = torch.tensor([stats['label_min'][k] for k in self.label_keys])
        self.label_max = torch.tensor([stats['label_max'][k] for k in self.label_keys])

    def __len__(self):
        return len(self.file_paths)

    def add_gaussian_noise(self, spectrum):
        """策略2：加入高斯白噪声模拟系统误差"""
        noise = torch.randn_like(spectrum) * self.noise_level
        return spectrum + noise

    def normalize_labels(self, labels):
        """对训练Label进行归一化 (Min-Max Normalization 映射到 0-1)"""
        # 公式: (x - min) / (max - min)
        return (labels - self.label_min) / (self.label_max - self.label_min)

    def standardize_spectrum(self, spectrum):
        """对光谱进行标准化 (Z-Score Standardization)"""
        # 公式: (x - mean) / std
        return (spectrum - self.spec_mean) / self.spec_std

    def __getitem__(self, idx):
        # 1. 懒加载数据
        file_path = self.file_paths[idx]
        try:
            # 注意：这里使用 item() 因为你的数据是 0-d array 包含 dict
            raw_data = np.load(file_path, allow_pickle=True).item()
        except Exception as e:
            # 容错处理：如果文件损坏，随机读取另一个（极少数情况）
            return self.__getitem__(np.random.randint(0, len(self)))

        # 提取光谱
        spectrum = torch.from_numpy(raw_data['spectra']).float()
        
        # 提取标签 [LAI, Cab, Cp]
        labels = torch.tensor([
            raw_data['params']['LAI'],
            raw_data['params']['Cab'],
            raw_data['params']['Cp']
        ]).float()

        # --- 训练模式逻辑 ---
        if self.mode == 'train':

            spectrum = self.add_gaussian_noise(spectrum)

            spectrum = self.standardize_spectrum(spectrum)
            
            labels = self.normalize_labels(labels)
            
        else:
            spectrum = self.standardize_spectrum(spectrum)

        spectrum = spectrum.unsqueeze(-1)

        return spectrum, labels

def main():
    # ================= 配置路径 =================
    DATA_ROOT = '/root/autodl-tmp/train_240/'  # 你的数据根目录
    STATS_PATH = 'dataset_240_stats.npy' # 刚才生成的统计文件路径
    BATCH_SIZE = 8

    # ================= 1. 加载统计量 =================
    if not os.path.exists(STATS_PATH):
        print(f"错误：找不到 {STATS_PATH}，请先运行 calculate_global_stats 生成统计文件。")
        return

    print(f"正在加载统计量: {STATS_PATH} ...")
    # 关键点：使用 .item() 还原字典对象
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    
    print("统计量加载成功！")
    print(f"光谱均值形状: {stats['spectra_mean'].shape}") # 应该是 (2101,)
    print(f"LAI 范围: {stats['label_min']['LAI']} ~ {stats['label_max']['LAI']}")

    # ================= 2. 获取文件列表 =================
    all_files = get_all_file_paths(DATA_ROOT)
    print(f"发现文件总数: {len(all_files)}")
    if len(all_files) == 0:
        print("错误：未找到数据文件，请检查 DATA_ROOT。")
        return

    # 简单切分一下用于测试
    train_files = all_files[:2000]
    test_files = all_files[2000:]

    # ================= 3. 测试 Train Dataset (训练模式) =================
    print("\n--- 测试 Train Dataset (Mode='train') ---")
    print("期待行为：开启噪声，Labels 被归一化到 0~1 之间")
    
    train_ds = HyperspectralDataset(train_files, stats, mode='train', noise_level=0.02)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    #以此取一个 Batch
    spectra, labels = next(iter(train_loader))
    
    print(f"Input Spectra Shape: {spectra.shape}") # [8, 2101]
    print(f"Output Labels Shape: {labels.shape}")  # [8, 3]
    
    # 验证 Label 归一化
    print(f"Labels Min (Batch): {labels.min():.4f}")
    print(f"Labels Max (Batch): {labels.max():.4f}")
    
    if labels.min() >= 0 and labels.max() <= 1.0:
        print("✅ 检测通过：训练集 Label 已归一化到 [0, 1] 区间。")
    else:
        print("❌ 检测失败：训练集 Label 超出 [0, 1] 区间！")

    # ================= 4. 测试 Test Dataset (测试模式) =================
    print("\n--- 测试 Test Dataset (Mode='test') ---")
    print("期待行为：无噪声，Labels 保持原始物理数值")
    
    test_ds = HyperspectralDataset(test_files, stats, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    spectra_test, labels_test = next(iter(test_loader))
    
    # 打印真实物理值示例
    print(f"真实标签示例 (前2行):\n{labels_test[:2]}")
    
    # 验证是否为原始物理值 (Cab通常几十，LAI通常0-10)
    # 只要最大值大于 1，说明肯定没有被归一化到 0-1
    if labels_test.max() > 1.0:
        print("✅ 检测通过：测试集 Label 为原始物理值。")
    else:
        print("⚠️ 警告：测试集 Label 看起来很小，可能是数据本身很小或者被错误归一化了。")

    # ================= 5. 模拟预测与反归一化验证 =================
    print("\n--- 模拟：模型预测与反归一化流程 ---")
    
    # 假设模型输出了归一化的结果 (这里我们用随机数模拟模型输出 0.5 左右的值)
    # 模拟 [Batch, 3]
    dummy_model_output = torch.rand((BATCH_SIZE, 3)) 
    print(f"模型输出 (Normalized, 模拟): \n{dummy_model_output[:2]}")
    
    # 反归一化
    restored_preds = denormalize_preds(dummy_model_output, stats)
    print(f"反归一化后的预测值 (Physical): \n{restored_preds[:2]}")
    
    print("\n✅ 所有 Dataset 测试流程结束。")

if __name__ == "__main__":
    # 确保在运行前定义了 HyperspectralDataset 类
    main()