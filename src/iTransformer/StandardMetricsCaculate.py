import os
import glob
import numpy as np
import torch
from tqdm import tqdm  # 建议安装 tqdm 显示进度: pip install tqdm

def get_all_npy_paths(root_dir):
    """
    递归查找 root_dir 下所有子文件夹中的 .npy 文件
    """
    # 使用 glob 递归查找所有 .npy 文件
    # root_dir 后面加上 /**/*.npy 并开启 recursive=True
    search_pattern = os.path.join(root_dir, '**', '*.npy')
    all_files = glob.glob(search_pattern, recursive=True)
    return all_files

def calculate_global_stats(root_dir, sample_ratio=0.05, min_samples=800000):
    """
    Args:
        root_dir: 数据集根目录 (包含 ZS31, ZS47 等文件夹)
        sample_ratio: 采样比例 (默认 5%)
        min_samples: 最少采样样本数 (防止数据量太少导致统计不准)
    """
    print(f"1. 正在扫描 {root_dir} 下的所有文件...")
    all_files = get_all_npy_paths(root_dir)
    total_files = len(all_files)
    
    if total_files == 0:
        raise ValueError(f"错误: 在 {root_dir} 下未找到任何 .npy 文件，请检查路径。")
    
    print(f"   共发现 {total_files} 个文件。")

    # 确定采样数量
    sample_count = int(total_files * sample_ratio)
    sample_count = max(sample_count, min_samples)
    sample_count = min(sample_count, total_files) # 不能超过总数

    print(f"2. 随机抽取 {sample_count} 个样本进行统计量计算...")
    
    # 随机打乱并切片
    np.random.shuffle(all_files)
    sampled_files = all_files[:sample_count]

    # 容器
    spectra_accum = []
    # 只需要统计这三个 Label
    labels_accum = {'LAI': [], 'Cab': [], 'Cp': []}

    # 遍历采样文件
    for f in tqdm(sampled_files, desc="Loading Samples"):
        try:
            # --- 关键修改点 ---
            # 1. allow_pickle=True: 因为里面存的是 dict 对象
            # 2. .item(): 将 0-d array 转回 Python dict
            data_obj = np.load(f, allow_pickle=True)
            data = data_obj.item() 
            
            # --- 提取数据 ---
            # 根据你的截图，spectra 在一级目录
            spec = data['spectra']  # shape: (2101,)
            
            # Label 在 params 字典里
            params = data['params']
            
            spectra_accum.append(spec)
            labels_accum['LAI'].append(params['LAI'])
            labels_accum['Cab'].append(params['Cab'])
            labels_accum['Cp'].append(params['Cp'])
            
        except Exception as e:
            print(f"警告: 无法读取文件 {f}, 错误: {e}")
            continue

    print("3. 正在计算均值和方差 (这可能需要一点时间)...")
    
    # 转换为 NumPy 数组进行向量化计算
    # shape: [N_samples, 2101]
    spectra_matrix = np.stack(spectra_accum)
    
    # 计算统计量
    stats = {
        # 光谱的均值和标准差 (按波段计算，axis=0)
        'spectra_mean': spectra_matrix.mean(axis=0),
        'spectra_std': spectra_matrix.std(axis=0),
        
        # Label 的最大最小值 (用于归一化)
        'label_min': {k: np.min(v) for k, v in labels_accum.items()},
        'label_max': {k: np.max(v) for k, v in labels_accum.items()}
    }
    
    print("统计完成！")
    print(f"光谱 Mean Shape: {stats['spectra_mean'].shape}")
    print("标签范围示例 (LAI):", stats['label_min']['LAI'], "~", stats['label_max']['LAI'])
    
    return stats

# --- 使用示例 ---
if __name__ == "__main__":
    DATA_ROOT = '/root/autodl-tmp/train_190/' 
    
    try:
        # 计算统计量
        stats = calculate_global_stats(DATA_ROOT, sample_ratio=0.02)
        print(stats)
        
        np.save('dataset_190_stats.npy', stats)
        print("统计量已保存为 dataset_stats.npy")
        
    except Exception as e:
        print(e)