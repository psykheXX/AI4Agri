import numpy as np
import pandas as pd
import prosail
import random
import time
import os
import concurrent.futures
import argparse 
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 0. [新增] 全局波段配置
# ==========================================
# 目标：400~900nm, 240个波段 (线性分布)
TARGET_WAVELENGTHS = np.linspace(400, 900, 190)
# PROSAIL 原生输出 (400-2500nm, 1nm间隔)
ORIGINAL_WAVELENGTHS = np.arange(400, 2501, 1)

print(f"设定光谱范围: 400-900nm, 波段数: {len(TARGET_WAVELENGTHS)}")

# ==========================================
# 1. 核心配置 (保持不变)
# ==========================================
WHEAT_CONFIG = {
    'ZS31': { # 拔节期
        'Normal':   {'LAI': (2.0, 5.0), 'Cab': (40, 65), 'Cw': (0.012, 0.018), 'Cp': (0.0012, 0.0020), 'Ccbc': (0.003, 0.005)},
        'Abnormal': {'LAI': (1.0, 2.0), 'Cab': (15, 35), 'Cw': (0.006, 0.012), 'Cp': (0.0006, 0.0012), 'Ccbc': (0.003, 0.005)}
    },
    'ZS47': { # 孕穗期
        'Normal':   {'LAI': (4.0, 7.0), 'Cab': (30, 75), 'Cw': (0.015, 0.018), 'Cp': (0.0015, 0.0025), 'Ccbc': (0.005, 0.008)},
        'Abnormal': {'LAI': (1.5, 3.5), 'Cab': (20, 45), 'Cw': (0.008, 0.015), 'Cp': (0.0008, 0.0015), 'Ccbc': (0.005, 0.008)}
    },
    'ZS65': { # 扬花期
        'Normal':   {'LAI': (3.5, 6.0), 'Cab': (40, 80), 'Cw': (0.012, 0.015), 'Cp': (0.0012, 0.0020), 'Ccbc': (0.006, 0.009)},
        'Abnormal': {'LAI': (1.5, 3.0), 'Cab': (15, 35), 'Cw': (0.005, 0.012), 'Cp': (0.0005, 0.0012), 'Ccbc': (0.006, 0.009)}
    },
    'ZS75': { # 灌浆期
        'Normal':   {'LAI': (2.5, 5.0), 'Cab': (20, 50), 'Cw': (0.008, 0.015), 'Cp': (0.0008, 0.0015), 'Ccbc': (0.008, 0.012)},
        'Abnormal': {'LAI': (0.5, 2.0), 'Cab': (0.1, 1),   'Cw': (0.003, 0.008), 'Cp': (0.0001, 0.0005), 'Ccbc': (0.008, 0.012)} 
    }
}

GLOBAL_PARAMS = {
    'N': (1.4, 1.6), 'LIDFa': (-0.5, -0.3), 'Hspot': (0.005, 0.05), 
    'TTS': (20, 50), 'TTO': (0, 10), 'PSI': (0, 360), 'Psoil': (0.3, 1.2), 'Car_Ratio': (0.20, 0.30)
}

PARAM_GROUPS = {'Nitrogen': ['Cab', 'Cp'], 'Water': ['Cw'], 'Structure':['LAI']}

# ==========================================
# 2. 逻辑采样 (保持不变)
# ==========================================
def get_stress_combination(stage):
    p = np.random.random()
    if p < 0.30: return ['Nitrogen']
    elif p < 0.60: return ['Water']
    elif p < 0.80: return ['Structure']
    else: return random.sample(['Nitrogen', 'Water', 'Structure'], np.random.choice([2, 3]))

def sample_range_values(range_dict, keys):
    res = {}
    for k in keys:
        if k in range_dict: res[k] = np.random.uniform(*range_dict[k])
    return res

def sample_advanced_stress(stage, config):
    row = {
        'Stage': stage, 'Condition': 'Abnormal',
        'N': np.random.uniform(*GLOBAL_PARAMS['N']),
        'LIDFa': np.random.uniform(*GLOBAL_PARAMS['LIDFa']),
        'TTS': np.random.uniform(*GLOBAL_PARAMS['TTS']),
        'TTO': np.random.uniform(*GLOBAL_PARAMS['TTO']),
        'PSI': np.random.uniform(*GLOBAL_PARAMS['PSI']),
        'Psoil': np.random.uniform(*GLOBAL_PARAMS['Psoil']),
        'Hspot': np.random.uniform(*GLOBAL_PARAMS['Hspot']),
        'Cbrown': 0.0 
    }
    row['Ccbc'] = np.random.uniform(*config['Abnormal']['Ccbc'])

    if stage == 'ZS75':
        row.update(sample_range_values(config['Abnormal'], ['LAI', 'Cw', 'Cp']))
        if np.random.random() < 0.5: 
            row['Cab'] = np.random.uniform(60, 80); row['Cbrown'] = np.random.uniform(0.0, 0.1); row['Label_Stress_Type'] = 'Greedy'
        else: 
            row['Cab'] = np.random.uniform(0.1, 10); row['Cbrown'] = np.random.uniform(0.5, 1.0); row['Label_Stress_Type'] = 'Dead'
        row['Car'] = max(0.0, row['Cab'] * np.random.uniform(*GLOBAL_PARAMS['Car_Ratio']))
        return row 

    active_stresses = get_stress_combination(stage)
    row['Label_Stress_Type'] = "+".join(active_stresses)

    for key in ['LAI', 'Cab', 'Cw', 'Cp']:
        source_dict = config['Normal']
        for group in active_stresses:
            if key in PARAM_GROUPS[group]: source_dict = config['Abnormal']; break
        row[key] = np.random.uniform(*source_dict[key])
    
    row['Car'] = row['Cab'] * np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
    return row

def generate_params_for_stage(stage, n_samples):
    config = WHEAT_CONFIG[stage]
    data_rows = []
    n_healthy = int(n_samples * 0.5)
    
    for _ in range(n_healthy):
        row = sample_range_values(config['Normal'], ['LAI', 'Cab', 'Cw', 'Cp', 'Ccbc'])
        row.update({
            'Stage': stage, 'Condition': 'Normal', 'Label_Stress_Type': 'Healthy',
            'N': np.random.uniform(*GLOBAL_PARAMS['N']), 'LIDFa': np.random.uniform(*GLOBAL_PARAMS['LIDFa']),
            'TTS': np.random.uniform(*GLOBAL_PARAMS['TTS']), 'TTO': np.random.uniform(*GLOBAL_PARAMS['TTO']),
            'PSI': np.random.uniform(*GLOBAL_PARAMS['PSI']), 'Psoil': np.random.uniform(*GLOBAL_PARAMS['Psoil']),
            'Hspot': np.random.uniform(*GLOBAL_PARAMS['Hspot']), 'Car': row['Cab'] * np.random.uniform(*GLOBAL_PARAMS['Car_Ratio']), 'Cbrown': 0.0 
        })
        if stage == 'ZS75': row['Cbrown'] = np.random.uniform(0.0, 0.3)
        data_rows.append(row)

    for _ in range(n_samples - n_healthy):
        data_rows.append(sample_advanced_stress(stage, config))
        
    random.shuffle(data_rows)
    return data_rows

# ==========================================
# 3. Worker 函数 (独立保存的核心) - [核心修改]
# ==========================================
def worker_save_single_file(args):
    """
    args: (row_dict, save_path)
    运行 PROSAIL -> 重采样至 400-900nm (240 bands) -> 保存
    """
    row, save_path = args # 解包参数
    
    try:
        cm_total = row['Cp'] + row['Ccbc']
        # 1. 获取全谱 (400-2500nm, 2101 bands)
        rho = prosail.run_prosail(
            n=float(row['N']), cab=float(row['Cab']), car=float(row['Car']), 
            cbrown=float(row['Cbrown']), cw=float(row['Cw']), cm=float(cm_total), 
            lai=float(row['LAI']), lidfa=float(row['LIDFa']), hspot=float(row['Hspot']), 
            tts=float(row['TTS']), tto=float(row['TTO']), psi=float(row['PSI']), 
            rsoil=float(row['Psoil']), psoil=1.0, typelidf=2
        )
        
        if rho is None: return False
        
        # 2. [重采样] 2101 bands -> 240 bands
        # rho.flatten() 是一维数组 (2101,)
        # np.interp(目标x, 原x, 原y) 进行线性插值
        rho_resampled = np.interp(TARGET_WAVELENGTHS, ORIGINAL_WAVELENGTHS, rho.flatten())
        
        # 3. 构造保存字典
        save_data = {
            'params': row,
            # 保存重采样后的数据，float32 足够精度且省空间
            'spectra': rho_resampled.astype(np.float32) 
        }
        
        # 保存
        np.save(save_path, save_data)
        return True
        
    except Exception as e:
        return False

# ==========================================
# 4. 并行管理器 (修改绘图函数以适配新波段)
# ==========================================
def process_stage_save_individual(stage, n_samples, output_dir):
    stage_dir = os.path.join(output_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    
    print(f"[{stage}] 正在生成参数...")
    params_list = generate_params_for_stage(stage, n_samples)
    
    tasks = []
    for i, row in enumerate(params_list):
        filename = f"{i:06d}.npy"
        save_path = os.path.join(stage_dir, filename)
        tasks.append((row, save_path))
        
    print(f"[{stage}] 开始模拟、重采样并保存 (CPU核数: {os.cpu_count()})...")
    success_count = 0
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(worker_save_single_file, tasks, chunksize=50), 
                            total=len(tasks), 
                            desc=f"Saving {stage}",
                            unit="file"))
        success_count = sum(results)
    
    print(f"[{stage}] 完成。成功保存: {success_count}/{n_samples} 个文件。")
    
    # 生成预览图
    plot_preview_from_files(stage_dir, stage)

def plot_preview_from_files(stage_dir, stage):
    """
    从保存的 .npy 文件中随机读取数据绘图
    [修改] 适配 400-900nm 坐标轴
    """
    files = [f for f in os.listdir(stage_dir) if f.endswith('.npy')]
    if not files: return
    
    sample_files = random.sample(files, min(20, len(files)))
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    # [修改] 使用全局定义的 TARGET_WAVELENGTHS 作为 X 轴
    x_axis = TARGET_WAVELENGTHS 
    
    for f in sample_files:
        path = os.path.join(stage_dir, f)
        data = np.load(path, allow_pickle=True).item()
        spec = data['spectra'] # 这里的 spec 已经是 240 维的了
        params = data['params']
        
        color = 'green' if params['Label_Stress_Type'] == 'Healthy' else 'red'
        plt.plot(x_axis, spec, color=color, alpha=0.5, linewidth=0.8)
        
    plt.title(f"{stage} Preview (400-900nm, 190 Bands)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 0.7) # 适当调整 Y 轴显示范围
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(stage_dir, "preview_check.png"))
    plt.close()

# ==========================================
# 5. 主入口 (保持不变)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wheat Hyperspectral Dataset Generator (PROSAIL)")
    
    parser.add_argument('--total_samples', type=int, default=1000000, help='生成的总样本数量')
    parser.add_argument('--output_dir', type=str, default='Wheat_Dataset_400_900_Individual', help='数据保存目录')
    
    args = parser.parse_args()
    
    TOTAL_SAMPLES = args.total_samples
    OUTPUT_DIR = args.output_dir
    
    samples_per_stage = TOTAL_SAMPLES // 4
    
    print(f"\n=== 开始生成 (PROSAIL 并行模式 - 光谱重采样版) ===")
    print(f"光谱范围: 400-900nm (240 Bands)")
    print(f"总目标 (Total): {TOTAL_SAMPLES}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=========================================\n")
    
    start_total = time.time()
    
    for stage in WHEAT_CONFIG.keys():
        process_stage_save_individual(stage, samples_per_stage, OUTPUT_DIR)
        print("-" * 30)
        
    print(f"\n🎉 所有任务完成！耗时: {(time.time() - start_total)/60:.2f} 分钟")