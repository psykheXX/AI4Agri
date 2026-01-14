import numpy as np
import pandas as pd
import prosail
import random
import time

import matplotlib
# [关键] 在 AutoDL/Linux 服务器上必须设置 'Agg' 后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 0. [新增] 光谱波段配置
# ==========================================
# 目标：400~900nm, 240个波段
# 使用 np.linspace 生成均匀分布的波段中心
TARGET_WAVELENGTHS = np.linspace(400, 900, 190) 
# PROSAIL 原生输出波段 (400-2500, 1nm)
ORIGINAL_WAVELENGTHS = np.arange(400, 2501, 1)

print(f"设定光谱范围: 400-900nm, 波段数: {len(TARGET_WAVELENGTHS)}")
print(f"光谱分辨率: ~{(900-400)/(240-1):.2f} nm")

# ==========================================
# 1. 核心配置字典 (Wheat Configuration) - 保持不变
# ==========================================
WHEAT_CONFIG = {
    'ZS31': { 
        'Normal':   {'LAI': (2.0, 5.0), 'Cab': (40, 65), 'Cw': (0.012, 0.018), 'Cp': (0.0012, 0.0020), 'Ccbc': (0.003, 0.005)},
        'Abnormal': {'LAI': (1.0, 2.0), 'Cab': (15, 35), 'Cw': (0.006, 0.012), 'Cp': (0.0006, 0.0012), 'Ccbc': (0.003, 0.005)}
    },
    'ZS47': { 
        'Normal':   {'LAI': (4.0, 7.0), 'Cab': (30, 75), 'Cw': (0.015, 0.018), 'Cp': (0.0015, 0.0025), 'Ccbc': (0.005, 0.008)},
        'Abnormal': {'LAI': (1.5, 3.5), 'Cab': (20, 45), 'Cw': (0.008, 0.015), 'Cp': (0.0008, 0.0015), 'Ccbc': (0.005, 0.008)}
    },
    'ZS65': { 
        'Normal':   {'LAI': (3.5, 6.0), 'Cab': (40, 80), 'Cw': (0.012, 0.015), 'Cp': (0.0012, 0.0020), 'Ccbc': (0.006, 0.009)},
        'Abnormal': {'LAI': (1.5, 3.0), 'Cab': (15, 35), 'Cw': (0.005, 0.012), 'Cp': (0.0005, 0.0012), 'Ccbc': (0.006, 0.009)}
    },
    'ZS75': { 
        'Normal':   {'LAI': (2.5, 5.0), 'Cab': (20, 50), 'Cw': (0.008, 0.015), 'Cp': (0.0008, 0.0015), 'Ccbc': (0.008, 0.012)},
        'Abnormal': {'LAI': (0.5, 2.0), 'Cab': (0.1, 1),   'Cw': (0.003, 0.008), 'Cp': (0.0001, 0.0005), 'Ccbc': (0.008, 0.012)} 
    }
}

GLOBAL_PARAMS = {
    'N': (1.4, 1.6), 
    'LIDFa': (-0.5, -0.3), 
    'Hspot': (0.005, 0.05),
    'TTS': (20, 50), 
    'TTO': (0, 10), 
    'PSI': (0, 360),
    'Psoil': (0.3, 1.2),    
    'Car_Ratio': (0.20, 0.30) 
}

PARAM_GROUPS = {
    'Nitrogen': ['Cab', 'Cp'], 
    'Water':    ['Cw'],
    'Structure':['LAI']
}

# ==========================================
# 2. 辅助函数 (Utils) - [修改] 适配新波段绘图
# ==========================================
def plot_and_save_spectra(df, filename="wheat_spectra_prosail_240bands.png"):
    """绘制小麦冠层光谱曲线并保存 (适配 400-900nm)"""
    print(f"正在绘制光谱曲线 (总样本数: {len(df)})...")
    
    # 获取列名 (对应 run_simulation 中的命名规则)
    wavelength_cols = [f"w{w:.2f}" for w in TARGET_WAVELENGTHS]

    plt.figure(figsize=(10, 6), dpi=150)
    
    stage_colors = {'ZS31': '#1f77b4', 'ZS47': '#2ca02c', 'ZS65': '#ff7f0e', 'ZS75': '#d62728'}
    
    for stage in df['Stage'].unique():
        stage_data = df[df['Stage'] == stage]
        n_display = min(20, len(stage_data))
        subset = stage_data.sample(n_display, random_state=42)
        color = stage_colors.get(stage, 'gray')
        
        first = True
        for _, row in subset.iterrows():
            # 绘制重采样后的波段
            plt.plot(TARGET_WAVELENGTHS, row[wavelength_cols].values, color=color, alpha=1, linewidth=1, label=stage if first else "")
            first = False

    plt.title("Simulated Wheat Canopy Spectra (400-900nm, 190 Bands)", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)
    plt.ylim(0, 0.6) # 可见光/近红外区域反射率通常稍低，稍微调整ylim
    
    # 标注红边区域
    plt.axvspan(680, 760, color='gray', alpha=0.1, label='Red Edge')
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    try:
        plt.savefig(filename)
        print(f"✅ 绘图成功！图片已保存至: {filename}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
    finally:
        plt.close()

# ... (中间的 get_stress_combination, sample_range_values, sample_advanced_stress, generate_dataset_v3 保持完全不变，省略以节省篇幅) ...
# 为了代码可运行，这里简单补全必须的辅助函数
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
        car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
        row['Car'] = max(0.0, row['Cab'] * car_ratio)
        return row 

    active_stresses = get_stress_combination(stage)
    row['Label_Stress_Type'] = "+".join(active_stresses)
    target_keys = ['LAI', 'Cab', 'Cw', 'Cp']
    for key in target_keys:
        source_dict = config['Normal']
        for group in active_stresses:
            if key in PARAM_GROUPS[group]: source_dict = config['Abnormal']; break
        low, high = source_dict[key]
        row[key] = np.random.uniform(low, high)
    
    car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
    row['Car'] = row['Cab'] * car_ratio
    return row

def generate_dataset_v3(samples_per_stage=1000):
    data_rows = []
    for stage, config in WHEAT_CONFIG.items():
        n_healthy = int(samples_per_stage * 0.5)
        n_stress = samples_per_stage - n_healthy
        # Normal
        for _ in range(n_healthy):
            row = sample_range_values(config['Normal'], ['LAI', 'Cab', 'Cw', 'Cp', 'Ccbc'])
            car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
            calculated_car = row['Cab'] * car_ratio
            row.update({'Stage': stage, 'Condition': 'Normal', 'Label_Stress_Type': 'Healthy',
                        'N': np.random.uniform(*GLOBAL_PARAMS['N']), 'LIDFa': np.random.uniform(*GLOBAL_PARAMS['LIDFa']),
                        'TTS': np.random.uniform(*GLOBAL_PARAMS['TTS']), 'TTO': np.random.uniform(*GLOBAL_PARAMS['TTO']),
                        'PSI': np.random.uniform(*GLOBAL_PARAMS['PSI']), 'Psoil': np.random.uniform(*GLOBAL_PARAMS['Psoil']),
                        'Hspot': np.random.uniform(*GLOBAL_PARAMS['Hspot']), 'Car': calculated_car, 'Cbrown': 0.0})
            if stage == 'ZS75': row['Cbrown'] = np.random.uniform(0.0, 0.3)
            data_rows.append(row)
        # Stress
        for _ in range(n_stress):
            data_rows.append(sample_advanced_stress(stage, config))
    return pd.DataFrame(data_rows).sample(frac=1).reset_index(drop=True)

# ==========================================
# 4. 模拟运行 (Simulation Execution) - [核心修改]
# ==========================================
def run_simulation(df):
    spectra_list = []
    valid_indices = []
    print(f"开始运行 PROSAIL-PRO 模拟 (N={len(df)})...")
    print(f"  -> 将自动重采样至 400-900nm (240 Bands)")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        cm_total = row['Cp'] + row['Ccbc']
        try:
            # 1. 运行 PROSAIL 获取全谱 (400-2500nm, 1nm step)
            rho = prosail.run_prosail(
                n = float(row['N']), 
                cab = float(row['Cab']), 
                car = float(row['Car']), 
                cbrown = float(row['Cbrown']), 
                cw = float(row['Cw']), 
                cm = float(cm_total), 
                lai = float(row['LAI']), 
                lidfa = float(row['LIDFa']), 
                hspot = float(row['Hspot']), 
                tts = float(row['TTS']), 
                tto = float(row['TTO']), 
                psi = float(row['PSI']), 
                rsoil = float(row['Psoil']), 
                psoil = 1.0,                 
                typelidf = 2
            )
            
            if rho is None: continue
            
            # 2. [新增] 核心步骤：重采样 (Resampling)
            # 使用 numpy 的线性插值将 1nm 数据映射到新的 240 个波段上
            # rho.flatten() 确保输入是一维数组
            rho_resampled = np.interp(TARGET_WAVELENGTHS, ORIGINAL_WAVELENGTHS, rho.flatten())
            
            spectra_list.append(rho_resampled)
            valid_indices.append(idx)
            
        except Exception as e:
            if idx == 0: print(f"Error at row {idx}: {e}")
            pass 

    elapsed = time.time() - start_time
    print(f"模拟完成。耗时: {elapsed:.2f}s, 成功率: {len(valid_indices)}/{len(df)}")

    if not valid_indices: return pd.DataFrame()

    # 3. 创建 DataFrame，使用带两位小数的列名 (例如 w400.00, w402.09)
    spec_df = pd.DataFrame(spectra_list, columns=[f"w{w:.2f}" for w in TARGET_WAVELENGTHS])
    
    return pd.concat([df.loc[valid_indices].reset_index(drop=True), spec_df], axis=1)

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 生成参数配置
    samples_per_stage = 50 
    print(f"1. 生成参数配置 (每个生育期 {samples_per_stage} 个样本)...")
    df_params = generate_dataset_v3(samples_per_stage=samples_per_stage)
    
    # 2. 运行 PROSAIL 模拟 (含重采样)
    final_df = run_simulation(df_params)

    # 4. 绘图验证
    if not final_df.empty:
        plot_and_save_spectra(final_df, filename="Wheat_Spectra_400_900nm_240bands.png")
        
        # 打印一下列名验证
        print("\n生成的列名示例 (前5个):")
        print([c for c in final_df.columns if c.startswith('w')][:5])
        print("生成的列名示例 (后5个):")
        print([c for c in final_df.columns if c.startswith('w')][-5:])