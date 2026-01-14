import numpy as np
import pandas as pd
import prosail
import random
import time

import matplotlib
# [关键] 在 AutoDL/Linux 服务器上必须设置 'Agg' 后端，否则 plt.figure() 会报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


# ==========================================
# 1. 核心配置字典 (Wheat Configuration)
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
    'ZS75': { # 灌浆期 (特殊处理)
        'Normal':   {'LAI': (2.5, 5.0), 'Cab': (20, 50), 'Cw': (0.008, 0.015), 'Cp': (0.0008, 0.0015), 'Ccbc': (0.008, 0.012)},
        'Abnormal': {'LAI': (0.5, 2.0), 'Cab': (0.1, 1),   'Cw': (0.003, 0.008), 'Cp': (0.0001, 0.0005), 'Ccbc': (0.008, 0.012)} 
        # ZS75 Abnormal 的 Cab/Cbrown 由代码逻辑动态生成，此处字典值仅占位
    }
}

# 全局参数与环境扰动 (Enhanced Version)
GLOBAL_PARAMS = {
    'N': (1.4, 1.6), 
    'LIDFa': (-0.5, -0.3), 
    'Hspot': (0.005, 0.05), # [优化] 动态热点因子
    'TTS': (20, 50), 
    'TTO': (0, 10), 
    'PSI': (0, 360),
    'Psoil': (0.3, 1.2),    # 土壤干湿因子 (对应 rsoil)
    'Car_Ratio': (0.20, 0.30) # [新增] 类胡萝卜素与叶绿素的比例约束
}

# 参数分组 (用于生长期混搭)
PARAM_GROUPS = {
    'Nitrogen': ['Cab', 'Cp'], 
    'Water':    ['Cw'],
    'Structure':['LAI']
}

# ==========================================
# 2. 辅助函数 (Utils)
# ==========================================
def plot_and_save_spectra(df, filename="wheat_spectra_prosail.png"):
    """绘制小麦冠层光谱曲线并保存"""
    print(f"正在绘制光谱曲线 (总样本数: {len(df)})...")
    wavelengths = np.arange(400, 2501)
    wavelength_cols = [f"w{w}" for w in wavelengths]

    plt.figure(figsize=(12, 7), dpi=150)
    
    stage_colors = {'ZS31': '#1f77b4', 'ZS47': '#2ca02c', 'ZS65': '#ff7f0e', 'ZS75': '#d62728'}
    
    for stage in df['Stage'].unique():
        stage_data = df[df['Stage'] == stage]
        n_display = min(20, len(stage_data))
        subset = stage_data.sample(n_display, random_state=42)
        color = stage_colors.get(stage, 'gray')
        
        first = True
        for _, row in subset.iterrows():
            plt.plot(wavelengths, row[wavelength_cols].values, color=color, alpha=0.6, linewidth=1, label=stage if first else "")
            first = False

    plt.title("Simulated Wheat Canopy Spectra (PROSAIL-PRO) - Bio-Enhanced", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)
    plt.ylim(0, 0.7)
    
    plt.axvspan(680, 760, color='gray', alpha=0.1, label='Red Edge')
    plt.axvline(1450, color='blue', linestyle=':', alpha=0.5, label='Water Abs (1450nm)')
    plt.axvline(1940, color='blue', linestyle=':', alpha=0.5, label='Water Abs (1940nm)')

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

def get_stress_combination(stage):
    """随机决定压力组合模式"""
    p = np.random.random()
    if p < 0.30: return ['Nitrogen']
    elif p < 0.60: return ['Water']
    elif p < 0.80: return ['Structure']
    else: return random.sample(['Nitrogen', 'Water', 'Structure'], np.random.choice([2, 3]))

def sample_range_values(range_dict, keys):
    """辅助批量采样"""
    res = {}
    for k in keys:
        if k in range_dict: res[k] = np.random.uniform(*range_dict[k])
    return res

# ==========================================
# 3. 核心采样逻辑 (Sampling Logic)
# ==========================================
def sample_advanced_stress(stage, config):
    """
    高级采样函数：包含 ZS75 安全锁、Cab-Car联动、动态Hspot
    """
    # A. 基础初始化 (通用参数)
    row = {
        'Stage': stage, 'Condition': 'Abnormal',
        'N': np.random.uniform(*GLOBAL_PARAMS['N']),
        'LIDFa': np.random.uniform(*GLOBAL_PARAMS['LIDFa']),
        'TTS': np.random.uniform(*GLOBAL_PARAMS['TTS']),
        'TTO': np.random.uniform(*GLOBAL_PARAMS['TTO']),
        'PSI': np.random.uniform(*GLOBAL_PARAMS['PSI']),
        'Psoil': np.random.uniform(*GLOBAL_PARAMS['Psoil']),
        'Hspot': np.random.uniform(*GLOBAL_PARAMS['Hspot']), # 动态 Hspot
        'Cbrown': 0.0 
    }
    
    row['Ccbc'] = np.random.uniform(*config['Abnormal']['Ccbc'])

    # =======================================================
    # B. 安全分流 (ZS75 特殊处理)
    # =======================================================
    if stage == 'ZS75':
        # 基础参数取 Abnormal 范围
        row.update(sample_range_values(config['Abnormal'], ['LAI', 'Cw', 'Cp']))
        
        if np.random.random() < 0.5: 
            # 剧本 A: 贪青 (Greedy)
            row['Cab'] = np.random.uniform(60, 80)
            row['Cbrown'] = np.random.uniform(0.0, 0.1)
            row['Label_Stress_Type'] = 'Greedy'
        else: 
            # 剧本 B: 枯死 (Dead)
            row['Cab'] = np.random.uniform(0.1, 10)
            row['Cbrown'] = np.random.uniform(0.5, 1.0)
            row['Label_Stress_Type'] = 'Dead'
        
        # [Bio-Constraint] Cab 确定后，计算 Car
        car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
        row['Car'] = max(0.0, row['Cab'] * car_ratio)
            
        return row 

    # =======================================================
    # C. 生长期混搭 (ZS31, ZS47, ZS65)
    # =======================================================
    active_stresses = get_stress_combination(stage)
    row['Label_Stress_Type'] = "+".join(active_stresses)

    target_keys = ['LAI', 'Cab', 'Cw', 'Cp']
    for key in target_keys:
        source_dict = config['Normal']
        for group in active_stresses:
            if key in PARAM_GROUPS[group]:
                source_dict = config['Abnormal']
                break
        low, high = source_dict[key]
        row[key] = np.random.uniform(low, high)
    
    # [Bio-Constraint] Cab 确定后，计算 Car
    car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
    row['Car'] = row['Cab'] * car_ratio
       
    return row

def generate_dataset_v3(samples_per_stage=1000):
    data_rows = []
    
    for stage, config in WHEAT_CONFIG.items():
        n_healthy = int(samples_per_stage * 0.5)
        n_stress = samples_per_stage - n_healthy
        
        # 1. 生成健康样本 (Normal)
        for _ in range(n_healthy):
            row = sample_range_values(config['Normal'], ['LAI', 'Cab', 'Cw', 'Cp', 'Ccbc'])
            
            # [Bio-Constraint] 立即计算 Car
            car_ratio = np.random.uniform(*GLOBAL_PARAMS['Car_Ratio'])
            calculated_car = row['Cab'] * car_ratio
            
            row.update({
                'Stage': stage, 'Condition': 'Normal', 'Label_Stress_Type': 'Healthy',
                'N': np.random.uniform(*GLOBAL_PARAMS['N']),
                'LIDFa': np.random.uniform(*GLOBAL_PARAMS['LIDFa']),
                'TTS': np.random.uniform(*GLOBAL_PARAMS['TTS']),
                'TTO': np.random.uniform(*GLOBAL_PARAMS['TTO']),
                'PSI': np.random.uniform(*GLOBAL_PARAMS['PSI']),
                'Psoil': np.random.uniform(*GLOBAL_PARAMS['Psoil']),
                'Hspot': np.random.uniform(*GLOBAL_PARAMS['Hspot']), # 动态 Hspot
                'Car': calculated_car,
                'Cbrown': 0.0 
            })
            
            if stage == 'ZS75': row['Cbrown'] = np.random.uniform(0.0, 0.3)
            data_rows.append(row)

        # 2. 生成压力样本 (Abnormal)
        for _ in range(n_stress):
            row = sample_advanced_stress(stage, config)
            data_rows.append(row)
            
    return pd.DataFrame(data_rows).sample(frac=1).reset_index(drop=True)

# ==========================================
# 4. 模拟运行 (Simulation Execution)
# ==========================================
def run_simulation(df):
    spectra_list = []
    valid_indices = []
    print(f"开始运行 PROSAIL-PRO 模拟 (N={len(df)})...")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        cm_total = row['Cp'] + row['Ccbc']
        try:
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
                rsoil = float(row['Psoil']), # 随机土壤亮度
                psoil = 1.0,                 # 必须设为1.0
                typelidf = 2
            )
            
            if rho is None: continue
            spectra_list.append(rho)
            valid_indices.append(idx)
            
        except Exception as e:
            if idx == 0: print(f"Error at row {idx}: {e}")
            pass 

    elapsed = time.time() - start_time
    print(f"模拟完成。耗时: {elapsed:.2f}s, 成功率: {len(valid_indices)}/{len(df)}")

    if not valid_indices: return pd.DataFrame()

    flat_spectra = [s.flatten() for s in spectra_list]
    n_bands = len(flat_spectra[0])
    spec_df = pd.DataFrame(flat_spectra, columns=[f"w{400+i}" for i in range(n_bands)])
    
    return pd.concat([df.loc[valid_indices].reset_index(drop=True), spec_df], axis=1)

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 生成参数配置 (在这里修改样本数量)
    samples_per_stage = 50 
    print(f"1. 生成参数配置 (每个生育期 {samples_per_stage} 个样本)...")
    df_params = generate_dataset_v3(samples_per_stage=samples_per_stage)
    
    # 2. 运行 PROSAIL 模拟
    final_df = run_simulation(df_params)

    # 4. 绘图验证
    plot_and_save_spectra(final_df, filename="Wheat_Spectra_Bio_Enhanced.png")