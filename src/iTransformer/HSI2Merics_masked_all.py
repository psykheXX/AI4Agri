import os
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from iTransformerSeq2Vec import Model 
from types import SimpleNamespace
import json
from LoadData import denormalize_preds

# --- 配置部分 ---
STATS_PATH = 'dataset_190_stats.npy'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    flat_config = {}
    for section in config_dict:
        for key, value in config_dict[section].items():
            flat_config[key] = value
    return SimpleNamespace(**flat_config)

def calculate_mask(hsi_cube, red_idx=103, nir_idx=152, threshold=0.03):
    """ 计算土壤掩膜 (NDVI) """
    red_band = hsi_cube[:, :, red_idx]
    nir_band = hsi_cube[:, :, nir_idx]
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    return ndvi > threshold

def save_plot_with_colorbar(data_map, mask, title, save_path):
    """ 
    核心绘图函数：
    1. 将背景(mask为False)设为白色
    2. 绘制伪彩色图
    3. 添加 Colorbar
    4. 保存
    """
    # 过滤异常极大值，防止破坏渲染效果
    data_map = np.clip(data_map, 0, np.max(data_map))
    
    # 准备绘图数据，背景设为 NaN 以便显示为白色
    vis_data = data_map.copy()
    vis_data[~mask] = np.nan 

    plt.figure(figsize=(10, 8))
    
    # 设置 colormap，并将 NaN 颜色设为白色
    cmap = plt.cm.jet
    cmap.set_bad(color='white') 
    
    plt.imshow(vis_data, cmap=cmap)
    plt.colorbar(label=title) # 添加图例
    plt.title(f'{title} Distribution')
    plt.axis('off') # 关闭坐标轴
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() # 关闭图像以释放内存

def batch_predict_and_plot(input_dir, output_dir, model_path, config_path, stats_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 准备环境
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"加载模型: {model_path}")
    cfg = load_config(config_path)
    stats = np.load(stats_path, allow_pickle=True).item()
    
    model = Model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 获取所有文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
    print(f"共发现 {len(files)} 个文件，开始处理...")

    # 3. 批量循环
    for idx, filename in enumerate(files):
        print(f"[{idx+1}/{len(files)}] 处理中: {filename}")
        file_path = os.path.join(input_dir, filename)
        file_base_name = os.path.splitext(filename)[0]

        try:
            # --- 加载数据 ---
            mat_data = sio.loadmat(file_path)
            # 兼容可能的键名
            if 'plot_HSI' in mat_data:
                hsi_cube = mat_data['plot_HSI'].astype(np.float32)
            elif 'hsi' in mat_data:
                hsi_cube = mat_data['hsi'].astype(np.float32)
            else:
                print(f"  警告: {filename} 中未找到 plot_HSI 或 hsi 变量，跳过。")
                continue
            
            H, W, C = hsi_cube.shape
            
            # --- 计算掩膜 ---
            soil_mask = calculate_mask(hsi_cube)

            # --- 推理 ---
            pixels = hsi_cube.reshape(-1, C)
            batch_size = 2048 # 加大batch_size加快速度
            preds_flat = np.zeros((H * W, 3)) # 存放三个变量的预测结果

            with torch.no_grad():
                for i in range(0, pixels.shape[0], batch_size):
                    batch_pixels = pixels[i:i+batch_size, :]
                    batch_tensor = torch.from_numpy(batch_pixels).to(device).float()
                    
                    outputs_norm = model(batch_tensor)
                    outputs_denorm = denormalize_preds(outputs_norm, stats)
                    
                    preds_flat[i:i+batch_size, :] = outputs_denorm.cpu().numpy()

            # --- 提取三个变量并绘图 ---
            # 根据你的要求：0->LAI, 1->Cab, 2->Cp
            var_mapping = [
                ('LAI', 0),
                ('Cab', 1),
                ('Cp',  2)
            ]

            for var_name, channel_idx in var_mapping:
                # 提取单通道并reshape回 (H, W)
                data_map = preds_flat[:, channel_idx].reshape(H, W)
                
                # 设置保存文件名: 原文件名_变量名.png
                save_name = f"{file_base_name}_{var_name}.png"
                save_full_path = os.path.join(output_dir, save_name)
                
                # 绘图保存
                save_plot_with_colorbar(data_map, soil_mask, var_name, save_full_path)

        except Exception as e:
            print(f"  处理 {filename} 时出错: {e}")

    print("所有文件处理完成！")

if __name__ == '__main__':
    # 请根据实际情况修改以下路径
    batch_predict_and_plot(
        input_dir='/root/autodl-tmp/C3/',              # 输入 .mat 文件夹
        output_dir='/root/autodl-tmp/C3_Results/',     # 输出图片保存文件夹
        model_path='checkpoints_190/best_model.pth',   # 模型路径
        config_path='config_190.json',                 # 配置文件
        stats_path='dataset_190_stats.npy'             # 统计文件
    )