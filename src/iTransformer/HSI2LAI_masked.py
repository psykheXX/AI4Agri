import os
import torch
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from iTransformerSeq2Vec import Model 
from types import SimpleNamespace
import json
from LoadData import denormalize_preds

STATS_PATH = 'dataset_190_stats.npy'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    flat_config = {}
    for section in config_dict:
        for key, value in config_dict[section].items():
            flat_config[key] = value
    return SimpleNamespace(**flat_config)

def calculate_mask(hsi_cube, red_idx=50, nir_idx=90, threshold=0.3):
    """
    利用 NDVI 计算植被掩膜以去除土壤背景。
    
    参数:
    - red_idx: 红光波段的通道索引 (通常在 670nm 附近)
    - nir_idx: 近红外波段的通道索引 (通常在 800nm 附近)
    - threshold: NDVI 阈值 (通常 0.3 - 0.5 之间用于区分土壤和植物)
    
    注意：请根据你的 HSI 数据具体的波段参数修改 red_idx 和 nir_idx。
    """
    # 提取单波段图像
    red_band = hsi_cube[:, :, red_idx]
    nir_band = hsi_cube[:, :, nir_idx]
    
    # 计算 NDVI: (NIR - Red) / (NIR + Red)
    # 加 1e-8 防止除以零
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    
    # 生成二值掩膜 (True 为植物, False 为土壤/背景)
    mask = ndvi > threshold
    return mask

def predict_lai_map(mat_path, model_path, config_path, stats_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载配置和统计信息
    cfg = load_config(config_path)
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # 2. 加载模型
    model = Model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 加载并预处理 HSI 数据
    mat_data = sio.loadmat(mat_path)
    hsi_cube = mat_data['plot_HSI'].astype(np.float32) # (H, W, C)
    
    H, W, C = hsi_cube.shape
    
    # === [新步骤] 计算土壤掩膜 ===
    """光谱范围：$900 - 400 = 500 \text{ nm}$波段数量：190光谱分辨率（每波段间隔）：$500 / 190 \approx 2.63 \text{ nm/band}$我们需要的波段中心波长通常为：Red (红光): 吸收谷，通常选在 670 nm 左右。NIR (近红外): 反射平台，通常选在 800 nm 左右 (只要大于 760nm 且小于 900nm 即可)。"""
    soil_mask = calculate_mask(hsi_cube, red_idx=103, nir_idx=152, threshold=0.05)
    # ==========================

    # 将图像展平为 (H*W, C) 以便批量输入
    pixels = hsi_cube.reshape(-1, C)
    
    # 4. 执行推理 (逻辑不变，对所有点进行预测)
    lai_map_flat = np.zeros(H * W)
    batch_size = 512 
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    
    print("正在进行像素级回归...")
    with torch.no_grad():
        for i in range(0, pixels.shape[0], batch_size):
            batch_pixels = pixels[i:i+batch_size, :]
            batch_tensor = torch.from_numpy(batch_pixels).to(device).float()
            
            # 模型输出是归一化后的 [LAI, Cab, Cp]
            outputs_norm = model(batch_tensor)
            
            # 提取 LAI (第一个通道) 并反归一化
            outputs_denorm = denormalize_preds(outputs_norm, stats)
            lai_phys =outputs_denorm[:, 2].cpu().numpy()
            
            lai_map_flat[i:i+batch_size] = lai_phys
            
    # 5. 重塑为二维图像
    lai_map = lai_map_flat.reshape(H, W)
    
    # 6. 伪彩色映射与保存 (带背景去除)
    
    # 过滤掉可能的异常值
    lai_map = np.clip(lai_map, 0, np.max(lai_map))
    
    # --- OpenCV 渲染 (白色背景) ---
    # 归一化到 0-255
    lai_map_rescaled = cv2.normalize(lai_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用 JET 颜色表
    color_map = cv2.applyColorMap(lai_map_rescaled, cv2.COLORMAP_JET)
    
    # [关键步骤] 利用掩膜将背景置为白色
    # mask 为 False (土壤) 的地方，颜色设为 [255, 255, 255]
    color_map[~soil_mask] = [255, 255, 255]
    
    cv2.imwrite('lai_prediction_map_white_bg.png', color_map)
    print("OpenCV 图像已保存 (背景为白色)")
    
    # --- Matplotlib 渲染 (白色背景) ---
    plt.figure(figsize=(10, 8))
    
    # 创建一个用于显示的副本，将背景设为 NaN
    lai_map_vis = lai_map.copy()
    lai_map_vis[~soil_mask] = np.nan 
    
    # 使用 jet colormap，set_bad 设置 NaN 的颜色为白色
    cmap = plt.cm.jet
    cmap.set_bad(color='white') # 或者 'none' 表示透明
    
    plt.imshow(lai_map_vis, cmap=cmap)
    plt.colorbar(label='LAI Value')
    plt.title('Wheat LAI Distribution Map (Background Removed)')
    plt.axis('off') # 可选：关闭坐标轴显示
    plt.savefig('lai_with_colorbar_masked.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    predict_lai_map(
        mat_path='/root/autodl-tmp/C3/C3_10706.mat', 
        model_path='checkpoints_190/best_model.pth',
        config_path='config_190.json',
        stats_path='dataset_190_stats.npy'
    )