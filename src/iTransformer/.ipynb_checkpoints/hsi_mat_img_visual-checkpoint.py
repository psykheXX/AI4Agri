import scipy.io as sio
import cv2
import numpy as np

# 1. 加载 .mat 文件
file_path = '/root/autodl-tmp/C3/C3_10706.mat'  # 请替换为你的文件名
data = sio.loadmat(file_path)

# 2. 获取数据矩阵
# 注意：你需要确认 .mat 文件中的变量名，通常可以使用 data.keys() 查看
# 假设变量名为 'hyperspectral_data'
print(f"文件中的键值有: {data.keys()}")
hsi_matrix = data['plot_HSI']

print(f"数据类型: {type(hsi_matrix)}")
print(f"数据维度 (Shape): {hsi_matrix.shape}")
# hsi_matrix = data['hyperspectral_data'] 

# 3. 检查数据维度
# 假设维度是 (H, W, C)，如果是 (C, H, W) 则需要转置

# 4. 提取其中一个通道 (例如第 100 个通道，对应约 610nm)
channel_idx = 100
single_channel = hsi_matrix[:, :, channel_idx].astype(np.float32)

# 5. 归一化到 0-255 范围 (OpenCV 保存 8 位图的要求)
# 高光谱数据通常是反射率或辐射亮度值，范围不在 0-255
min_val = np.min(single_channel)
max_val = np.max(single_channel)
normalized_img = ((single_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 6. 使用 cv2 保存并显示图像
output_name = f'channel_{channel_idx}_gray.png'
cv2.imwrite(output_name, normalized_img)

print(f"已成功将通道 {channel_idx} 保存为 {output_name}")