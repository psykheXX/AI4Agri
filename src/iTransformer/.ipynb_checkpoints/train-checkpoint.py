import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path
import swanlab

from LoadData import HyperspectralDataset, get_all_file_paths, denormalize_preds
from iTransformerSeq2Vec import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 工具函数：加载配置 =================
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    flat_config = {}
    for section in config_dict:
        for key, value in config_dict[section].items():
            flat_config[key] = value
            
    config_obj = SimpleNamespace(**flat_config)
    return config_obj

# ================= 验证函数=================
def validate(model, val_loader, stats, device):
    model.eval()
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for spectra, labels_phys in val_loader:
            spectra = spectra.to(device).float()
            outputs_norm = model(spectra)
            outputs_phys = denormalize_preds(outputs_norm, stats, device=device)
            
            preds_list.append(outputs_phys.cpu().numpy())
            targets_list.append(labels_phys.numpy())
            
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    metrics = {}
    param_names = ['LAI', 'Cab', 'Cp']
    total_r2 = 0
    total_rmse = 0
    
    print("\n--- Validation Report ---")
    for i, name in enumerate(param_names):
        try:
            r2 = r2_score(targets[:, i], preds[:, i])
            rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        except:
            r2 = 0.0; rmse = 999.0
            
        metrics[f'{name}_R2'] = r2
        metrics[f'{name}_RMSE'] = rmse
        total_r2 += r2
        total_rmse += rmse
        print(f"  {name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    
    avg_r2 = total_r2 / 3
    avg_rmse = total_rmse / 3
    print(f"  [AVG]: R2 = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}")
    print("-------------------------")
    
    model.train()
    return avg_r2, metrics

# ================= 主训练循环 =================
def train(args):
    # 1. 加载配置
    print(f"正在加载配置文件: {args.config}")
    cfg = load_config(args.config)
    
    if not hasattr(cfg, 'epochs'):
        print("Warning: 'epochs' not found in config, defaulting to 100.")
        cfg.epochs = 100

    # ================= 2. SwanLab 初始化 =================
    print("初始化 SwanLab...")
    swanlab.init(
        project=cfg.project_name, 
        workspace=cfg.workspace,
        experiment_name=f"iTrans_BS{cfg.batch_size}_LR{cfg.learning_rate}", # 实验名称
        config=vars(cfg), # 记录所有配置参数
        description="Hyperspectral regression using iTransformer"
    )

    # 3. 数据准备
    print("正在读取数据文件路径...")
    train_files = get_all_file_paths(cfg.data_root)
    if not train_files: raise ValueError(f"训练集目录 {cfg.data_root} 下未找到数据文件！")
    
    val_files = get_all_file_paths(cfg.val_root)
    if not val_files: raise ValueError(f"验证集目录 {cfg.val_root} 下未找到数据文件！")
    
    stats = np.load(cfg.stats_path, allow_pickle=True).item()
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 4. Dataset & DataLoader
    train_ds = HyperspectralDataset(train_files, stats, mode='train', noise_level=0.01)
    val_ds = HyperspectralDataset(val_files, stats, mode='test')
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    # 5. Model, Loss, Optimizer
    model = Model(cfg).to(device)
    huber_criterion = nn.HuberLoss(delta=1.0, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_iters, eta_min=cfg.eta_min)
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. 训练循环
    print(f"开始训练... 总 Epochs: {cfg.epochs}, 总 Iterations 上限: {cfg.total_iters}")
    
    global_step = 0
    best_val_r2 = -float('inf')
    should_stop = False

    progress_bar = tqdm(total=cfg.total_iters, desc="Training", unit="step")
    
    for epoch in range(1, cfg.epochs + 1):
        if should_stop: break

        for i, (spectra, labels_norm) in enumerate(train_loader):
            global_step += 1

            # --- 新增检查 ---
            if torch.isnan(spectra).any() or torch.isinf(spectra).any():
                print(f"检测到输入数据包含 NaN/Inf，跳过 Batch {i}")
                continue
            if torch.isnan(labels_norm).any() or torch.isinf(labels_norm).any():
                print(f"检测到标签数据包含 NaN/Inf，跳过 Batch {i}")
                continue
            # ----------------

            spectra, labels_norm = spectra.to(device).float(), labels_norm.to(device).float()
            
            optimizer.zero_grad()
            
            """混合精度"""
            # with torch.cuda.amp.autocast():
            #     preds_norm = model(spectra)
            #     loss_LAI = huber_criterion(preds_norm[:, 0], labels_norm[:, 0])
            #     loss_Cab = huber_criterion(preds_norm[:, 1], labels_norm[:, 1])
            #     loss_Cp  = huber_criterion(preds_norm[:, 2], labels_norm[:, 2])
            #     total_loss = loss_LAI + loss_Cab + loss_Cp
            
            # scaler.scale(total_loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # scaler.step(optimizer)
            # scaler.update()

            """fp32"""
            preds_norm = model(spectra)
            loss_LAI = huber_criterion(preds_norm[:, 0], labels_norm[:, 0])
            loss_Cab = huber_criterion(preds_norm[:, 1], labels_norm[:, 1])
            loss_Cp  = huber_criterion(preds_norm[:, 2], labels_norm[:, 2])
            total_loss = 2*loss_LAI + loss_Cab + 5*loss_Cp

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            # ================= 7. 记录训练 Metrics 到 SwanLab =================
            swanlab.log({
                "train/total_loss": total_loss.item(),
                "train/loss_LAI": loss_LAI.item(),
                "train/loss_Cab": loss_Cab.item(),
                "train/loss_Cp": loss_Cp.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
            }, step=global_step)

            progress_bar.update(1)
            progress_bar.set_postfix({'Epoch': f"{epoch}/{cfg.epochs}", 'Loss': f"{total_loss.item():.4f}"})
            
            # 验证与保存
            if global_step % cfg.val_interval == 0:
                avg_r2, metrics = validate(model, val_loader, stats, device)
                
                # ================= 8. 记录验证 Metrics 到 SwanLab =================
                # 构建用于 SwanLab 的字典，为了区分，建议加 'val/' 前缀
                log_val_dict = {f"val/{k}": v for k, v in metrics.items()}
                log_val_dict["val/avg_r2"] = avg_r2
                swanlab.log(log_val_dict, step=global_step)
                
                checkpoint = {
                    'epoch': epoch,
                    'iter': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': vars(cfg)
                }
                
                torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, f'{global_step}_checkpoint.pth'))
                
                if avg_r2 > best_val_r2:
                    best_val_r2 = avg_r2
                    torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
                    print(f"🔥 新的最佳模型! R2: {best_val_r2:.4f}")
            
            if global_step >= cfg.total_iters:
                print("\n达到最大迭代次数，停止训练。")
                should_stop = True
                break
    
    progress_bar.close()

    final_path = os.path.join(cfg.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Saved to {final_path}")
    
    # 结束 SwanLab 实验
    swanlab.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to json config file')
    args = parser.parse_args()
    
    train(args)