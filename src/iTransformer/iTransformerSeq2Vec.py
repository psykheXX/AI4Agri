import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Model(nn.Module):
    """
    修改版 iTransformer (针对单变量光谱回归任务)
    架构：Patching -> Linear Embedding -> Transformer Encoder -> Pooling -> Regression Head
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. 配置参数
        self.output_attention = configs.output_attention
        self.seq_len = configs.seq_len      # 光谱长度 (L)
        self.patch_len = configs.patch_len  # 每个 Patch 的长度 (建议设为 16, 32 等)
        self.stride = configs.stride        # Patch 的步长 (通常等于 patch_len，即无重叠)
        self.d_model = configs.d_model
        
        # 计算 Patch 的数量
        # 公式: (L - P) / S + 1
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        
        # 2. Patch Embedding
        # 将每个 Patch (长度为 patch_len) 映射为 d_model 维度的向量
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # 3. Encoder (保持原 Transformer 结构)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 4. 回归头 (Regression Head)
        # 输入是 d_model (经过池化后)，输出是 3 (三个回归值)
        self.head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, 3)  # 输出 3 个参数
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [Batch, Seq_Len, 1]  (单变量光谱数据)
        """
        
        # --- Step 1: Input Normalization (可选，但推荐) ---
        # 对每条光谱做标准化，消除基线漂移等影响
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # --- Step 2: Patching ---
        # Input: [B, L, 1] -> squeeze -> [B, L]
        x = x_enc.squeeze(-1) 
        
        # Unfold 操作实现切片
        # [B, L] -> [B, Patch_Num, Patch_Len]
        # unfold(dimension, size, step)
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # --- Step 3: Embedding ---
        # [B, Patch_Num, Patch_Len] -> [B, Patch_Num, d_model]
        enc_out = self.patch_embedding(x_patched)
        enc_out = self.dropout(enc_out)

        # --- Step 4: Encoder ---
        # [B, Patch_Num, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # --- Step 5: Pooling & Output ---
        # 此时 enc_out 是所有 patches 的特征。
        # 我们使用 Global Average Pooling 聚合整条光谱的信息
        # [B, Patch_Num, d_model] -> [B, d_model]
        out_pooled = enc_out.mean(dim=1) 
        
        # 回归预测
        # [B, d_model] -> [B, 3]
        outputs = self.head(out_pooled)
        
        # 注意：这里不再进行 De-Normalization (反归一化)
        # 因为输出的是物理参数，而不是光谱本身。
        # 请确保你的 Label 在外部也做了标准化处理。

        if self.output_attention:
            return outputs, attns
        else:
            return outputs