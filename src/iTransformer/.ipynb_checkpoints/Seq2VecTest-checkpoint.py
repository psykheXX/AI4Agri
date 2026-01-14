import iTransformerSeq2Vec
import torch

class Configs:
    def __init__(self):
        self.seq_len = 240     # 假设你的光谱长度是 1000
        self.patch_len = 64     # 切片长度：每 50 个波长作为一个 Token
        self.stride = 64        # 步长：无重叠切割
        
        # 其他原有的参数
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 5
        self.d_ff = 2048
        self.dropout = 0.1
        self.factor = 1
        self.activation = 'gelu'
        self.output_attention = False
        
configs = Configs()
model = iTransformerSeq2Vec.Model(configs)

# 测试输入 [Batch=32, Length=1000, Channel=1]
dummy_input = torch.randn(32, 2101, 1)
output = model(dummy_input)

print("Output shape:", output.shape) # 应该是 [32, 3]