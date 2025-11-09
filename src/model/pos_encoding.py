import torch
import math


def get_positional_encoding(d_model=128, max_seq_len=256):
    # 实现文档3.5的正弦位置编码
    # 初始化位置编码矩阵（维度：max_seq_len × d_model）
    pe = torch.zeros(max_seq_len, d_model)
    pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
    # 计算频率因子
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # 填充正弦（偶数维）和余弦（奇数维）
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0)
    return pe


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=128, max_seq_len=256):
        super().__init__()
        self.pe = get_positional_encoding(d_model, max_seq_len)  # 非训练参数
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        # 位置编码与token嵌入同等缩放
        scaled_pe = self.pe[:, :seq_len, :].to(x.device) * math.sqrt(self.d_model)
        x = x + scaled_pe
        return x
