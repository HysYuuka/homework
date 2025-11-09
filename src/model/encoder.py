import torch
import math
from .attention import MultiHeadAttention
from .ffn import PositionWiseFFN
from .residual_norm import ResidualNormLayer
from .pos_encoding import PositionalEncoding


class EncoderBlock(torch.nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.res_norm1 = ResidualNormLayer(d_model, dropout)
        self.res_norm2 = ResidualNormLayer(d_model, dropout)

    def forward(self, x, mask):
        x = self.res_norm1(x, lambda x: self.self_attn(x, x, x, mask)[0])
        x = self.res_norm2(x, self.ffn)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 d_ff=512, max_seq_len=256, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)  # 位置编码模块
        self.encoder_blocks = torch.nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.d_model = d_model
        self.output_proj = torch.nn.Linear(d_model, vocab_size)
        self.use_pos_encoding = use_pos_encoding  # 控制是否使用位置编码的标志

    def forward(self, x, mask):
        batch_size, seq_len = x.size()
        # 嵌入层缩放
        x = self.embedding(x) * math.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=x.device))

        # 根据标志决定是否添加位置编码
        if self.use_pos_encoding:
            x = self.pos_encoding(x)  # 有位置编码（默认）
        # 无位置编码时，直接跳过位置编码叠加

        x = self.dropout(x)
        # 经过所有Encoder Block
        for block in self.encoder_blocks:
            x = block(x, mask)
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits

