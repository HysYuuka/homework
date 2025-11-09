import torch
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    # 实现文档3.1的Scaled Dot-Product Attention

    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output, attn_weights


class MultiHeadAttention(torch.nn.Module):
    # 实现文档3.2的Multi-Head Attention

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 定义Q/K/V投影层
        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        self.W_O = torch.nn.Linear(d_model, d_model)  # 多头结果拼接后投影

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        # 投影并分割为多头（维度：(batch, n_heads, seq_len, d_k)）
        Q_proj = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K_proj = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 并行计算每个头的注意力（调用Scaled Dot-Product）
        attn_output, attn_weights = scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask)

        # 拼接多头结果并投影（文档3.2公式：Linear(concat(head1))）
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_O(attn_output)
        return output, attn_weights
