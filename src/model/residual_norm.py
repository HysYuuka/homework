import torch


class ResidualNormLayer(torch.nn.Module):
    # 实现文档3.4的残差连接+LayerNorm
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(d_model)  # 层归一化
        self.dropout = torch.nn.Dropout(dropout)  # 防止过拟合

    def forward(self, x, sublayer):
        # Pre-Norm流程：LayerNorm 子模块 Dropout 残差相加
        output = x + self.dropout(sublayer(self.layer_norm(x)))
        return output
