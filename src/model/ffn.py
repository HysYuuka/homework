import torch


class PositionWiseFFN(torch.nn.Module):
    # 实现文档3.3的Position-Wise Feed-Forward Network
    def __init__(self, d_model=128, d_ff=512):  # 超参参考文档表3
        super().__init__()
        # 两层全连接（y = max(0, xW1 + b1)W2 + b2）
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 前向流程：逐token独立处理
        output = self.fc2(self.relu(self.fc1(x)))
        return output
