# Transformer 期中作业（基于PTB数据集）
实现Encoder-only Transformer用于PTB词级语言建模。

## 1. 数据集说明
- 选用数据集：Penn Treebank (PTB)
- 任务类型：词级语言建模（Word-level LM）
- 数据集规模：~1M tokens
- 加载方式：通过Hugging Face `ptb-text-only` 数据集加载
- 预处理：词级分词、序列长度256（截断/填充）、自回归样本构建

## 2.仓库结构
```
├── src/                  # 源代码文件夹
│   ├── model/            # 模型模块定义（核心模块实现）
│   │   ├── attention.py  # Scaled Dot-Product、Multi-Head Attention
│   │   ├── ffn.py        # Position-Wise FFN
│   │   ├── residual_norm.py # 残差连接与LayerNorm
│   │   ├── pos_encoding.py # 正弦位置编码
│   │   └── encoder.py    # Encoder Block与完整Encoder
│   ├── results/              # 实验结果
│   │   └── ptb_train_val_curves.png # PTB训练损失与困惑度曲线
│   ├── checkpoints/          # 模型权重
│   │   └── best_model_ptb.pth # PTB最优模型权重
│   ├── data/             # 数据处理
│   │   └── data_loader.py # PTB数据集加载、分词、批处理
│   ├── train.py          # 训练+评估脚本
│   ├── predict_validation.py          # 预测
│   └── utils.py          # 工具函数
├── requirements.txt      # 依赖库列表
├── scripts/              # 运行脚本
│   ├── run.bat           # PTB一键训练/评估脚本 windows
│   └── run.sh            # PTB一键训练/评估脚本
└── README.md             # 说明文档
```

## 3.环境要求（硬件+软件）
- 硬件：GPU（建议显存≥4GB，支持CUDA 12.1）
- 软件：Python 3.10、CUDA 12.1

## 4. 环境配置
```bash
# 1. 创建conda虚拟环境（文档建议Python 3.10）
conda create -n transformer python=3.10 -y
conda activate transformer

# 2. 安装依赖库（对应requirements.txt）
pip install -r requirements.txt
```

## 5.运行
直接运行，随机种子硬编码为42。
```angular
# PTB训练
python src/train.py
# PTB评估 如果有模型
python src/train.py --eval_only --model_path src/checkpoints/best_model_ptb.pth
```

## 6.超参设置

| 超参数	       | 取值 |
|:------------:|:-:|
| 嵌入维度       | 128 |
| 多头注意力头数    | 4 |
| FFN 中间层维度  | 512 |
| Encoder 层数 | 4 |
| 最大序列长度     | 256 |
| Dropout 概率 | 0.1 |
| 批大小        | 32 |
| 学习率        | 3e-4 |
| 优化器        | Adam（权重衰减 1e-4）|
| 训练轮次       | 10 |

