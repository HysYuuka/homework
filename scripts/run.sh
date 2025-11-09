#!/bin/bash
# scripts/run.sh
set -e  # 出错即停止

# 1. 创建虚拟环境
conda create -n transformer python=3.10 -y
conda activate transformer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练
python src/train.py
