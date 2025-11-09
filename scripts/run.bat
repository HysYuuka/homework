@echo off
setlocal enabledelayedexpansion

:: 1. 配置虚拟环境
echo 配置虚拟环境...
conda create -n transformer_ptb python=3.10 -y
call conda activate transformer_ptb

:: 2. 安装依赖
echo 安装依赖库...
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install datasets==2.20.0 transformers==4.41.0 matplotlib==3.10.3 numpy==2.2.6 tqdm==4.67.1

:: 3. 训练实验
echo 开始模型训练...
python src\train.py

echo 实验完成！结果已保存至results/目录
endlocal