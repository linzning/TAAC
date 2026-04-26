#!/bin/bash
# Conda环境配置脚本 - TAAC项目
# 使用方法: bash setup_env.sh

# 创建conda环境
conda create -n taac python=3.10.20 -y

# 激活环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate taac

# 安装PyTorch (使用conda-forge和pytorch频道)
echo "安装PyTorch..."
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 cpuonly -c pytorch -y

# 安装其他依赖
echo "安装其他依赖包..."
conda install pyarrow=18.1.0 numpy=1.26.4 scikit-learn=1.6.1 -c conda-forge -y

# 使用pip安装剩余包
echo "安装tqdm和tensorboard..."
pip install tqdm==4.67.1 tensorboard==2.18.0

echo "========================================="
echo "环境配置完成！"
echo "使用以下命令激活环境:"
echo "  conda activate taac"
echo "========================================="