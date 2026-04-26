# TAAC

## 项目结构

```
TAAC/
├── baseline/                 # 基线模型代码（只读参考）
├── experiments/              # 实验目录
├── scripts/                  # 实用脚本
├── demo_sample_1000/         # 示例数据
├── requirements.txt          # 依赖清单
└── setup_env.sh             # 环境设置脚本
```

## 快速开始

### 1. 环境配置

```bash
bash setup_env.sh
```

或手动安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行基线模型

```bash
cd baseline
bash run.sh
```

或直接运行：

```bash
python train.py
```

### 3. 运行推理

```bash
python infer.py
```

## 实验管理

### 目录结构

所有实验存放在 `experiments/` 目录下，命名格式为 `YYYY-MM-DD_实验描述/`：

```
experiments/
├── 2026-04-26_attention_flash/     # Flash Attention 实验
│   ├── code/                       # 实验代码（从 baseline 复制）
│   ├── outputs/                    # 运行输出
│   │   ├── checkpoints/            # 模型检查点
│   │   └── logs/                   # 训练日志
│   ├── config.yaml                 # 超参数配置
│   └── README.md                   # 实验说明
│
└── 2026-04-27_encoder_longformer/  # Longformer 编码器实验
    ├── code/
    ├── outputs/
    ├── config.yaml
    └── README.md
```

## 基线模型

### 模型架构

- **序列编码器**：支持 Transformer、Longer、SwiGLU 等变体
- **注意力机制**：标准注意力 + 位置编码
- **查询生成**：自适应查询生成机制

### 训练配置

- **优化器**：AdamW
- **学习率调度**：支持多种调度策略
- **早停机制**：基于验证集性能
- **检查点管理**：自动保存最佳模型

## 实验记录

详见 [EXPERIMENTS.md](EXPERIMENTS.md)

## 依赖

主要依赖：

- PyTorch
- Transformers (Hugging Face)
- PyArrow (Parquet 数据读取)
- 其他依赖见 `requirements.txt`

## 许可

[待添加]
