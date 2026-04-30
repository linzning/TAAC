#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 激活conda环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taac

# 本地数据路径配置（相对于项目根目录）
export TRAIN_DATA_PATH="${SCRIPT_DIR}/../../demo_sample_1000"
export TRAIN_CKPT_PATH="${SCRIPT_DIR}/checkpoints"
export TRAIN_LOG_PATH="${SCRIPT_DIR}/logs"
export TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/tf_events"

# 创建必要的输出目录
mkdir -p "${TRAIN_CKPT_PATH}"
mkdir -p "${TRAIN_LOG_PATH}"
mkdir -p "${TRAIN_TF_EVENTS_PATH}"

# ---- Active config: RankMixer NS tokenizer ----
# 梯度累积: --gradient_accumulation_steps 2 等效 batch_size = 32*2 = 64
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 11 \
    --item_ns_tokens 4 \
    --num_queries 4 \
    --d_model 128 \
    --emb_dim 128 \
    --hidden_mult 6 \
    --num_hyformer_blocks 3 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 4 \
    --batch_size 32 \
    --num_epochs 3 \
    --valid_ratio 0.0 \
    --gradient_accumulation_steps 2 \
    "$@"
