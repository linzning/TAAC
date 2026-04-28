# HyFormer Baseline 改进方向分析

## 一、论文核心 vs 代码实现对比

HyFormer 的核心设计是**"交替优化"**：每层内先通过 **Query Decoding**（Cross-Attention 让 Query 读取序列 K/V）再 **Query Boosting**（Token Mixing + FFN 让 Query 与 NS 特征交互），形成序列与非序列的双向信息流。

代码基本复现了这个框架，支持 3 种序列编码器（SwiGLU / Transformer / Longer）、2 种 NS Tokenizer（Group / RankMixer），并扩展到**多序列**场景。

---

## 二、改进方向

### 1. 序列间交互机制（Inter-Sequence Interaction）已实验

**问题**：当前每个序列独立编码、独立 Cross-Attention，序列之间**没有显式信息交换**。多个序列（点击 / 加购 / 支付等）的 Query tokens 只是简单 concat 后进 RankMixer。

**改进**：

- 在 `MultiSeqHyFormerBlock` 中增加**序列间 Cross-Attention 层**：让序列 A 的 Query 能关注序列 B 的 Key/Value
- 或在 RankMixer 之前增加一个轻量的 `InterSeqAttention` 模块，使不同序列的 decoded Q tokens 相互交互

```python
# 示例：在 Query Decoding 后、Token Fusion 前增加
for i in range(S):
    for j in range(S):
        if i != j:
            q_i = cross_attn_seq_i_to_j(q_i, seq_j)
```

---

### 2. 最终输出显式融合 NS Tokens【高优先级】

**问题**：当前输出只用了 Q tokens：

```python
all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
output = self.output_proj(all_q.view(B, -1))  # (B, D)
```

NS tokens 经过每层的 RankMixer 后也蕴含丰富的交互信息，但**完全被丢弃**。

**改进**：
- 类似 BERT 的 `[CLS]` 设计，将 NS tokens 与 Q tokens 一起 flatten 后投影
- 或增加一个独立的 NS aggregation head，与 Q-based 输出融合

```python
# 改进后的输出
all_q = torch.cat(curr_qs, dim=1)          # (B, Nq*S, D)
ns_out = curr_ns.mean(dim=1)               # (B, D) 或取特定 token
output = self.output_proj(
    torch.cat([all_q.view(B, -1), ns_out], dim=-1)
)
```

---

### 3. 候选物品（Candidate Item）的显式 Query 注入【中优先级】

**问题**：论文提到 Global Tokens "derived from original candidate item"。当前代码中 item 特征通过 `item_ns_tokenizer` 进入 NS tokens，但**没有生成候选感知的专用 Query**。

**改进**：
- 在 `MultiSeqQueryGenerator` 中，将候选 item 特征单独抽取出来，生成 `candidate-aware query tokens`
- 这些 query 专门用于解码与候选物品最相关的序列信息

---

### 4. 时间感知的增强【中优先级】

**问题**：当前时间建模仅为简单的 `time_embedding` 加法：

```python
token_emb = token_emb + self.time_embedding(time_bucket_ids)
```

没有体现**时间衰减**（越近的行为越重要）或**相对时间差**的精细建模。

**改进**：
- 在序列编码器中引入时间衰减权重：$w_t = \exp(-\lambda \cdot \Delta t)$
- 或使用相对时间编码替代绝对分桶

---

### 5. RankMixer 约束松绑与模式探索【中优先级】

**问题**：`full` 模式下要求 `d_model % T == 0`，其中 `T = num_queries * num_sequences + num_ns`。当序列数多或 NS token 数多时，**严重限制 d_model 的可选值**。

**改进**：
- 当前已有 `ffn_only` 和 `none` 模式作为 fallback
- 可探索**自适应分组 Token Mixing**：不强制所有 token 参与 mixing，而是按语义（User / Item / Query）分组 mixing 后再融合

---

### 6. 序列编码器的深度与参数共享【中优先级】

**问题**：每个 block 内的 `seq_encoder` 是单层的。如果序列很长（如 `seq_d:512`）或语义复杂，单层自注意力的建模能力可能不足。

**改进**：
- 在 `TransformerEncoder` 内部堆叠 2 层（增加 `num_encoder_layers` 参数）
- 或探索**跨层参数共享**的序列编码器，在增加深度的同时控制参数量

---

### 7. LongerEncoder 的 Top-K 选择策略优化【中优先级】

**问题**：当前 `_gather_top_k` 只取**最新的 top_k 个 token**：

```python
start_pos = valid_len - actual_k  # 总是取尾部
```

对于某些序列（如支付序列），最早的几个行为可能同样重要。

**改进**：
- 引入**重要性采样**：基于时间衰减 + 频率的重要性分数选择 top_k
- 或混合策略：最近 k/2 + 最重要 k/2

---

### 8. 训练策略增强【中优先级】

**当前已支持**：Focal Loss、双优化器（Adagrad + AdamW）、Sparse Reinit、EarlyStopping。

**可补充**：
- **辅助任务**：增加序列长度预测、时间间隔预测等辅助 loss，增强序列表示学习
- **Dropout 策略差异化**：当前 `seq_id_emb_dropout` 固定为 `dropout_rate * 2`，可改为基于特征频率的自适应 dropout
- **学习率预热（Warmup）**：当前优化器没有 warmup，对大规模 embedding 训练不稳定

---

### 9. 多序列编码器异构化【低优先级】

**问题**：当前所有序列强制使用同一种 `seq_encoder_type`。

**改进**：
- 不同序列采用不同编码器：例如点击序列长用 `longer`，曝光序列短用 `transformer`，收藏序列用 `swiglu`
- 在 `train.py` 中支持按 domain 指定 encoder 类型

---

### 10. 推理优化：KV-Cache 与模型编译【低优先级】

**问题**：当前 `infer.py` 逐 batch 推理，没有利用序列的增量特性。

**改进**：
- 对于自回归场景，为 `TransformerEncoder` 和 `LongerEncoder` 添加 **KV-Cache**
- 在 `infer.py` 中使用 `torch.compile()` 加速（代码中 RoPE cache 已兼容 `torch.compile`）

### 11.改用 sigmoid_focal_loss

---

## 三、建议的实验优先级

| 优先级 | 改进方向 | 预期收益 | 实现复杂度 |
|--------|---------|---------|-----------|
| P0 | 序列间交互机制 | 显著提升多序列场景效果 | 中 |
| P0 | 输出融合 NS tokens | 提升信息利用效率 | 低 |
| P1 | 候选物品显式 Query | 增强候选感知能力 | 中 |
| P1 | 时间衰减权重 | 提升长序列建模质量 | 低 |
| P2 | RankMixer 约束松绑 | 更灵活的模型配置 | 中 |
| P2 | 学习率预热 + 辅助任务 | 训练稳定性与表示质量 | 低 |
