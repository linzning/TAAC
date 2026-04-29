# 模型参数量翻倍方案

> 目标：在当前配置基础上，合理增加模型参数量以提升模型容量

## 当前配置分析

基于 `baseline/run.sh` 的配置：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `d_model` | 64 | 主干网络隐藏维度 |
| `num_queries` | 2 | 每个序列域生成的查询token数 |
| `num_hyformer_blocks` | 2 | HyFormer块堆叠层数 |
| `num_heads` | 4 | 注意力头数 |
| `emb_dim` | 64 | Embedding维度 |
| `hidden_mult` | 4 | FFN隐藏层倍数 |
| `user_ns_tokens` | 5 | 用户侧NS token数 |
| `item_ns_tokens` | 2 | 物品侧NS token数 |

### 关键约束

**T = num_queries × num_sequences + num_ns 必须能整除 d_model**

当前配置：
- `ns_tokenizer_type` = rankmixer
- `emb_skip_threshold` = 1000000 (跳过超大基数字段)
- num_sequences = 4 (seq_a/b/c/d)
- num_ns = user_ns_tokens + 1(user_dense) + item_ns_tokens = 5 + 1 + 2 = 8
- T = 2 × 4 + 8 = **16**
- 64 % 16 = 0 ✓

## 参数量翻倍方案

### 方案1：增大 d_model（推荐）

```bash
--d_model 128 --num_queries 2 --user_ns_tokens 5 --item_ns_tokens 2
```

- **T = 16**，128 % 16 = 0 ✓
- 参数量变化：~3.5倍（d_model呈平方关系影响主要参数）
- 优点：最大化提升模型容量，T约束保持不变
- 缺点：显存占用增加较多

### 方案2：增加 num_queries + d_model

```bash
--d_model 96 --num_queries 4 --user_ns_tokens 5 --item_ns_tokens 2
```

- T = 4 × 4 + 8 = **24**，96 % 24 = 0 ✓
- 参数量变化：~2.2倍
- 优点：增加序列表达能力
- 缺点：计算复杂度增加

### 方案3：增加层数

```bash
--num_hyformer_blocks 4
```

- 参数量变化：~2倍（线性增长）
- 优点：增加模型深度，不改变T约束
- 缺点：训练时间增加

### 方案4：增加 emb_dim

```bash
--emb_dim 128
```

- 参数量变化：Embedding表参数翻倍（约占总参数10-20%）
- 优点：特征表示能力增强
- 缺点：对总参数量提升有限

### 方案5：调整 hidden_mult

```bash
--hidden_mult 8
```

- 参数量变化：FFN层参数约2倍
- 优点：增加非线性变换能力
- 缺点：对总参数量提升有限

### 方案6：综合组合

```bash
--d_model 96 --num_hyformer_blocks 3 --emb_dim 96 --hidden_mult 6
```

- T = 16，96 % 16 = 0 ✓
- 参数量变化：~2倍
- 优点：均衡提升各方面能力
- 缺点：调参复杂度增加

## 方案对比表

| 方案 | d_model | num_queries | num_blocks | emb_dim | hidden_mult | T | 预估倍数 |
|------|---------|-------------|------------|---------|-------------|---|----------|
| **基线** | 64 | 2 | 2 | 64 | 4 | 16 | 1× |
| 方案1 | 128 | 2 | 2 | 64 | 4 | 16 | ~3.5× |
| 方案2 | 96 | 4 | 2 | 64 | 4 | 24 | ~2.2× |
| 方案3 | 64 | 2 | 4 | 64 | 4 | 16 | ~2× |
| 方案4 | 64 | 2 | 2 | 128 | 4 | 16 | ~1.2× |
| 方案5 | 64 | 2 | 2 | 64 | 8 | 16 | ~1.5× |
| 方案6 | 96 | 2 | 3 | 96 | 6 | 16 | ~2× |

## 参数量影响分析

### 影响最大的参数（按降序）

1. **d_model** - 影响所有线性层和Attention（呈平方关系）
2. **num_hyformer_blocks** - 线性增加主网络参数
3. **emb_dim** - 影响所有Embedding表
4. **num_queries** - 影响Query Generator和输出层
5. **hidden_mult** - 影响FFN中间层

### T约束检查

当调整参数时，必须确保：
```
d_model % (num_queries × num_sequences + num_ns) == 0
```

对于 d_model = 96，有效T值：1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96
对于 d_model = 128，有效T值：1, 2, 4, 8, 16, 32, 64, 128

## 推荐实施顺序

1. **先试方案1**（d_model=128）：最大化容量提升
2. **再试方案6**（综合组合）：均衡提升
3. **最后试方案2**（增加queries）：提升序列表达能力

## 注意事项

- 显存占用：d_model和num_queries影响最大
- 训练速度：num_hyformer_blocks和num_queries影响较大
- 过拟合风险：参数量增加后需调整dropout和正则化
- 学习率调整：参数量翻倍后建议适当降低学习率

---

# 模型架构改进方案

## 方案7: 多尺度特征融合 (Multi-Scale Feature Fusion)

### 设计思路
在HyFormerBlock中引入多尺度特征融合机制，让不同层级的特征能够跨层交互：

```python
class MultiScaleFusionBlock(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        # 不同尺度的卷积核或池化操作
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model),
                nn.LayerNorm(d_model)
            )
            for k in [1, 3, 5][:num_scales]
        ])
        self.fusion = nn.Linear(d_model * num_scales, d_model)

    def forward(self, x):
        # x: (B, L, D)
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x.transpose(1, 2)).transpose(1, 2)
            scale_features.append(feat)
        fused = torch.cat(scale_features, dim=-1)
        return self.fusion(fused)
```

- **优点**: 捕获不同粒度的序列模式，增强特征表达能力
- **实现**: 在HyFormerBlock的FFN前插入多尺度融合
- **参数增加**: 约30-50%

## 方案8: 门控注意力机制 (Gated Attention)

### 设计思路
为交叉注意力添加门控机制，让模型动态调整不同序列域的重要性：

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, query, key_value, ...):
        attn_out = self.cross_attn(query, key_value, ...)
        gate_weight = self.gate(torch.cat([query, attn_out], dim=-1))
        return gate_weight * attn_out + (1 - gate_weight) * query
```

- **优点**: 自适应调整信息流动，防止某个域主导
- **实现**: 替换现有的CrossAttention模块
- **参数增加**: 约10-15%

## 方案9: 层归一化变体 (RMSNorm + Pre-LN)

### 设计思路
使用RMSNorm替代LayerNorm，减少计算开销并稳定训练：

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.weight * x / (norm + self.eps)
```

- **优点**: 减少约10%计算量，训练更稳定
- **实现**: 替换所有nn.LayerNorm
- **资源约束**: 19GiB显存足够

---

# 训练策略改进方案

## 方案10: 混合精度训练 (AMP)

```bash
# 启用混合精度训练
torch.cuda.amp.autocast(enabled=True)
scaler = torch.cuda.amp.GradScaler()
```

- **优点**: 显存占用减半，训练速度提升30-50%
- **实现**: 在trainer.py中添加AMP支持
- **注意事项**: 需调整loss scale防止梯度下溢

## 方案11: 梯度累积

```bash
# 参数配置
--batch_size 64 --accum_steps 2  # 有效batch_size=128
--batch_size 32 --accum_steps 4  # 有效batch_size=128
```

- **优点**: 在显存受限时使用更大有效batch size
- **当前配置**: batch_size=32可尝试accum_steps=2-4
- **效果**: 稳定训练，提升模型泛化性

## 方案12: 学习率调度优化

```python
# 当前: AdamW betas=(0.9, 0.98)
# 改进: 使用cosine annealing with warm restart
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

- **优点**: 帮助模型跳出局部最优
- **实现**: 在trainer.py中添加scheduler
- **参数**: T_0=10, T_mult=2可根据验证集调整

## 方案13: EMA (指数移动平均)

```python
# 训练时维护EMA模型
EMA_decay = 0.9999
for param, ema_param in zip(model.parameters(), ema_model.parameters()):
    ema_param.data = EMA_decay * ema_param.data + (1 - EMA_decay) * param.data
```

- **优点**: 提升模型稳定性和泛化性能
- **推理时**: 使用EMA参数而非原始参数
- **额外开销**: 约1倍模型参数的显存

---

# 数据增强方案

## 方案14: 序列时间扰动

```python
def time_augment(seq_time_buckets, aug_range=(-2, 2)):
    """随机偏移时间桶ID，模拟时间不确定性"""
    if self.training:
        shift = random.randint(aug_range[0], aug_range[1])
        return torch.clamp(seq_time_buckets + shift, 0, NUM_TIME_BUCKETS-1)
    return seq_time_buckets
```

- **优点**: 增强模型对时间噪声的鲁棒性
- **实现**: 在dataset.py的_convert_batch中添加
- **风险**: 可能影响时间敏感的预测

## 方案15: 序列特征Dropout

```python
def seq_feature_dropout(seq_tokens, p=0.1):
    """随机mask掉序列中的某些特征位置"""
    mask = torch.rand(seq_tokens.shape[:2], device=seq_tokens.device) > p
    return seq_tokens * mask.unsqueeze(-1)
```

- **优点**: 防止过拟合，提升泛化性
- **实现**: 在模型forward的seq_tokens处添加
- **推荐p值**: 0.05-0.15

## 方案16: 特征交叉

```python
class FeatureCross(nn.Module):
    """显式特征交叉：为每对序列域生成交叉特征"""
    def __init__(self, d_model, num_sequences):
        super().__init__()
        self.cross_proj = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(num_sequences * (num_sequences - 1) // 2)
        ])

    def forward(self, seq_tokens_list):
        cross_features = []
        for i, j in combinations(range(len(seq_tokens_list)), 2):
            cross = torch.cat([seq_tokens_list[i].mean(1),
                              seq_tokens_list[j].mean(1)], dim=-1)
            cross_features.append(self.cross_proj[idx](cross))
        return cross_features
```

- **优点**: 显式建模序列域间关系
- **实现**: 在ns_tokens生成后添加
- **参数增加**: 约20%

---

# 正则化增强方案

## 方案17: DropPath (随机深度)

```python
def drop_path(x, drop_prob=0.1):
    if drop_prob > 0 and self.training:
        keep_prob = 1 - drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x / keep_prob * mask
    return x
```

- **优点**: 防止深层网络过拟合
- **实现**: 在每个HyFormerBlock的残差连接处添加
- **推荐drop_prob**: 0.05-0.2（随层数线性增加）

## 方案18: 标签平滑

```python
# 替换硬标签为平滑标签
label_smooth = 0.1
smooth_labels = labels * (1 - label_smooth) + label_smooth / 2
loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
```

- **优点**: 防止模型过度自信
- **实现**: 在trainer.py的_train_step中添加
- **推荐值**: 0.05-0.15

## 方案19:对抗训练 (FGM/PGD)

```python
class FGM:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon

    def attack(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if emb_name in name:
                param.data += self.epsilon * param.grad.sign()
```

- **优点**: 提升模型鲁棒性
- **实现**: 仅对Embedding层添加对抗扰动
- **时间开销**: 训练时间增加约20%

---

# 损失函数优化方案

## 方案20: 组合损失

```python
def combined_loss(logits, labels):
    # 结合BCE和Focal Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
    focal_loss = sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0)

    # AUC优化损失 (tweedie loss近似)
    probs = torch.sigmoid(logits)
    rank_loss = -(probs * labels).sum() / (labels.sum() + 1e-8)

    return 0.3 * bce_loss + 0.5 * focal_loss + 0.2 * rank_loss
```

- **优点**: 兼顾分类准确性和排序质量
- **权重可调**: 根据验证集AUC调整
- **风险**: 需仔细调参避免梯度冲突

## 方案21: 困难样本挖掘

```python
def hard_mining_loss(logits, labels, ratio=0.3):
    """聚焦于预测错误的样本"""
    probs = torch.sigmoid(logits)
    errors = torch.abs(probs - labels)
    n_hard = int(len(labels) * ratio)
    hard_idx = errors.topk(n_hard).indices
    return F.binary_cross_entropy_with_logits(
        logits[hard_idx], labels[hard_idx]
    )
```

- **优点**: 加速模型收敛
- **实现**: 替换或附加于主损失
- **推荐ratio**: 0.2-0.5

---

# 推荐实施优先级

## 立即尝试 (高ROI)
1. **混合精度训练 (方案10)** - 显存减半，速度提升30%
2. **标签平滑 (方案18)** - 一行代码，防止过拟合
3. **梯度累积 (方案11)** - 更稳定的训练
4. **学习率调度 (方案12)** - 跳出局部最优

## 短期优化 (1-2周)
5. **多尺度特征融合 (方案7)** - 架构改进
6. **门控注意力 (方案8)** - 增强模型容量
7. **DropPath (方案17)** - 深层网络必备
8. **组合损失 (方案20)** - 优化训练目标

## 中期探索 (2-4周)
9. **序列特征Dropout (方案15)** - 数据增强
10. **特征交叉 (方案16)** - 显式建模关系
11. **快速几何集成 (方案23)** - 低成本提升
12. **对抗训练 (方案19)** - 鲁棒性提升

## 长期研究 (1-2月)
13. **多模型融合 (方案22)** - 性能上限
14. **EMA (方案13)** - 稳定性提升
15. **困难样本挖掘 (方案21)** - 训练策略优化
