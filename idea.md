# HyFormer Baseline 改进方向分析

## 一、论文核心 vs 代码实现对比

HyFormer 的核心设计是**"交替优化"**：每层内先通过 **Query Decoding**（Cross-Attention 让 Query 读取序列 K/V）再 **Query Boosting**（Token Mixing + FFN 让 Query 与 NS 特征交互），形成序列与非序列的双向信息流。

代码基本复现了这个框架，支持 3 种序列编码器（SwiGLU / Transformer / Longer）、2 种 NS Tokenizer（Group / RankMixer），并扩展到**多序列**场景。

---

## 二、改进方向

### 1. 序列间交互机制（Inter-Sequence Interaction）已实验-有效

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

### 2. 最终输出显式融合 NS Tokens【高优先级】已实验-较为有效

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

## 三、特征工程

### 1. 基础属性特征（Raw Features）

直接对原始字段进行清理和编码，这是所有特征的基础。

- **类别特征（Categorical）：** 对 ID 类（UserID, ItemID）、属性类（城市、类别）进行 `Label Encoding` 或 `One-hot Encoding`。
- **数值特征（Numerical）：** 对价格、时长等进行归一化（Normalization）或标准化（Standardization），减少极值影响。
- **连续变量离散化：** 比如将“年龄”分桶（Binning），可以增强特征的鲁棒性，捕捉非线性关系。

### 2. 统计特征（Statistical Features）

通过历史数据统计，刻画实体的活跃度或受欢迎程度。

- **计数特征（Count）：** 比如用户近 7 天的点击数、某个商品的领取次数。
- **转化率/CTR（Target Encoding）：** 核心特征。计算某个特征（如 `BrandID`）在历史数据中的点击率。
  - *注意：* 为防止标签泄露，必须使用 **五折交叉统计** 或 **Leave-one-out** 方法。
- **分位数与偏离度：** 比如“当前商品价格”相对于“用户历史购买均价”的偏离程度，刻画用户的价格敏感度。

#### 3. 序列特征（Sequence Features）

推荐系统的本质是时序预测，挖掘用户的兴趣演变至关重要。

- **历史行为序列：** 用户最近点击的 5/10/50 个商品 ID。
- **时间间隔（Time Gap）：** * 距离上次行为的时间（判断用户意图是否还存在）。
  - 距离该商品上一次被点击的时间（刻画商品的实时热度）。
- **位置特征：** 物品在瀑布流中的曝光位置（用于消除位置偏见/Positional Bias）。

### 4. 交叉特征（Interaction Features）

手动模拟模型难以捕捉的高阶非线性组合。

- **笛卡尔积：** 比如 `UserID & Category`，刻画用户对特定类别的偏好。
- **业务逻辑交叉：** “性别 + 肤质”、“机型 + 游戏类型”。
- **统计交叉：** 计算用户在某个类别下的点击占比。

### 5. 向量化与 Embedding（Representation Learning）

利用特征提取技术将离散 ID 映射到连续空间。

- **协同过滤向量：** 通过矩阵分解（MF）得到的 User/Item Embedding。
- **图特征（Graph）：** 构建“用户-商品”二部图，利用 **DeepWalk** 或 **Node2vec** 生成 Embedding，捕捉二阶甚至高阶的相似性。
- **语义特征：** 利用预训练模型（如 BERT, CLIP）提取商品标题、图片的向量，计算 User-Item 的余弦相似度。

### 6. 趋势与动态特征（Dynamic Features）

- **窗口统计：** 计算近 1 小时、1 天、3 天的活跃度变化趋势。
- **衰减特征：** 给历史行为加上时间衰减因子（Time Decay），近期的行为权重更高。

------

### 💡 比赛中的提分技巧（Trick）

1. **穿越特征检查：** 确保特征提取时只使用了“当前时刻”之前的数据，严禁使用未来信息。
2. **特征选择：** 使用 `LightGBM` 或 `XGBoost` 跑一个基础模型，观察 **Feature Importance**，剔除重要性极低的噪声特征。
3. **零值处理：** 对于冷启动（新用户/新物品），需要设计专门的默认值填充逻辑。

---

## 四、竞赛特征工程方案（AUC 排序任务，当前 Baseline 0.811）

**前提约束**：
- 完整数据 2 亿条在官方平台，本地仅 demo 数据用于流程验证
- 除 `uid`、`item_id`、`label_type`、`label_time`、`timestamp` 外，所有特征匿名
- 损失函数为逐样本 BCE，label=1 为正样本，label=0 为负样本
- 评估指标：AUC

### 4.1 特征工程总览

| 层级 | 特征类别 | 优先级 | 实现方式 | 预期收益 |
|------|----------|--------|----------|----------|
| P0 | 候选物品-序列匹配特征 | 高 | 在线计算（dataset.py）| +0.003~0.008 |
| P0 | 时间感知增强特征 | 高 | 在线计算 | +0.002~0.005 |
| P1 | 序列内部统计特征 | 高 | 在线计算 | +0.001~0.003 |
| P1 | 跨序列聚合特征 | 中 | 在线计算 | +0.001~0.003 |
| P2 | 全局统计特征（CTR/计数）| 中 | 离线预计算 + 五折交叉 | +0.003~0.010 |
| P2 | Embedding 相似度特征 | 中 | 离线预训练 + 在线查表 | +0.002~0.005 |
| P3 | 匿名特征交叉 | 低 | 在线计算 | +0.001~0.002 |

---

### 4.3 P0：时间感知增强特征

当前仅有 `time_bucket`（粗粒度分桶）。增加以下精细时间特征：

| 特征名 | 计算方式 | 作用 |
|--------|----------|------|
| `{domain}_last_action_diff` | 当前 timestamp - 序列最近一次行为时间戳 | 用户活跃度/沉默度 |
| `{domain}_time_decay_score` | Σ exp(-λ × Δt_i) | 近期行为加权聚合 |
| `{domain}_time_gap_mean` | 序列内相邻行为时间差的均值 | 行为规律性 |
| `{domain}_time_gap_std` | 序列内相邻行为时间差的方差 | 行为规律性波动 |

**补充全局时间特征**：

| 特征名 | 计算方式 |
|--------|----------|
| `hour_of_day` | `timestamp % 86400 // 3600` | 一天中的小时（周期性） |
| `day_of_week` | `timestamp // 86400 % 7` | 星期几（周期性） |

**为什么有效**：HyFormer 中的 `time_embedding` 只是绝对时间分桶的加法，无法显式建模**相对时间差**和**时间衰减**。BCE 损失下，时间敏感的样本（如近期活跃用户的正样本）会被更准确地 scoring。

---

### 4.4 P1：序列内部统计特征

不依赖特征语义，仅从序列结构计算：

| 特征名 | 计算方式 |
|--------|----------|
| `{domain}_seq_len` | 已有 |
| `{domain}_unique_ratio` | 唯一 item 数 / seq_len | 兴趣集中度 |
| `{domain}_repeat_count` | seq_len - 唯一 item 数 | 重复行为强度 |
| `{domain}_top_freq` | 序列中出现最多次 item 的频率 | 主导兴趣强度 |

**跨序列聚合**：
| 特征名 | 计算方式 |
|--------|----------|
| `total_seq_len` | 4 个 domain seq_len 之和 |
| `active_domain_count` | seq_len > 0 的 domain 数 |
| `max_domain_ratio` | max(seq_len) / total_seq_len | 用户偏好哪个 domain |

---

### 4.5 P2：全局统计特征（离线预计算，收益通常最高）

由于完整数据在平台，可以在**训练集**上预计算以下统计量。**必须做防穿越处理**。

| 特征名 | 计算方式 | 防穿越方法 |
|--------|----------|-----------|
| `user_ctr` | 用户历史点击率 = 正样本数 / 总样本数 | 五折交叉统计 |
| `item_ctr` | 物品历史点击率 | 五折交叉统计 |
| `user_item_cooccur` | 用户-物品共现次数 | Leave-one-out |
| `user_avg_seq_len` | 用户平均序列长度 | 五折交叉统计 |
| `item_domain_dist` | 物品在各 domain 出现的分布熵 | 全局平滑 |

**五折交叉统计流程**：
1. 将训练数据随机分为 5 份
2. 对每份数据，用其他 4 份计算统计量（如 CTR）
3. 将统计量作为特征合并回原数据
4. 冷启动用户/物品用全局均值填充

**为什么对 BCE + AUC 最有效**：Target Encoding 直接将历史标签信息编码为特征，BCE 损失可以线性利用这些信号，AUC 对单调特征特别敏感。

---

### 4.6 P2：Embedding 相似度特征

**方案 A：Item2Vec（无需平台数据，本地用 demo 预训练流程）**
1. 提取所有序列中的 item_id 作为语料
2. 用 Word2Vec/Skip-gram 训练 item embedding（dim=16/32）
3. 保存 embedding 表，随代码提交到平台

**在线计算特征**：

| 特征名 | 计算方式 |
|--------|----------|
| `seq_item_emb_mean` | 序列中所有 item embedding 的均值 |
| `candidate_emb` | 候选 item 的 embedding |
| `emb_cos_sim` | cosine_similarity(seq_item_emb_mean, candidate_emb) | 兴趣匹配度 |

**方案 B：User-Item 协同过滤向量**
- 若平台允许，用矩阵分解（SVD/ALS）离线计算 user/item 隐向量
- 在线计算 user 向量与 item 向量的内积作为特征

---

### 4.7 P3：匿名特征交叉

由于特征匿名，只能做**盲目交叉**（blind crossing），收益不确定但可尝试：

| 交叉方式 | 示例 |
|----------|------|
| 标量 ID 哈希交叉 | `hash(user_int_scalar_fid_1, item_int_scalar_fid_1) % vocab` |
| 多值特征密度 | `user_int_array_15` 的非零元素占比 |
| Dense 特征统计 | `user_dense` 各维度的均值/最大值 |

**注意事项**：盲目交叉容易引入噪声，建议先用小数据验证有效性。

---

### 4.8 实施路线图（建议迭代顺序）

```
Week 1（快速验证）
├── Step 1: 实现候选物品-序列匹配特征（flag/count/last_pos/time_diff）
├── Step 2: 实现时间增强特征（last_action_diff, hour_of_day, day_of_week）
└── 目标：AUC +0.003~0.005

Week 2（深度挖掘）
├── Step 3: 序列统计特征（unique_ratio, repeat_count, cross-domain聚合）
├── Step 4: 尝试五折交叉的 user_ctr / item_ctr（若平台支持离线预计算）
└── 目标：AUC +0.002~0.005

Week 3（高阶特征）
├── Step 5: Item2Vec embedding 相似度特征
├── Step 6: 时间衰减加权 score（time_decay_sum）
└── 目标：AUC +0.001~0.003
```

---

### 4.9 风险与注意事项

1. **穿越风险**：所有全局统计特征必须用五折交叉或 Leave-one-out，严禁用未来信息。
2. **维度爆炸**：新增特征建议先作为 `user_dense` 或 `item_dense` 传入（连续值），而非扩展 `user_int`（离散值），避免 Embedding 层参数量激增。
3. **本地验证局限**：demo 数据仅 1000 条，分布可能与全量差异巨大。特征工程代码需保证在任意规模数据上都能稳定运行（如冷启动填充）。
4. **与模型协同**：HyFormer 已经能捕捉复杂的序列交互，特征工程应聚焦于**模型难以显式计算的统计量**（如全局 CTR、精确匹配信号），而非重复建模 Attention 已覆盖的能力。
5. **BCE 损失特性**：BCE 对正负样本比例敏感。若数据极度不平衡，建议同步尝试 Focal Loss（项目中已支持 `--use_focal_loss`）。

