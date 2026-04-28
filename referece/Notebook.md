# Notebook

## Rankmixer

## Hyformer

![image-20260426222540931](/Users/linzhengning/Library/Application Support/typora-user-images/image-20260426222540931.png)

### 问题构建

单样本的binary cross-entropy损失，采用label_type作为正负样本标志，

### Query generation

#### **1. 输入特征的准备（Input Tokenization）**

在生成查询之前，模型首先对输入进行分类。HyFormer 采用语义分组（Semantic Grouping）策略，将特征划分为不同的语义簇（如用户画像、上下文等），以保持结构化的归纳偏置 。

#### **2. 全局信息的集成（Global Info Integration）**

查询向量不仅包含当前的候选项目信息，还吸纳了全场景的上下文。其计算逻辑如下：

- **非序列特征 (Non-Sequential Features)**：记为 $F_1, F_2, ..., F_M$ 。

- **序列全局摘要 (Global Sequence Summary)**：通过对原始行为序列进行平均池化（MeanPool）得到，记为 $MeanPool(Seq)$ 。

- **公式 4：全局信息拼接**

  $$Global\ Info = Concat(F_1, ..., F_M, MeanPool(Seq))$$

#### **3. 查询向量的生成（Query Generation）**

模型利用轻量级的前馈神经网络（FFN）将上述全局信息映射为 $N$ 个并行查询 Token。

- **公式 3：查询矩阵生成**

  $$Q = [FFN_1(Global\ Info), ..., FFN_N(Global\ Info)] \in \mathbb{R}^{N \times D}$$

  **解释**： * $N$：生成的查询 Token 数量。通过生成多个查询（而非单个），模型可以从不同维度（如长期兴趣、短期意图等）并行捕捉用户的序列特征 。

  $D$：特征维度。 * 这些生成的 $Q$ 将作为 **Query Decoding** 模块的输入，去“询问”长序列中的 Key-Value 表征 

### Query decoding

#### **1. 序列表示编码（Sequence Representation Encoding）**

在进行解码之前，原始的行为序列 $S$ 必须先转换为神经网络可处理的层级化 Key-Value（K/V）形式 。HyFormer 提供了三种灵活的编码方案以适配不同的算力预算：

- **全 Transformer 编码**：使用标准 Transformer Encoder 捕捉全序列内部的精细交互 。

  $$H_l = \text{TransformerEnc}_l(S) \quad \text{[公式 5] [cite: 175]}$$

- **LONGER 风格的高效编码**：利用较短的压缩序列 $S_{short}$ 作为查询，与长序列 $S$ 进行交叉注意，将复杂度从 $O(L_S^2)$ 降至 $O(L_H L_S)$ 。

  $$H_l = \text{CrossAttn}(S_{short}, S, S) \quad \text{[公式 6] [cite: 179]}$$

- **轻量级解码编码**：完全抛弃注意力机制，仅通过 SwiGLU 等前馈网络变换，追求极致的推理延迟 

  $$H_l = \text{SwiGLU}_l(S) \quad \text{[公式 7] [cite: 185]}$$

最终，通过线性投影生成每一层特有的 Key 和 Value ：

$$K_l = H_l W_l^K, \quad V_l = H_l W_l^V \quad \text{[公式 8] [cite: 188, 189]}$$

#### **2. 交叉注意力解码（Query Decoding via Cross-Attention）**

这是解码模块的核心步骤。它使用从 **Query Generation** 获得的全局查询 $Q$（包含非序列特征和序列汇总信息）去检索上述生成的序列 K/V 。

- **公式 9：交叉注意力计算**

  $$\tilde{Q}_{(l)} = \text{CrossAttn}(Q_{(l)}, K_{(l)}, V_{(l)}) \quad \text{[cite: 194]}$$

  > 模型让非序列特征（如“用户今天在上海”）直接去长序列（如“过去一年的点击历史”）中寻找相关的信号 。这种**目标触发（Target-aware）**的提取方式比简单的序列池化要精准得多 。

### Query Boosting

#### **1. 统一查询表示（Unified Representation）**

在增强之前，模型将 Decoding 输出的序列感知查询 $\tilde{Q}_{(l)}$ 与原始的非序列特征（NS Tokens） $F$ 进行拼接，形成一个统一的任务空间 。

- **公式 10：统一矩阵构建**

  $$Q = [\tilde{Q}_{(l)}, F_1, ..., F_M] \in \mathbb{R}^{T \times D} \quad \text{}$$

#### **2. MLP-Mixer 风格的特征混合**

HyFormer 借鉴了 **RankMixer** 的思路，采用一种轻量级的 **MLP-Mixer** 机制进行 Token 混合 。它不使用计算昂贵的 Self-Attention，而是通过维度的重新排列和线性变换来实现信息交换 。

**(1) 空间切分（Subspace Partitioning）**

每个 Token $q_t$ 被切分为 $T$ 个通道子空间（Channel Subspaces） 。

- **公式 11：维度切分**

  $$q_t = [q_t^{(1)} || q_t^{(2)} || \cdots || q_t^{(T)}], \quad q_t^{(h)} \in \mathbb{R}^{D/T} \quad \text{[cite: 210]}$$

**(2) 跨 Token 聚合（Cross-Token Aggregation）**

对于每个子空间索引 $h$，模型将所有 Token 在该位置的子向量拼接起来，形成一个新的向量进行混合 。

- **公式 12：Token 混合聚合**

  $$\overline{q}_h = \text{Concat}(q_1^{(h)}, q_2^{(h)}, ..., q_T^{(h)}) \in \mathbb{R}^D \quad \text{[cite: 235]}$$

**(3) 逐 Token 细化（Per-Token Refinement）**

在完成横向混合后，模型再对每个 Token 独立应用一个前馈网络（FFN），进行非线性映射 。

- **公式 14：线性变换与激活**

  $$\tilde{Q} = \text{PerToken-FFN}(\hat{Q}) \quad \text{[cite: 245]}$$

最后，通过残差连接稳定训练，并保留原始解码的语义 。

- **公式 15：残差融合**

  $$Q_{boost} = Q + \tilde{Q} \quad \text{[cite: 249]}$$

### 多序列处理

![image-20260427192510085](/Users/linzhengning/Library/Application Support/typora-user-images/image-20260427192510085.png)

#### **独立查询解码机制**

HyFormer 为每条序列分配了一套专用的查询 Token 。

- **专用查询构建**：对于 $S$ 条不同的行为序列，模型会生成 $S$ 组对应的全局查询 Token 。
-  **序列特定解码**：在每个 HyFormer 块中，每组查询仅对其对应的序列表示进行 **Query Decoding**（交叉注意力操作）。
- **语义保留**：这种设计确保了在解码阶段，各序列的特定语义（如“搜索意图”与“购买偏好”）被完整保留，而不会相互干扰 。

#### **3. 跨序列的信息交互**

虽然解码过程是独立的，但异构序列之间的关联性通过随后的 **Query Boosting** 模块来实现 。

- **查询级混合**：所有序列解码后的输出（Cross Outputs）会与非序列特征（NS Tokens）一起进入 **MLP-Mixer** 。
- **异步交互**：在这种架构下，不同序列的信息不需要通过显式的长序列拼接来交互，而是通过各自的“摘要 Token”在 Mixer 模块中进行高阶特征交叉 。
- **资源动态分配**：该框架允许根据序列的重要性，为不同的序列自适应地分配不同数量的全局 Token，从而实现更精准的建模 。
