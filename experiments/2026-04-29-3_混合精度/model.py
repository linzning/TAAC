"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) 旋转位置编码模块。

    基于论文《RoFormer: Enhanced Transformer with Rotary Position Embedding》实现。
    核心思想：通过旋转矩阵将绝对位置信息注入到注意力机制中的相对位置表示中，
    使得内积运算能够自然地编码相对位置信息。

    数学原理：
        对于位置 m 的向量 x，将其按维度两两分组，对每组应用旋转矩阵：
        [x_{2i}  ]   [cos(m*θ_i)  -sin(m*θ_i)] [x_{2i}  ]
        [x_{2i+1}] = [sin(m*θ_i)   cos(m*θ_i)] [x_{2i+1}]
        其中 θ_i = base^{-2i/dim}，i ∈ [0, dim/2)

    预计算策略：
        在初始化时预先计算并缓存 cos/sin 值，forward 时仅做切片和 device 转移，
        避免运行时重复计算，同时保证与 torch.compile() 的兼容性。

    Attributes:
        dim: 旋转位置编码的维度，必须是偶数。对应注意力头中 key/query 向量的维度。
        max_seq_len: 缓存的最大序列长度，默认 2048。超过该长度需重新初始化。
        base: 旋转角度频率基数，默认 10000.0。控制位置编码的波长，
              值越大，长距离位置的区分度越低（波长越长）。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """初始化 RoPE 模块并预计算缓存。

        Args:
            dim: 旋转编码维度（必须是偶数，用于两两分组旋转）。
            max_seq_len: 最大序列长度，默认 2048。决定缓存大小。
            base: 频率基数，默认 10000.0。
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算每对维度的旋转角度倒数 inv_freq，形状为 (dim // 2,)
        # 公式：1 / (base ^ (2i / dim))，其中 i 取值范围为 [0, dim/2)
        # 该频率向量与位置索引 m 相乘后得到实际旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # persistent=False：该缓冲区不参与模型保存/加载（可运行时重建）
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 预计算全量 cos/sin 缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """构建位置编码缓存。

        计算从位置 0 到 seq_len-1 的所有 cos/sin 值，并注册为 persistent=False 的 buffer。
        仅在初始化或序列长度扩展时调用一次。

        Args:
            seq_len: 需要缓存的序列长度。

        张量形状流转：
            t:      (seq_len,)
            freqs:  (seq_len, dim // 2)  # 外积：每个位置 × 每个频率
            emb:    (seq_len, dim)        # 复制拼接为完整维度
            cached: (1, seq_len, dim)      # 添加 batch 维度便于广播
        """
        # 生成位置索引向量 t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # 计算每个位置在每个频率上的旋转角度：outer(t, inv_freq) = t_i * inv_freq_j
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)

        # 将 freqs 沿最后一个维度复制一份并拼接，得到 (seq_len, dim)
        # 因为旋转操作是逐对进行的，每对维度共享相同的旋转角度
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        # 预计算并缓存 cos 和 sin 值，添加维度 (1, seq_len, dim) 便于后续广播到 batch 维度
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据目标序列长度和设备返回预计算的 cos/sin 缓存切片。

        运行时仅执行切片和 device 转移操作，时间复杂度 O(1)，不影响 torch.compile()。

        Args:
            seq_len: 当前输入序列的实际长度（必须 <= max_seq_len）。
            device: 目标计算设备（如 cuda/cpu），用于将缓存转移到正确设备。

        Returns:
            cos: 形状为 (1, seq_len, dim) 的余弦缓存。
            sin: 形状为 (1, seq_len, dim) 的正弦缓存。

        Raises:
            IndexError: 当 seq_len > max_seq_len 时，切片操作可能越界。
        """
        # 从缓存中截取前 seq_len 个位置，并转移到目标设备
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """支持 Rotary Position Embedding (RoPE) 的多头自注意力模块。

    本模块实现了标准的 Multi-Head Self-Attention (MHSA) 机制，并在 Q/K 向量上注入
    旋转位置编码。区别于 PyTorch 原生 nn.MultiheadAttention，本实现手动完成
    Q/K/V 投影与 reshape，以便在点积前精确控制 RoPE 的注入时机。

    核心计算流程：
        1. 线性投影：query/key/value → Q/K/V，形状 (B, L, D)
        2. 多头拆分：将 D 拆分为 num_heads × head_dim，转置为 (B, num_heads, L, head_dim)
        3. RoPE 注入：对 Q 和 K 分别应用旋转位置编码（V 不加位置信息）
        4. 缩放点积注意力：调用 F.scaled_dot_product_attention 高效计算
        5. 门控融合：通过 sigmoid 门控 G 对注意力输出进行自适应加权
        6. 输出投影：合并多头结果并映射回 d_model 维度

    门控机制说明：
        输出 = W_o(out) * sigmoid(W_g(query))，其中 W_g 初始化为零权重、偏置 1.0，
        使得初始状态下门控近似恒等映射（sigmoid(1.0) ≈ 0.731），训练过程中逐步学习
        对注意力输出的自适应抑制/增强。

    Attributes:
        d_model: 模型维度，默认 64。必须是 num_heads 的整数倍。
        num_heads: 注意力头数，默认 4。决定并行注意力子空间的数量。
        head_dim: 每个注意力头的维度，等于 d_model // num_heads。
        rope_on_q: 是否为 Q 侧应用 RoPE，默认 True。在交叉注意力中可设为 False，
                   使 Q 侧（如全局查询 token）不携带序列位置信息。
        dropout: 注意力 dropout 概率，默认 0.0。仅在 training=True 时生效。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        """初始化多头注意力模块及投影矩阵。

        Args:
            d_model: 模型特征维度，必须是 num_heads 的整数倍。
            num_heads: 注意力头数。
            dropout: 注意力权重 dropout 概率，默认 0.0。
            rope_on_q: 是否为 query 侧应用 RoPE，默认 True。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        # 保证模型维度可被头数整除，否则 reshape 会失败
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Q/K/V/O 投影矩阵：均将 d_model 映射到 d_model
        # 标准 Transformer 中输出投影 W_o 承担合并多头的职责
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 门控投影 W_g：生成与输出同维度的门控信号，用于自适应调节注意力贡献
        self.W_g = nn.Linear(d_model, d_model)
        # 零权重初始化：初始门控信号仅由偏置决定
        nn.init.zeros_(self.W_g.weight)
        # 偏置初始化为 1.0：sigmoid(1.0) ≈ 0.731，近似恒等门控
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """执行带 RoPE 的多头注意力前向计算。

        完整的数据流形状：
            query (B, Lq, D) ──W_q──→ Q (B, Lq, D) ──reshape──→ (B, H, Lq, Hd)
            key   (B, Lk, D) ──W_k──→ K (B, Lk, D) ──reshape──→ (B, H, Lk, Hd)
            value (B, Lk, D) ──W_v──→ V (B, Lk, D) ──reshape──→ (B, H, Lk, Hd)
                                      ↓ RoPE 注入
                                Q_rot, K_rot ──SDPA──→ out (B, H, Lq, Hd)
                                      ↓ 合并 + 门控 + 输出投影
                                output (B, Lq, D)

        Args:
            query: 查询张量，形状 (B, Lq, D)。B 为 batch size，Lq 为查询序列长度。
            key: 键张量，形状 (B, Lk, D)。Lk 为键/值序列长度。
            value: 值张量，形状 (B, Lk, D)。通常 key 与 value 来自同一来源。
            key_padding_mask: 键侧填充掩码，形状 (B, Lk)。
                              True 表示该位置为填充（padding），不应被注意力关注。
            attn_mask: 注意力掩码，形状 (Lq, Lk) 或 (B*H, Lq, Lk)。
                       采用加法掩码形式：值为 -inf 的位置表示禁止关注，0 表示允许关注。
            rope_cos: RoPE 余弦缓存，形状 (1, L, head_dim)。
                      用于 KV 侧的位置编码；当未提供 q_rope_cos 时，也复用于 Q 侧。
            rope_sin: RoPE 正弦缓存，形状与 rope_cos 相同。用于 KV 侧。
            q_rope_cos: Q 侧专用 RoPE 余弦缓存，形状 (B, Lq, head_dim) 或 (1, Lq, head_dim)。
                        在交叉注意力场景中使用（如 LongerEncoder 的 top-k 聚合位置），
                        使 Q 侧位置编码与 KV 侧解耦。
            q_rope_sin: Q 侧专用 RoPE 正弦缓存，形状与 q_rope_cos 相同。
            need_weights: 兼容性参数，始终返回 None，不返回注意力权重。

        Returns:
            Tuple[Tensor, None]:
                - output: 注意力输出，形状 (B, Lq, D)。
                - None: 占位符，保持与 nn.MultiheadAttention 的接口兼容。
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # ── Step 1: 线性投影 ──
        # 分别通过独立线性层将输入映射到 Q/K/V 空间
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # ── Step 2: 多头拆分 ──
        # view: 将最后一个维度 D 拆分为 (num_heads, head_dim)
        # transpose: 将 head 维度提到序列长度前，便于后续并行计算
        # 最终形状：(B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # ── Step 3: 应用 RoPE 旋转位置编码 ──
        # RoPE 仅作用于 Q 和 K，V 不添加位置信息（值向量不需要位置感知）
        if rope_cos is not None and rope_sin is not None:
            # K 侧始终使用 rope_cos/rope_sin（标准 KV 位置编码）
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q 侧优先使用专用的 q_rope_cos/sin（如交叉注意力中 gathered 的位置）
                # 若未提供专用缓存，则回退到与 K 侧共享的 rope_cos/sin
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # ── Step 4: 掩码格式转换（适配 SDPA）──
        # PyTorch 的 scaled_dot_product_attention 要求 attn_mask 为 bool 类型，
        # True 表示"允许关注"，False 表示"忽略"。
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk)，True = padding（需忽略）
            # SDPA 期望: (B, num_heads, Lq, Lk)，True = attend（允许关注）
            # 因此先取反 (~)，再扩展维度以广播到所有头和查询位置
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: 加法掩码，-inf 表示禁止关注，0 表示允许关注
            # 转换为 bool：值为 0 的位置为 True（允许关注）
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            # 扩展到 (B, num_heads, Lq, Lk) 以匹配 SDPA 的广播要求
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                # 两种掩码取交集：同时满足填充掩码和注意力掩码的位置才允许关注
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # ── Step 5: 缩放点积注意力（SDPA）──
        # 调用 PyTorch 优化内核，自动选择 FlashAttention / Memory-Efficient Attention / Cudnn 后端
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # 处理全填充序列的边界情况：当某条序列所有 key 位置均为 padding 时，
        # softmax 的分母为 0，导致输出 NaN。将其替换为 0，使得残差连接后
        # 输出近似等于原始输入（零向量不改变加法结果）。
        out = torch.nan_to_num(out, nan=0.0)

        # ── Step 6: 合并多头、门控加权与输出投影 ──
        # transpose: (B, num_heads, Lq, head_dim) → (B, Lq, num_heads, head_dim)
        # contiguous + view: 合并多头维度为 (B, Lq, d_model)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)

        # 门控信号：基于原始 query 生成逐元素权重，sigmoid 保证范围在 (0, 1)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)

        # 最终线性投影：将门控后的注意力输出映射回 d_model 维度
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        """初始化 RankMixerBlock 实例。

        根据 mode 参数决定模块行为：
        - 'full'  : 启用 Token Mixing + Per-token FFN + 残差连接（完整功能）。
        - 'ffn_only': 仅启用 Per-token FFN + 残差连接（跳过 Token Mixing）。
        - 'none'  : 纯恒等映射，不创建任何子模块，forward 直接返回输入。

        Args:
            d_model (int): 模型隐藏维度，即每个 token 的向量维度 D。
            n_total (int): 输入序列总长度 T = Nq + Nns（查询 token 数 + 负采样 token 数）。
            hidden_mult (int, optional): FFN 中间层维度相对于 d_model 的倍数。
                中间层实际维度为 d_model * hidden_mult。默认为 4。
            dropout (float, optional): FFN 中的 Dropout 概率。默认为 0.0。
            mode (str, optional): 模块运行模式，可选 'full'、'ffn_only'、'none'。默认为 'full'。

        Raises:
            ValueError: 当 mode='full' 且 d_model 无法被 n_total 整除时抛出。
                Token Mixing 要求将 D 维均分为 T 个子空间，因此必须满足整除约束。
        """
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        # ---------------------------- 模式分支处理 ----------------------------
        if mode == 'none':
            # Pure identity mapping, no submodules created
            # 纯恒等映射：不创建任何可训练子模块，forward 直接透传输入
            return

        if mode == 'full':
            # Token Mixing 要求将 d_model 均分为 n_total 个子空间
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total  # 每个子空间的维度 d_sub = D / T

        # ---------------------------- Per-token FFN 定义 ----------------------------
        # 所有 token 共享同一套 FFN 参数（共享权重），适用于 'full' 和 'ffn_only' 两种模式
        self.norm = nn.LayerNorm(d_model)  # FFN 前的层归一化，稳定输入分布
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)  # 升维投影：D -> D * hidden_mult
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)  # 降维投影：D * hidden_mult -> D
        self.dropout = nn.Dropout(dropout)  # 防止过拟合的随机失活层

        # ---------------------------- 残差后处理 ----------------------------
        # Post-LN after residual to stabilize stacked block outputs
        # 在残差连接之后进行层归一化，缓解深层网络堆叠时的数值不稳定问题
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        """初始化 MultiSeqQueryGenerator 实例。

        为每条序列独立构建查询生成网络。核心思想是：
        将共享的 NS tokens 与该条序列的均值池化特征拼接为全局信息向量，
        再通过序列专属、查询独立的 FFN 网络生成 Nq 个查询 token。

        Args:
            d_model (int): 模型隐藏维度 D，所有 token 和全局信息向量的维度基础。
            num_ns (int): 负采样 token 数量 M，即共享 NS tokens 的个数。
            num_queries (int): 每条序列需要生成的查询 token 数量 Nq。
            num_sequences (int): 输入序列的总条数 S。
            hidden_mult (int, optional): 每个查询 FFN 中间层维度相对于 d_model 的倍数。
                中间层实际维度为 d_model * hidden_mult。默认为 4。
        """
        super().__init__()
        self.num_queries = num_queries    # 每条序列生成的查询 token 数 Nq
        self.num_sequences = num_sequences  # 序列总数 S
        self.d_model = d_model            # 隐藏维度 D

        # ---------------------------- 全局信息维度计算 ----------------------------
        # GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        # NS tokens 扁平化后维度 = num_ns * d_model，序列池化后维度 = d_model
        # 因此全局信息总维度 = (num_ns + 1) * d_model
        global_info_dim = (num_ns + 1) * d_model

        # ---------------------------- 全局信息层归一化 ----------------------------
        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        # 对拼接后的高维全局信息向量进行层归一化，防止大维度拼接导致梯度爆炸
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # ---------------------------- 查询生成 FFN 网络 ----------------------------
        # Each sequence has N independent FFNs
        # 构建嵌套 ModuleList 结构：外层按序列索引，内层按查询索引
        # 总参数量 = S * Nq 个独立 FFN，每个 FFN 结构为：
        #   Linear(global_info_dim -> d_model * hidden_mult)
        #   -> SiLU 激活
        #   -> Linear(d_model * hidden_mult -> d_model)
        #   -> LayerNorm(d_model)
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),  # 升维投影
                    nn.SiLU(),  # Swish 激活函数：x * sigmoid(x)，平滑非线性
                    nn.Linear(d_model * hidden_mult, d_model),  # 降维投影回 D 维
                    nn.LayerNorm(d_model),  # 输出层归一化，稳定查询 token 分布
                )
                for _ in range(num_queries)  # 为当前序列创建 Nq 个独立 FFN
            ])
            for _ in range(num_sequences)  # 为 S 条序列各创建一组查询生成器
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        """初始化 MultiSeqHyFormerBlock 实例。

        该模块是多序列 HyFormer 的核心构建块，包含三个子阶段：
        1. Sequence Evolution：为每条序列独立进行序列编码（如 SwiGLU / Transformer / Longer）。
        2. Query Decoding：为每条序列独立进行交叉注意力（Query tokens 关注编码后的序列）。
        3. Query Boosting：将所有序列的 Query tokens 与共享 NS tokens 合并，
           通过 RankMixerBlock 进行联合增强。

        Args:
            d_model (int): 模型隐藏维度 D。
            num_heads (int): 注意力头数，用于交叉注意力模块。
            num_queries (int): 每条序列生成的查询 token 数量 Nq。
            num_ns (int): 共享负采样 token 数量 Nns。
            num_sequences (int): 输入序列的总条数 S。
            seq_encoder_type (str, optional): 序列编码器类型，可选 'swiglu'、'transformer'、'longer'。
                默认为 'swiglu'。
            hidden_mult (int, optional): FFN / 编码器中间层维度相对于 d_model 的倍数。默认为 4。
            dropout (float, optional): Dropout 概率，应用于编码器和交叉注意力。默认为 0.0。
            top_k (int, optional): LongerEncoder 的 top-k 稀疏注意力参数。默认为 50。
            causal (bool, optional): 是否启用因果掩码（用于 LongerEncoder）。默认为 False。
            rank_mixer_mode (str, optional): RankMixerBlock 运行模式，可选 'full'、'ffn_only'、'none'。
                默认为 'full'。
        """
        super().__init__()
        self.num_sequences = num_sequences  # 序列总数 S
        self.num_queries = num_queries      # 每条序列的查询 token 数 Nq
        self.num_ns = num_ns                # 共享 NS token 数 Nns

        # ---------------------------- 序列编码器（Sequence Evolution） ----------------------------
        # Independent sequence encoder per sequence
        # 为每条序列独立创建一个序列编码器，各序列的编码过程互不干扰
        # 编码器类型由 seq_encoder_type 决定（SwiGLU / Transformer / Longer）
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)  # 共 S 个独立编码器
        ])

        # ---------------------------- 交叉注意力（Query Decoding） ----------------------------
        # Independent cross-attention per sequence
        # 为每条序列独立创建一个交叉注意力模块
        # Query tokens 作为 Query，编码后的序列特征作为 Key 和 Value
        # ln_mode='pre' 表示在注意力前进行层归一化（Pre-LN 架构）
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)  # 共 S 个独立交叉注意力模块
        ])

        # ---------------------------- RankMixer（Query Boosting） ----------------------------
        # RankMixer: input token count = Nq * S + Nns
        # 将所有序列的查询 token 与共享 NS tokens 拼接后送入 RankMixer
        # 总 token 数 = 每条序列 Nq 个查询 × S 条序列 + Nns 个共享 NS token
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        """初始化 GroupNSTokenizer 实例。

        将离散特征按组进行嵌入和投影，每组生成一个 NS token。
        工作流程：
        1. 为每个特征 fid 创建 Embedding 表（高基数特征可选择跳过）。
        2. 对多值特征进行均值池化，单值特征直接查表。
        3. 将每组内所有特征的嵌入向量拼接，通过投影层映射到 d_model 维度。

        Args:
            feature_specs (List[Tuple[int, int, int]]): 每个特征的三元组信息列表，
                每个元组格式为 (vocab_size, offset, length)。
                - vocab_size: 该特征的词典大小（决定 Embedding 表行数）。
                - offset: 该特征在 int_feats 张量中的起始列索引。
                - length: 该特征的取值个数（1 表示单值特征，>1 表示多值特征）。
            groups (List[List[int]]): 特征分组列表，每个组是 fid 索引的列表。
                每组内的特征嵌入后拼接，再投影为一个 NS token。
            emb_dim (int): 每个特征的嵌入维度。
            d_model (int): 模型隐藏维度，即投影层输出维度。
            emb_skip_threshold (int, optional): 高基数特征跳过阈值。
                当 vocab_size > emb_skip_threshold 且 emb_skip_threshold > 0 时，
                该特征不创建 Embedding 表，forward 时输出零向量。默认为 0（不跳过）。
        """
        super().__init__()
        self.feature_specs = feature_specs  # 特征元数据列表
        self.groups = groups                # 特征分组方案
        self.emb_dim = emb_dim              # 嵌入维度
        self.emb_skip_threshold = emb_skip_threshold  # 高基数跳过阈值

        # ---------------------------- Embedding 表构建 ----------------------------
        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        # 为每个特征 fid 创建 Embedding 表，满足以下任一条件则跳过：
        #   1. vocab_size <= 0（无有效词典信息）
        #   2. emb_skip_threshold > 0 且 vocab_size > emb_skip_threshold（高基数特征）
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)  # 标记为跳过，不创建 Embedding 表
            else:
                # +1 是为了容纳 padding_idx=0（零值作为填充标记，不参与梯度更新）
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        # 过滤掉 None，仅保留实际创建的 Embedding 表
        self.embs = nn.ModuleList([e for e in embs if e is not None])

        # ---------------------------- 特征索引映射 ----------------------------
        # Map from fid index to position in self.embs (or -1 if filtered)
        # 建立原始 fid 索引到 self.embs 中实际位置的映射：
        # - 若该特征未跳过，记录其在 self.embs 中的位置（real_idx）
        # - 若该特征被跳过，标记为 -1，forward 时输出零向量
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # ---------------------------- 组级投影层 ----------------------------
        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        # 为每个特征组创建一个投影网络：
        #   输入维度 = 组内特征数 × emb_dim（拼接后的嵌入向量）
        #   输出维度 = d_model（通过 LayerNorm 稳定分布）
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),  # 线性投影到模型维度
                nn.LayerNorm(d_model),  # 层归一化，稳定各组 token 的数值分布
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """初始化 RankMixerNSTokenizer 实例。

        遵循 RankMixer 论文思路的 NS token 生成器。与 GroupNSTokenizer 的核心区别在于：
        - GroupNSTokenizer：每组特征生成一个 NS token（token 数量 = 组数，不可调）。
        - RankMixerNSTokenizer：所有特征嵌入拼接为长向量后，等分成 num_ns_tokens 段，
          每段独立投影为一个 NS token（token 数量可自由设定，与组数解耦）。

        工作流程：
        1. 为每个特征 fid 创建 Embedding 表（高基数特征可选择跳过）。
        2. 将所有组内特征的嵌入向量拼接为一个超长向量。
        3. 将超长向量填充至可被 num_ns_tokens 整除，然后等分为 num_ns_tokens 段。
        4. 每段通过独立的投影网络映射到 d_model 维度，生成对应 NS token。

        Args:
            feature_specs (List[Tuple[int, int, int]]): 每个特征的三元组信息列表，
                每个元组格式为 (vocab_size, offset, length)。
                - vocab_size: 该特征的词典大小。
                - offset: 该特征在 int_feats 张量中的起始列索引。
                - length: 该特征的取值个数（1 表示单值，>1 表示多值）。
            groups (List[List[int]]): 特征分组列表，定义特征拼接的顺序。
                仅影响拼接顺序，不决定 NS token 数量。
            emb_dim (int): 每个特征的嵌入维度。
            d_model (int): 模型隐藏维度，即投影层输出维度。
            num_ns_tokens (int): 目标 NS token 数量 T。
                将所有特征嵌入等分为 T 段，每段生成一个 token。
            emb_skip_threshold (int, optional): 高基数特征跳过阈值。
                当 vocab_size > emb_skip_threshold 且 emb_skip_threshold > 0 时，
                该特征不创建 Embedding 表，forward 时输出零向量。默认为 0（不跳过）。
        """
        super().__init__()
        self.feature_specs = feature_specs  # 特征元数据列表
        self.groups = groups                # 特征分组方案（决定拼接顺序）
        self.emb_dim = emb_dim              # 嵌入维度
        self.num_ns_tokens = num_ns_tokens  # 目标 NS token 数量 T
        self.emb_skip_threshold = emb_skip_threshold  # 高基数跳过阈值

        # ---------------------------- Embedding 表构建 ----------------------------
        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        # 为每个特征 fid 创建 Embedding 表，满足以下任一条件则跳过：
        #   1. vocab_size <= 0（无有效词典信息）
        #   2. emb_skip_threshold > 0 且 vocab_size > emb_skip_threshold（高基数特征）
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)  # 标记为跳过，不创建 Embedding 表
            else:
                # +1 是为了容纳 padding_idx=0（零值作为填充标记，不参与梯度更新）
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        # 过滤掉 None，仅保留实际创建的 Embedding 表
        self.embs = nn.ModuleList([e for e in embs if e is not None])

        # ---------------------------- 特征索引映射 ----------------------------
        # Map from fid index to position in self.embs (or -1 if filtered)
        # 建立原始 fid 索引到 self.embs 中实际位置的映射：
        # - 若该特征未跳过，记录其在 self.embs 中的位置（real_idx）
        # - 若该特征被跳过，标记为 -1，forward 时输出零向量
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # ---------------------------- 分块维度计算 ----------------------------
        # Compute total embedding dim: sum of all fids across all groups
        # 计算所有特征嵌入拼接后的总维度：
        #   总特征数 = 所有组的特征数之和
        #   总嵌入维度 = 总特征数 × 每个特征的嵌入维度 emb_dim
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        # 将总嵌入维度填充至可被 num_ns_tokens 整除：
        #   chunk_dim = ceil(total_emb_dim / num_ns_tokens)  每段的目标维度
        #   padded_total_dim = chunk_dim * num_ns_tokens      填充后的总维度
        #   _pad_size = 填充的零值维度数
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # ---------------------------- 分段投影层 ----------------------------
        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        # 为每个 NS token 段创建一个独立的投影网络：
        #   输入维度 = chunk_dim（每段填充后的维度）
        #   输出维度 = d_model（通过 LayerNorm 稳定分布）
        # 共 num_ns_tokens 个投影器，彼此不共享参数
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),  # 线性投影到模型维度
                nn.LayerNorm(d_model),  # 层归一化，稳定各 token 的数值分布
            )
            for _ in range(num_ns_tokens)
        ])

        # ---------------------------- 初始化日志 ----------------------------
        # 打印 tokenizer 的关键维度信息，便于调试和验证配置
        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()

        # ================== 核心超参数与元数据保存 ==================
        self.d_model = d_model              # 模型隐藏维度，所有 token 的统一表示维度
        self.emb_dim = emb_dim              # Embedding 层输出维度，序列特征先嵌入到 emb_dim 再投影到 d_model
        self.action_num = action_num        # 分类任务输出维度，默认 1（二分类 logits）
        self.num_queries = num_queries      # 每序列查询 token 数量，控制信息压缩程度
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # 序列域名称列表，排序保证确定性顺序
        self.num_sequences = len(self.seq_domains)         # 序列域总数量（如点击序列、加购序列等）
        self.num_time_buckets = num_time_buckets             # 时间间隔分桶数，0 表示不启用时间嵌入
        self.rank_mixer_mode = rank_mixer_mode               # RankMixer 模式：'full' 或 'separate'
        self.use_rope = use_rope                             # 是否启用旋转位置编码（RoPE）
        self.emb_skip_threshold = emb_skip_threshold         # 词表大小超过此阈值时跳过 Embedding 创建，节省内存
        self.seq_id_threshold = seq_id_threshold             # 判断序列特征是否为 ID 类特征的阈值，ID 特征应用更强 Dropout
        self.ns_tokenizer_type = ns_tokenizer_type           # NS tokenizer 变体：'group' 或 'rankmixer'

        # ================== NS Tokens 构建（非序列特征压缩） ==================
        # NS（Non-Sequence）Tokenizer 的作用：将用户/物品的静态离散特征（如用户画像、商品类目）
        # 按预定义分组聚合成少量压缩 token，作为模型输入的一部分。
        # 支持两种变体：
        #   - 'group'：每组生成 1 个 NS token，结构简单。
        #   - 'rankmixer'（默认）：参考 RankMixer 论文，先将所有特征 Embedding 拼接，
        #     再分割投影为固定数量的 NS token，表达能力更强。
        if ns_tokenizer_type == 'group':
            # 原始方案：每个特征组对应 1 个 NS token
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer 风格：所有 Embedding 拼接 → 分割 → 投影为 num_ns_tokens 个 token
            # user_ns_tokens / item_ns_tokens 为 0 时自动回退到组数，保证兼容性
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # ================== 稠密特征投影层 ==================
        # 若输入中包含连续型（dense）特征，通过 Linear + LayerNorm 投影到 d_model 维度，
        # 使其与 NS token 和序列 token 处于同一语义空间。
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== NS Token 总数统计 ==================
        # 总 NS token 数 = 用户侧 NS token + 用户稠密 token（如有）
        #                  + 物品侧 NS token + 物品稠密 token（如有）
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== d_model 整除约束检查（仅 full 模式） ==================
        # full 模式下，RankMixer 需要将 d_model 均分为 T 份，每份对应一个 token 的专属子空间。
        # T = 每序列查询数 × 序列数 + NS token 总数。
        # 若不能整除，计算并提示所有合法的 T 值供调参参考。
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== 序列特征 Embedding 构建 ==================
        # seq_id_threshold 用于判断序列内的哪些特征属于“ID 类特征”（如 item_id、shop_id），
        # 这类特征词表通常极大，需施加更强的 Dropout（dropout_rate * 2）防止过拟合。
        # 注意：seq_id_threshold 与 emb_skip_threshold 完全独立：
        #   - emb_skip_threshold：决定是否创建 Embedding 层（内存优化）
        #   - seq_id_threshold：决定是否对特征施加额外 Dropout（正则化优化）
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """为单个序列域构建 Embedding 层列表。

            规则：
            1. 若词表大小 vs <= 0，或启用 emb_skip_threshold 且 vs > threshold，则跳过该特征（返回 None）。
            2. 否则创建 nn.Embedding(vs+1, emb_dim, padding_idx=0)，+1 为 padding 预留索引 0。

            返回：
                - module_list：实际创建的 Embedding 层（nn.ModuleList）
                - index_map：原始特征位置 → module_list 真实索引的映射，-1 表示被跳过
                - is_id：标记每个特征是否为 ID 类特征（vs > seq_id_threshold）
            """
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # 建立原始特征索引到 module_list 实际索引的映射，-1 表示该特征被跳过
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== 动态序列 Embedding 注册 ==================
        # 每个序列域（domain）独立维护一套 Embedding 和投影层，支持异构序列长度与特征维度。
        # _seq_embs[domain]：该域实际创建的 Embedding 层列表
        # _seq_emb_index[domain]：特征位置到 Embedding 层索引的映射（处理跳过特征）
        # _seq_is_id[domain]：标记各特征是否为 ID 类特征，用于前向时施加不同 Dropout
        # _seq_vocab_sizes[domain]：保存原始词表大小，供跳过统计和后续查找使用
        # _seq_proj[domain]：将拼接后的多特征 Embedding（len(vs)*emb_dim）投影到 d_model
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== 时间间隔分桶 Embedding（可选） ==================
        # 若 num_time_buckets > 0，为序列中相邻行为的时间间隔创建可学习嵌入。
        # 例如：65 个桶可覆盖从秒级到月级的行为间隔，增强模型对时间序列模式的感知。
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer 核心组件 ==================
        # QueryGenerator：基于 NS token 生成各序列的初始查询向量（learnable queries 与 NS 信息融合）
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # HyFormer Block 堆叠：每块内部包含序列内自注意力 + 序列间交叉注意力 + FFN，
        # 重复 num_hyformer_blocks 次以逐步提炼多序列交互表征。
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== 旋转位置编码（RoPE，可选） ==================
        # RoPE 通过旋转矩阵为注意力注入相对位置信息，替代绝对位置编码。
        # 若启用，为每个注意力头计算对应维度的旋转编码，base 控制波长。
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # ================== 输出投影层 ==================
        # 将各序列、各 query 的 d_model 维度表征拼接后，投影回 d_model，
        # 实现多源信息的深度融合与维度统一，供后续分类器使用。
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # ================== Dropout 与分类器 ==================
        # emb_dropout：对最终输入模型的 token 表征施加正则化
        self.emb_dropout = nn.Dropout(dropout_rate)

        # clsfier：两层 MLP 分类头，含 SiLU 激活、LayerNorm 和 Dropout，
        # 输出维度为 action_num（默认 1，对应二分类 logit）。
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # ================== 参数初始化 ==================
        # 统一初始化模型中所有 Linear 和 Embedding 层的权重，保证训练初期数值稳定性。
        self._init_params()

        # ================== emb_skip_threshold 过滤统计日志 ==================
        # 若启用了大词表跳过（emb_skip_threshold > 0），打印各序列域和 NS tokenizer 中
        # 被跳过特征的比例，方便确认内存优化效果及是否有重要特征被误跳过。
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
