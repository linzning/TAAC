"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    设计思路:
    在 CTR/CVR 模型中，同一类特征（如 user_int）通常由多个 feature 组成，
    为了高效地传给 nn.Embedding 或 Linear，我们会把它们 concat 成一个
    flatten 向量。FeatureSchema 负责记录每个 feature 在这个 flatten 向量中的
    起始位置 (offset) 和长度 (length)，以便后续模型层能精准切分。

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # 有序列表，存储 (feature_id, offset, length) 三元组。
        # 顺序即特征被 add 的顺序，也对应 flatten 张量中的排列顺序。
        self.entries: List[Tuple[int, int, int]] = []
        # 当前已分配的总维度；等于所有已 add 特征的 length 之和。
        # 也作为下一个新特征的 offset 起始点。
        self.total_dim: int = 0
        # 快速查找表: feature_id -> (offset, length)，避免遍历 entries。
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        # 当前总维度即为新特征的偏移量；特征连续排列，无间隙。
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        # 同步更新字典，保证 O(1) 查询。
        self._fid_to_entry[feature_id] = (offset, length)
        # 累加长度，为下一个特征准备 offset。
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        # 通过字典直接定位，用于模型层根据 fid 取对应的 embedding 段。
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        # 保持原始 add 顺序返回，常用于遍历或初始化 embedding 层。
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        # 便于保存到 schema.json 或 checkpoint，支持跨进程/跨会话重建。
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        # 反序列化：从 JSON dict 恢复完整的 Schema 状态。
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        # 可视化调试：打印每个 feature 的占用区间，方便核对维度。
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# 时间差分桶边界（64 个边界 → 65 个桶：0=padding，1..64）。
# 每个边界值单位是秒，覆盖从几秒到约一年的范围。
# 在序列特征中，用当前样本的 timestamp 减去序列项的时间戳得到 time_diff，
# 再通过 np.searchsorted 映射到对应的桶 ID，作为时间感知的 Embedding 输入。
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# 时间桶 Embedding 槽位总数（= 边界数量 + 1，其中 0 表示 padding）。
#
# 该常数由 BUCKET_BOUNDARIES 的长度唯一确定；在模型侧，
# ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` 必须与此值严格一致，
# 否则运行时可能抛出 IndexError。
#
# 因此 ``train.py`` / ``infer.py`` 只暴露布flag ``--use_time_buckets``，
# 具体的桶数量从此处自动推导，避免硬编码不一致。
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.

    设计思路与核心机制:
    1. 数据源与加载: 直接读取多列 Parquet 原始文件，通过 ``schema.json`` 解析特征布局。
       支持单文件或目录输入，并将文件按 Row Group 切分以支持多进程 DataLoader。
    2. 内存预分配: 预先分配 numpy 缓冲区（如 ``_buf_user_int``），避免每次 ``__iter__``
       时重复调用 ``np.zeros`` 和 ``np.stack`` 带来的开销，直接将数据写入预分配内存。
    3. 特征处理:
       - int 特征: 支持标量和变长数组（多热点），将 <= 0 的值映射为 0（padding），
         并根据词表大小检测及裁剪越界词表 ID。
       - dense 特征: 变长浮点数组，自动 padding 到最大维度 ``max_dim``。
       - 序列特征: 按 domain 分组，包含 side-info 和可选的时间戳列。
         时间戳与当前样本时间做差后，通过 ``BUCKET_BOUNDARIES`` 分桶，作为时间感知 Embedding 输入。
    4. 标签生成: 训练模式下，标签由 ``label_type == 2`` 转换而来；非训练模式返回全零标签。
    5. 数据混洗: 内部维护一个基于 batch 数量的 Shuffle Buffer，在 buffer 满时进行行级混洗并切片输出。
    6. 多进程安全: 通过 ``file_system`` tensor 共享策略和按 worker 切分 Row Group 解决多 worker 下的内存与数据重复问题。
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.

        中文注释:
            parquet_path: Parquet 数据路径，可以是包含 ``*.parquet`` 文件的目录，也可以是单个 Parquet 文件路径。
            schema_path: Schema JSON 文件路径，用于描述各特征的布局信息。
            batch_size: 固定批大小，用于预分配 numpy 缓冲区。
            seq_max_lens: 可选的按域序列截断长度覆盖，例如 ``{'seq_d': 256}``，未指定的域默认回退到 256。
            shuffle: 是否在 ``buffer_batches`` 大小的窗口内进行混洗。
            buffer_batches: 混洗缓冲区大小，单位为批次数。
            row_group_range: Row Group 的切片范围 ``(start, end)``，``None`` 表示使用全部 Row Group。
            clip_vocab: 若为 True，将越界的词表 ID 裁剪为 0；若为 False，则直接抛出异常。
            is_training: 若为 True，从 ``label_type == 2`` 推导标签；若为 False，返回全零标签列。
        """
        super().__init__()

        # 支持传入目录或单个 Parquet 文件路径。
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        # 越界词表统计: {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        # 用于训练结束后诊断数据质量，不干扰主流程。
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # 构建 Row Group 列表：Parquet 的 Row Group 是列式存储的最小读取单元，
        # 比文件粒度更细，便于多 worker 并行和数据切分。
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        # 如果指定了 row_group_range，只保留该范围内的 Row Group。
        # 训练集和验证集通过此参数划分，避免重复读取同一文件。
        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # 加载 schema.json，解析各类特征的 id、维度、词表大小及序列域配置。
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- 预计算列名到列索引的映射 ----
        # Arrow 按列名读取数据，但在 batch 转换时频繁做字符串查找开销大。
        # 此处一次性建立 name -> index 字典，后续全部用整数索引访问列。
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- 预分配 numpy 缓冲区 ----
        # 核心性能优化：避免在 __iter__ 循环中反复调用 np.zeros 和 np.stack。
        # 直接在初始化时分配最大 batch 所需的内存，每次只 slice 前 B 行并重置为 0。
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            # 3D 缓冲区: [batch_size, n_sideinfo_features, max_seq_len]
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            # 时间桶缓冲区: [batch_size, max_seq_len]
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            # 实际序列长度: [batch_size]
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- 预计算 int 列的读取计划 ----
        # 将 (feature_id, vocab_size, dim) 映射为 (col_idx, dim, offset, vocab_size)。
        # 在 _convert_batch 中按此 plan 直接取列、写 buffer，无需再查 schema。
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # 序列列读取计划: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        # side_plan 描述每个 side-info 特征在 Arrow 中的列索引、在 3D buffer 中的槽位、词表大小。
        # ts_col_idx 为时间戳列索引，用于后续 Time Bucketing。
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``.

        Parses ``schema.json`` and builds ``FeatureSchema`` objects, vocab size
        tables, and sequence-domain metadata needed by the dataset.

        中文说明:
        从 schema.json 加载各类特征的元数据，构建 FeatureSchema、词表大小表
        以及序列域配置。这些结构在后续 ``_convert_batch`` 中用于快速定位
        每列数据在预分配 buffer 中的写入位置。
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        # dim == 1 表示标量 int；dim > 1 表示变长 int 数组（如多热点特征），
        # 实际长度不超过 dim，不足部分 padding 到 dim。
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            # 每个特征槽位对应一个词表大小；标量重复 1 次，数组重复 dim 次。
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        # 逻辑与 user_int 完全一致，只是作用于 item（物品/广告）侧。
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        # dense 特征只有 dim 信息，不需要词表大小（直接传入 MLP）。
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        # 当前项目中 item_dense 未使用，保留空 schema 以保持接口一致性。
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        # seq 配置按 domain 分组，每个 domain 包含 prefix、ts_fid（时间戳特征id）、
        # features（[(fid, vocab_size), ...]）等字段。
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            # prefix 用于拼接列名，例如 prefix='seq_a' + fid=3 -> 列名 'seq_a_3'。
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            # sideinfo 是除时间戳外的所有特征，用于序列编码器的主输入。
            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            # 按 sideinfo 顺序收集词表大小，用于模型侧创建 embedding 层。
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: 优先使用用户传入的 seq_max_lens 覆盖；未指定则回退到 256。
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """按 worker 分片迭代 Row Group，读取 Arrow batch 并转换为训练张量。

        核心流程:
        1. 若多 worker 运行，按 worker_id 对 Row Group 取模分片，避免数据重复。
        2. 遍历分片后的 Row Group，用 ``pyarrow.parquet`` 以 ``batch_size`` 为粒度迭代。
        3. 每个 Arrow RecordBatch 经 ``_convert_batch`` 转为 dict of tensors。
        4. 若开启 shuffle，则将 batch 暂存到 buffer，满 ``buffer_batches`` 后行级混洗输出。
        5. 迭代结束后清空剩余 buffer，释放内存。
        """
        # ---- 多 worker 分片 ----
        # IterableDataset 不会自动划分数据，必须由我们在 __iter__ 内部根据 worker_id 切片。
        # 每个 worker 只处理索引满足 ``i % num_workers == worker_id`` 的 Row Group，
        # 从而保证不同进程之间数据互不重叠。
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        # Shuffle buffer: 暂存尚未混洗的 batch，攒够 buffer_batches 后统一打乱并切片输出。
        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            # 指定 row_groups=[rg_idx] 实现精细读取：只读当前 Row Group，不加载整个文件。
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    # 攒 batch：先将当前 batch 加入 buffer，等待后续统一混洗。
                    buffer.append(batch_dict)
                    # 当 buffer 中的 batch 数量达到阈值，触发一次行级 shuffle 并清空 buffer。
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    # 不开启 shuffle 或 buffer_batches <= 1 时直接 yield，保持数据顺序。
                    yield batch_dict

        # 迭代结束：处理 buffer 中剩余的不足 buffer_batches 的尾批。
        if buffer:
            yield from self._flush_buffer(buffer)

        # 显式删除 buffer 并触发 gc，避免多 epoch 迭代时内存泄漏。
        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """检测并记录越界词表 ID，根据 ``clip_vocab`` 决定裁剪或抛异常。

        在 CTR/CVR 场景中，原始数据可能包含比 schema.json 声明的 vocab_size
        更大的 ID（如数据漂移、新词未更新词表）。本函数在训练时静默统计这些
        越界值，避免大量日志刷屏，同时保证模型不会因 Embedding 越界而崩溃。

        Args:
            group: 特征分组标识，如 ``'user_int'``、``'seq_click'``，用于统计归类。
            col_idx: Arrow 列索引，精确定位是哪一列出现越界。
            arr: 待检测的 int64 数组（会被原地修改若 clip_vocab=True）。
            vocab_size: 该列允许的最大 ID（不含），即合法范围 ``[0, vocab_size)``。

        Raises:
            ValueError: 当 ``clip_vocab=False`` 且检测到越界值时抛出。
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """输出越界统计信息，用于训练结束后诊断数据质量。

        遍历 ``_oob_stats`` 中收集的所有越界记录，按 group 和列索引汇总输出。
        若提供 ``path`` 则写入文件，否则通过 ``logging.info`` 打印。

        Args:
            path: 输出文件路径；若为 ``None`` 则仅打印日志。
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """将 Arrow ``ListArray<int>`` 填充/截断为固定形状 ``[B, max_len]``。

        Parquet 中变长 int 列（如多热点特征）以 Arrow ListArray 存储，
        每行的长度不一。本函数将其展开为稠密二维数组，并记录每行的实际长度。
        所有 ``<= 0`` 的值（含原始缺失值 ``-1``）统一映射为 ``0``（padding）。

        Args:
            arrow_col: PyArrow ListArray，每行是一个变长 int 列表。
            max_len: 目标最大长度；超过则截断，不足则补 0。
            B: 当前 batch 的行数。

        Returns:
            ``(padded, lengths)``，其中 ``padded`` 形状为 ``[B, max_len]``，
            ``lengths`` 形状为 ``[B]``，记录每行原始有效长度（已截断到 max_len）。
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """将 Arrow ``ListArray<float>`` 填充/截断为固定形状 ``[B, max_dim]``。

        与 ``_pad_varlen_int_column`` 逻辑相同，但作用于 float 列（dense 特征）。
        缺失值在 Arrow 层通常以空列表表示，直接补 0 即可，无需额外 <=0 过滤。

        Args:
            arrow_col: PyArrow ListArray，每行是一个变长 float 列表。
            max_dim: 目标最大维度；超过则截断，不足则补 0.0。
            B: 当前 batch 的行数。

        Returns:
            形状为 ``[B, max_dim]`` 的 float32 稠密数组。
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """将 Arrow RecordBatch 转换为训练就绪的张量字典。

        这是数据管道的核心转换函数，负责把 PyArrow 的列式数据映射到预分配的
        numpy buffer，再转为 PyTorch Tensor。处理流程包括：

        1. 提取元信息（timestamp、label、user_id）。
        2. 处理 user_int / item_int：标量直接取值，变长列表经 padding 后写入 buffer。
        3. 处理 user_dense：变长 float 列 padding 后写入 buffer。
        4. 处理序列特征：将各 side-info 列 fuse 写入 3D buffer，并计算 time bucket。
        5. 越界检测：对所有 int 特征调用 ``_record_oob``，确保 Embedding 安全。

        所有操作尽量原地写入预分配 buffer，避免频繁的 ``np.zeros`` 和内存拷贝。

        Args:
            batch: PyArrow RecordBatch，包含一个 Row Group 内的若干行数据。

        Returns:
            Dict，键包括 ``user_int_feats``、``user_dense_feats``、``item_int_feats``、
            ``item_dense_feats``、``label``、``timestamp``、``user_id``、
            各 ``seq_domain``、``seq_domain_len``、``seq_domain_time_bucket`` 等。
        """
        # 当前 batch 的样本数（batch size）。
        B = batch.num_rows

        # ────────────────────────────── meta 信息提取 ──────────────────────────────
        # 从全局 'timestamp' 列提取当前样本的时间戳（Unix 秒级），用于后续 time_diff 计算。
        # shape: (B,)
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        # label_type == 2 表示正样本（转化），其余为负样本。
        # fill_null(0) 处理缺失值，避免 Arrow null 导致 numpy 转换失败。
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            # 推理模式下没有 label_type 列，统一填充 0。
            labels = np.zeros(B, dtype=np.int64)

        # user_id 用于后续可能的用户级采样或调试，以 Python list 形式返回（非张量）。
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ────────────────────────────── user_int ──────────────────────────────
        # 将用户侧整型特征写入预分配 buffer。
        # 处理规则：
        #   - null / -1 统一映射为 0（padding），因为 Embedding 层 padding_idx=0。
        #   - vocab_size == 0 的特征（无词表信息），在 dataset 侧强制置 0，防止模型侧
        #     1-slot Embedding 越界（模型会为 vs=0 创建占位 Embedding，但只有索引 0 有效）。
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                # 标量整型特征：直接展平为一维数组。
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                # 变长整型列表（如 multi-hot）：先 padding/truncation 到固定长度 dim。
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ────────────────────────────── item_int ──────────────────────────────
        # 逻辑与 user_int 完全一致：候选物品侧的整型特征（item_id、类目等）。
        # 注意：item_int 通常包含候选 item 的静态属性，与序列特征中的历史 item 区分开。
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ────────────────────────────── user_dense ──────────────────────────────
        # 用户侧稠密特征（浮点型），直接 padding 后写入 buffer。
        # 当前 item_dense 未使用，故后续 result 中置为空张量。
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        # ────────────────────────────── 组装非序列输出 ──────────────────────────────
        # 将 numpy buffer 转为 PyTorch Tensor 并组装为字典。
        # 注意：所有 buffer 均做 .copy()，避免 numpy 视图与后续 batch 的写操作冲突。
        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),      # (B, user_int_dim)
            'user_dense_feats': torch.from_numpy(user_dense.copy()),  # (B, user_dense_dim)
            'item_int_feats': torch.from_numpy(item_int.copy()),      # (B, item_int_dim)
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),  # 占位，当前未使用
            'label': torch.from_numpy(labels),                        # (B,)
            'timestamp': torch.from_numpy(timestamps),                # (B,)  当前样本时间戳
            'user_id': user_ids,                                       # list[str]
            '_seq_domains': self.seq_domains,                          # 告知模型有哪些序列域
        }

        # ────────────────────────────── Sequence features ──────────────────────────────
        # 核心：将变长序列（list<struct>）转为定长 3D 张量 (B, num_sideinfo, max_len)。
        # 每个 domain（如点击序列、加购序列）独立处理，最终输出：
        #   - result[domain]:          序列 sideinfo 张量    (B, num_features, max_len)
        #   - result[f'{domain}_len']: 实际序列长度          (B,)
        #   - result[f'{domain}_time_bucket']: 时间分桶 ID   (B, max_len)
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            # side_plan: [(col_idx, slot_idx, vocab_size), ...]  sideinfo 列的读取计划
            # ts_ci:     序列时间戳列在 batch 中的列索引；若为 None 表示该 domain 无时间戳。
            side_plan, ts_ci = self._seq_plan[domain]

            # 3D buffer: (B, num_sideinfo_features, max_len)。
            # 例如 seq_a 有 9 个 sideinfo，max_len=256，则 shape=(B, 9, 256)。
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # ── Step 1: 收集所有 sideinfo 列的 Arrow ListArray 元信息 ──
            # Arrow 的 ListArray 底层由 offsets（每个元素的起止位置）和 values（扁平数据）组成。
            # 这里一次性提取所有列的 (offsets, values, vocab_size, col_idx)，避免在嵌套循环中
            # 重复调用 batch.column() 和类型转换。
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            # ── Step 2: 将变长序列 fuse 写入 3D buffer ──
            # 对每个样本 i、每个特征 c：
            #   - 根据 offsets 找到该样本在该列中的原始值段 [s, e)
            #   - 取实际长度 rl = e - s，若 rl > max_len 则截断（只保留最近的 max_len 个）
            #   - 写入 out[i, c, :ul]，同时更新 lengths[i] 为当前最大实际长度。
            #
            # 注意：不同 sideinfo 列的序列长度理论上一致（来自同一行为的多个属性），
            # 但若出现不一致，以最长的为准（这样短列的尾部保持 0-padding）。
            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # 对所有 int 值做统一后处理：<=0 视为非法或 padding，强制置 0。
            # （Parquet 中缺失值、-1 占位符等都会落在这个区间。）
            out[out <= 0] = 0

            # ── Step 3: 越界检测与清洗 ──
            # 对每一列检查 vocab_size：
            #   - vs > 0：正常特征，调用 _record_oob 记录越界值（用于调试监控）。
            #   - vs == 0：该特征无词表信息，该列全部置 0，防止模型 Embedding 越界。
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            # 将处理后的序列数据加入 result。
            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # ── Step 4: Time bucketing（时间分桶）──
            # 目标：为序列中每个行为计算"距离当前样本有多久"，并映射到离散桶 ID。
            # 最终输出 shape: (B, max_len)，作为 nn.Embedding 的输入。
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                # ts_col: Arrow ListArray，结构与其他 sideinfo 列相同，存储每个行为的时间戳。
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()

                # 将变长时间戳序列 padding 到定长 (B, max_len)。
                # 未填充位置保持 0（后续通过 ts_padded == 0 将对应 bucket 也置 0）。
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                # 计算时间差：当前样本时间 - 序列项时间。
                # timestamps shape: (B,)  -> 扩展为 (B, 1) 以便广播。
                # 结果 shape: (B, max_len)。
                # np.maximum(..., 0) 保证时间差非负（防止数据异常导致负值）。
                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)

                # np.searchsorted(BUCKET_BOUNDARIES, time_diff) 返回每个 time_diff 在
                # BUCKET_BOUNDARIES 中的插入位置，即桶索引的"原始值"。
                #
                # 数值映射说明：
                #   - time_diff < 5s          -> raw_bucket = 0  -> bucket = 1
                #   - 5s <= time_diff < 10s   -> raw_bucket = 1  -> bucket = 2
                #   - ...
                #   - time_diff >= 1年        -> raw_bucket = 63 -> bucket = 64
                #
                # 为防止 time_diff 超过最大边界导致 raw_bucket = 64（越界），
                # 先 clip 到 [0, len(BUCKET_BOUNDARIES)-1]，再加 1 得到最终桶 ID [1, 64]。
                # bucket = 0 保留给 padding（序列中无行为或 ts_fid 为 null 的位置）。
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1

                # 序列 padding 位置（ts_padded == 0）对应 bucket 也置 0，
                # 这样模型侧的 nn.Embedding(padding_idx=0) 会输出零向量。
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    中文说明:
    从原始多列 Parquet 文件创建训练/验证 DataLoader 的工厂函数。
    验证集取自所有 Row Group 的末尾 ``valid_ratio`` 比例（按 glob 返回的文件顺序），
    避免训练集和验证集数据交叉污染。

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    # 固定随机种子，保证 Row Group 的划分顺序可复现。
    # 注意：这里的 random 只用于 get_pcvr_data 内部的分割逻辑，
    # Dataset 内部的 shuffle 使用独立的 torch 随机生成器。
    random.seed(seed)

    import glob as _glob
    # 排序保证多机/多进程环境下文件顺序一致，避免训练/验证划分出现不确定性。
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    # 收集所有 Row Group 的元信息：(文件路径, RowGroup索引, 该Group行数)。
    # Row Group 是 Parquet 的物理读取单元，比文件粒度更细，便于精确切分。
    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    # 按 valid_ratio 从末尾切分验证集；至少保留 1 个 Row Group 给验证，
    # 防止 total_rgs 很小时 valid_ratio 四舍五入为 0 导致没有验证数据。
    # 特殊处理：当 total_rgs=1 且 valid_ratio=0 时，全部用于训练
    if total_rgs == 1 and valid_ratio == 0:
        n_valid_rgs = 0
        n_train_rgs = 1
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: 仅使用训练 Row Group 的前 N%（用于快速实验或子集训练）。
    # 例如 train_ratio=0.1 时，只用 10% 的训练数据，加速调参迭代。
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    # 训练 Dataset：开启 shuffle 和 buffer_batches，使用 [0, n_train_rgs) 的 Row Group。
    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
    )

    # 根据是否可用 CUDA 决定 pin_memory；开启 persistent_workers 可减少多 worker 进程创建开销。
    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    # 训练 DataLoader: batch_size=None 因为 Dataset 本身已经产出 batch；
    # num_workers 负责多进程读取，pin_memory 加速 CPU->GPU 传输。
    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    # 验证 Dataset：关闭 shuffle 和 buffer，使用 [n_train_rgs, total_rgs) 的 Row Group。
    # 验证时数据顺序固定，便于复现评估结果。
    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    # 验证 DataLoader: num_workers=0 避免多进程开销（验证通常数据量小，且需要确定性）。
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
