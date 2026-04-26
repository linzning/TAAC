---
license: cc-by-nc-4.0
tags:
  - TAAC2026
  - recommendation
---

# TAAC2026 Demo Dataset (1000 Samples) 


> [!WARNING] ⚠️**Update[2026.04.10]:**
> This demo dataset has been updated to newest version with the following changes:
>
> - The parquet file is now a **flat column layout**, with all features as top-level columns.
> - Add a sequence feature, rename feature names and update some features.
> Participants should refer to the updated `demo_1000.parquet` and this `README.md` for the latest schema and data details.


A sample dataset containing 1000 user-item interaction records for the [TAAC2026 competition](https://algo.qq.com/). This dataset uses a **flat column layout** — all features are stored as individual top-level columns instead of nested structs/arrays.

## Dataset Overview

| Property | Value |
|---|---|
| **File** | `demo_1000.parquet` |
| **Rows** | 1,000 |
| **Columns** | 120 |
| **File Size** | ~39 MB |

## Columns

The 120 columns fall into **6 categories**:

| Category | Count | Data Type | Description |
|---|---|---|---|
| **ID & Label** | 5 | `int64` / `int32` | Core identifiers, label, and timestamp |
| **User Int Features** | 46 | `int64` / `list<int64>` | Integer-valued user features (scalar or array) |
| **User Dense Features** | 10 | `list<float>` | Float-array user features |
| **Item Int Features** | 14 | `int64` / `list<int64>` | Integer-valued item features (scalar or array) |
| **Domain Sequence Features** | 45 | `list<int64>` | Behavioral sequence features from 4 domains |

---

## Detailed Column Schema

### ID & Label Columns (5 columns)

All these 5 columns have no `null` value.

| Column | Data Type | 
|---|---|
| `user_id` | `int64` |
| `item_id` | `int64` |
| `label_type` | `int32` | 
| `label_time` | `int64` | 
| `timestamp` | `int64` |

> [!NOTE] **Note:**
> When `user_int_feats_{fid}` and `user_dense_feats_{fid}` share the same `{fid}`, they are aligned and jointly describe the same entity or signal.

### User Int Features (46 columns)

- `user_int_feats_{1,3,4,48-59,82,86,92-109}`: Scalar `int64`, total 35 columns.
- `user_int_feats_{15, 60, 62-66, 80, 89-91}`: Array `list<int64>`, total 11 columns.


### User Dense Features (10 columns)

- `user_dense_feats_{61-66, 87, 89-91}`: Array `list<float>`, total 10 columns.


### Item Int Features (14 columns)

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`: Scalar `int64`, total 13 columns.
- `item_int_feats_{11}`: Array `list<int64>`, total 1 column.


### Domain Sequence Features (45 columns)

`list<int64>` sequences from 4 behavioral domains:

- `domain_a_seq_{38-46}`: 9 columns
- `domain_b_seq_{67-79, 88}`: 14 columns
- `domain_c_seq_{27-37, 47}`: 12 columns
- `domain_d_seq_{17-26}`: 10 columns

---

## Usage

```python
import pyarrow.parquet as pq
import pandas as pd

# Read the parquet file
df = pd.read_parquet("demo_1000.parquet")

print(df.shape)       # (1000, 120)
print(df.columns)     # ['user_id', 'item_id', 'label_type', ...]
```

With Hugging Face `datasets`:
```python
from datasets import load_dataset

ds = load_dataset("TAAC2026/data_sample_1000")
print(ds)
```

---

# TAAC2026 演示数据集（1000 条样本）

> [!WARNING] ⚠️**更新[2026.04.10]：**
> 该演示数据集已更新至最新版本，主要变更如下：
>
> - Parquet 文件现为**扁平列布局**，所有特征均为顶层列。
> - 新增序列特征，重命名部分特征名并更新若干特征。
> - 参赛者请参考更新后的 `demo_1000.parquet` 及本 `README.md` 以获取最新的数据结构和数据详情。

本文档为 [TAAC2026 竞赛](https://algo.qq.com/) 提供的包含 1000 条用户-物品交互记录的样本数据集。该数据集采用**扁平列布局** —— 所有特征均以独立的顶层列存储，而非嵌套的 struct/array。

## 数据集概览

| 属性 | 值 |
|---|---|
| **文件** | `demo_1000.parquet` |
| **行数** | 1,000 |
| **列数** | 120 |
| **文件大小** | ~39 MB |

## 列说明

这 120 列分为 **6 大类**：

| 类别 | 数量 | 数据类型 | 说明 |
|---|---|---|---|
| **ID & 标签** | 5 | `int64` / `int32` | 核心标识符、标签及时间戳 |
| **用户整型特征** | 46 | `int64` / `list<int64>` | 整数值用户特征（标量或数组） |
| **用户稠密特征** | 10 | `list<float>` | 浮点数组用户特征 |
| **物品整型特征** | 14 | `int64` / `list<int64>` | 整数值物品特征（标量或数组） |
| **域序列特征** | 45 | `list<int64>` | 来自 4 个行为域的序列特征 |

---

## 详细列结构

### ID & 标签列（5 列）

以下 5 列均不含 `null` 值。

| 列名 | 数据类型 |
|---|---|
| `user_id` | `int64` |
| `item_id` | `int64` |
| `label_type` | `int32` |
| `label_time` | `int64` |
| `timestamp` | `int64` |

> [!NOTE] **注意：**
> 当 `user_int_feats_{fid}` 与 `user_dense_feats_{fid}` 具有相同的 `{fid}` 时，它们是对齐的，共同描述同一实体或信号。

### 用户整型特征（46 列）

- `user_int_feats_{1,3,4,48-59,82,86,92-109}`：标量 `int64`，共 35 列。
- `user_int_feats_{15, 60, 62-66, 80, 89-91}`：数组 `list<int64>`，共 11 列。

### 用户稠密特征（10 列）

- `user_dense_feats_{61-66, 87, 89-91}`：数组 `list<float>`，共 10 列。

### 物品整型特征（14 列）

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`：标量 `int64`，共 13 列。
- `item_int_feats_{11}`：数组 `list<int64>`，共 1 列。

### 域序列特征（45 列）

来自 4 个行为域的 `list<int64>` 序列：

- `domain_a_seq_{38-46}`：9 列
- `domain_b_seq_{67-79, 88}`：14 列
- `domain_c_seq_{27-37, 47}`：12 列
- `domain_d_seq_{17-26}`：10 列

---

## 使用方式

```python
import pyarrow.parquet as pq
import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("demo_1000.parquet")

print(df.shape)       # (1000, 120)
print(df.columns)     # ['user_id', 'item_id', 'label_type', ...]
```

使用 Hugging Face `datasets`：

```python
from datasets import load_dataset

ds = load_dataset("TAAC2026/data_sample_1000")
print(ds)
```
