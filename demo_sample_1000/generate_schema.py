#!/usr/bin/env python3
"""从parquet数据生成schema.json文件"""

import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# 读取数据统计vocab size和维度
df = pd.read_parquet('demo_1000.parquet')
pf = pq.ParquetFile('demo_1000.parquet')
schema_types = {field.name: str(field.type) for field in pf.schema_arrow}

def get_vocab_size(series):
    """计算列的vocab size（最大值+1）"""
    if series.dtype == 'object':  # list/ndarray类型
        # 展开所有列表统计最大值
        all_vals = []
        for x in series.dropna():
            arr = np.asarray(x)
            valid_vals = arr[arr > 0]
            if len(valid_vals) > 0:
                all_vals.extend(valid_vals.tolist())
        return max(all_vals) + 1 if all_vals else 100000
    else:  # 标量int类型
        vals = series[series > 0]
        return int(vals.max()) + 1 if len(vals) > 0 else 100000

def get_max_dim(series):
    """获取list类型的最大长度"""
    max_len = 0
    for x in series.dropna():
        arr = np.asarray(x)
        max_len = max(max_len, len(arr))
    return max_len if max_len > 0 else 1

def is_list_type(col_name):
    """检查列是否为list类型"""
    col_type = schema_types.get(col_name, '')
    return 'list' in col_type.lower()

# 按照README的分类构建schema

# user_int_feats: 标量35列，数组11列
user_int_scalars = [1, 3, 4, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    82, 86, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                    104, 105, 106, 107, 108, 109]
user_int_arrays = [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]

user_int = []
# 标量特征 (dim=1)
for fid in sorted(user_int_scalars):
    col = f'user_int_feats_{fid}'
    if col in df.columns:
        vs = get_vocab_size(df[col])
        user_int.append([fid, vs, 1])

# 数组特征 (dim>1) - 根据实际数据类型确定
for fid in sorted(user_int_arrays):
    col = f'user_int_feats_{fid}'
    if col in df.columns:
        vs = get_vocab_size(df[col])
        if is_list_type(col):
            dim = get_max_dim(df[col])
        else:
            dim = 1
        user_int.append([fid, vs, dim])

# item_int_feats: 标量13列，数组1列
item_int_scalars = [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
item_int_arrays = [11]

item_int = []
for fid in sorted(item_int_scalars):
    col = f'item_int_feats_{fid}'
    if col in df.columns:
        vs = get_vocab_size(df[col])
        item_int.append([fid, vs, 1])

for fid in sorted(item_int_arrays):
    col = f'item_int_feats_{fid}'
    if col in df.columns:
        vs = get_vocab_size(df[col])
        if is_list_type(col):
            dim = get_max_dim(df[col])
        else:
            dim = 1
        item_int.append([fid, vs, dim])

# user_dense_feats: 10列
user_dense_fids = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]
user_dense = []
for fid in user_dense_fids:
    col = f'user_dense_feats_{fid}'
    if col in df.columns:
        dim = get_max_dim(df[col])
        user_dense.append([fid, dim])

# 序列特征 - 按domain分组
seq = {}

# domain_a: prefix=domain_a_seq, fids=38-46 (9列)
domain_a_fids = list(range(38, 47))
seq['seq_a'] = {
    'prefix': 'domain_a_seq',
    'ts_fid': None,
    'features': [[fid, get_vocab_size(df[f'domain_a_seq_{fid}'])] for fid in domain_a_fids
                 if f'domain_a_seq_{fid}' in df.columns]
}

# domain_b: prefix=domain_b_seq, fids=67-79, 88 (14列)
domain_b_fids = list(range(67, 80)) + [88]
seq['seq_b'] = {
    'prefix': 'domain_b_seq',
    'ts_fid': None,
    'features': [[fid, get_vocab_size(df[f'domain_b_seq_{fid}'])] for fid in domain_b_fids
                 if f'domain_b_seq_{fid}' in df.columns]
}

# domain_c: prefix=domain_c_seq, fids=27-37, 47 (12列)
domain_c_fids = list(range(27, 38)) + [47]
seq['seq_c'] = {
    'prefix': 'domain_c_seq',
    'ts_fid': None,
    'features': [[fid, get_vocab_size(df[f'domain_c_seq_{fid}'])] for fid in domain_c_fids
                 if f'domain_c_seq_{fid}' in df.columns]
}

# domain_d: prefix=domain_d_seq, fids=17-26 (10列)
domain_d_fids = list(range(17, 27))
seq['seq_d'] = {
    'prefix': 'domain_d_seq',
    'ts_fid': None,
    'features': [[fid, get_vocab_size(df[f'domain_d_seq_{fid}'])] for fid in domain_d_fids
                 if f'domain_d_seq_{fid}' in df.columns]
}

schema = {
    'user_int': user_int,
    'item_int': item_int,
    'user_dense': user_dense,
    'seq': seq
}

# 写入schema.json
with open('schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print("✓ schema.json 生成成功！")
print(f"  - user_int: {len(user_int)} features")
print(f"  - item_int: {len(item_int)} features")
print(f"  - user_dense: {len(user_dense)} features")
print(f"  - seq domains: {list(seq.keys())}")
