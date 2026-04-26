---

## 🛠️ 开发与集成速查手册

### 1. 运行环境规格 (Environment Specs)
在编写代码（尤其是设置 Batch Size 或模型大小）时，请参考以下限制：
* **操作系统**: Ubuntu 22.04
* **Python 版本**: 3.10.20 (Conda 26.1.1)
* **计算资源**: 单卡 **20% 算力**，**19GiB 显存**
* **内存/CPU**: 55GiB 内存，9 核 CPU
* **驱动环境**: CUDA 12.6, cuDNN 9.5.1

---

### 2. 训练阶段规范 (Training Phase)

#### **文件要求**
* **入口文件**: 必须包含一个 `run.sh` 脚本。

#### **环境变量**
| 变量名 | 用途 |
| :--- | :--- |
| `TRAIN_DATA_PATH` | 训练数据集所在目录 |
| `TRAIN_CKPT_PATH` | **必须**将模型权重保存至此目录 |
| `TRAIN_TF_EVENTS_PATH` | TensorBoard 日志保存目录 |
| `USER_CACHE_PATH` | 20GB 缓存空间，数据可在训练与评估间共享 |

#### **输出路径命名 (严格执行)**
保存模型检查点时，子文件夹命名必须遵循：
* **格式**: `global_step[步数].[自定义参数]`
* **示例**: `global_step1000.lr=0.001.batch=32`
* **限制**: 长度 < 300 字符；仅限字母、数字、`_`、`-`、`=`、`.`。

---

### 3. 评估阶段规范 (Evaluation Phase)

#### **文件要求**
1.  **`prepare.sh`** (可选): 推理前自动运行，用于 `pip/conda install` 缺失的依赖。
2.  **`infer.py`** (**强制**): 必须包含 `def main():` 函数且**不带参数**。

#### **环境变量**
| 变量名 | 用途 |
| :--- | :--- |
| `EVAL_DATA_PATH` | 推理用的测试集目录 |
| `MODEL_OUTPUT_PATH` | 读取训练产出的模型权重目录 |
| `EVAL_RESULT_PATH` | **必须**将 `predictions.json` 保存至此 |

#### **输出格式 (`predictions.json`)**
必须严格遵守以下 JSON 结构，确保 `user_id` 与测试集完全对应：
```json
{
    "predictions": {
        "user_001": 0.8732,
        "user_002": 0.1245
    }
}
```

---

### 4. 代码实现模板 (Python)

#### **读取数据与保存模型 (训练阶段)**
```python
import os
import torch

# 获取路径
data_dir = os.environ.get('TRAIN_DATA_PATH')
ckpt_dir = os.environ.get('TRAIN_CKPT_PATH')

# 保存示例
step = 100
save_path = os.path.join(ckpt_dir, f"global_step{step}")
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
```

#### **推理与生成结果 (评估阶段)**
```python
import os
import json

def main():
    # 获取环境变量
    test_data = os.environ.get('EVAL_DATA_PATH')
    result_path = os.environ.get('EVAL_RESULT_PATH')
    
    # ... 推理逻辑 ...
    results = {"predictions": {"user_id_1": 0.95, "user_id_2": 0.02}}
    
    # 强制要求保存为 predictions.json
    with open(os.path.join(result_path, "predictions.json"), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
```

---

### 5. 任务状态查阅表
* **Pending**: 排队中。
* **Inference Running**: 正在跑你的 `infer.py`。
* **Success**: 跑完了，可以看分数了。
* **Failed**: 报错了，立即去 **Logs** 检查是否是 `predictions.json` 格式或文件命名问题。

---