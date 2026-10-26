# tests/fixtures — checkpoint 测试数据

只生成数据，**不模拟训练过程**：没有训练循环、参数演化或 loss/lr。
产出的数据形态对齐 PyTorch 训练时的 `state_dict`：模型参数 + Adam 的 `exp_avg` / `exp_avg_sq`。

## 用法

```bash
python tests/fixtures/make_data.py --out /tmp/ckpt-data --scale small
python tests/fixtures/make_data.py --out /tmp/ckpt-data --scale tiny --dtype bf16
python tests/fixtures/make_data.py --out /tmp/ckpt-data --world-size 2 --rank 0
python tests/fixtures/make_data.py --out /tmp/ckpt-data --no-optimizer
```

## 输出

```
<out>/
├── manifest.json          数据清单
└── tensors/
    ├── model_embed_tokens_weight.bin
    ├── model_blocks_0_attn_qkv_weight.bin
    └── ...
```

`manifest.json` 字段：

| 字段 | 含义 |
|---|---|
| `format` | 夹具格式版本，当前 `ckpt-fixture/1` |
| `step` | 仅作标记的 step 编号（不影响数据内容） |
| `seed` | 生成种子 |
| `dtype` | 全局 dtype |
| `world_size` / `rank` | 分片配置 |
| `model` | 模型规模：`hidden` / `layers` / `vocab` / `intermediate` |
| `total_bytes` | 全部 tensor 字节数之和 |
| `tensors[]` | 每项含 `name` / `file` / `kind` / `dtype` / `shape` / `nbytes` / `crc32` |

`kind` 为 `model` 或 `optimizer`；`crc32` 是 zlib 的 CRC32（C++ 侧可直接用 zlib 校验）。

## 规模

| `--scale` | hidden / layers / vocab / intermediate | f32 全量约 |
|---|---|---|
| `tiny` | 32 / 1 / 64 / 64 | 0.2 MB |
| `small` | 128 / 2 / 1000 / 512 | 7.8 MB |
| `medium` | 512 / 4 / 8000 / 2048 | 约 480 MB |

也可用 `--hidden --layers --vocab --intermediate` 单项覆盖。

## dtype

`f64` `f32` `f16` `bf16` `i64` `i32` `i16` `i8` `u8`。
`bf16` 没有 numpy 原生类型，按 IEEE 位模式用 `uint16` 承载（截断 float32 高 16 位）。

## 确定性

同一 `(seed, tensor name)` 必然产出相同字节，与机器、运行次数无关。
测试里可以据此做字节级断言：

```python
from make_data import write_dataset
write_dataset("/tmp/a", seed=7, scale="tiny")
write_dataset("/tmp/b", seed=7, scale="tiny")  # 两个目录内容逐字节相同
```

## 分片规则

`--world-size N --rank R` 模拟 DDP 的数据分布：

- **模型参数**：每个 rank 都持有完整副本（与 DDP 一致）
- **optimizer 状态**：按参数名 CRC32 分到某个 rank，各 rank 不重不漏

## 自测

```bash
.venv/bin/python -m pytest tests/fixtures/test_make_data.py -q
```

## 与 C++ 侧的衔接

数据是裸二进制 + JSON，C++ 测试不依赖 Python 运行时：

1. 用生成器产出一份数据到 `build/fixtures/`
2. C++ 读 `manifest.json`（可用任意 JSON 库）与 `tensors/*.bin`
3. 用 `crc32` 校验；`shape` + `dtype` 驱动 `TensorView` 的构造
