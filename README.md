# ckpt — 模型训练 Checkpoint 系统

C++ 核心 + Python 绑定，存储后端 S3 兼容（SeaweedFS / MinIO）。

## 目录结构

```
checkpoint/
├── include/ckpt/             C++ 公共头文件（对外接口）
│   ├── storage/              Storage 抽象：local / s3 实现
│   ├── s3/                   SigV4 签名、凭据、S3 客户端
│   ├── serialize/            格式读写（manifest、shard）
│   └── util/                 byte buffer、hash、线程池、日志
├── src/                      与 include/ 镜像的实现目录
│   ├── core/                 checkpoint 编排、metadata、manifest、state_dict
│   ├── storage/              后端实现
│   │   └── s3/               S3 客户端、SigV4、HTTP 传输
│   ├── serialize/
│   └── util/
├── python/
│   ├── bindings/             pybind11 绑定（生成 _core 扩展）
│   ├── src/ckpt/             Python 包：读取接口、torch state_dict 对接
│   └── tests/                pytest
├── tests/                    C++ 单测
│   └── fixtures/             测试数据生成器（numpy，只造数据）
├── bench/                    读写吞吐基准
├── examples/                 C++ / Python 最小示例
├── scripts/                  依赖引导、格式化、本地 S3 起停
├── docs/                     设计文档、存储布局
└── cmake/                    CMake 辅助模块与包配置模板
```

## 读取接口

`python/src/ckpt` 的接口按「从 GPU 读取」设计；当前没有 GPU，cuda 分支由 mock backend
在 host 内存上承载，读取路径与真机完全一致。

```python
import ckpt

data = ckpt.load("/tmp/ckpt-data", device="cuda:0")   # 对应 torch.load(map_location=...)
t = data["model.embed_tokens.weight"]
t.device        # cuda:0
t.numpy()       # 经 D2H 拷贝取 host 副本

data.to("cpu")  # 批量迁回
```

数据路径固定为 `文件 -> staging buffer -> H2D -> device buffer`，每个张量一次拷贝，
`backend.stats` 可以断言拷贝次数与字节数。

| 组件 | 作用 |
|---|---|
| `Device` | `"cpu"` / `"cuda:N"`，写法与 `torch.device` 一致 |
| `DeviceBackend` | `allocate` / `copy_h2d` / `copy_d2h` / `synchronize` |
| `CpuBackend` / `MockCudaBackend` | 默认注册的两个后端，后者 `is_mock == True` |
| `register_backend()` | 接入真卡时用真实 `CudaBackend` 覆盖 `"cuda"`，调用方与测试不用改 |

```bash
.venv/bin/python -m pytest      # 26 个用例：数据生成器 + 读取接口
```
