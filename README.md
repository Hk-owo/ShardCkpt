# ckpt — 模型训练 Checkpoint 系统

Python 侧按组件分层（`core` → `planner` → `storage` / `commit` / `dist` / `async_` / `adapters` / `tiers`，
由 `api.py` 串联）。S3 兼容存储（SeaweedFS / MinIO）的接入走 boto3（`storage/object.py`）。

## 目录结构

```
ShardCkpt/
├── core/             纯数据结构，零依赖
│   ├── metadata.py     分片元数据、索引序列化：Device、dtype 映射
│   ├── region.py       N 维区域、求交、一维区间（展平切分）
│   └── __init__.py     错误类型（CheckpointError 及其子类）
├── planner/          纯函数，不碰 IO 和通信
│   ├── save_plan.py    去重、写者分配、覆盖校验
│   └── load_plan.py    求交 → 读取计划，合并相邻读请求
├── storage/          存储后端，统一接口
│   ├── base.py         write / read / read_range / rename_or_put / list / delete
│   ├── fs.py           本地盘、并行文件系统（manifest 解析、按张量读取）
│   └── object.py       对象存储（boto3：PUT / Range GET / list / delete / copy+delete）
├── commit/           提交协议、版本目录、清理、保留策略
├── dist/             通信抽象：gather / broadcast / all-reduce，带超时
├── async_/           pinned buffer 池、D2H stream、后台进程、状态机（DeviceBackend / Tensor）
├── adapters/         框架相关，把本地张量翻译成全局坐标
│   ├── dtensor.py      FSDP2 / DTensor
│   ├── megatron.py     PP 层号、EP 专家号、融合参数、vocab padding
│   └── stateful.py     优化器状态转 FQN 键、小状态
├── tiers/            分层存储、节点间副本、恢复时选版本（最后做）
├── api.py            save_async / load：唯一把以上组件串起来的地方
├── tests/            pytest 与测试数据生成器（tests/fixtures/）、C++ 用例（tests/cpp/）
├── requirements.txt  Python 依赖（numpy / boto3）
├── include/ckpt/     C++ 公共头文件：Status、io_uring 封装、本地文件读写
├── src/              C++ 实现：io/uring.cpp、storage/local/local_file.cpp
└── CMakeLists.txt    C++ 静态库 ckpt_core（依赖 liburing）
```

## 读取接口

`api.py` 的接口按「从 GPU 读取」设计；当前没有 GPU，cuda 分支由 mock backend
在 host 内存上承载，读取路径与真机完全一致。

```python
import api

data = api.load("/tmp/ckpt-data", device="cuda:0")   # 对应 torch.load(map_location=...)
t = data["model.embed_tokens.weight"]
t.device        # cuda:0
t.numpy()       # 经 D2H 拷贝取 host 副本

data.to("cpu")  # 批量迁回
```

数据路径固定为 `文件 -> staging buffer -> H2D -> device buffer`，每个张量一次拷贝，
`backend.stats` 可以断言拷贝次数与字节数。

| 组件 | 位置 | 作用 |
|---|---|---|
| `Device` | `core/metadata.py` | `"cpu"` / `"cuda:N"`，写法与 `torch.device` 一致 |
| `DeviceBackend` | `async_/` | `allocate` / `copy_h2d` / `copy_d2h` / `synchronize` |
| `CpuBackend` / `MockCudaBackend` | `async_/` | 默认注册的两个后端，后者 `is_mock == True` |
| `register_backend()` | `async_/` | 接入真卡时用真实 `CudaBackend` 覆盖 `"cuda"`，调用方与测试不用改 |
| `CheckpointReader` | `storage/fs.py` | 读 manifest、校验 CRC32、逐张量 H2D |

## 对象存储后端（S3）

`storage/object.py` 用 boto3 接 S3 兼容存储（SeaweedFS / MinIO / AWS S3），
是 `storage/base.py` 接口的一个实现，调用语义与 `storage/fs.py` 一致：

```python
from storage.object import S3Config, S3Storage

s3 = S3Storage(S3Config(
    bucket="ckpt",
    prefix="runs/step_0",
    endpoint_url="http://127.0.0.1:8333",   # SeaweedFS / MinIO；AWS 留空
    access_key_id="...",                    # 留空则走 boto3 默认凭据链
    secret_access_key="...",
))
s3.write("tensors/model_embed.bin", payload)       # PUT 返回即可读
s3.read("tensors/model_embed.bin")
s3.read_range("tensors/model_embed.bin", 0, 4096)  # Range GET，按张量读取
s3.list("tensors")                                 # 相对 prefix 的 key
s3.rename_or_put("tensors/tmp.bin", "tensors/final.bin")
s3.delete("tensors/final.bin")                     # 幂等
```

- key 一律相对 `prefix`，与 fs 后端保持一致；绝对路径、`..`、空段直接 `ValueError`
- 对象不存在抛 `ObjectNotFoundError`，其余失败抛 `StorageError`，两者都继承 `CheckpointError`
- 超过 `multipart_threshold`（默认 64 MiB）自动分片：create → upload_part × N → complete，
  任一步失败都会 abort 掉本次上传；`part_size` 默认 8 MiB（S3 要求除末片外 ≥ 5 MiB）
- `rename_or_put` 在源对象超过 5 GiB 时自动改用 `UploadPartCopy`，分片大小按 10000 片上限放大；
  复制失败会 abort 并保留源对象
- `tests/s3_stub.py` 是进程内的最小 S3 兼容服务，跑测试不需要外部 S3

```bash
uv pip install --python .venv/bin/python -r requirements.txt   # numpy + boto3
.venv/bin/python -m pytest      # 54 个用例：数据生成器 + 读取接口 + 对象存储
```

## 本地读写（C++ / io_uring）

`include/ckpt/io/uring.hpp` 与 `include/ckpt/storage/local_file.hpp` 是 C++ 侧的本地
读写骨架，数据面全部走 io_uring：调用线程自己填 SQE、自己收割 CQE，随机读的线程数
恒等于调用线程数，与请求数量无关（`tests/cpp/local_file_test.cpp` 用 `/proc/self/task`
把这一点变成了断言）。

```cpp
#include "ckpt/io/uring.hpp"
#include "ckpt/storage/local_file.hpp"

using namespace ckpt;

io::Ring ring;
io::Ring::Create(128, &ring);

// 跨文件批量随机读：一次提交、一次等待，收完所有 CQE 才返回，结果与 ops 一一对应
std::vector<storage::ReadResult> results;
std::vector<storage::ReadOp> ops{
    {fd_a, 3 * 4096, 4096, dst + 0 * 4096},
    {fd_b, 7 * 4096, 4096, dst + 1 * 4096},
};
storage::ReadMany(&ring, ops, &results);

// 大文件顺序写：O_DIRECT 大块 + fsync，内部攒批，同样不建线程
storage::LocalFileOptions options;
options.direct_io = true;
storage::LocalFile* file = nullptr;
storage::LocalFile::OpenWrite("__0_0.distcp", options, &file);
file->WriteAt(0, {chunk0.data(), chunk0.size()});  // offset 必须递增
file->Sync();                                      // 等完所有写 + fsync
file->Close();
```

- 随机读默认 buffered：读区间是张量在文件里的偏移，通常不按块对齐，O_DIRECT 会放大读
  并失去 page cache；只有大块顺序读才建议 `direct_io = true`
- 顺序写默认 O_DIRECT：buffer 地址、offset、length 都要按 `alignment`（默认 4096）对齐，
  不对齐返回 `kInvalidArgument`；补齐责任在 planner，C++ 层不做 read-modify-write
- `WriteFile()` 是整文件便捷写法：临时文件 → 顺序 WRITE → fsync → rename → 目录 fsync，
  最后一段不足一块时自动补零到块边界
- `RenameAtomic()` / `FsyncDir()` 给 commit 层做原子提交（COMMIT 临时文件 + rename + 目录 fsync）
- 短读不静默截断：`ReadMany` 返回 `kIoError`，同时 `bytes` 保留已读长度
- 一个 `io::Ring` 只能给一个线程用（io_uring 的 SQ/CQ 不支持多线程共享）

```bash
apt-get install -y liburing-dev                      # 需要 liburing.h + liburing.so
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```
