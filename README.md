# ckpt — 模型训练 Checkpoint 系统

Python 侧按组件分层（`core` → `planner` → `storage` / `commit` / `dist` / `async_` / `adapters` / `tiers`，
由 `api.py` 串联）。S3 兼容存储（SeaweedFS / MinIO）的接入走 boto3（`storage/object.py`）。

## 目录结构

```
ShardCkpt/
├── core/             纯数据结构，零依赖
│   ├── metadata.py     设备描述、dtype 映射
│   ├── region.py       N 维区域、求交、一维区间（展平切分）
│   ├── index.py        分片元数据与索引（.metadata）：序列化 + 覆盖校验
│   └── __init__.py     错误类型（CheckpointError 及其子类）
├── planner/          纯函数，不碰 IO 和通信
│   ├── save_plan.py    去重、写者分配、对齐、覆盖校验
│   └── load_plan.py    求交 → 读取计划，合并相邻读请求
├── storage/          存储后端，统一接口
│   ├── base.py         write / read / read_range / rename_or_put / list / delete
│   ├── fs.py           本地盘、并行文件系统（manifest 解析、按张量读取）
│   └── object.py       对象存储（boto3：PUT / Range GET / list / delete / copy+delete）
├── commit/           提交协议、版本目录、清理、保留策略
├── dist/             通信抽象：gather / broadcast / all-reduce，带超时
├── async_/           DeviceBackend / Tensor、D2H 搬运、快照状态机
│                     （pinned buffer 池、D2H stream、后台进程还没做）
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
| `copy_to_host()` | `async_/d2h.py` | device → host 的唯一实现，一次整块 D2H |
| `SnapshotManager` | `async_/machine.py` | 一代快照的状态机 + 多代门闸（单/双缓冲） |

## D2H 搬运

`async_/d2h.py` 是全项目唯一的 device → host 拷贝实现，`Tensor.tobytes()` /
`numpy()` / `to("cpu")` 都走它。搬运固定为「host 副本 → `backend.copy_d2h` →
`synchronize`」，所以 `backend.stats.device_to_host_*` 精确反映搬运次数与字节，
换成真机 `CudaBackend` 后调用方与测试都不用改。

```python
import api as ckpt
from async_ import copy_to_host, get_backend

backend = get_backend(ckpt.Device.parse("cuda:0"))
buf = backend.allocate(len(payload))                 # device buffer
backend.copy_h2d(buf, memoryview(payload))
host = copy_to_host(buf, device="cuda:0", backend=backend)   # 与 tensor.tobytes() 同一条路径
```

普通调用方拿不到裸 device buffer，直接用 `tensor.tobytes()` / `tensor.numpy()` 即可；
`copy_to_host` 是给搬运算子（保存路径、未来的 pinned 快照）准备的显式入口。

`tests/test_gpu_to_s3.py` 端到端验证这条链路：`load(device="cuda:0")` → 逐张量 D2H →
PUT 到 S3 兼容存储 → 读回与源文件逐字节、CRC32 一致（大张量另走 multipart 分支）。

## 快照状态机

`async_/machine.py` 把设计文档 §4 的状态机落成可测的代码：一个 step 一代
（`Snapshot`），状态只能沿合法边推进，非法转移直接抛 `SnapshotError`。

```
SNAPSHOTTING ──d2h_done()──> PERSISTING ──persist_ok()──> COMMITTING ──commit_ok()──> COMMITTED
      │                          │                           │
      └─fail()→ DROPPED          └───────fail()→ ABORTED <───┘
```

文档没写、但实现必须做对的三件事：

- **`SNAPSHOTTING` 失败回 `DROPPED`，不是 `ABORTED`**：此时存储侧一个字节都没写，
  没有目录需要清理；进 `ABORTED` 会让清理逻辑去删不存在的目录。
- **buffer 的归还点是「数据写完/放弃」，不是「D2H 完成」**：`COMMITTED` / `ABORTED` /
  `DROPPED` 三个终态各自归还一次，且只归还一次。
- **门闸看「上一代是否还占着 buffer」**：`max_inflight=1` 是单缓冲，`2` 即双缓冲。
  越线时先 `SKIP`（训练不等），连续跳过 `force_after_skips` 次之后才 `FORCE_WAIT`
  （阻塞训练，对应文档「距上次成功超过阈值」）。

```python
pool = ...                # 任意实现 acquire / release / has_free 的池；pinned 池还没做
manager = SnapshotManager(pool, max_inflight=1, force_after_skips=3)

trigger = manager.trigger(step=12000, nbytes=total)
if trigger.decision is TriggerDecision.START:
    snap = trigger.snapshot                  # 提交 D2H，训练继续
    snap.d2h_done(); snap.persist_ok(); snap.commit_ok()
elif trigger.decision is TriggerDecision.SKIP:
    ...                                      # 本次不保存，训练不等
```

状态机不认识真实内存（buffer 是不透明句柄），所以整套转移在 mock 环境下可测；
pinned 池、独立 stream、后台进程还没实现，那三样才需要真卡。

## 读写计划（planner）

`planner/` 是纯函数层：输入输出都是数据，不碰 IO、不碰 GPU、不做集合通信。
保存侧决定「本 rank 写哪些分片、写到文件的哪个字节」，恢复侧决定「从旧分片的
哪个字节区间读多少、拷到本地张量的哪个位置」。索引结构（文档 §3 的 `.metadata`）
在 `core/index.py`。

```python
from core.index import CheckpointIndex
from core.region import Interval, Region
from planner import ShardSource, TargetTensor, build_load_plan, build_save_plan

# 保存：ZeRO-1 的展平区间 [0, numel/2) 记成 flat，不做机内通信
plan = build_save_plan(
    [ShardSource(fqn="optim.model.w.exp_avg", global_shape=(4096, 4096), dtype="f32",
                 flat=Interval(0, 4096 * 2048))],
    rank=3, blobs={"rng/rank_3": 64},
)
plan.items[0].file          # "__3_0.distcp"
plan.items[0].byte_offset   # 0
plan.items[0].length        # 32 MiB
plan.partial_index()        # 本 rank 的分片元数据；gather 到 rank 0 后合并成 .metadata

# 恢复：新布局 TP=4 的 rank 3 需要 W 的列 [3072, 4096)
index = CheckpointIndex.loads(metadata_text)
load = build_load_plan(
    index,
    [TargetTensor.for_shard("model.layers.5.attn.q_proj.weight", (4096, 4096), "f32",
                            dim=1, index=3, count=4)],
)
for window in load.windows(64 << 20):   # 按 64 MiB 切 staging 窗口
    ...
```

- **去重按拓扑指定写者**：Replicate 维度上的 N 份副本内容相同，`hash(FQN) % N`
  决定哪一份落盘（`default_writer`），其余进 `skipped`。不做内容比对，也不会把
  写入量全压给 DP rank 0。
- **对齐责任在 planner**：顺序写默认 O_DIRECT，每项之后补齐到 4096（`align`），
  padding 记在 `WriteItem.padding` 里，最后一项不补。
- **展平切分（方案 A）**：ZeRO / FSDP1 / Megatron distributed optimizer 的 buffer
  区间在索引里记成 `flat = [start, length]`，`flatten_slice()` 负责把它拆回 N 维
  矩形块（段数 ≤ 2 × rank），保存时零通信。
- **连续性按分片自己的布局判断**：矩形分片在文件里按本地 shape 紧凑排列，所以
  整读一个 TP 列分片是一次 `read_range` —— 尽管它在全局行主序下是碎片。
- **列子集两种读法**：`column_strategy="whole"` 整读覆盖该行的整行块再在内存里切
  （TP 8→16 时读放大 2 倍、一次 I/O，文档 §3 的结论）；`"strided"` 每行一次
  `read_range`，读放大 1 但碎片 I/O 多。
- **布局不变时自然退化**：TP 8→8 时每个张量只有一个 `ReadItem`、`length` 等于旧
  分片全长，没有单独的快路径代码 —— 用语例锁住这个性质。
- **覆盖校验分两层**：`TensorMeta.validate(complete=False)` 给单 rank 自检（只看得到
  自己那片），`CheckpointIndex.validate()` 在 rank 0 汇总后做全局覆盖 + 同一文件内
  字节区间不重叠。

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
.venv/bin/python -m pytest      # 157 个用例：数据生成器 + 读取接口 + 对象存储 + D2H + GPU→S3 + 状态机 + 区域/索引/读写计划
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
