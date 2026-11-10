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
│   ├── local.py        本地层写闭环用的后端（native O_DIRECT 写 / 纯 Python 回退）
│   └── object.py       对象存储（boto3：PUT / Range GET / list / delete / copy+delete）
├── commit/           提交协议、版本目录、清理、保留策略
│   └── local.py        .metadata → COMMIT → latest，版本扫描与删除
├── dist/             通信抽象：gather / broadcast / all-reduce，带超时
├── async_/           DeviceBackend / Tensor、D2H 搬运、pinned 池、拷贝 stream、状态机
│                     （后台进程还没做）
├── adapters/         框架相关，把本地张量翻译成全局坐标
│   ├── dtensor.py      FSDP2 / DTensor
│   ├── megatron.py     PP 层号、EP 专家号、融合参数、vocab padding
│   └── stateful.py     优化器状态转 FQN 键、小状态
├── tiers/            分层存储、节点间副本、恢复时选版本（最后做）
├── native/           ckpt_native（C++ 内核）的加载点与对齐 buffer 桥接
├── bindings/         Python 绑定源码：python.cpp（异常映射 / GIL / 零拷贝 / 线程归属）
├── api.py            save_async / load：唯一把以上组件串起来的地方
├── tests/            pytest 与测试数据生成器（tests/fixtures/）、C++ 用例（tests/cpp/）
├── requirements.txt  Python 依赖（numpy / boto3）
├── include/ckpt/     C++ 公共头文件：Status、io_uring 封装、本地文件读写
├── src/              C++ 实现：io/uring.cpp、storage/local/local_file.cpp
└── CMakeLists.txt    C++ 静态库 ckpt_core（liburing）+ Python 扩展 ckpt_native（pybind11）
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
| `copy_into()` / `copy_to_host()` | `async_/d2h.py` | device → host 的唯一实现；前者不分配、直接落进调用方 buffer |
| `D2HStream` / `CopyHandle` | `async_/d2h.py` | 独立拷贝 stream：提交即返回，`wait_stream` / `synchronize` / `max_inflight` |
| `PinnedBufferPool` | `async_/pinned.py` | 预分配 + 复用 + 容量上限；无可用块时阻塞（反压点） |
| `SnapshotManager` | `async_/machine.py` | 一代快照的状态机 + 多代门闸（单/双缓冲） |
| `LocalStorage` | `storage/local.py` | 本地目录后端：native 写（O_DIRECT+fsync+rename）/ 纯 Python 回退 |
| `CommitRecord` | `commit/local.py` | COMMIT 内容：step、metadata 校验和、文件清单 |
| `save_local()` | `api.py` | 本地层写闭环：plan → D2H → 数据 + `.metadata` + COMMIT |

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
`copy_to_host` / `copy_into` 是给搬运算子（保存路径）准备的显式入口。

需要异步（提交即返回、训练继续）时走 `D2HStream`：

```python
from async_ import D2HStream, PinnedBufferPool

pool = PinnedBufferPool(chunk_bytes=8 << 20, min_chunks=1, max_bytes=None)
buffer = pool.acquire(total_bytes)          # 一块 pinned（mock 下是普通内存）

with D2HStream(max_inflight=1) as stream:
    stream.wait_stream(compute_stream)      # 对应 copy_stream.wait_stream(compute_stream)
    handles = []
    offset = 0
    for name, tensor in state.items():
        handles.append(stream.submit_copy(tensor, buffer, offset=offset))
        offset += tensor.nbytes
    compute_stream.wait_event(...)          # 下一次 optimizer step 前再 wait，正常不阻塞
    for h in handles:
        h.wait()                            # CopyHandle 等价于 copy_done event
    stream.synchronize()
buffer.release()                            # 数据写完才归还，见下方状态机
```

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
pool = PinnedBufferPool(chunk_bytes=8 << 20, min_chunks=1)   # 只要 acquire / release / has_free
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
.venv/bin/python -m pytest      # 204 个用例：数据生成器 + 读取接口 + 对象存储 + D2H/stream/pinned 池 + GPU→S3
#              + 状态机 + 区域/索引/读写计划 + C++ 绑定（未构建时自动 skip）
#              + 本地层写闭环（数据 + .metadata + COMMIT 可恢复）
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
- 顺序写默认 O_DIRECT：buffer 地址、offset、length 都要按 `alignment`（默认 4096）对齐；
  offset 与 length 不对齐返回 `kInvalidArgument`（**buffer 地址目前没校验**，靠调用方保证，
  `native.aligned_view()` 就是为此准备的）；补齐责任在 planner，C++ 层不做 read-modify-write
- `WriteFile()` 是整文件便捷写法：临时文件 → 顺序 WRITE → fsync → rename → 目录 fsync，
  最后一段不足一块时自动补零到块边界
- `RenameAtomic()` / `FsyncDir()` 给 commit 层做原子提交（COMMIT 临时文件 + rename + 目录 fsync）
- 短读不静默截断：`ReadMany` 返回 `kIoError`，同时 `bytes` 保留已读长度
- 一个 `io::Ring` 只能给一个线程用（io_uring 的 SQ/CQ 不支持多线程共享）

```bash
apt-get install -y liburing-dev                      # 需要 liburing.h + liburing.so
uv pip install --python .venv/bin/python pybind11     # 构建期依赖，不进 requirements.txt
cmake -S . -B build -G Ninja \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
      -Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)
cmake --build build
ctest --test-dir build --output-on-failure
```

`Python3_EXECUTABLE` 必须显式给：CMake 默认可能找到系统里另一个版本的 python，
编出来的 `.so` ABI 后缀会对不上（导入时报 "compiled with ... but ... is required"）。

## 本地层写闭环

`api.save_local()` 把「计划 → D2H → 落盘 → 提交」串成一次同步保存，落点是本地目录；
远端（S3）上传留给后续的后台 worker，不在训练路径上。

```
build_save_plan ──> trigger(拿对齐 pinned 块) ──> 按 item.byte_offset D2H
   ──> write_view(数据文件: 临时→fsync→rename→目录 fsync) ──> .metadata
   ──> persist_ok() ──> COMMIT(临时→rename) + latest ──> commit_ok()
```

产物就是设计文档 §5 的版本目录：

```
ckpt/
  step_42/
    __0_0.distcp      # 数据：planner 布局，按 4096 补齐到块边界
    .metadata         # 索引：分片全局坐标 + 字节区间 + crc32
    COMMIT            # 提交标记：索引校验和 + 文件清单
  latest              # 指向 42
```

- **只有带 `COMMIT` 的目录算可恢复**：`committed_steps()` / `latest_step()` 只认它
- 写失败或中途异常 → 状态机 `fail()`（未写文件则 `DROPPED`，有残缺目录则 `ABORTED`），
  `latest` 不更新，这一版对恢复端不可见
- 数据文件与 `.metadata` 都走「临时文件 → fsync → rename → 目录 fsync」，崩溃后只会留下
  一个完整的旧版本
- 两块写入引擎：`engine="native"`（`ckpt_native`，O_DIRECT + io_uring）与 `"python"`
  （功能等价、不绕 page cache）；`"auto"` 按扩展是否可用来选
- pinned 块由 C++ 的 `AlignedBuffer` 分配（`aligned_allocator()`），D2H 落点天然满足
  O_DIRECT 的地址对齐，写盘全程零拷贝

```python
import api as ckpt

data = ckpt.load("/data/step_0", device="cuda:0")
report = ckpt.save_local(
    data.tensors, "/ckpt", step=42,
    blobs={"rng/rank_0": rng_bytes, "lr_scheduler": lr_bytes},
    layout={"dp": 1, "tp": 1},
)
report.state        # SnapshotState.COMMITTED
report.engine       # "native" / "python"
report.file_size    # 数据文件字节数（含尾块补齐）
```

`save_async()` 仍是 `NotImplementedError`：后台进程与上传队列还没做。

## Python 绑定（`ckpt_native`）

`bindings/python.cpp` 把 `ckpt_core` 包成扩展模块，产物落在 `build/python/`，
由 `native/__init__.py` 负责挂 `sys.path` 并给出可读的缺失提示。绑定层只做四件事：

| 职责 | 做法 |
|---|---|
| 异常映射 | `Status` + `ErrorCode` → 项目异常（`StorageError` / `ObjectNotFoundError` / `FormatError` / `ValueError` / `NotImplementedError`），上层见不到 `ErrorCode` |
| GIL 释放 | `WriteFile` / `WriteAt` / `Sync` / `rename_atomic` / `fsync_dir` 全程 `gil_scoped_release`，后台写不与训练线程抢 GIL |
| 零拷贝 | 通过 Python buffer 协议直接取指针，`write_at` 期间用 keepalive 保住调用方的 buffer；`AlignedView` 让 C++ 分配的页对齐内存变成可写 `memoryview` |
| 线程归属 | `LocalFile` 记住创建它的线程，跨线程调用直接 `RuntimeError`（io_uring 的 SQ/CQ 不支持共享） |

```python
import native
from native import aligned_view

block = aligned_view(1 << 20, 4096)          # 页对齐，可以直接 O_DIRECT 写
tensor.copy_to(block.view)                    # D2H 落到对齐块上（零拷贝）
native.write_file("step_12000/__3_0.distcp", [block.view[: tensor.nbytes]],
                  direct_io=True)             # 临时文件 → fsync → rename → 目录 fsync
block.release()

native.rename_atomic("COMMIT.tmp", "COMMIT")  # 提交标记
native.fsync_dir("step_12000")
```

`tests/test_native.py` 覆盖：写读回、多段写、末段自动补零、`LocalFile` 增量写、
原子 rename、错误映射、池块 → D2H → 落盘的闭环，以及两条绑定层必须做对的事
（跨线程用 `LocalFile` 报错；写入期间另一个 Python 线程能推进 —— 证明 GIL 真的放掉了）。
扩展没构建时这些用例整体 skip，不影响纯 Python 链路。
