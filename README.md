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
│   ├── load_plan.py    求交 → 读取计划，合并相邻读请求
│   └── targets.py      目标布局（tp / shard_dims）→ 本 rank 要读的区域
├── storage/          存储后端，统一接口
│   ├── base.py         write / read / download / read_range / rename_or_put / list / delete
│   ├── data.py         加载结果容器（api.load 的返回类型）
│   ├── local.py        本地层后端（native O_DIRECT 写 / preadv 读 / 纯 Python 回退）
│   ├── checksum.py     CRC-32C（默认 1 MiB 分块，可配置）+ google-crc32c / 纯 Python 回退
│   ├── crc_io.py       逻辑读块 → 可校验 IO 子区间
│   ├── local_load.py   本地层读闭环：窗口化 staging + 批量读 + 块校验 + H2D
│   ├── recover.py      COMMIT 门禁：选版本、校验 .metadata、读出索引
│   ├── object.py       对象存储（boto3：PUT / 流式下载 / Range GET / list / delete）
│   └── upload.py       上传闭环 + 对象存储恢复（整文件下载 → 本地读闭环）
├── commit/           提交协议、版本目录、垃圾回收
│   ├── local.py        .metadata → COMMIT，版本扫描与删除
│   └── gc.py           回收判据（纯函数）+ 执行器 + 周期线程
├── dist/             通信抽象：gather / broadcast / all-reduce，带超时（**未实现**，只有 docstring）
├── async_/           DeviceBackend / Tensor、D2H 搬运、pinned 池、拷贝 stream、状态机
│                     （后台进程还没做，`save_async` 仍是 NotImplementedError）
├── adapters/         框架相关，把本地张量翻译成全局坐标
│   ├── dtensor.py      FSDP2 / DTensor：placements（Shard/Replicate）→ ShardSource
│   ├── megatron.py     PP 层号、EP 专家号、融合参数、vocab padding（**未实现**）
│   └── stateful.py     优化器状态转 FQN 键、小状态（**未实现**）
├── tiers/            分层存储、节点间副本、恢复时选版本（**未实现**，只有 docstring）
├── examples/         e2e_sim.py：端到端模拟（伪训练 → 保存 → 崩溃 → 上传 → 恢复 → 回收）
├── native/           ckpt_native（C++ 内核）的加载点与对齐 buffer 桥接
├── bindings/         Python 绑定源码：python.cpp（异常映射 / GIL / 零拷贝 / 线程归属）
├── api.py            save_local / load_local / load：唯一把以上组件串起来的地方
├── tests/            pytest、测试数据生成器（tests/fixtures/）、仿真测试（tests/sim/）、C++ 用例（tests/cpp/）
├── requirements.txt  Python 依赖（numpy / boto3 / google-crc32c）
├── include/ckpt/     C++ 公共头文件：Status、io_uring 封装、本地文件读写
├── src/              C++ 实现：io/uring.cpp、storage/local/local_file.cpp
└── CMakeLists.txt    C++ 静态库 ckpt_core（liburing）+ Python 扩展 ckpt_native（pybind11）
```

## 端到端模拟（先跑这个）

一条命令跑完「伪训练 → 分片保存 → 崩溃 → 上传 → 从对象存储恢复 → 续训 → 回收」，
**不需要 GPU，也不需要外部对象存储**：显存由 `MockCudaBackend` 用 host 内存承载，
S3 用进程内的 `tests/s3_stub.py`。读取路径、提交点语义、Range GET、H2D 都是正式实现。

```bash
.venv/bin/python examples/e2e_sim.py              # 2 个 rank（TP=2），约 5 秒
.venv/bin/python examples/e2e_sim.py --ranks 4    # 换 world size
.venv/bin/python examples/e2e_sim.py --ranks 1 --keep   # 单 rank，保留工作目录供查看
.venv/bin/python examples/e2e_sim.py --endpoint-url http://127.0.0.1:8333 \
    --bucket ckpt --prefix runs/e2e --access-key ... --secret-key ...   # 真 SeaweedFS / MinIO
```

七个阶段，每一步都断言，任何一条不成立就非零退出：

| 阶段 | 演示什么 |
|---|---|
| 1 | 伪训练 3/6/9 步，每 rank 存自己那片（TP 列切分）→ 独立上传 → rank 0 只 `list` 写版本 `COMMIT` |
| 2 | 本地门禁：只有带 `COMMIT` 的版本可恢复；单 rank 目录单独读会被全局覆盖校验拒掉 |
| 3 | 崩溃注入：step 12 的某个 rank 在写 `COMMIT` 前被 `SIGKILL`（数据齐、提交点缺失） |
| 4 | 提交点语义：`finalize_upload(12)` 失败、`load_object(12)` 报错，最新仍是 9 |
| 5 | 从对象存储恢复：整读一份 + 每 rank 取自己那片拼回全局，两者逐字节一致 |
| 6 | 续训：从 step 9 恢复再跑 3 步 == 一口气训到 step 12（无容差） |
| 7 | 回收：已上传的本地版本、被更新版本越过的残片都删掉；`dry_run` 出计划、重跑幂等 |

脚本本身就是「外部调用方怎么用这套 API」的示例：`to_device()` 演示 host 数组 →
device 张量，`sources_for_rank()` 演示分片全局坐标，`save_rank()` 是一个 rank 的完整动作。
`tests/test_examples_e2e.py` 跑这个脚本（1 / 2 个 rank 各一遍），防止 demo 腐化。

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

data = api.load("/ckpt", device="cuda:0")           # 本地版本目录自动走 load_local
```

数据路径固定为 `文件 -> staging buffer -> H2D -> device buffer`，每个张量一次拷贝，
`backend.stats` 可以断言拷贝次数与字节数。

| 组件 | 位置 | 作用 |
|---|---|---|
| `Device` | `core/metadata.py` | `"cpu"` / `"cuda:N"`，写法与 `torch.device` 一致 |
| `DeviceBackend` | `async_/` | `allocate` / `copy_h2d` / `copy_d2h` / `synchronize` |
| `CpuBackend` / `MockCudaBackend` | `async_/` | 默认注册的两个后端，后者 `is_mock == True` |
| `register_backend()` | `async_/` | 接入真卡时用真实 `CudaBackend` 覆盖 `"cuda"`，调用方与测试不用改 |
| `CheckpointData` | `storage/data.py` | 加载结果容器：按名字取张量、迁移设备 |
| `copy_into()` / `copy_to_host()` | `async_/d2h.py` | device → host 的唯一实现；前者不分配、直接落进调用方 buffer |
| `D2HStream` / `CopyHandle` | `async_/d2h.py` | 独立拷贝 stream：提交即返回，`wait_stream` / `synchronize` / `max_inflight` |
| `PinnedBufferPool` | `async_/pinned.py` | 预分配 + 复用 + 容量上限；无可用块时阻塞（反压点） |
| `SnapshotManager` | `async_/machine.py` | 一代快照的状态机 + 多代门闸（单/双缓冲） |
| `LocalStorage` | `storage/local.py` | 本地目录后端：native 写（O_DIRECT+fsync+rename）/ preadv 读 |
| `CommitRecord` | `commit/local.py` | COMMIT 内容：step、metadata 字节数、文件清单 |
| `save_local()` | `api.py` | 本地层写闭环：plan → D2H → 数据 + `.metadata` + COMMIT |
| `open_version()` | `storage/recover.py` | 恢复门禁：只认带 COMMIT 的版本，校验 `.metadata` 与数据文件 |
| `targets_for_layout()` | `planner/targets.py` | 新布局 → 本 rank 的目标区域（`tp=1` 时整读） |
| `load_local()` | `api.py` | 本地层读闭环：COMMIT → 读取计划 → 按块 IO → 块 CRC-32C → H2D |
| `upload_local()` / `finalize_upload()` | `storage/upload.py` | 对象存储上传：每 rank 独立上传 + rank 0 只 `list` 检查 |
| `load_object()` | `storage/upload.py` | 对象存储恢复：COMMIT → 合并索引 → 整文件下载 → 本地读闭环 |
| `RankCommit` | `storage/upload.py` | 分片提交点：`step_N/rank_i/COMMIT`，证明本 rank 传完了 |
| `plan()` | `commit/gc.py` | 回收判据（纯函数）：删哪些、留哪些、为什么 |
| `collect()` / `gc_local()` | `commit/gc.py` / `api.py` | 回收执行器：一次 `list` + 目录扫描 + 幂等删除，默认 `dry_run` |
| `GcWorker` | `commit/gc.py` | 周期回收线程：启动时跑一次，之后每 `interval` 秒 |

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
PUT 到 S3 兼容存储 → 读回与源文件逐字节一致（大张量另走 multipart 分支）。上传本身带
对象级校验和（`x-amz-checksum-crc32`），见「本地分片 CRC 与对象存储校验的分工」。

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
pinned 池与 `D2HStream` 已经实现，还没做的是后台进程（`save_async`）和真卡上的
`CudaBackend`。

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

## 框架适配（`adapters/dtensor.py`）

把 DTensor 的 `placements` 翻译成 `ShardSource` 的全局坐标，**不 import torch**：
真 `torch.distributed.tensor.DTensor` 与测试里的 stub 走同一条鸭子类型路径
（`placements` / `device_mesh` / `to_local()` / `shape` / `dtype`）。

```python
from adapters.dtensor import describe_state

shards = describe_state(model.state_dict())       # 每个 DTensor -> 一个分片描述
ckpt.save_local(
    {s.fqn: to_device(s.local) for s in shards},  # 本地那片（数据）
    "/ckpt", step=step,
    sources=[s.source for s in shards],           # 全局坐标（metadata）
)
```

| placement | 处理 |
|---|---|
| `Shard(dim)` | 沿 `dim` 切成 mesh 该维的份数，取本 rank 那一份；不整除时前 `shape[dim] % n` 份各多一个（与 FSDP2 / `torch.chunk` 一致） |
| `Replicate()` | 该维上所有副本内容相同 → 只影响 `n_replicas` / `replica_index`，写者按 `hash(FQN) % n` 挑一个，不做内容比对 |
| `Partial` / `_StridedShard` | 直接报错：前者要先 all-reduce，后者是交错切分，都不落成矩形区间 |

多维 mesh 按 `placements[i]` 对应 `mesh.shape[i]`，可以混着来（`[Replicate(), Shard(1)]`
= DP 副本 + TP 列切）。普通张量（没有 `placements`）当成 1 副本整体，可以和 DTensor
混在同一个 `state_dict` 里。本地分片形状与推出的区域不一致会报错 —— `Partial` /
交错切分正是靠这个前提被挡住的。

`tests/test_dtensor_adapter.py` 覆盖：多维 Shard/Replicate 组合、不整除余数规则、
dtype 归一（`torch.float32` / numpy / 短名）、各类拒绝路径，以及「每 rank 存自己那片 →
上传 → rank 0 提交 → 从对象存储恢复拼回全局」的端到端往返。

## 对象存储后端（S3）

`storage/object.py` 用 boto3 接 S3 兼容存储（SeaweedFS / MinIO / AWS S3），
是 `storage/base.py` 接口的一个实现，调用语义与 `storage/local.py` 一致：

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
uv pip install --python .venv/bin/python -r requirements.txt   # numpy + boto3 + google-crc32c
.venv/bin/python -m pytest      # 344 个用例：数据生成器 + 读取接口 + 对象存储 + D2H/stream/pinned 池 + GPU→S3
#              + 状态机 + 区域/索引/读写计划 + C++ 绑定（**未构建 native 时相关用例自动 skip**）
#              + 本地层写闭环（数据 + .metadata + COMMIT 可恢复）
#              + 本地层读闭环（布局不变 / TP 换卡数 / 默认 1 MiB 块 CRC-32C + 扩读与 COMMIT 门禁）
#              + 对象存储上传（每 rank 独立上传 / rank 0 list 门禁 / 整文件下载恢复）
#              + 垃圾回收（判据纯函数 / 上传后回收 / 残片 / 幂等 / 降级 / 周期线程）
#              + 仿真：训练 loop / 多 rank 分片 / 故障注入（见 tests/sim/README.md）
#              + DTensor 适配（placements → 全局坐标；每 rank 上传后恢复拼回全局）
#              + examples/e2e_sim.py 端到端剧本（1 / 2 个 rank 各跑一遍）
.venv/bin/python -m pytest tests/sim -m workload -s   # 规模与性能仿真，默认不跑
```

`tests/sim/` 是模拟测试层：没有 GPU 与对象存储时，用伪训练 loop、多进程多 rank、
故障注入（SIGKILL / 提交点缺失 / 破坏索引）和 64 MiB 规模仿真来验证生命周期行为。
方案、覆盖矩阵与待补全清单见 [`tests/sim/README.md`](tests/sim/README.md)。

## 对象存储上传（每 rank 独立上传）

`storage/upload.py` 把本地版本推到对象存储，**每个 rank 各传各的**，rank 0 只在
所有人都传完之后 `list` 一次前缀做检查，不做任何索引汇总：

```
每个 rank（独立进程，互不等）
  save_local() ──> 本地 step_N/ 已提交（本地 COMMIT 是第一道门）
  upload_local(root, s3, step=42, rank=i)
      PUT step_42/rank_i/__i_0.distcp     数据
      PUT step_42/rank_i/.metadata        本 rank 的分片索引
      PUT step_42/rank_i/COMMIT           分片提交点（最后写；它一出现本地就能回收）

rank 0
  finalize_upload(s3, step=42, ranks=range(N))
      list("step_42") ──> 每个 rank 的 .metadata + rank_i/COMMIT 都在？
      PUT step_42/COMMIT                  版本提交点（只记 step / ranks / layout）
```

```python
from storage.object import S3Config, S3Storage

s3 = S3Storage(S3Config(bucket="ckpt", prefix="runs",
                        endpoint_url="http://127.0.0.1:8333",   # SeaweedFS / MinIO
                        access_key_id="...", secret_access_key="..."))

api.upload_local("/ckpt", s3, step=42, rank=3)          # 每个 rank 调自己的
api.finalize_upload(s3, step=42, ranks=range(8), layout={"tp": 8})   # 只有 rank 0 调

report = api.load_object(s3, device="cuda:0")           # 取最新已提交版本
report.step        # 42
report.tensors["model.embed_tokens.weight"].numpy()
report.amplification                                    # 读放大（Range GET / 真正用到的字节）
```

- **三层提交点**：本地 `step_N/COMMIT` 管「本 rank 写完了，可以传」；
  `rank_i/COMMIT` 管「本 rank 传完了，本地可以删」；版本 `COMMIT` 管「这一版齐了，
  可以恢复」。每一层只放开一个动作（见「垃圾回收」一节）
- **rank 0 不读别人的索引**：`finalize_upload` 只有 `list` 和 `PUT`，`tests/`
  里用 `s3_server.store.requests` 断言这一点。齐全性由每个 rank 的分片提交点证明 ——
  「对象真的在」比「索引拼起来完整」更强，代价也不随 rank 数增长
- **提交点顺序**：`rank_i/COMMIT` 是每个 rank 的最后一步；版本 `COMMIT` 在全部
  分片提交点齐了之后写。少任何一个都让 `finalize_upload` 失败且不写 `COMMIT`，
  这一版对恢复端不可见；反过来它是**幂等可重跑**的，rank 0 崩在最后一步，重启
  补调一次就能把这一版补成可恢复
- **恢复端才做全局校验**：`open_object_version` 按 `COMMIT` 的 rank 清单读各 rank 的
  `.metadata`，把分片 `file` 加上 `rank_<i>/` 前缀后合并，再 `validate()` 查覆盖与重叠。
  缺任何一个 rank 的索引都在读数据之前报错，不会交出半份张量
- **rank 0 不逐个对象核对**：它只信分片提交点。真缺了数据对象由恢复端的 Range GET
  （短读即报错）抓 —— 数据对象清单记在 `rank_i/COMMIT` 里，但那是给人和工具看的，
  提交路径不为它付 `HEAD` 往返
- **残片会被指出来**：`finalize_upload` 会把 `COMMIT` 没声明的 `rank_*` 前缀当错误
  报出（多半是上一次重试的残留）。正式的 attempt/epoch 隔离还没做，见下
- 字节级完整性仍在上传边界：`S3Storage` 的 PUT 带对象级校验和，服务端独立重算比对

待补：`step_N/` 里带尝试号（避免重试残片混进同一前缀）、大文件分块流式上传、
上传失败重试与后台上传队列（`save_async`）。上传成功后回收本地暂存已由
「垃圾回收」一节覆盖。

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

// 单文件批量读：句柄自带 ring，短读直接报错（Python 侧 native.read_file_ranges 走这条）
storage::LocalFile* reader = nullptr;
storage::LocalFile::OpenRead("step_42/__0_0.distcp", {}, &reader);
std::vector<storage::RangeRead> ranges{
    {0, 4096, dst + 0 * 4096},
    {1 << 20, 2048, dst + 4096},
};
reader->ReadRanges(ranges);
reader->Close();

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
  并失去 page cache；只有大块顺序读才建议 `direct_io = true`（此时 offset/length/落点
  地址都要对齐）
- `LocalFile::ReadRanges()` 是读侧的批量入口：打开时自带 ring（深度至少 2 × 单批上限），
  一次提交读多段；`RangeRead` 只描述「文件内 offset + length + 落点」，fd 由句柄提供
- 顺序写默认 O_DIRECT：buffer 地址、offset、length 都要按 `alignment`（默认 4096）对齐；
  三者任一不对齐都返回 `kInvalidArgument`，由 C++ 层显式校验 —— 不同文件系统/内核对
  非对齐 O_DIRECT 的宽容度不一样（ext4 读会报 EINVAL，某些内核的写会静默接受），
  把判断留给内核会让同一次写入在不同机器上表现不一致。`native.aligned_view()` 就是
  为拿对齐 buffer 准备的；补齐责任在 planner，C++ 层不做 read-modify-write
- `WriteFile()` 是整文件便捷写法：临时文件 → 顺序 WRITE → fsync → rename → 目录 fsync，
  最后一段不足一块时自动补零到块边界（这块兜底缓冲活到 fsync 之后，避免零拷贝写在途
  时被释放）
- `RenameAtomic()` / `FsyncDir()` 给 commit 层做原子提交（COMMIT 临时文件 + rename + 目录 fsync）
- 短读不静默截断：`ReadMany` 返回 `kIoError`，同时 `bytes` 保留已读长度；
  `ReadRanges` 逐项检查结果，短读直接返回错误，不会交出半份数据
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

C++ 用例的临时目录读 `TMPDIR`（默认 `/tmp`），并会先探测该文件系统是否支持
O_DIRECT：tmpfs 会被判为不支持、整组用例退回 buffered。要在真实文件系统上跑，
显式给目录：

```bash
TMPDIR=$PWD/build/tmp ctest --test-dir build --output-on-failure
```

## 本地层写闭环

`api.save_local()` 把「计划 → D2H → 落盘 → 提交」串成一次同步保存，落点是本地目录；
远端（S3）上传留给后续的后台 worker，不在训练路径上。

```
build_save_plan ──> trigger(拿对齐 pinned 块) ──> 按 item.byte_offset D2H
   ──> write_view(数据文件: 临时→fsync→rename→目录 fsync) ──> .metadata
   ──> persist_ok() ──> COMMIT(临时→rename) ──> commit_ok()
```

产物就是设计文档 §5 的版本目录：

```
ckpt/
  step_42/
    __0_0.distcp      # 数据：planner 布局，按 4096 补齐到块边界
    .metadata         # 索引：分片全局坐标 + 字节区间
    COMMIT            # 提交标记：索引字节数 + 文件清单
```

没有 `latest` 指针：`step=None` 的答案恒等于「所有带 `COMMIT` 的版本里最大的 step」，
省下的只是一次 `list`（反正不读对象内容），却会带来「指针失效就恢复不了」这类问题。

- **只有带 `COMMIT` 的目录算可恢复**：`committed_steps()` / `latest_step()` 只认它
- 写失败或中途异常 → 状态机 `fail()`（未写文件则 `DROPPED`，有残缺目录则 `ABORTED`），
  `COMMIT` 不会写，这一版对恢复端不可见
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

## 垃圾回收（上传完即回收）

本地层只是**上传前的中转**：本 rank 的分片一到远端，本地那份就是纯冗余，可以直接删。
判据只有一条不变量 —— **step 单调推进** —— 所以不需要保留策略、租约、时间阈值或
尝试号（`commit/gc.py`）。

| 位置 | 候选 | 删除条件 | 理由 |
|---|---|---|---|
| 本地 | `step_N/` | 远端有 `step_N/rank_<本rank>/COMMIT` | `uploaded`：本分片已到远端 |
| 本地 | `step_N/` | 存在 `M > N` 的版本 `COMMIT` | `superseded`：训练已越过第 N 代 |
| 远端 | `step_N/`（整个前缀） | 没有版本 `COMMIT` 且有 `M > N` 的版本 `COMMIT` | `superseded`：半途上传的残片 |

**保留的一律是「最新且未完成」**：本地最新且没有自己的分片提交点 → 可能正在写；
远端最新且没有版本提交点 → 可能正在传，也可能 rank 0 崩在最后一步（重跑
`finalize_upload` 就能补成可恢复）。回收延迟上界是一个 checkpoint 间隔。

```python
import api as ckpt
from commit.gc import GcWorker

report = ckpt.gc_local("/ckpt/rank_0", s3, rank=0, dry_run=True)   # 默认只看计划
report.planned      # 命中的候选 (location, step, reason)
report.skipped      # 保留的版本与理由
report.deleted      # 真跑时删掉的版本；远端字节记 0（list 不带对象大小）
report.reclaimed_bytes
report.versions     # 本次看到的已提交 step

# 周期回收：进程启动时先跑一次（收掉上次崩溃的残目录），之后每 interval 秒一次
with GcWorker("/ckpt/rank_0", s3, rank=0, interval=60.0) as worker:
    ...                     # worker.reports / worker.last_error 可观测
```

- **不需要问「写入者是否还活着」**：兄弟 rank 正在写的目录一定是最新 step，两条判据
  都不命中。这也是本地 root 必须**按 rank 独立**的原因 —— 本地判据删的是整个
  `step_N/` 目录，只有目录里全是本 rank 的文件时这句话才成立
- **删除顺序与写入镜像**：先删提交点（本地 `COMMIT` / 远端 `rank_*/COMMIT`），再删
  数据，最后移空目录。判据只看提交点在不在，所以删到一半被打断、重跑一次就收敛
- **远端不可达就什么都不删**：本地判据依赖远端状态，「查不到」不能当成「可以删」，
  原因记在 `GcReport.aborted`，周期线程继续跑
- **`save_local` 失败路径就地清理**：`ABORTED` 时直接删掉残缺目录并 `mark_cleaned()`，
  所以「本地无 `COMMIT` 的残目录」几乎只剩 SIGKILL 一种来源，定时回收只兜底
- **不做保留策略**：远端已提交的版本永不删除；本地不保留任何可恢复版本，正常恢复
  一律走 `load_object()`

## 本地层读闭环

`api.load_local()` 是写闭环的镜像：从版本目录把张量读回内存（先在 staging、最后在
device）。落盘时怎么切的，恢复时按新布局重新求交，所以换 TP 大小不用重新存一遍。

```
open_version(COMMIT 门禁 + 长度校验) ──> targets_for_layout(tp/shard_dims)
   ──> build_load_plan(求交 → 读取计划) ──> 窗口化取 pinned staging 块
   ──> 按索引里的 CRC 块切/扩读 ──> 按文件分组 native.read_file_ranges
   ──> 逐块 CRC-32C 校验 ──> 按 CopySpan 拷进目标 host 数组 ──> H2D ──> Tensor
```

```python
import api as ckpt

report = ckpt.load_local("/ckpt", device="cuda:0")      # step=None 取最新提交版本
report.step          # 42
report.engine        # "native"（io_uring）/ "python"（preadv）
report.requests      # ReadItem 数（合并相邻请求之后）
report.io_requests   # 实际 IO 单元数（扩读/同块合并之后）
report.amplification # 读放大：读进来的字节 / 真正用到的字节
report.crc_verified  # 本次是否校验（默认 True）
report.verified_blocks  # 通过 CRC-32C 的块数
report.unverified       # 没有扩读到完整块的读块（默认空；none/auto 未达阈值时非空）
report.unverified_bytes # 上述读块里没有可比 CRC 的字节数
report.tensors["model.embed_tokens.weight"].numpy()
report.blobs["lr_scheduler"]                             # 小状态原样读回

# 布局变了：TP 2 存、TP 4 读，dims 说清每个张量沿哪个维切
report = ckpt.load_local(
    "/ckpt", step=42, device="cuda:0",
    tp=4, tp_rank=1, shard_dims={"model.layers.0.attn.q_proj.weight": 1},
)
```

- **COMMIT 是第一道门**：没有 `COMMIT` 的目录、`.metadata` 长度与 `COMMIT` 不符、
  `COMMIT` 清单缺文件、数据文件大小不符 —— 全部在读第一个张量之前报错，不会吐出半份数据
- **staging 复用**：一个窗口内所有读请求紧凑排进一块 pinned 块（`PinnedBufferPool`，
  与写路径同一个 `aligned_allocator`），读完、拷完才归还；`budget_bytes` 控制窗口大小，
  单块大于预算时自己占一个窗口
- **CRC 的边界**：默认校验，索引里每个分片/小状态按 `crc_chunk_max`（默认 1 MiB）
  切块，每块一个 CRC-32C。读路径按同样的块边界切 IO，并默认把不完整的读块扩读到
  完整块（`crc_padding="block"`），完整块逐个比对；只有显式 `crc_padding="none"`
  或 `"auto"` 未达阈值时，头/尾部分块才记进 `LoadReport.unverified` 并计入
  `unverified_bytes`，不假装校验过
- **短读不静默截断**：C++ 的 `ReadRanges` 与 Python 的 `preadv` 回退都会报错
- `api.load()` 只接受本地版本目录（`COMMIT` / `.metadata`）；旧的 `manifest.json`
  格式已删除，历史数据集需要重新保存

## 本地分片 CRC 与对象存储校验的分工

本地版本目录的完整性由 `tmp → fsync → rename → fsync` 与 `COMMIT` 的文件清单兜结构性问题；
字节级正确性由**块级 CRC-32C** 兜，默认开启。

- **写侧总是算**。`save_local` 为每个分片与小状态按 `crc_chunk_max`（默认 1 MiB，
  可配置）切块，每块算一次 CRC-32C（Castagnoli；默认用 `google-crc32c`，走
  SSE4.2 / ARMv8 硬件指令），写进索引的 `crc_chunk_bytes` / `crc32c_chunks`。
  算的是 `[byte_offset, byte_offset + length)` 这段有效字节，不含对齐补齐。
- **块大小由长度推导**：`blocks = ceil(length / crc_chunk_max)`，
  `chunk = ceil(length / blocks)`。小分片整片一块，大分片块大小落在
  `crc_chunk_max/2` 到 `crc_chunk_max` 之间；尾块接近块大小，不会出现几 KB 的零头。
- **读侧默认校验 + 扩读**。读路径按 `crc_chunk_bytes` 切 IO；`crc_padding="block"`
  （默认）把不完整的读块扩读到完整块，同块/相邻块合并成一次 IO。每块重算后与索引
  比对，不匹配在**读出坏数据之前**抛 `ChecksumError`。关掉校验要显式传
  `verify_crc=False`；只想控制读放大可以传 `crc_padding="auto"` / `"none"`。
- **扩读的代价是读放大**。`strided` 逐行读时，一行可能只有几 KB，扩读到 1 MiB 块
  意味着最多几百倍放大；`crc_padding="auto"` 用“实际使用字节 / 扩展后字节”的阈值
  控制，`"none"` 则完全不扩、如实记 `unverified`。

### 为什么是 CRC-32C，而对象存储用 CRC32

两层算法**故意不同**，各有各的理由：

| | 本地层（`storage/checksum.py`） | 对象存储（`storage/object.py`） |
|---|---|---|
| 谁算 | 我们自己算 | 服务端独立算，客户端只声明算法 |
| 算法 | CRC-32C（Castagnoli） | CRC32（IEEE） |
| 理由 | 需要自己算才有得比，`google-crc32c` 有硬件指令 | botocore 无 `awscrt` 时只实现 `crc32`/`sha1`/`sha256`/`sha512`；换 CRC32C 要么加 `awscrt` 要么自己写 header，而 32 位校验的检错能力同量级，收益为零 |

两层的值互不流通：本地块校验和不进对象存储，对象存储的校验和不落本地索引。

### 对象存储恢复：整文件下载 + 本地块校验

对象存储恢复走 `load_object()`：先读版本 `COMMIT`、合并各 rank 的 `.metadata`，然后把
版本数据文件**流式下载**到本地临时目录（`StorageBackend.download`，峰值内存是一块下载
缓冲，不是整个文件），写本地 `.metadata` 与 `COMMIT`，最后调用 `load_local()` 按块校验。

- 对象存储侧只负责「发回来的 = 存的」：整对象 GET 带 `ChecksumMode=ENABLED`，
  PUT / multipart 带 `ChecksumAlgorithm=CRC32`，服务端独立重算；
- 本地 `load_local()` 负责「存的 = 当初写的」：按索引里的 CRC 块比对 CRC-32C；
- 下载完临时目录在返回前删除；`report.local_staging` 只留路径作诊断。

### 旧格式已删除

`manifest.json` 读取路径、`.metadata` / `COMMIT` 里的旧整片 `crc32c`、`COMMIT` 的
`metadata.crc32` 都已删除。旧 checkpoint 不再兼容：要么重新保存，要么用旧版本程序读取。

注意区分：`planner/save_plan.py` 的 `default_writer` 仍用 `zlib.crc32(fqn) % n`，那是
**哈希分桶**（决定 Replicate 副本里谁落盘），不是校验，不能跟着改——否则历史版本的
分片归属会变。

CRC 本来也抓不到最主要的那类故障：写错位置、rank 之间拼错、张量串了。它只证明「读回来
的 = 当时写进去的」，结构性错误由 `CheckpointIndex.validate()` 与 `COMMIT` 门禁负责。

## Python 绑定（`ckpt_native`）

`bindings/python.cpp` 把 `ckpt_core` 包成扩展模块，产物落在 `build/python/`，
由 `native/__init__.py` 负责挂 `sys.path` 并给出可读的缺失提示。绑定层只做四件事：

| 职责 | 做法 |
|---|---|
| 异常映射 | `Status` + `ErrorCode` → 项目异常（`StorageError` / `ObjectNotFoundError` / `FormatError` / `ValueError` / `NotImplementedError`），上层见不到 `ErrorCode` |
| GIL 释放 | `WriteFile` / `WriteAt` / `Sync` / `ReadRanges` / `rename_atomic` / `fsync_dir` 全程 `gil_scoped_release`，IO 期间不与训练线程抢 GIL |
| 零拷贝 | 通过 Python buffer 协议直接取指针：写用 `PyBUF_SIMPLE` + keepalive，读用 `PyBUF_WRITABLE` 直接落进调用方 buffer；`AlignedView` 让 C++ 分配的页对齐内存变成可写 `memoryview` |
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

# 读回来：多段随机读一次提交，数据直接落进 staging 块（零拷贝）
staging = bytearray(2 * 4096)
native.read_file_ranges(
    "step_12000/__3_0.distcp",
    [(0, 4096, 0), (4 * 4096, 4096, 4096)],   # (offset, length, dst_offset)
    staging,
)
reader = native.LocalFile.open_read("step_12000/__3_0.distcp")   # 复用 fd + ring
reader.read_ranges([(8 * 4096, 4096, 0)], staging)
reader.close()
```

`tests/test_native.py` 覆盖：写读回、多段写、末段自动补零、`LocalFile` 增量写、
批量读（多区间 / 短读 / 越界落点 / fd 复用）、原子 rename、错误映射、
池块 → D2H → 落盘的闭环，以及两条绑定层必须做对的事（跨线程用 `LocalFile` 报错；
写入期间另一个 Python 线程能推进 —— 证明 GIL 真的放掉了）。
扩展没构建时这些用例整体 skip，不影响纯 Python 链路。
