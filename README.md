# ShardCkpt — 分布式模型训练 Checkpoint 系统

面向分布式训练的分片 Checkpoint 系统：通过规划器（`planner`）完成分片去重、对齐与重切分求交，本地走 `O_DIRECT` + `io_uring`（可纯 Python 回退）原子落盘，远端通过 S3 条件写（CAS）实现代际隔离与单点提交，并支持本地热副本优先恢复与有界垃圾回收。

> **当前实现状态**：本地写/读闭环（`save_local` / `load_local` / `load`）、异步三层分离保存（`start_saver` / `save_async` / `close_saver` + `saver/` 存储子进程与节点通信层）、S3 上传与单点 `STATUS` 代际 fence（`begin_upload` / `upload_local` / `finalize_upload` / `load_object` / `load_checkpoint`）、GC 与有界保留（`gc_local` / `retain_remote_versions` / `GcWorker`）、跨 rank 一致门闸（`dist` + `SnapshotManager`）、DTensor 适配（`adapters/dtensor.py`）、C++ `io_uring` 扩展（`ckpt_native`）均已落地；`adapters/megatron.py`、`adapters/stateful.py`、`tiers/` 尚未实现。

## 目录

- [目录结构与分层](#目录结构与分层)
- [快速开始](#快速开始)
  - [安装依赖与构建原生扩展](#安装依赖与构建原生扩展)
  - [运行端到端模拟](#运行端到端模拟)
- [核心工作流与 API](#核心工作流与-api)
  - [1. 本地保存与恢复闭环](#1-本地保存与恢复闭环)
  - [2. 对象存储上传与代际 Fence](#2-对象存储上传与代际-fence)
  - [3. 垃圾回收与版本保留](#3-垃圾回收与版本保留)
- [核心模块设计](#核心模块设计)
  - [读写计划](#读写计划)
  - [框架适配](#框架适配)
  - [设备搬运与快照状态机](#设备搬运与快照状态机)
  - [数据完整性校验](#数据完整性校验)
  - [本地读写内核与 Python 绑定](#本地读写内核与-python-绑定)
- [测试与文档索引](#测试与文档索引)

---

## 目录结构与分层

Python 侧按依赖分层：`core` → `planner` → `storage` / `commit` / `dist` / `async_` / `adapters` / `tiers`，由 [`api.py`](api.py) 串联对外接口。

```text
ShardCkpt/
├── api.py              统一对外入口：save_local / load_local / load / 上传与恢复 / gc_local
├── core/               纯数据结构与异常，零外部依赖
│   ├── __init__.py       CheckpointError 及其子类
│   ├── metadata.py       Device 描述、dtype 映射
│   ├── region.py         N 维区域、求交、一维展平切分
│   └── index.py          分片元数据与索引（.metadata）序列化及覆盖校验
├── planner/            纯函数计划层，不碰 IO 和集合通信
│   ├── save_plan.py      写者去重、字节对齐、覆盖校验
│   ├── load_plan.py      求交生成读取计划、合并相邻读请求
│   └── targets.py        目标布局（tp / shard_dims）→ 本 rank 要读的区域
├── storage/            本地与对象存储后端及读写闭环
│   ├── base.py           StorageBackend / ConditionalBackend 协议与 WriteOutcome
│   ├── data.py           CheckpointData（api.load 返回容器）
│   ├── local.py          本地后端（native O_DIRECT 写 / preadv 读 / Python 回退）
│   ├── local_load.py     本地读闭环：窗口化 staging + 批量读 + 块 CRC 校验 + H2D
│   ├── recover.py        COMMIT 门禁：选版本、校验 .metadata、读出索引
│   ├── checksum.py       分块 CRC-32C（默认 1 MiB）与 google-crc32c / Python 回退
│   ├── crc_io.py         逻辑读块 → 可校验 IO 子区间切分与扩读合并
│   ├── object.py         S3 兼容存储（boto3：分片上传 / Range GET / 条件写 / 能力探针）
│   └── upload.py         S3 上传闭环与恢复（代际隔离 / 热副本优先）
├── commit/             提交协议、代际 fence 与垃圾回收
│   ├── local.py          本地 .metadata → COMMIT 原子提交、版本扫描与删除
│   ├── generation.py     代际（gen_0007-xxx）与单点 STATUS CAS 状态机
│   └── gc.py             GC 判据（纯函数）、执行器、GcWorker 线程、远端有界保留
├── async_/             DeviceBackend / Tensor、D2HStream、PinnedBufferPool、SnapshotManager
├── saver/              异步保存三层分离：memfd 池、控制面协议、纯 host 落盘、存储子进程与节点通信层
├── dist/               集合通信抽象（all_reduce(MAX) / gather / scatter / broadcast 已落地）
├── adapters/           框架适配（dtensor.py 已实现；megatron.py / stateful.py 未实现）
├── tiers/              分层存储与节点间副本（未实现，仅 docstring）
├── native/             ckpt_native 加载点与对齐 buffer 桥接
├── bindings/           pybind11 绑定源码（python.cpp）
├── include/ckpt/       C++ 公共头文件（Status、io_uring 封装、LocalFile）
├── src/                C++ 实现（io/uring.cpp、storage/local/local_file.cpp）
├── examples/           端到端模拟脚本（e2e_sim.py）
├── tests/              pytest 单元测试、fixtures 生成器、sim 仿真测试、C++ GTest/CTest
└── docs/               详细设计规格文档
```

<details>
<summary>核心对外 API 与关键组件速查表</summary>

| 类别 | 组件 / 函数 | 位置 | 职责 |
|---|---|---|---|
| **总入口** | `save_local` / `load_local` / `load` | `api.py` | 本地保存与加载闭环（`load` 只接受本地版本目录） |
| | `start_saver` / `save_async` / `close_saver` | `api.py` / `saver/` | 跨进程异步保存、节点独立通信协调、后台上传与对称收敛 |
| | `begin_upload` / `upload_local` / `finalize_upload` | `api.py` | 开代广播、per-rank 上传、rank 0 单点 CAS 提交与保留 |
| | `load_object` / `load_checkpoint` / `gc_local` | `api.py` | 远端恢复、本地热副本优先恢复、本地/残片回收 |
| **设备与池** | `Device` / `DeviceBackend` / `register_backend()` | `core/`, `async_/` | `"cpu"` / `"cuda:N"` 抽象，默认含 `CpuBackend` 与 `MockCudaBackend` |
| | `D2HStream` / `PinnedBufferPool` / `default_pool()` | `async_/` | 异步 D2H 拷贝与常驻池复用（`configure_default_pool` / `close_default_pool`） |
| | `SnapshotManager` | `async_/machine.py` | 快照状态机与跨 rank 一致门闸（`all_reduce(MAX)`） |
| **存储与提交** | `LocalStorage` / `S3Storage` | `storage/` | 本地目录与 S3 兼容存储后端（均支持条件写协议） |
| | `Generation` / `FencedError` | `commit/generation.py` | 代际身份与 fence 裁决异常 |
| | `GcWorker` / `retain_remote_versions` | `commit/gc.py` | 周期残片清理与远端最近 `keep` 代有界保留 |

</details>

---

## 快速开始

### 安装依赖与构建原生扩展

纯 Python 链路仅需 `requirements.txt`（`numpy` / `boto3` / `google-crc32c`）：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

如需启用 `O_DIRECT` + `io_uring` 原生读写加速（未构建时自动回退到纯 Python，不影响测试通过）：

```bash
apt-get install -y liburing-dev                      # 需要 liburing.h + liburing.so
uv pip install --python .venv/bin/python pybind11     # 构建期依赖，不进 requirements.txt

cmake -S . -B build -G Ninja \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python \
      -Dpybind11_DIR=$(uv run --python .venv/bin/python python -m pybind11 --cmakedir)
cmake --build build
```

> [!NOTE]
> `Python3_EXECUTABLE` 必须显式指定，避免 CMake 绑定到系统其他 Python 版本导致 `.so` ABI 后缀不匹配。

### 运行端到端模拟

[`examples/e2e_sim.py`](examples/e2e_sim.py) 一条命令跑完「伪训练 → 分片保存 → 崩溃 → 上传 → 从对象存储恢复 → 续训 → 回收」，**不需要 GPU 也不需要外部 S3**：显存由 `MockCudaBackend` 用 host 内存承载，S3 由进程内 [`tests/s3_stub.py`](tests/s3_stub.py) 模拟，而读取路径、提交点语义、Range GET 与 H2D 均走正式实现。

```bash
.venv/bin/python examples/e2e_sim.py              # 2 个 rank（TP=2），约 5 秒
.venv/bin/python examples/e2e_sim.py --ranks 4    # 自定义 world size
.venv/bin/python examples/e2e_sim.py --ranks 1 --keep   # 保留工作目录供检查
.venv/bin/python examples/e2e_sim.py --endpoint-url http://127.0.0.1:8333 \
    --bucket ckpt --prefix runs/e2e --access-key ... --secret-key ...   # 连接真实 SeaweedFS / MinIO
```

脚本按顺序验证 7 个阶段，任何断言失败即非零退出：

| 阶段 | 验证内容 |
|---|---|
| 1 | 伪训练 step 3 / 6 / 9：rank 0 开代广播 → 各 rank 按 TP 列切分保存本地 → 独立上传 → rank 0 `list` 后单点 CAS 提交 |
| 2 | 本地门禁：只有带 `COMMIT` 的版本可恢复；单 rank 目录单独读会被全局覆盖校验拒绝 |
| 3 | 崩溃注入：step 12 的最后一个 rank 在写 `COMMIT` 前被 `SIGKILL` |
| 4 | 提交点语义：`finalize_upload(12)` 失败、`load_object(12)` 报错，最新可用版本仍为 step 9 |
| 5 | 对象存储恢复：整读一份 + 多 rank 分别按分片拼回全局，两者逐字节一致 |
| 6 | 续训一致性：从 step 9 恢复再跑 3 步，结果与连续训练到 step 12 逐字节一致 |
| 7 | 有界保留与 GC：step 15 提交后远端仅留最近 3 代（step 3 不可恢复），本地回收仅保留最新 1 代热副本 |

---

## 核心工作流与 API

### 1. 本地保存与恢复闭环

本地保存（`api.save_local`）与本地恢复（`api.load_local` / `api.load`）构成互为镜像的本地闭环，不涉及远端上传。

#### 版本目录布局

```text
ckpt/
└── step_42/
    ├── __0_0.distcp      数据：按 planner 布局写入，按 4096 字节对齐
    ├── .metadata         索引：分片全局坐标 + 字节区间 + 分块 CRC-32C
    └── COMMIT            提交标记：step、metadata 字节数与 CRC-32C、文件清单
```

> [!IMPORTANT]
> 系统**没有 `latest` 指针**，也没有旧版 `manifest.json` 兼容路径：`step=None` 恒取所有带 `COMMIT` 目录中的最大 `step`。只有通过 `COMMIT` 长度与 CRC-32C 校验的版本才允许进入恢复流程。

#### 保存路径（`save_local`）

```text
build_save_plan ──> trigger(从常驻池拿对齐块) ──> D2H 拷贝至 pinned buffer
   ──> write_view(数据文件: tmp → fsync → rename → dir fsync)
   ──> .metadata ──> persist_ok() ──> COMMIT(tmp → rename) ──> commit_ok()
```

- **两套写入引擎**：`engine="native"`（`ckpt_native` + `O_DIRECT` + `io_uring`）与 `"python"`（等价缓冲写）；`"auto"` 自动选择。
- **常驻内存池复用**：`pool=None` 时默认使用进程级 `default_pool()`（块粒度默认 8 MiB）。每次租块前通过 `trim()` 裁掉多余空闲块，稳态只占用一块，`report.pool_reused` 指示本次是否复用。
- **失败清理**：写失败触发状态机 `fail()`（未写盘回 `DROPPED`，已建残目录进 `ABORTED` 并就地清理），绝不写出 `COMMIT`。

#### 恢复路径（`load_local` / `load`）

```text
open_version(COMMIT 门禁 + 索引校验) ──> targets_for_layout(tp / shard_dims)
   ──> build_load_plan(求交 + 合并) ──> 窗口化获取 pinned staging 块
   ──> 按索引 CRC 块切分/扩读 ──> 按文件批量读 ──> 逐块 CRC-32C 校验 ──> 拷贝进 host 数组 ──> H2D
```

数据流固定为 `文件 → staging buffer → H2D → device buffer`（每个张量一次 H2D 拷贝）。支持在读取时重新指定 `tp` / `tp_rank` / `shard_dims`，按新布局求交读取。

```python
import api as ckpt

# 可选：启动时配置常驻 pinned 池
ckpt.configure_default_pool(chunk_bytes=16 << 20, max_bytes=1 << 30)

# 保存本地版本
report = ckpt.save_local(
    tensors, "/ckpt/rank_0", step=42,
    blobs={"rng/rank_0": rng_bytes, "lr_scheduler": lr_bytes},
    layout={"dp": 1, "tp": 1},
)
# report.state == SnapshotState.COMMITTED, report.engine, report.file_size, report.pool_reused

# 便捷加载（类似 torch.load(..., map_location=...)）
data = ckpt.load("/ckpt/rank_0", device="cuda:0")
t = data["model.embed_tokens.weight"]   # Device(type='cuda', index=0)
arr = t.numpy()                         # 经 D2H 取 host 副本

# 换布局加载：TP 2 保存 → TP 4 读取
load_rep = ckpt.load_local(
    "/ckpt/rank_0", step=42, device="cuda:0",
    tp=4, tp_rank=1, shard_dims={"model.layers.0.attn.q_proj.weight": 1},
)
# load_rep.amplification, load_rep.verified_blocks, load_rep.unverified_bytes

ckpt.close_default_pool()               # 训练结束或显存紧张时释放常驻块
```

---

### 2. 对象存储上传与代际 Fence

[`storage/upload.py`](storage/upload.py) 与 [`commit/generation.py`](commit/generation.py) 负责把各 rank 的本地提交版本推送到 S3 兼容存储（SeaweedFS / MinIO / AWS S3）。

#### 协议流程（单点 `STATUS` CAS 状态机）

```text
rank 0（开代，一次 all_reduce(MAX) 广播给全体；多 rank 必须先调此步）
  begin_upload(s3, step=42, ranks=range(N), collective=...)
      GET step_42/STATUS ── 404 → PUT If-None-Match: *   (active=gen_0001-..., committed=null)
                         └─ 200 → PUT If-Match: <etag>    (推进 active，原样保留 committed)

每个 rank（独立进程，互不等待）
  save_local() ──> 本地 step_42/COMMIT（允许上传）
  upload_local(root, s3, step=42, rank=i, generation=gen)
      GET step_42/STATUS                                  ← fence 检查：active 仍是我？
      PUT step_42/gen_0003-xxx/rank_i/__i_0.distcp        数据（If-None-Match + 流式分片）
      PUT step_42/gen_0003-xxx/rank_i/.metadata           本 rank 分片索引（If-None-Match）
      PUT step_42/gen_0003-xxx/rank_i/COMMIT              L2 分片提交点（最后写，含索引校验）

rank 0（单点原子提交）
  finalize_upload(s3, step=42, ranks=range(N), generation=gen, keep=3)
      GET step_42/STATUS                                  ← 检查 active / committed 状态
      list("step_42/gen_0003-xxx")                        ← 仅检查每个 rank 的 .metadata + COMMIT 齐全
      PUT step_42/STATUS  If-Match: <etag>                ← 单次 CAS 写入 committed（L3 提交）
      reclaim_stale_generations(...)                      ← 清理同 step 更早的残代
      retain_remote_versions(..., keep=3)                 ← 远端只留最近 keep 个已提交 step
```

#### 关键设计保证

- **代际隔离与混代拒绝**：每次保存尝试落在独立的 `gen_<seq>-<writer>` 前缀下；恢复时校验所有 `rank_i/COMMIT` 声明的 generation 与 `STATUS.committed` 一致，杜绝旧尝试与新尝试的残片混拼。
- **无跨 key 并发窗口**：`step_N/STATUS` 同时承载写权（`active`）和版本提交指针（`committed`），`acquire` 与 `commit` 共享同一 `ETag`。新代一旦抢走写权，旧代提交必遭服务端 `412 Precondition Failed` 拒绝（`FencedError`）。重试新代失败时，已有 `committed` 不受影响。
- **三层提交点语义**：
  1. 本地 `step_N/COMMIT` → 本 rank 写完，可上传；
  2. 远端 `rank_i/COMMIT`（L2）→ 本 rank 上传完成；
  3. 远端 `step_N/STATUS.committed`（L3）→ 全版本齐全，可恢复（且本地该代可进入热副本窗口）。
- **rank 0 不汇总索引**：`finalize_upload` 只做 `list` + 单点 CAS，不读各 rank `.metadata`；缺失任何 rank 时报错不写 `committed`，且支持幂等重跑补交。全局合并与 `CheckpointIndex.validate()` 推迟到恢复端（`open_object_version`）执行。
- **S3 条件写硬前置**：[`S3Storage`](storage/object.py) 默认 `conditional_writes="require"`，首次使用执行 4 步 `probe_conditional_writes()` 探针；若端点不严格执行 `If-None-Match` / `If-Match` 则拒绝写入。`conditional_writes="off"` 时降为**只读**模式。超过 `multipart_threshold`（默认 64 MiB）自动分片上传并将条件头挂在 `CompleteMultipartUpload` 上。

```python
from storage.object import S3Config, S3Storage
import api as ckpt

s3 = S3Storage(S3Config(
    bucket="ckpt", prefix="runs",
    endpoint_url="http://127.0.0.1:8333",   # SeaweedFS / MinIO；AWS S3 留空
    access_key_id="...", secret_access_key="...",
))
s3.ensure_conditional_writes()   # 验证端点真正执行条件写

# 上传
gen = ckpt.begin_upload(s3, step=42, ranks=range(8), collective=collective)
ckpt.upload_local("/ckpt/rank_3", s3, step=42, rank=3, generation=gen)
ckpt.finalize_upload(s3, step=42, ranks=range(8), generation=gen, keep=3)

# 恢复：优先命中本地热副本（零下载），缺失时回退 load_object()
report = ckpt.load_checkpoint(s3, local_root="/ckpt/rank_0", step=42, device="cuda:0")
# report.source == "local" 或 "object"，report.download_bytes 对应 0 或下载字节数
```

> 详细协议、失败模式矩阵与探针规格见 [`docs/implemented/generation.md`](docs/implemented/generation.md)。

---

### 3. 垃圾回收与版本保留

[`commit/gc.py`](commit/gc.py) 通过**三条互不重叠的判据**管理本地中转与远端存储空间，仅依赖「step 单调推进」与「代际 CAS 推进」，无需租约、TTL 或心跳：

| 机制 | 触发点 | 清理对象 |
|---|---|---|
| **残片回收** | `gc_local` / `GcWorker` | 本地已上传且超出热副本窗口的副本、本地/远端被越过的残片、被取代的旧代 |
| **同 step 旧代** | `finalize_upload` 提交后（`reclaim_stale=True`） | 同一 step 下序号早于当前已提交代的 `gen_*` 前缀 |
| **远端有界保留** | `finalize_upload` 提交后（`keep=3`） / `retain_remote_versions` | 远端完整但超出最近 `keep` 代的旧版本 |

#### 残片判据明细

| 位置 | 候选 | 删除条件 | 理由标签 |
|---|---|---|---|
| 本地 | `step_N/` | 远端**最新一次尝试**存在 `rank_<r>/COMMIT`，且超出本地热副本窗口（`keep_local`，默认 1） | `uploaded` |
| 本地 | `step_N/` | 存在 `M > N` 的远端已提交版本 | `superseded` |
| 远端 | `step_N/` | 无已提交版本、无代际前缀，且存在 `M > N` 的远端已提交版本 | `superseded` |
| 远端 | `step_N/<gen>/` | 既非 `STATUS.committed` 声明的代，且序号严格小于「最新一次尝试」 | `stale_generation` |

- **本地热副本层**：保留最近 `keep_local`（默认 1）个「远端已提交且本 rank 已上传」的本地版本，供 `load_checkpoint()` 零下载快速回退；未在远端提交的本地版本永不被当作热副本或被提前误删。
- **不误删在途代**：「最新一次尝试」取 `max(STATUS.active.seq, 目录最大代序号)` 而非 `STATUS.committed`；若 `STATUS` 损坏/不可读或远端不可达，则整体跳过（`aborted` / `unknown_versions`），绝不盲删。
- **安全删除顺序**：整版删除时先删 `STATUS`（立即禁恢复并 fence 残留写者）→ 再删 `rank_*/COMMIT` → 最后删数据；本地目录按 rank 独立存放，互不干扰。

```python
from commit.gc import GcWorker
import api as ckpt

# 预览 / 执行本地及残片回收（默认 dry_run=True）
rep = ckpt.gc_local("/ckpt/rank_0", s3, rank=0, dry_run=False, keep_local=1)
# rep.planned / rep.deleted / rep.skipped / rep.recovery_steps()

# 单独裁剪远端已提交版本
ret = ckpt.retain_remote_versions(s3, keep=3, dry_run=False)

# 后台周期回收线程
with GcWorker("/ckpt/rank_0", s3, rank=0, interval=60.0) as worker:
    ...  # 可观测 worker.reports / worker.last_error
```

> 详细规格见 [`docs/implemented/gc.md`](docs/implemented/gc.md)。待补全功能：「每 M 步永久归档一份」与 `save_async` 失败重传队列。

---

## 核心模块设计

### 读写计划

[`planner/`](planner/__init__.py) 是纯函数层：保存侧计算各 rank 写入布局与偏移，恢复侧按新拓扑求交生成最小 IO 计划。

- **副本去重**：Replicate 维上的多份相同副本按 `hash(FQN) % N`（`zlib.crc32`）确定唯一落盘写者，避免集中在 rank 0 写入。
- **块对齐与 ZeRO 展平切分**：每项写后按 4096 字节补齐（末项除外）以适配 `O_DIRECT`；ZeRO / FSDP1 / distributed optimizer 的 1D 展平区间通过 `flat=Interval(start, end)` 拆回 N 维矩形块，保存时零机内通信。
- **恢复策略**：分片在文件内按本地形状紧凑存放，整读单个分片只需一次连续读；列子集切分支持 `column_strategy="whole"`（整行块读后内存切片，读放大换少 I/O）与 `"strided"`（逐行精确读）。布局不变时自动退化为单段全长读。
- **两层覆盖校验**：`TensorMeta.validate(complete=False)` 供单 rank 自检；`CheckpointIndex.validate()` 在合并后校验全局全覆盖与文件内字节区间无重叠。

```python
from core.index import CheckpointIndex
from core.region import Interval
from planner import ShardSource, TargetTensor, build_load_plan, build_save_plan

plan = build_save_plan(
    [ShardSource(fqn="optim.model.w.exp_avg", global_shape=(4096, 4096), dtype="f32",
                 flat=Interval(0, 4096 * 2048))],
    rank=3, blobs={"rng/rank_3": 64},
)
# plan.items[0]: file="__3_0.distcp", byte_offset=0, length=32 MiB

index = CheckpointIndex.loads(metadata_text)
load_plan = build_load_plan(
    index,
    [TargetTensor.for_shard("model.layers.5.attn.q_proj.weight", (4096, 4096), "f32",
                            dim=1, index=3, count=4)],
)
for window in load_plan.windows(64 << 20):
    ...
```

---

### 框架适配

[`adapters/dtensor.py`](adapters/dtensor.py) 通过鸭子类型（`placements` / `device_mesh` / `to_local()` / `shape` / `dtype`）将 FSDP2 / `DTensor` 与普通张量混合翻译为 `ShardSource`，**不 import `torch`**：

| placement | 处理方式 |
|---|---|
| `Shard(dim)` | 沿 `dim` 切分，余数分摊给前 `shape[dim] % n` 份（与 `torch.chunk` 一致） |
| `Replicate()` | 标记 `n_replicas` / `replica_index`，由 planner 按 `hash(FQN) % n` 选写者 |
| `Partial` / `_StridedShard` | 直接拒绝报错（前者需先 all-reduce，后者非矩形连续区间） |

```python
from adapters.dtensor import describe_state
import api as ckpt

shards = describe_state(model.state_dict())
ckpt.save_local(
    {s.fqn: to_device(s.local) for s in shards},
    "/ckpt/rank_0", step=step,
    sources=[s.source for s in shards],
)
```

> `adapters/megatron.py`（PP/EP/融合参数）与 `adapters/stateful.py`（优化器状态 FQN 映射）目前为占位文件。

---

### 设备搬运与快照状态机

- **设备后端与 D2H**：[`async_/`](async_/__init__.py) 提供 `DeviceBackend`（默认注册 `CpuBackend` 与 `MockCudaBackend`；真机可通过 `register_backend()` 替换 `"cuda"`）。`copy_to_host` / `copy_into` 是全项目唯一的 D2H 路径，配合 `D2HStream`（`wait_stream` / `CopyHandle`）与 `PinnedBufferPool` 实现异步拷贝与反压。
- **快照状态机（[`async_/machine.py`](async_/machine.py)）**：

```text
SNAPSHOTTING ──d2h_done()──> PERSISTING ──persist_ok()──> COMMITTING ──commit_ok()──> COMMITTED
      │                          │                           │
      └─fail()→ DROPPED          └───────fail()→ ABORTED <───┘
```

- **三项关键语义**：
  1. `SNAPSHOTTING` 失败进入 `DROPPED`（未触碰存储，无需清理目录），写盘后失败进 `ABORTED`；
  2. buffer 仅在 `COMMITTED` / `ABORTED` / `DROPPED` 三个终态精确归还一次；
  3. **跨 rank 一致门闸**：多 rank 场景下 `trigger()` 将各 rank 的在途代数与池状态通过 [`dist/`](dist/__init__.py) 执行一次 `all_reduce(MAX)` 得出统一裁决（`START` / `SKIP` / `FORCE_WAIT`）。详细规格见 [`docs/implemented/gate.md`](docs/implemented/gate.md)；异步后台进程设计见 [`docs/planned/save_async.md`](docs/planned/save_async.md)。

---

### 数据完整性校验

本地与远端采用分层互补的校验设计：

| 维度 | 本地层（[`storage/checksum.py`](storage/checksum.py)） | 对象存储（[`storage/object.py`](storage/object.py)） |
|---|---|---|
| **算法** | CRC-32C（Castagnoli，默认 `google-crc32c` 硬件指令） | CRC32（IEEE，`ChecksumAlgorithm=CRC32`） |
| **算力归属** | 客户端按分块计算并写入 `.metadata`（含 `COMMIT` 里的 `metadata.crc32c`） | 服务端在 PUT / GET 时独立重算比对（无 `awscrt` 下的 boto3 原生支持） |
| **语义边界** | 证明「存的 = 当初写的」 | 证明「发回来的 = 服务端存的」 |

- **分块与扩读**：写侧按 `crc_chunk_max`（默认 1 MiB）对有效数据区间切块计算；读侧默认 `crc_padding="block"` 将部分块扩读到整块边界逐块校验，不匹配即抛 `ChecksumError`。可通过 `crc_padding="auto"` / `"none"` 或 `verify_crc=False` 权衡读放大。
- **对象存储恢复**：`load_object()` 流式下载数据文件至临时目录，再调用 `load_local()` 完成按块 CRC-32C 校验。
- **哈希分桶区分**：`planner/save_plan.py` 中 `default_writer` 使用 `zlib.crc32(fqn) % n` 做副本分配，属于哈希分桶而非数据校验。

---

### 本地读写内核与 Python 绑定

[`include/ckpt/`](include/ckpt/storage/local_file.hpp) 与 [`src/`](src/storage/local/local_file.cpp) 实现基于 `io_uring` 的零额外线程本地文件 IO；[`bindings/python.cpp`](bindings/python.cpp) 导出为 `ckpt_native`。

- **读写策略**：随机读默认 buffered（避免非对齐 `O_DIRECT` 放大读并保留 page cache）；顺序写默认 `O_DIRECT` + `fsync` + `rename` + 目录 `fsync`，由 C++ 层显式检查地址/偏移/长度 4096 对齐，短读一律报 `kIoError` 不静默截断。
- **Python 绑定四项职责**：`Status` → Python 异常映射、IO 全程 `gil_scoped_release` 释放 GIL、基于 buffer 协议与 `AlignedView` 的零拷贝读写、以及 `LocalFile` 单线程归属检查（跨线程调用报 `RuntimeError`）。

```python
import native
from native import aligned_view

block = aligned_view(1 << 20, 4096)            # 页对齐缓冲，可直接用于 O_DIRECT
tensor.copy_to(block.view)
native.write_file("step_12000/__3_0.distcp", [block.view[: tensor.nbytes]], direct_io=True)
block.release()

staging = bytearray(2 * 4096)
native.read_file_ranges("step_12000/__3_0.distcp", [(0, 4096, 0), (16384, 4096, 4096)], staging)
```

---

## 测试与文档索引

### 运行测试

```bash
# 1. Python 全套单元测试 + 仿真 + e2e 剧本（未构建 ckpt_native 时 C++ 绑定用例自动 skip）
.venv/bin/python -m pytest

# 2. 规模与性能仿真（默认跳过）
.venv/bin/python -m pytest tests/sim -m workload -s

# 3. C++ 单元测试（要在支持 O_DIRECT 的真实文件系统上跑可显式指定 TMPDIR）
TMPDIR=$PWD/build/tmp ctest --test-dir build --output-on-failure
```

### 相关文档

| 文档 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | 设计与计划文档目录总览（含目录分类与完成进度分类） |
| [`docs/implemented/generation.md`](docs/implemented/generation.md) | S3 单点 `STATUS` 代际隔离、fence 协议与失败模式矩阵（已完成） |
| [`docs/implemented/gc.md`](docs/implemented/gc.md) | 本地与远端 GC 判据、安全删除顺序与有界保留规格（已完成） |
| [`docs/implemented/gate.md`](docs/implemented/gate.md) | 跨 rank 一致门闸设计与实现偏差（已完成） |
| [`docs/implemented/recovery_and_coordination.md`](docs/implemented/recovery_and_coordination.md) | 崩溃恢复补全（R1-R16a）与提交协调监控（C1-C4）统一计划（已完成） |
| [`docs/planned/save_async.md`](docs/planned/save_async.md) | 异步保存进程分离设计规格（M0–M3 已实现，M4 可选优化） |
| [`tests/sim/README.md`](tests/sim/README.md) | 仿真测试方案、覆盖矩阵与待补全清单 |
| [`tests/fixtures/README.md`](tests/fixtures/README.md) | 确定性测试数据集生成器用法（`make_data.py`） |
