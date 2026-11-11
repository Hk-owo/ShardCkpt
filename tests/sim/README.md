# tests/sim — 模拟测试方案

没有 GPU、没有对象存储、没有多机的前提下，用**仿真环境**验证 checkpoint 系统的
生命周期行为。这里只模拟「训练世界」和「故障」，不模拟真实训练数值。

## 为什么需要这一层

现有 261 个用例大多是**单点**的：保存能写、索引能解析、planner 求交正确、S3 后端
语义对。它们各自的假设没人交叉验证过：

- 「保存时写的字节」和「恢复时读的字节」是不是同一份（版本隔离、换布局）
- 多 rank 各写一片之后，拼起来的索引是否真能恢复（rank 间写者分配）
- 写到一半崩溃时，恢复端会不会吐出半份数据（提交点语义）
- 一次训练要存几百次，pinned 池、fd、请求数会不会随步数增长（资源）

这些只有把组件拼成**生命周期**才暴露，而且要能在 CI 上跑几秒而不是几小时。

## 运行方式

```bash
.venv/bin/python -m pytest tests/sim -q              # 默认套件（22 个用例，约 2s）
.venv/bin/python -m pytest tests/sim -m workload -s  # 规模/性能仿真（另 2 个，约 1s）
.venv/bin/python -m pytest -q                        # 全仓（sim 已并入默认套件）
```

| marker | 含义 |
|---|---|
| `sim` | 仿真测试（伪训练 loop / 多 rank / 故障注入），默认就跑 |
| `workload` | 规模与性能仿真，默认排除（`pytest.ini` 的 `-m "not workload"`） |

`tests/sim/sim.py` 是仿真部件（伪训练状态、device 张量构造、rank0 汇总），
`tests/sim/conftest.py` 把本目录加进 `sys.path` 并提供默认 `state` 夹具。

## 已有基础（可以放心依赖）

| 能力 | 位置 | 仿真里怎么用 |
|---|---|---|
| `MockCudaBackend` | `async_/__init__.py` | 用 host 内存冒充显存，`stats` 可断言拷贝次数/字节 |
| `s3_stub` | `tests/s3_stub.py` | 进程内 S3，真算校验和、multipart、copy |
| 数据生成器 | `tests/fixtures/make_data.py` | `build_specs` + `make_tensor` 造确定性 state |
| `PinnedBufferPool` / `SnapshotManager` | `async_/` | 长跑复用同一个池，统计 `reuses` / `in_use_bytes` |
| `LoadReport` / `SaveReport` | `storage/local_load.py` / `api.py` | 成本口径：`requests` / `amplification` / `d2h_bytes` |

## 分层方案

### L0 契约与等价性 — ⏳ 待补全

防止 mock 与真实实现漂移，成本最低，但需要真实现才有意义。

- `DeviceBackend` 契约用例：同一组断言参数化跑 `CpuBackend` / `MockCudaBackend`，
  将来 `CudaBackend` 原样复用（`allocate` / `copy_h2d` / `copy_d2h` / 越界 / stats）
- 引擎差分：`save_local(engine="python")` 与 `"native"` 产出的数据文件字节、
  `.metadata` JSON 逐字节相同（native 未构建时自动 skip）
- S3 后端契约：`s3_stub` 与 docker MinIO 跑同一套用例，靠 `mark.s3_real` 分流
- 现状：`tests/test_local_read.py` 已有一条「python 引擎读出同样字节」，但**没有**
  写侧的引擎差分，也没有 backend 契约层

### L1 单进程训练 loop 仿真 — ✅ 已落地

`tests/sim/test_train_loop.py`，8 个用例：

| 用例 | 守的性质 |
|---|---|
| `test_periodic_saves_are_individually_recoverable` | 12 步跑完，3/6/9/12 各自可恢复、内容与当时快照逐字节一致 |
| `test_each_version_keeps_its_own_bytes` | 版本隔离：step 3 不被 6/9/12 覆盖 |
| `test_resume_from_checkpoint_continues_the_same_trajectory` | 从 step 6 恢复再跑 6 步 = 一口气跑 12 步（确定性，无容差） |
| `test_tp_change_after_save_rebuilds_the_global_state` | TP 1 存 → TP 4 读，各 rank 拼回全局；whole 策略有读放大 |
| `test_many_steps_reuse_one_pool_without_leaking` | 18 步共用一块池：`in_use_bytes=0`、`reuses>0`、池不涨、fd 不涨 |
| `test_training_loop_works_for_other_dtypes[f16/bf16]` | f16/bf16 走完整闭环 |

伪训练在 `sim.TrainingState`：`array * 0.999 + noise(step, fqn)`，对同一
`(seed, step, fqn)` 必然产出同一字节，所以「恢复后继续训练」可以逐字节比对。

### L2 多 rank 分片仿真 — ✅ 已落地（rank0 汇总走 test-only 旁路）

`tests/sim/test_multirank.py`，6 个用例。`spawn` 起 N 个进程，每个 rank 用
`save_local(sources=..., rank=i)` 写自己那份，rank0 汇总成可恢复版本：

| 用例 | 守的性质 |
|---|---|
| `test_merged_version_is_globally_consistent` | 合并索引 `validate()` 通过；分片张量 tp 份、Replicate 1 份 |
| `test_every_tp_rank_rebuilds_the_global_state` | TP=4 存，每个 tp_rank 拼回全局；`tp=1` 整读同一份数据 |
| `test_replicate_shards_have_exactly_one_writer` | Replicate 只由 `default_writer` 选中的 rank 落盘；D2H 总量 = 数据量（无整份复制） |
| `test_blobs_are_written_by_their_own_rank` | `rng/rank_i` 各 rank 各一份，读回不串 |
| `test_merge_rejects_incomplete_coverage` | 少一个 rank → 汇总直接 `PlanError`，不留半份索引 |
| `test_merge_is_byte_identical_across_runs` | 同样输入汇总两次，`.metadata` 逐字节相同 |

**汇总旁路**：`dist/` 的 gather 还没实现（`dist/__init__.py` 只有 docstring），所以
`sim.merge_rank_versions()` 用「读回各 rank 的 `.metadata` + 硬链接数据文件」代替
collective。真 gather 落地后把这个函数换成通信实现，**用例一条都不用改**。

### L3 故障注入 — ✅ 部分落地（崩溃 / 提交点 / 写失败）

`tests/sim/test_fault.py`，8 个用例：

| 用例 | 守的性质 |
|---|---|
| `test_crash_before_commit_leaves_version_invisible` | 子进程写完数据与 `.metadata`、在写 COMMIT 前被 `SIGKILL`：目录有残缺、无 COMMIT，没有更新的提交，旧版本仍完整可读 |
| `test_commit_gate_rejects_damaged_version[5 种]` | COMMIT 缺失 / `.metadata` 截断 / 数据文件缺失 / 数据文件截断 / COMMIT 清单漏文件 —— 全部在读第一个张量前 `FormatError` |
| `test_write_failure_leaves_no_newer_commit` | 写失败 → 没有更新的已提交版本，`latest_step()` 仍指向上一个版本 |
| `test_local_layer_does_not_checksum_bytes` | 改一个字节本地层不报错（设计决定），`verify_crc=True` 时如实报告「校验不了」 |

崩溃注入是**确定性**的：不按时间随机 kill，而是让 `write_commit` 那一步直接
`SIGKILL`，所以每次留下的都是「数据齐、提交点缺失」这一种半成品。

### L4 规模与性能仿真 — ✅ 最小版本落地

`tests/sim/test_scale.py`，2 个用例（`mark.workload`）。64 MiB state（8 × 8 MiB）：

| 用例 | 断言（只 gate 定量不变量） |
|---|---|
| `test_workload_save_load_invariants` | `requests == 张量数`、`amplification == 1.0`、`read_bytes == copied_bytes == h2d_bytes == 数据量` |
| `test_workload_tp_change_amplifies_reads_exactly` | TP 1→8 时 `amplification == 8.0`（整读策略的精确放大），本 rank 拿到的列逐字节正确 |

墙钟与峰值 RSS 只 `print`（`-s` 可见），**不设阈值** —— CI 抖动会把时间断言变成
flaky。实测参考值（本机，64 MiB）：save 250–700 MiB/s（首次含分配开销）、
load ≈ 500 MiB/s、峰值 RSS ≈ 240 MiB。

## 上传链路（已落地，测试不在本目录）

`storage/upload.py` 的上传闭环按「每 rank 独立上传 + rank 0 只 `list` 检查」实现，
用例在 `tests/test_object_upload.py`（10 个，走进程内 `s3_stub`）。它和本目录的分工：

| | tests/sim/ | tests/test_object_upload.py |
|---|---|---|
| 对象 | 本地目录闭环（发现 `save_local` 单 rank 校验缺陷的地方） | S3 上传闭环（PUT / Range GET / list / 分片 COMMIT / 版本 COMMIT） |
| 多 rank | `sim.merge_rank_versions()` test-only 汇总旁路 | 每 rank 独立上传，**没有汇总**，rank 0 只 list |
| 全局覆盖校验 | 在 rank0 的旁路里做 | 在**恢复端**合并索引后做 |

上传侧待补（与本目录的待补全清单互补）：`step_N/` 带尝试号（epoch/attempt）
隔离重试残片、大文件分块流式上传、失败重试与后台上传队列。**上传成功后的本地清理
已落地**，见「回收链路」一节的 ``commit/gc.py``（用例在 `tests/test_gc.py`）。

## 回收链路（已落地，测试不在本目录）

`commit/gc.py` 的判据只有一条不变量（step 单调推进）：远端有本 rank 的分片提交点
⇒ 本地这份是冗余；存在更新的版本提交点 ⇒ 这一代是残片。没有保留策略、租约、时间
阈值或尝试号，所以本地回收与「最慢的 rank」「写入者是否还活着」都无关。

| | 谁测 |
|---|---|
| 判据纯函数（删谁 / 留谁 / 为什么） | `tests/test_gc.py::test_plan_*` 等 9 条 |
| 执行器（dry_run / 上传后回收 / 残片 / 幂等 / 远端不可达） | `tests/test_gc.py` 8 条 |
| 周期线程（启动即跑 / 远端挂了继续跑） | `tests/test_gc.py` 2 条 |
| 长跑不泄漏（目录数、池、fd） | 本目录的 `test_many_steps_reuse_one_pool_without_leaking` 尚未接入 GC |

## 待补全清单

按优先级排列，都是「现在缺前置条件」而不是「没想清楚」：

1. **`dist/` 真 gather**：落地后把 `sim.merge_rank_versions()` 换成 collective，
   L2 用例不变；同时可以删掉 rank 目录硬链接这一步。
2. **S3 故障注入**：`s3_stub` 加故障模式（500/503、连接重置、超时、限速、
   `store.objects` 改字节模拟静默损坏），断言 `upload_local` 的重试 / abort / 幂等；
   再配合 `storage/object.py` 的对象级校验和验证「静默损坏必须被发现」。
   现在已有的是「缺分片提交点 / 缺索引 / 缺数据对象」三类结构性故障（见
   `tests/test_object_upload.py`）。
3. **磁盘满**：小 tmpfs 或 quota 下保存，断言状态机走 `ABORTED`、没有更新的提交、
   旧版本可恢复。
4. **反压链路**：`s3_stub` 限速 + `PinnedBufferPool` 满 → 验证 `D2HStream` 与
   `SnapshotManager` 的 `SKIP → FORCE_WAIT` 门闸在训练侧真的生效。
5. **L0 契约层**：`DeviceBackend` 契约参数化 + 写侧 python/native 引擎差分。
6. **真机切换**：`CudaBackend` 骨架 + `mark.gpu`；docker MinIO（`mark.s3_real`）；
   两台机器上的跨机读取演练。
7. **后台保存**：`save_async` 落地后，仿真「训练不阻塞 + 上传队列重试 + 崩溃后
   不丢已提交版本」。
8. **`tiers/`**：分层存储落地后，仿真「本地命中 / 远端回源 / 副本选版本」。
9. **规模外推表**：把 L4 的 64 MiB 实测按 `state_dict` 体量线性外推到 7B/70B，
   给出「一次检查点预计耗时/带宽」的换算表。

## 写新仿真用例的约定

1. **确定性**：不按时间/随机数注入，固定 `seed` 与 `step`；失败要能一次复现。
2. **以字节为准**：断言「文件里的字节 / 读回的字节」，不信内存中间态。
3. **只 gate 定量不变量**：读放大、请求数、拷贝次数、覆盖完整性；墙钟只记录。
4. **每个用例结尾自检不漏**：池 `in_use_bytes == 0`、`manager.active == ()`、
   fd 数不涨。
5. **模拟要能整体替换**：任何「因为 X 还没实现所以先这样」的旁路（目前是
   `merge_rank_versions`）必须集中在一个函数里，并在 docstring 写清替换条件。

## 已知发现

`save_local` 原来在单 rank 上调用 `index.validate()`（全局覆盖校验），任何带分片的
保存（TP / ZeRO）都会 `PlanError: shards cover ...`，与 README「覆盖校验分两层」
的描述不符。已在 `api.py` 改为 `index.validate(complete=False)`（单 rank 自检），
由 L2 的 `test_merged_version_is_globally_consistent` 锁住「汇总后全局校验通过」。
