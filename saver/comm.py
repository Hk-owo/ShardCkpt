"""节点独立通信转发层（§1 Q1 / §5.2，纯 CPU，不碰 GPU 与训练梯度通信）。

职责：
1. 静态平衡多叉树组网：节点间 TCP 互联，本机内通过 ``AF_UNIX``（或回环 TCP）
   同时连接本机的训练进程（``role="trainer"``）与存储子进程（``role="storage"``）；
2. 差异化四类跨节点原语 + 恢复/退出原语：
   - ``GATE_REDUCE``：本机归约 → 跨节点树形 ``all_reduce(MAX)``（强同步）
   - ``PLAN_AUDIT``：树形逐层预去重、区间合并与覆盖审计 → 广播确认指纹（强同步）
   - ``GEN_BROADCAST`` / ``COMMIT_BROADCAST``：主节点异步单向广播
   - ``INVALIDATE_EVENT``：事件驱动即时全集群广播作废止损
   - ``TREE_BROADCAST`` / ``STORAGE_BARRIER`` / ``PROTECT_BROADCAST``：恢复选版、GC 保护与末代退出收敛
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue
import selectors
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core import PlanError
from core.region import Interval, Region, numel
from dist import CollectiveError, DEFAULT_TIMEOUT
from planner.save_plan import ShardSource
from saver.config import CommEndpoint
from saver.protocol import ProtocolError, recv_msg, send_msg

ROLE_TRAINER = "trainer"
ROLE_STORAGE = "storage"

# 控制面消息操作码
_OP_REGISTER = "REGISTER"
_OP_REGISTER_ACK = "REGISTER_ACK"
_OP_HELLO_NODE = "HELLO_NODE"
_OP_HELLO_ACK = "HELLO_ACK"

_OP_GATE_REDUCE = "GATE_REDUCE"
_OP_GATE_RESP = "GATE_RESP"
_OP_TREE_GATE_UP = "TREE_GATE_UP"
_OP_TREE_GATE_DOWN = "TREE_GATE_DOWN"

_OP_PLAN_AUDIT = "PLAN_AUDIT"
_OP_PLAN_AUDIT_RESP = "PLAN_AUDIT_RESP"
_OP_TREE_AUDIT_UP = "TREE_AUDIT_UP"
_OP_TREE_AUDIT_DOWN = "TREE_AUDIT_DOWN"

_OP_TREE_BCAST = "TREE_BCAST"
_OP_TREE_BCAST_RESP = "TREE_BCAST_RESP"
_OP_TREE_BCAST_UP = "TREE_BCAST_UP"
_OP_TREE_BCAST_DOWN = "TREE_BCAST_DOWN"

_OP_STORAGE_BARRIER = "STORAGE_BARRIER"
_OP_STORAGE_BARRIER_RESP = "STORAGE_BARRIER_RESP"
_OP_TREE_BARRIER_UP = "TREE_BARRIER_UP"
_OP_TREE_BARRIER_DOWN = "TREE_BARRIER_DOWN"

OP_GEN_BROADCAST = "GEN_BROADCAST"
OP_COMMIT_BROADCAST = "COMMIT_BROADCAST"
OP_INVALIDATE_EVENT = "INVALIDATE_EVENT"
OP_PROTECT_BROADCAST = "PROTECT_BROADCAST"


class CommError(CollectiveError):
    """节点独立通信转发层错误：超时、节点失联或树内强同步一致放弃。"""


def parent_of(node_id: int, fanout: int = 2) -> int | None:
    """静态平衡 ``fanout`` 叉树的父节点编号；根节点（0）返回 ``None``。"""
    if node_id <= 0:
        return None
    return (node_id - 1) // fanout


def children_of(node_id: int, num_nodes: int, fanout: int = 2) -> tuple[int, ...]:
    """静态平衡 ``fanout`` 叉树的子节点编号列表。"""
    start = fanout * node_id + 1
    end = min(num_nodes, start + fanout)
    return tuple(range(start, end))


@dataclass(frozen=True)
class NodeTreeConfig:
    """单个节点通信转发服务的静态树配置。"""

    node_id: int
    num_nodes: int
    node_addrs: tuple[tuple[str, int], ...]
    local_address: str
    fanout: int = 2
    world_size: int = 1
    local_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        if not 0 <= self.node_id < self.num_nodes:
            raise ValueError(f"node_id {self.node_id} out of range [0, {self.num_nodes})")
        if self.fanout < 1:
            raise ValueError(f"fanout must be >= 1, got {self.fanout}")
        if len(self.node_addrs) != self.num_nodes:
            raise ValueError(
                f"node_addrs length {len(self.node_addrs)} != num_nodes {self.num_nodes}"
            )
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not self.local_ranks and self.num_nodes == 1:
            object.__setattr__(self, "local_ranks", tuple(range(self.world_size)))


def _reserve_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def make_tree_configs(
    *,
    num_nodes: int,
    ranks_per_node: int | Sequence[int] = 1,
    base_dir: str | Path | None = None,
    fanout: int = 2,
    host: str = "127.0.0.1",
) -> tuple[NodeTreeConfig, ...]:
    """生成 ``num_nodes`` 台虚拟物理机的静态多叉树配置（适合单机多进程/多线程仿真）。"""
    if isinstance(ranks_per_node, int):
        counts = [ranks_per_node] * num_nodes
    else:
        counts = list(ranks_per_node)
        if len(counts) != num_nodes:
            raise ValueError("ranks_per_node length must match num_nodes")
    world_size = sum(counts)
    node_addrs = tuple((host, _reserve_tcp_port(host)) for _ in range(num_nodes))

    configs: list[NodeTreeConfig] = []
    cursor = 0
    for nid in range(num_nodes):
        ranks = tuple(range(cursor, cursor + counts[nid]))
        cursor += counts[nid]
        if base_dir is not None:
            unix_path = str(Path(base_dir) / f"comm-n{nid}.sock")
            if len(unix_path.encode("utf-8")) < 96:
                local_addr = f"unix://{unix_path}"
            else:
                local_addr = f"tcp://{host}:{_reserve_tcp_port(host)}"
        else:
            local_addr = f"tcp://{host}:{_reserve_tcp_port(host)}"
        configs.append(
            NodeTreeConfig(
                node_id=nid,
                num_nodes=num_nodes,
                node_addrs=node_addrs,
                local_address=local_addr,
                fanout=fanout,
                world_size=world_size,
                local_ranks=ranks,
            )
        )
    return tuple(configs)


def _bind_address(address: str) -> tuple[socket.socket, str]:
    if address.startswith("unix://"):
        path = Path(address[len("unix://") :])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        srv.listen(128)
        return srv, address
    if address.startswith("tcp://"):
        host_port = address[len("tcp://") :]
        host, raw_port = host_port.rsplit(":", 1)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, int(raw_port)))
        srv.listen(128)
        bound_host, bound_port = srv.getsockname()[:2]
        return srv, f"tcp://{bound_host}:{bound_port}"
    # 裸路径视为 unix socket
    path = Path(address)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(128)
    return srv, f"unix://{path}"


def _connect_address(address: str, *, timeout: float = 10.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_exc: OSError | None = None
    while True:
        try:
            if address.startswith("unix://"):
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(address[len("unix://") :])
                return sock
            if address.startswith("tcp://"):
                host, raw_port = address[len("tcp://") :].rsplit(":", 1)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((host, int(raw_port)))
                return sock
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(address)
            return sock
        except OSError as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise CommError(f"连接节点通信端 {address!r} 超时: {last_exc}") from exc
            time.sleep(0.01)


# --- 树形布局预去重与区间合并审计纯函数（§1 Q1 / §5.2） --------------------


def _extent_key(src: Mapping[str, Any]) -> str:
    if "flat" in src and src["flat"] is not None:
        return f"F:{src['flat'][0]}:{src['flat'][1]}"
    reg = src["region"]
    return f"R:{','.join(str(v) for v in reg[0])}:{','.join(str(v) for v in reg[1])}"


def _parse_extent(key: str) -> tuple[str, Any]:
    if key.startswith("F:"):
        _, s, l = key.split(":")
        return "flat", (int(s), int(l))
    _, off_s, shp_s = key.split(":")
    offs = tuple(int(v) for v in off_s.split(","))
    shps = tuple(int(v) for v in shp_s.split(","))
    return "region", (offs, shps)


def _merge_flat_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start < prev_end:
            raise PlanError(
                f"展平分片存在重叠区间 [{prev_start}, {prev_end}) 与 [{start}, {end})"
            )
        if start == prev_end:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _check_and_add_region(
    regions: list[list[list[int]]], new_reg: tuple[tuple[int, ...], tuple[int, ...]]
) -> None:
    candidate = Region(new_reg[0], new_reg[1])
    for existing in regions:
        other = Region(tuple(existing[0]), tuple(existing[1]))
        if candidate.overlaps(other):
            raise PlanError(f"分片区域重叠：{candidate} 与 {other}")
    regions.append([list(new_reg[0]), list(new_reg[1])])


def source_to_audit_dict(src: ShardSource) -> dict[str, Any]:
    d: dict[str, Any] = {
        "fqn": src.fqn,
        "global_shape": list(src.global_shape),
        "dtype": src.dtype,
        "replica_index": src.replica_index,
        "n_replicas": src.n_replicas,
    }
    if src.flat is not None:
        d["flat"] = [src.flat.start, src.flat.length]
    else:
        assert src.region is not None
        d["region"] = [list(src.region.offsets), list(src.region.shape)]
    return d


def build_local_audit_summary(
    sources: Sequence[Mapping[str, Any]],
    *,
    blob_names: Sequence[str] = (),
) -> dict[str, Any]:
    """把单卡分片描述转为可逐层预去重、合并的节点审计摘要。"""
    tensors: dict[str, dict[str, Any]] = {}
    for src in sources:
        fqn = str(src["fqn"])
        shape = [int(v) for v in src["global_shape"]]
        dtype = str(src["dtype"])
        n_rep = int(src["n_replicas"])
        rep_idx = int(src["replica_index"])
        ek = _extent_key(src)

        entry = tensors.get(fqn)
        if entry is None:
            entry = {
                "global_shape": shape,
                "dtype": dtype,
                "n_replicas": n_rep,
                "complete": False,
                "incomplete_extents": {},
                "flat_intervals": [],
                "regions": [],
                "covered_numel": 0,
            }
            tensors[fqn] = entry
        else:
            if entry["global_shape"] != shape or entry["dtype"] != dtype:
                raise PlanError(
                    f"{fqn}: 分片元信息不一致 {entry['global_shape']}/{entry['dtype']} vs {shape}/{dtype}"
                )
            if entry["n_replicas"] != n_rep:
                raise PlanError(
                    f"{fqn}: n_replicas 标注不一致 ({entry['n_replicas']} vs {n_rep})"
                )

        reps = entry["incomplete_extents"].setdefault(ek, [])
        if rep_idx in reps:
            raise PlanError(f"{fqn}: 覆盖范围 {ek} 的 replica_index={rep_idx} 重复声明")
        reps.append(rep_idx)

    for fqn, entry in tensors.items():
        _promote_completed_extents(fqn, entry)

    return {
        "tensors": tensors,
        "blobs": sorted({str(b) for b in blob_names}),
    }


def _promote_completed_extents(fqn: str, entry: dict[str, Any]) -> None:
    if entry["complete"]:
        return
    n_rep = int(entry["n_replicas"])
    total_numel = numel(tuple(int(v) for v in entry["global_shape"]))
    inc = entry["incomplete_extents"]
    done_keys: list[str] = []
    for ek, reps in list(inc.items()):
        if len(reps) == n_rep:
            if set(reps) != set(range(n_rep)):
                raise PlanError(f"{fqn}: 覆盖范围 {ek} 的 replica_index 非法 {sorted(reps)}")
            kind, geom = _parse_extent(ek)
            if kind == "flat":
                start, length = geom
                entry["flat_intervals"].append([start, start + length])
                entry["flat_intervals"] = _merge_flat_intervals(entry["flat_intervals"])
                entry["covered_numel"] += length
            else:
                _check_and_add_region(entry["regions"], geom)
                entry["covered_numel"] += Region(geom[0], geom[1]).numel
            done_keys.append(ek)
    for ek in done_keys:
        del inc[ek]

    # 混合 flat + rectangular 时若不完全覆盖会留给上面或 root 报错
    if not inc and entry["covered_numel"] == total_numel:
        if entry["flat_intervals"] and entry["regions"]:
            # 确认混合情况转矩形无重叠（罕见，仅兜底）
            rects: list[list[list[int]]] = list(entry["regions"])
            shape_t = tuple(int(v) for v in entry["global_shape"])
            from core.region import flatten_slice

            for start, end in entry["flat_intervals"]:
                for reg in flatten_slice(shape_t, Interval(start, end - start)):
                    _check_and_add_region(rects, (reg.offsets, reg.shape))
        entry["complete"] = True
        entry["flat_intervals"] = []
        entry["regions"] = []


def merge_audit_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """树形内部节点归并子摘要：检查 n_replicas / 副本唯一性 / 空间重叠并折叠已完成张量。"""
    merged_tensors: dict[str, dict[str, Any]] = {}
    merged_blobs: set[str] = set()

    for summary in summaries:
        merged_blobs.update(str(b) for b in summary.get("blobs", ()))
        for fqn, src_entry in summary.get("tensors", {}).items():
            dst = merged_tensors.get(fqn)
            if dst is None:
                merged_tensors[fqn] = {
                    "global_shape": list(src_entry["global_shape"]),
                    "dtype": str(src_entry["dtype"]),
                    "n_replicas": int(src_entry["n_replicas"]),
                    "complete": bool(src_entry["complete"]),
                    "incomplete_extents": {
                        k: list(v) for k, v in src_entry.get("incomplete_extents", {}).items()
                    },
                    "flat_intervals": [list(x) for x in src_entry.get("flat_intervals", ())],
                    "regions": [
                        [list(r[0]), list(r[1])] for r in src_entry.get("regions", ())
                    ],
                    "covered_numel": int(src_entry.get("covered_numel", 0)),
                }
                continue

            if (
                dst["global_shape"] != src_entry["global_shape"]
                or dst["dtype"] != src_entry["dtype"]
            ):
                raise PlanError(
                    f"{fqn}: 分片元信息不一致 {dst['global_shape']}/{dst['dtype']} "
                    f"vs {src_entry['global_shape']}/{src_entry['dtype']}"
                )
            if dst["n_replicas"] != int(src_entry["n_replicas"]):
                raise PlanError(
                    f"{fqn}: n_replicas 标注不一致 ({dst['n_replicas']} vs {src_entry['n_replicas']})"
                )
            if dst["complete"] or src_entry["complete"]:
                # 其中一方已声明整张量完整覆盖：另一方若还有任何分片，必是重叠或多余副本
                raise PlanError(f"{fqn}: 存在重复或重叠的分片覆盖")

            for ek, reps in src_entry.get("incomplete_extents", {}).items():
                target_reps = dst["incomplete_extents"].setdefault(ek, [])
                overlap_reps = set(target_reps) & set(reps)
                if overlap_reps:
                    raise PlanError(
                        f"{fqn}: 覆盖范围 {ek} 的 replica_index {sorted(overlap_reps)} 被多个 rank 同时声明"
                    )
                target_reps.extend(reps)

            for interval in src_entry.get("flat_intervals", ()):
                dst["flat_intervals"].append(list(interval))
                dst["covered_numel"] += int(interval[1]) - int(interval[0])
            if dst["flat_intervals"]:
                dst["flat_intervals"] = _merge_flat_intervals(dst["flat_intervals"])

            for reg in src_entry.get("regions", ()):
                reg_t = (tuple(int(v) for v in reg[0]), tuple(int(v) for v in reg[1]))
                _check_and_add_region(dst["regions"], reg_t)
                dst["covered_numel"] += Region(reg_t[0], reg_t[1]).numel

            _promote_completed_extents(fqn, dst)

    return {
        "tensors": merged_tensors,
        "blobs": sorted(merged_blobs),
    }


def finalize_audit_summary(summary: Mapping[str, Any]) -> str:
    """根节点最终覆核并生成极小确认指纹。"""
    tensors = summary.get("tensors", {})
    blobs = sorted(str(b) for b in summary.get("blobs", ()))
    if not tensors and not blobs:
        raise PlanError("gathered 全局保存计划为空：没有任何分片或小状态")

    schema: list[Any] = []
    for fqn in sorted(tensors):
        entry = tensors[fqn]
        inc = entry.get("incomplete_extents", {})
        if inc:
            ek = sorted(inc)[0]
            expected = set(range(int(entry["n_replicas"])))
            got = set(inc[ek])
            raise PlanError(
                f"{fqn}: 覆盖范围 {ek} 的副本声明不完整，缺 replica_index {sorted(expected - got)}"
            )
        if not entry.get("complete"):
            total = numel(tuple(int(v) for v in entry["global_shape"]))
            raise PlanError(
                f"{fqn}: 分片未覆盖完整全局张量（已覆盖 {entry.get('covered_numel', 0)}/{total} 元素）"
            )
        schema.append([fqn, entry["global_shape"], entry["dtype"], entry["n_replicas"]])

    canonical = json.dumps(
        {"tensors": schema, "blobs": blobs},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


# --- 节点通信转发服务端（纯 CPU） -------------------------------------------


@dataclass
class _ConnState:
    sock: socket.socket
    kind: str = "unknown"  # "local" | "child" | "parent"
    role: str = ""  # "trainer" | "storage"
    rank: int | None = None
    peer_node_id: int | None = None
    send_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _PendingRound:
    seq: int
    op: str
    local_values: dict[int, Any] = field(default_factory=dict)
    local_clients: dict[int, tuple[_ConnState, str]] = field(default_factory=dict)
    child_values: dict[int, Any] = field(default_factory=dict)
    sent_up: bool = False
    finished: bool = False


class NodeCommServer:
    """每台物理机一个的纯 CPU 节点通信转发模块（§1 Q1 / §5.2）。"""

    def __init__(self, config: NodeTreeConfig) -> None:
        self.config = config
        self.node_id = config.node_id
        self.num_nodes = config.num_nodes
        self.parent_id = parent_of(config.node_id, config.fanout)
        self.children_ids = children_of(config.node_id, config.num_nodes, config.fanout)
        self.expected_local_ranks = frozenset(config.local_ranks)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._started = False
        self._closed = False

        self._local_srv: socket.socket | None = None
        self._ext_srv: socket.socket | None = None
        self._bound_local_address: str = config.local_address
        self._bound_ext_address: tuple[str, int] = config.node_addrs[config.node_id]

        self._conns: dict[int, _ConnState] = {}  # fd -> _ConnState
        self._local_clients: list[_ConnState] = []
        self._child_conns: dict[int, _ConnState] = {}
        self._parent_conn: _ConnState | None = None

        self._gate_rounds: dict[int, _PendingRound] = {}
        self._audit_rounds: dict[int, _PendingRound] = {}
        self._bcast_rounds: dict[int, _PendingRound] = {}
        self._barrier_rounds: dict[int, _PendingRound] = {}

        self._thread: threading.Thread | None = None
        self._wakeup_r, self._wakeup_w = socket.socketpair()
        self._wakeup_r.setblocking(False)
        self._wakeup_w.setblocking(False)

    @property
    def local_address(self) -> str:
        return self._bound_local_address

    @property
    def endpoint(self) -> CommEndpoint:
        return CommEndpoint(
            local_address=self._bound_local_address,
            node_id=self.node_id,
            num_nodes=self.num_nodes,
            world_size=self.config.world_size,
        )

    def bind(self) -> "NodeCommServer":
        """仅绑定监听套接字（先 bind 再 connect_tree，避免多节点启动竞态）。"""
        if self._local_srv is not None:
            return self
        self._local_srv, self._bound_local_address = _bind_address(
            self.config.local_address
        )
        self._local_srv.setblocking(False)
        if self.num_nodes > 1:
            host, port = self.config.node_addrs[self.node_id]
            self._ext_srv, bound = _bind_address(f"tcp://{host}:{port}")
            self._ext_srv.setblocking(False)
            bhost, bport = bound[len("tcp://") :].rsplit(":", 1)
            self._bound_ext_address = (bhost, int(bport))
        return self

    def start(self) -> "NodeCommServer":
        if self._started:
            return self
        self.bind()
        self._started = True
        self._thread = threading.Thread(
            target=self._run_loop, name=f"node-comm-{self.node_id}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise CommError(f"node {self.node_id}: NodeCommServer 启动超时")
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._wakeup_w.send(b"\x01")
        except OSError:
            pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._cleanup_sockets()

    def __enter__(self) -> "NodeCommServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    # --- 内部通信循环 -------------------------------------------------------

    def _run_loop(self) -> None:
        sel = selectors.DefaultSelector()
        assert self._local_srv is not None
        sel.register(self._local_srv, selectors.EVENT_READ, data="local_accept")
        sel.register(self._wakeup_r, selectors.EVENT_READ, data="wakeup")
        if self._ext_srv is not None:
            sel.register(self._ext_srv, selectors.EVENT_READ, data="ext_accept")

        # 监听套接字已由 bind() 就绪，先释放 start() 等待，避免多节点启动顺序耦合
        self._ready.set()

        # 若有父节点，主动连向父节点
        if self.parent_id is not None and self.num_nodes > 1:
            phost, pport = self.config.node_addrs[self.parent_id]
            psock = _connect_address(f"tcp://{phost}:{pport}", timeout=10.0)
            send_msg(psock, {"op": _OP_HELLO_NODE, "node_id": self.node_id})
            ack = recv_msg(psock)
            if ack is None or ack[0].get("op") != _OP_HELLO_ACK:
                psock.close()
                raise CommError(f"node {self.node_id}: 父节点握手失败")
            psock.setblocking(True)
            pconn = _ConnState(sock=psock, kind="parent", peer_node_id=self.parent_id)
            with self._lock:
                self._parent_conn = pconn
                self._conns[psock.fileno()] = pconn
            sel.register(psock, selectors.EVENT_READ, data=pconn)

        try:
            while not self._stop.is_set():
                events = sel.select(timeout=0.5)
                for key, _mask in events:
                    tag = key.data
                    if tag == "wakeup":
                        try:
                            self._wakeup_r.recv(1024)
                        except OSError:
                            pass
                    elif tag == "local_accept":
                        self._accept_local(sel)
                    elif tag == "ext_accept":
                        self._accept_ext(sel)
                    elif isinstance(tag, _ConnState):
                        self._handle_conn_read(sel, tag)
        finally:
            sel.close()

    def _accept_local(self, sel: selectors.DefaultSelector) -> None:
        assert self._local_srv is not None
        while True:
            try:
                sock, _addr = self._local_srv.accept()
            except BlockingIOError:
                return
            except OSError:
                return
            sock.setblocking(True)
            conn = _ConnState(sock=sock, kind="local")
            with self._lock:
                self._conns[sock.fileno()] = conn
                self._local_clients.append(conn)
            sel.register(sock, selectors.EVENT_READ, data=conn)

    def _accept_ext(self, sel: selectors.DefaultSelector) -> None:
        assert self._ext_srv is not None
        while True:
            try:
                sock, _addr = self._ext_srv.accept()
            except BlockingIOError:
                return
            except OSError:
                return
            sock.setblocking(True)
            conn = _ConnState(sock=sock, kind="child")
            with self._lock:
                self._conns[sock.fileno()] = conn
            sel.register(sock, selectors.EVENT_READ, data=conn)

    def _handle_conn_read(self, sel: selectors.DefaultSelector, conn: _ConnState) -> None:
        try:
            res = recv_msg(conn.sock)
        except (OSError, EOFError, ProtocolError):
            res = None
        if res is None:
            self._disconnect(sel, conn)
            return
        msg, fds = res
        for fd in fds:
            try:
                import os

                os.close(fd)
            except OSError:
                pass
        self._dispatch_message(conn, msg)

    def _disconnect(self, sel: selectors.DefaultSelector, conn: _ConnState) -> None:
        try:
            sel.unregister(conn.sock)
        except Exception:
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        with self._lock:
            self._conns.pop(conn.sock.fileno(), None)
            if conn in self._local_clients:
                self._local_clients.remove(conn)
            if conn.peer_node_id is not None and self._child_conns.get(conn.peer_node_id) is conn:
                self._child_conns.pop(conn.peer_node_id, None)
            if self._parent_conn is conn:
                self._parent_conn = None

        # 强同步断连策略：任一子节点/父节点/参会 trainer 断连，立即中止所有在途强同步轮次
        if conn.kind in ("child", "parent") or (
            conn.kind == "local" and conn.role == ROLE_TRAINER
        ):
            reason = f"peer disconnected (node={self.node_id}, kind={conn.kind}, rank={conn.rank})"
            self._abort_all_strong_sync(reason)

    def _send_conn(self, conn: _ConnState, msg: Mapping[str, Any]) -> bool:
        with conn.send_lock:
            try:
                send_msg(conn.sock, msg)
                return True
            except OSError:
                return False

    def _dispatch_message(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        op = msg.get("op")
        if op == _OP_REGISTER:
            conn.rank = int(msg["rank"])
            conn.role = str(msg["role"])
            self._send_conn(conn, {"op": _OP_REGISTER_ACK, "node_id": self.node_id})
            return

        if op == _OP_HELLO_NODE:
            child_id = int(msg["node_id"])
            conn.peer_node_id = child_id
            with self._lock:
                self._child_conns[child_id] = conn
            self._send_conn(conn, {"op": _OP_HELLO_ACK, "node_id": self.node_id})
            return

        # --- 异步单向广播与事件广播（代际 / 提交 / 即时作废 / 恢复保护） ---
        if op in (
            OP_GEN_BROADCAST,
            OP_COMMIT_BROADCAST,
            OP_INVALIDATE_EVENT,
            OP_PROTECT_BROADCAST,
        ):
            self._forward_broadcast(conn, msg)
            return

        # --- 强同步：门闸归约 ---
        if op == _OP_GATE_REDUCE:
            self._on_local_gate(conn, msg)
        elif op == _OP_TREE_GATE_UP:
            self._on_tree_gate_up(conn, msg)
        elif op == _OP_TREE_GATE_DOWN:
            self._on_tree_gate_down(msg)

        # --- 强同步：布局预去重覆盖审计 ---
        elif op == _OP_PLAN_AUDIT:
            self._on_local_audit(conn, msg)
        elif op == _OP_TREE_AUDIT_UP:
            self._on_tree_audit_up(conn, msg)
        elif op == _OP_TREE_AUDIT_DOWN:
            self._on_tree_audit_down(msg)

        # --- 强同步：树形对象广播（恢复选版） ---
        elif op == _OP_TREE_BCAST:
            self._on_local_bcast(conn, msg)
        elif op == _OP_TREE_BCAST_UP:
            self._on_tree_bcast_up(conn, msg)
        elif op == _OP_TREE_BCAST_DOWN:
            self._on_tree_bcast_down(msg)

        # --- 强同步：存储端退出收敛屏障 ---
        elif op == _OP_STORAGE_BARRIER:
            self._on_local_barrier(conn, msg)
        elif op == _OP_TREE_BARRIER_UP:
            self._on_tree_barrier_up(conn, msg)
        elif op == _OP_TREE_BARRIER_DOWN:
            self._on_tree_barrier_down(msg)

    # --- 异步与事件广播转发 -------------------------------------------------

    def _forward_broadcast(self, sender: _ConnState, msg: Mapping[str, Any]) -> None:
        with self._lock:
            tree_targets = [
                c
                for c in ([self._parent_conn] + list(self._child_conns.values()))
                if c is not None and c is not sender
            ]
            local_targets = [
                c
                for c in self._local_clients
                if c is not sender and c.role in (ROLE_STORAGE, ROLE_TRAINER)
            ]
        for target in tree_targets:
            self._send_conn(target, msg)
        for target in local_targets:
            # 正常只向存储客户端推（作废事件也向训练客户端推便于感知）
            if msg.get("op") == OP_INVALIDATE_EVENT or target.role == ROLE_STORAGE:
                self._send_conn(target, msg)

    # --- 强同步辅助 ---------------------------------------------------------

    def _abort_all_strong_sync(self, reason: str) -> None:
        for table, down_op, resp_op in (
            (self._gate_rounds, _OP_TREE_GATE_DOWN, _OP_GATE_RESP),
            (self._audit_rounds, _OP_TREE_AUDIT_DOWN, _OP_PLAN_AUDIT_RESP),
            (self._bcast_rounds, _OP_TREE_BCAST_DOWN, _OP_TREE_BCAST_RESP),
            (self._barrier_rounds, _OP_TREE_BARRIER_DOWN, _OP_STORAGE_BARRIER_RESP),
        ):
            with self._lock:
                rounds = list(table.values())
                table.clear()
                children = list(self._child_conns.values())
                parent = self._parent_conn
            for rnd in rounds:
                if rnd.finished:
                    continue
                rnd.finished = True
                err_down = {"op": down_op, "seq": rnd.seq, "ok": False, "error": reason}
                for child in children:
                    self._send_conn(child, err_down)
                if parent is not None and rnd.sent_up:
                    pass
                for _rank, (client, req_id) in rnd.local_clients.items():
                    self._send_conn(
                        client,
                        {"op": resp_op, "req_id": req_id, "ok": False, "error": reason},
                    )

    # --- 1. 门闸树形归约 ----------------------------------------------------

    def _on_local_gate(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        rank = int(conn.rank if conn.rank is not None else msg["rank"])
        req_id = str(msg["req_id"])
        values = [int(v) for v in msg["values"]]
        with self._lock:
            rnd = self._gate_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_GATE_REDUCE))
            rnd.local_values[rank] = values
            rnd.local_clients[rank] = (conn, req_id)
        self._try_advance_gate(rnd)

    def _on_tree_gate_up(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        child_id = int(msg["node_id"])
        with self._lock:
            rnd = self._gate_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_GATE_REDUCE))
            if not msg.get("ok", True):
                rnd.child_values[child_id] = {"error": str(msg.get("error", "child failed"))}
            else:
                rnd.child_values[child_id] = [int(v) for v in msg["values"]]
        self._try_advance_gate(rnd)

    def _try_advance_gate(self, rnd: _PendingRound) -> None:
        with self._lock:
            if rnd.finished or rnd.sent_up:
                return
            if set(rnd.local_values) != self.expected_local_ranks:
                return
            if set(rnd.child_values) != set(self.children_ids):
                return
            # 汇总全部本地与子树向量
            vectors: list[list[int]] = list(rnd.local_values.values())
            for cv in rnd.child_values.values():
                if isinstance(cv, dict) and "error" in cv:
                    self._finish_gate(rnd, ok=False, error=cv["error"])
                    return
                vectors.append(cv)
            widths = {len(v) for v in vectors}
            if len(widths) != 1:
                self._finish_gate(
                    rnd, ok=False, error=f"各 rank 门闸归约入参宽度不一致: {sorted(widths)}"
                )
                return
            width = next(iter(widths))
            merged = [max(v[i] for v in vectors) for i in range(width)]
            if self.parent_id is not None:
                parent = self._parent_conn
                if parent is None:
                    self._finish_gate(rnd, ok=False, error="父节点已断连")
                    return
                rnd.sent_up = True
                up_msg = {
                    "op": _OP_TREE_GATE_UP,
                    "seq": rnd.seq,
                    "node_id": self.node_id,
                    "ok": True,
                    "values": merged,
                }
            else:
                up_msg = None
        if up_msg is not None:
            assert parent is not None
            if not self._send_conn(parent, up_msg):
                with self._lock:
                    self._finish_gate(rnd, ok=False, error="发送到父节点失败")
        else:
            with self._lock:
                self._finish_gate(rnd, ok=True, values=merged)

    def _on_tree_gate_down(self, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        with self._lock:
            rnd = self._gate_rounds.get(seq)
            if rnd is None or rnd.finished:
                return
            if msg.get("ok", True):
                self._finish_gate(rnd, ok=True, values=[int(v) for v in msg["values"]])
            else:
                self._finish_gate(rnd, ok=False, error=str(msg.get("error", "gate failed")))

    def _finish_gate(
        self,
        rnd: _PendingRound,
        *,
        ok: bool,
        values: list[int] | None = None,
        error: str = "",
    ) -> None:
        if rnd.finished:
            return
        rnd.finished = True
        self._gate_rounds.pop(rnd.seq, None)
        down_msg = {
            "op": _OP_TREE_GATE_DOWN,
            "seq": rnd.seq,
            "ok": ok,
            "values": values or [],
            "error": error,
        }
        for child in self._child_conns.values():
            self._send_conn(child, down_msg)
        for _rank, (client, req_id) in rnd.local_clients.items():
            self._send_conn(
                client,
                {
                    "op": _OP_GATE_RESP,
                    "req_id": req_id,
                    "ok": ok,
                    "values": values or [],
                    "error": error,
                },
            )

    # --- 2. 树形布局预去重与覆盖审计 ----------------------------------------

    def _on_local_audit(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        rank = int(conn.rank if conn.rank is not None else msg["rank"])
        req_id = str(msg["req_id"])
        try:
            summary = build_local_audit_summary(
                msg.get("sources", ()), blob_names=msg.get("blobs", ())
            )
            payload: Any = {"ok": True, "summary": summary}
        except PlanError as exc:
            payload = {"ok": False, "error": str(exc)}
        with self._lock:
            rnd = self._audit_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_PLAN_AUDIT))
            rnd.local_values[rank] = payload
            rnd.local_clients[rank] = (conn, req_id)
        self._try_advance_audit(rnd)

    def _on_tree_audit_up(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        child_id = int(msg["node_id"])
        with self._lock:
            rnd = self._audit_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_PLAN_AUDIT))
            if msg.get("ok", True):
                rnd.child_values[child_id] = {"ok": True, "summary": msg["summary"]}
            else:
                rnd.child_values[child_id] = {"ok": False, "error": str(msg.get("error", "audit failed"))}
        self._try_advance_audit(rnd)

    def _try_advance_audit(self, rnd: _PendingRound) -> None:
        with self._lock:
            if rnd.finished or rnd.sent_up:
                return
            if set(rnd.local_values) != self.expected_local_ranks:
                return
            if set(rnd.child_values) != set(self.children_ids):
                return

            summaries: list[Mapping[str, Any]] = []
            for item in list(rnd.local_values.values()) + list(rnd.child_values.values()):
                if not item.get("ok", True):
                    self._finish_audit(rnd, ok=False, error=str(item.get("error", "audit failed")))
                    return
                summaries.append(item["summary"])

            try:
                merged = merge_audit_summaries(summaries)
            except PlanError as exc:
                self._finish_audit(rnd, ok=False, error=str(exc))
                return

            if self.parent_id is not None:
                parent = self._parent_conn
                if parent is None:
                    self._finish_audit(rnd, ok=False, error="父节点已断连")
                    return
                rnd.sent_up = True
                up_msg = {
                    "op": _OP_TREE_AUDIT_UP,
                    "seq": rnd.seq,
                    "node_id": self.node_id,
                    "ok": True,
                    "summary": merged,
                }
            else:
                up_msg = None
                try:
                    fp = finalize_audit_summary(merged)
                except PlanError as exc:
                    self._finish_audit(rnd, ok=False, error=str(exc))
                    return
        if up_msg is not None:
            assert parent is not None
            if not self._send_conn(parent, up_msg):
                with self._lock:
                    self._finish_audit(rnd, ok=False, error="发送到父节点失败")
        else:
            with self._lock:
                self._finish_audit(rnd, ok=True, fingerprint=fp)

    def _on_tree_audit_down(self, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        with self._lock:
            rnd = self._audit_rounds.get(seq)
            if rnd is None or rnd.finished:
                return
            if msg.get("ok", True):
                self._finish_audit(
                    rnd, ok=True, fingerprint=str(msg.get("fingerprint", ""))
                )
            else:
                self._finish_audit(
                    rnd, ok=False, error=str(msg.get("error", "audit failed"))
                )

    def _finish_audit(
        self,
        rnd: _PendingRound,
        *,
        ok: bool,
        fingerprint: str = "",
        error: str = "",
    ) -> None:
        if rnd.finished:
            return
        rnd.finished = True
        self._audit_rounds.pop(rnd.seq, None)
        down_msg = {
            "op": _OP_TREE_AUDIT_DOWN,
            "seq": rnd.seq,
            "ok": ok,
            "fingerprint": fingerprint,
            "error": error,
        }
        for child in self._child_conns.values():
            self._send_conn(child, down_msg)
        for _rank, (client, req_id) in rnd.local_clients.items():
            self._send_conn(
                client,
                {
                    "op": _OP_PLAN_AUDIT_RESP,
                    "req_id": req_id,
                    "ok": ok,
                    "fingerprint": fingerprint,
                    "error": error,
                },
            )

    # --- 3. 树形强同步广播（恢复选版用） ------------------------------------

    def _on_local_bcast(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        rank = int(conn.rank if conn.rank is not None else msg["rank"])
        req_id = str(msg["req_id"])
        is_root_rank = rank == int(msg.get("root", 0))
        with self._lock:
            rnd = self._bcast_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_TREE_BCAST))
            rnd.local_values[rank] = msg.get("payload") if is_root_rank else None
            rnd.local_clients[rank] = (conn, req_id)
        self._try_advance_bcast(rnd)

    def _on_tree_bcast_up(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        child_id = int(msg["node_id"])
        with self._lock:
            rnd = self._bcast_rounds.setdefault(seq, _PendingRound(seq=seq, op=_OP_TREE_BCAST))
            rnd.child_values[child_id] = msg.get("payload")
        self._try_advance_bcast(rnd)

    def _try_advance_bcast(self, rnd: _PendingRound) -> None:
        with self._lock:
            if rnd.finished or rnd.sent_up:
                return
            if set(rnd.local_values) != self.expected_local_ranks:
                return
            if set(rnd.child_values) != set(self.children_ids):
                return
            payload = next((v for v in rnd.local_values.values() if v is not None), None)
            if payload is None:
                payload = next((v for v in rnd.child_values.values() if v is not None), None)
            if self.parent_id is not None:
                parent = self._parent_conn
                if parent is None:
                    self._finish_bcast(rnd, ok=False, error="父节点已断连")
                    return
                rnd.sent_up = True
                up_msg = {
                    "op": _OP_TREE_BCAST_UP,
                    "seq": rnd.seq,
                    "node_id": self.node_id,
                    "payload": payload,
                }
            else:
                up_msg = None
        if up_msg is not None:
            assert parent is not None
            if not self._send_conn(parent, up_msg):
                with self._lock:
                    self._finish_bcast(rnd, ok=False, error="发送到父节点失败")
        else:
            with self._lock:
                self._finish_bcast(rnd, ok=True, payload=payload)

    def _on_tree_bcast_down(self, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        with self._lock:
            rnd = self._bcast_rounds.get(seq)
            if rnd is None or rnd.finished:
                return
            if msg.get("ok", True):
                self._finish_bcast(rnd, ok=True, payload=msg.get("payload"))
            else:
                self._finish_bcast(rnd, ok=False, error=str(msg.get("error", "bcast failed")))

    def _finish_bcast(
        self,
        rnd: _PendingRound,
        *,
        ok: bool,
        payload: Any = None,
        error: str = "",
    ) -> None:
        if rnd.finished:
            return
        rnd.finished = True
        self._bcast_rounds.pop(rnd.seq, None)
        down_msg = {
            "op": _OP_TREE_BCAST_DOWN,
            "seq": rnd.seq,
            "ok": ok,
            "payload": payload,
            "error": error,
        }
        for child in self._child_conns.values():
            self._send_conn(child, down_msg)
        for _rank, (client, req_id) in rnd.local_clients.items():
            self._send_conn(
                client,
                {
                    "op": _OP_TREE_BCAST_RESP,
                    "req_id": req_id,
                    "ok": ok,
                    "payload": payload,
                    "error": error,
                },
            )

    # --- 4. 存储侧退出收敛屏障 ----------------------------------------------

    def _on_local_barrier(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        rank = int(conn.rank if conn.rank is not None else msg["rank"])
        req_id = str(msg["req_id"])
        with self._lock:
            rnd = self._barrier_rounds.setdefault(
                seq, _PendingRound(seq=seq, op=_OP_STORAGE_BARRIER)
            )
            rnd.local_values[rank] = dict(msg.get("data", {}))
            rnd.local_clients[rank] = (conn, req_id)
        self._try_advance_barrier(rnd)

    def _on_tree_barrier_up(self, conn: _ConnState, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        child_id = int(msg["node_id"])
        with self._lock:
            rnd = self._barrier_rounds.setdefault(
                seq, _PendingRound(seq=seq, op=_OP_STORAGE_BARRIER)
            )
            rnd.child_values[child_id] = dict(msg.get("data", {}))
        self._try_advance_barrier(rnd)

    def _try_advance_barrier(self, rnd: _PendingRound) -> None:
        with self._lock:
            if rnd.finished or rnd.sent_up:
                return
            if set(rnd.local_values) != self.expected_local_ranks:
                return
            if set(rnd.child_values) != set(self.children_ids):
                return
            errors: list[str] = []
            for v in list(rnd.local_values.values()) + list(rnd.child_values.values()):
                errors.extend(v.get("errors", ()))
            data = {"errors": errors}
            if self.parent_id is not None:
                parent = self._parent_conn
                if parent is None:
                    self._finish_barrier(rnd, ok=False, error="父节点已断连")
                    return
                rnd.sent_up = True
                up_msg = {
                    "op": _OP_TREE_BARRIER_UP,
                    "seq": rnd.seq,
                    "node_id": self.node_id,
                    "data": data,
                }
            else:
                up_msg = None
        if up_msg is not None:
            assert parent is not None
            if not self._send_conn(parent, up_msg):
                with self._lock:
                    self._finish_barrier(rnd, ok=False, error="发送到父节点失败")
        else:
            with self._lock:
                self._finish_barrier(rnd, ok=True, data=data)

    def _on_tree_barrier_down(self, msg: dict[str, Any]) -> None:
        seq = int(msg["seq"])
        with self._lock:
            rnd = self._barrier_rounds.get(seq)
            if rnd is None or rnd.finished:
                return
            self._finish_barrier(
                rnd,
                ok=bool(msg.get("ok", True)),
                data=msg.get("data"),
                error=str(msg.get("error", "")),
            )

    def _finish_barrier(
        self,
        rnd: _PendingRound,
        *,
        ok: bool,
        data: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> None:
        if rnd.finished:
            return
        rnd.finished = True
        self._barrier_rounds.pop(rnd.seq, None)
        down_msg = {
            "op": _OP_TREE_BARRIER_DOWN,
            "seq": rnd.seq,
            "ok": ok,
            "data": dict(data or {}),
            "error": error,
        }
        for child in self._child_conns.values():
            self._send_conn(child, down_msg)
        for _rank, (client, req_id) in rnd.local_clients.items():
            self._send_conn(
                client,
                {
                    "op": _OP_STORAGE_BARRIER_RESP,
                    "req_id": req_id,
                    "ok": ok,
                    "data": dict(data or {}),
                    "error": error,
                },
            )

    def _cleanup_sockets(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
            self._local_clients.clear()
            self._child_conns.clear()
            self._parent_conn = None
        for conn in conns:
            try:
                conn.sock.close()
            except OSError:
                pass
        for sock in (self._local_srv, self._ext_srv, self._wakeup_r, self._wakeup_w):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self._bound_local_address.startswith("unix://"):
            Path(self._bound_local_address[len("unix://") :]).unlink(missing_ok=True)
        self._local_srv = None
        self._ext_srv = None


# --- 客户端（训练进程 / 存储进程接入点） -----------------------------------


class NodeCommClient:
    """连接本机 ``NodeCommServer`` 的客户端句柄（实现 ``GateCollective`` 语义）。"""

    def __init__(
        self,
        endpoint: CommEndpoint,
        *,
        rank: int,
        role: str = ROLE_TRAINER,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if role not in (ROLE_TRAINER, ROLE_STORAGE):
            raise ValueError(f"role must be trainer or storage, got {role!r}")
        self.endpoint = endpoint
        self.rank = int(rank)
        self.world_size = int(endpoint.world_size)
        self.role = role
        self._on_event = on_event

        self._sock = _connect_address(endpoint.local_address, timeout=endpoint.timeout)
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._gate_seq = 0
        self._audit_seq = 0
        self._bcast_seq = 0
        self._barrier_seq = 0
        self._closed = False

        send_msg(self._sock, {"op": _OP_REGISTER, "rank": self.rank, "role": self.role})
        ack = recv_msg(self._sock)
        if ack is None or ack[0].get("op") != _OP_REGISTER_ACK:
            self._sock.close()
            raise CommError(f"rank {self.rank}: 注册到 NodeCommServer 失败")

        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"comm-client-{self.role}-r{self.rank}",
            daemon=True,
        )
        self._reader.start()

    def set_event_handler(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_event = handler

    def poll_events(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._fail_pending("NodeCommClient closed")
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)

    def __enter__(self) -> "NodeCommClient":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    # --- 强同步原语 ---------------------------------------------------------

    def all_reduce_max_int64(
        self, values: list[int], *, timeout: float = DEFAULT_TIMEOUT
    ) -> list[int]:
        self._gate_seq += 1
        resp = self._rpc(
            {
                "op": _OP_GATE_REDUCE,
                "seq": self._gate_seq,
                "rank": self.rank,
                "values": [int(v) for v in values],
            },
            timeout=timeout,
        )
        if not resp.get("ok", True):
            raise CommError(str(resp.get("error", "GATE_REDUCE failed")))
        return [int(v) for v in resp["values"]]

    def audit_save_plan(
        self,
        sources: Sequence[ShardSource],
        *,
        blob_names: Sequence[str] = (),
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str:
        """``gathered`` 模式树形预去重覆盖审计，返回主节点确认指纹。"""
        self._audit_seq += 1
        resp = self._rpc(
            {
                "op": _OP_PLAN_AUDIT,
                "seq": self._audit_seq,
                "rank": self.rank,
                "sources": [source_to_audit_dict(s) for s in sources],
                "blobs": [str(b) for b in blob_names],
            },
            timeout=timeout,
        )
        if not resp.get("ok", True):
            raise PlanError(str(resp.get("error", "PLAN_AUDIT failed")))
        return str(resp.get("fingerprint", ""))

    def broadcast_object(
        self, obj: Any, *, root: int = 0, timeout: float = DEFAULT_TIMEOUT
    ) -> Any:
        """树形强同步广播（用于恢复选版广播等小对象同步）。"""
        self._bcast_seq += 1
        encoded = _encode_bcast_payload(obj) if self.rank == root else None
        resp = self._rpc(
            {
                "op": _OP_TREE_BCAST,
                "seq": self._bcast_seq,
                "rank": self.rank,
                "root": int(root),
                "payload": encoded,
            },
            timeout=timeout,
        )
        if not resp.get("ok", True):
            raise CommError(str(resp.get("error", "TREE_BCAST failed")))
        return _decode_bcast_payload(resp.get("payload"))

    def storage_barrier(
        self, *, errors: Sequence[str] = (), timeout: float = DEFAULT_TIMEOUT
    ) -> list[str]:
        """存储侧全集群收敛屏障，返回全集群上报的错误摘要。"""
        self._barrier_seq += 1
        resp = self._rpc(
            {
                "op": _OP_STORAGE_BARRIER,
                "seq": self._barrier_seq,
                "rank": self.rank,
                "data": {"errors": [str(e) for e in errors]},
            },
            timeout=timeout,
        )
        if not resp.get("ok", True):
            raise CommError(str(resp.get("error", "STORAGE_BARRIER failed")))
        return list(resp.get("data", {}).get("errors", ()))

    # --- 异步与事件广播原语 -------------------------------------------------

    def broadcast_generation(
        self,
        *,
        run_id: str | None,
        step: int,
        generation: str,
        seq: int,
        writer: str,
        ranks: Sequence[int],
    ) -> None:
        self._send(
            {
                "op": OP_GEN_BROADCAST,
                "run_id": run_id,
                "step": int(step),
                "generation": generation,
                "seq": int(seq),
                "writer": writer,
                "ranks": [int(r) for r in ranks],
            }
        )

    def broadcast_commit(
        self, *, run_id: str | None, step: int, generation: str
    ) -> None:
        self._send(
            {
                "op": OP_COMMIT_BROADCAST,
                "run_id": run_id,
                "step": int(step),
                "generation": generation,
            }
        )

    def broadcast_invalidation(
        self,
        *,
        run_id: str | None,
        step: int,
        generation: str | None,
        reason: str,
    ) -> None:
        self._send(
            {
                "op": OP_INVALIDATE_EVENT,
                "run_id": run_id,
                "step": int(step),
                "generation": generation,
                "reason": reason,
                "source_rank": self.rank,
            }
        )

    def broadcast_protect(
        self,
        *,
        add_steps: Sequence[int] = (),
        release_steps: Sequence[int] = (),
        pause_gc: bool | None = None,
    ) -> None:
        self._send(
            {
                "op": OP_PROTECT_BROADCAST,
                "add_steps": [int(s) for s in add_steps],
                "release_steps": [int(s) for s in release_steps],
                "pause_gc": pause_gc,
            }
        )

    # --- 内部 RPC 与接收线程 ------------------------------------------------

    def _send(self, msg: Mapping[str, Any]) -> None:
        with self._send_lock:
            if self._closed:
                raise CommError("NodeCommClient is closed")
            try:
                send_msg(self._sock, msg)
            except OSError as exc:
                raise CommError(f"发送消息失败: {exc}") from exc

    def _rpc(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        req_id = uuid.uuid4().hex
        payload["req_id"] = req_id
        ev = threading.Event()
        holder: dict[str, Any] = {}
        with self._pending_lock:
            if self._closed:
                raise CommError("NodeCommClient is closed")
            self._pending[req_id] = (ev, holder)
        try:
            self._send(payload)
        except BaseException:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise
        if not ev.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise CommError(f"{payload['op']} 超时（{timeout}s）")
        if "comm_error" in holder:
            raise CommError(holder["comm_error"])
        return holder["resp"]

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            items = list(self._pending.values())
            self._pending.clear()
        for ev, holder in items:
            holder["comm_error"] = reason
            ev.set()

    def _read_loop(self) -> None:
        while not self._closed:
            try:
                res = recv_msg(self._sock)
            except (OSError, EOFError, ProtocolError):
                res = None
            if res is None:
                self._fail_pending("NodeCommServer connection closed")
                return
            msg, _fds = res
            req_id = msg.get("req_id")
            if req_id is not None:
                with self._pending_lock:
                    slot = self._pending.pop(str(req_id), None)
                if slot is not None:
                    ev, holder = slot
                    holder["resp"] = msg
                    ev.set()
                    continue
            self._events.put(msg)
            if self._on_event is not None:
                try:
                    self._on_event(msg)
                except Exception:
                    pass


def _encode_bcast_payload(value: Any) -> Any:
    if isinstance(value, BaseException):
        return {
            "__ckpt_exc__": {
                "type": type(value).__name__,
                "message": str(value),
            }
        }
    if isinstance(value, dict):
        return {str(k): _encode_bcast_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_bcast_payload(v) for v in value]
    return value


def _decode_bcast_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ckpt_exc__" in value and len(value) == 1:
            info = value["__ckpt_exc__"]
            ename = str(info.get("type", "CheckpointError"))
            emsg = str(info.get("message", ""))
            from core import CheckpointError, ChecksumError, FormatError, InvalidTargetError
            from storage.base import StorageError

            exc_map: dict[str, type[BaseException]] = {
                "FormatError": FormatError,
                "PlanError": PlanError,
                "ChecksumError": ChecksumError,
                "InvalidTargetError": InvalidTargetError,
                "StorageError": StorageError,
                "CheckpointError": CheckpointError,
                "ValueError": ValueError,
                "TimeoutError": TimeoutError,
            }
            cls = exc_map.get(ename, CheckpointError)
            return cls(emsg)
        return {k: _decode_bcast_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_bcast_payload(v) for v in value]
    return value


class NodeComm:
    """便捷封装：启动单机节点通信服务并按 rank 分发 ``NodeCommClient``。"""

    def __init__(
        self,
        *,
        world_size: int = 1,
        base_dir: str | Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        cfg = make_tree_configs(
            num_nodes=1,
            ranks_per_node=world_size,
            base_dir=base_dir,
        )[0]
        self._server = NodeCommServer(cfg).start()
        self._timeout = float(timeout)
        self._clients: dict[tuple[int, str], NodeCommClient] = {}

    @property
    def endpoint(self) -> CommEndpoint:
        ep = self._server.endpoint
        return CommEndpoint(
            local_address=ep.local_address,
            node_id=ep.node_id,
            num_nodes=ep.num_nodes,
            world_size=ep.world_size,
            timeout=self._timeout,
        )

    def client(self, rank: int = 0, *, role: str = ROLE_TRAINER) -> NodeCommClient:
        key = (int(rank), role)
        cli = self._clients.get(key)
        if cli is None:
            cli = NodeCommClient(self.endpoint, rank=rank, role=role)
            self._clients[key] = cli
        return cli

    def close(self) -> None:
        for cli in list(self._clients.values()):
            cli.close()
        self._clients.clear()
        self._server.close()

    def __enter__(self) -> "NodeComm":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


def _comm_process_main(
    config: NodeTreeConfig, ready_conn: Any, stop_event: Any
) -> None:  # pragma: no cover - 跨进程分支
    server = NodeCommServer(config).start()
    ready_conn.send(server.endpoint)
    ready_conn.close()
    try:
        stop_event.wait()
    finally:
        server.close()


class NodeCommProcess:
    """独立纯 CPU 进程形态的节点通信转发器（多机集群真实隔离用）。"""

    def __init__(self, config: NodeTreeConfig) -> None:
        self.config = config
        self._ctx = multiprocessing.get_context("spawn")
        self._stop_event = self._ctx.Event()
        parent_pipe, child_pipe = self._ctx.Pipe(duplex=False)
        self._proc = self._ctx.Process(
            target=_comm_process_main,
            args=(config, child_pipe, self._stop_event),
            name=f"shardckpt-comm-n{config.node_id}",
        )
        self._proc.start()
        child_pipe.close()
        if not parent_pipe.poll(timeout=10.0):
            parent_pipe.close()
            self.close()
            raise CommError(f"node {config.node_id}: NodeCommProcess 启动超时")
        self.endpoint: CommEndpoint = parent_pipe.recv()
        parent_pipe.close()

    def close(self) -> None:
        self._stop_event.set()
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=2.0)

    def __enter__(self) -> "NodeCommProcess":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


__all__ = [
    "CommEndpoint",
    "CommError",
    "NodeComm",
    "NodeCommClient",
    "NodeCommProcess",
    "NodeCommServer",
    "NodeTreeConfig",
    "OP_COMMIT_BROADCAST",
    "OP_GEN_BROADCAST",
    "OP_INVALIDATE_EVENT",
    "OP_PROTECT_BROADCAST",
    "ROLE_STORAGE",
    "ROLE_TRAINER",
    "build_local_audit_summary",
    "children_of",
    "finalize_audit_summary",
    "make_tree_configs",
    "merge_audit_summaries",
    "parent_of",
]
