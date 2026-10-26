"""读取接口测试：设备语义、数据一致性、错误分支。"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import ckpt


def manifest_of(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_load_cpu_marks_tensors_on_cpu(dataset: Path) -> None:
    data = ckpt.load(dataset, device="cpu")
    assert len(data) > 0
    for tensor in data.tensors.values():
        assert tensor.device == ckpt.CPU
        assert str(tensor.device) == "cpu"


def test_load_cuda_goes_through_h2d_path(dataset: Path) -> None:
    """无 GPU 时 cuda 走 mock backend，但读取路径与真机一致：每个张量一次 H2D。"""
    manifest = manifest_of(dataset)
    data = ckpt.load(dataset, device="cuda:0")

    backend = ckpt.get_backend(ckpt.Device.parse("cuda:0"))
    assert backend.is_mock, "当前环境应使用 mock cuda backend"
    assert backend.stats.host_to_device_copies == len(manifest["tensors"])
    assert backend.stats.host_to_device_bytes == manifest["total_bytes"]

    for tensor in data.tensors.values():
        assert tensor.device == ckpt.Device("cuda", 0)
        assert str(tensor.device) == "cuda:0"


def test_bytes_match_source_files(dataset: Path) -> None:
    """读回来的字节必须与磁盘上的 bin 完全一致（含 bf16 这类原始位模式）。"""
    manifest = manifest_of(dataset)
    data = ckpt.load(dataset, device="cuda:0")
    for entry in manifest["tensors"]:
        expected = (dataset / entry["file"]).read_bytes()
        tensor = data[entry["name"]]
        assert tensor.tobytes() == expected, entry["name"]
        assert tensor.crc32() == entry["crc32"], entry["name"]


def test_cpu_and_cuda_values_are_identical(dataset: Path) -> None:
    cpu = ckpt.load(dataset, device="cpu")
    gpu = ckpt.load(dataset, device="cuda:0")
    assert cpu.names() == gpu.names()
    for name in cpu.names():
        np.testing.assert_array_equal(cpu[name].numpy(), gpu[name].numpy())


def test_shape_dtype_metadata(dataset: Path) -> None:
    manifest = manifest_of(dataset)
    data = ckpt.load(dataset)
    for entry in manifest["tensors"]:
        tensor = data[entry["name"]]
        assert tensor.dtype == entry["dtype"]
        assert tensor.shape == tuple(entry["shape"])
        assert tensor.nbytes == entry["nbytes"]
        assert tensor.numel == int(np.prod(entry["shape"], dtype=np.int64))


def test_read_single_tensor(dataset: Path) -> None:
    reader = ckpt.CheckpointReader.open(dataset, device="cuda:1")
    name = reader.names()[0]
    tensor = reader.read(name)
    assert tensor.name == name
    assert str(tensor.device) == "cuda:1"


def test_missing_tensor_raises(dataset: Path) -> None:
    reader = ckpt.CheckpointReader.open(dataset)
    with pytest.raises(ckpt.MissingTensorError):
        reader.read("model.does_not_exist.weight")


def test_corrupted_payload_detected_by_crc(dataset: Path) -> None:
    manifest = manifest_of(dataset)
    entry = manifest["tensors"][0]
    path = dataset / entry["file"]
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0xFF
    path.write_bytes(payload)

    reader = ckpt.CheckpointReader.open(dataset, verify_crc=True)
    with pytest.raises(ckpt.ChecksumError):
        reader.read(entry["name"])

    # 关掉校验就不再报错，说明失败确实来自 CRC 而非读文件
    lenient = ckpt.CheckpointReader.open(dataset, verify_crc=False)
    assert lenient.read(entry["name"]).nbytes == entry["nbytes"]


def test_truncated_file_is_format_error(dataset: Path) -> None:
    manifest = manifest_of(dataset)
    entry = manifest["tensors"][0]
    path = dataset / entry["file"]
    path.write_bytes(path.read_bytes()[:-4])

    reader = ckpt.CheckpointReader.open(dataset)
    with pytest.raises(ckpt.FormatError):
        reader.read(entry["name"])


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ckpt.FormatError):
        ckpt.load(tmp_path)


def test_unknown_device_spec() -> None:
    with pytest.raises(ValueError):
        ckpt.Device.parse("tpu:0")
    with pytest.raises(ValueError):
        ckpt.Device.parse("cuda:-1")


def test_cuda_index_out_of_range(dataset: Path) -> None:
    with pytest.raises(ckpt.UnsupportedDeviceError):
        ckpt.load(dataset, device="cuda:99")


def test_to_cpu_moves_data(dataset: Path) -> None:
    gpu = ckpt.load(dataset, device="cuda:0")
    cpu = gpu.to("cpu")
    assert cpu.device == ckpt.CPU
    for name in gpu.names():
        assert cpu[name].device == ckpt.CPU
        np.testing.assert_array_equal(cpu[name].numpy(), gpu[name].numpy())


def test_staging_off_gives_same_bytes(dataset: Path) -> None:
    staged = ckpt.load(dataset, device="cuda:0", staging=True)
    direct = ckpt.load(dataset, device="cuda:0", staging=False)
    for name in staged.names():
        assert staged[name].tobytes() == direct[name].tobytes()


def test_bf16_dataset(tmp_path: Path) -> None:
    import make_data

    out = tmp_path / "bf16"
    make_data.write_dataset(out, seed=5, scale="tiny", dtype="bf16")
    data = ckpt.load(out, device="cuda:0")
    for tensor in data.tensors.values():
        assert tensor.dtype == "bf16"
        assert tensor.nbytes == tensor.numel * 2


def test_sharded_rank_can_be_loaded(tmp_path: Path) -> None:
    import make_data

    for rank in (0, 1):
        out = tmp_path / f"rank{rank}"
        make_data.write_dataset(out, seed=9, scale="tiny", world_size=2, rank=rank)
        data = ckpt.load(out, device="cuda:0")
        assert len(data) > 0
        assert any(name.startswith("optimizer.") for name in data.names())


def test_backend_is_replaceable(dataset: Path) -> None:
    """接口只依赖 DeviceBackend：换一个实现即可改变底层行为，调用方不变。"""

    class RecordingCuda(ckpt.MockCudaBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[int] = []

        def copy_h2d(self, dst, src) -> None:
            self.calls.append(len(src))
            super().copy_h2d(dst, src)

    backend = RecordingCuda()
    ckpt.register_backend(backend)

    data = ckpt.load(dataset, device="cuda:0")
    assert len(backend.calls) == len(data)
    assert sum(backend.calls) == manifest_of(dataset)["total_bytes"]


def test_checkpoint_data_api(dataset: Path) -> None:
    data = ckpt.load(dataset)
    first = data.names()[0]
    assert first in data
    assert data[first].name == first
    assert len(list(data.items())) == len(data)
    assert data.meta["seed"] == 1234
    assert "tensors" not in data.meta


def test_crc32_matches_zlib(dataset: Path) -> None:
    manifest = manifest_of(dataset)
    entry = manifest["tensors"][0]
    payload = (dataset / entry["file"]).read_bytes()
    assert entry["crc32"] == zlib.crc32(payload) & 0xFFFFFFFF
