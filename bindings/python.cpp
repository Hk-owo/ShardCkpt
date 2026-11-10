// Python 绑定：把 ckpt_core 的本地读写能力包装成 ckpt_native 扩展模块。
//
// 绑定层只做四件事，其余全部交给 C++：
//   1. Status -> 项目异常的映射（上层永远见不到 ErrorCode）
//   2. 阻塞 IO 期间释放 GIL（否则后台写线程会和训练线程抢 GIL）
//   3. host buffer 零拷贝：直接从 Python buffer 协议取指针交给 C++
//   4. 线程归属检查：io_uring / LocalFile 只允许在创建它的线程上使用
//
// 数据生命周期：LocalFile::WriteAt 是零拷贝的（Chunk 的 data 在 Sync() 前必须
// 有效），所以每次 write_at 都会把对应的 Python 对象记在 keepalive 里，Sync()
// 之后才放开，避免 Python 侧提前释放。

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ckpt/error.hpp"
#include "ckpt/storage/local_file.hpp"

namespace py = pybind11;

namespace {

using ckpt::ErrorCode;
using ckpt::Status;
using ckpt::storage::AlignedBuffer;
using ckpt::storage::Chunk;
using ckpt::storage::LocalFile;
using ckpt::storage::LocalFileOptions;
using ckpt::storage::WriteOptions;

// 项目异常类型，加载扩展时导入一次；导入失败则回退到内置异常，保证扩展本身
// 在没有项目 Python 包的环境里也能加载。
py::object g_storage_error;
py::object g_not_found_error;
py::object g_format_error;

py::object ExceptionFor(ErrorCode code) {
  switch (code) {
    case ErrorCode::kInvalidArgument:
      return py::reinterpret_borrow<py::object>(PyExc_ValueError);
    case ErrorCode::kNotFound:
      if (g_not_found_error) return g_not_found_error;
      return py::reinterpret_borrow<py::object>(PyExc_FileNotFoundError);
    case ErrorCode::kFormatError:
      if (g_format_error) return g_format_error;
      return py::reinterpret_borrow<py::object>(PyExc_ValueError);
    case ErrorCode::kUnsupported:
    case ErrorCode::kNotImplemented:
      return py::reinterpret_borrow<py::object>(PyExc_NotImplementedError);
    case ErrorCode::kAuthError:
    case ErrorCode::kNetworkError:
    case ErrorCode::kIoError:
    case ErrorCode::kOk:
      break;
  }
  if (g_storage_error) return g_storage_error;
  return py::reinterpret_borrow<py::object>(PyExc_OSError);
}

[[noreturn]] void RaiseStatus(const Status& status) {
  py::object type = ExceptionFor(status.code());
  std::string message = status.ToString();
  PyErr_SetString(type.ptr(), message.c_str());
  throw py::error_already_set();
}

void Check(const Status& status) {
  if (!status.ok()) RaiseStatus(status);
}

// 从 Python 的 buffer 协议取一段连续字节；零拷贝。
class SimpleBuffer {
 public:
  explicit SimpleBuffer(py::handle object) : held_(false) {
    if (PyObject_GetBuffer(object.ptr(), &view_, PyBUF_SIMPLE) < 0) {
      throw py::error_already_set();
    }
    held_ = true;
  }

  SimpleBuffer(const SimpleBuffer&) = delete;
  SimpleBuffer& operator=(const SimpleBuffer&) = delete;

  ~SimpleBuffer() {
    if (held_) PyBuffer_Release(&view_);
  }

  std::span<const std::byte> bytes() const {
    return {static_cast<const std::byte*>(view_.buf), static_cast<std::size_t>(view_.len)};
  }

 private:
  Py_buffer view_{};
  bool held_;
};

// LocalFile 的 Python 句柄。绑定层额外记住创建它的线程：io_uring 的 SQ/CQ 不支持
// 跨线程共享，误用要立刻报错，而不是留给用户踩未定义行为。
class PyLocalFile {
 public:
  static std::shared_ptr<PyLocalFile> OpenWrite(const std::string& path, bool direct_io,
                                                unsigned alignment, unsigned chunk_bytes,
                                                unsigned depth, bool fsync_on_sync) {
    LocalFileOptions options;
    options.direct_io = direct_io;
    options.alignment = alignment;
    options.chunk_bytes = chunk_bytes;
    options.depth = depth;
    options.fsync_on_sync = fsync_on_sync;

    LocalFile* file = nullptr;
    Status status;
    {
      py::gil_scoped_release release;
      status = LocalFile::OpenWrite(path, options, &file);
    }
    Check(status);
    return std::shared_ptr<PyLocalFile>(new PyLocalFile(path, file));
  }

  static std::shared_ptr<PyLocalFile> OpenRead(const std::string& path, bool direct_io,
                                               unsigned alignment) {
    LocalFileOptions options;
    options.direct_io = direct_io;
    options.alignment = alignment;

    LocalFile* file = nullptr;
    Status status;
    {
      py::gil_scoped_release release;
      status = LocalFile::OpenRead(path, options, &file);
    }
    Check(status);
    return std::shared_ptr<PyLocalFile>(new PyLocalFile(path, file));
  }

  int fd() const { return file_ ? file_->fd() : -1; }

  const std::string& path() const { return path_; }

  void WriteAt(std::uint64_t offset, py::object data) {
    EnsureOwnerThread();
    SimpleBuffer buffer(data);
    auto bytes = buffer.bytes();

    Status status;
    {
      py::gil_scoped_release release;
      status = file_->WriteAt(offset, bytes);
    }
    Check(status);
    // 零拷贝：C++ 侧没有复制这段内存，Sync() 之前不能让它被回收
    keepalive_.push_back(std::move(data));
  }

  void Sync(bool fsync) {
    EnsureOwnerThread();
    Status status;
    {
      py::gil_scoped_release release;
      status = file_->Sync(fsync);
    }
    Check(status);
    keepalive_.clear();
  }

  void Close() {
    EnsureOwnerThread();
    if (!file_) return;
    Status status;
    {
      py::gil_scoped_release release;
      status = file_->Close();
    }
    keepalive_.clear();
    file_.reset();
    Check(status);
  }

 private:
  PyLocalFile(std::string path, LocalFile* file)
      : path_(std::move(path)), file_(file), owner_(std::this_thread::get_id()) {}

  void EnsureOwnerThread() const {
    if (std::this_thread::get_id() != owner_) {
      // io_uring 的 SQ/CQ 不支持跨线程共享：误用要立刻报错，而不是留给用户踩 UB
      throw std::runtime_error(
          "LocalFile 只能在创建它的线程上使用（io_uring 的 SQ/CQ 不支持跨线程共享）");
    }
  }

  std::string path_;
  std::unique_ptr<LocalFile> file_;
  std::thread::id owner_;
  std::vector<py::object> keepalive_;
};

std::shared_ptr<PyLocalFile> LocalFileOpenWrite(const std::string& path, bool direct_io,
                                               unsigned alignment, unsigned chunk_bytes,
                                               unsigned depth, bool fsync_on_sync) {
  return PyLocalFile::OpenWrite(path, direct_io, alignment, chunk_bytes, depth, fsync_on_sync);
}

void WriteFileBinding(const std::string& path, const py::sequence& chunks, bool direct_io,
                      bool fsync_data, bool fsync_dir, unsigned alignment,
                      unsigned chunk_bytes, unsigned depth) {
  // 用 unique_ptr 持有 Py_buffer guard：vector 扩容不会影响 guard 的地址，也不会
  // 复制出重复的 Py_buffer（那会导致重复 Release）。
  std::vector<std::unique_ptr<SimpleBuffer>> guards;
  std::vector<Chunk> raw;
  guards.reserve(chunks.size());
  raw.reserve(chunks.size());
  for (py::handle item : chunks) {
    guards.push_back(std::make_unique<SimpleBuffer>(item));
    auto bytes = guards.back()->bytes();
    raw.push_back(Chunk{bytes.data(), bytes.size()});
  }

  WriteOptions options;
  options.direct_io = direct_io;
  options.fsync_data = fsync_data;
  options.fsync_dir = fsync_dir;
  options.alignment = alignment;
  options.chunk_bytes = chunk_bytes;
  options.depth = depth;

  Status status;
  {
    py::gil_scoped_release release;
    status = ckpt::storage::WriteFile(path, raw, options);
  }
  Check(status);
}

void RenameAtomicBinding(const std::string& src, const std::string& dst) {
  Status status;
  {
    py::gil_scoped_release release;
    status = ckpt::storage::RenameAtomic(src, dst);
  }
  Check(status);
}

void FsyncDirBinding(const std::string& path) {
  Status status;
  {
    py::gil_scoped_release release;
    status = ckpt::storage::FsyncDir(path);
  }
  Check(status);
}

class PyAlignedBuffer {
 public:
  static std::unique_ptr<PyAlignedBuffer> Allocate(std::size_t bytes, unsigned alignment) {
    AlignedBuffer buffer;
    Status status = AlignedBuffer::Allocate(bytes, alignment, &buffer);
    Check(status);
    return std::unique_ptr<PyAlignedBuffer>(new PyAlignedBuffer(std::move(buffer)));
  }

  std::size_t size() const { return buffer_.size(); }

  // 地址用整数返回：Python 侧据此构造零拷贝的 memoryview（见 native/__init__.py）
  std::uintptr_t address() const { return reinterpret_cast<std::uintptr_t>(buffer_.data()); }

 private:
  explicit PyAlignedBuffer(AlignedBuffer buffer) : buffer_(std::move(buffer)) {}

  AlignedBuffer buffer_;
};

}  // namespace

PYBIND11_MODULE(ckpt_native, m) {
  m.doc() = "ShardCkpt 的 C++ 内核：本地文件读写（io_uring）、原子 rename、对齐 buffer";

  // 项目异常：语义化映射，Python 上层不需要认识 ErrorCode
  try {
    py::object base = py::module_::import("storage.base");
    g_storage_error = base.attr("StorageError");
    g_not_found_error = base.attr("ObjectNotFoundError");
    g_format_error = py::module_::import("core").attr("FormatError");
  } catch (const py::error_already_set&) {
    PyErr_Clear();  // 没有项目包时退回内置异常
  }

  m.attr("DEFAULT_ALIGNMENT") = ckpt::storage::kDefaultAlignment;
  m.attr("DEFAULT_CHUNK_BYTES") = ckpt::storage::kDefaultChunkBytes;
  m.attr("DEFAULT_DEPTH") = ckpt::storage::kDefaultDepth;

  m.def("write_file", &WriteFileBinding, py::arg("path"), py::arg("chunks"),
        py::arg("direct_io") = true, py::arg("fsync_data") = true, py::arg("fsync_dir") = true,
        py::arg("alignment") = ckpt::storage::kDefaultAlignment,
        py::arg("chunk_bytes") = ckpt::storage::kDefaultChunkBytes,
        py::arg("depth") = ckpt::storage::kDefaultDepth,
        "整文件顺序写：临时文件 -> WRITE -> fsync -> rename -> 目录 fsync。chunks 里的每个"
        " buffer 都是零拷贝。");

  m.def("rename_atomic", &RenameAtomicBinding, py::arg("src"), py::arg("dst"),
        "rename(2)；同路径直接成功，跨文件系统失败。");
  m.def("fsync_dir", &FsyncDirBinding, py::arg("path"), "fsync 目录项（rename 之后调用）。");

  py::class_<PyLocalFile, std::shared_ptr<PyLocalFile>>(m, "LocalFile")
      .def_static("open_write", &LocalFileOpenWrite, py::arg("path"), py::arg("direct_io") = true,
                  py::arg("alignment") = ckpt::storage::kDefaultAlignment,
                  py::arg("chunk_bytes") = ckpt::storage::kDefaultChunkBytes,
                  py::arg("depth") = ckpt::storage::kDefaultDepth,
                  py::arg("fsync_on_sync") = true)
      .def_static("open_read", &PyLocalFile::OpenRead, py::arg("path"),
                  py::arg("direct_io") = false,
                  py::arg("alignment") = ckpt::storage::kDefaultAlignment)
      .def("write_at", &PyLocalFile::WriteAt, py::arg("offset"), py::arg("data"),
           "顺序追加一段；offset 必须等于上次写到的位置。数据零拷贝，Sync() 前有效。")
      .def("sync", &PyLocalFile::Sync, py::arg("fsync") = true, "收完在途写，可选 fsync。")
      .def("close", &PyLocalFile::Close)
      .def("fileno", &PyLocalFile::fd)
      .def_property_readonly("path", &PyLocalFile::path);

  py::class_<PyAlignedBuffer>(m, "AlignedBuffer")
      .def_static("allocate", &PyAlignedBuffer::Allocate, py::arg("bytes"),
                  py::arg("alignment") = ckpt::storage::kDefaultAlignment,
                  "按 alignment 对齐分配；O_DIRECT 写需要对齐的 buffer。")
      .def_property_readonly("size", &PyAlignedBuffer::size)
      .def_property_readonly("address", &PyAlignedBuffer::address);
}
