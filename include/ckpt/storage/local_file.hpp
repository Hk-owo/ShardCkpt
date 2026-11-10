#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ckpt/error.hpp"
#include "ckpt/io/uring.hpp"

namespace ckpt::storage {

inline constexpr unsigned kDefaultAlignment = 4096;
inline constexpr unsigned kDefaultChunkBytes = 8u << 20;
inline constexpr unsigned kDefaultDepth = 32;

// 本地文件的打开选项。
//
// direct_io 默认关闭：随机读的区间是张量在文件里的偏移，基本不按块对齐，
// 开 O_DIRECT 会把每次读放大到块边界并失去 page cache；大文件顺序写则建议
// 打开（见 WriteOptions::direct_io）。
struct LocalFileOptions {
  bool direct_io = false;
  unsigned alignment = kDefaultAlignment;     // direct_io 时的块大小
  unsigned chunk_bytes = kDefaultChunkBytes;  // 单次 READ/WRITE 的最大长度
  unsigned depth = kDefaultDepth;             // 未完成写请求上限（= 并发度）
  bool fsync_on_sync = true;                  // Sync() 是否顺便 fsync 数据文件
};

// 一次跨文件随机读：fd 由调用方打开，dst 必须能容纳 length 字节。
struct ReadOp {
  int fd = -1;
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  void* dst = nullptr;
};

struct ReadResult {
  std::size_t bytes = 0;  // 实际读到的字节数（短读时小于请求长度）
  Status status;
};

// 同一文件上的一次批量读：从 offset 起读 length 字节到 dst。
//
// 读的典型用法是「按索引里的分片区间成批读回来」，所以这里只描述区间；
// fd（以及 ring）由 LocalFile::ReadRanges 提供，不用每次重复填。
struct RangeRead {
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  void* dst = nullptr;
};

// 大文件顺序写的一段数据。data 必须在 Sync() 返回前保持有效：写路径零拷贝，
// 不复制调用方的 buffer。
struct Chunk {
  const std::byte* data = nullptr;
  std::size_t size = 0;
};

struct WriteOptions {
  // 大文件顺序写默认 O_DIRECT：绕过 page cache，避免 checkpoint 写入把它冲爆。
  bool direct_io = true;
  bool fsync_data = true;  // Sync() 时 fsync 数据文件（保证落盘）
  bool fsync_dir = true;   // WriteFile() 的 rename 之后 fsync 父目录
  unsigned alignment = kDefaultAlignment;
  unsigned chunk_bytes = kDefaultChunkBytes;
  unsigned depth = kDefaultDepth;
};

// O_DIRECT 需要页对齐的 buffer；planner 拼对齐缓冲时用它。
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  ~AlignedBuffer();

  AlignedBuffer(AlignedBuffer&&) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&&) noexcept;
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  // alignment 必须是 2 的幂且不小于 sizeof(void*)；bytes 为 0 时只分配对齐的最小块。
  static Status Allocate(std::size_t bytes, unsigned alignment, AlignedBuffer* out);

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return data_ == nullptr; }

 private:
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

// 一个本地文件句柄。
//
// 写模式：在已打开的大文件上按 offset 顺序追加多段（planner 规划好分段），
// 内部攒批提交到 io_uring，不额外创建线程；Sync() 收完在途请求并 fsync。
// 读模式：打开时自带一个 ring，随机读走 ReadRanges()（一次提交、一次回收 +
// 逐项结果）；跨文件批量读仍可用 ReadMany()。
class LocalFile {
 public:
  ~LocalFile();

  LocalFile(LocalFile&&) noexcept;
  LocalFile& operator=(LocalFile&&) noexcept;
  LocalFile(const LocalFile&) = delete;
  LocalFile& operator=(const LocalFile&) = delete;

  static Status OpenRead(std::string_view path, const LocalFileOptions& options,
                         LocalFile** out);
  // O_WRONLY|O_CREAT|O_TRUNC，direct_io 时再加 O_DIRECT。
  static Status OpenWrite(std::string_view path, const LocalFileOptions& options,
                          LocalFile** out);

  int fd() const;

  // 顺序写：offset 必须等于上次写到的位置（首段为 0），direct_io 时 offset 与
  // data.size() 都要按 alignment 对齐。数据不拷贝，Sync() 前要一直有效。
  Status WriteAt(std::uint64_t offset, std::span<const std::byte> data);

  // 批量随机读：一次提交、一次回收，短读报 kIoError（不会静默截断）。
  // direct_io 打开时 offset 与 length 都要按 alignment 对齐。
  Status ReadRanges(std::span<const RangeRead> ranges);

  // 等待所有在途写完成；fsync 为 true 时再 fsync 数据文件。
  Status Sync(bool fsync);
  Status Sync();

  Status Close();

 private:
  LocalFile();

  class Impl;
  std::unique_ptr<Impl> impl_;
};

// 一次提交、一次等待，收完所有 CQE 才返回。可跨多个 fd，结果与 ops 一一对应。
// 短读（文件尾或请求超出范围）会返回 kIoError，同时在 bytes 里保留已读长度。
Status ReadMany(io::Ring* ring, std::span<const ReadOp> ops, std::vector<ReadResult>* out);

// 整文件顺序写：临时文件 -> 顺序 WRITE -> fsync -> rename -> 目录 fsync。
// direct_io 时除最后一段外的 Chunk 都要按 alignment 对齐，最后一段自动补零到
// 块边界（文件因此可能比逻辑数据大，读取端按真实长度截断）。
Status WriteFile(std::string_view path, std::span<const Chunk> chunks,
                 const WriteOptions& options = {});

// rename(2)。同路径直接成功；跨文件系统会失败，因为本地后端只处理同一个 root。
Status RenameAtomic(std::string_view src, std::string_view dst);

// fsync 目录项（rename 之后调用，保证目录本身也落盘）。
Status FsyncDir(std::string_view path);

}  // namespace ckpt::storage
