// O_DIRECT / O_DIRECTORY 需要 _GNU_SOURCE，必须定义在所有 include 之前。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ckpt/storage/local_file.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "ckpt/error.hpp"
#include "ckpt/io/uring.hpp"

namespace ckpt::storage {
namespace {

constexpr std::size_t kMaxReadBatch = 64;
constexpr std::size_t kMaxCompletionBatch = 64;

Status ErrnoStatus(int err, std::string_view what) {
  std::string message(what);
  message += ": ";
  message += std::strerror(err);
  switch (err) {
    case ENOENT:
      return Status::Error(ErrorCode::kNotFound, std::move(message));
    case EBADF:
    case EINVAL:
      return Status::Error(ErrorCode::kInvalidArgument, std::move(message));
    default:
      return Status::Error(ErrorCode::kIoError, std::move(message));
  }
}

bool IsPowerOfTwo(unsigned value) { return value != 0 && (value & (value - 1)) == 0; }

bool IsAligned(std::uint64_t value, unsigned alignment) { return value % alignment == 0; }

Status ValidateOptions(const LocalFileOptions& options) {
  if (!IsPowerOfTwo(options.alignment)) {
    return Status::Error(ErrorCode::kInvalidArgument, "alignment must be a power of two");
  }
  if (options.alignment < 512) {
    return Status::Error(ErrorCode::kInvalidArgument, "alignment must be >= 512 bytes");
  }
  if (options.chunk_bytes == 0 ||
      options.chunk_bytes > std::numeric_limits<unsigned>::max()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "chunk_bytes must be in [1, UINT32_MAX]");
  }
  if (options.chunk_bytes % options.alignment != 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "chunk_bytes must be a multiple of alignment");
  }
  if (options.depth == 0 || options.depth > 4096) {
    return Status::Error(ErrorCode::kInvalidArgument, "depth must be in [1, 4096]");
  }
  return Status::OK();
}

// 临时文件放在目标文件同目录，rename 才是原子的。
std::string MakeTempPath(std::string_view path) {
  static std::atomic<std::uint64_t> counter{0};
  std::string temp(path);
  temp += ".tmp.";
  temp += std::to_string(::getpid());
  temp += '.';
  temp += std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
  return temp;
}

}  // namespace

class LocalFile::Impl {
 public:
  ~Impl() { CloseQuietly(); }

  Status ReapOne() {
    if (submitted == 0) return Status::OK();
    if (ring == nullptr) {
      return Status::Error(ErrorCode::kIoError, "ring is not initialized");
    }
    CKPT_RETURN_IF_ERROR(ring->SubmitAndWait(1));
    return Collect(std::min<std::size_t>(submitted, kMaxCompletionBatch));
  }

  Status DrainAll() {
    Status first_error;
    while (submitted > 0) {
      Status submitted_status = ring->SubmitAndWait(1);
      if (!submitted_status.ok()) {
        return first_error.ok() ? submitted_status : first_error;
      }
      Status st = Collect(std::min<std::size_t>(submitted, kMaxCompletionBatch));
      if (!st.ok() && first_error.ok()) first_error = st;
    }
    return first_error;
  }

  Status Collect(std::size_t at_most) {
    io::Completion completions[kMaxCompletionBatch];
    const unsigned got = ring->PeekBatch(completions, static_cast<unsigned>(at_most));
    if (got == 0) {
      return Status::Error(ErrorCode::kIoError,
                           "io_uring: wait returned without any completion");
    }
    Status first_error;
    for (unsigned i = 0; i < got; ++i) {
      Status st = CheckCompletion(completions[i]);
      if (!st.ok() && first_error.ok()) first_error = st;
      if (submitted > 0) --submitted;
    }
    return first_error;
  }

  Status CheckCompletion(const io::Completion& completion) {
    auto it = in_flight.find(completion.user_data);
    if (it == in_flight.end()) {
      return Status::Error(ErrorCode::kIoError,
                           "io_uring: completion for unknown request");
    }
    const unsigned expected = it->second;
    in_flight.erase(it);

    if (completion.res < 0) {
      return ErrnoStatus(-completion.res, "write");
    }
    if (static_cast<unsigned>(completion.res) != expected) {
      return Status::Error(ErrorCode::kIoError,
                           "short write: got " + std::to_string(completion.res) + " of " +
                               std::to_string(expected) + " bytes");
    }
    return Status::OK();
  }

  void CloseQuietly() {
    if (writable) (void)SyncInternal(fsync_on_sync);
    if (fd >= 0) {
      (void)::close(fd);
      fd = -1;
    }
  }

  Status SyncInternal(bool fsync) {
    Status st = DrainAll();
    if (!st.ok()) return st;
    if (!fsync || !dirty) return Status::OK();

    const std::uint64_t id = next_request_id++;
    st = ring->PrepFsync(fd, /*datasync=*/false, id);
    if (!st.ok()) return st;
    st = ring->SubmitAndWait(1);
    if (!st.ok()) return st;

    io::Completion completion;
    if (!ring->PeekCompletion(&completion)) {
      return Status::Error(ErrorCode::kIoError, "io_uring: fsync did not complete");
    }
    if (completion.res < 0) return ErrnoStatus(-completion.res, "fsync");
    dirty = false;
    return Status::OK();
  }

  int fd = -1;
  bool writable = false;
  bool direct_io = false;
  bool fsync_on_sync = true;
  unsigned alignment = kDefaultAlignment;
  unsigned chunk_bytes = kDefaultChunkBytes;
  unsigned depth = kDefaultDepth;

  std::uint64_t next_offset = 0;
  std::uint64_t next_request_id = 1;
  std::size_t submitted = 0;
  bool dirty = false;

  std::unique_ptr<io::Ring> ring;
  std::unordered_map<std::uint64_t, unsigned> in_flight;
};

LocalFile::LocalFile() = default;

LocalFile::~LocalFile() = default;

LocalFile::LocalFile(LocalFile&&) noexcept = default;
LocalFile& LocalFile::operator=(LocalFile&&) noexcept = default;

Status LocalFile::OpenRead(std::string_view path, const LocalFileOptions& options,
                           LocalFile** out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument, "out must not be null");
  }
  *out = nullptr;
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "path must not be empty");
  }
  CKPT_RETURN_IF_ERROR(ValidateOptions(options));

  const std::string name(path);
  const int raw_fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
  if (raw_fd < 0) return ErrnoStatus(errno, "open " + name);

  auto file = std::unique_ptr<LocalFile>(new LocalFile());
  file->impl_ = std::make_unique<Impl>();
  file->impl_->fd = raw_fd;
  file->impl_->writable = false;
  file->impl_->direct_io = options.direct_io;
  file->impl_->fsync_on_sync = options.fsync_on_sync;
  file->impl_->alignment = options.alignment;
  file->impl_->chunk_bytes = options.chunk_bytes;
  file->impl_->depth = options.depth;
  *out = file.release();
  return Status::OK();
}

Status LocalFile::OpenWrite(std::string_view path, const LocalFileOptions& options,
                            LocalFile** out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument, "out must not be null");
  }
  *out = nullptr;
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "path must not be empty");
  }
  CKPT_RETURN_IF_ERROR(ValidateOptions(options));

  int flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC;
  if (options.direct_io) flags |= O_DIRECT;

  const std::string name(path);
  const int raw_fd = ::open(name.c_str(), flags, 0644);
  if (raw_fd < 0) return ErrnoStatus(errno, "open " + name);

  auto ring = std::make_unique<io::Ring>();
  const unsigned entries = std::max(options.depth * 2, 2u);
  Status st = io::Ring::Create(entries, ring.get());
  if (!st.ok()) {
    (void)::close(raw_fd);
    return st;
  }

  auto file = std::unique_ptr<LocalFile>(new LocalFile());
  file->impl_ = std::make_unique<Impl>();
  file->impl_->fd = raw_fd;
  file->impl_->writable = true;
  file->impl_->direct_io = options.direct_io;
  file->impl_->fsync_on_sync = options.fsync_on_sync;
  file->impl_->alignment = options.alignment;
  file->impl_->chunk_bytes = options.chunk_bytes;
  file->impl_->depth = options.depth;
  file->impl_->ring = std::move(ring);
  *out = file.release();
  return Status::OK();
}

int LocalFile::fd() const { return impl_ ? impl_->fd : -1; }

Status LocalFile::WriteAt(std::uint64_t offset, std::span<const std::byte> data) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "file is not open");
  }
  Impl& self = *impl_;
  if (!self.writable) {
    return Status::Error(ErrorCode::kInvalidArgument, "file is not opened for writing");
  }
  if (data.empty()) return Status::OK();
  if (offset != self.next_offset) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "sequential write expected offset " +
                             std::to_string(self.next_offset) + ", got " +
                             std::to_string(offset));
  }
  if (offset > std::numeric_limits<std::uint64_t>::max() - data.size()) {
    return Status::Error(ErrorCode::kInvalidArgument, "offset + size overflows");
  }
  if (self.direct_io) {
    if (!IsAligned(offset, self.alignment) || !IsAligned(data.size(), self.alignment)) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "direct_io requires offset and size to be aligned to " +
                               std::to_string(self.alignment) + " bytes");
    }
  }

  std::size_t written = 0;
  while (written < data.size()) {
    const std::size_t piece = std::min<std::size_t>(self.chunk_bytes, data.size() - written);
    if (self.submitted >= self.depth) {
      CKPT_RETURN_IF_ERROR(self.ReapOne());
    }
    const std::uint64_t piece_offset = offset + written;
    const std::uint64_t id = self.next_request_id++;
    Status st = self.ring->PrepWrite(self.fd, data.data() + written,
                                     static_cast<unsigned>(piece), piece_offset, id);
    if (!st.ok()) return st;
    self.in_flight.emplace(id, static_cast<unsigned>(piece));
    ++self.submitted;
    written += piece;
  }

  self.next_offset = offset + data.size();
  self.dirty = true;
  return self.ring->Submit();
}

Status LocalFile::Sync(bool fsync) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "file is not open");
  }
  if (!impl_->writable) return Status::OK();
  return impl_->SyncInternal(fsync);
}

Status LocalFile::Sync() { return Sync(impl_ ? impl_->fsync_on_sync : false); }

Status LocalFile::Close() {
  if (!impl_) return Status::OK();
  Impl& self = *impl_;
  Status st = Status::OK();
  if (self.writable) st = self.SyncInternal(self.fsync_on_sync);
  if (self.fd >= 0) {
    if (::close(self.fd) != 0 && st.ok()) st = ErrnoStatus(errno, "close");
    self.fd = -1;
  }
  self.in_flight.clear();
  self.submitted = 0;
  return st;
}

Status ReadMany(io::Ring* ring, std::span<const ReadOp> ops, std::vector<ReadResult>* out) {
  if (ring == nullptr || out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring and out are required");
  }
  out->assign(ops.size(), ReadResult{});
  if (ops.empty()) return Status::OK();

  for (const ReadOp& op : ops) {
    if (op.fd < 0) {
      return Status::Error(ErrorCode::kInvalidArgument, "fd must be >= 0");
    }
    if (op.length == 0) {
      return Status::Error(ErrorCode::kInvalidArgument, "length must be > 0");
    }
    if (op.dst == nullptr) {
      return Status::Error(ErrorCode::kInvalidArgument, "dst must not be null");
    }
    if (op.offset > std::numeric_limits<std::uint64_t>::max() - op.length) {
      return Status::Error(ErrorCode::kInvalidArgument, "offset + length overflows");
    }
  }

  std::vector<std::size_t> progress(ops.size(), 0);
  std::vector<bool> done(ops.size(), false);
  std::vector<std::size_t> pending;
  pending.reserve(ops.size());
  for (std::size_t i = 0; i < ops.size(); ++i) pending.push_back(i);

  while (!pending.empty()) {
    const std::size_t batch = std::min(pending.size(), kMaxReadBatch);

    // 一轮只提交尚未读完的部分；短读会在下一轮继续提交剩余区间。
    for (std::size_t k = 0; k < batch; ++k) {
      const std::size_t index = pending[k];
      const ReadOp& op = ops[index];
      const std::size_t have = progress[index];
      const auto want = static_cast<unsigned>(op.length - have);
      Status st = ring->PrepRead(op.fd, static_cast<std::byte*>(op.dst) + have, want,
                                 op.offset + have, index);
      if (!st.ok()) return st;
    }
    CKPT_RETURN_IF_ERROR(ring->SubmitAndWait(static_cast<unsigned>(batch)));

    io::Completion completions[kMaxCompletionBatch];
    std::size_t collected = 0;
    while (collected < batch) {
      const unsigned got =
          ring->PeekBatch(completions, static_cast<unsigned>(batch - collected));
      if (got == 0) {
        CKPT_RETURN_IF_ERROR(ring->SubmitAndWait(1));
        continue;
      }
      for (unsigned j = 0; j < got; ++j) {
        const std::size_t index = static_cast<std::size_t>(completions[j].user_data);
        ReadResult& result = out->at(index);
        if (completions[j].res < 0) {
          result.status = ErrnoStatus(-completions[j].res, "read");
          done[index] = true;
        } else {
          progress[index] += static_cast<std::size_t>(completions[j].res);
          result.bytes = progress[index];
          if (completions[j].res == 0) {
            result.status = Status::Error(ErrorCode::kIoError,
                                          "short read: unexpected EOF at offset " +
                                              std::to_string(ops[index].offset + progress[index]));
            done[index] = true;
          } else if (progress[index] == ops[index].length) {
            done[index] = true;
          }
        }
      }
      collected += got;
    }
    std::erase_if(pending, [&](std::size_t index) { return done[index]; });
  }
  return Status::OK();
}

Status WriteFile(std::string_view path, std::span<const Chunk> chunks,
                 const WriteOptions& options) {
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "path must not be empty");
  }
  for (const Chunk& chunk : chunks) {
    if (chunk.size != 0 && chunk.data == nullptr) {
      return Status::Error(ErrorCode::kInvalidArgument, "chunk data must not be null");
    }
  }

  LocalFileOptions file_options;
  file_options.direct_io = options.direct_io;
  file_options.alignment = options.alignment;
  file_options.chunk_bytes = options.chunk_bytes;
  file_options.depth = options.depth;
  file_options.fsync_on_sync = options.fsync_data;

  const std::string final_path(path);
  const std::string temp_path = MakeTempPath(path);

  LocalFile* raw = nullptr;
  CKPT_RETURN_IF_ERROR(LocalFile::OpenWrite(temp_path, file_options, &raw));
  std::unique_ptr<LocalFile> file(raw);

  auto fail = [&](const Status& st) {
    (void)file->Close();
    (void)::unlink(temp_path.c_str());
    return st;
  };

  std::size_t last_non_empty = chunks.size();
  for (std::size_t i = chunks.size(); i > 0; --i) {
    if (chunks[i - 1].size != 0) {
      last_non_empty = i - 1;
      break;
    }
  }

  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const Chunk& chunk = chunks[i];
    if (chunk.size == 0) continue;
    const bool last = (i == last_non_empty);

    if (!options.direct_io) {
      Status st = file->WriteAt(offset, {chunk.data, chunk.size});
      if (!st.ok()) return fail(st);
      offset += chunk.size;
      continue;
    }

    if (IsAligned(offset, options.alignment) && IsAligned(chunk.size, options.alignment)) {
      Status st = file->WriteAt(offset, {chunk.data, chunk.size});
      if (!st.ok()) return fail(st);
      offset += chunk.size;
      continue;
    }

    if (!last) {
      return fail(Status::Error(ErrorCode::kInvalidArgument,
                                "direct_io requires every non-final chunk to be aligned to " +
                                    std::to_string(options.alignment) + " bytes"));
    }

    // 最后一段不足一个块：先写对齐部分，再用零补齐一整块。
    const std::size_t aligned = chunk.size - chunk.size % options.alignment;
    if (aligned > 0) {
      Status st = file->WriteAt(offset, {chunk.data, aligned});
      if (!st.ok()) return fail(st);
      offset += aligned;
    }
    AlignedBuffer tail;
    Status st = AlignedBuffer::Allocate(options.alignment, options.alignment, &tail);
    if (!st.ok()) return fail(st);
    std::memset(tail.data(), 0, tail.size());
    std::memcpy(tail.data(), chunk.data + aligned, chunk.size - aligned);
    st = file->WriteAt(offset, {tail.data(), tail.size()});
    if (!st.ok()) return fail(st);
    offset += tail.size();
  }

  Status st = file->Sync(options.fsync_data);
  Status close_status = file->Close();
  if (!st.ok()) return fail(st);
  if (!close_status.ok()) return fail(close_status);

  st = RenameAtomic(temp_path, final_path);
  if (!st.ok()) return fail(st);

  if (options.fsync_dir) {
    std::filesystem::path parent = std::filesystem::path(final_path).parent_path();
    if (parent.empty()) parent = ".";
    CKPT_RETURN_IF_ERROR(FsyncDir(parent.string()));
  }
  return Status::OK();
}

Status RenameAtomic(std::string_view src, std::string_view dst) {
  if (src.empty() || dst.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "src and dst must not be empty");
  }
  if (src == dst) return Status::OK();

  const std::string from(src);
  const std::string to(dst);
  if (::rename(from.c_str(), to.c_str()) != 0) {
    return ErrnoStatus(errno, "rename " + from + " -> " + to);
  }
  return Status::OK();
}

Status FsyncDir(std::string_view path) {
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "path must not be empty");
  }
  const std::string name(path);
  const int fd = ::open(name.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return ErrnoStatus(errno, "open directory " + name);

  Status st = Status::OK();
  if (::fsync(fd) != 0) st = ErrnoStatus(errno, "fsync directory " + name);
  if (::close(fd) != 0 && st.ok()) st = ErrnoStatus(errno, "close directory " + name);
  return st;
}

AlignedBuffer::~AlignedBuffer() { std::free(data_); }

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
  if (this != &other) {
    std::free(data_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

Status AlignedBuffer::Allocate(std::size_t bytes, unsigned alignment, AlignedBuffer* out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument, "out must not be null");
  }
  if (!IsPowerOfTwo(alignment) || alignment < sizeof(void*)) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "alignment must be a power of two and >= sizeof(void*)");
  }
  if (bytes > std::numeric_limits<std::size_t>::max() - alignment) {
    return Status::Error(ErrorCode::kInvalidArgument, "size + alignment overflows");
  }
  const std::size_t rounded =
      bytes == 0 ? alignment : (bytes + alignment - 1) / alignment * alignment;

  std::free(out->data_);
  out->data_ = nullptr;
  out->size_ = 0;

  void* pointer = nullptr;
  if (::posix_memalign(&pointer, alignment, rounded) != 0) {
    return Status::Error(ErrorCode::kIoError, "posix_memalign failed");
  }
  out->data_ = static_cast<std::byte*>(pointer);
  out->size_ = rounded;
  return Status::OK();
}

}  // namespace ckpt::storage
