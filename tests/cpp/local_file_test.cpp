#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "ckpt/error.hpp"
#include "ckpt/io/uring.hpp"
#include "ckpt/storage/local_file.hpp"

namespace fs = std::filesystem;
using ckpt::ErrorCode;
using ckpt::Status;
using ckpt::io::Ring;
using ckpt::storage::AlignedBuffer;
using ckpt::storage::Chunk;
using ckpt::storage::LocalFile;
using ckpt::storage::LocalFileOptions;
using ckpt::storage::ReadMany;
using ckpt::storage::ReadOp;
using ckpt::storage::ReadResult;
using ckpt::storage::WriteFile;
using ckpt::storage::WriteOptions;

namespace {

bool g_direct_io = false;

#define CHECK_TRUE(expr)                                                        \
  do {                                                                          \
    if (!(expr)) {                                                              \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,     \
                   #expr);                                                      \
      return false;                                                             \
    }                                                                           \
  } while (0)

#define CHECK_OK(expr)                                                          \
  do {                                                                          \
    Status _st = (expr);                                                        \
    if (!_st.ok()) {                                                            \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,      \
                   _st.ToString().c_str());                                     \
      return false;                                                             \
    }                                                                           \
  } while (0)

#define CHECK_CODE(expr, want)                                                  \
  do {                                                                          \
    Status _st = (expr);                                                        \
    if (_st.code() != (want)) {                                                 \
      std::fprintf(stderr, "%s:%d: %s -> %s, want %s\n", __FILE__, __LINE__,    \
                   #expr, _st.ToString().c_str(),                               \
                   ckpt::ErrorCodeName(want));                                  \
      return false;                                                             \
    }                                                                           \
  } while (0)

fs::path MakeTempDir() {
  // 临时目录可配：O_DIRECT 是否可用依赖文件系统（tmpfs 与 ext4/xfs 行为不同），
  // 要在指定文件系统上跑用例，就设 TMPDIR。
  const char* base = std::getenv("TMPDIR");
  std::string templ =
      (base != nullptr && base[0] == '/') ? std::string(base) : std::string("/tmp");
  if (templ.back() != '/') templ += '/';
  templ += "ckpt-local-test-XXXXXX";
  char* made = ::mkdtemp(templ.data());
  return made == nullptr ? fs::path() : fs::path(made);
}

// O_DIRECT 要求 buffer 地址按块对齐，测试数据因此统一放在 AlignedBuffer 里；
// 逻辑长度单独记（分配长度会被向上取整到块边界）。
class Block {
 public:
  Block() = default;
  Block(std::size_t size, std::uint32_t seed) { (void)Fill(size, seed); }

  Block(const Block& other) { *this = other; }
  Block& operator=(const Block& other) {
    if (this == &other) return *this;
    if (!Fill(other.size_, 0)) return *this;
    if (size_ != 0) std::memcpy(data(), other.data(), size_);
    return *this;
  }
  Block(Block&&) noexcept = default;
  Block& operator=(Block&&) noexcept = default;

  // 按 4096 对齐分配并用固定种子填满（确定性数据，便于逐字节比对）。
  bool Fill(std::size_t size, std::uint32_t seed) {
    if (!AlignedBuffer::Allocate(size, 4096, &buffer_).ok()) return false;
    size_ = size;
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < size_; ++i) {
      data()[i] = static_cast<std::byte>(rng() & 0xFF);
    }
    return true;
  }

  std::byte* data() { return buffer_.data(); }
  const std::byte* data() const { return buffer_.data(); }
  std::size_t size() const { return size_; }

 private:
  AlignedBuffer buffer_;
  std::size_t size_ = 0;
};

std::vector<std::byte> ReadWholeFile(const fs::path& path) {
  std::vector<std::byte> out;
  LocalFile* file = nullptr;
  Status st = LocalFile::OpenRead(path.string(), {}, &file);
  if (!st.ok()) return out;
  std::unique_ptr<LocalFile> guard(file);

  Ring ring;
  st = Ring::Create(64, &ring);
  if (!st.ok()) return out;

  std::error_code ec;
  const auto size = static_cast<std::uint32_t>(fs::file_size(path, ec));
  if (ec || size == 0) return out;

  out.resize(size);
  std::vector<ReadResult> results;
  ReadOp op{file->fd(), 0, size, out.data()};
  st = ReadMany(&ring, std::span<const ReadOp>(&op, 1), &results);
  if (!st.ok() || results.size() != 1 || !results[0].status.ok()) return {};
  return out;
}

bool WriteChunks(const fs::path& path, const std::vector<Block>& parts,
                 const WriteOptions& options) {
  std::vector<Chunk> chunks;
  chunks.reserve(parts.size());
  for (const auto& part : parts) {
    chunks.push_back(Chunk{part.data(), part.size()});
  }
  return WriteFile(path.string(), chunks, options).ok();
}

bool DirectIoSupported(const fs::path& dir) {
  const fs::path probe = dir / "direct-probe.bin";
  const std::vector<Block> parts{Block(4096, 1)};
  const bool ok = WriteChunks(probe, parts, WriteOptions{.direct_io = true});
  std::error_code ec;
  fs::remove(probe, ec);
  return ok;
}

std::size_t CountThreads() {
  std::error_code ec;
  std::size_t count = 0;
  for (const auto& entry : fs::directory_iterator("/proc/self/task", ec)) {
    (void)entry;
    ++count;
  }
  return count;
}

// --- 用例 -----------------------------------------------------------------

bool TestWriteFileRoundTrip(const fs::path& dir) {
  const fs::path path = dir / "roundtrip.bin";
  const std::vector<Block> parts{Block(1u << 20, 1), Block(1u << 20, 2),
                                 Block(1u << 20, 3)};

  WriteOptions options;
  options.direct_io = g_direct_io;
  options.chunk_bytes = 256u << 10;
  options.depth = 4;
  CHECK_TRUE(WriteChunks(path, parts, options));

  // 目录里只应留下目标文件，临时文件必须被 rename 掉。
  std::size_t files = 0;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) ++files;
  }
  CHECK_TRUE(files == 1);

  const std::vector<std::byte> got = ReadWholeFile(path);
  CHECK_TRUE(got.size() == 3u << 20);
  std::size_t offset = 0;
  for (const auto& part : parts) {
    CHECK_TRUE(std::memcmp(got.data() + offset, part.data(), part.size()) == 0);
    offset += part.size();
  }
  return true;
}

bool TestWriteFileTailPadding(const fs::path& dir) {
  const fs::path path = dir / "tail.bin";
  // 尾段故意用非块对齐长度：direct_io 时应补零到块边界，buffered 时按原样写。
  const std::vector<Block> parts{Block(4096, 7), Block(1000, 8)};

  WriteOptions options;
  options.direct_io = g_direct_io;
  CHECK_TRUE(WriteChunks(path, parts, options));

  const std::vector<std::byte> got = ReadWholeFile(path);
  CHECK_TRUE(got.size() == (options.direct_io ? 4096u + 4096u : 4096u + 1000u));
  CHECK_TRUE(std::memcmp(got.data(), parts[0].data(), parts[0].size()) == 0);
  CHECK_TRUE(std::memcmp(got.data() + 4096, parts[1].data(), parts[1].size()) == 0);
  if (options.direct_io) CHECK_TRUE(got[4096 + 1000] == std::byte{0});
  return true;
}

bool TestWriteFileUnalignedTailBuffer(const fs::path& dir) {
  // 最后一段的 buffer 地址不对齐也必须能写：内部转一次对齐缓冲并把尾部补零。
  const fs::path path = dir / "tail-unaligned.bin";
  Block data(4096, 91);
  CHECK_TRUE(data.data() != nullptr);
  const Chunk chunks[] = {{data.data() + 3, data.size() - 3}};

  WriteOptions options;
  options.direct_io = true;
  const Status st = WriteFile(path.string(), std::span<const Chunk>(chunks, 1), options);
  std::error_code ec;
  const auto written = fs::file_size(path, ec);
  fs::remove(path, ec);
  if (!g_direct_io) {
    // 文件系统不支持 O_DIRECT 时 open 就会失败；这条用例只要求不崩。
    std::printf("     (skip: O_DIRECT unsupported on this filesystem)\n");
    return true;
  }
  CHECK_OK(st);
  CHECK_TRUE(!ec);
  CHECK_TRUE(written == 4096);
  return true;
}

bool TestWriteFileRejectsUnalignedNonFinalBuffer(const fs::path& dir) {
  // 非末段的 buffer 地址不对齐必须报错：不同内核对非对齐 O_DIRECT 的宽容度不同
  // （有的 EINVAL，有的静默接受），把判断留在内核里会让行为随环境漂移。
  const fs::path path = dir / "mid-unaligned.bin";
  if (!g_direct_io) {
    std::printf("     (skip: O_DIRECT unsupported on this filesystem)\n");
    return true;
  }
  Block first(4096, 92);
  Block second(4096, 93);
  CHECK_TRUE(first.data() != nullptr && second.data() != nullptr);
  const Chunk chunks[] = {{first.data() + 1, first.size()}, {second.data(), second.size()}};

  WriteOptions options;
  options.direct_io = true;
  CHECK_CODE(WriteFile(path.string(), std::span<const Chunk>(chunks, 2), options),
             ErrorCode::kInvalidArgument);
  CHECK_TRUE(!fs::exists(path));
  return true;
}

bool TestReadManyAcrossFds(const fs::path& dir) {
  const fs::path a = dir / "a.bin";
  const fs::path b = dir / "b.bin";
  const Block data_a(1u << 20, 11);
  const Block data_b(512u << 10, 12);

  CHECK_TRUE(WriteChunks(a, {data_a}, WriteOptions{.direct_io = g_direct_io}));
  CHECK_TRUE(WriteChunks(b, {data_b}, WriteOptions{.direct_io = g_direct_io}));

  LocalFile* fa = nullptr;
  LocalFile* fb = nullptr;
  CHECK_OK(LocalFile::OpenRead(a.string(), {}, &fa));
  std::unique_ptr<LocalFile> guard_a(fa);
  CHECK_OK(LocalFile::OpenRead(b.string(), {}, &fb));
  std::unique_ptr<LocalFile> guard_b(fb);

  Ring ring;
  CHECK_OK(Ring::Create(128, &ring));

  // 交错、乱序地读两个文件的不同区间。
  Block out(6 * 4096, 0);
  std::vector<ReadOp> ops{
      {fa->fd(), 3 * 4096, 4096, out.data() + 0 * 4096},
      {fb->fd(), 7 * 4096, 4096, out.data() + 1 * 4096},
      {fa->fd(), 0, 4096, out.data() + 2 * 4096},
      {fb->fd(), 0, 4096, out.data() + 3 * 4096},
      {fa->fd(), 100 * 4096, 4096, out.data() + 4 * 4096},
      {fb->fd(), 50 * 4096, 4096, out.data() + 5 * 4096},
  };
  std::vector<ReadResult> results;
  CHECK_OK(ReadMany(&ring, ops, &results));
  CHECK_TRUE(results.size() == ops.size());
  for (const auto& result : results) CHECK_TRUE(result.status.ok());

  CHECK_TRUE(std::memcmp(out.data() + 0, data_a.data() + 3 * 4096, 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 1 * 4096, data_b.data() + 7 * 4096, 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 2 * 4096, data_a.data(), 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 3 * 4096, data_b.data(), 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 4 * 4096, data_a.data() + 100 * 4096, 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 5 * 4096, data_b.data() + 50 * 4096, 4096) == 0);
  return true;
}

bool TestReadManyShortRead(const fs::path& dir) {
  const fs::path path = dir / "short.bin";
  CHECK_TRUE(WriteChunks(path, {Block(4096, 21)}, WriteOptions{.direct_io = g_direct_io}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  Ring ring;
  CHECK_OK(Ring::Create(16, &ring));

  Block out(8192, 0);
  ReadOp op{file->fd(), 0, 8192, out.data()};
  std::vector<ReadResult> results;
  CHECK_OK(ReadMany(&ring, std::span<const ReadOp>(&op, 1), &results));
  CHECK_TRUE(results.size() == 1);
  CHECK_CODE(results[0].status, ErrorCode::kIoError);
  CHECK_TRUE(results[0].bytes == 4096);
  return true;
}

bool TestReadManyMissingFd(const fs::path& dir) {
  (void)dir;
  Ring ring;
  CHECK_OK(Ring::Create(16, &ring));

  Block out(4096, 0);
  ReadOp op{-1, 0, 4096, out.data()};
  std::vector<ReadResult> results;
  CHECK_CODE(ReadMany(&ring, std::span<const ReadOp>(&op, 1), &results),
             ErrorCode::kInvalidArgument);
  return true;
}

// --- LocalFile::ReadRanges（读模式自带 ring，一次提交读多段）----------------

bool TestReadRangesBatch(const fs::path& dir) {
  const fs::path path = dir / "ranges.bin";
  const Block data(1u << 20, 51);
  CHECK_TRUE(WriteChunks(path, {data}, WriteOptions{.direct_io = g_direct_io}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  // 三段读进一个连续 buffer：验证区间与落点各自独立，且都按请求填满。
  Block out(3 * 4096, 0);
  std::vector<ckpt::storage::RangeRead> ranges{
      {0, 4096, out.data()},
      {5000, 2048, out.data() + 4096},
      {600000, 4096, out.data() + 4096 + 2048},
  };
  CHECK_OK(file->ReadRanges(ranges));

  CHECK_TRUE(std::memcmp(out.data(), data.data(), 4096) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 4096, data.data() + 5000, 2048) == 0);
  CHECK_TRUE(std::memcmp(out.data() + 4096 + 2048, data.data() + 600000, 4096) == 0);
  return true;
}

bool TestReadRangesManySegments(const fs::path& dir) {
  // 超过 ReadMany 内部的单批上限（64），逼出「多轮提交」这条路径。
  const fs::path path = dir / "ranges-many.bin";
  const Block data(128u * 4096, 52);
  CHECK_TRUE(WriteChunks(path, {data}, WriteOptions{.direct_io = g_direct_io}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  constexpr std::size_t kCount = 100;
  Block out(kCount * 4096, 0);
  std::vector<ckpt::storage::RangeRead> ranges;
  ranges.reserve(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    // 逆序 + 稀疏：每段 4096 字节，起点在文件内不重叠
    const std::size_t index = kCount - 1 - i;
    ranges.push_back({index * 4096, 4096, out.data() + i * 4096});
  }
  CHECK_OK(file->ReadRanges(ranges));

  for (std::size_t i = 0; i < kCount; ++i) {
    const std::size_t index = kCount - 1 - i;
    CHECK_TRUE(std::memcmp(out.data() + i * 4096, data.data() + index * 4096, 4096) == 0);
  }
  return true;
}

bool TestReadRangesShortRead(const fs::path& dir) {
  const fs::path path = dir / "ranges-short.bin";
  CHECK_TRUE(WriteChunks(path, {Block(4096, 53)}, WriteOptions{.direct_io = false}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  Block out(16, 0);
  std::vector<ckpt::storage::RangeRead> ranges{{4096 - 4, 16, out.data()}};
  // 短读必须报错，不能返回半份数据
  CHECK_CODE(file->ReadRanges(ranges), ErrorCode::kIoError);
  return true;
}

bool TestReadRangesRejectsBadArguments(const fs::path& dir) {
  const fs::path path = dir / "ranges-bad.bin";
  CHECK_TRUE(WriteChunks(path, {Block(8192, 54)}, WriteOptions{.direct_io = false}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  Block out(32, 0);
  std::vector<ckpt::storage::RangeRead> zero_length{{0, 0, out.data()}};
  CHECK_CODE(file->ReadRanges(zero_length), ErrorCode::kInvalidArgument);

  std::vector<ckpt::storage::RangeRead> null_dst{{0, 16, nullptr}};
  CHECK_CODE(file->ReadRanges(null_dst), ErrorCode::kInvalidArgument);

  // 空区间序列是合法的空操作
  CHECK_OK(file->ReadRanges({}));
  return true;
}

bool TestReadRangesRequiresReadMode(const fs::path& dir) {
  const fs::path path = dir / "ranges-write-mode.bin";
  LocalFile* file = nullptr;
  LocalFileOptions options;
  options.direct_io = false;
  CHECK_OK(LocalFile::OpenWrite(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);

  Block out(16, 0);
  std::vector<ckpt::storage::RangeRead> ranges{{0, 16, out.data()}};
  CHECK_CODE(guard->ReadRanges(ranges), ErrorCode::kInvalidArgument);
  return true;
}

bool TestReadRangesDirectIoAligned(const fs::path& dir) {
  // O_DIRECT 读的区间与落点都必须按块对齐；写侧已按 4096 对齐落盘，所以整块读成立。
  // 文件系统不支持 O_DIRECT（如部分 tmpfs 配置）时这个用例整体跳过。
  if (!g_direct_io) return true;
  const fs::path path = dir / "ranges-direct.bin";
  CHECK_TRUE(WriteChunks(path, {Block(8192, 55)}, WriteOptions{.direct_io = true}));

  LocalFileOptions options;
  options.direct_io = true;
  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);

  Block out(8192, 0);
  std::vector<ckpt::storage::RangeRead> aligned{{4096, 4096, out.data()}};
  CHECK_OK(guard->ReadRanges(aligned));
  // offset/length 不对齐：拒绝
  std::vector<ckpt::storage::RangeRead> misaligned{{100, 16, out.data()}};
  CHECK_CODE(guard->ReadRanges(misaligned), ErrorCode::kInvalidArgument);
  // 落点地址不对齐：也拒绝（否则 EINVAL 会晚一步才从 io_uring 里冒出来）
  std::vector<ckpt::storage::RangeRead> bad_dst{{0, 4096, out.data() + 1}};
  CHECK_CODE(guard->ReadRanges(bad_dst), ErrorCode::kInvalidArgument);
  return true;
}

bool TestThreadCountStable(const fs::path& dir) {
  const fs::path path = dir / "threads.bin";
  CHECK_TRUE(WriteChunks(path, {Block(1u << 20, 31)}, WriteOptions{.direct_io = g_direct_io}));

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenRead(path.string(), {}, &file));
  std::unique_ptr<LocalFile> guard(file);

  Ring ring;
  CHECK_OK(Ring::Create(128, &ring));

  const std::size_t before = CountThreads();
  Block out(64 * 4096, 0);
  std::vector<ReadOp> ops;
  ops.reserve(64);
  for (std::size_t i = 0; i < 64; ++i) {
    ops.push_back(ReadOp{file->fd(), (i * 31 % 256) * 4096, 4096, out.data() + i * 4096});
  }
  std::vector<ReadResult> results;
  CHECK_OK(ReadMany(&ring, ops, &results));
  const std::size_t after = CountThreads();

  // 批量随机读不应该创建任何线程：线程数与请求数无关。
  CHECK_TRUE(before == after);
  return true;
}

bool TestSequentialWriter(const fs::path& dir) {
  const fs::path path = dir / "sequential.bin";
  LocalFileOptions options;
  options.direct_io = g_direct_io;
  options.chunk_bytes = 64u << 10;
  options.depth = 2;

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenWrite(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);

  const Block first(64u << 10, 41);
  const Block second(32u << 10, 42);
  CHECK_OK(file->WriteAt(0, {first.data(), first.size()}));
  // 非递增的 offset 必须被拒绝，否则随机写会把顺序写的优势抹掉。
  CHECK_CODE(file->WriteAt(first.size() + 4096, {second.data(), second.size()}),
             ErrorCode::kInvalidArgument);
  CHECK_OK(file->WriteAt(first.size(), {second.data(), second.size()}));
  CHECK_OK(file->Sync());
  CHECK_OK(file->Close());

  const std::vector<std::byte> got = ReadWholeFile(path);
  CHECK_TRUE(got.size() == first.size() + second.size());
  CHECK_TRUE(std::memcmp(got.data(), first.data(), first.size()) == 0);
  CHECK_TRUE(std::memcmp(got.data() + first.size(), second.data(), second.size()) == 0);
  return true;
}

bool TestDirectIoRejectsMisaligned(const fs::path& dir) {
  if (!g_direct_io) {
    std::printf("     (skip: O_DIRECT unsupported on this filesystem)\n");
    return true;
  }
  const fs::path path = dir / "misaligned.bin";
  LocalFileOptions options;
  options.direct_io = true;
  options.alignment = 4096;

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenWrite(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);

  const Block data(4096, 51);
  CHECK_CODE(file->WriteAt(100, {data.data(), data.size()}), ErrorCode::kInvalidArgument);
  CHECK_CODE(file->WriteAt(0, {data.data(), 1000}), ErrorCode::kInvalidArgument);
  // offset 与长度都对、只有 buffer 地址不对：同样必须拒绝
  CHECK_CODE(file->WriteAt(0, {data.data() + 1, data.size()}),
             ErrorCode::kInvalidArgument);
  CHECK_OK(file->WriteAt(0, {data.data(), data.size()}));
  CHECK_OK(file->Sync());
  return true;
}

bool TestBufferedWrite(const fs::path& dir) {
  const fs::path path = dir / "buffered.bin";
  LocalFileOptions options;
  options.direct_io = false;

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenWrite(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);

  // buffered 模式不要求对齐：普通 vector 也能当数据源。
  const std::vector<std::byte> first(1000, std::byte{0xAB});
  const std::vector<std::byte> second(500, std::byte{0xCD});
  CHECK_OK(file->WriteAt(0, {first.data(), first.size()}));
  CHECK_OK(file->WriteAt(first.size(), {second.data(), second.size()}));
  CHECK_OK(file->Sync());
  CHECK_OK(file->Close());

  const std::vector<std::byte> got = ReadWholeFile(path);
  CHECK_TRUE(got.size() == 1500);
  CHECK_TRUE(std::memcmp(got.data(), first.data(), first.size()) == 0);
  CHECK_TRUE(std::memcmp(got.data() + first.size(), second.data(), second.size()) == 0);
  return true;
}

bool TestRenameAndFsyncDir(const fs::path& dir) {
  const fs::path src = dir / "commit.tmp";
  const fs::path dst = dir / "COMMIT";
  CHECK_TRUE(WriteChunks(src, {Block(128, 71)}, WriteOptions{.direct_io = false}));
  CHECK_OK(ckpt::storage::RenameAtomic(src.string(), dst.string()));
  CHECK_TRUE(fs::exists(dst));
  CHECK_TRUE(!fs::exists(src));
  CHECK_OK(ckpt::storage::FsyncDir(dir.string()));
  CHECK_CODE(ckpt::storage::RenameAtomic((dir / "missing").string(), dst.string()),
             ErrorCode::kNotFound);
  return true;
}

bool TestOpenMissing(const fs::path& dir) {
  LocalFile* file = nullptr;
  CHECK_CODE(LocalFile::OpenRead((dir / "nope.bin").string(), {}, &file),
             ErrorCode::kNotFound);
  CHECK_TRUE(file == nullptr);
  return true;
}

bool TestAlignedBuffer(const fs::path& dir) {
  (void)dir;
  AlignedBuffer buffer;
  CHECK_OK(AlignedBuffer::Allocate(1000, 4096, &buffer));
  CHECK_TRUE(buffer.size() == 4096);
  CHECK_TRUE(reinterpret_cast<std::uintptr_t>(buffer.data()) % 4096 == 0);
  CHECK_CODE(AlignedBuffer::Allocate(4096, 1234, &buffer), ErrorCode::kInvalidArgument);
  return true;
}

bool TestFlushWithoutFsync(const fs::path& dir) {
  const fs::path path = dir / "no-fsync.bin";
  LocalFileOptions options;
  options.direct_io = false;
  options.fsync_on_sync = false;

  LocalFile* file = nullptr;
  CHECK_OK(LocalFile::OpenWrite(path.string(), options, &file));
  std::unique_ptr<LocalFile> guard(file);
  const Block data(8192, 81);
  CHECK_OK(file->WriteAt(0, {data.data(), data.size()}));
  // Sync(false) 只等写完，不 fsync；数据此时仍在 page cache 里。
  CHECK_OK(file->Sync(false));
  CHECK_OK(file->Close());
  CHECK_TRUE(fs::file_size(path) == 8192);
  return true;
}

}  // namespace

int main() {
  const fs::path dir = MakeTempDir();
  if (dir.empty()) {
    std::fprintf(stderr, "failed to create temp dir\n");
    return 1;
  }
  g_direct_io = DirectIoSupported(dir);
  std::printf("temp dir: %s (O_DIRECT %s)\n", dir.string().c_str(),
              g_direct_io ? "supported" : "unsupported -> buffered only");

  struct Case {
    const char* name;
    bool (*fn)(const fs::path&);
  };
  const Case cases[] = {
      {"write_file_roundtrip", TestWriteFileRoundTrip},
      {"write_file_tail_padding", TestWriteFileTailPadding},
      {"write_file_unaligned_tail_buffer", TestWriteFileUnalignedTailBuffer},
      {"write_file_rejects_unaligned_non_final_buffer",
       TestWriteFileRejectsUnalignedNonFinalBuffer},
      {"read_many_across_fds", TestReadManyAcrossFds},
      {"read_many_short_read", TestReadManyShortRead},
      {"read_many_missing_fd", TestReadManyMissingFd},
      {"read_ranges_batch", TestReadRangesBatch},
      {"read_ranges_many_segments", TestReadRangesManySegments},
      {"read_ranges_short_read", TestReadRangesShortRead},
      {"read_ranges_bad_arguments", TestReadRangesRejectsBadArguments},
      {"read_ranges_requires_read_mode", TestReadRangesRequiresReadMode},
      {"read_ranges_direct_io_aligned", TestReadRangesDirectIoAligned},
      {"thread_count_stable", TestThreadCountStable},
      {"sequential_writer", TestSequentialWriter},
      {"direct_io_rejects_misaligned", TestDirectIoRejectsMisaligned},
      {"buffered_write", TestBufferedWrite},
      {"rename_and_fsync_dir", TestRenameAndFsyncDir},
      {"open_missing", TestOpenMissing},
      {"aligned_buffer", TestAlignedBuffer},
      {"flush_without_fsync", TestFlushWithoutFsync},
  };

  int failed = 0;
  for (const Case& test : cases) {
    if (test.fn(dir)) {
      std::printf("[ OK ] %s\n", test.name);
    } else {
      std::printf("[FAIL] %s\n", test.name);
      ++failed;
    }
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failed != 0) {
    std::fprintf(stderr, "%d case(s) failed\n", failed);
    return 1;
  }
  std::printf("all %zu cases passed\n", sizeof(cases) / sizeof(cases[0]));
  return 0;
}
