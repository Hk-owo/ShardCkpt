#include "ckpt/io/uring.hpp"

#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "ckpt/error.hpp"

namespace ckpt::io {
namespace {

Status ErrnoStatus(int err, std::string_view what) {
  std::string message(what);
  message += ": ";
  message += std::strerror(err);
  // 内核没编译 io_uring（或被 seccomp 拦掉）时明确报 Unsupported，调用方不用猜。
  const ErrorCode code = (err == ENOSYS || err == EPERM) ? ErrorCode::kUnsupported
                                                        : ErrorCode::kIoError;
  return Status::Error(code, std::move(message));
}

}  // namespace

class Ring::Impl {
 public:
  ~Impl() { io_uring_queue_exit(&ring); }

  ::io_uring ring;
};

Ring::Ring() = default;
Ring::~Ring() = default;
Ring::Ring(Ring&&) noexcept = default;
Ring& Ring::operator=(Ring&&) noexcept = default;

Status Ring::Create(unsigned entries, Ring* out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument, "out must not be null");
  }
  if (entries == 0) {
    return Status::Error(ErrorCode::kInvalidArgument, "entries must be > 0");
  }

  auto impl = std::make_unique<Impl>();
  // 不启用 SQPOLL / IOPOLL：SQPOLL 需要 CAP_SYS_NICE 且常驻内核线程，
  // IOPOLL 只在 O_DIRECT + 可轮询设备时有意义，都不属于默认行为。
  const int rc = io_uring_queue_init(entries, &impl->ring, /*flags=*/0);
  if (rc < 0) {
    std::string message = "io_uring_queue_init(entries=" + std::to_string(entries) + ")";
    return ErrnoStatus(-rc, message);
  }
  out->impl_ = std::move(impl);
  return Status::OK();
}

Status Ring::PrepRead(int fd, void* dst, unsigned len, std::uint64_t offset,
                      std::uint64_t user_data) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring is not initialized");
  }
  if (dst == nullptr || len == 0) {
    return Status::Error(ErrorCode::kInvalidArgument, "read destination and length are required");
  }
  io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
  if (sqe == nullptr) {
    return Status::Error(ErrorCode::kIoError,
                         "submission queue is full; submit outstanding requests first");
  }
  io_uring_prep_read(sqe, fd, dst, len, offset);
  sqe->user_data = user_data;
  return Status::OK();
}

Status Ring::PrepWrite(int fd, const void* src, unsigned len, std::uint64_t offset,
                       std::uint64_t user_data) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring is not initialized");
  }
  if (src == nullptr || len == 0) {
    return Status::Error(ErrorCode::kInvalidArgument, "write source and length are required");
  }
  io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
  if (sqe == nullptr) {
    return Status::Error(ErrorCode::kIoError,
                         "submission queue is full; submit outstanding requests first");
  }
  io_uring_prep_write(sqe, fd, src, len, offset);
  sqe->user_data = user_data;
  return Status::OK();
}

Status Ring::PrepFsync(int fd, bool datasync, std::uint64_t user_data) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring is not initialized");
  }
  io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
  if (sqe == nullptr) {
    return Status::Error(ErrorCode::kIoError,
                         "submission queue is full; submit outstanding requests first");
  }
  io_uring_prep_fsync(sqe, fd, datasync ? IORING_FSYNC_DATASYNC : 0);
  sqe->user_data = user_data;
  return Status::OK();
}

Status Ring::Submit() {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring is not initialized");
  }
  const int rc = io_uring_submit(&impl_->ring);
  if (rc < 0) return ErrnoStatus(-rc, "io_uring_submit");
  return Status::OK();
}

Status Ring::SubmitAndWait(unsigned min_complete) {
  if (!impl_) {
    return Status::Error(ErrorCode::kInvalidArgument, "ring is not initialized");
  }
  const int rc = io_uring_submit_and_wait(&impl_->ring, min_complete);
  if (rc < 0) return ErrnoStatus(-rc, "io_uring_submit_and_wait");
  return Status::OK();
}

bool Ring::PeekCompletion(Completion* out) {
  return PeekBatch(out, 1) == 1;
}

unsigned Ring::PeekBatch(Completion* out, unsigned max) {
  if (!impl_ || out == nullptr || max == 0) return 0;

  constexpr unsigned kBatch = 64;
  io_uring_cqe* cqes[kBatch];
  unsigned collected = 0;
  while (collected < max) {
    const unsigned want = std::min(max - collected, kBatch);
    const unsigned got = io_uring_peek_batch_cqe(&impl_->ring, cqes, want);
    if (got == 0) break;
    for (unsigned i = 0; i < got; ++i) {
      out[collected + i].user_data = cqes[i]->user_data;
      out[collected + i].res = cqes[i]->res;
    }
    io_uring_cq_advance(&impl_->ring, got);
    collected += got;
  }
  return collected;
}

}  // namespace ckpt::io
