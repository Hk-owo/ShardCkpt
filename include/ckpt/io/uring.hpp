#pragma once

#include <cstdint>
#include <memory>

#include "ckpt/error.hpp"

namespace ckpt::io {

// 一条完成事件的抽象：user_data 由提交方指定，res 与 io_uring 的 cqe->res 一致
// （成功为正的字节数，失败为 -errno）。
struct Completion {
  std::uint64_t user_data = 0;
  int res = 0;
};

// io_uring 的 RAII 封装。一个 Ring 只能由创建它的线程使用：io_uring 的 SQ/CQ
// 本身不支持多线程共享，这也是「随机读不额外创建线程」的前提——调用线程自己
// 填 SQE、自己收割 CQE。
//
// 所有方法失败时返回 Status，不抛异常；liburing 的类型不出现在公共头里。
class Ring {
 public:
  Ring();
  ~Ring();

  Ring(Ring&&) noexcept;
  Ring& operator=(Ring&&) noexcept;
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;

  // entries 决定 SQ/CQ 深度，即单次可并发的请求上限（内核会向上取整到 2 的幂）。
  static Status Create(unsigned entries, Ring* out);

  bool valid() const { return impl_ != nullptr; }

  // 填 SQE，不提交。user_data 会在对应的 Completion 里原样带回。
  Status PrepRead(int fd, void* dst, unsigned len, std::uint64_t offset,
                  std::uint64_t user_data);
  Status PrepWrite(int fd, const void* src, unsigned len, std::uint64_t offset,
                   std::uint64_t user_data);
  Status PrepFsync(int fd, bool datasync, std::uint64_t user_data);

  // 只提交已填的 SQE，不等待。
  Status Submit();
  // 提交已填的 SQE，并阻塞到至少 min_complete 个请求完成。
  Status SubmitAndWait(unsigned min_complete);

  // 收割一条已完成的 CQE，不阻塞；没有就绪的返回 false。
  bool PeekCompletion(Completion* out);
  // 批量收割，最多 max 条；返回实际条数。
  unsigned PeekBatch(Completion* out, unsigned max);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ckpt::io
