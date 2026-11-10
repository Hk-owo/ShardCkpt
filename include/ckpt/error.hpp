#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace ckpt {

enum class ErrorCode : std::uint8_t {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAuthError,
  kNetworkError,
  kIoError,
  kFormatError,
  kUnsupported,
  kNotImplemented,
};

inline const char* ErrorCodeName(ErrorCode code) {
  switch (code) {
    case ErrorCode::kOk: return "OK";
    case ErrorCode::kInvalidArgument: return "InvalidArgument";
    case ErrorCode::kNotFound: return "NotFound";
    case ErrorCode::kAuthError: return "AuthError";
    case ErrorCode::kNetworkError: return "NetworkError";
    case ErrorCode::kIoError: return "IoError";
    case ErrorCode::kFormatError: return "FormatError";
    case ErrorCode::kUnsupported: return "Unsupported";
    case ErrorCode::kNotImplemented: return "NotImplemented";
  }
  return "Unknown";
}

// 轻量错误值。C++ 层不抛异常，调用方统一检查 Status。
class Status {
 public:
  Status() = default;

  static Status OK() { return Status(); }
  static Status Error(ErrorCode code, std::string message) {
    return Status(code, std::move(message));
  }

  bool ok() const { return code_ == ErrorCode::kOk; }
  explicit operator bool() const { return ok(); }

  ErrorCode code() const { return code_; }
  const std::string& message() const { return message_; }

  std::string ToString() const {
    if (ok()) return "OK";
    return std::string(ErrorCodeName(code_)) + ": " + message_;
  }

 private:
  Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

  ErrorCode code_ = ErrorCode::kOk;
  std::string message_;
};

#define CKPT_RETURN_IF_ERROR(expr) \
  do {                             \
    ::ckpt::Status _st = (expr);   \
    if (!_st.ok()) return _st;     \
  } while (0)

}  // namespace ckpt
