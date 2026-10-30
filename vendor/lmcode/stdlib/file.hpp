#pragma once
// Raw byte file helpers. Paths use the VM byte-string contract; content is
// never decoded or newline-translated.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#endif
#include "dslvm.hpp"
#ifndef DSLVM_STD_BUILDING
#undef DSLVM_STD_DEFER
#endif
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_BUILDING
#include "collections.hpp"
#undef DSLVM_STD_BUILDING
#else
#include "collections.hpp"
#endif
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <stdexcept>
#include <vector>

namespace dslvm::stdlib {
namespace file_detail {

inline Error failure(const std::string& function, const std::string& path, const std::string& reason) {
  return Error(function + ": " + path + ": " + (reason.empty() ? "file operation failed" : reason));
}

inline std::string os_reason() {
  return errno ? std::strerror(errno) : "I/O error";
}

inline std::string reason(std::errc code) { return std::make_error_code(code).message(); }

inline void reject_nul(const std::string& function, const std::string& path) {
  if (path.find('\0') != std::string::npos) throw failure(function, path, reason(std::errc::invalid_argument));
}

#ifdef _WIN32
inline std::filesystem::path native_path(const std::string& function, const std::string& path) {
  reject_nul(function, path);
  if (path.empty()) return std::filesystem::path();
  std::filesystem::path converted;
  try { converted = std::filesystem::u8path(path); }
  catch (const std::filesystem::filesystem_error& e) { throw failure(function, path, e.code().message()); }
  catch (const std::range_error&) { throw failure(function, path, reason(std::errc::illegal_byte_sequence)); }
  std::error_code ec;
  std::filesystem::path absolute = std::filesystem::absolute(converted, ec);
  if (ec) throw failure(function, path, ec.message());
  absolute = absolute.lexically_normal().make_preferred();
  std::wstring wide = absolute.native();
  if (wide.size() >= 248 && wide.rfind(L"\\\\?\\", 0) != 0) {
    if (wide.rfind(L"\\\\", 0) == 0) wide = L"\\\\?\\UNC" + wide.substr(1);
    else wide = L"\\\\?\\" + wide;
  }
  return std::filesystem::path(wide);
}
#else
inline std::filesystem::path native_path(const std::string& function, const std::string& path) {
  reject_nul(function, path);
  return std::filesystem::path(path);
}
#endif

template<class Path> inline void ensure_regular_target(const Path& path, const std::string& function, const std::string& original, bool allow_missing) {
  std::error_code ec;
  auto result = std::filesystem::status(path, ec);
  if (ec == std::errc::no_such_file_or_directory && allow_missing) return;
  if (ec) throw failure(function, original, ec.message());
  if (!std::filesystem::is_regular_file(result)) {
    const auto code = std::filesystem::is_directory(result) ? std::errc::is_a_directory : std::errc::invalid_argument;
    throw failure(function, original, reason(code));
  }
}

template<class Path> inline std::string read_path(const Path& native, const std::string& original, const std::string& function) {
  ensure_regular_target(native, function, original, false);
  errno = 0;
  std::ifstream input(native, std::ios::binary);
  if (!input) throw failure(function, original, os_reason());
  std::string data;
  char buffer[64 * 1024];
  while (input.read(buffer, sizeof buffer) || input.gcount()) data.append(buffer, static_cast<size_t>(input.gcount()));
  if (input.bad() || !input.eof()) throw failure(function, original, os_reason());
  // EOF leaves failbit set after the final read; clear it before checking close.
  input.clear();
  input.close();
  if (input.fail()) throw failure(function, original, os_reason());
  return data;
}

template<class Path> inline void write_path(const Path& native, const std::string& bytes, const std::string& original, const std::string& function, std::ios::openmode mode) {
  ensure_regular_target(native, function, original, true);
  errno = 0;
  std::ofstream output(native, std::ios::binary | mode);
  if (!output) throw failure(function, original, os_reason());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) throw failure(function, original, os_reason());
  output.flush();
  if (!output) throw failure(function, original, os_reason());
  output.close();
  if (output.fail()) throw failure(function, original, os_reason());
}

inline Value lines_value(Collections& arena, const std::string& bytes) {
  std::vector<std::string> lines;
  size_t start = 0;
  for (size_t i = 0; i < bytes.size(); ++i) if (bytes[i] == '\n') {
    lines.push_back(bytes.substr(start, i - start));
    start = i + 1;
  }
  lines.push_back(bytes.substr(start));
  return Value::object("StrList", arena.make_str_list(std::move(lines)));
}

} // namespace file_detail

inline Registry& register_file(Registry& r, Collections& arena) {
  if (r.has_std()) return r;
  r.function("file_read", {"string"}, "string", [](Args& a) {
    const auto& path = a[0].s;
    return Value::string(file_detail::read_path(file_detail::native_path("file_read", path), path, "file_read"));
  });
  r.function("file_write", {"string", "string"}, "void", [](Args& a) {
    const auto& path = a[0].s;
    file_detail::write_path(file_detail::native_path("file_write", path), a[1].s, path, "file_write", std::ios::trunc);
    return Value::none();
  });
  r.function("file_append", {"string", "string"}, "void", [](Args& a) {
    const auto& path = a[0].s;
    file_detail::write_path(file_detail::native_path("file_append", path), a[1].s, path, "file_append", std::ios::app);
    return Value::none();
  });
  r.function("file_exists", {"string"}, "bool", [](Args& a) {
    const auto& path = a[0].s;
    if (path.empty()) return Value::boolean(false);
    auto native = file_detail::native_path("file_exists", path);
    std::error_code ec;
    auto result = std::filesystem::status(native, ec);
    if (!ec) return Value::boolean(std::filesystem::is_regular_file(result));
    if (ec == std::errc::no_such_file_or_directory || ec == std::errc::not_a_directory) return Value::boolean(false);
    throw file_detail::failure("file_exists", path, ec.message());
  });
  r.function("file_lines", {"string"}, "StrList", [&arena](Args& a) {
    const auto& path = a[0].s;
    auto native = file_detail::native_path("file_lines", path);
    return file_detail::lines_value(arena, file_detail::read_path(native, path, "file_lines"));
  });
  return r;
}

} // namespace dslvm::stdlib

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
