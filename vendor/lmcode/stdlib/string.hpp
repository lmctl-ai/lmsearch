#pragma once
// Optional host-side string catalog for lmscript. These are ordinary host
// functions: copy the signatures from string.schema-fragment.json into the
// host schema, then call register_string(registry).
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#endif
#include "dslvm.hpp"
#ifndef DSLVM_STD_BUILDING
#undef DSLVM_STD_DEFER
#endif
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif
#include <string>
#include <system_error>

namespace dslvm::stdlib {

inline bool str_starts_with(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}
inline bool str_ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
inline bool str_contains(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }
inline int64_t str_len(const std::string& s) { return static_cast<int64_t>(s.size()); }

// Byte offsets. Negative starts clamp to 0; starts beyond the end return empty;
// negative counts return empty; counts beyond the remaining bytes clamp.
inline std::string str_substr(const std::string& s, int64_t start, int64_t count) {
  if (start < 0) start = 0;
  if (count <= 0 || static_cast<uint64_t>(start) >= s.size()) return {};
  const size_t offset = static_cast<size_t>(start), available = s.size() - offset;
  return s.substr(offset, static_cast<uint64_t>(count) > available ? available : static_cast<size_t>(count));
}
inline std::string str_lower(const std::string& s) {
  std::string out = s;
  for (char& c : out) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  return out;
}
inline std::string str_upper(const std::string& s) {
  std::string out = s;
  for (char& c : out) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  return out;
}
inline int64_t str_find(const std::string& s, const std::string& needle) {
  const auto p = s.find(needle);
  return p == std::string::npos ? -1 : static_cast<int64_t>(p);
}
// Replaces every non-overlapping occurrence; an empty search string is a no-op.
inline std::string str_replace(const std::string& s, const std::string& from, const std::string& to) {
  constexpr size_t max_bytes = 64u * 1024u * 1024u;
  size_t result_size = 0;
  auto add_size = [&](size_t n) {
    if (n > max_bytes - result_size) throw Error("str_replace: result must be at most 67108864 bytes");
    result_size += n;
  };
  if (from.empty()) { add_size(s.size()); return s; }
  // Preflight the exact result size, without overflowing or allocating output.
  size_t at = 0, found;
  while ((found = s.find(from, at)) != std::string::npos) {
    add_size(found - at); add_size(to.size()); at = found + from.size();
  }
  add_size(s.size() - at);
  std::string out; out.reserve(result_size);
  at = 0;
  while ((found = s.find(from, at)) != std::string::npos) {
    out.append(s, at, found - at); out += to; at = found + from.size();
  }
  out.append(s, at, std::string::npos);
  return out;
}
inline std::string str_trim(const std::string& s) {
  auto ws = [](unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };
  size_t a = 0, b = s.size();
  while (a < b && ws(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && ws(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}
inline std::string str_repeat(const std::string& s, int64_t count) {
  constexpr size_t max_bytes = 64u * 1024u * 1024u;
  if (s.empty() && count >= 0) return {};
  if (count < 0 || (s.size() && static_cast<uint64_t>(count) > max_bytes / s.size()))
    throw Error("str_repeat: count must be nonnegative and result at most 67108864 bytes");
  std::string out; out.reserve(s.size() * static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) out += s;
  return out;
}
inline bool str_is_int(const std::string& s) {
  if (s.empty()) return false;
  const char* begin = s.data();
  if (s[0] == '+') { if (s.size() == 1 || s[1] == '-' || s[1] == '+') return false; ++begin; }
  int64_t value = 0;
  const auto r = std::from_chars(begin, s.data() + s.size(), value, 10);
  return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
inline int64_t str_to_int(const std::string& s) {
  if (!str_is_int(s)) return 0;
  int64_t value = 0; const char* begin = s.data() + (s[0] == '+' ? 1 : 0);
  std::from_chars(begin, s.data() + s.size(), value, 10); return value;
}
namespace detail {
// Per-object C locale: conversion never reads or changes the process locale.
// Unlike a global setlocale switch, this is safe for concurrent host callbacks.
struct FloatLocale {
#if defined(_WIN32)
  _locale_t value = _create_locale(LC_NUMERIC, "C");
  ~FloatLocale() { if (value) _free_locale(value); }
#else
  locale_t value = newlocale(LC_NUMERIC_MASK, "C", nullptr);
  ~FloatLocale() { if (value) freelocale(value); }
#endif
  FloatLocale() = default;
  FloatLocale(const FloatLocale&) = delete;
  FloatLocale& operator=(const FloatLocale&) = delete;
};
// strtod_l is available on POSIX; Windows spells it _strtod_l. Validate decimal
// syntax first to exclude whitespace, hexadecimal floats, NaN/Inf and prefixes.
inline bool parse_float(const std::string& s, double& value) {
  size_t i = 0;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
  bool digits = false, nonzero = false;
  auto scan_digits = [&]() {
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      digits = true;
      nonzero = nonzero || s[i] != '0';
      ++i;
    }
  };
  scan_digits();
  if (i < s.size() && s[i] == '.') { ++i; scan_digits(); }
  if (!digits) return false;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    ++i;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
    const size_t exponent = i;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i == exponent) return false;
  }
  if (i != s.size()) return false;
  static const FloatLocale locale;
  if (!locale.value) throw Error("str_to_float: cannot create C numeric locale");
  char* end = nullptr;
#if defined(_WIN32)
  value = _strtod_l(s.c_str(), &end, locale.value);
#else
  value = strtod_l(s.c_str(), &end, locale.value);
#endif
  // ERANGE alone is not a rejection: some C libraries set it for representable
  // subnormals. Match from_chars by rejecting only overflow/underflow to zero.
  return end == s.c_str() + s.size() && std::isfinite(value) &&
         (value != 0.0 || !nonzero);
}
} // namespace detail
inline bool str_is_float(const std::string& s) {
  double value = 0;
  return detail::parse_float(s, value);
}
inline double str_to_float(const std::string& s) {
  double value = 0;
  return detail::parse_float(s, value) ? value : 0.0;
}

namespace detail {
struct GlobWork {
  uint64_t steps = 0;
  void step() {
    if (steps == (uint64_t(1) << 27)) throw Error("str_match: pattern exceeds the 2^27 work limit");
    ++steps;
  }
};
// One glob token (never '*'): ?; escaped byte; or [class], [!class], ranges.
// Unterminated '[' is a literal, ']' is literal at the start, descending ranges
// are empty, matching lmhtml's documented byte-oriented catalog behavior.
inline bool glob_token(const std::string& p, size_t i, unsigned char c, size_t& next, GlobWork& work) {
  if (p[i] == '\\' && i + 1 < p.size()) { next = i + 2; return static_cast<unsigned char>(p[i + 1]) == c; }
  if (p[i] == '?') { next = i + 1; return true; }
  if (p[i] == '[') {
    size_t q = i + 1; bool negate = q < p.size() && p[q] == '!'; if (negate) ++q;
    size_t first = q; bool hit = false;
    while (q < p.size() && (p[q] != ']' || q == first)) {
      work.step();
      auto byte = [&]() {
        if (p[q] == '\\' && q + 1 < p.size()) ++q;
        return static_cast<unsigned char>(p[q++]);
      };
      const unsigned char lo = byte();
      if (q + 1 < p.size() && p[q] == '-' && p[q + 1] != ']') {
        ++q; const unsigned char hi = byte();
        if (lo <= hi && c >= lo && c <= hi) hit = true;
      } else if (lo == c) hit = true;
    }
    if (q < p.size()) { next = q + 1; return hit != negate; }
  }
  next = i + 1; return static_cast<unsigned char>(p[i]) == c;
}
}
// *, ?, classes and backslash escapes operate on bytes (including NUL). The
// last-star algorithm uses O(1) auxiliary memory. Charge each matcher iteration,
// character-class entry scanned, and trailing star skipped to a 2^27 work limit.
inline bool str_match(const std::string& text, const std::string& pattern) {
  detail::GlobWork work;
  size_t t = 0, p = 0, star_p = std::string::npos, star_t = 0;
  while (t < text.size()) {
    work.step();
    size_t next = 0;
    if (p < pattern.size() && pattern[p] == '*') { star_p = p++; star_t = t; }
    else if (p < pattern.size() && detail::glob_token(pattern, p, static_cast<unsigned char>(text[t]), next, work)) { p = next; ++t; }
    else if (star_p != std::string::npos) { p = star_p + 1; t = ++star_t; }
    else return false;
  }
  while (p < pattern.size() && pattern[p] == '*') { work.step(); ++p; }
  return p == pattern.size();
}

inline Registry& register_string(Registry& r) {
  if (r.has_std()) return r;
  r.function("str_starts_with", {"string", "string"}, "bool", [](Args& a) { return Value::boolean(str_starts_with(a[0].s, a[1].s)); });
  r.function("str_ends_with", {"string", "string"}, "bool", [](Args& a) { return Value::boolean(str_ends_with(a[0].s, a[1].s)); });
  r.function("str_contains", {"string", "string"}, "bool", [](Args& a) { return Value::boolean(str_contains(a[0].s, a[1].s)); });
  r.function("str_lower", {"string"}, "string", [](Args& a) { return Value::string(str_lower(a[0].s)); });
  r.function("str_upper", {"string"}, "string", [](Args& a) { return Value::string(str_upper(a[0].s)); });
  r.function("str_substr", {"string", "int", "int"}, "string", [](Args& a) { return Value::string(str_substr(a[0].s, a[1].i, a[2].i)); });
  r.function("str_len", {"string"}, "int", [](Args& a) { return Value::integer(str_len(a[0].s)); });
  r.function("str_match", {"string", "string"}, "bool", [](Args& a) { return Value::boolean(str_match(a[0].s, a[1].s)); });
  r.function("str_find", {"string", "string"}, "int", [](Args& a) { return Value::integer(str_find(a[0].s, a[1].s)); });
  r.function("str_replace", {"string", "string", "string"}, "string", [](Args& a) { return Value::string(str_replace(a[0].s, a[1].s, a[2].s)); });
  r.function("str_trim", {"string"}, "string", [](Args& a) { return Value::string(str_trim(a[0].s)); });
  r.function("str_repeat", {"string", "int"}, "string", [](Args& a) { return Value::string(str_repeat(a[0].s, a[1].i)); });
  r.function("str_to_int", {"string"}, "int", [](Args& a) { return Value::integer(str_to_int(a[0].s)); });
  r.function("str_is_int", {"string"}, "bool", [](Args& a) { return Value::boolean(str_is_int(a[0].s)); });
  r.function("str_to_float", {"string"}, "float", [](Args& a) { return Value::real(str_to_float(a[0].s)); });
  r.function("str_is_float", {"string"}, "bool", [](Args& a) { return Value::boolean(str_is_float(a[0].s)); });
  return r;
}

} // namespace dslvm::stdlib

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
