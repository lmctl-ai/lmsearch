#pragma once
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#endif
#include "dslvm.hpp"
#ifndef DSLVM_STD_BUILDING
#undef DSLVM_STD_DEFER
#endif
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace dslvm::stdlib {

namespace detail {
inline size_t checked_base64_size(size_t input_size) {
  const size_t groups = input_size / 3 + (input_size % 3 != 0 ? 1 : 0);
  if (groups > std::numeric_limits<size_t>::max() / 4)
    throw Error("str_base64_encode: size overflow");
  return groups * 4;
}
inline size_t checked_hex_size(size_t input_size) {
  if (input_size > std::numeric_limits<size_t>::max() / 2)
    throw Error("str_hex_encode: size overflow");
  return input_size * 2;
}
} // namespace detail

inline std::string str_base64_encode(const std::string& input) {
  static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const size_t output_size = detail::checked_base64_size(input.size());
  std::string out;
  out.reserve(output_size);
  for (size_t i = 0; i < input.size(); i += 3) {
    const size_t remaining = input.size() - i;
    const uint32_t a = static_cast<unsigned char>(input[i]);
    const uint32_t b = remaining > 1 ? static_cast<unsigned char>(input[i + 1]) : 0;
    const uint32_t c = remaining > 2 ? static_cast<unsigned char>(input[i + 2]) : 0;
    out.push_back(alphabet[(a >> 2) & 63]);
    out.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
    out.push_back(remaining > 1 ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
    out.push_back(remaining > 2 ? alphabet[c & 63] : '=');
  }
  return out;
}

inline int encoding_base64_value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

inline std::string str_base64_decode(const std::string& input) {
  if (input.size() % 4 != 0) throw Error("str_base64_decode: invalid encoding");
  std::string out;
  if (input.size() / 4 > std::numeric_limits<size_t>::max() / 3)
    throw Error("str_base64_decode: invalid encoding");
  out.reserve((input.size() / 4) * 3);
  for (size_t i = 0; i < input.size(); i += 4) {
    const bool last = i + 4 == input.size();
    const unsigned char c0 = static_cast<unsigned char>(input[i]);
    const unsigned char c1 = static_cast<unsigned char>(input[i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(input[i + 2]);
    const unsigned char c3 = static_cast<unsigned char>(input[i + 3]);
    const int a = encoding_base64_value(c0), b = encoding_base64_value(c1);
    if (a < 0 || b < 0) throw Error("str_base64_decode: invalid encoding");
    if (c2 == '=') {
      if (!last || c3 != '=' || (b & 15) != 0) throw Error("str_base64_decode: invalid encoding");
      out.push_back(static_cast<char>((a << 2) | (b >> 4)));
      continue;
    }
    const int d = encoding_base64_value(c2);
    if (d < 0) throw Error("str_base64_decode: invalid encoding");
    out.push_back(static_cast<char>((a << 2) | (b >> 4)));
    out.push_back(static_cast<char>((b << 4) | (d >> 2)));
    if (c3 == '=') {
      if (!last || (d & 3) != 0) throw Error("str_base64_decode: invalid encoding");
      continue;
    }
    const int e = encoding_base64_value(c3);
    if (e < 0) throw Error("str_base64_decode: invalid encoding");
    out.push_back(static_cast<char>((d << 6) | e));
  }
  return out;
}

inline std::string str_hex_encode(const std::string& input) {
  const size_t output_size = detail::checked_hex_size(input.size());
  static constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(output_size);
  for (unsigned char c : input) { out.push_back(digits[c >> 4]); out.push_back(digits[c & 15]); }
  return out;
}

inline int encoding_hex_value(unsigned char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

inline std::string str_hex_decode(const std::string& input) {
  if (input.size() % 2 != 0) throw Error("str_hex_decode: invalid encoding");
  std::string out;
  out.reserve(input.size() / 2);
  for (size_t i = 0; i < input.size(); i += 2) {
    const int hi = encoding_hex_value(static_cast<unsigned char>(input[i]));
    const int lo = encoding_hex_value(static_cast<unsigned char>(input[i + 1]));
    if (hi < 0 || lo < 0) throw Error("str_hex_decode: invalid encoding");
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  return out;
}

inline Registry& register_encoding(Registry& r) {
  if (r.has_std()) return r;
  r.function("str_base64_encode", {"string"}, "string", [](Args& a) { return Value::string(str_base64_encode(a[0].s)); });
  r.function("str_base64_decode", {"string"}, "string", [](Args& a) { return Value::string(str_base64_decode(a[0].s)); });
  r.function("str_hex_encode", {"string"}, "string", [](Args& a) { return Value::string(str_hex_encode(a[0].s)); });
  r.function("str_hex_decode", {"string"}, "string", [](Args& a) { return Value::string(str_hex_decode(a[0].s)); });
  return r;
}

} // namespace dslvm::stdlib

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
