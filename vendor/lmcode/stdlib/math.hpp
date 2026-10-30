#pragma once
// Optional ordinary host functions. Copy math.schema-fragment.json into the
// host schema and register once per registry; no VM opcodes are involved.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#endif
#include "dslvm.hpp"
#ifndef DSLVM_STD_BUILDING
#undef DSLVM_STD_DEFER
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dslvm::stdlib {

inline int64_t math_abs(int64_t x) { return x < 0 ? from_twos_complement(0 - uint64_t(x)) : x; }
inline double math_abs(double x) { return std::fabs(x); }
inline int64_t math_min(int64_t a, int64_t b) { return std::min(a, b); }
inline int64_t math_max(int64_t a, int64_t b) { return std::max(a, b); }
// Propagate either NaN; otherwise ties retain the first operand, including its zero sign.
inline double math_min(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
  return a <= b ? a : b;
}
inline double math_max(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
  return a >= b ? a : b;
}
inline int64_t math_floor(int64_t x) { return x; }
inline int64_t math_ceil(int64_t x) { return x; }
inline int64_t math_round(int64_t x) { return x; }
inline double math_floor(double x) { return std::floor(x); }
inline double math_ceil(double x) { return std::ceil(x); }
// Halfway cases round away from zero; the return type remains binary64.
inline double math_round(double x) { return std::round(x); }
inline int64_t math_pow(int64_t base, int64_t exponent) {
  if (exponent < 0) throw Error("math_pow: integer exponent must be nonnegative");
  uint64_t result = 1, factor = uint64_t(base), remaining = uint64_t(exponent);
  while (remaining) {
    if (remaining & 1) result *= factor;
    remaining >>= 1;
    if (remaining) factor *= factor;
  }
  return from_twos_complement(result);
}
inline double math_pow(double base, double exponent) {
  if (exponent == 0.0 || base == 1.0 || (base == -1.0 && std::isinf(exponent))) return 1.0;
  if (std::isnan(base) || std::isnan(exponent)) return std::numeric_limits<double>::quiet_NaN();
  return std::pow(base, exponent);
}
// Exact floor square root: division avoids overflow, including at INT64_MAX.
inline int64_t math_sqrt(int64_t x) {
  if (x < 0) throw Error("math_sqrt: integer argument must be nonnegative");
  if (x < 2) return x;
  int64_t low = 1, high = std::min(x, INT64_C(3037000499)), result = 1;
  while (low <= high) {
    const int64_t mid = low + (high - low) / 2;
    if (mid <= x / mid) { result = mid; low = mid + 1; }
    else high = mid - 1;
  }
  return result;
}
inline double math_sqrt(double x) { return std::sqrt(x); }

inline Registry& register_math(Registry& r) {
  if (r.has_std()) return r;
  r.function("math_abs", {"int"}, "int", [](Args& a) { return Value::integer(math_abs(a[0].i)); });
  r.function("math_abs", {"float"}, "float", [](Args& a) { return Value::real(math_abs(a[0].f)); });
  r.function("math_min", {"int", "int"}, "int", [](Args& a) { return Value::integer(math_min(a[0].i, a[1].i)); });
  r.function("math_min", {"float", "float"}, "float", [](Args& a) { return Value::real(math_min(a[0].f, a[1].f)); });
  r.function("math_max", {"int", "int"}, "int", [](Args& a) { return Value::integer(math_max(a[0].i, a[1].i)); });
  r.function("math_max", {"float", "float"}, "float", [](Args& a) { return Value::real(math_max(a[0].f, a[1].f)); });
  r.function("math_floor", {"int"}, "int", [](Args& a) { return Value::integer(math_floor(a[0].i)); });
  r.function("math_floor", {"float"}, "float", [](Args& a) { return Value::real(math_floor(a[0].f)); });
  r.function("math_ceil", {"int"}, "int", [](Args& a) { return Value::integer(math_ceil(a[0].i)); });
  r.function("math_ceil", {"float"}, "float", [](Args& a) { return Value::real(math_ceil(a[0].f)); });
  r.function("math_round", {"int"}, "int", [](Args& a) { return Value::integer(math_round(a[0].i)); });
  r.function("math_round", {"float"}, "float", [](Args& a) { return Value::real(math_round(a[0].f)); });
  r.function("math_pow", {"int", "int"}, "int", [](Args& a) { return Value::integer(math_pow(a[0].i, a[1].i)); });
  r.function("math_pow", {"float", "float"}, "float", [](Args& a) { return Value::real(math_pow(a[0].f, a[1].f)); });
  r.function("math_sqrt", {"int"}, "int", [](Args& a) { return Value::integer(math_sqrt(a[0].i)); });
  r.function("math_sqrt", {"float"}, "float", [](Args& a) { return Value::real(math_sqrt(a[0].f)); });
  return r;
}

} // namespace dslvm::stdlib

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
