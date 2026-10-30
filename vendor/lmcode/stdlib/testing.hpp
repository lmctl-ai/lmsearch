#pragma once
// Optional host-side testing helpers. Copy testing.schema-fragment.json into
// the host schema and call register_testing(registry).
//
// The sink pointer is captured by the registered callbacks; when supplied it
// must remain alive and writable for every call made through the registry. A
// null sink selects std::cout at registration time.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#endif
#include "dslvm.hpp"
#ifndef DSLVM_STD_BUILDING
#undef DSLVM_STD_DEFER
#endif
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <iostream>
#include <ostream>
#include <string>

namespace dslvm::stdlib {

inline void testing_print(std::ostream& sink, const std::string& value) {
  sink.write(value.data(), static_cast<std::streamsize>(value.size()));
  sink.put('\n');
}

inline void testing_assert_close(double a, double b, double tolerance) {
  // Keep this deliberately as an explicit comparison: equal infinities
  // subtract to NaN and therefore fail, as do NaN operands and tolerances.
  if (std::fabs(a - b) <= tolerance) return;
  // Match %.15g in the C locale without reading or changing the host's locale.
  std::ostringstream message;
  message.imbue(std::locale::classic());
  message << std::setprecision(15);
  auto number = [&](double value) {
    if (std::isnan(value)) message << "nan"; // NaN sign/payload are not portable.
    else message << value;
  };
  message << "assert_close failed: "; number(a);
  message << " vs "; number(b);
  message << " (tol "; number(tolerance); message << ")";
  throw Error(message.str());
}

inline Registry& register_testing(Registry& r, std::ostream* sink = nullptr) {
  std::ostream* output = sink ? sink : &std::cout;
  r.function("print", {"string"}, "void", [output](Args& a) {
    testing_print(*output, a[0].s);
    return Value::none();
  });
  r.function("assert_close", {"float", "float", "float"}, "void", [](Args& a) {
    testing_assert_close(a[0].f, a[1].f, a[2].f);
    return Value::none();
  });
  return r;
}

} // namespace dslvm::stdlib

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
