#pragma once
#include <atomic>
// The complete built-in standard. Include dslvm.hpp normally; individual catalog
// headers also finish this aggregator after their own declarations are complete.
#define DSLVM_STD_BUILDING
#include "string.hpp"
#include "math.hpp"
#include "collections.hpp"
#include "file.hpp"
#include "containers.hpp"
#include "json.hpp"
#include "encoding.hpp"
#include "result.hpp"
#undef DSLVM_STD_BUILDING

namespace dslvm {
namespace stdlib {
struct StdRuntime {
  Collections collections;
  ResultSink* result_sink = nullptr; // external lifetime, outside collection reset
  std::atomic<bool> running{false};
};
} // namespace stdlib

inline Registry::Registry(StdMode mode) {
  if (mode == StdMode::Enabled) register_std(*this);
}

inline Registry& register_std(Registry& registry) {
  if (registry.has_std()) return registry;
  // Build transactionally: a conflicting host declaration leaves the original intact.
  Registry candidate(StdMode::Disabled);
  candidate = registry;
  candidate.std_runtime_ = std::make_shared<stdlib::StdRuntime>();
  const auto before = candidate.signatures();
  stdlib::register_string(candidate);
  stdlib::register_encoding(candidate);
  stdlib::register_math(candidate);
  stdlib::register_collections(candidate, candidate.std_runtime_->collections);
  stdlib::register_file(candidate, candidate.std_runtime_->collections);
  stdlib::register_containers(candidate, candidate.std_runtime_->collections);
  stdlib::register_json(candidate, candidate.std_runtime_->collections);
  stdlib::register_results(candidate, candidate.std_runtime_->result_sink);
  const auto after = candidate.signatures();
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(candidate.std_signatures_));
  candidate.std_initialized_ = true;
  registry = std::move(candidate);
  return registry;
}

inline stdlib::Collections& Registry::std_collections() const {
  if (!has_std()) throw Error("std: registry has standard library disabled");
  return std_runtime_->collections;
}
inline void Registry::set_result_sink(ResultSink* sink) {
  if (!has_std()) throw Error("std: registry has standard library disabled");
  std_runtime_->result_sink = sink;
}
inline void Registry::set_std_output(std::ostream* sink) {
  std_output_ = sink ? sink : &std::cout;
  auto* output = std_output_;
  for (const char* sig : {"fn print(string) -> void", "fn println(string) -> void"}) {
    auto it = functions_.find(sig);
    if (it != functions_.end()) {
      it->second.entry.fn = [output](Args& args) {
        output->write(args[0].s.data(), static_cast<std::streamsize>(args[0].s.size()));
        output->put('\n'); return Value::none();
      };
    }
  }
}
inline void Registry::begin_std_run() const {
  bool idle = false;
  if (!std_runtime_->running.compare_exchange_strong(idle, true, std::memory_order_acquire))
    throw Error("std: registry already has an active run");
  std_runtime_->collections.reset();
}
inline void Registry::end_std_run() const noexcept {
  std_runtime_->collections.reset();
  std_runtime_->running.store(false, std::memory_order_release);
}
} // namespace dslvm
