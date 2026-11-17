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
#include "invocation.hpp"
#include "rest.hpp"
#undef DSLVM_STD_BUILDING

namespace dslvm {
namespace stdlib {
struct StdRuntime {
  Collections collections;
  ResultSink* result_sink = nullptr; // external lifetime, outside collection reset
  std::atomic<bool> running{false};
};
namespace rest {
inline void RestController::call_entry_isolated(size_t entry_idx, Exchange& x) {
  auto rt = registry_ ? registry_->std_runtime() : nullptr;
  const bool was_running = rt && rt->running.load(std::memory_order_relaxed);
  Collections saved_collections;
  struct Restore {
    std::shared_ptr<StdRuntime> rt;
    Collections& saved;
    bool was_running;
    ~Restore() {
      if (was_running && rt) {
        std::swap(rt->collections, saved);
        rt->running.store(true, std::memory_order_release);
      }
    }
  } restore{rt, saved_collections, was_running};
  if (was_running && rt) {
    std::swap(rt->collections, saved_collections);
    rt->running.store(false, std::memory_order_release);
  }
  program_->call(entry_idx, context_, *registry_, {Value::object("RestExchange", &x)});
}
} // namespace rest
} // namespace stdlib

inline Registry::Registry(StdMode mode) {
  std_runtime_ = std::make_shared<stdlib::StdRuntime>();
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
  auto rest_ctrl = stdlib::rest::register_rest(candidate);
  const auto after = candidate.signatures();
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(candidate.std_signatures_));
  candidate.std_initialized_ = true;
  registry = std::move(candidate);
  rest_ctrl->bind_registry(&registry);
  return registry;
}

inline stdlib::Collections& Registry::std_collections() const {
  if (!has_std()) throw Error("std: registry has standard library disabled");
  return std_runtime_->collections;
}
inline stdlib::Collections& Registry::dry_collections() const {
  if (!std_runtime_) throw Error("std: collection arena unavailable");
  return std_runtime_->collections;
}
inline Value dry_dummy(const Registry& registry, const std::string& declared) {
  if (declared == "void") return Value::none();
  if (declared == "null") return Value::dry_null(declared);
  if (declared == "bool") return Value::boolean(true);
  if (declared == "int" || declared == "integer" || declared == "int64") return Value::integer(0);
  if (declared == "float" || declared == "double" || declared == "number") return Value::real(0.0);
  if (declared == "string") return Value::string("");
  auto& arena = registry.dry_collections();
  if (declared == "StrList") return Value::object(declared, arena.make_str_list());
  if (declared == "IntList") return Value::object(declared, arena.make_int_list());
  if (declared == "FloatList") return Value::object(declared, arena.make_float_list());
  if (declared == "StrMap") return Value::object(declared, arena.make_str_map());
  if (declared == "IntMap") return Value::object(declared, arena.make_object<stdlib::IntMap>("IntMap"));
  if (declared == "FloatMap") return Value::object(declared, arena.make_object<stdlib::FloatMap>("FloatMap"));
  return Value::dry_null(declared);
}
inline void Registry::set_result_sink(ResultSink* sink) {
  if (!has_std()) throw Error("std: registry has standard library disabled");
  if(sink && sink->sealed()) throw Error("lmcode return: result sink is sealed");
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
      if (it->second.entry.dry_fn) it->second.entry.dry_fn = it->second.entry.fn;
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
