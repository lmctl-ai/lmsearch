// lmcode-cpp: the C++ runtime for lmscript `_lmcode_` bytecode. Header-only, C++17, standard library only.
// Single source of truth: github.com/lmctl-ai/lmcode-cpp/include/dslvm.hpp (hosts vendor it by pinned sha256, see README).
//
// Public API (everything else in this header is an implementation detail):
//   dslvm::Error                         exception for load, verify and run-time failures: what() names the problem; code is one of
//                                        LOAD_BAD_BYTECODE, LOAD_SCHEMA_MISMATCH, LOAD_UNBOUND, RUN_ERROR (VM), RUN_HOST (thrown by a host
//                                        closure), APP_FAIL (the host's own application failure, see report_app_failure)
//   dslvm::Value                         tagged value: VOID, INT (int64, two's complement), FLOAT (binary64), STR (arbitrary bytes), BOOL, OBJ (host pointer + schema type name)
//   dslvm::Args                          std::vector<Value> of call arguments, in order
//   dslvm::Registry                      the implementation of the schema, declared with its types: function(name, {param types}, return
//                                        type, fn), method(type, name, {param types}, return type, fn), context(name, type); entries are
//                                        keyed by canonical signature, distinct parameter tuples overload a name, and duplicate parameter
//                                        tuples are refused; signatures() lists the canonical strings `lmscript --lmcode-signatures SCHEMA` prints, so a
//                                        host test diffs registry and schema (a consistency check between two sources of truth,
//                                        not a run-time defense: the VM checks nothing at call time)
//   dslvm::Context                       the live objects of one run: set(name, pointer)
//   dslvm::Program::load(bytes, schema, registry)  parses, binds and verifies; throws before anything runs. The Program keeps
//                                        pointers to entries in that Registry: keep the same Registry alive and unchanged until
//                                        the Program is destroyed.
//   dslvm::Program::run(context, registry)         executes the chunk once; pass the same unchanged Registry used by load()
//                                        (the argument remains in the public host API); throws dslvm::Error on a run-time failure
//   dslvm::Program::entries()              the rule entries of the file (v6): {name, annotation, iface, params, returns}
//   dslvm::Program::call(entry, context, registry, args, stats)   executes one entry with args stored into its parameter
//                                        locals; returns the stack value at HALT (Value::none() for a void interface)
//   dslvm::Program::RunStats             optional dispatch count for host measurements; pass a pointer to run(), never a limit
//   dslvm::report_ok / report_error / report_app_failure   the standard one-line JSON result on stdout and the
//                                        exit-code convention 2 compile (lmscript) / 3 host (load or run) / 4 application
//   dslvm::VERSION                       the bytecode format version this SDK reads (and nothing else: no backward compatibility)
//   dslvm::sha256 / hex / read_file      helpers the hosts use for the schema binding and files
//
// Bytecode v6 (big-endian, fixed widths, IEEE-754 binary64): magic "DSLB", u16 version 6, sha256(schema bytes);
// constants: u32 count, each u8 tag (1 = string) + u32 length + raw bytes; imports: u16 count, each u32 constant index of a canonical
// signature (`fn NAME(T1, T2) -> R`, `method Type.name(T1) -> R`) resolved against the registry at load; context: u16 count,
// each (u32 name constant, u32 type constant) checked against the registry's declarations; locals: u16 count, each (u32 type-name
// constant, u8 hidden flag); u16 max_stack (the compiler's measured operand-stack depth: the VM reserves it, nothing is checked
// against it); code: u32 length + code ending with HALT; lines: u32 count, each (u32 pc, u32 line, u16 column), may be empty.
// Then the entries section (v6): u16 entry count; each entry is a self-contained mini-program sharing the file's constant,
// import and context pools: u32 name constant, u32 annotation constant, u32 interface constant (canonical
// `interface I(T1, T2) -> R` text), u16 nparams, u16 nlocals + (u32 type constant, u8 hidden) x nlocals (parameter slots
// 0..nparams-1 are unhidden; the host stores the arguments there before execution), u16 max_stack, u32 code length + code
// ending with HALT (a non-void entry leaves exactly one value on the stack there: the result), u32 nlines +
// (u32 pc, u32 line, u16 column) x nlines. PCs and jumps are relative to their own region. The file ends after the last
// entry; a file without rules has u16 0.
// Opcodes: enum Op below (PUSH_INT carries an i64, PUSH_STR a u32 constant index, LOAD_CTX a u16 context index, CALL_FN and
// CALL_METHOD a u16 import index + u8 argc). FOR_RANGE carries counter/end local slots and an exit displacement; FOR_NEXT carries
// the hidden end slot and a back displacement. WHILE_BEGIN marks a paired loop head, WHILE_TEST branches to its exit on false,
// and LOOP_BACK returns only to that paired head. Int arithmetic is 64-bit two's complement with wraparound (no overflow trap);
// division and remainder by zero are run-time errors. The language is defined by the lmscript compiler
// (_lmcode_ blocks, npm package @lmctl-ai/lmctl); this header implements the bytecode that compiler emits.
//
// Guarantees of load(): wrong magic, other version (including v3), schema-hash mismatch, truncation, trailing bytes, unknown
// constant tags, undecodable instructions, out-of-range constants/locals/imports/context indices, jumps that do not land on an
// instruction start or go backwards except for paired structured loops, missing HALT, context declarations
// and import signatures the registry does not provide are all refused with a message. The language and the VM impose no
// memory or time limits: there is no stack cap, no locals cap and no string-size cap.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dslvm {

struct Error : std::runtime_error {
  std::string code;
  std::string message;
  bool has_location = false;
  bool has_column = false;
  uint32_t line = 0;
  uint16_t column = 0;
  explicit Error(const std::string& m) : std::runtime_error(m), code("RUN_HOST"), message(m) {}   // a plain message is what host closures throw
  Error(const std::string& c, const std::string& m) : std::runtime_error(m), code(c), message(m) {}
  const char* what() const noexcept override { return message.c_str(); }
  void locate(uint32_t l, uint16_t c) { if (has_location || l == 0) return; has_location = true; has_column = c != 0; line = l; column = c; message = "dsl line " + std::to_string(l) + (has_column ? ":" + std::to_string(c) : "") + ": " + message; }
};

// ---- SHA-256 (FIPS 180-4), used only to bind the bytecode to the exact schema bytes ----
inline std::string sha256(const std::string& data) {
  static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::string msg = data;
  uint64_t bits = static_cast<uint64_t>(data.size()) * 8;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bits >> (i * 8)) & 0xff));
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      const unsigned char* p = reinterpret_cast<const unsigned char*>(msg.data() + off + i * 4);
      w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
      uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22), maj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  std::string out(32, '\0');
  for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) out[i*4+j] = static_cast<char>((h[i] >> (24 - j*8)) & 0xff);
  return out;
}

inline std::string hex(const std::string& bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (unsigned char c : bytes) { out.push_back(digits[c >> 4]); out.push_back(digits[c & 15]); }
  return out;
}

// ---- Values ----
struct Value {
  enum Kind { VOID, INT, FLOAT, STR, BOOL, OBJ } kind = VOID;
  int64_t i = 0; double f = 0.0; std::string s; bool b = false; void* obj = nullptr; std::string type;  // type: schema type name of obj
  std::string context_name;
  static Value none() { return Value(); }
  static Value integer(int64_t v) { Value x; x.kind = INT; x.i = v; return x; }
  static Value real(double v) { Value x; x.kind = FLOAT; x.f = v; return x; }
  static Value string(std::string v) { Value x; x.kind = STR; x.s = std::move(v); return x; }
  static Value boolean(bool v) { Value x; x.kind = BOOL; x.b = v; return x; }
  static Value object(const std::string& type, void* p) { Value x; x.kind = OBJ; x.obj = p; x.type = type; return x; }
  const char* kind_name() const { static const char* n[] = {"void","int","float","string","bool","object"}; return n[kind]; }
};
using Args = std::vector<Value>;
using Function = std::function<Value(Args&)>;
using Method = std::function<Value(void* self, Args&)>;

struct CallSite {
  std::string file;
  int line = 0;
  int column = 0;
};

struct PlanEntry {
  int64_t seq = 0;
  std::string kind = "host_effect";
  std::string function;
  Args arguments;
  std::string arguments_json;
  int line = 0;
  int column = 0;
  std::string file;
  bool simulated = true;
  std::string detail_json;
  bool budget_exceeded = false;
};

struct VmConfig {
  uint64_t step_budget = 0;  // 0 = unlimited; max mutating host-effect steps in dry mode
};

// ---- Host registry: the implementation of the schema, declared with the schema's types ----
struct Signature { std::vector<std::string> params; std::string returns; };
struct FunctionEntry { Signature sig; Function fn; Function plan_fn; bool mutating = false; };
struct MethodEntry { Signature sig; Method fn; Method plan_fn; bool mutating = false; };
inline std::string signature_text(const Signature& s) { std::string out = "("; for (size_t i = 0; i < s.params.size(); ++i) out += (i ? ", " : "") + s.params[i]; return out + ") -> " + s.returns; }

namespace stdlib { struct Collections; struct StdRuntime; }
enum class StdMode { Enabled, Disabled };
class Registry;
class ResultSink;
inline Registry& register_std(Registry&);

class Registry {
  struct NamedFunction { std::string name; FunctionEntry entry; };
  struct NamedMethod { std::string type, name; MethodEntry entry; };
 public:
  explicit Registry(StdMode mode = StdMode::Enabled);
  bool has_std() const { return std_initialized_; }
  stdlib::Collections& std_collections() const;
  void set_std_output(std::ostream* sink);
  // Host-owned invocation sink; never reset by a per-region VM run.
  void set_result_sink(ResultSink* sink);
  std::ostream& std_output() const { return *std_output_; }
  std::vector<std::string> std_signatures() const { return std_signatures_; }
  std::vector<std::string> host_signatures() const {
    auto all = signatures(); std::vector<std::string> out;
    std::set_difference(all.begin(), all.end(), std_signatures_.begin(), std_signatures_.end(), std::back_inserter(out));
    return out;
  }
  bool matches_schema_signatures(std::vector<std::string> expected) const {
    expected.insert(expected.end(), std_signatures_.begin(), std_signatures_.end());
    std::sort(expected.begin(), expected.end()); expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    return signatures() == expected;
  }
  // One active run per shared std runtime; separate registries own independent runtimes.
  class StdRunScope {
   public:
    explicit StdRunScope(const Registry& r) : registry_(r.has_std() ? &r : nullptr) { if (registry_) registry_->begin_std_run(); }
    ~StdRunScope() { if (registry_) registry_->end_std_run(); }
    StdRunScope(const StdRunScope&) = delete;
    StdRunScope& operator=(const StdRunScope&) = delete;
   private:
    const Registry* registry_;
  };
  StdRunScope std_run_scope() const { return StdRunScope(*this); }
  // Type names are the schema's: int, float, string, bool, void (returns only) or an object type name. The arity the loader
  // checks is params.size(). Registering a name with the same parameter tuple twice is an error; distinct tuples are overloads.
  Registry& function(const std::string& name, std::vector<std::string> params, std::string returns, Function fn) {
    Signature sig{std::move(params), std::move(returns)};
    for (const auto& f : functions_) if (f.second.name == name && f.second.entry.sig.params == sig.params) throw Error("LOAD_UNBOUND", "registry: function '" + name + signature_text(sig) + "' is registered twice");
    std::string key = "fn " + name + signature_text(sig);
    functions_.emplace(std::move(key), NamedFunction{name, FunctionEntry{std::move(sig), std::move(fn), Function{}, false}});
    return *this;
  }
  Registry& mutating_function(const std::string& name, std::vector<std::string> params, std::string returns, Function plan_fn, Function apply_fn) {
    Signature sig{std::move(params), std::move(returns)};
    for (const auto& f : functions_) if (f.second.name == name && f.second.entry.sig.params == sig.params) throw Error("LOAD_UNBOUND", "registry: function '" + name + signature_text(sig) + "' is registered twice");
    std::string key = "fn " + name + signature_text(sig);
    functions_.emplace(std::move(key), NamedFunction{name, FunctionEntry{std::move(sig), std::move(apply_fn), std::move(plan_fn), true}});
    return *this;
  }
  Registry& method(const std::string& type, const std::string& name, std::vector<std::string> params, std::string returns, Method fn) {
    Signature sig{std::move(params), std::move(returns)};
    for (const auto& m : methods_) if (m.second.type == type && m.second.name == name && m.second.entry.sig.params == sig.params) throw Error("LOAD_UNBOUND", "registry: method '" + type + "." + name + signature_text(sig) + "' is registered twice");
    std::string key = "method " + type + "." + name + signature_text(sig);
    methods_.emplace(std::move(key), NamedMethod{type, name, MethodEntry{std::move(sig), std::move(fn), Method{}, false}});
    return *this;
  }
  Registry& mutating_method(const std::string& type, const std::string& name, std::vector<std::string> params, std::string returns, Method plan_fn, Method apply_fn) {
    Signature sig{std::move(params), std::move(returns)};
    for (const auto& m : methods_) if (m.second.type == type && m.second.name == name && m.second.entry.sig.params == sig.params) throw Error("LOAD_UNBOUND", "registry: method '" + type + "." + name + signature_text(sig) + "' is registered twice");
    std::string key = "method " + type + "." + name + signature_text(sig);
    methods_.emplace(std::move(key), NamedMethod{type, name, MethodEntry{std::move(sig), std::move(apply_fn), std::move(plan_fn), true}});
    return *this;
  }
  Registry& context(const std::string& name, const std::string& type) {
    if (!context_.emplace(name, type).second) throw Error("LOAD_UNBOUND", "registry: context '" + name + "' is declared twice");
    return *this;
  }
  std::string context_name_for_type(const std::string& type) const {
    for (const auto& c : context_) if (c.second == type) return c.first;
    return "";
  }
  void record_plan_detail(std::string detail_json, bool budget_exceeded = false) const {
    has_pending_plan_detail_ = true;
    pending_plan_detail_ = std::move(detail_json);
    pending_budget_exceeded_ = budget_exceeded;
  }
  void clear_pending_plan_detail() const {
    has_pending_plan_detail_ = false;
    pending_plan_detail_.clear();
    pending_budget_exceeded_ = false;
  }
  bool has_pending_plan_detail() const { return has_pending_plan_detail_; }
  std::string take_pending_plan_detail(bool& budget_exceeded) const {
    budget_exceeded = pending_budget_exceeded_;
    has_pending_plan_detail_ = false;
    pending_budget_exceeded_ = false;
    return std::move(pending_plan_detail_);
  }
  // Canonical strings, sorted: exactly what `lmscript --lmcode-signatures SCHEMA` prints for the schema this registry implements.
  std::vector<std::string> signatures() const {
    std::vector<std::string> out;
    for (auto& c : context_) out.push_back("context " + c.first + ": " + c.second);
    for (auto& f : functions_) out.push_back(f.first);
    for (auto& m : methods_) out.push_back(m.first);
    std::sort(out.begin(), out.end()); return out;
  }
  // Name-only lookup is retained for host introspection; overloaded names are intentionally ambiguous.
  const FunctionEntry* find_function(const std::string& name) const { const FunctionEntry* found = nullptr; for (auto& f : functions_) if (f.second.name == name) { if (found) return nullptr; found = &f.second.entry; } return found; }
  const MethodEntry* find_method(const std::string& type, const std::string& name) const { const MethodEntry* found = nullptr; for (auto& m : methods_) if (m.second.type == type && m.second.name == name) { if (found) return nullptr; found = &m.second.entry; } return found; }
  const std::string* context_type(const std::string& name) const { auto it = context_.find(name); return it == context_.end() ? nullptr : &it->second; }
  // The entry whose canonical string is `fn NAME(...) -> R`; null when the host declares no such signature (the loader names
  // the host's own signature for that name in its message, so a type drift between schema and host is one line to read).
  const FunctionEntry* function_by_signature(const std::string& canonical) const { auto it = functions_.find(canonical); return it == functions_.end() ? nullptr : &it->second.entry; }
  const MethodEntry* method_by_signature(const std::string& canonical) const { auto it = methods_.find(canonical); return it == methods_.end() ? nullptr : &it->second.entry; }
  std::string declared_for(const std::string& canonical) const {  // "the host declares fn x(float) -> int" or "" when the name is unknown
    size_t sp = canonical.find(' '), par = canonical.find('(');
    if (sp == std::string::npos || par == std::string::npos || par < sp) return "";
    std::string name = canonical.substr(sp + 1, par - sp - 1);
    if (canonical.compare(0, 3, "fn ") == 0) { std::vector<std::string> have; for (auto& f : functions_) if (f.second.name == name) have.push_back(f.first); std::sort(have.begin(), have.end()); std::string out; for (const auto& s : have) out += (out.empty() ? "" : ", ") + s; return out; }
    auto dot = name.find('.'); if (dot == std::string::npos) return "";
    std::vector<std::string> have; for (auto& m : methods_) if (m.second.type + "." + m.second.name == name) have.push_back(m.first); std::sort(have.begin(), have.end()); std::string out; for (const auto& s : have) out += (out.empty() ? "" : ", ") + s; return out;
  }
 private:
  friend Registry& register_std(Registry&);
  void begin_std_run() const;
  void end_std_run() const noexcept;
  bool std_initialized_ = false;
  std::ostream* std_output_ = &std::cout;
  mutable bool has_pending_plan_detail_ = false;
  mutable std::string pending_plan_detail_;
  mutable bool pending_budget_exceeded_ = false;
  std::vector<std::string> std_signatures_;
  std::shared_ptr<stdlib::StdRuntime> std_runtime_;
  std::map<std::string, NamedFunction> functions_;
  std::map<std::string, NamedMethod> methods_;
  std::map<std::string, std::string> context_;
};

inline std::string json_escape(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) { switch (c) { case '"': out += "\\\""; break; case '\\': out += "\\\\"; break; case '\n': out += "\\n"; break; case '\r': out += "\\r"; break; case '\t': out += "\\t"; break; default: if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; } else out += char(c); } }
  return out + "\"";
}
inline std::string value_to_json(const Value& v) {
  switch (v.kind) {
    case Value::VOID: return "null";
    case Value::INT: return std::to_string(v.i);
    case Value::FLOAT: {
      if (!std::isfinite(v.f)) return "null";
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.15g", v.f);
      return buf;
    }
    case Value::STR: return json_escape(v.s);
    case Value::BOOL: return v.b ? "true" : "false";
    case Value::OBJ: return json_escape(v.type);
  }
  return "null";
}
inline std::string args_to_json(const Args& args) {
  std::string out = "[";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) out.push_back(',');
    out += value_to_json(args[i]);
  }
  out.push_back(']');
  return out;
}
// ---- The standard result line: exactly one JSON line on stdout; the return value is the process exit code. ----
// `result_json` must already be JSON text ("null", "42", json_escape(text), an object...).
inline int report_ok(const std::string& result_json = "null") { std::cout << "{\"ok\": true, \"result\": " << result_json << "}" << std::endl; return 0; }
// phase is "load" or "run" (compile failures are lmscript's, exit 2); exit 4 for APP_FAIL, else 3.
inline int report_error(const std::string& phase, const Error& e) {
  std::cout << "{\"ok\": false, \"phase\": " << json_escape(phase) << ", \"code\": " << json_escape(e.code) << ", \"message\": " << json_escape(e.what());
  if (e.has_location) { std::cout << ", \"line\": " << e.line; if (e.has_column) std::cout << ", \"column\": " << e.column; }
  std::cout << "}" << std::endl;
  return e.code == "APP_FAIL" ? 4 : 3;
}
inline int report_app_failure(const std::string& message) { return report_error("app", Error("APP_FAIL", message)); }

// ---- Per-run context: the host's live objects, by declared name ----
class Context {
 public:
  void set(const std::string& name, void* object) { objects_[name] = object; }
  void* get(const std::string& name) const { auto it = objects_.find(name); return it == objects_.end() ? nullptr : it->second; }
  void set_dry(bool dry) { dry_ = dry; }
  bool dry() const { return dry_; }
  void set_file(std::string file) { file_ = std::move(file); call_site_.file = file_; }
  const std::string& file() const { return file_; }
  void set_config(const VmConfig& cfg) { config_ = cfg; }
  const VmConfig& config() const { return config_; }
  void set_step_budget(uint64_t budget) { config_.step_budget = budget; }
  uint64_t step_budget() const { return config_.step_budget; }
  void set_dry_loop_limit(int64_t limit) { config_.step_budget = limit > 0 ? static_cast<uint64_t>(limit) : 0; }
  int64_t dry_loop_limit() const { return static_cast<int64_t>(config_.step_budget); }
  const std::vector<PlanEntry>& plan_entries() const { return plan_entries_; }
  std::vector<PlanEntry>& plan_entries() { return plan_entries_; }
  void clear_plan_entries() { plan_entries_.clear(); truncated_ = false; stop_reason_.clear(); }
  bool truncated() const { return truncated_; }
  const std::string& stop_reason() const { return stop_reason_; }
  const CallSite& call_site() const { return call_site_; }
 private:
  friend class Program;
  std::map<std::string, void*> objects_;
  bool dry_ = false;
  std::string file_;
  VmConfig config_;
  mutable std::vector<PlanEntry> plan_entries_;
  mutable bool truncated_ = false;
  mutable std::string stop_reason_;
  mutable CallSite call_site_;
};

// ---- Bytecode ----
enum Op : uint8_t {
  PUSH_INT = 0x01, PUSH_STR = 0x02, PUSH_BOOL = 0x03, LOAD_LOCAL = 0x04, STORE_LOCAL = 0x05, LOAD_CTX = 0x06,
  CALL_FN = 0x07, CALL_METHOD = 0x08, EQ = 0x09, NE = 0x0A, NOT = 0x0B, JUMP = 0x0C, JUMP_IF_FALSE = 0x0D,
  JUMP_IF_TRUE = 0x0E, DUP = 0x0F, POP = 0x10, HALT = 0x11,
  // v2: floats, typed arithmetic and ordering, explicit widening
  PUSH_FLOAT = 0x12, ADD_I = 0x13, SUB_I = 0x14, MUL_I = 0x15, DIV_I = 0x16, ADD_F = 0x17, SUB_F = 0x18, MUL_F = 0x19, DIV_F = 0x1A,
  NEG_I = 0x1B, NEG_F = 0x1C, LT_I = 0x1D, LE_I = 0x1E, GT_I = 0x1F, GE_I = 0x20, LT_F = 0x21, LE_F = 0x22, GT_F = 0x23, GE_F = 0x24, I2F = 0x25,
  // v3: string concatenation and str() conversions
  CONCAT = 0x26, I2S = 0x27, F2S = 0x28, B2S = 0x29,
  // Portable primitive math. The operand is a one-byte function id.
  MATH = 0x32,
  // Integer remainder (the dividend's sign; zero divisor is a run-time error).
  MOD_I = 0x33,
  FOR_RANGE = 0x2f, FOR_NEXT = 0x31,
  // v5: structured while loop markers. LOOP_BACK is the only general backward edge.
  WHILE_BEGIN = 0x34, WHILE_TEST = 0x35, LOOP_BACK = 0x36
};
enum MathFn : uint8_t {
  MATH_SIN_F = 0, MATH_COS_F = 1, MATH_SQRT_F = 2, MATH_ABS_F = 3,
  MATH_FLOOR_F = 4, MATH_POW_F = 5, MATH_MIN_F = 6, MATH_MAX_F = 7,
  MATH_CLAMP_F = 8, MATH_LERP_F = 9, MATH_EASE_F = 10,
  MATH_EASE_IN_F = 11, MATH_EASE_OUT_F = 12, MATH_INT_F = 13,
  MATH_ABS_I = 14, MATH_MIN_I = 15, MATH_MAX_I = 16, MATH_CLAMP_I = 17
};
static const char MAGIC[4] = {'D', 'S', 'L', 'B'};
static const uint16_t VERSION = 6;
// Convert the wire representation explicitly so decoding does not depend on whether a C++ implementation
// defines an out-of-range unsigned-to-signed conversion as a two's-complement reinterpretation.
inline int64_t from_twos_complement(uint64_t u) {
  return u <= uint64_t(INT64_MAX) ? static_cast<int64_t>(u) : -1 - static_cast<int64_t>(~u);
}
class Reader {
 public:
  explicit Reader(const std::string& bytes) : b_(&bytes) {}
  void seek(size_t pos) { if (pos > b_->size()) throw Error("RUN_ERROR", "jump target " + std::to_string(pos) + " is outside the code"); pos_ = pos; }
  uint8_t u8() { need(1); return static_cast<uint8_t>((*b_)[pos_++]); }
  uint16_t u16() { need(2); uint16_t v = (uint16_t(uint8_t((*b_)[pos_])) << 8) | uint8_t((*b_)[pos_+1]); pos_ += 2; return v; }
  uint32_t u32() { need(4); uint32_t v = 0; for (int i = 0; i < 4; ++i) v = (v << 8) | uint8_t((*b_)[pos_+i]); pos_ += 4; return v; }
  int32_t i32() { return static_cast<int32_t>(u32()); }
  int64_t i64() { need(8); uint64_t v = 0; for (int i = 0; i < 8; ++i) v = (v << 8) | uint8_t((*b_)[pos_+i]); pos_ += 8; return from_twos_complement(v); }
  double f64() { need(8); uint64_t bits = 0; for (int i = 0; i < 8; ++i) bits = (bits << 8) | uint8_t((*b_)[pos_+i]); pos_ += 8; double d; std::memcpy(&d, &bits, 8); return d; }
  std::string bytes(size_t n) { need(n); std::string s = b_->substr(pos_, n); pos_ += n; return s; }
  size_t pos() const { return pos_; }
  bool done() const { return pos_ == b_->size(); }
 private:
  void need(size_t n) { if (pos_ + n > b_->size()) throw Error("LOAD_BAD_BYTECODE", "bytecode is truncated: wanted " + std::to_string(n) + " byte(s) at offset " + std::to_string(pos_) + ", file has " + std::to_string(b_->size())); }
  const std::string* b_; size_t pos_ = 0;
};

class Program {
 public:
  struct Import { const FunctionEntry* fn = nullptr; const MethodEntry* method = nullptr; std::string type, name; };
  struct Local { std::string type; bool hidden = false; };
  struct Line { uint32_t pc, line; uint16_t column; };
  struct RunStats { uint64_t instructions = 0; };
  struct Loop { uint16_t counter, end; size_t opener_next; };
  struct VerifiedInstr { size_t pc, next, target = 0; uint8_t op; size_t depth; size_t owner; };
  // One rule of the file: name is the fn identifier, annotation the node kind, iface the canonical
  // `interface I(T1, T2) -> R` text; params are the type names of parameter slots 0..n-1, returns the interface's.
  struct Entry { std::string name, annotation, iface, returns; std::vector<std::string> params; };

  // Parses, binds and verifies. Nothing executes here. Throws dslvm::Error with a message naming the problem.
  static Program load(const std::string& bytecode, const std::string& schemaBytes, const Registry& reg) {
    Program p;
    p.bound_registry_ = &reg;
    Reader r(bytecode);
    if (bytecode.size() < 4 || std::memcmp(bytecode.data(), MAGIC, 4) != 0) throw Error("LOAD_BAD_BYTECODE", "not a DSL bytecode file: bad magic (expected 'DSLB')");
    r.bytes(4);
    uint16_t version = r.u16();
    if (version != VERSION) throw Error("LOAD_BAD_BYTECODE", "unsupported bytecode version " + std::to_string(version) + " (this runtime reads version " + std::to_string(VERSION) + ")");
    std::string hash = r.bytes(32), expected = sha256(schemaBytes);
    if (hash != expected) throw Error("LOAD_SCHEMA_MISMATCH", "schema mismatch: bytecode was compiled against schema sha256 " + hex(hash) + ", the given schema has sha256 " + hex(expected));
    uint32_t nconst = r.u32();
    for (uint32_t i = 0; i < nconst; ++i) {
      uint8_t tag = r.u8();
      if (tag != 1) throw Error("LOAD_BAD_BYTECODE", "constant " + std::to_string(i) + " has unknown tag " + std::to_string(tag) + " (1 = string)");
      uint32_t len = r.u32(); p.consts_.push_back(r.bytes(len));
    }
    auto cidx = [&](uint32_t i, const char* what) -> const std::string& { if (i >= p.consts_.size()) throw Error("LOAD_BAD_BYTECODE", std::string(what) + ": constant index " + std::to_string(i) + " out of range (" + std::to_string(p.consts_.size()) + " constants)"); return p.consts_[i]; };
    uint16_t nimports = r.u16();
    for (uint16_t i = 0; i < nimports; ++i) {
      const std::string& canonical = cidx(r.u32(), "import");
      Import imp;
      if (canonical.compare(0, 3, "fn ") == 0) {
        imp.fn = reg.function_by_signature(canonical);
        size_t par = canonical.find('(');
        if (par != std::string::npos && par > 3) imp.name = canonical.substr(3, par - 3);
      } else if (canonical.compare(0, 7, "method ") == 0) {
        imp.method = reg.method_by_signature(canonical);
        size_t dot = canonical.find('.'), par = canonical.find('(');
        if (dot != std::string::npos && par != std::string::npos && dot < par) { imp.type = canonical.substr(7, dot - 7); imp.name = canonical.substr(dot + 1, par - dot - 1); }
      }
      if (!imp.fn && !imp.method) {
        std::string have = reg.declared_for(canonical);
        throw Error("LOAD_UNBOUND", "import '" + canonical + "' is not provided by the host" + (have.empty() ? "" : " (the host declares " + have + ")"));
      }
      p.imports_.push_back(std::move(imp));
    }
    uint16_t ncontext = r.u16();
    for (uint16_t i = 0; i < ncontext; ++i) {
      const std::string& name = cidx(r.u32(), "context name"); const std::string& type = cidx(r.u32(), "context type");
      const std::string* declared = reg.context_type(name);
      if (!declared) throw Error("LOAD_UNBOUND", "context variable '" + name + "' is not declared by the host");
      if (*declared != type) throw Error("LOAD_UNBOUND", "context variable '" + name + "' is declared as " + type + " by the bytecode, the host provides " + *declared);
      p.context_.push_back(name);
    }
    uint16_t nlocals = r.u16();
    for (uint16_t i = 0; i < nlocals; ++i) { Local l; l.type = cidx(r.u32(), "local type"); uint8_t hidden = r.u8(); if (hidden > 1) throw Error("LOAD_BAD_BYTECODE", "local " + std::to_string(i) + " hidden flag must be 0 or 1"); l.hidden = hidden == 1; p.locals_.push_back(std::move(l)); }
    p.max_stack_ = r.u16();
    uint32_t codeLen = r.u32();
    p.code_ = r.bytes(codeLen);
    uint32_t nlines = r.u32();
    for (uint32_t i = 0; i < nlines; ++i) { Line l; l.pc = r.u32(); l.line = r.u32(); l.column = r.u16(); if (i && l.pc < p.lines_[i - 1].pc) throw Error("LOAD_BAD_BYTECODE", "line table is not sorted by pc at entry " + std::to_string(i)); p.lines_.push_back(l); }
    uint16_t nentries = r.u16();
    for (uint16_t i = 0; i < nentries; ++i) {
      EntryRegion e;
      e.meta.name = cidx(r.u32(), "entry name");
      for (const auto& prev : p.entry_meta_) if (prev.name == e.meta.name) throw Error("LOAD_BAD_BYTECODE", "duplicate entry name '" + e.meta.name + "'");
      e.meta.annotation = cidx(r.u32(), "entry annotation");
      e.meta.iface = cidx(r.u32(), "entry interface");
      // The interface text is canonical `interface I(T1, T2) -> R`; type names never contain commas or parens.
      size_t lpar = e.meta.iface.find('(');
      size_t rpar = lpar == std::string::npos ? std::string::npos : e.meta.iface.find(')', lpar + 1);
      size_t arrow = e.meta.iface.rfind("->");
      std::vector<std::string> iparams;
      std::string ret;
      if (lpar != std::string::npos && rpar != std::string::npos && arrow != std::string::npos && arrow > rpar) {
        std::string inner = e.meta.iface.substr(lpar + 1, rpar - lpar - 1);
        for (size_t b = 0, e2; b <= inner.size(); b = e2 + 1) {
          e2 = inner.find(',', b);
          std::string part = inner.substr(b, e2 == std::string::npos ? e2 : e2 - b);
          size_t fb = part.find_first_not_of(' '), lb = part.find_last_not_of(' ');
          if (fb != std::string::npos) iparams.push_back(part.substr(fb, lb - fb + 1));
          if (e2 == std::string::npos) break;
        }
        ret = e.meta.iface.substr(arrow + 2);
        size_t fb = ret.find_first_not_of(' '), lb = ret.find_last_not_of(' ');
        ret = fb == std::string::npos ? "" : ret.substr(fb, lb - fb + 1);
      }
      if (ret.empty()) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "': interface '" + e.meta.iface + "' is not an interface signature");
      e.meta.returns = ret;
      e.nparams = r.u16();
      uint16_t enlocals = r.u16();
      if (e.nparams > enlocals) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "': nparams " + std::to_string(e.nparams) + " exceeds nlocals " + std::to_string(enlocals));
      for (uint16_t j = 0; j < enlocals; ++j) {
        Local l; l.type = cidx(r.u32(), "entry local type");
        uint8_t hidden = r.u8();
        if (hidden > 1) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "': local " + std::to_string(j) + " hidden flag must be 0 or 1");
        l.hidden = hidden == 1;
        if (j < e.nparams) e.meta.params.push_back(l.type);
        e.locals.push_back(std::move(l));
      }
      if (e.nparams != iparams.size()) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "' declares " + std::to_string(e.nparams) + " parameter(s); " + e.meta.iface + " has " + std::to_string(iparams.size()));
      for (uint16_t j = 0; j < e.nparams; ++j) {
        if (e.locals[j].type != iparams[j]) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "' parameter " + std::to_string(j + 1) + " has type " + e.locals[j].type + "; the interface expects " + iparams[j]);
        if (e.locals[j].hidden) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "' parameter " + std::to_string(j + 1) + " is hidden");
      }
      e.max_stack = r.u16();
      uint32_t ecode_len = r.u32();
      e.code = r.bytes(ecode_len);
      uint32_t enlines = r.u32();
      for (uint32_t j = 0; j < enlines; ++j) { Line l; l.pc = r.u32(); l.line = r.u32(); l.column = r.u16(); if (j && l.pc < e.lines[j - 1].pc) throw Error("LOAD_BAD_BYTECODE", "entry '" + e.meta.name + "': line table is not sorted by pc at entry " + std::to_string(j)); e.lines.push_back(l); }
      p.entry_meta_.push_back(e.meta);
      p.entries_.push_back(std::move(e));
    }
    if (!r.done()) throw Error("LOAD_BAD_BYTECODE", "bytecode has " + std::to_string(bytecode.size() - r.pos()) + " trailing byte(s) after the entries");
    p.verify_region(p.code_, p.locals_, p.lines_, p.loops_, -1);
    for (auto& e : p.entries_) {
      try {
        p.verify_region(e.code, e.locals, e.lines, e.loops, e.meta.returns == "void" ? 0 : 1);
      } catch (Error& err) {
        throw Error(err.code, "entry '" + e.meta.name + "': " + err.message);
      }
    }
    return p;
  }

  // Executes the chunk once. The host's objects come from ctx (declared names only).
  void run(const Context& ctx, const Registry& reg, RunStats* stats = nullptr) const {
    if (ctx.dry() && ctx.truncated_) return;
    execute(code_, lines_, loops_, max_stack_, std::vector<Value>(locals_.size()), false, ctx, reg, stats);
  }

  // The rule entries, in source order.
  const std::vector<Entry>& entries() const { return entry_meta_; }

  // Executes one entry: args are stored into its parameter locals 0..n-1 (kinds are the host's contract, unchecked),
  // the result is the value left on the stack at HALT (Value::none() for a void interface). A wrong argument count or
  // an out-of-range entry index is a host programming error (RUN_HOST).
  Value call(size_t entry, const Context& ctx, const Registry& reg, std::vector<Value> args, RunStats* stats = nullptr) const {
    if (entry >= entries_.size()) throw Error("RUN_HOST", "call entry index " + std::to_string(entry) + " out of range (" + std::to_string(entries_.size()) + " entries)");
    const EntryRegion& e = entries_[entry];
    if (args.size() != e.nparams) throw Error("RUN_HOST", "call entry '" + e.meta.name + "' expects " + std::to_string(e.nparams) + " argument(s); found " + std::to_string(args.size()));
    if (ctx.dry() && ctx.truncated_) return Value::none();
    std::vector<Value> locals(e.locals.size());
    for (size_t i = 0; i < args.size(); ++i) locals[i] = std::move(args[i]);
    return execute(e.code, e.lines, e.loops, e.max_stack, std::move(locals), e.meta.returns != "void", ctx, reg, stats);
  }

 private:
  // The one interpreter: run() and call() differ only in the region, the initial locals and whether HALT's stack
  // value is the result. Dry mode, RunStats, the std run scope and located errors behave the same for both.
  Value execute(const std::string& code, const std::vector<Line>& lines, const std::map<size_t, Loop>& loops, uint16_t max_stack,
                std::vector<Value> locals, bool capture_result, const Context& ctx, const Registry& reg, RunStats* stats) const {
    auto std_scope = (bound_registry_ ? *bound_registry_ : reg).std_run_scope();
    std::vector<Value> stack;
    stack.reserve(max_stack);
    Reader r(code);
    auto pop = [&]() { if (stack.empty()) throw Error("RUN_ERROR", "stack underflow"); Value v = std::move(stack.back()); stack.pop_back(); return v; };
    auto push = [&](Value v) { stack.push_back(std::move(v)); };
    auto jump = [&](int32_t rel) { int64_t target = int64_t(r.pos()) + rel; if (target < 0) throw Error("RUN_ERROR", "jump before the code start"); r.seek(size_t(target)); };
    auto wrap = [](uint64_t u) { return from_twos_complement(u); };  // two's complement wraparound through unsigned arithmetic
    auto min_float = [](double a, double b) {
      if (std::isnan(a)) return a;
      if (std::isnan(b)) return b;
      if (a < b) return a;
      return b < a ? b : a;  // equal operands, including signed zero, choose the first
    };
    auto max_float = [](double a, double b) {
      if (std::isnan(a)) return a;
      if (std::isnan(b)) return b;
      if (a > b) return a;
      return b > a ? b : a;  // equal operands, including signed zero, choose the first
    };
    auto int_from_float = [](double x) -> int64_t {
      // Java-like narrowing conversion: truncation toward zero, with NaN -> 0
      // and saturation at the representable endpoints.  The comparisons also
      // cover infinities; values strictly inside the interval are safe to cast.
      if (std::isnan(x)) return 0;
      constexpr double TWO63 = 0x1p63;
      if (x >= TWO63) return INT64_MAX;
      if (x <= -TWO63) return INT64_MIN;
      return static_cast<int64_t>(x);
    };
    auto update_call_site = [&](size_t pc) {
      const Line* ln = line_of(lines, pc);
      ctx.call_site_.file = ctx.file_;
      ctx.call_site_.line = ln ? static_cast<int>(ln->line) : 0;
      ctx.call_site_.column = ln ? static_cast<int>(ln->column) : 0;
    };
    auto record_plan_entry = [&](const std::string& fn_label, const Args& args, const Value& result) {
      PlanEntry entry;
      entry.seq = static_cast<int64_t>(ctx.plan_entries_.size() + 1);
      entry.kind = "host_effect";
      entry.function = fn_label;
      entry.arguments = args;
      entry.arguments_json = args_to_json(args);
      entry.line = ctx.call_site_.line;
      entry.column = ctx.call_site_.column;
      entry.file = ctx.file_;
      entry.simulated = true;
      if (reg.has_pending_plan_detail()) {
        entry.detail_json = reg.take_pending_plan_detail(entry.budget_exceeded);
      } else if (result.kind == Value::OBJ) {
        if (const auto* jm = reg.method_by_signature("method " + result.type + ".json() -> string")) {
          Args empty;
          Value jv = jm->fn(result.obj, empty);
          if (jv.kind == Value::STR) entry.detail_json = std::move(jv.s);
        }
        if (entry.detail_json.find("\"budget_exceeded\":true") != std::string::npos ||
            entry.detail_json.find("\"budget_exceeded\": true") != std::string::npos) {
          entry.budget_exceeded = true;
        }
      }
      ctx.plan_entries_.push_back(std::move(entry));
    };
    while (true) {
      const size_t fault_pc = r.pos();
      try {
        uint8_t op = r.u8();
        if (stats) ++stats->instructions;
        switch (op) {
        case PUSH_INT: push(Value::integer(r.i64())); break;
        case PUSH_FLOAT: push(Value::real(r.f64())); break;
        case PUSH_STR: push(Value::string(consts_[r.u32()])); break;
        case PUSH_BOOL: push(Value::boolean(r.u8() != 0)); break;
        case LOAD_LOCAL: {
          uint16_t s = r.u16();
          if (locals[s].kind == Value::VOID) throw Error("RUN_ERROR", "local slot " + std::to_string(s) + " read before assignment");
          push(locals[s]); break;
        }
        case STORE_LOCAL: { uint16_t s = r.u16(); locals[s] = pop(); break; }
        case LOAD_CTX: {
          const std::string& name = context_[r.u16()];
          void* obj = ctx.get(name);
          if (!obj) throw Error("RUN_ERROR", "context variable '" + name + "' was not provided by the host for this run");
          Value v = Value::object(*reg.context_type(name), obj);
          v.context_name = name;
          push(std::move(v)); break;
        }
        case CALL_FN: {
          const Import& imp = imports_[r.u16()]; uint8_t argc = r.u8();
          Args args(argc); for (int i = argc - 1; i >= 0; --i) args[i] = pop();
          update_call_site(fault_pc);
          if (imp.fn->mutating && ctx.dry()) {
            if (ctx.config_.step_budget > 0 && ctx.plan_entries_.size() >= ctx.config_.step_budget) {
              ctx.truncated_ = true;
              ctx.stop_reason_ = "dry loop limit";
              return Value::none();
            }
            reg.clear_pending_plan_detail();
            const Function& target_fn = imp.fn->plan_fn ? imp.fn->plan_fn : imp.fn->fn;
            Value result = target_fn(args);
            record_plan_entry(imp.name, args, result);
            if (result.kind != Value::VOID) push(std::move(result));
          } else {
            Value result = imp.fn->fn(args);
            if (result.kind != Value::VOID) push(std::move(result));
          }
          break;
        }
        case CALL_METHOD: {
          const Import& imp = imports_[r.u16()]; uint8_t argc = r.u8();
          Args args(argc); for (int i = argc - 1; i >= 0; --i) args[i] = pop();
          Value self = pop();
          if (self.kind != Value::OBJ || self.type != imp.type) throw Error("RUN_ERROR", "method " + imp.type + "." + imp.name + " called on a " + (self.kind == Value::OBJ ? self.type : std::string(self.kind_name())));
          update_call_site(fault_pc);
          if (imp.method->mutating && ctx.dry()) {
            if (ctx.config_.step_budget > 0 && ctx.plan_entries_.size() >= ctx.config_.step_budget) {
              ctx.truncated_ = true;
              ctx.stop_reason_ = "dry loop limit";
              return Value::none();
            }
            reg.clear_pending_plan_detail();
            const Method& target_fn = imp.method->plan_fn ? imp.method->plan_fn : imp.method->fn;
            Value result = target_fn(self.obj, args);
            std::string receiver = !self.context_name.empty() ? self.context_name : reg.context_name_for_type(imp.type);
            record_plan_entry((receiver.empty() ? imp.type : receiver) + "." + imp.name, args, result);
            if (result.kind != Value::VOID) push(std::move(result));
          } else {
            Value result = imp.method->fn(self.obj, args);
            if (result.kind != Value::VOID) push(std::move(result));
          }
          break;
        }
        case EQ: case NE: {
          Value b = pop(), a = pop();
          if (a.kind != b.kind) throw Error("RUN_ERROR", std::string("cannot compare ") + a.kind_name() + " with " + b.kind_name());
          bool eq = a.kind == Value::INT ? a.i == b.i : a.kind == Value::FLOAT ? a.f == b.f : a.kind == Value::STR ? a.s == b.s : a.kind == Value::BOOL ? a.b == b.b : a.obj == b.obj;
          push(Value::boolean(op == EQ ? eq : !eq)); break;
        }
        case ADD_I: case SUB_I: case MUL_I: case DIV_I: case MOD_I: case LT_I: case LE_I: case GT_I: case GE_I: {
          Value b = pop(), a = pop();
          if (a.kind != Value::INT || b.kind != Value::INT) throw Error("RUN_ERROR", std::string("int operator applied to ") + a.kind_name() + " and " + b.kind_name());
          int64_t x = a.i, y = b.i; uint64_t ux = uint64_t(x), uy = uint64_t(y);
          switch (op) {
            case ADD_I: push(Value::integer(wrap(ux + uy))); break;
            case SUB_I: push(Value::integer(wrap(ux - uy))); break;
            case MUL_I: push(Value::integer(wrap(ux * uy))); break;
            case DIV_I: if (y == 0) throw Error("RUN_ERROR", "int division by zero"); push(Value::integer(y == -1 ? wrap(0 - ux) : x / y)); break;  // truncates toward zero; INT64_MIN / -1 wraps
            case MOD_I: if (y == 0) throw Error("RUN_ERROR", "int modulo by zero"); push(Value::integer(y == -1 ? 0 : x % y)); break;  // remainder with the dividend's sign
            case LT_I: push(Value::boolean(x < y)); break; case LE_I: push(Value::boolean(x <= y)); break;
            case GT_I: push(Value::boolean(x > y)); break; default: push(Value::boolean(x >= y)); break;
          }
          break;
        }
        case ADD_F: case SUB_F: case MUL_F: case DIV_F: case LT_F: case LE_F: case GT_F: case GE_F: {
          Value b = pop(), a = pop();
          if (a.kind != Value::FLOAT || b.kind != Value::FLOAT) throw Error("RUN_ERROR", std::string("float operator applied to ") + a.kind_name() + " and " + b.kind_name());
          switch (op) {
            case ADD_F: push(Value::real(a.f + b.f)); break; case SUB_F: push(Value::real(a.f - b.f)); break;
            case MUL_F: push(Value::real(a.f * b.f)); break; case DIV_F: push(Value::real(a.f / b.f)); break;  // IEEE: inf/nan allowed
            case LT_F: push(Value::boolean(a.f < b.f)); break; case LE_F: push(Value::boolean(a.f <= b.f)); break;
            case GT_F: push(Value::boolean(a.f > b.f)); break; default: push(Value::boolean(a.f >= b.f)); break;
          }
          break;
        }
        case NEG_I: { Value a = pop(); if (a.kind != Value::INT) throw Error("RUN_ERROR", std::string("int negation applied to ") + a.kind_name()); push(Value::integer(wrap(0 - uint64_t(a.i)))); break; }  // INT64_MIN wraps to itself
        case NEG_F: { Value a = pop(); if (a.kind != Value::FLOAT) throw Error("RUN_ERROR", std::string("float negation applied to ") + a.kind_name()); push(Value::real(-a.f)); break; }
        case I2F: { Value a = pop(); if (a.kind != Value::INT) throw Error("RUN_ERROR", std::string("float() applied to ") + a.kind_name()); push(Value::real(double(a.i))); break; }
        case CONCAT: {
          Value b = pop(), a = pop();
          if (a.kind != Value::STR || b.kind != Value::STR) throw Error("RUN_ERROR", std::string("string + applied to ") + a.kind_name() + " and " + b.kind_name());
          push(Value::string(a.s + b.s)); break;
        }
        case I2S: { Value a = pop(); if (a.kind != Value::INT) throw Error("RUN_ERROR", std::string("str(int) applied to ") + a.kind_name()); push(Value::string(std::to_string(a.i))); break; }
        case F2S: {  // %.15g: deterministic across hosts
          Value a = pop(); if (a.kind != Value::FLOAT) throw Error("RUN_ERROR", std::string("str(float) applied to ") + a.kind_name());
          if (std::isnan(a.f)) { push(Value::string("nan")); break; } // NaN sign/payload are not portable text.
          char buf[64]; std::snprintf(buf, sizeof buf, "%.15g", a.f); push(Value::string(buf)); break;
        }
        case B2S: { Value a = pop(); if (a.kind != Value::BOOL) throw Error("RUN_ERROR", std::string("str(bool) applied to ") + a.kind_name()); push(Value::string(a.b ? "true" : "false")); break; }
        case MATH: {
          const uint8_t fn = r.u8();
          switch (fn) {
            case MATH_SIN_F: { Value a = pop(); push(Value::real(std::sin(a.f))); break; }
            case MATH_COS_F: { Value a = pop(); push(Value::real(std::cos(a.f))); break; }
            case MATH_SQRT_F: { Value a = pop(); push(Value::real(std::sqrt(a.f))); break; }
            case MATH_ABS_F: { Value a = pop(); push(Value::real(std::fabs(a.f))); break; }
            case MATH_FLOOR_F: { Value a = pop(); push(Value::real(std::floor(a.f))); break; }
            case MATH_POW_F: { Value b = pop(), a = pop(); push(Value::real(std::pow(a.f, b.f))); break; }
            case MATH_MIN_F: { Value b = pop(), a = pop(); push(Value::real(min_float(a.f, b.f))); break; }
            case MATH_MAX_F: { Value b = pop(), a = pop(); push(Value::real(max_float(a.f, b.f))); break; }
            case MATH_CLAMP_F: { Value hi = pop(), lo = pop(), x = pop(); push(Value::real(min_float(max_float(x.f, lo.f), hi.f))); break; }
            case MATH_LERP_F: { Value t = pop(), b = pop(), a = pop(); double d = b.f - a.f; volatile double product = d * t.f; push(Value::real(a.f + product)); break; }
            case MATH_EASE_F: { Value a = pop(); double t = min_float(max_float(a.f, 0.0), 1.0); volatile double square = t * t; volatile double twice = 2.0 * t; push(Value::real(square * (3.0 - twice))); break; }
            case MATH_EASE_IN_F: { Value a = pop(); double t = min_float(max_float(a.f, 0.0), 1.0); volatile double square = t * t; push(Value::real(square)); break; }
            case MATH_EASE_OUT_F: { Value a = pop(); double t = min_float(max_float(a.f, 0.0), 1.0); volatile double d = 1.0 - t; volatile double square = d * d; push(Value::real(1.0 - square)); break; }
            case MATH_INT_F: { Value a = pop(); push(Value::integer(int_from_float(a.f))); break; }
            case MATH_ABS_I: { Value a = pop(); push(Value::integer(a.i < 0 ? wrap(0 - uint64_t(a.i)) : a.i)); break; }
            case MATH_MIN_I: { Value b = pop(), a = pop(); push(Value::integer(a.i < b.i ? a.i : b.i)); break; }
            case MATH_MAX_I: { Value b = pop(), a = pop(); push(Value::integer(a.i > b.i ? a.i : b.i)); break; }
            case MATH_CLAMP_I: { Value hi = pop(), lo = pop(), x = pop(); int64_t lower = x.i < lo.i ? lo.i : x.i; push(Value::integer(lower < hi.i ? lower : hi.i)); break; }
            // The loader structurally rejects unknown IDs before execution.
            // Bytecode is trusted: there is no run-time validation path.
            default: break;
          }
          break;
        }
        case NOT: {
          Value a = pop();
          if (a.kind != Value::BOOL) throw Error("RUN_ERROR", std::string("! applied to a ") + a.kind_name());
          push(Value::boolean(!a.b)); break;
        }
        case JUMP: jump(r.i32()); break;
        case JUMP_IF_FALSE: case JUMP_IF_TRUE: {
          int32_t rel = r.i32(); Value c = pop();
          if (c.kind != Value::BOOL) throw Error("RUN_ERROR", std::string("condition is a ") + c.kind_name() + ", not a bool");
          if (c.b == (op == JUMP_IF_TRUE)) jump(rel);
          break;
        }
        case FOR_RANGE: {
          uint16_t counter = r.u16(), end = r.u16(); int32_t rel = r.i32();
          Value finish = pop(), begin = pop();
          locals[end] = std::move(finish);
          locals[counter] = std::move(begin);
          if (locals[counter].i >= locals[end].i) jump(rel);
          break;
        }
        case FOR_NEXT: {
          uint16_t encoded_end = r.u16(); int32_t rel = r.i32();
          const auto it = loops.find(r.pos() - 7);
          // The verifier has paired this instruction and recorded both hidden slots;
          // the encoded slot is retained as a structural consistency marker.
          const Loop& loop = it->second;
          (void)encoded_end;
          locals[loop.counter].i = from_twos_complement(uint64_t(locals[loop.counter].i) + 1);
          if (locals[loop.counter].i < locals[loop.end].i) jump(rel);
          break;
        }
        case WHILE_BEGIN: break;
        case WHILE_TEST: {
          int32_t rel = r.i32(); Value condition = pop();
          if (!condition.b) jump(rel);
          break;
        }
        case LOOP_BACK: jump(r.i32()); break;
        case DUP: {
          if (stack.empty()) throw Error("RUN_ERROR", "stack underflow");
          push(stack.back()); break;
        }
        case POP: pop(); break;
        case HALT:
          if (capture_result && !stack.empty()) { Value result = std::move(stack.back()); return result; }
          return Value::none();
        default: throw Error("RUN_ERROR", "unknown opcode " + std::to_string(op));
        }
      } catch (Error& e) {
        const Line* line = line_of(lines, fault_pc);
        if (line) e.locate(line->line, line->column);
        throw;
      }
    }
  }

 public:
  size_t constant_count() const { return consts_.size(); }
  size_t code_size() const { return code_.size(); }
  size_t import_count() const { return imports_.size(); }
  size_t local_count() const { return locals_.size(); }
  const std::vector<Local>& locals() const { return locals_; }
  uint16_t max_stack() const { return max_stack_; }
  const std::vector<Line>& lines() const { return lines_; }
  // The source position recorded for a pc (the last line-table entry at or before it); null when the table has none.
  const Line* line_of(size_t pc) const { return line_of(lines_, pc); }
  static const Line* line_of(const std::vector<Line>& lines, size_t pc) { const Line* best = nullptr; for (auto& l : lines) { if (l.pc <= pc) best = &l; else break; } return best; }

 private:
  // Static verification of one code region (the body or an entry): every instruction decodes, every index is in range,
  // ordinary jumps remain in their lexical region (or break exactly one loop), and paired range/while loops nest properly.
  // Imports and context are already bound. halt_depth >= 0 additionally requires every reachable HALT to leave exactly
  // that many values (entries: 0 for a void interface, 1 otherwise) and disables the no-loop early return in verify_stack.
  void verify_region(const std::string& code, const std::vector<Local>& locals, const std::vector<Line>& lines,
                     std::map<size_t, Loop>& loops, int halt_depth) {
    struct Open { uint8_t op; size_t pc, next, exit; uint16_t counter, end; bool tested; size_t parent; size_t test = 0; };
    struct LoopRegion { size_t exit, parent, test; uint8_t op; uint16_t counter, end; };
    std::vector<VerifiedInstr> insns; std::vector<Open> open; std::vector<std::pair<size_t, int64_t>> ordinary;
    std::vector<std::pair<size_t, int64_t>> while_exits;
    std::map<size_t, LoopRegion> regions;
    std::vector<uint16_t> used_loop_slots;
    loops.clear();
    Reader r(code);
    auto index = [&](size_t i, size_t n, const char* what) { if (i >= n) throw Error("LOAD_BAD_BYTECODE", std::string(what) + " index " + std::to_string(i) + " out of range (" + std::to_string(n) + " entries)"); };
    auto local_int_hidden = [&](uint16_t slot, const char* role) {
      index(slot, locals.size(), role);
      if (locals[slot].type != "int" || !locals[slot].hidden) throw Error("LOAD_BAD_BYTECODE", std::string(role) + " slot " + std::to_string(slot) + " must be a hidden int local");
    };
    bool halted = false;
    while (!r.done()) {
      const size_t pc = r.pos();
      const size_t depth = open.size();
      const size_t owner = open.empty() ? size_t(-1) : open.back().pc;
      const uint8_t op = r.u8();
      insns.push_back({pc, 0, 0, op, depth, owner});
      halted = false;
      switch (op) {
        case PUSH_INT: r.i64(); break;
        case PUSH_FLOAT: r.f64(); break;
        case PUSH_STR: index(r.u32(), consts_.size(), "constant"); break;
        case PUSH_BOOL: {
          uint8_t v = r.u8();
          if (v > 1) throw Error("LOAD_BAD_BYTECODE", "PUSH_BOOL operand must be 0 or 1");
          break;
        }
        case LOAD_LOCAL: case STORE_LOCAL: index(r.u16(), locals.size(), "local slot"); break;
        case LOAD_CTX: index(r.u16(), context_.size(), "context"); break;
        case CALL_FN: {
          uint16_t i = r.u16(); uint8_t argc = r.u8(); index(i, imports_.size(), "import");
          if (!imports_[i].fn) throw Error("LOAD_BAD_BYTECODE", "CALL_FN at offset " + std::to_string(pc) + " names import " + std::to_string(i) + ", which is a method");
          if (imports_[i].fn->sig.params.size() != argc) throw Error("LOAD_BAD_BYTECODE", "CALL_FN at offset " + std::to_string(pc) + " passes " + std::to_string(argc) + " argument(s), the import takes " + std::to_string(imports_[i].fn->sig.params.size()));
          break;
        }
        case CALL_METHOD: {
          uint16_t i = r.u16(); uint8_t argc = r.u8(); index(i, imports_.size(), "import");
          if (!imports_[i].method) throw Error("LOAD_BAD_BYTECODE", "CALL_METHOD at offset " + std::to_string(pc) + " names import " + std::to_string(i) + ", which is a function");
          if (imports_[i].method->sig.params.size() != argc) throw Error("LOAD_BAD_BYTECODE", "CALL_METHOD at offset " + std::to_string(pc) + " passes " + std::to_string(argc) + " argument(s), the import takes " + std::to_string(imports_[i].method->sig.params.size()));
          break;
        }
        case EQ: case NE: case NOT: case DUP: case POP: break;
        case ADD_I: case SUB_I: case MUL_I: case DIV_I: case MOD_I: case ADD_F: case SUB_F: case MUL_F: case DIV_F: case NEG_I: case NEG_F:
        case LT_I: case LE_I: case GT_I: case GE_I: case LT_F: case LE_F: case GT_F: case GE_F: case I2F: case CONCAT: case I2S: case F2S: case B2S: break;
        case MATH: {
          const uint8_t fn = r.u8();
          if (fn > MATH_CLAMP_I) throw Error("LOAD_BAD_BYTECODE", "unknown MATH function " + std::to_string(fn) + " at offset " + std::to_string(pc));
          break;
        }
        case JUMP: case JUMP_IF_FALSE: case JUMP_IF_TRUE: {
          int32_t rel = r.i32();
          if (rel < 0) throw Error("LOAD_BAD_BYTECODE", "jump at offset " + std::to_string(pc) + " goes backwards (only paired LOOP_BACK may jump backward)");
          insns.back().target = size_t(int64_t(r.pos()) + rel); ordinary.emplace_back(pc, int64_t(r.pos()) + rel); break;
        }
        case FOR_RANGE: {
          uint16_t counter = r.u16(), end = r.u16(); int32_t rel = r.i32();
          local_int_hidden(counter, "FOR_RANGE counter"); local_int_hidden(end, "FOR_RANGE end");
          if (counter == end) throw Error("LOAD_BAD_BYTECODE", "FOR_RANGE counter and end slots must be distinct");
          for (uint16_t used : used_loop_slots) {
            if (counter == used || end == used) throw Error("LOAD_BAD_BYTECODE", "FOR_RANGE slot " + std::to_string(counter == used ? counter : end) + " is already owned by another range loop");
          }
          used_loop_slots.push_back(counter); used_loop_slots.push_back(end);
          if (rel < 0) throw Error("LOAD_BAD_BYTECODE", "FOR_RANGE at offset " + std::to_string(pc) + " exits backwards");
          open.push_back({FOR_RANGE, pc, r.pos(), size_t(int64_t(r.pos()) + rel), counter, end, false, owner});
          insns.back().target = size_t(int64_t(r.pos()) + rel);
          break;
        }
        case FOR_NEXT: {
          uint16_t encoded_end = r.u16(); int32_t rel = r.i32();
          if (open.empty() || open.back().op != FOR_RANGE) throw Error("LOAD_BAD_BYTECODE", "FOR_NEXT at offset " + std::to_string(pc) + " has no matching FOR_RANGE");
          Open loop = open.back(); open.pop_back();
          if (encoded_end != loop.end) throw Error("LOAD_BAD_BYTECODE", "FOR_NEXT at offset " + std::to_string(pc) + " names end slot " + std::to_string(encoded_end) + ", expected " + std::to_string(loop.end));
          if (int64_t(r.pos()) + rel != int64_t(loop.next)) throw Error("LOAD_BAD_BYTECODE", "FOR_NEXT at offset " + std::to_string(pc) + " must return to its FOR_RANGE body");
          if (loop.exit != r.pos()) throw Error("LOAD_BAD_BYTECODE", "FOR_RANGE at offset " + std::to_string(loop.pc) + " must exit immediately after its FOR_NEXT");
          loops[pc] = Loop{loop.counter, loop.end, loop.next};
          insns.back().target = size_t(int64_t(r.pos()) + rel);
          regions[loop.pc] = LoopRegion{loop.exit, loop.parent, 0, FOR_RANGE, loop.counter, loop.end};
          break;
        }
        case WHILE_BEGIN:
          open.push_back({WHILE_BEGIN, pc, r.pos(), 0, 0, 0, false, owner}); break;
        case WHILE_TEST: {
          int32_t rel = r.i32();
          if (rel < 0) throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST at offset " + std::to_string(pc) + " exits backwards");
          if (open.empty() || open.back().op != WHILE_BEGIN) throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST at offset " + std::to_string(pc) + " has no matching WHILE_BEGIN");
          if (open.back().tested) throw Error("LOAD_BAD_BYTECODE", "WHILE_BEGIN at offset " + std::to_string(open.back().pc) + " has more than one WHILE_TEST");
          open.back().test = pc; open.back().tested = true; open.back().exit = size_t(int64_t(r.pos()) + rel); insns.back().target = open.back().exit; while_exits.emplace_back(pc, int64_t(r.pos()) + rel); break;
        }
        case LOOP_BACK: {
          int32_t rel = r.i32();
          if (open.empty() || open.back().op != WHILE_BEGIN) throw Error("LOAD_BAD_BYTECODE", "LOOP_BACK at offset " + std::to_string(pc) + " has no matching WHILE_BEGIN");
          Open loop = open.back(); open.pop_back();
          if (!loop.tested) throw Error("LOAD_BAD_BYTECODE", "WHILE_BEGIN at offset " + std::to_string(loop.pc) + " has no WHILE_TEST");
          if (int64_t(r.pos()) + rel != int64_t(loop.pc)) throw Error("LOAD_BAD_BYTECODE", "LOOP_BACK at offset " + std::to_string(pc) + " does not target its paired WHILE_BEGIN");
          if (loop.exit != r.pos()) throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST at offset " + std::to_string(loop.pc) + " must exit immediately after its LOOP_BACK");
          regions[loop.pc] = LoopRegion{loop.exit, loop.parent, loop.test, WHILE_BEGIN, 0, 0};
          insns.back().target = size_t(int64_t(r.pos()) + rel);
          break;
        }
        case HALT: if (depth != 0) throw Error("LOAD_BAD_BYTECODE", "HALT inside loop"); halted = true; break;
        default: throw Error("LOAD_BAD_BYTECODE", "unknown opcode " + std::to_string(op) + " at offset " + std::to_string(pc));
      }
      insns.back().next = r.pos();
    }
    if (!open.empty()) throw Error("LOAD_BAD_BYTECODE", std::string(open.back().op == FOR_RANGE ? "unclosed FOR_RANGE at offset " : "unclosed WHILE_BEGIN at offset ") + std::to_string(open.back().pc));
    if (!halted) throw Error("LOAD_BAD_BYTECODE", "code does not end with HALT");
    auto find_insn = [&](int64_t target) -> const VerifiedInstr* { for (auto& i : insns) if (int64_t(i.pc) == target) return &i; return nullptr; };
    for (auto& j : ordinary) {
      const VerifiedInstr* from = find_insn(int64_t(j.first)); const VerifiedInstr* to = find_insn(j.second);
      if (!to) throw Error("LOAD_BAD_BYTECODE", "jump at offset " + std::to_string(j.first) + " targets " + std::to_string(j.second) + ", not an instruction start");
      if (!from) throw Error("LOAD_BAD_BYTECODE", "internal verifier error: missing jump instruction");
      auto loop = regions.find(from->owner);
      if (loop != regions.end() && loop->second.op == WHILE_BEGIN &&
          from->pc < loop->second.test && to->pc > loop->second.test)
        throw Error("LOAD_BAD_BYTECODE", "jump at offset " + std::to_string(from->pc) + " skips its WHILE_TEST");
      if (from->owner == to->owner) continue;
      if (loop == regions.end() || size_t(j.second) != loop->second.exit || to->owner != loop->second.parent)
        throw Error("LOAD_BAD_BYTECODE", "jump at offset " + std::to_string(j.first) + " crosses a loop boundary (only a break to the innermost loop exit is allowed)");
    }
    for (auto& i : insns) if (i.op == STORE_LOCAL) {
      Reader q(code); q.seek(i.pc + 1); uint16_t slot = q.u16();
      for (auto& l : loops) if (slot == l.second.counter || slot == l.second.end)
        throw Error("LOAD_BAD_BYTECODE", "STORE_LOCAL at offset " + std::to_string(i.pc) + " writes a read-only range slot " + std::to_string(slot));
      if (locals[slot].hidden) throw Error("LOAD_BAD_BYTECODE", "STORE_LOCAL at offset " + std::to_string(i.pc) + " writes compiler-owned loop local " + std::to_string(slot));
    }
    for (auto& i : insns) if (i.op == FOR_RANGE) {
      uint16_t counter = uint16_t(uint8_t(code[i.pc + 1]) << 8 | uint8_t(code[i.pc + 2]));
      (void)counter;
      // Re-read the encoded exit displacement; its target must be a parent-region instruction.
      Reader q(code); q.seek(i.pc + 1); q.u16(); q.u16(); int32_t rel = q.i32();
      const VerifiedInstr* to = find_insn(int64_t(q.pos()) + rel);
      if (!to || to->owner != i.owner) throw Error("LOAD_BAD_BYTECODE", "FOR_RANGE at offset " + std::to_string(i.pc) + " exits to an invalid loop region");
    }
    for (auto& j : while_exits) {
      const VerifiedInstr* from = find_insn(int64_t(j.first)); const VerifiedInstr* to = find_insn(j.second);
      if (!to || !from || from->owner == size_t(-1)) throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST at offset " + std::to_string(j.first) + " exits to an invalid loop region");
      auto loop = regions.find(from->owner);
      if (loop == regions.end() || loop->second.op != WHILE_BEGIN || loop->second.exit != size_t(j.second) || to->owner != loop->second.parent)
        throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST at offset " + std::to_string(j.first) + " exits to an invalid loop region");
    }
    verify_stack(insns, code, halt_depth);
    for (auto& l : lines) if (l.pc >= code.size()) throw Error("LOAD_BAD_BYTECODE", "line table entry points at pc " + std::to_string(l.pc) + ", beyond the code");
  }

  void verify_stack(const std::vector<VerifiedInstr>& insns, const std::string& code, int halt_depth) const {
    if (halt_depth < 0 && std::none_of(insns.begin(), insns.end(), [](const VerifiedInstr& i) { return i.op == FOR_RANGE || i.op == WHILE_BEGIN; })) return;
    std::map<size_t, size_t> at;
    for (size_t i = 0; i < insns.size(); ++i) at[insns[i].pc] = i;
    std::vector<int64_t> incoming(insns.size(), -1);
    std::vector<size_t> work;
    if (insns.empty()) throw Error("LOAD_BAD_BYTECODE", "code is empty");
    incoming[0] = 0; work.push_back(0);
    auto merge = [&](size_t pc, int64_t depth) {
      auto found = at.find(pc);
      if (found == at.end()) throw Error("LOAD_BAD_BYTECODE", "control flow targets " + std::to_string(pc) + ", not an instruction start");
      int64_t& old = incoming[found->second];
      if (old < 0) { old = depth; work.push_back(found->second); }
      else if (old != depth) throw Error("LOAD_BAD_BYTECODE", "inconsistent operand stack depth at control-flow join " + std::to_string(pc) + ": " + std::to_string(old) + " versus " + std::to_string(depth));
    };
    auto math_arity = [](uint8_t fn) -> int {
      switch (fn) {
        case MATH_SIN_F: case MATH_COS_F: case MATH_SQRT_F: case MATH_ABS_F: case MATH_FLOOR_F: case MATH_EASE_F:
        case MATH_EASE_IN_F: case MATH_EASE_OUT_F: case MATH_INT_F: case MATH_ABS_I: return 1;
        case MATH_POW_F: case MATH_MIN_F: case MATH_MAX_F: case MATH_MIN_I: case MATH_MAX_I: return 2;
        case MATH_CLAMP_F: case MATH_LERP_F: case MATH_CLAMP_I: return 3;
        default: throw Error("LOAD_BAD_BYTECODE", "unknown MATH function " + std::to_string(fn));
      }
    };
    while (!work.empty()) {
      const size_t ix = work.back(); work.pop_back();
      const VerifiedInstr& in = insns[ix];
      Reader r(code); r.seek(in.pc); const uint8_t op = r.u8();
      int pop = 0, push = 0;
      bool fallthrough = true, branch = false, conditional = false;
      switch (op) {
        case PUSH_INT: case PUSH_FLOAT: case PUSH_STR: case PUSH_BOOL: case LOAD_LOCAL: case LOAD_CTX: push = 1; break;
        case STORE_LOCAL: case POP: pop = 1; break;
        case CALL_FN: {
          uint16_t i = r.u16(); uint8_t argc = r.u8(); pop = argc; push = imports_[i].fn->sig.returns == "void" ? 0 : 1; break;
        }
        case CALL_METHOD: {
          uint16_t i = r.u16(); uint8_t argc = r.u8(); pop = int(argc) + 1; push = imports_[i].method->sig.returns == "void" ? 0 : 1; break;
        }
        case EQ: case NE: case ADD_I: case SUB_I: case MUL_I: case DIV_I: case MOD_I: case ADD_F: case SUB_F: case MUL_F: case DIV_F:
        case LT_I: case LE_I: case GT_I: case GE_I: case LT_F: case LE_F: case GT_F: case GE_F: case CONCAT: pop = 2; push = 1; break;
        case NOT: case NEG_I: case NEG_F: case I2F: case I2S: case F2S: case B2S: pop = 1; push = 1; break;
        case MATH: { uint8_t fn = r.u8(); pop = math_arity(fn); push = 1; break; }
        case DUP: pop = 1; push = 2; break;
        case JUMP: branch = true; fallthrough = false; break;
        case JUMP_IF_FALSE: case JUMP_IF_TRUE: pop = 1; branch = true; conditional = true; break;
        case FOR_RANGE: pop = 2; branch = true; conditional = true; break;
        case FOR_NEXT: branch = true; conditional = true; break;
        case WHILE_BEGIN: break;
        case WHILE_TEST: pop = 1; branch = true; conditional = true; break;
        case LOOP_BACK: branch = true; fallthrough = false; break;
        case HALT: fallthrough = false; break;
        default: throw Error("LOAD_BAD_BYTECODE", "unknown opcode during stack verification");
      }
      const int64_t depth = incoming[ix];
      if (depth < pop) throw Error("LOAD_BAD_BYTECODE", "operand stack underflow at offset " + std::to_string(in.pc));
      const int64_t after = depth - pop + push;
      if (branch) merge(in.target, after);
      if (conditional && (op == JUMP_IF_FALSE || op == JUMP_IF_TRUE || op == FOR_RANGE || op == FOR_NEXT || op == WHILE_TEST)) merge(in.next, after);
      else if (fallthrough) {
        if (ix + 1 < insns.size()) merge(in.next, after);
        else throw Error("LOAD_BAD_BYTECODE", "control flow falls past the end of code");
      }
    }
    for (size_t ix = 0; ix < insns.size(); ++ix) {
      const auto& in = insns[ix];
      if (in.op == WHILE_TEST && incoming[ix] >= 0 &&
          incoming[ix] != incoming[at.at(in.owner)] + 1)
        throw Error("LOAD_BAD_BYTECODE", "WHILE_TEST must leave the loop-head stack depth at offset " + std::to_string(in.pc));
    }
    if (halt_depth >= 0) {
      for (size_t ix = 0; ix < insns.size(); ++ix) {
        if (insns[ix].op == HALT && incoming[ix] >= 0 && incoming[ix] != halt_depth)
          throw Error("LOAD_BAD_BYTECODE", "HALT leaves " + std::to_string(incoming[ix]) + " values");
      }
    }
  }
  const Registry* bound_registry_ = nullptr; // Same lifetime as the bound import entries.
  struct EntryRegion {  // an entry's private code region; meta is the public Entry view
    Entry meta;
    uint16_t nparams = 0;
    std::vector<Local> locals;
    uint16_t max_stack = 0;
    std::string code;
    std::vector<Line> lines;
    std::map<size_t, Loop> loops;
  };
  std::vector<EntryRegion> entries_;
  std::vector<Entry> entry_meta_;
  std::vector<std::string> consts_; std::vector<Import> imports_; std::vector<std::string> context_; std::vector<Local> locals_; uint16_t max_stack_ = 0; std::string code_; std::vector<Line> lines_; std::map<size_t, Loop> loops_;
};

inline std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("LOAD_BAD_BYTECODE", "cannot read " + path);
  std::ostringstream s; s << in.rdbuf(); return s.str();
}

}  // namespace dslvm

// Support the repository include/ layout and the hosts' pinned cpp/stdlib layout.
#ifndef DSLVM_STD_DEFER
#if __has_include("stdlib/std.hpp")
#include "stdlib/std.hpp"
#else
#include "../stdlib/std.hpp"
#endif
#endif
