#pragma once
// Opt-in `rest` std module (lmcode_std_rest_v1): a REST server over stdio. NOT part of the default std catalog and
// NOT included by std.hpp / register_std. Spec: spec/std-rest.md of github.com/lmctl-ai/lmcode.
//
//   dslvm::Registry registry;                                  // default std only (206 signatures)
//   auto rest = dslvm::stdlib::register_rest(registry, {});    // + exactly 18 rest signatures
//   auto program = dslvm::Program::load(bytecode, hostSchemaBytes, registry);
//   rest->run(program, context);   // runs the body, then serves input -> output until EOF / rest_stop() / framing error
//
// Copy rest.schema-fragment.json (== std/rest/lmcode-rest.schema.json) into the host schema. There is no network
// listener: requests are read from `options.input` (default std::cin) and responses written to `options.output`
// (default std::cout). While serving, stdout belongs to the protocol: point the registry's `print` at stderr
// (registry.set_std_output(&std::cerr)). On Windows put the process streams in binary mode (see examples/rest_serve.cpp).
// Strings are lmcode byte strings and stay byte-faithful. Everything is single threaded: handlers run strictly one at a
// time, after the program body has returned.
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
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <istream>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dslvm::stdlib::rest {

using Headers = std::vector<std::pair<std::string, std::string>>;
struct Request { std::string method, target; Headers headers; std::string body; };
struct Response { int status = 200; Headers headers; std::string body; };

constexpr size_t MAX_HEADER_BYTES = 16 * 1024;
inline const char* HANDLER_INTERFACE = "interface RestHandler(RestExchange) -> void";
inline const char* REST_GET_INTERFACE = "interface RestGet(RestExchange) -> void";
inline const char* REST_POST_INTERFACE = "interface RestPost(RestExchange) -> void";
inline const char* REST_PUT_INTERFACE = "interface RestPut(RestExchange) -> void";
inline const char* REST_DELETE_INTERFACE = "interface RestDelete(RestExchange) -> void";
inline const char* REST_PATCH_INTERFACE = "interface RestPatch(RestExchange) -> void";
inline const char* REST_HEAD_INTERFACE = "interface RestHead(RestExchange) -> void";
inline const char* REST_OPTIONS_INTERFACE = "interface RestOptions(RestExchange) -> void";
inline const char* CLI_COMMAND_INTERFACE = "interface CliCommand(RestExchange) -> void";
inline const char* CLI_OPTION_INTERFACE = "interface CliOption(RestExchange) -> void";

inline const char* method_for_rest_interface(const std::string& iface) {
  if (iface == REST_GET_INTERFACE) return "GET";
  if (iface == REST_POST_INTERFACE) return "POST";
  if (iface == REST_PUT_INTERFACE) return "PUT";
  if (iface == REST_DELETE_INTERFACE) return "DELETE";
  if (iface == REST_PATCH_INTERFACE) return "PATCH";
  if (iface == REST_HEAD_INTERFACE) return "HEAD";
  if (iface == REST_OPTIONS_INTERFACE) return "OPTIONS";
  return nullptr;
}

inline bool is_rest_or_cli_interface(const std::string& iface) {
  return iface == HANDLER_INTERFACE || method_for_rest_interface(iface) != nullptr ||
         iface == CLI_COMMAND_INTERFACE || iface == CLI_OPTION_INTERFACE;
}

// ---- bytes helpers: whitespace is ASCII only (0x00-0x20, 0x7f); 0xA0 is an ordinary byte ----
inline bool is_ws(char c) { unsigned char u = static_cast<unsigned char>(c); return u <= 0x20 || u == 0x7f; }
inline std::string lower(std::string s) { for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32); return s; }
inline std::string upper(std::string s) { for (char& c : s) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32); return s; }
inline int hex_digit(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }
inline Error host_error(const std::string& message) { return Error("RUN_HOST", message); }

// ---- router --------------------------------------------------------------------------------------------------
inline const std::vector<std::string>& methods() {
  static const std::vector<std::string> m{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"};
  return m;
}

/** Percent-decode a byte string; false on an invalid escape. `plus` turns '+' into a space (queries). */
inline bool percent_decode(const std::string& text, std::string& out, bool plus = false) {
  out.clear();
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c == '%') {
      int hi = i + 1 < text.size() ? hex_digit(text[i + 1]) : -1, lo = i + 2 < text.size() ? hex_digit(text[i + 2]) : -1;
      if (hi < 0 || lo < 0) return false;
      out.push_back(static_cast<char>(hi * 16 + lo));
      i += 2;
    } else if (plus && c == '+') out.push_back(' ');
    else out.push_back(c);
  }
  return true;
}

struct Segment { enum Kind { LITERAL = 0, PARAM = 1, REST = 2 } kind = LITERAL; std::string text; };  // text: literal value or parameter name
struct Route { std::string method, path, annotation, shape; std::vector<Segment> segments; };

inline std::vector<std::string> split_slash(const std::string& s) {
  std::vector<std::string> out; size_t from = 0;
  for (;;) { size_t at = s.find('/', from); if (at == std::string::npos) { out.push_back(s.substr(from)); return out; } out.push_back(s.substr(from, at - from)); from = at + 1; }
}

inline Route parse_route(const std::string& method, const std::string& path) {
  const std::string label = method + " " + path;
  auto invalid = [&](const std::string& reason) { throw host_error("rest: invalid route " + label + ": " + reason); };
  if (std::find(methods().begin(), methods().end(), method) == methods().end()) {
    std::string all; for (const auto& m : methods()) all += (all.empty() ? "" : ", ") + m;
    invalid("method must be one of " + all);
  }
  if (path.empty() || path[0] != '/') invalid("path must start with '/'");
  for (char c : path) if (is_ws(c) || c == '?' || c == '#') invalid("path must not contain whitespace, control characters, ? or #");
  Route route; route.method = method; route.path = path;
  const auto raw = split_slash(path.substr(1));
  std::set<std::string> names;
  for (size_t index = 0; index < raw.size(); ++index) {
    const std::string& text = raw[index];
    const bool last = index + 1 == raw.size();
    if (text.empty() && !last) invalid("empty segment ('//')");
    Segment seg;
    if (text.size() >= 2 && text.front() == '{' && text.back() == '}') {
      std::string name = text.substr(1, text.size() - 2);
      const bool rest = name.size() >= 3 && name.compare(name.size() - 3, 3, "...") == 0;
      if (rest) name.resize(name.size() - 3);
      bool ok = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
      for (char c : name) if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) ok = false;
      if (!ok) invalid("invalid parameter name '" + name + "'");
      if (!names.insert(name).second) invalid("duplicate parameter '" + name + "'");
      if (rest && !last) invalid("'{" + name + "...}' must be the last segment");
      seg.kind = rest ? Segment::REST : Segment::PARAM; seg.text = name;
    } else if (!text.empty() && text.front() == ':') {
      std::string name = text.substr(1);
      const bool rest = name.size() >= 3 && name.compare(name.size() - 3, 3, "...") == 0;
      if (rest) name.resize(name.size() - 3);
      bool ok = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
      for (char c : name) if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) ok = false;
      if (!ok) invalid("invalid parameter name '" + name + "'");
      if (!names.insert(name).second) invalid("duplicate parameter '" + name + "'");
      if (rest && !last) invalid("':" + name + "...' must be the last segment");
      seg.kind = rest ? Segment::REST : Segment::PARAM; seg.text = name;
    } else {
      if (text.find_first_of("{}") != std::string::npos) invalid("malformed placeholder in segment '" + text + "'");
      seg.kind = Segment::LITERAL; seg.text = text;
    }
    route.shape += (index ? "/" : "");
    route.shape += seg.kind == Segment::LITERAL ? "L" + seg.text : seg.kind == Segment::PARAM ? "P" : "R";
    route.segments.push_back(std::move(seg));
  }
  return route;
}

/** Outcome of RouteTable::match: a route + captured parameters, or 404, or 405 + Allow. */
struct Match {
  int status = 0;                // 0 = route matched, 404, 405
  const Route* route = nullptr;
  std::map<std::string, std::string> params;
  std::string allow;
};

class RouteTable {
 public:
  size_t size() const { return routes_.size(); }
  const std::vector<Route>& routes() const { return routes_; }
  void add(const std::string& method, const std::string& path, const std::string& annotation) {
    Route route = parse_route(method, path);
    for (const auto& r : routes_) if (r.method == method && r.shape == route.shape) throw host_error("rest: duplicate route " + method + " " + path);
    route.annotation = annotation;
    routes_.push_back(std::move(route));
  }
  /** `raw_path` is the undecoded request path without the query. */
  Match match(const std::string& method, const std::string& raw_path) const {
    const auto raw = split_slash(raw_path.substr(1));
    struct Hit { const Route* route; std::map<std::string, std::string> params; };
    std::vector<Hit> hits;
    for (const auto& route : routes_) { Hit hit{&route, {}}; if (match_segments(route, raw, hit.params)) hits.push_back(std::move(hit)); }
    Match out;
    if (hits.empty()) { out.status = 404; return out; }
    auto best = [&](const std::string& verb) -> const Hit* {
      const Hit* found = nullptr;
      for (const auto& h : hits) if (h.route->method == verb && (!found || more_specific(*h.route, *found->route))) found = &h;
      return found;
    };
    const Hit* direct = best(method);
    if (!direct && method == "HEAD") direct = best("GET");
    if (direct) { out.route = direct->route; out.params = direct->params; return out; }
    std::set<std::string> allow;
    for (const auto& h : hits) allow.insert(h.route->method);
    if (allow.count("GET")) allow.insert("HEAD");
    out.status = 405;
    for (const auto& m : allow) out.allow += (out.allow.empty() ? "" : ", ") + m;
    return out;
  }
 private:
  // True when a is strictly more specific than b (segment by segment: literal < {p} < {p...}; shorter wins a full tie).
  static bool more_specific(const Route& a, const Route& b) {
    size_t n = std::min(a.segments.size(), b.segments.size());
    for (size_t i = 0; i < n; ++i) if (a.segments[i].kind != b.segments[i].kind) return a.segments[i].kind < b.segments[i].kind;
    return a.segments.size() < b.segments.size();
  }
  static bool match_segments(const Route& route, const std::vector<std::string>& raw, std::map<std::string, std::string>& params) {
    const auto& segs = route.segments;
    for (size_t i = 0; i < segs.size(); ++i) {
      const Segment& s = segs[i];
      if (s.kind == Segment::REST) {
        if (raw.size() <= i) return false;
        std::string joined, d;
        for (size_t j = i; j < raw.size(); ++j) { if (!percent_decode(raw[j], d)) return false; if (j > i) joined.push_back('/'); joined += d; }
        params[s.text] = joined;
        return true;
      }
      if (i >= raw.size()) return false;
      std::string d;
      if (!percent_decode(raw[i], d)) return false;
      if (s.kind == Segment::LITERAL) { if (d != s.text) return false; }
      else { if (d.empty()) return false; params[s.text] = d; }
    }
    return raw.size() == segs.size();
  }
  std::vector<Route> routes_;
};

// ---- HTTP subset: framing, request parsing, response encoding ------------------------------------------------------
inline std::string reason_phrase(int status) {
  switch (status) {
    case 200: return "OK"; case 201: return "Created"; case 202: return "Accepted"; case 204: return "No Content";
    case 301: return "Moved Permanently"; case 302: return "Found"; case 304: return "Not Modified";
    case 400: return "Bad Request"; case 401: return "Unauthorized"; case 403: return "Forbidden"; case 404: return "Not Found";
    case 405: return "Method Not Allowed"; case 408: return "Request Timeout"; case 409: return "Conflict"; case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type"; case 422: return "Unprocessable Entity"; case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large"; case 500: return "Internal Server Error"; case 501: return "Not Implemented";
    case 502: return "Bad Gateway"; case 503: return "Service Unavailable";
  }
  return status < 300 ? "OK" : status < 400 ? "Redirect" : status < 500 ? "Client Error" : "Server Error";
}

inline const char* error_body(int status) {
  switch (status) {
    case 400: return "{\"error\":\"bad request\"}"; case 404: return "{\"error\":\"not found\"}";
    case 405: return "{\"error\":\"method not allowed\"}"; case 413: return "{\"error\":\"payload too large\"}";
    case 431: return "{\"error\":\"request header too large\"}"; case 500: return "{\"error\":\"internal error\"}";
    case 501: return "{\"error\":\"not implemented\"}";
  }
  return "";
}

inline Response error_response(int status, Headers extra = {}) {
  Response r; r.status = status; r.headers.push_back({"Content-Type", "application/json"});
  for (auto& h : extra) r.headers.push_back(std::move(h));
  r.body = error_body(status);
  return r;
}

struct Head { int error = 0; std::string method, target; Headers headers; };  // error: 0 = ok, else the status to answer

/** Parse the header section (without the terminating blank line). */
inline Head parse_head(const std::string& text) {
  Head head;
  auto bad = [&]() { Head h; h.error = 400; return h; };
  const size_t eol = text.find("\r\n");
  const std::string first = text.substr(0, eol);
  // METHOD SP target SP HTTP/1.0|1.1 ; METHOD = [A-Z]+, target = non-whitespace bytes (ASCII whitespace only)
  const size_t sp = first.find(' ');
  if (sp == std::string::npos || sp == 0) return bad();
  for (size_t i = 0; i < sp; ++i) if (first[i] < 'A' || first[i] > 'Z') return bad();
  if (first.size() < sp + 1 + 1 + 9) return bad();
  const std::string tail = first.substr(first.size() - 9);
  if (tail != " HTTP/1.0" && tail != " HTTP/1.1") return bad();
  const std::string target = first.substr(sp + 1, first.size() - 9 - (sp + 1));
  if (target.empty()) return bad();
  for (char c : target) if (is_ws(c)) return bad();
  if (target[0] != '/' || target.find('#') != std::string::npos) return bad();
  head.method = first.substr(0, sp); head.target = target;
  size_t pos = eol == std::string::npos ? text.size() : eol + 2;
  while (eol != std::string::npos && pos <= text.size()) {
    size_t next = text.find("\r\n", pos);
    const std::string line = text.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
    const size_t colon = line.find(':');
    if (colon == std::string::npos || colon == 0) return bad();
    for (size_t i = 0; i < colon; ++i) if (is_ws(line[i])) return bad();
    for (char c : line) { unsigned char u = static_cast<unsigned char>(c); if ((u <= 0x08) || (u >= 0x0a && u <= 0x1f) || u == 0x7f) return bad(); }
    std::string value = line.substr(colon + 1);
    const size_t a = value.find_first_not_of(" \t"), b = value.find_last_not_of(" \t");
    value = a == std::string::npos ? "" : value.substr(a, b - a + 1);
    head.headers.push_back({line.substr(0, colon), value});
    if (next == std::string::npos) break;
    pos = next + 2;
  }
  return head;
}

struct BodyLength { int error = 0; uint64_t length = 0; };
/** Content-Length (or 0), or an error status. Transfer-Encoding is unsupported (501). */
inline BodyLength body_length(const Headers& headers) {
  BodyLength out;
  for (const auto& h : headers) if (lower(h.first) == "transfer-encoding") { out.error = 501; return out; }
  std::vector<std::string> values;
  for (const auto& h : headers) if (lower(h.first) == "content-length") values.push_back(h.second);
  if (values.empty()) return out;
  for (const auto& v : values) if (v != values[0]) { out.error = 400; return out; }
  const std::string& v = values[0];
  if (v.empty() || v.size() > 15) { out.error = 400; return out; }
  for (char c : v) if (c < '0' || c > '9') { out.error = 400; return out; }
  out.length = std::stoull(v);
  return out;
}

inline std::string encode_response(const Response& r, bool head = false) {
  std::string out = "HTTP/1.1 " + std::to_string(r.status) + " " + reason_phrase(r.status) + "\r\n";
  for (const auto& h : r.headers) out += h.first + ": " + h.second + "\r\n";
  out += "Content-Length: " + std::to_string(r.body.size()) + "\r\n\r\n";
  return head ? out : out + r.body;
}

/** Outcome of reading one request off a stream. */
struct Taken { enum Kind { REQUEST, END, FATAL } kind = END; int fatal = 0; Request request; };

/**
 * Read one framed request. END: EOF before a complete request (a truncated one is dropped silently). FATAL: a framing
 * error that cannot be resynchronised (answer once, then stop). Reads exactly the bytes of the request, no more.
 */
inline Taken take_request(std::istream& in, uint64_t max_body_bytes) {
  Taken out;
  std::streambuf* buf = in.rdbuf();
  std::string head;
  for (;;) {
    int c = buf ? buf->sbumpc() : std::char_traits<char>::eof();
    if (c == std::char_traits<char>::eof()) return out;
    head.push_back(static_cast<char>(c));
    const size_t n = head.size();
    if (n >= 4 && head.compare(n - 4, 4, "\r\n\r\n") == 0) { head.resize(n - 4); break; }
    if (n >= MAX_HEADER_BYTES + 4) { out.kind = Taken::FATAL; out.fatal = 431; return out; }
  }
  Head parsed = parse_head(head);
  if (parsed.error) { out.kind = Taken::FATAL; out.fatal = parsed.error; return out; }
  BodyLength size = body_length(parsed.headers);
  if (size.error) { out.kind = Taken::FATAL; out.fatal = size.error; return out; }
  if (size.length > max_body_bytes) { out.kind = Taken::FATAL; out.fatal = 413; return out; }
  std::string body(static_cast<size_t>(size.length), '\0');
  if (size.length) {
    std::streamsize got = buf->sgetn(&body[0], static_cast<std::streamsize>(size.length));
    if (got != static_cast<std::streamsize>(size.length)) return out;
  }
  out.kind = Taken::REQUEST;
  out.request = Request{parsed.method, parsed.target, std::move(parsed.headers), std::move(body)};
  return out;
}

// ---- the controller --------------------------------------------------------------------------------------------------
struct State { std::map<std::string, std::string> values; std::map<std::string, int64_t> counts; };

struct Options {
  std::istream* input = nullptr;                 // default std::cin
  std::ostream* output = nullptr;                // default std::cout
  uint64_t max_body_bytes = 1024 * 1024;
  std::function<void(const std::string& entry, const std::string& message)> on_handler_error;  // default: one line on stderr
};

/** The per-request object behind `RestExchange`. */
struct Exchange {
  State* state = nullptr;
  Request request;
  std::map<std::string, std::string> params, query;
  std::string path;
  int status = 200;
  struct Header { std::string lower, name, value; };
  std::vector<Header> headers;                   // insertion order; a repeated name replaces in place
  std::string body;
  std::vector<std::string> request_header(const std::string& name) const {
    std::vector<std::string> out; const std::string wanted = lower(name);
    for (const auto& h : request.headers) if (rest::lower(h.first) == wanted) out.push_back(h.second);
    return out;
  }
  void put_header(const std::string& name, const std::string& value) {
    const std::string key = lower(name);
    for (auto& h : headers) if (h.lower == key) { h.name = name; h.value = value; return; }
    headers.push_back({key, name, value});
  }
  bool has_response_header(const std::string& key) const { for (const auto& h : headers) if (h.lower == key) return true; return false; }
};

class RestController {
 public:
  RestController(Registry& registry, Options options) : registry_(&registry), options_(std::move(options)) {
    if (!options_.input) options_.input = &std::cin;
    if (!options_.output) options_.output = &std::cout;
    if (!options_.on_handler_error) options_.on_handler_error = [](const std::string& entry, const std::string& message) { std::cerr << "rest: handler " << entry << " failed: " << message << "\n"; };
  }
  RestController(const RestController&) = delete;
  RestController& operator=(const RestController&) = delete;

  void bind_registry(Registry* registry) { registry_ = registry; }

  State& state() { return state_; }
  const RouteTable& table() const { return table_; }
  bool serving() const { return serving_; }
  bool stopped() const { return stopped_; }

  /**
   * Binds `program`, runs its body, then (if the body called rest_serve and not rest_stop) serves the input stream until EOF,
   * rest_stop() or a framing error. `program` and the registry must outlive the call.
   */
  void run(const Program& program, const Context& context = Context()) {
    struct RunFlag { bool& f; ~RunFlag() { f = false; } } flag{in_controller_run_};
    in_controller_run_ = true;
    program_ = &program; context_ = context; stopped_ = false;
    bind_entries(program.entries());
    program.run(context_, *registry_);
    if (serving_ && !stopped_) pump();
  }

  bool before_body_run(const Program& program, const Context& context, const Registry& registry) {
    if (in_controller_run_) return false;
    registry_ = const_cast<Registry*>(&registry);
    program_ = &program;
    context_ = context;
    table_ = RouteTable{};
    serving_ = false;
    stopped_ = false;
    bind_entries(program.entries());
    return true;
  }

  void after_body_run() {
    if (serving_ && !stopped_) pump();
  }

  // ---- lmcode-facing operations -------------------------------------------------------------------------
  void config(const std::string& key, const std::string& value) {
    if (serving_) throw host_error("rest: routes are frozen once rest_serve has been called");
    state_.values[key] = value;
    if (key == "prefix" || key == "base_path") {
      config_prefix_ = value;
    } else if (key == "max_body_bytes" && !value.empty()) {
      uint64_t n = 0;
      bool ok = true;
      for (char c : value) {
        if (c < '0' || c > '9') { ok = false; break; }
        n = n * 10 + uint64_t(c - '0');
      }
      if (ok && n > 0) options_.max_body_bytes = n;
    }
  }

  void route(const std::string& method, const std::string& path, const std::string& annotation) {
    if (serving_) throw host_error("rest: routes are frozen once rest_serve has been called");
    table_.add(method, path, annotation);
  }

  /** Records that serving was requested after the body; validates now, before any byte is read. */
  void serve() {
    if (serving_) throw host_error("rest: rest_serve already called");
    if (!bound_) throw host_error("rest: no program is bound (run the program through the controller)");
    materialize_auto_routes();
    validate();
    serving_ = true;
  }

  /** Ends serving once the current response is written. */
  void stop() { stopped_ = true; }

  /** Offers the entries a program provides (run() does this); exposed for validation tests. */
  void bind_entries(const std::vector<Program::Entry>& entries) {
    entries_ = entries;
    bound_ = true;
    auto_routes_materialized_ = false;
    auto_entry_index_.clear();
    auto_entry_indices_.clear();
    entry_index_.clear();
    cli_options_.clear();
    config_prefix_.clear();
  }

  std::string cli(const std::vector<std::string>& argv) {
    if (!bound_ || !program_ || !registry_)
      throw host_error("rest: no program is bound (run the program through the controller)");
    ensure_dispatch_ready();

    std::string method = "GET";
    std::string body;
    bool method_explicit = false;
    std::vector<std::string> positionals;
    struct OptVal { std::string flag, key, val; };
    std::vector<OptVal> parsed_opts;

    for (size_t i = 0; i < argv.size(); ++i) {
      const std::string& arg = argv[i];
      if (arg == "--") {
        std::string rest_body;
        for (size_t j = i + 1; j < argv.size(); ++j) {
          if (!rest_body.empty()) rest_body += " ";
          rest_body += argv[j];
        }
        if (body.empty()) body = rest_body;
        break;
      }
      if (arg.size() > 1 && arg[0] == '-') {
        std::string flag = arg, key, val = "true";
        const size_t eq = arg.find('=');
        if (eq != std::string::npos) {
          flag = arg.substr(0, eq);
          val = arg.substr(eq + 1);
          key = flag.rfind("--", 0) == 0 ? flag.substr(2) : flag.substr(1);
        } else {
          key = flag.rfind("--", 0) == 0 ? flag.substr(2) : flag.substr(1);
          if (i + 1 < argv.size() && (argv[i + 1].empty() || argv[i + 1][0] != '-')) {
            const std::string& next = argv[i + 1];
            const bool is_special = (key == "method" || key == "X" || key == "body" || key == "d");
            const std::string probe_path = next.front() == '/' ? next : ("/" + next);
            const bool next_is_root_cmd = positionals.empty() && !is_special &&
                                          table_.match("GET", probe_path).status != 404;
            if (!next_is_root_cmd) {
              val = argv[++i];
            }
          }
        }
        if (key == "method" || key == "X") {
          method = upper(val);
          method_explicit = true;
        } else if (key == "body" || key == "d") {
          body = val;
        }
        parsed_opts.push_back({flag, key, val});
      } else {
        positionals.push_back(arg);
      }
    }

    std::string path = "/";
    if (!positionals.empty()) {
      std::string joined;
      for (size_t i = 0; i < positionals.size(); ++i) {
        const std::string& seg = positionals[i];
        if (i == 0 && !seg.empty() && seg.front() == '/') joined = seg;
        else joined += "/" + seg;
      }
      path = joined.empty() ? "/" : joined;
    }
    const std::string prefix = normalized_prefix();
    if (!prefix.empty() && path.rfind(prefix, 0) != 0) {
      path = (path == "/" ? prefix : prefix + path);
    }

    Exchange x;
    x.state = &state_;
    x.request.method = method;
    x.request.target = path;
    x.request.body = body;
    x.path = path;
    for (const auto& opt : parsed_opts) {
      x.query[opt.key] = opt.val;
      x.query[opt.flag] = opt.val;
      state_.values[opt.key] = opt.val;
    }

    for (const auto& opt : parsed_opts) {
      auto it = cli_options_.find(opt.flag);
      if (it == cli_options_.end()) it = cli_options_.find(opt.key);
      if (it != cli_options_.end()) {
        call_entry_isolated(it->second.first, x);
      }
    }

    if (stopped_) return response_of(x).body;

    Match match = table_.match(method, path);
    if (match.status == 405 && !method_explicit) {
      if (!body.empty() && table_.match("POST", path).status == 0) {
        x.request.method = "POST";
        match = table_.match("POST", path);
      } else {
        for (const char* cand : {"POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}) {
          Match probe = table_.match(cand, path);
          if (probe.status == 0) {
            x.request.method = cand;
            match = std::move(probe);
            break;
          }
        }
      }
    }

    if (match.status == 404) {
      if (positionals.empty() && !x.body.empty()) return response_of(x).body;
      return error_response(404).body;
    }
    if (match.status == 405) return error_response(405, {{"Allow", match.allow}}).body;
    auto target = entry_index_.find(match.route->annotation);
    if (target == entry_index_.end()) return error_response(500).body;
    x.params = std::move(match.params);
    try {
      call_entry_isolated(target->second.first, x);
    } catch (const std::exception& e) {
      options_.on_handler_error(target->second.second, e.what());
      return error_response(500).body;
    } catch (...) {
      options_.on_handler_error(target->second.second, "unknown error");
      return error_response(500).body;
    }
    return response_of(x).body;
  }

  // ---- the stream-independent core ---------------------------------------------------------------------------
  Response dispatch(const Request& request) {
    ensure_dispatch_ready();
    const size_t mark = request.target.find('?');
    const std::string raw_path = mark == std::string::npos ? request.target : request.target.substr(0, mark);
    const std::string query_text = mark == std::string::npos ? "" : request.target.substr(mark + 1);
    Exchange x; x.state = &state_;
    if (!percent_decode(raw_path, x.path)) return error_response(400);
    size_t from = 0;
    while (!query_text.empty() && from <= query_text.size()) {
      size_t amp = query_text.find('&', from);
      const std::string pair = query_text.substr(from, amp == std::string::npos ? std::string::npos : amp - from);
      from = amp == std::string::npos ? query_text.size() + 1 : amp + 1;
      if (pair.empty()) continue;
      const size_t eq = pair.find('=');
      std::string key, value;
      if (!percent_decode(eq == std::string::npos ? pair : pair.substr(0, eq), key, true) ||
          !percent_decode(eq == std::string::npos ? std::string() : pair.substr(eq + 1), value, true)) return error_response(400);
      x.query.emplace(key, value);                // the first occurrence wins
    }
    Match match = table_.match(request.method, raw_path);
    if (match.status == 404) return error_response(404);
    if (match.status == 405) return error_response(405, {{"Allow", match.allow}});
    auto target = entry_index_.find(match.route->annotation);
    if (target == entry_index_.end()) throw host_error("rest: no program is serving (call rest_serve first)");
    x.request = request; x.params = std::move(match.params);
    try {
      call_entry_isolated(target->second.first, x);
    } catch (const std::exception& e) {
      options_.on_handler_error(target->second.second, e.what());
      return error_response(500);
    } catch (...) {
      options_.on_handler_error(target->second.second, "unknown error");
      return error_response(500);
    }
    return response_of(x);
  }

 private:
  std::string normalized_prefix() const {
    if (config_prefix_.empty()) return {};
    std::string p = config_prefix_;
    if (p.front() != '/') p = "/" + p;
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    return p == "/" ? std::string{} : p;
  }

  void materialize_auto_routes() {
    if (auto_routes_materialized_) return;
    auto_routes_materialized_ = true;
    const std::string prefix = normalized_prefix();
    for (size_t i = 0; i < entries_.size(); ++i) {
      const auto& e = entries_[i];
      if (e.iface == CLI_OPTION_INTERFACE) {
        cli_options_[e.annotation] = {i, e.name};
        std::string stripped = e.annotation;
        while (!stripped.empty() && stripped.front() == '-') stripped.erase(stripped.begin());
        if (!stripped.empty()) cli_options_[stripped] = {i, e.name};
        auto_entry_indices_.insert(i);
        continue;
      }
      std::string method, path;
      if (const char* m = method_for_rest_interface(e.iface)) {
        const size_t sp = e.annotation.find(' ');
        if (sp != std::string::npos) {
          method = e.annotation.substr(0, sp);
          path = e.annotation.substr(sp + 1);
        } else if (!e.annotation.empty() && e.annotation.front() == '/') {
          method = m;
          path = e.annotation;
        } else {
          continue;
        }
      } else if (e.iface == HANDLER_INTERFACE) {
        const size_t sp = e.annotation.find(' ');
        if (sp != std::string::npos) {
          method = e.annotation.substr(0, sp);
          path = e.annotation.substr(sp + 1);
        } else if (!e.annotation.empty() && e.annotation.front() == '/') {
          method = "GET";
          path = e.annotation;
        } else {
          continue;
        }
      } else if (e.iface == CLI_COMMAND_INTERFACE) {
        const size_t sp = e.annotation.find(' ');
        if (sp != std::string::npos) {
          method = e.annotation.substr(0, sp);
          path = e.annotation.substr(sp + 1);
        } else {
          method = "GET";
          path = e.annotation;
        }
        if (path.empty() || path.front() != '/') path = "/" + path;
      } else {
        continue;
      }
      if (!prefix.empty() && path.rfind(prefix, 0) != 0) {
        path = (path == "/" ? prefix : prefix + path);
      }
      const std::string tag = "__auto_entry_" + std::to_string(i);
      table_.add(method, path, tag);
      auto_entry_index_[tag] = {i, e.name};
      auto_entry_indices_.insert(i);
    }
  }

  void ensure_dispatch_ready() {
    if (!bound_) return;
    materialize_auto_routes();
    if (entry_index_.empty() && !auto_entry_index_.empty()) {
      entry_index_ = auto_entry_index_;
      for (size_t i = 0; i < entries_.size(); ++i) {
        if (!is_rest_or_cli_interface(entries_[i].iface) || entries_[i].iface == CLI_OPTION_INTERFACE) continue;
        if (auto_entry_indices_.count(i)) continue;
        entry_index_.emplace(entries_[i].annotation, std::make_pair(i, entries_[i].name));
      }
    }
  }

  void call_entry_isolated(size_t entry_idx, Exchange& x);

  void validate() {
    if (!table_.size()) throw host_error("rest: no routes registered");
    std::map<std::string, std::pair<size_t, std::string>> index = auto_entry_index_;
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (!is_rest_or_cli_interface(entries_[i].iface) || entries_[i].iface == CLI_OPTION_INTERFACE) continue;
      if (auto_entry_indices_.count(i)) continue;
      if (!index.emplace(entries_[i].annotation, std::make_pair(i, entries_[i].name)).second) throw host_error("rest: duplicate annotation @" + entries_[i].annotation);
    }
    for (const auto& route : table_.routes()) {
      if (index.count(route.annotation)) continue;
      for (const auto& e : entries_) if (e.annotation == route.annotation) throw host_error("rest: entry " + e.name + " must implement RestHandler(RestExchange) -> void");
      throw host_error("rest: route " + route.method + " " + route.path + " references missing annotation @" + route.annotation);
    }
    entry_index_ = std::move(index);
  }

  static Response response_of(const Exchange& x) {
    Response r; r.status = x.status; r.body = x.body;
    for (const auto& h : x.headers) r.headers.push_back({h.name, h.value});
    if (x.status == 204 || x.status == 304) r.body.clear();
    else if (!r.body.empty() && !x.has_response_header("content-type")) r.headers.insert(r.headers.begin(), {"Content-Type", "text/plain; charset=utf-8"});
    return r;
  }

  void write(const Response& r, bool head) {
    const std::string bytes = encode_response(r, head);
    options_.output->write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    options_.output->flush();
  }

  void pump() {
    for (;;) {
      Taken taken = take_request(*options_.input, options_.max_body_bytes);
      if (taken.kind == Taken::END) return;
      if (taken.kind == Taken::FATAL) { write(error_response(taken.fatal), false); return; }
      write(dispatch(taken.request), taken.request.method == "HEAD");
      if (stopped_) return;
    }
  }

  Registry* registry_;
  Options options_;
  State state_;
  RouteTable table_;
  const Program* program_ = nullptr;
  Context context_;
  std::vector<Program::Entry> entries_;
  std::map<std::string, std::pair<size_t, std::string>> auto_entry_index_;
  std::set<size_t> auto_entry_indices_;
  std::map<std::string, std::pair<size_t, std::string>> entry_index_;   // annotation -> (entry index, function name)
  std::map<std::string, std::pair<size_t, std::string>> cli_options_;
  std::string config_prefix_;
  bool bound_ = false, serving_ = false, stopped_ = false;
  bool auto_routes_materialized_ = false, in_controller_run_ = false;
};

// ---- registration ----------------------------------------------------------------------------------------------------
inline Exchange& exchange_of(void* self) { if (!self) throw host_error("invalid exchange"); return *static_cast<Exchange*>(self); }

/** Registers the 20 `rest` signatures in `registry` and returns the controller that owns serving. */
inline std::shared_ptr<RestController> register_rest(Registry& registry, Options options = {}) {
  auto controller = std::make_shared<RestController>(registry, std::move(options));
  // The closures keep the controller alive (registry copies share it); the controller only points back at the registry.
  auto c = controller;
  registry.set_rest_hooks(
      c,
      [c](const Program& program, const Context& ctx, const Registry& reg) {
        return c->before_body_run(program, ctx, reg);
      },
      [c]() { c->after_body_run(); });
  registry.override_std_function(
      "rest_config", {"string", "string"}, "void",
      [c](Args& a) { c->config(a[0].s, a[1].s); return Value::none(); },
      dry([](Args&) { return Value::none(); }));
  registry.override_std_function(
      "rest_route", {"string", "string", "string"}, "void",
      [c](Args& a) { c->route(a[0].s, a[1].s, a[2].s); return Value::none(); },
      dry([](Args&) { return Value::none(); }));
  registry.override_std_function(
      "rest_serve", {}, "void",
      [c](Args&) { c->serve(); return Value::none(); },
      dry([](Args&) { return Value::none(); }));
  registry.override_std_function(
      "rest_stop", {}, "void",
      [c](Args&) { c->stop(); return Value::none(); },
      dry([](Args&) { return Value::none(); }));
  registry.override_std_function(
      "rest_cli", {"StrList"}, "string",
      [c](Args& a) {
        const std::vector<std::string> argv = static_cast<stdlib::StrList*>(a[0].obj)->items;
        return Value::string(c->cli(argv));
      },
      dry([](Args&) { return Value::string(""); }));
  const std::string T = "RestExchange";
  auto method = [&](const char* name, std::vector<std::string> params, const char* returns, Method fn) {
    Method preview = fn;
    registry.override_std_method(T, name, std::move(params), returns, std::move(fn), dry(std::move(preview)));
  };
  method("method", {}, "string", [](void* s, Args&) { return Value::string(exchange_of(s).request.method); });
  method("path", {}, "string", [](void* s, Args&) { return Value::string(exchange_of(s).path); });
  method("query", {"string"}, "string", [](void* s, Args& a) { auto& q = exchange_of(s).query; auto it = q.find(a[0].s); return Value::string(it == q.end() ? "" : it->second); });
  method("has_query", {"string"}, "bool", [](void* s, Args& a) { return Value::boolean(exchange_of(s).query.count(a[0].s) > 0); });
  method("param", {"string"}, "string", [](void* s, Args& a) { auto& p = exchange_of(s).params; auto it = p.find(a[0].s); return Value::string(it == p.end() ? "" : it->second); });
  method("header", {"string"}, "string", [](void* s, Args& a) {
    std::string joined; for (const auto& v : exchange_of(s).request_header(a[0].s)) joined += (joined.empty() ? "" : ", ") + v;
    return Value::string(joined);
  });
  method("has_header", {"string"}, "bool", [](void* s, Args& a) { return Value::boolean(!exchange_of(s).request_header(a[0].s).empty()); });
  method("body", {}, "string", [](void* s, Args&) { return Value::string(exchange_of(s).request.body); });
  method("status", {"int"}, "void", [](void* s, Args& a) {
    Exchange& x = exchange_of(s);
    if (a[0].i < 200 || a[0].i > 599) throw host_error("rest: status " + std::to_string(a[0].i) + " out of range 200..599");
    x.status = static_cast<int>(a[0].i); return Value::none();
  });
  method("set_header", {"string", "string"}, "void", [](void* s, Args& a) {
    Exchange& x = exchange_of(s); const std::string& name = a[0].s;
    bool bad = name.empty() || a[1].s.find_first_of("\r\n") != std::string::npos;
    for (char ch : name) if (is_ws(ch) || ch == ':') bad = true;
    if (bad) throw host_error("rest: invalid response header");
    const std::string key = lower(name);
    if (key == "content-length" || key == "connection" || key == "transfer-encoding") throw host_error("rest: response header " + name + " is managed by the server");
    x.put_header(name, a[1].s); return Value::none();
  });
  method("send", {"string"}, "void", [](void* s, Args& a) { exchange_of(s).body = a[0].s; return Value::none(); });
  method("json", {"string"}, "void", [](void* s, Args& a) { Exchange& x = exchange_of(s); x.body = a[0].s; x.put_header("Content-Type", "application/json"); return Value::none(); });
  method("get", {"string"}, "string", [](void* s, Args& a) { auto& m = exchange_of(s).state->values; auto it = m.find(a[0].s); return Value::string(it == m.end() ? "" : it->second); });
  method("set", {"string", "string"}, "void", [](void* s, Args& a) { exchange_of(s).state->values[a[0].s] = a[1].s; return Value::none(); });
  method("increment", {"string"}, "int", [](void* s, Args& a) {
    int64_t& n = exchange_of(s).state->counts[a[0].s];
    n = from_twos_complement(static_cast<uint64_t>(n) + 1u);   // int64 wraparound like all lmcode ints
    return Value::integer(n);
  });
  return controller;
}

}  // namespace dslvm::stdlib::rest

#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
