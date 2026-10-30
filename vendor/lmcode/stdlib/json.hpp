#pragma once
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
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_BUILDING
#include "string.hpp"
#undef DSLVM_STD_BUILDING
#else
#include "string.hpp"
#endif

namespace dslvm::stdlib {
struct JsonNode {
  enum class Kind {
    Null,
    Bool,
    Int,
    Float,
    String,
    Array,
    Object
  } kind = Kind::Null;
  bool boolean = false;
  int64_t integer = 0;
  double floating = 0;
  std::string text, raw;
  std::vector<JsonNode> array;
  struct Member {
    std::string key;
    std::string raw_key;
    std::unique_ptr<JsonNode> value;
  };
  std::vector<Member> object;
};
struct Json {
  JsonNode node;
};
namespace json_detail {
[[noreturn]] inline void fail(const std::string &r, size_t p) {
  throw Error("json_parse: " + r + " at byte " + std::to_string(p));
}
inline void append_utf8(std::string &o, uint32_t c) {
  if (c <= 127)
    o.push_back(char(c));
  else if (c <= 2047) {
    o.push_back(char(192 | (c >> 6)));
    o.push_back(char(128 | (c & 63)));
  } else if (c <= 65535) {
    o.push_back(char(224 | (c >> 12)));
    o.push_back(char(128 | ((c >> 6) & 63)));
    o.push_back(char(128 | (c & 63)));
  } else {
    o.push_back(char(240 | (c >> 18)));
    o.push_back(char(128 | ((c >> 12) & 63)));
    o.push_back(char(128 | ((c >> 6) & 63)));
    o.push_back(char(128 | (c & 63)));
  }
}
inline void utf8(const std::string &s, size_t b = 0) {
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t n = 0;
    uint32_t cp = 0;
    if (c < 128) {
      ++i;
      continue;
    }
    if (c >= 194 && c <= 223) {
      n = 2;
      cp = c & 31;
    } else if (c >= 224 && c <= 239) {
      n = 3;
      cp = c & 15;
    } else if (c >= 240 && c <= 244) {
      n = 4;
      cp = c & 7;
    } else
      fail("invalid UTF-8", b + i);
    if (i + n > s.size())
      fail("invalid UTF-8", b + i);
    for (size_t j = 1; j < n; j++)
      if ((s[i + j] & 192) != 128)
        fail("invalid UTF-8", b + i);
    for (size_t j = 1; j < n; j++)
      cp = (cp << 6) | (s[i + j] & 63);
    if ((n == 3 && cp < 2048) || (n == 4 && cp < 65536) || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      fail("invalid UTF-8", b + i);
    i += n;
  }
}
inline bool parse_float(const std::string &s, double &value) {
  static const dslvm::stdlib::detail::FloatLocale locale;
  if (!locale.value)
    return false;
  char *end = nullptr;
#if defined(_WIN32)
  value = _strtod_l(s.c_str(), &end, locale.value);
#else
  value = strtod_l(s.c_str(), &end, locale.value);
#endif
  return end == s.c_str() + s.size() && std::isfinite(value);
}
struct Parser {
  const std::string &s;
  size_t p = 0;
  explicit Parser(const std::string &x) : s(x) {}
  void ws() {
    while (p < s.size() &&
           (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n'))
      p++;
  }
  bool take(char c) {
    if (p < s.size() && s[p] == c) {
      p++;
      return true;
    }
    return false;
  }
  void lit(const char *w) {
    size_t n = strlen(w);
    if (s.compare(p, n, w))
      fail("invalid literal", p);
    p += n;
  }
  std::string str() {
    if (!take('"'))
      fail("expected string", p);
    std::string o;
    while (p < s.size()) {
      unsigned char c = s[p++];
      if (c == '"')
        return o;
      if (c < 32)
        fail("control byte in string", p - 1);
      if (c != '\\') {
        o.push_back(char(c));
        continue;
      }
      if (p >= s.size())
        fail("unterminated escape", p);
      char e = s[p++];
      switch (e) {
      case '"':
      case '\\':
      case '/':
        o.push_back(e);
        break;
      case 'b':
        o += '\b';
        break;
      case 'f':
        o += '\f';
        break;
      case 'n':
        o += '\n';
        break;
      case 'r':
        o += '\r';
        break;
      case 't':
        o += '\t';
        break;
      case 'u': {
        auto hex = [&](size_t at) {
          uint32_t c = 0;
          for (int i = 0; i < 4; i++) {
            char h = s[at + i];
            int v = h >= '0' && h <= '9'   ? h - '0'
                    : h >= 'a' && h <= 'f' ? h - 'a' + 10
                    : h >= 'A' && h <= 'F' ? h - 'A' + 10
                                           : -1;
            if (v < 0)
              fail("invalid Unicode escape", at + i);
            c = (c << 4) | v;
          }
          return c;
        };
        if (p + 4 > s.size())
          fail("invalid Unicode escape", p);
        uint32_t cp = hex(p);
        p += 4;
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (p + 6 > s.size() || s[p] != '\\' || s[p + 1] != 'u')
            fail("unpaired high surrogate", p);
          uint32_t lo = hex(p + 2);
          if (lo < 0xdc00 || lo > 0xdfff)
            fail("unpaired high surrogate", p);
          p += 6;
          cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
        } else if (cp >= 0xdc00 && cp <= 0xdfff)
          fail("unpaired low surrogate", p - 4);
        append_utf8(o, cp);
        break;
      }
      default:
        fail("invalid escape", p - 1);
      }
    }
    fail("unterminated string", p);
    return {};
  }
  void num(JsonNode &o) {
    size_t st = p;
    if (s[p] == '-')
      p++;
    if (p >= s.size() || s[p] < '0' || s[p] > '9')
      fail("invalid number", p);
    if (s[p] == '0')
      p++;
    else
      while (p < s.size() && s[p] >= '0' && s[p] <= '9')
        p++;
    bool fl = false;
    if (p < s.size() && s[p] == '.') {
      fl = true;
      p++;
      if (p >= s.size() || s[p] < '0' || s[p] > '9')
        fail("invalid number", p);
      while (p < s.size() && s[p] >= '0' && s[p] <= '9')
        p++;
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
      fl = true;
      p++;
      if (p < s.size() && (s[p] == '+' || s[p] == '-'))
        p++;
      if (p >= s.size() || s[p] < '0' || s[p] > '9')
        fail("invalid number", p);
      while (p < s.size() && s[p] >= '0' && s[p] <= '9')
        p++;
    }
    o.raw = s.substr(st, p - st);
    if (!fl) {
      auto [e, ec] =
          std::from_chars(o.raw.data(), o.raw.data() + o.raw.size(), o.integer);
      if (ec != std::errc())
        fail("integer out of range", st);
      o.kind = JsonNode::Kind::Int;
    } else {
      if (!json_detail::parse_float(o.raw, o.floating))
        fail("nonfinite number", st);
      o.kind = JsonNode::Kind::Float;
    }
  }
  JsonNode val(size_t d = 0) {
    ws();
    if (p >= s.size())
      fail("expected value", p);
    size_t st = p;
    JsonNode o;
    if (s[p] == 'n') {
      lit("null");
      o.kind = JsonNode::Kind::Null;
    } else if (s[p] == 't') {
      lit("true");
      o.kind = JsonNode::Kind::Bool;
      o.boolean = true;
    } else if (s[p] == 'f') {
      lit("false");
      o.kind = JsonNode::Kind::Bool;
    } else if (s[p] == '"') {
      o.kind = JsonNode::Kind::String;
      o.text = str();
      utf8(o.text, st);
    } else if (s[p] == '[') {
      if (++d > 256)
        fail("maximum depth exceeded", p);
      o.kind = JsonNode::Kind::Array;
      p++;
      ws();
      if (!take(']'))
        for (;;) {
          o.array.push_back(val(d));
          ws();
          if (take(']'))
            break;
          if (!take(','))
            fail("expected comma", p);
          ws();
        }
    } else if (s[p] == '{') {
      if (++d > 256)
        fail("maximum depth exceeded", p);
      o.kind = JsonNode::Kind::Object;
      p++;
      ws();
      std::unordered_set<std::string> keys;
      if (!take('}'))
        for (;;) {
          ws();
          size_t kp = p;
          if (p >= s.size() || s[p] != '"')
            fail("object key must be a string", p);
          auto k = str();
          utf8(k, kp);
          if (!keys.insert(k).second)
            fail("duplicate object key", kp);
          const size_t key_end = p;
          ws();
          if (!take(':'))
            fail("expected colon", p);
          const std::string raw_key = s.substr(kp, key_end - kp);
          o.object.push_back(
              {std::move(k), raw_key, std::make_unique<JsonNode>(val(d))});
          ws();
          if (take('}'))
            break;
          if (!take(','))
            fail("expected comma", p);
          ws();
        }
    } else if (s[p] == '-' || (s[p] >= '0' && s[p] <= '9'))
      num(o);
    else
      fail("unexpected value", p);
    if (o.kind == JsonNode::Kind::String)
      o.raw = s.substr(st, p - st);
    return o;
  }
};
inline std::string quoted(const std::string &s) {
  std::string o = "\"";
  static const char *d = "0123456789abcdef";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += char(c);
    } else if (c == '\b')
      o += "\\b";
    else if (c == '\f')
      o += "\\f";
    else if (c == '\n')
      o += "\\n";
    else if (c == '\r')
      o += "\\r";
    else if (c == '\t')
      o += "\\t";
    else if (c < 32) {
      o += "\\u00";
      o += d[c >> 4];
      o += d[c & 15];
    } else
      o += char(c);
  }
  return o + '"';
}
inline std::string emit(const JsonNode &n) {
  switch (n.kind) {
  case JsonNode::Kind::Null:
    return "null";
  case JsonNode::Kind::Bool:
    return n.boolean ? "true" : "false";
  case JsonNode::Kind::Int:
  case JsonNode::Kind::Float:
    return n.raw;
  case JsonNode::Kind::String:
    // Preserve the source spelling, including escapes, in normalized output.
    return n.raw;
  case JsonNode::Kind::Array: {
    std::string o = "[";
    for (size_t i = 0; i < n.array.size(); i++) {
      if (i)
        o += ',';
      o += emit(n.array[i]);
    }
    return o + ']';
  }
  case JsonNode::Kind::Object: {
    std::string o = "{";
    for (size_t i = 0; i < n.object.size(); i++) {
      if (i)
        o += ',';
      o += n.object[i].raw_key + ":" + emit(*n.object[i].value);
    }
    return o + '}';
  }
  }
  return {};
}

struct TemplateNode {
  std::string text, path, item, separator;
  std::vector<TemplateNode> body, otherwise;
  bool conditional = false;
};
inline std::vector<TemplateNode>
parse_template(const std::string &s, size_t &pos, const std::string &mode,
               std::string &separator, unsigned depth = 0) {
  if (depth > 256)
    throw Error("template_render: maximum nesting exceeded");
  static const std::regex head(
      "^<([A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*|\\[[0-9]+\\])*)(?:"
      ":\\{([A-Za-z_][A-Za-z0-9_]*)\\||>)");
  static const std::regex close(
      "^\\};[ ]*separator[ ]*=[ ]*(\"(?:[^\"\\\\]|\\\\.)*\")>");
  std::vector<TemplateNode> nodes;
  std::string literal;
  auto flush = [&]() {
    if (!literal.empty()) {
      TemplateNode n;
      n.text = std::move(literal);
      nodes.push_back(std::move(n));
      literal.clear();
    }
  };
  while (pos < s.size()) {
    auto rest = s.substr(pos);
    if (mode == "loop" && rest.rfind("}>", 0) == 0) {
      flush();
      pos += 2;
      return nodes;
    }
    if (mode == "loop" && rest.rfind("};", 0) == 0) {
      std::smatch m;
      if (!std::regex_search(rest, m, close))
        throw Error("template_render: invalid loop separator");
      auto raw = m[1].str();
      try {
        utf8(raw);
        Parser p(raw);
        separator = p.val().text;
      } catch (const Error &) {
        throw Error("template_render: invalid loop separator");
      }
      flush();
      pos += size_t(m.length());
      return nodes;
    }
    if (rest.rfind("<<", 0) == 0) {
      literal += '<';
      pos += 2;
      continue;
    }

    if (rest.rfind("<else>", 0) == 0 || rest.rfind("<endif>", 0) == 0) {
      auto tag = rest.rfind("<else>", 0) == 0 ? "else" : "endif";
      if ((mode != "if" && mode != "else") ||
          (mode == "else" && std::string(tag) == "else"))
        throw Error("template_render: unexpected conditional delimiter");
      flush();
      pos += std::string(tag) == "else" ? 6 : 7;
      separator = tag;
      return nodes;
    }
    static const std::regex condition(
        "^<if\\(([A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*|\\[[0-9]+"
        "\\])*)\\)>");
    std::smatch cm;
    if (std::regex_search(rest, cm, condition)) {
      flush();
      pos += size_t(cm.length());
      TemplateNode n;
      n.path = cm[1].str();
      n.conditional = true;
      std::string end;
      n.body = parse_template(s, pos, "if", end, depth + 1);
      if (end == "else")
        n.otherwise = parse_template(s, pos, "else", end, depth + 1);
      nodes.push_back(std::move(n));
      continue;
    }
    std::smatch m;
    if (!std::regex_search(rest, m, head)) {
      literal += s[pos++];
      continue;
    }
    flush();
    pos += size_t(m.length());
    TemplateNode n;
    n.path = m[1].str();
    n.item = m[2].str();
    if (!n.item.empty()) {
      if (n.item == "index")
        throw Error("template_render: index is reserved");
      n.body = parse_template(s, pos, "loop", n.separator, depth + 1);
    }
    nodes.push_back(std::move(n));
  }
  if (!mode.empty())
    throw Error("template_render: unterminated block");
  flush();
  return nodes;
}
inline const JsonNode &
template_lookup(const JsonNode &context,
                const std::map<std::string, const JsonNode *> &locals,
                const std::string &path) {
  static const std::regex part("[A-Za-z_][A-Za-z0-9_]*|\\[[0-9]+\\]");
  const JsonNode *n = &context;
  bool first = true;
  for (auto it = std::sregex_iterator(path.begin(), path.end(), part);
       it != std::sregex_iterator(); ++it) {
    auto key = it->str();
    if (first) {
      first = false;
      auto local = locals.find(key);
      if (local != locals.end()) {
        n = local->second;
        continue;
      }
    }
    const JsonNode *next = nullptr;
    if (key[0] == '[') {
      auto digits = key.substr(1, key.size() - 2);
      size_t i = 0;
      auto parsed =
          std::from_chars(digits.data(), digits.data() + digits.size(), i);
      if (n->kind == JsonNode::Kind::Array && parsed.ec == std::errc{} &&
          std::to_string(i) == digits && i < n->array.size())
        next = &n->array[i];
    } else if (n->kind == JsonNode::Kind::Object)
      for (auto &m : n->object)
        if (m.key == key) {
          next = m.value.get();
          break;
        }
    if (!next)
      throw Error("template_render: missing variable " + path);
    n = next;
  }
  return *n;
}
inline void evaluate_template(
    const std::vector<TemplateNode> &nodes, const JsonNode &context,
    const std::map<std::string, const JsonNode *> &locals, std::string &out) {
  for (auto &n : nodes) {
    if (n.path.empty()) {
      out += n.text;
      continue;
    }
    auto &v = template_lookup(context, locals, n.path);
    if (n.conditional) {
      if (v.kind != JsonNode::Kind::Bool)
        throw Error("template_render: condition requires a bool: " + n.path);
      evaluate_template(v.boolean ? n.body : n.otherwise, context, locals, out);
      continue;
    }
    if (n.item.empty()) {
      out += v.kind == JsonNode::Kind::String ? v.text : emit(v);
      continue;
    }
    if (v.kind != JsonNode::Kind::Array)
      throw Error("template_render: loop requires a list: " + n.path);
    auto scope = locals;
    for (size_t i = 0; i < v.array.size(); ++i) {
      if (i)
        out += n.separator;
      JsonNode index;
      index.kind = JsonNode::Kind::Int;
      index.raw = std::to_string(i);
      index.integer = int64_t(i);
      scope[n.item] = &v.array[i];
      scope["index"] = &index;
      evaluate_template(n.body, context, scope, out);
    }
  }
}
inline std::string render_template(const std::string &source,
                                   const JsonNode &context) {
  size_t pos = 0;
  std::string separator, out;
  auto nodes = parse_template(source, pos, "", separator);
  evaluate_template(nodes, context, {}, out);
  return out;
}
inline const char *kind(const JsonNode &n) {
  switch (n.kind) {
  case JsonNode::Kind::Null:
    return "null";
  case JsonNode::Kind::Bool:
    return "bool";
  case JsonNode::Kind::Int:
    return "int";
  case JsonNode::Kind::Float:
    return "float";
  case JsonNode::Kind::String:
    return "string";
  case JsonNode::Kind::Array:
    return "array";
  case JsonNode::Kind::Object:
    return "object";
  }
  return "null";
}
} // namespace json_detail
inline Registry &register_json(Registry &r, Collections &arena) {
  r.function("json_parse", {"string"}, "Json", [&arena](Args &a) {
    json_detail::utf8(a[0].s);
    if (a[0].s.rfind("\xEF\xBB\xBF", 0) == 0)
      json_detail::fail("BOM is not allowed", 0);
    json_detail::Parser p(a[0].s);
    JsonNode parsed = p.val();
    p.ws();
    if (p.p != a[0].s.size())
      json_detail::fail("trailing input", p.p);
    Json *root = arena.make_object<Json>("Json");
    root->node = std::move(parsed);
    return Value::object("Json", &root->node);
  });
  r.function("template_render", {"string", "Json"}, "string", [](Args &a) {
    auto &context = *static_cast<JsonNode *>(a[1].obj);
    if (context.kind != JsonNode::Kind::Object)
      throw Error("template_render: context must be an object");
    return Value::string(json_detail::render_template(a[0].s, context));
  });
#define JSON_METHOD(name, params, ret, body)                                   \
  r.method("Json", name, params, ret, [](void *p, Args &a) {                   \
    (void)a;                                                                   \
    body                                                                       \
  })
  JSON_METHOD("kind", {}, "string", {
    return Value::string(json_detail::kind(*static_cast<JsonNode *>(p)));
  });
  JSON_METHOD("len", {}, "int", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind == JsonNode::Kind::Array)
      return Value::integer(n.array.size());
    if (n.kind == JsonNode::Kind::Object)
      return Value::integer(n.object.size());
    throw Error("Json.len: expected array or object");
  });
  JSON_METHOD("at", {"int"}, "Json", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Array)
      throw Error("Json.at: expected array");
    if (a[0].i < 0 || uint64_t(a[0].i) >= n.array.size())
      throw Error("Json.at: index out of range: " + std::to_string(a[0].i));
    return Value::object("Json", &n.array[size_t(a[0].i)]);
  });
  JSON_METHOD("get", {"string"}, "Json", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Object)
      throw Error("Json.get: expected object");
    for (auto &x : n.object)
      if (x.key == a[0].s)
        return Value::object("Json", x.value.get());
    throw Error("Json.get: missing key");
  });
  JSON_METHOD("has", {"string"}, "bool", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Object)
      throw Error("Json.has: expected object");
    for (auto &x : n.object)
      if (x.key == a[0].s)
        return Value::boolean(true);
    return Value::boolean(false);
  });
  r.method("Json", "keys", {}, "StrList", [&arena](void *p, Args &) {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Object)
      throw Error("Json.keys: expected object");
    std::vector<std::string> k;
    for (auto &x : n.object)
      k.push_back(x.key);
    return Value::object("StrList", arena.make_str_list(std::move(k)));
  });
  JSON_METHOD("as_string", {}, "string", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::String)
      throw Error("Json.as_string: expected string");
    return Value::string(n.text);
  });
  JSON_METHOD("as_int", {}, "int", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Int)
      throw Error("Json.as_int: expected int");
    return Value::integer(n.integer);
  });
  JSON_METHOD("as_float", {}, "float", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind == JsonNode::Kind::Float)
      return Value::real(n.floating);
    if (n.kind == JsonNode::Kind::Int)
      return Value::real(double(n.integer));
    throw Error("Json.as_float: expected number");
  });
  JSON_METHOD("as_bool", {}, "bool", {
    auto &n = *static_cast<JsonNode *>(p);
    if (n.kind != JsonNode::Kind::Bool)
      throw Error("Json.as_bool: expected bool");
    return Value::boolean(n.boolean);
  });
  JSON_METHOD("to_json", {}, "string", {
    return Value::string(json_detail::emit(*static_cast<JsonNode *>(p)));
  });
#undef JSON_METHOD
  return r;
}
} // namespace dslvm::stdlib
#ifndef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
