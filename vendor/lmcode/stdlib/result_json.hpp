#pragma once

// The result boundary owns values before the VM's per-run collection arena
// can be reset.  It deliberately does not use Json::raw: raw number spellings
// have to be checked before conversion and re-emitted with a round-tripping
// formatter.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#include "dslvm.hpp"
#undef DSLVM_STD_DEFER
#define DSLVM_STD_BUILDING
#define RESULT_JSON_FINISH_STD
#endif
#include "json.hpp"
#include "containers.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>

namespace dslvm::stdlib::result_detail {
inline constexpr int64_t kSafeInteger = 9007199254740991LL;

inline std::string ptr(const std::string& parent, const std::string& key) {
  std::string out = parent + "/";
  for (char c : key) { if (c == '~') out += "~0"; else if (c == '/') out += "~1"; else out += c; }
  return out;
}
[[noreturn]] inline void invalid(const std::string& path, const char* reason) {
  throw Error("lmcode return: invalid JSON value at " + path + ": " + reason);
}
inline void valid_utf8(const std::string& s, const std::string& path) {
  for (size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]); size_t n = 0; uint32_t cp = 0;
    if (c < 0x80) { ++i; continue; }
    if (c >= 0xc2 && c <= 0xdf) { n=2; cp=c&31; }
    else if (c >= 0xe0 && c <= 0xef) { n=3; cp=c&15; }
    else if (c >= 0xf0 && c <= 0xf4) { n=4; cp=c&7; }
    else invalid(path,"utf8");
    if (i+n>s.size()) invalid(path,"utf8");
    for (size_t j=1;j<n;++j) { unsigned char x=s[i+j]; if ((x&0xc0)!=0x80) invalid(path,"utf8"); cp=(cp<<6)|(x&63); }
    if ((n==3 && cp<0x800)||(n==4 && cp<0x10000)||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)) invalid(path,"utf8");
    i+=n;
  }
}
inline void number(double value, const std::string& path) {
  if (!std::isfinite(value)) invalid(path,"nonfinite");
  if (value == 0) return;
  if (std::trunc(value)==value && std::abs(value)>double(kSafeInteger)) invalid(path,"unsafe_integer");
}
// Examine the mathematical decimal value before binary64 can round it. The
// exponent bound below is derived from token length: exceeding it determines
// integral-vs-fractional classification without allocating exponent-sized text.
inline bool decimal_unsafe_integer(const std::string& raw, const std::string& path="") {
  if (raw.empty()) return false; // host-created numeric nodes need no raw token
  size_t at=0, fractional=0; std::string digits;
  if (raw[at]=='-') ++at;
  if (at==raw.size()) invalid(path,"handle");
  auto digit=[](char c){return c>='0'&&c<='9';};
  if (raw[at]=='0') digits+=raw[at++];
  else {
    if (raw[at]<'1'||raw[at]>'9') invalid(path,"handle");
    while(at<raw.size()&&digit(raw[at])) digits+=raw[at++];
  }
  if(at<raw.size()&&raw[at]=='.') {
    ++at;const size_t first=at;
    while(at<raw.size()&&digit(raw[at])) {digits+=raw[at++];++fractional;}
    if(at==first) invalid(path,"handle");
  }
  bool negative_exp=false, huge=false;size_t exponent=0;
  if(at<raw.size()&&(raw[at]=='e'||raw[at]=='E')) {
    ++at;
    if(at<raw.size()&&(raw[at]=='+'||raw[at]=='-')) negative_exp=raw[at++]=='-';
    const size_t first=at,bound=raw.size()+16;
    while(at<raw.size()&&digit(raw[at])) {
      if(!huge) {
        const size_t d=static_cast<size_t>(raw[at]-'0');
        if(exponent>bound/10||(exponent==bound/10&&d>bound%10)) huge=true;
        else exponent=exponent*10+d;
      }
      ++at;
    }
    if(at==first) invalid(path,"handle");
  }
  if(at!=raw.size()) invalid(path,"handle");
  const size_t first=digits.find_first_not_of('0');
  if(first==std::string::npos) return false;
  digits.erase(0,first);
  if(huge) return !negative_exp;
  size_t denominator=0,zeros=0;
  if(negative_exp) {
    if(exponent>digits.size()||fractional>digits.size()-exponent) return false;
    denominator=fractional+exponent;
  } else if(exponent>=fractional) zeros=exponent-fractional;
  else denominator=fractional-exponent;
  if(denominator) {
    if(denominator>=digits.size()) return false;
    for(size_t i=0;i<denominator;++i) if(digits[digits.size()-1-i]!='0') return false;
    digits.resize(digits.size()-denominator);
  }
  const std::string safe="9007199254740991";
  if(digits.size()>safe.size()||zeros>safe.size()-digits.size()) return true;
  digits.append(zeros,'0');
  return digits.size()==safe.size()&&digits>safe;
}
inline void raw_number(const JsonNode& n, const std::string& path) {
  if (n.kind == JsonNode::Kind::Int) {
    if ((!n.raw.empty() && decimal_unsafe_integer(n.raw,path)) || n.integer > kSafeInteger || n.integer < -kSafeInteger) invalid(path,"unsafe_integer");
    return;
  }
  number(n.floating,path);
  bool nonzero=false; for(char c:n.raw) { if(c=='e'||c=='E') break; if(c>='1'&&c<='9'){nonzero=true;break;} }
  if (nonzero && n.floating==0) invalid(path,"underflow");
  if (decimal_unsafe_integer(n.raw,path)) invalid(path,"unsafe_integer");
}
inline JsonNode copy_node_active(const JsonNode& n, const std::string& path, std::unordered_set<const JsonNode*>& active) {
  if (!active.insert(&n).second) invalid(path,"cycle");
  if(n.kind<JsonNode::Kind::Null||n.kind>JsonNode::Kind::Object) invalid(path,"handle");
  JsonNode out; out.kind=n.kind; out.boolean=n.boolean; out.integer=n.integer; out.floating=n.floating; out.text=n.text;
  if (n.kind==JsonNode::Kind::Int || n.kind==JsonNode::Kind::Float) raw_number(n,path);
  if (n.kind==JsonNode::Kind::String) valid_utf8(n.text,path);
  if (n.kind==JsonNode::Kind::Array) { for(size_t i=0;i<n.array.size();++i) out.array.push_back(copy_node_active(n.array[i],path+"/"+std::to_string(i),active)); }
  if (n.kind==JsonNode::Kind::Object) {
    std::unordered_set<std::string> keys;
    for(const auto& m:n.object) {
      valid_utf8(m.key,path);
      if (!m.value||!keys.insert(m.key).second) invalid(ptr(path,m.key),"handle");
      out.object.push_back({m.key,"",std::make_unique<JsonNode>(copy_node_active(*m.value,ptr(path,m.key),active))});
    }
  }
  active.erase(&n);
  return out;
}
inline JsonNode copy_node(const JsonNode& n, const std::string& path="") { std::unordered_set<const JsonNode*> active; return copy_node_active(n,path,active); }
inline JsonNode scalar_int(int64_t v,const std::string& path) { if(v>kSafeInteger||v<-kSafeInteger) invalid(path,"unsafe_integer"); JsonNode n; n.kind=JsonNode::Kind::Int;n.integer=v;return n; }
inline JsonNode scalar_float(double v,const std::string& path) { number(v,path); JsonNode n; n.kind=JsonNode::Kind::Float;n.floating=v;return n; }
inline JsonNode copy_value(const Value& v, const std::string& path="") {
  JsonNode n;
  switch(v.kind) {
    case Value::INT: return scalar_int(v.i,path);
    case Value::FLOAT: return scalar_float(v.f,path);
    case Value::BOOL: n.kind=JsonNode::Kind::Bool;n.boolean=v.b;return n;
    case Value::STR: valid_utf8(v.s,path);n.kind=JsonNode::Kind::String;n.text=v.s;return n;
    case Value::VOID: n.kind=JsonNode::Kind::Null;return n;
    case Value::OBJ: break;
  }
  if(!v.obj) invalid(path,"handle");
  const std::string& t=v.type;
  if(t=="Json") return copy_node(*static_cast<const JsonNode*>(v.obj),path);
  if(t=="StrList" || t=="IntList" || t=="FloatList") {
    n.kind=JsonNode::Kind::Array;
    if(t=="StrList") for(size_t i=0;i<static_cast<StrList*>(v.obj)->items.size();++i) n.array.push_back(copy_value(Value::string(static_cast<StrList*>(v.obj)->items[i]),path+"/"+std::to_string(i)));
    if(t=="IntList") for(size_t i=0;i<static_cast<IntList*>(v.obj)->items.size();++i) n.array.push_back(copy_value(Value::integer(static_cast<IntList*>(v.obj)->items[i]),path+"/"+std::to_string(i)));
    if(t=="FloatList") for(size_t i=0;i<static_cast<FloatList*>(v.obj)->items.size();++i) n.array.push_back(copy_value(Value::real(static_cast<FloatList*>(v.obj)->items[i]),path+"/"+std::to_string(i)));
    return n;
  }
  n.kind=JsonNode::Kind::Object;
  std::unordered_set<std::string> keys;
  const auto add=[&](const std::string& key,const Value& value) {
    valid_utf8(key,path);
    if(!keys.insert(key).second) invalid(ptr(path,key),"handle");
    n.object.push_back({key,"",std::make_unique<JsonNode>(copy_value(value,ptr(path,key)))});
  };
  if(t=="StrMap") for(const auto& p:static_cast<StrMap*>(v.obj)->entries()) add(p.first,Value::string(p.second));
  else if(t=="IntMap") for(const auto& p:static_cast<IntMap*>(v.obj)->items) add(p.first,Value::integer(p.second));
  else if(t=="FloatMap") for(const auto& p:static_cast<FloatMap*>(v.obj)->items) add(p.first,Value::real(p.second));
  else invalid(path,"handle");
  return n;
}
inline std::string encode(const JsonNode& n) {
  switch(n.kind) {
    case JsonNode::Kind::Null:return "null"; case JsonNode::Kind::Bool:return n.boolean?"true":"false";
    case JsonNode::Kind::Int:return std::to_string(n.integer);
    case JsonNode::Kind::Float:{ double v=n.floating;if(v==0)v=0;char b[64];auto r=std::to_chars(b,b+sizeof b,v,std::chars_format::general);if(r.ec!=std::errc())throw Error("lmcode return: invalid JSON value at /: nonfinite");return std::string(b,r.ptr); }
    case JsonNode::Kind::String:return json_detail::quoted(n.text);
    case JsonNode::Kind::Array:{std::string s="[";for(size_t i=0;i<n.array.size();++i){if(i)s+=',';s+=encode(n.array[i]);}return s+"]";}
    case JsonNode::Kind::Object:{std::string s="{";for(size_t i=0;i<n.object.size();++i){if(i)s+=',';s+=json_detail::quoted(n.object[i].key)+":"+encode(*n.object[i].value);}return s+"}";}
  } return "null";
}
#ifdef RESULT_JSON_FINISH_STD
#undef DSLVM_STD_BUILDING
#undef RESULT_JSON_FINISH_STD
} // namespace dslvm::stdlib::result_detail
#include "std.hpp"
#else
} // namespace dslvm::stdlib::result_detail
#endif
