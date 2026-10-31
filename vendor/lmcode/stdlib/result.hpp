#pragma once
// Invocation-owned results are deliberately separate from collection arenas.
// Hosts bind a sink once before the body, retain it across selected entry calls,
// and reset it only before the next logical program. Native code never prints
// implicitly: a successful host explicitly chooses when to frame its result.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#include "dslvm.hpp"
#undef DSLVM_STD_DEFER
#define DSLVM_RESULT_FINISH_STD
#define DSLVM_STD_BUILDING
#endif
#include "result_json.hpp"
#include <ostream>

namespace dslvm {
class ResultSink {
  stdlib::JsonNode root_;
  void put_owned(const std::string& key, stdlib::JsonNode value) {
    auto next=std::make_unique<stdlib::JsonNode>(std::move(value));
    if (root_.kind==stdlib::JsonNode::Kind::Null) {
      stdlib::JsonNode object; object.kind=stdlib::JsonNode::Kind::Object;
      object.object.push_back({key,"",std::move(next)});
      root_=std::move(object);
      return;
    }
    for (auto& member:root_.object) if(member.key==key) {
      member.value=std::move(next); // overwrite preserves insertion position
      return;
    }
    root_.object.push_back({key,"",std::move(next)});
  }
  void field_key(const std::string& key) const {
    if (root_.kind!=stdlib::JsonNode::Kind::Null && root_.kind!=stdlib::JsonNode::Kind::Object)
      throw Error("lmcode return: field assignment requires object or null");
    // An invalid key has no valid JSON Pointer segment. Report its parent.
    stdlib::result_detail::valid_utf8(key,"");
  }
public:
  void reset() { root_=stdlib::JsonNode{}; }
  void set_null() { reset(); }
  void set(const Value& value) { root_=stdlib::result_detail::copy_value(value); }
  void put(const std::string& key,const Value& value) {
    field_key(key);
    put_owned(key,stdlib::result_detail::copy_value(value,stdlib::result_detail::ptr("",key)));
  }
  void put_null(const std::string& key) { field_key(key);put_owned(key,stdlib::JsonNode{}); }
  std::string json() const { return stdlib::result_detail::encode(root_); }
};

inline std::string result_frame(const ResultSink& sink) {
  return "LMCODE_RETURN_BEGIN\n"+sink.json()+"\nLMCODE_RETURN_END\n";
}
// Call only AFTER body, selected entries, and application validation succeed.
// An exception or application failure must take the host's diagnostic path
// without calling this function. A host processing several programs frames each.
inline void write_result_frame(std::ostream& output,const ResultSink& sink) {
  const std::string frame=result_frame(sink);
  output.write(frame.data(),static_cast<std::streamsize>(frame.size()));
  output.flush(); // Surface a buffered write failure before reporting success.
  if(!output) throw Error("lmcode return: cannot write result frame");
}
namespace stdlib {
inline Registry& register_results(Registry& registry,ResultSink*& configured) {
  for(const char* type:{"int","float","bool","string","Json","StrList","IntList","FloatList","StrMap","IntMap","FloatMap"}) {
    registry.function("lmcode_return_set",{type},"void",[&configured](Args& args){
      if(!configured) throw Error("lmcode return: no result sink");
      configured->set(args[0]);return Value::none();
    });
    registry.function("lmcode_return_put",{"string",type},"void",[&configured](Args& args){
      if(!configured) throw Error("lmcode return: no result sink");
      configured->put(args[0].s,args[1]);return Value::none();
    });
  }
  registry.function("lmcode_return_null",{},"void",[&configured](Args&){
    if(!configured) throw Error("lmcode return: no result sink");
    configured->set_null();return Value::none();
  });
  registry.function("lmcode_return_put_null",{"string"},"void",[&configured](Args& args){
    if(!configured) throw Error("lmcode return: no result sink");
    configured->put_null(args[0].s);return Value::none();
  });
  return registry;
}
} // namespace stdlib
} // namespace dslvm
#ifdef DSLVM_RESULT_FINISH_STD
#undef DSLVM_RESULT_FINISH_STD
#undef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
