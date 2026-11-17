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
#include "result_contract.hpp"
#include <ostream>

namespace dslvm {
class ResultSink {
  ResultContract contract_;
  stdlib::JsonNode root_;
  bool sealed_=false;
  bool written_=false;
  void writable() const { if(sealed_)throw Error("lmcode return: result sink is sealed"); }
  void field_key(const std::string& key) const {
    writable();
    if (root_.kind!=stdlib::JsonNode::Kind::Null && root_.kind!=stdlib::JsonNode::Kind::Object)
      throw Error("lmcode return: field assignment requires object or null");
    stdlib::result_detail::valid_utf8(key,"");
  }
  void put_owned(const std::string& key,stdlib::JsonNode value) {
    auto next=stdlib::result_detail::copy_node(root_);
    if(next.kind==stdlib::JsonNode::Kind::Null)next.kind=stdlib::JsonNode::Kind::Object;
    bool found=false;
    for(auto&m:next.object)if(m.key==key){m.value=std::make_unique<stdlib::JsonNode>(std::move(value));found=true;break;}
    if(!found)next.object.push_back({key,"",std::make_unique<stdlib::JsonNode>(std::move(value))});
    contract_.validate(next);root_=std::move(next);
  }
public:
  explicit ResultSink(ResultContract contract={}):contract_(std::move(contract)){}
  ResultSink(const ResultSink&)=delete;
  ResultSink& operator=(const ResultSink&)=delete;
  void reset() { writable();root_=stdlib::JsonNode{};written_=false; }
  bool written() const noexcept { return written_; }
  void set_null() { writable();stdlib::JsonNode n;contract_.validate(n);root_=std::move(n);written_=true; }
  void set(const Value& value) {
    writable();auto next=stdlib::result_detail::copy_value(value);
    contract_.validate_source(value);contract_.validate(next);root_=std::move(next);written_=true;
  }
  void put(const std::string& key,const Value& value) {
    field_key(key);auto next=stdlib::result_detail::copy_value(value,stdlib::result_detail::ptr("",key));
    contract_.validate_field_source(key,value);put_owned(key,std::move(next));written_=true;
  }
  void put_null(const std::string& key) { field_key(key);put_owned(key,stdlib::JsonNode{});written_=true; }
  std::string json() const { return stdlib::result_detail::encode(root_); }
  bool sealed()const{return sealed_;}
  const ResultContract& contract()const{return contract_;}
  stdlib::JsonNode seal(){writable();contract_.validate(root_);auto copy=stdlib::result_detail::copy_node(root_);sealed_=true;return copy;}
};
namespace stdlib {
inline Registry& register_results(Registry& registry,ResultSink*& configured) {
  for(const char* type:{"int","float","bool","string","Json","StrList","IntList","FloatList","StrMap","IntMap","FloatMap"}) {
    pure_std_function(registry, "lmcode_return_set",{type},"void",[&configured](Args& args){
      if(!configured) throw Error("lmcode return: no result sink");
      configured->set(args[0]);return Value::none();
    });
    pure_std_function(registry, "lmcode_return_put",{"string",type},"void",[&configured](Args& args){
      if(!configured) throw Error("lmcode return: no result sink");
      configured->put(args[0].s,args[1]);return Value::none();
    });
  }
  pure_std_function(registry, "lmcode_return_null",{},"void",[&configured](Args&){
    if(!configured) throw Error("lmcode return: no result sink");
    configured->set_null();return Value::none();
  });
  pure_std_function(registry, "lmcode_return_put_null",{"string"},"void",[&configured](Args& args){
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
