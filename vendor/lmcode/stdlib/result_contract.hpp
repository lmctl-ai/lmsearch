#pragma once
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#include "dslvm.hpp"
#undef DSLVM_STD_DEFER
#define DSLVM_CONTRACT_FINISH_STD
#define DSLVM_STD_BUILDING
#endif
#include "result_json.hpp"

namespace dslvm {
namespace invocation_detail {
using Node=stdlib::JsonNode;
using Kind=Node::Kind;
inline const Node* member(const Node& n,const std::string& key){for(const auto&m:n.object)if(m.key==key)return m.value.get();return nullptr;}
inline Node parse(const std::string& raw,bool validate_numbers=true){
  stdlib::json_detail::utf8(raw);
  if(raw.rfind("\xef\xbb\xbf",0)==0)throw Error("JSON BOM is not allowed");
  stdlib::json_detail::Parser parser(raw);auto n=parser.val();parser.ws();
  if(parser.p!=raw.size())throw Error("trailing JSON input");
  return validate_numbers?stdlib::result_detail::copy_node(n):std::move(n);
}
inline const char* kind(const Node& n){switch(n.kind){case Kind::Null:return "null";case Kind::Bool:return "bool";case Kind::Int:return "int";case Kind::Float:return "float";case Kind::String:return "string";case Kind::Array:return "array";case Kind::Object:return "object";}return "handle";}
[[noreturn]] inline void mismatch(const std::string& path,const std::string& type,const std::string& got){throw Error("lmcode return: result type mismatch at "+path+": expected "+type+"; got "+got);}
inline bool integer(const Node& n){const auto safe=stdlib::result_detail::kSafeInteger;return(n.kind==Kind::Int&&n.integer>=-safe&&n.integer<=safe)||(n.kind==Kind::Float&&std::isfinite(n.floating)&&std::abs(n.floating)<=double(safe)&&std::trunc(n.floating)==n.floating);}
inline int64_t integer_value(const Node& n){return n.kind==Kind::Int?n.integer:static_cast<int64_t>(n.floating);}
inline bool primitive(const Node& n,const std::string&t){return(t=="int"&&integer(n))||(t=="float"&&(n.kind==Kind::Int||n.kind==Kind::Float))||(t=="bool"&&n.kind==Kind::Bool)||(t=="string"&&n.kind==Kind::String);}
}
// Host-owned schema metadata; never inferred from an invocation manifest.
class ResultContract {
  std::string type_;
  bool nullable_=false;
public:
  ResultContract()=default;
  const std::string& type()const{return type_;}
  bool nullable()const{return nullable_;}
  bool operator==(const ResultContract&o)const{return type_==o.type_&&nullable_==o.nullable_;}
  bool operator!=(const ResultContract&o)const{return!(*this==o);}
  static ResultContract from_schema(const std::string& schema){
    using namespace invocation_detail;
    try{
      auto root=parse(schema,false);if(root.kind!=Kind::Object)throw Error("schema must be an object");
      const auto* version=member(root,"dsl.schema");if(!version||!integer(*version)||integer_value(*version)!=2)throw Error("requires schema 2");
      const auto* result=member(root,"result");ResultContract c;if(!result)return c;
      if(result->kind!=Kind::Object)throw Error("result must be an object");
      for(const auto&m:result->object)if(m.key!="type"&&m.key!="nullable")throw Error("unknown key "+m.key);
      const auto*t=member(*result,"type"),*nullable=member(*result,"nullable");
      if(!t||t->kind!=Kind::String)throw Error("type must be a string");
      c.type_=t->text;
      bool known=false;for(const char* type:{"null","bool","int","float","string","Json","StrList","IntList","FloatList","StrMap","IntMap","FloatMap"})if(c.type_==type)known=true;
      if(!known)throw Error("unknown type "+c.type_);
      if(nullable){if(c.type_=="null")throw Error("type null forbids nullable");if(nullable->kind!=Kind::Bool)throw Error("nullable must be bool");c.nullable_=nullable->boolean;}
      return c;
    }catch(const Error&e){throw Error("LOAD_BAD_SCHEMA","lmcode result contract: "+e.message);}
  }
  void validate(const stdlib::JsonNode& n,const std::string& path="")const{
    using namespace invocation_detail;
    if(type_.empty())return;
    if(n.kind==Kind::Null){if(nullable_||type_=="null")return;invocation_detail::mismatch(path,type_,"null");}
    if(type_=="Json")return; // any non-null JSON shape; nested null remains valid
    if(primitive(n,type_))return;
    const bool list=type_=="StrList"||type_=="IntList"||type_=="FloatList";
    const bool map=type_=="StrMap"||type_=="IntMap"||type_=="FloatMap";
    if((list&&n.kind==Kind::Array)||(map&&n.kind==Kind::Object)){
      const std::string element=type_.rfind("Str",0)==0?"string":type_.rfind("Int",0)==0?"int":"float";
      if(list)for(size_t i=0;i<n.array.size();++i){const auto&value=n.array[i];if(!primitive(value,element))invocation_detail::mismatch(path+"/"+std::to_string(i),element,kind(value));}
      if(map)for(const auto&m:n.object)if(!primitive(*m.value,element))invocation_detail::mismatch(stdlib::result_detail::ptr(path,m.key),element,kind(*m.value));
      return;
    }
    invocation_detail::mismatch(path,type_,kind(n));
  }
  // Preserve known collection distinctions even when an empty value has no
  // elements to inspect. Json handles deliberately defer to their actual shape.
  static std::string source_type(const Value& value){
    switch(value.kind){case Value::INT:return "int";case Value::FLOAT:return "float";case Value::BOOL:return "bool";case Value::STR:return "string";case Value::VOID:return "null";case Value::DRY_NULL:return value.is_dry_null()?"null":"dry null";case Value::OBJ:return value.type;}
    return "handle";
  }
  static bool assignable(const std::string& actual,const std::string& expected){
    return actual==expected||(actual=="int"&&expected=="float")||(actual=="IntList"&&expected=="FloatList")||(actual=="IntMap"&&expected=="FloatMap");
  }
  void validate_source(const Value& v)const{
    if(type_.empty()||type_=="Json"||v.kind==Value::VOID||v.is_dry_null()||(v.kind==Value::OBJ&&v.type=="Json"))return;
    const auto actual=source_type(v);
    if(!assignable(actual,type_))invocation_detail::mismatch("",type_,actual);
  }
  void validate_field_source(const std::string& key,const Value& v)const{
    if(type_.empty()||type_=="Json"||(v.kind==Value::OBJ&&v.type=="Json"))return;
    std::string element;
    if(type_=="IntMap")element="int";
    else if(type_=="FloatMap")element="float";
    else if(type_=="StrMap")element="string";
    else return; // the constructed object fails root validation below
    const auto actual=source_type(v);
    if(!assignable(actual,element))invocation_detail::mismatch(stdlib::result_detail::ptr("",key),element,actual);
  }
};
} // namespace dslvm
#ifdef DSLVM_CONTRACT_FINISH_STD
#undef DSLVM_CONTRACT_FINISH_STD
#undef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
