#pragma once
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#include "dslvm.hpp"
#undef DSLVM_STD_DEFER
#define DSLVM_INVOCATION_FINISH_STD
#define DSLVM_STD_BUILDING
#endif
#include "result.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

namespace dslvm {
namespace invocation_detail {
inline std::string framed(const std::string& json){return "LMCODE_RETURN_BEGIN\n"+json+"\nLMCODE_RETURN_END\n";}
inline int exit_code(const std::string& phase){
  if(phase=="cli")return 2;
  if(phase=="load"||phase=="run")return 3;
  if(phase=="app")return 4;
  throw Error("lmcode invocation: invalid failure phase");
}
inline std::string failure(const std::string& phase,const Error& e,std::optional<size_t> part={},bool dry=false){
  (void)exit_code(phase);
  stdlib::result_detail::valid_utf8(e.code,"");stdlib::result_detail::valid_utf8(e.message,"");
  std::string out="{\"ok\":false,\"phase\":"+json_escape(phase)+",\"code\":"+json_escape(e.code)+",\"message\":"+json_escape(e.message);
  if(e.has_column&&(!e.has_location||!e.line||!e.column))throw Error("lmcode invocation: column requires a positive line");
  if(e.has_location){if(!e.line)throw Error("lmcode invocation: line must be positive");out+=",\"line\":"+std::to_string(e.line);if(e.has_column)out+=",\"column\":"+std::to_string(e.column);}
  if(part)out+=",\"part\":"+std::to_string(*part);
  if(dry)out+=",\"dry\":true";
  return out+"}";
}
inline void output_error()noexcept{try{std::cerr<<"lmcode invocation: cannot write envelope\n";}catch(...){}}
inline int emit(std::ostream& out,const std::string& json,int code)noexcept{
  try{const auto bytes=framed(json);out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));out.flush();if(out)return code;}
  catch(...){} // Never append a second envelope after partial output.
  output_error();return 4;
}
[[noreturn]] inline void metadata_error(const std::string& text){throw Error("CLI_USAGE","lmcode invocation: "+text);}
inline void keys(const Node& n,std::initializer_list<const char*> allowed){
  if(n.kind!=Kind::Object)metadata_error("object required");
  for(const auto&m:n.object){bool known=false;for(auto key:allowed)if(m.key==key)known=true;if(!known)metadata_error("unknown key "+m.key);}
}
}
// Errors before collector construction have no submitted slot association.
inline int failure_frame(std::ostream& out,const std::string& phase,const Error& error){return invocation_detail::emit(out,invocation_detail::failure(phase,error,{},error.has_invocation_dry&&error.invocation_dry),invocation_detail::exit_code(phase));}
inline int failure_frame(const std::string& phase,const Error& error){return failure_frame(std::cout,phase,error);}

struct InvocationRegistration {std::string file;size_t index;};
class InvocationResults {
  struct Slot {std::unique_ptr<ResultSink> sink;std::optional<stdlib::JsonNode> result;};
  struct Failure {std::string phase;Error error;std::optional<size_t> part;};
  bool builder_=false,dry_=false,writing_=false;
  std::vector<InvocationRegistration> registrations_;
  std::vector<Slot> slots_;
  std::optional<Failure> failure_;
  Slot& slot(size_t index){if(index>=slots_.size())throw Error("lmcode invocation: invalid slot index");return slots_[index];}
  InvocationResults(bool builder,bool dry,std::vector<InvocationRegistration> registrations,size_t count):builder_(builder),dry_(dry),registrations_(std::move(registrations)),slots_(count){}
public:
  InvocationResults(InvocationResults&&)=default;
  InvocationResults& operator=(InvocationResults&&)=default;
  InvocationResults(const InvocationResults&)=delete;
  InvocationResults& operator=(const InvocationResults&)=delete;
  static InvocationResults from_metadata(const std::string& raw,const std::vector<std::string>& parsed_files){
    using namespace invocation_detail;
    bool mode_known=false,dry_mode=false;
    try{
      auto root=parse(raw);if(root.kind!=Kind::Object)metadata_error("object required");
      const auto*version=member(root,"version"),*mode=member(root,"mode"),*dry=member(root,"dry"),*parts=member(root,"parts");
      if(!version||!integer(*version)||integer_value(*version)!=1)metadata_error("unsupported version");
      if(!mode||mode->kind!=Kind::String||(mode->text!="single"&&mode->text!="builder"))metadata_error("invalid mode");
      if(!dry||dry->kind!=Kind::Bool)metadata_error("dry must be boolean");
      mode_known=true;dry_mode=dry->boolean;
      keys(root,{"version","mode","dry","parts"});
      if(!parts||parts->kind!=Kind::Array||parts->array.empty()||(mode->text=="single"&&parts->array.size()!=1))metadata_error("invalid parts");
      std::vector<std::string> files;
      for(size_t i=0;i<parts->array.size();++i){
        const auto& part=parts->array[i];keys(part,{"index","file"});const auto*index=member(part,"index"),*file=member(part,"file");
        if(!index||!integer(*index)||integer_value(*index)<0||static_cast<uint64_t>(integer_value(*index))!=i)metadata_error("noncontiguous index");
        if(!file||file->kind!=Kind::String||file->text.find('\0')!=std::string::npos||!std::filesystem::u8path(file->text).is_absolute())metadata_error("file must be absolute");
        if(std::find(files.begin(),files.end(),file->text)!=files.end())metadata_error("duplicate file");
        files.push_back(file->text);
      }
      if(parsed_files.size()!=files.size())metadata_error("registration count mismatch");
      std::vector<InvocationRegistration> registrations;std::vector<bool> seen(files.size(),false);
      for(const auto&file:parsed_files){
        const auto found=std::find(files.begin(),files.end(),file);
        if(found==files.end())metadata_error("extra registration");
        const auto index=static_cast<size_t>(found-files.begin());if(seen[index])metadata_error("duplicate registration");seen[index]=true;registrations.push_back({file,index});
      }
      return InvocationResults(mode->text=="builder",dry->boolean,std::move(registrations),files.size());
    }catch(const Error& error){
      Error e=error;
      if(mode_known){e.has_invocation_dry=true;e.invocation_dry=dry_mode;throw e;}
      if(e.code=="CLI_USAGE")throw;
      metadata_error(e.message);
    }
  }
  static InvocationResults from_command(const std::vector<std::string>& parsed_files,const std::optional<std::string>& manifest){
    if(manifest){
      if(manifest->empty()||manifest->find('\0')!=std::string::npos||!std::filesystem::u8path(*manifest).is_absolute())invocation_detail::metadata_error("manifest path must be absolute");
      std::ifstream input(std::filesystem::u8path(*manifest),std::ios::binary);
      if(!input)invocation_detail::metadata_error("cannot read manifest");
      std::ostringstream raw;raw<<input.rdbuf();if(input.bad())invocation_detail::metadata_error("cannot read manifest");
      return from_metadata(raw.str(),parsed_files);
    }
    if(parsed_files.empty())invocation_detail::metadata_error("empty parts");
    std::vector<InvocationRegistration> registrations;
    for(size_t i=0;i<parsed_files.size();++i){if(parsed_files[i].empty()||parsed_files[i].find('\0')!=std::string::npos)invocation_detail::metadata_error("invalid command file");try{stdlib::result_detail::valid_utf8(parsed_files[i],"");}catch(const Error&){invocation_detail::metadata_error("file must be UTF-8");}registrations.push_back({parsed_files[i],i});}
    return InvocationResults(parsed_files.size()>1,false,std::move(registrations),parsed_files.size());
  }
  static InvocationResults from_command(const std::vector<std::string>& parsed_files,bool embedded_dry){
    auto result=from_command(parsed_files,std::nullopt);result.dry_=embedded_dry;return result;
  }
  static InvocationResults from_command(const std::vector<std::string>& parsed_files){
#ifdef _WIN32
    const wchar_t* value=_wgetenv(L"LMCODE_INVOCATION_FILE");
    if(!value)return from_command(parsed_files,std::nullopt);
    std::string manifest;
    try{const auto bytes=std::filesystem::path(value).u8string();manifest.assign(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
    catch(const std::exception&){invocation_detail::metadata_error("invalid manifest path encoding");}
    return from_command(parsed_files,manifest);
#else
    const char* value=std::getenv("LMCODE_INVOCATION_FILE");return from_command(parsed_files,value?std::optional<std::string>(value):std::nullopt);
#endif
  }
  const std::vector<InvocationRegistration>& registrations()const{return registrations_;}
  bool dry()const{return dry_;}
  ResultSink& select(size_t index,Registry& registry,const ResultContract& contract){
    if(writing_)throw Error("lmcode invocation: finalized");
    auto& s=slot(index);
    if(!s.sink){if(failure_)throw Error("lmcode invocation: cannot start work after failure");s.sink=std::make_unique<ResultSink>(contract);}
    if(s.sink->sealed())throw Error("lmcode return: result sink is sealed");
    if(s.sink->contract()!=contract)throw Error("lmcode invocation: result contract changed");
    registry.set_result_sink(s.sink.get());return *s.sink;
  }
  void complete(size_t index){if(writing_)throw Error("lmcode invocation: finalized");
    auto& s=slot(index);if(!s.sink)throw Error("lmcode invocation: slot was not selected");try{s.result=s.sink->seal();}catch(const Error&e){fail(index,"run",e);throw;} }
  void fail(std::optional<size_t> index,const std::string& phase,const Error& error){
    if(writing_)throw Error("lmcode invocation: finalized");
    if(failure_)return;
    if(index)(void)slot(*index);
    (void)invocation_detail::failure(phase,error); // validate before latching
    if(!failure_)failure_.emplace(Failure{phase,error,builder_?index:std::nullopt});
  }
  int write_frame(std::ostream& out)noexcept{
    if(writing_)return 4;
    try{
    if(!failure_)for(size_t i=0;i<slots_.size();++i)if(!slots_[i].result){fail(i,"run",Error("lmcode invocation: slot not completed"));break;}
    std::string payload;int code=0;
    if(failure_){payload=invocation_detail::failure(failure_->phase,failure_->error,failure_->part,dry_);code=invocation_detail::exit_code(failure_->phase);}
    else{
      payload="{\"ok\":true,\"result\":";
      if(builder_){payload+='[';for(size_t i=0;i<slots_.size();++i){if(i)payload+=',';payload+=stdlib::result_detail::encode(*slots_[i].result);}payload+=']';payload+=",\"parts\":"+std::to_string(slots_.size());}
      else payload+=stdlib::result_detail::encode(*slots_[0].result);
      payload+='}';
      if(dry_) payload.insert(payload.size()-1,",\"dry\":true");
    }
    writing_=true;return invocation_detail::emit(out,payload,code);
    }catch(...){writing_=true;invocation_detail::output_error();return 4;}
  }
};
} // namespace dslvm
#ifdef DSLVM_INVOCATION_FINISH_STD
#undef DSLVM_INVOCATION_FINISH_STD
#undef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
