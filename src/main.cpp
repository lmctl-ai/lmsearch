// SPDX-License-Identifier: MIT
#include "lmsearch.hpp"
#include "lmsearch_schema.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <fcntl.h>
#include <io.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifndef LMSEARCH_VERSION
#define LMSEARCH_VERSION "0.1.0"
#endif

namespace {

std::string read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw dslvm::Error("DSL_IO_ERROR", "cannot read DSLB file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw dslvm::Error("DSL_IO_ERROR", "cannot read input file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Minimal JSON string extractor for {"program": "..."} in --input-json / --input-file
std::string extract_json_string_field(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    std::string out;
    for (size_t i = pos + 1; i < json.size(); ++i) {
        char c = json[i];
        if (c == '\\' && i + 1 < json.size()) {
            out.push_back(json[++i]);
        } else if (c == '"') {
            return out;
        } else {
            out.push_back(c);
        }
    }
    return "";
}

std::string compact_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            out.push_back(c);
            if (c == '\\' && i + 1 < s.size()) {
                out.push_back(s[++i]);
            } else if (c == '"') {
                in_str = false;
            }
        } else {
            if (c == '"') {
                in_str = true;
                out.push_back(c);
            } else if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                out.push_back(c);
            }
        }
    }
    return out;
}

int run_bytecode(lmsearch::Host& host, const std::string& dsl_path, bool dryrun, bool readonly) {
    if (readonly) {
        return dslvm::report_error(
            "policy",
            dslvm::Error("POLICY_READONLY", "run is a write operation and is refused in readonly mode"));
    }

    std::string bytes;
    try {
        bytes = read_binary_file(dsl_path);
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("io", e);
    }

    try {
        auto program = dslvm::Program::load(bytes, lmsearch::schema, host.registry);
        if (dryrun) {
            std::ostringstream out;
            out << "{\"dryrun\": true, \"program\": "
                << dslvm::json_escape(dsl_path)
                << ", \"entries\": " << program.entries().size() << "}";
            return dslvm::report_ok(out.str());
        }

        host.bytecode_bytes = bytes;
        host.schema_bytes = lmsearch::schema;

        dslvm::Context ctx;
        program.run(ctx, host.registry);

        // Execute post-body @audit rules against all evaluated IndexMetrics
        const auto& entries = program.entries();
        for (uint32_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (entry.annotation == "audit" &&
                entry.iface == "interface AuditRule(IndexMetrics) -> void") {
                for (const auto& m : host.evaluated_metrics) {
                    program.call(
                        i,
                        ctx,
                        host.registry,
                        {dslvm::Value::object("IndexMetrics", m)});
                    host.entries_called += 1;
                }
            }
        }
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("dsl", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("dsl", dslvm::Error("DSL_HOST_EXCEPTION", e.what()));
    }

    if (std::getenv("LMCODE_INVOCATION_FILE") != nullptr) {
        std::cout << "LMCODE_RETURN_BEGIN\n"
                  << "{\"ok\":true,\"result\":" << compact_json(host.build_result_json()) << "}\n"
                  << "LMCODE_RETURN_END\n";
        return 0;
    }
    return dslvm::report_ok(host.build_result_json());
}

struct CliFilterConfig {
    std::string dsl_path;
    std::string config_rule;
    std::string filter_rule;
    std::string rerank_rule;
    std::string file_filter_rule;
    std::string ignore_rule;
    std::string embed_rule;

    std::string ext_filter;
    std::string path_prefix;
    std::string path_contains;
    std::string exclude_path;
    std::string text_contains;
    std::map<std::string, std::string> required_fields;
    int64_t min_size = -1;
    int64_t max_size = -1;
    bool has_min_score = false;
    double min_score = -1e30;
    bool has_max_distance = false;
    double max_distance = 1e30;
    bool has_respect_gitignore = false;
    bool respect_gitignore = true;
    bool has_include_hidden = false;
    bool include_hidden = false;
    std::vector<std::string> ignore_patterns;
    std::vector<std::string> ignore_files;
    int64_t nprobe = -1;
    int64_t ef_search = -1;
    std::string params;

    bool parse_opt(const std::vector<std::string>& args, size_t& i) {
        const std::string& a = args[i];
        if ((a == "--lmcode") && i + 1 < args.size()) {
            dsl_path = args[++i];
            return true;
        } else if (a == "--config-rule" && i + 1 < args.size()) {
            config_rule = args[++i];
            return true;
        } else if (a == "--filter-rule" && i + 1 < args.size()) {
            filter_rule = args[++i];
            return true;
        } else if (a == "--rerank-rule" && i + 1 < args.size()) {
            rerank_rule = args[++i];
            return true;
        } else if (a == "--file-filter-rule" && i + 1 < args.size()) {
            file_filter_rule = args[++i];
            return true;
        } else if (a == "--ignore-rule" && i + 1 < args.size()) {
            ignore_rule = args[++i];
            return true;
        } else if (a == "--embed-rule" && i + 1 < args.size()) {
            embed_rule = args[++i];
            return true;
        } else if (a == "--no-gitignore") {
            respect_gitignore = false;
            has_respect_gitignore = true;
            return true;
        } else if (a == "--gitignore") {
            respect_gitignore = true;
            has_respect_gitignore = true;
            return true;
        } else if (a == "--hidden") {
            include_hidden = true;
            has_include_hidden = true;
            return true;
        } else if (a == "--ignore" && i + 1 < args.size()) {
            ignore_patterns.push_back(args[++i]);
            return true;
        } else if (a == "--ignore-file" && i + 1 < args.size()) {
            ignore_files.push_back(args[++i]);
            return true;
        } else if (a == "--ext" && i + 1 < args.size()) {
            ext_filter = args[++i];
            return true;
        } else if (a == "--path-prefix" && i + 1 < args.size()) {
            path_prefix = args[++i];
            return true;
        } else if (a == "--path-contains" && i + 1 < args.size()) {
            path_contains = args[++i];
            return true;
        } else if (a == "--exclude-path" && i + 1 < args.size()) {
            exclude_path = args[++i];
            return true;
        } else if ((a == "--contains" || a == "--text-contains") && i + 1 < args.size()) {
            text_contains = args[++i];
            return true;
        } else if (a == "--field" && i + 1 < args.size()) {
            std::string kv = args[++i];
            auto eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) {
                throw dslvm::Error("CLI_USAGE", "--field requires KEY=VALUE format: " + kv);
            }
            required_fields[kv.substr(0, eq)] = kv.substr(eq + 1);
            return true;
        } else if (a == "--min-size" && i + 1 < args.size()) {
            min_size = std::stoll(args[++i]);
            return true;
        } else if (a == "--max-size" && i + 1 < args.size()) {
            max_size = std::stoll(args[++i]);
            return true;
        } else if (a == "--min-score" && i + 1 < args.size()) {
            min_score = std::stod(args[++i]);
            has_min_score = true;
            return true;
        } else if (a == "--max-distance" && i + 1 < args.size()) {
            max_distance = std::stod(args[++i]);
            has_max_distance = true;
            return true;
        } else if (a == "--nprobe" && i + 1 < args.size()) {
            nprobe = std::stoll(args[++i]);
            return true;
        } else if (a == "--ef-search" && i + 1 < args.size()) {
            ef_search = std::stoll(args[++i]);
            return true;
        } else if (a == "--params" && i + 1 < args.size()) {
            params = args[++i];
            return true;
        }
        return false;
    }

    void apply(lmsearch::Host& host, lmsearch::FaissIndex* idx) const {
        if (!dsl_path.empty()) {
            std::string bytes = read_binary_file(dsl_path);
            host.bytecode_bytes = bytes;
            host.schema_bytes = lmsearch::schema;
            auto program = dslvm::Program::load(bytes, lmsearch::schema, host.registry);
            std::string auto_config_rule;
            for (const auto& entry : program.entries()) {
                if (entry.iface == "interface VectorFilter(int, Document) -> bool" && idx->filter_rule.empty()) {
                    idx->filter_rule = entry.name;
                } else if (entry.iface == "interface Reranker(SearchHit, string) -> float" && idx->rerank_rule.empty()) {
                    idx->rerank_rule = entry.name;
                } else if (entry.iface == "interface FileFilter(Document) -> bool" && idx->file_filter_rule.empty()) {
                    idx->file_filter_rule = entry.name;
                } else if (entry.iface == "interface IgnoreRule(Document) -> bool" && idx->ignore_rule.empty()) {
                    idx->ignore_rule = entry.name;
                } else if (entry.iface == "interface EmbeddingTransform(FloatList, int) -> FloatList" &&
                           idx->embedder && idx->embedder->transform_rule.empty()) {
                    idx->embedder->transform_rule = entry.name;
                } else if (entry.iface == "interface SearchConfig(FaissIndex) -> void" && auto_config_rule.empty()) {
                    auto_config_rule = entry.name;
                }
            }
            std::string cfg = !config_rule.empty() ? config_rule : auto_config_rule;
            if (!cfg.empty()) {
                host.invoke_rule(
                    cfg,
                    "interface SearchConfig(FaissIndex) -> void",
                    {dslvm::Value::object("FaissIndex", idx)});
            }
        }

        if (!filter_rule.empty()) idx->filter_rule = filter_rule;
        if (!rerank_rule.empty()) idx->rerank_rule = rerank_rule;
        if (!file_filter_rule.empty()) idx->file_filter_rule = file_filter_rule;
        if (!ignore_rule.empty()) idx->ignore_rule = ignore_rule;
        if (!embed_rule.empty() && idx->embedder) idx->embedder->transform_rule = embed_rule;
        if (has_respect_gitignore) idx->filter_opts.respect_gitignore = respect_gitignore;
        if (has_include_hidden) idx->filter_opts.include_hidden = include_hidden;
        for (const auto& f : ignore_files) {
            idx->filter_opts.load_ignore_file(f);
        }
        for (const auto& p : ignore_patterns) {
            idx->filter_opts.ignore_patterns.push_back(p);
        }

        if (!ext_filter.empty()) idx->filter_opts.ext_filter = ext_filter;
        if (!path_prefix.empty()) idx->filter_opts.path_prefix = path_prefix;
        if (!path_contains.empty()) idx->filter_opts.path_contains = path_contains;
        if (!exclude_path.empty()) idx->filter_opts.exclude_path = exclude_path;
        if (!text_contains.empty()) idx->filter_opts.text_contains = text_contains;
        for (const auto& kv : required_fields) {
            idx->filter_opts.required_fields[kv.first] = kv.second;
        }
        if (min_size >= 0) idx->filter_opts.min_size = min_size;
        if (max_size >= 0) idx->filter_opts.max_size = max_size;
        if (has_min_score) idx->filter_opts.min_score = min_score;
        if (has_max_distance) idx->filter_opts.max_distance = max_distance;
        if (!params.empty()) idx->apply_param_string(params);
        if (nprobe > 0) {
            idx->cur_nprobe = nprobe;
            idx->param_str = "nprobe=" + std::to_string(nprobe);
        }
        if (ef_search > 0) {
            idx->cur_ef_search = ef_search;
            idx->param_str = "efSearch=" + std::to_string(ef_search);
        }
    }
};

int run_index_cli(lmsearch::Host& host, const std::vector<std::string>& args) {
    std::string output_path = "build/index.faiss";
    std::string model_id = "all-MiniLM-L6-v2";
    std::string cache_dir;
    std::string index_key = "HNSW16";
    std::string metric = "COSINE";
    int64_t dim = 0;
    bool dryrun = false;
    bool readonly = false;
    std::vector<std::string> paths;
    CliFilterConfig fcfg;

    try {
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            if (a == "--dryrun") {
                dryrun = true;
            } else if (a == "--readonly") {
                readonly = true;
            } else if (a == "--json") {
            } else if ((a == "--output" || a == "-o" || a == "--index") && i + 1 < args.size()) {
                output_path = args[++i];
            } else if ((a == "--model" || a == "-m") && i + 1 < args.size()) {
                model_id = args[++i];
            } else if (a == "--cache-dir" && i + 1 < args.size()) {
                cache_dir = args[++i];
            } else if (a == "--key" && i + 1 < args.size()) {
                index_key = args[++i];
            } else if (a == "--metric" && i + 1 < args.size()) {
                metric = args[++i];
            } else if (a == "--dim" && i + 1 < args.size()) {
                dim = std::stoll(args[++i]);
            } else if (fcfg.parse_opt(args, i)) {
            } else if (!a.empty() && a[0] != '-') {
                paths.push_back(a);
            } else {
                return dslvm::report_error(
                    "cli",
                    dslvm::Error("CLI_USAGE", "unknown or incomplete index option: " + a));
            }
        }
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("cli", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("cli", dslvm::Error("CLI_USAGE", e.what()));
    }

    if (readonly) {
        return dslvm::report_error(
            "policy",
            dslvm::Error("POLICY_READONLY", "index is a write operation and is refused in readonly mode"));
    }
    if (paths.empty()) {
        return dslvm::report_error(
            "cli",
            dslvm::Error("CLI_USAGE", "index requires at least one file or directory path"));
    }
    if (dryrun) {
        std::ostringstream out;
        out << "{\"dryrun\": true, \"operation\": \"index\", \"model_id\": "
            << dslvm::json_escape(model_id)
            << ", \"output\": " << dslvm::json_escape(output_path)
            << ", \"paths_count\": " << paths.size() << "}";
        return dslvm::report_ok(out.str());
    }

    try {
        lmsearch::OnnxModel loaded = lmsearch::OnnxModel::load_or_download_cached(&host, model_id, dim, cache_dir);
        lmsearch::OnnxModel* m = host.alloc_model();
        *m = std::move(loaded);
        m->host = &host;

        lmsearch::FaissIndex* idx = host.alloc_index(m->output_dim, index_key, metric);
        idx->embedder = m;
        fcfg.apply(host, idx);
        int64_t files_indexed = idx->add_path_list(paths, "");
        idx->train_index();
        idx->save_to_file(output_path);

        std::ostringstream out;
        out << "{\"operation\": \"index\""
            << ", \"model_id\": " << dslvm::json_escape(m->model_id)
            << ", \"cache_path\": " << dslvm::json_escape(m->cache_path)
            << ", \"cached\": " << (m->from_cache ? "true" : "false")
            << ", \"dim\": " << m->output_dim
            << ", \"index_key\": " << dslvm::json_escape(idx->index_key)
            << ", \"metric\": " << dslvm::json_escape(idx->metric)
            << ", \"files_indexed\": " << files_indexed
            << ", \"ntotal\": " << idx->entries.size()
            << ", \"file_filter_rule\": " << dslvm::json_escape(idx->file_filter_rule)
            << ", \"entries_called\": " << host.entries_called
            << ", \"index_path\": " << dslvm::json_escape(output_path) << "}";
        return dslvm::report_ok(out.str());
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("index", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("index", dslvm::Error("INDEX_ERROR", e.what()));
    }
}

int run_search_cli(lmsearch::Host& host, const std::vector<std::string>& args) {
    std::string index_path = "build/index.faiss";
    std::string query_text;
    std::string model_id;
    std::string cache_dir;
    int64_t k = 5;
    std::vector<std::string> positional;
    CliFilterConfig fcfg;

    try {
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            if (a == "--json" || a == "--readonly" || a == "--dryrun") {
            } else if ((a == "--index" || a == "-i") && i + 1 < args.size()) {
                index_path = args[++i];
            } else if ((a == "--query" || a == "-q") && i + 1 < args.size()) {
                query_text = args[++i];
            } else if ((a == "-k" || a == "--top-k") && i + 1 < args.size()) {
                k = std::stoll(args[++i]);
            } else if ((a == "--model" || a == "-m") && i + 1 < args.size()) {
                model_id = args[++i];
            } else if (a == "--cache-dir" && i + 1 < args.size()) {
                cache_dir = args[++i];
            } else if (fcfg.parse_opt(args, i)) {
            } else if (!a.empty() && a[0] != '-') {
                positional.push_back(a);
            } else {
                return dslvm::report_error(
                    "cli",
                    dslvm::Error("CLI_USAGE", "unknown or incomplete search option: " + a));
            }
        }
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("cli", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("cli", dslvm::Error("CLI_USAGE", e.what()));
    }

    if (query_text.empty()) {
        if (positional.size() == 1) {
            query_text = positional[0];
        } else if (positional.size() >= 2) {
            index_path = positional[0];
            query_text = positional[1];
        }
    }
    if (query_text.empty()) {
        return dslvm::report_error(
            "cli",
            dslvm::Error("CLI_USAGE", "search requires a query string (--query <TEXT> or positional)"));
    }

    try {
        lmsearch::FaissIndex* idx = lmsearch::FaissIndex::load_from_file(&host, index_path);
        if (!model_id.empty() || !idx->embedder) {
            std::string mid = model_id.empty() ? "all-MiniLM-L6-v2" : model_id;
            lmsearch::OnnxModel loaded = lmsearch::OnnxModel::load_or_download_cached(&host, mid, idx->dim, cache_dir);
            lmsearch::OnnxModel* m = host.alloc_model();
            *m = std::move(loaded);
            m->host = &host;
            idx->embedder = m;
        }
        fcfg.apply(host, idx);
        auto qvec = idx->embedder->run_text(query_text);
        lmsearch::SearchResults* res = idx->run_search(qvec, k, query_text);

        std::ostringstream out;
        out << "{\"operation\": \"search\""
            << ", \"index_path\": " << dslvm::json_escape(index_path)
            << ", \"query\": " << dslvm::json_escape(query_text)
            << ", \"k\": " << k
            << ", \"filter_rule\": " << dslvm::json_escape(idx->filter_rule)
            << ", \"rerank_rule\": " << dslvm::json_escape(idx->rerank_rule)
            << ", \"entries_called\": " << host.entries_called
            << ", \"hits\": [";
        for (size_t i = 0; i < res->hits.size(); ++i) {
            if (i > 0) out << ", ";
            const auto* h = res->hits[i];
            std::string fpath;
            std::string fname;
            if (h->doc) {
                auto it_p = h->doc->fields.find("path");
                if (it_p != h->doc->fields.end()) fpath = it_p->second;
                auto it_f = h->doc->fields.find("filename");
                if (it_f != h->doc->fields.end()) fname = it_f->second;
            }
            std::string snippet = h->doc ? h->doc->text : "";
            if (snippet.size() > 160) snippet = snippet.substr(0, 160) + "...";
            out << "{\"rank\": " << h->rank
                << ", \"id\": " << h->id
                << ", \"score\": " << h->score
                << ", \"distance\": " << h->distance
                << ", \"path\": " << dslvm::json_escape(fpath)
                << ", \"filename\": " << dslvm::json_escape(fname)
                << ", \"snippet\": " << dslvm::json_escape(snippet) << "}";
        }
        out << "]}";
        return dslvm::report_ok(out.str());
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("search", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("search", dslvm::Error("SEARCH_ERROR", e.what()));
    }
}

std::string format_hits_array_json(const lmsearch::SearchResults* res) {
    std::ostringstream out;
    out << "[";
    if (res) {
        for (size_t i = 0; i < res->hits.size(); ++i) {
            if (i > 0) out << ", ";
            const auto* h = res->hits[i];
            std::string fpath;
            std::string fname;
            std::string fext;
            int64_t fsize = h->doc ? static_cast<int64_t>(h->doc->text.size()) : 0;
            if (h->doc) {
                auto it_p = h->doc->fields.find("path");
                if (it_p != h->doc->fields.end()) fpath = it_p->second;
                auto it_f = h->doc->fields.find("filename");
                if (it_f != h->doc->fields.end()) fname = it_f->second;
                auto it_e = h->doc->fields.find("ext");
                if (it_e != h->doc->fields.end()) fext = it_e->second;
                auto it_sn = h->doc->numbers.find("size");
                if (it_sn != h->doc->numbers.end()) {
                    fsize = it_sn->second;
                } else {
                    auto it_s = h->doc->fields.find("size");
                    if (it_s != h->doc->fields.end() && !it_s->second.empty()) {
                        try { fsize = std::stoll(it_s->second); } catch (...) {}
                    }
                }
            }
            std::string snippet = h->doc ? h->doc->text : "";
            if (snippet.size() > 160) snippet = snippet.substr(0, 160) + "...";
            out << "{\"rank\": " << h->rank
                << ", \"id\": " << h->id
                << ", \"score\": " << h->score
                << ", \"distance\": " << h->distance
                << ", \"path\": " << dslvm::json_escape(fpath)
                << ", \"filename\": " << dslvm::json_escape(fname)
                << ", \"ext\": " << dslvm::json_escape(fext)
                << ", \"size\": " << fsize
                << ", \"snippet\": " << dslvm::json_escape(snippet) << "}";
        }
    }
    out << "]";
    return out.str();
}

std::map<std::string, std::string> parse_query_params(const std::string& qs) {
    std::map<std::string, std::string> params;
    size_t start = 0;
    while (start < qs.size()) {
        size_t amp = qs.find('&', start);
        std::string pair = (amp == std::string::npos) ? qs.substr(start) : qs.substr(start, amp - start);
        if (!pair.empty()) {
            size_t eq = pair.find('=');
            if (eq == std::string::npos) {
                params[lmsearch::url_decode(pair)] = "";
            } else {
                params[lmsearch::url_decode(pair.substr(0, eq))] = lmsearch::url_decode(pair.substr(eq + 1));
            }
        }
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
    return params;
}

void apply_http_filter_params(
    lmsearch::Host& host,
    lmsearch::FaissIndex* idx,
    const std::map<std::string, std::string>& params) {
    auto it_reset = params.find("reset");
    if (it_reset != params.end() && (it_reset->second == "1" || it_reset->second == "true")) {
        idx->filter_opts.ext_filter.clear();
        idx->filter_opts.path_prefix.clear();
        idx->filter_opts.path_contains.clear();
        idx->filter_opts.exclude_path.clear();
        idx->filter_opts.text_contains.clear();
        idx->filter_opts.required_fields.clear();
        idx->filter_opts.min_size = -1;
        idx->filter_opts.max_size = -1;
        idx->filter_opts.min_score = -1e30;
        idx->filter_opts.max_distance = 1e30;
        idx->filter_rule.clear();
        idx->rerank_rule.clear();
    }

    auto it_dsl = params.find("dsl");
    if (it_dsl != params.end() && !it_dsl->second.empty()) {
        CliFilterConfig fcfg;
        fcfg.dsl_path = it_dsl->second;
        fcfg.apply(host, idx);
    }
    auto it_ext = params.find("ext");
    if (it_ext != params.end()) idx->filter_opts.ext_filter = it_ext->second;
    auto it_pp = params.find("path_prefix");
    if (it_pp != params.end()) idx->filter_opts.path_prefix = it_pp->second;
    auto it_pc = params.find("path_contains");
    if (it_pc != params.end()) idx->filter_opts.path_contains = it_pc->second;
    auto it_ep = params.find("exclude_path");
    if (it_ep != params.end()) idx->filter_opts.exclude_path = it_ep->second;
    auto it_tc = params.find("contains");
    if (it_tc != params.end()) idx->filter_opts.text_contains = it_tc->second;
    auto it_tc2 = params.find("text_contains");
    if (it_tc2 != params.end()) idx->filter_opts.text_contains = it_tc2->second;
    auto it_fk = params.find("field_key");
    auto it_fv = params.find("field_val");
    if (it_fk != params.end() && it_fv != params.end() && !it_fk->second.empty()) {
        idx->filter_opts.required_fields[it_fk->second] = it_fv->second;
    }
    auto it_field = params.find("field");
    if (it_field != params.end() && !it_field->second.empty()) {
        auto eq = it_field->second.find('=');
        if (eq != std::string::npos && eq > 0) {
            idx->filter_opts.required_fields[it_field->second.substr(0, eq)] = it_field->second.substr(eq + 1);
        }
    }
    auto it_mins = params.find("min_size");
    if (it_mins != params.end() && !it_mins->second.empty()) idx->filter_opts.min_size = std::stoll(it_mins->second);
    auto it_maxs = params.find("max_size");
    if (it_maxs != params.end() && !it_maxs->second.empty()) idx->filter_opts.max_size = std::stoll(it_maxs->second);
    auto it_minsc = params.find("min_score");
    if (it_minsc != params.end() && !it_minsc->second.empty()) idx->filter_opts.min_score = std::stod(it_minsc->second);
    auto it_maxd = params.find("max_distance");
    if (it_maxd != params.end() && !it_maxd->second.empty()) idx->filter_opts.max_distance = std::stod(it_maxd->second);
    auto it_frule = params.find("filter_rule");
    if (it_frule != params.end()) idx->filter_rule = it_frule->second;
    auto it_rrule = params.find("rerank_rule");
    if (it_rrule != params.end()) idx->rerank_rule = it_rrule->second;
}

std::string build_filter_state_json(const lmsearch::FaissIndex* idx) {
    std::ostringstream out;
    out << "{\"ext\": " << dslvm::json_escape(idx->filter_opts.ext_filter)
        << ", \"path_prefix\": " << dslvm::json_escape(idx->filter_opts.path_prefix)
        << ", \"path_contains\": " << dslvm::json_escape(idx->filter_opts.path_contains)
        << ", \"exclude_path\": " << dslvm::json_escape(idx->filter_opts.exclude_path)
        << ", \"text_contains\": " << dslvm::json_escape(idx->filter_opts.text_contains)
        << ", \"min_size\": " << idx->filter_opts.min_size
        << ", \"max_size\": " << idx->filter_opts.max_size
        << ", \"filter_rule\": " << dslvm::json_escape(idx->filter_rule)
        << ", \"rerank_rule\": " << dslvm::json_escape(idx->rerank_rule) << "}";
    return out.str();
}

void send_http_json(
#ifdef _WIN32
    SOCKET client_fd,
#else
    int client_fd,
#endif
    int status_code,
    const std::string& status_text,
    const std::string& json_body) {
    std::ostringstream resp;
    resp << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
         << "Content-Type: application/json\r\n"
         << "Content-Length: " << json_body.size() << "\r\n"
         << "Connection: close\r\n\r\n"
         << json_body;
    std::string data = resp.str();
    send(client_fd, data.data(), static_cast<int>(data.size()), 0);
}

int run_serve_cli(lmsearch::Host& host, const std::vector<std::string>& args) {
    std::string bind_host = "127.0.0.1";
    int port = 8080;
    std::string port_file;
    int64_t timeout_ms = 300000;  // 5 minutes default idle auto-shutdown
    bool weights_only_cache = true;
    std::string model_id = "all-MiniLM-L6-v2";
    std::string cache_dir;
    std::string index_key = "HNSW16";
    std::string metric = "COSINE";
    int64_t dim = 0;
    bool dryrun = false;
    std::vector<std::string> paths;
    CliFilterConfig fcfg;

    try {
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            if (a == "--dryrun") {
                dryrun = true;
            } else if (a == "--json" || a == "--readonly") {
            } else if (a == "--host" && i + 1 < args.size()) {
                bind_host = args[++i];
            } else if ((a == "--port" || a == "-p") && i + 1 < args.size()) {
                port = std::stoi(args[++i]);
            } else if (a == "--port-file" && i + 1 < args.size()) {
                port_file = args[++i];
            } else if (a == "--timeout" && i + 1 < args.size()) {
                timeout_ms = static_cast<int64_t>(std::stod(args[++i]) * 1000.0);
            } else if (a == "--timeout-ms" && i + 1 < args.size()) {
                timeout_ms = std::stoll(args[++i]);
            } else if (a == "--weights-only-cache" || a == "--clean-start") {
                weights_only_cache = true;
            } else if ((a == "--model" || a == "-m") && i + 1 < args.size()) {
                model_id = args[++i];
            } else if (a == "--cache-dir" && i + 1 < args.size()) {
                cache_dir = args[++i];
            } else if (a == "--key" && i + 1 < args.size()) {
                index_key = args[++i];
            } else if (a == "--metric" && i + 1 < args.size()) {
                metric = args[++i];
            } else if (a == "--dim" && i + 1 < args.size()) {
                dim = std::stoll(args[++i]);
            } else if (fcfg.parse_opt(args, i)) {
            } else if (!a.empty() && a[0] != '-') {
                paths.push_back(a);
            } else {
                return dslvm::report_error(
                    "cli",
                    dslvm::Error("CLI_USAGE", "unknown or incomplete serve option: " + a));
            }
        }
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("cli", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("cli", dslvm::Error("CLI_USAGE", e.what()));
    }

    if (dryrun) {
        std::ostringstream out;
        out << "{\"dryrun\": true, \"operation\": \"serve\", \"host\": "
            << dslvm::json_escape(bind_host)
            << ", \"port\": " << port
            << ", \"timeout_ms\": " << timeout_ms
            << ", \"weights_only_disk_cache\": " << (weights_only_cache ? "true" : "false")
            << ", \"clean_start\": true"
            << ", \"model_id\": " << dslvm::json_escape(model_id)
            << ", \"paths_count\": " << paths.size() << "}";
        return dslvm::report_ok(out.str());
    }

    // 1. Cache ONLY model weights on disk; index & broad result cache start clean in RAM
    lmsearch::OnnxModel loaded = lmsearch::OnnxModel::load_or_download_cached(&host, model_id, dim, cache_dir);
    lmsearch::OnnxModel* m = host.alloc_model();
    *m = std::move(loaded);
    m->host = &host;

    lmsearch::FaissIndex* idx = host.alloc_index(m->output_dim, index_key, metric);
    idx->embedder = m;
    idx->weights_only_disk_cache = weights_only_cache;
    fcfg.apply(host, idx);
    if (!paths.empty()) {
        idx->register_lazy_paths(paths);
    }

#ifdef _WIN32
    static bool wsa_init = false;
    if (!wsa_init) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        wsa_init = true;
    }
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd == INVALID_SOCKET) {
        return dslvm::report_error("serve", dslvm::Error("SOCKET_ERROR", "failed to create listening socket"));
    }
    char opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        return dslvm::report_error("serve", dslvm::Error("SOCKET_ERROR", "failed to create listening socket"));
    }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind_host == "0.0.0.0" || bind_host == "*") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) <= 0) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
#ifdef _WIN32
        closesocket(server_fd);
#else
        close(server_fd);
#endif
        return dslvm::report_error(
            "serve",
            dslvm::Error("BIND_ERROR", "cannot bind to " + bind_host + ":" + std::to_string(port)));
    }

    if (listen(server_fd, 16) < 0) {
#ifdef _WIN32
        closesocket(server_fd);
#else
        close(server_fd);
#endif
        return dslvm::report_error("serve", dslvm::Error("LISTEN_ERROR", "listen failed"));
    }

    sockaddr_in bound_addr{};
#ifdef _WIN32
    int addr_len = sizeof(bound_addr);
#else
    socklen_t addr_len = sizeof(bound_addr);
#endif
    if (getsockname(server_fd, reinterpret_cast<sockaddr*>(&bound_addr), &addr_len) == 0) {
        port = ntohs(bound_addr.sin_port);
    }

    if (!port_file.empty()) {
        std::ofstream pf(port_file);
        pf << port << "\n";
        pf.flush();
    }

    auto last_activity = std::chrono::steady_clock::now();
    bool running = true;
    std::string shutdown_reason = "client_shutdown";
    int64_t requests_served = 0;

    while (running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);

        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms tick for responsive idle timeout

        int ready = select(
#ifdef _WIN32
            0,
#else
            server_fd + 1,
#endif
            &readfds,
            nullptr,
            nullptr,
            &tv);

        if (ready <= 0) {
            if (timeout_ms > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - last_activity)
                                   .count();
                if (elapsed >= timeout_ms) {
                    shutdown_reason = "idle_timeout";
                    running = false;
                    break;
                }
            }
            continue;
        }

        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
        SOCKET client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd == INVALID_SOCKET) continue;
#else
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) continue;
#endif

        last_activity = std::chrono::steady_clock::now();
        requests_served += 1;

        char req_buf[16384];
        int received = recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
        if (received <= 0) {
#ifdef _WIN32
            closesocket(client_fd);
#else
            close(client_fd);
#endif
            continue;
        }
        req_buf[received] = '\0';
        std::string raw_req(req_buf, static_cast<size_t>(received));

        // Parse request line: METHOD /path?query HTTP/1.1
        std::string method = "GET";
        std::string target = "/";
        auto line_end = raw_req.find("\r\n");
        std::string req_line = (line_end == std::string::npos) ? raw_req : raw_req.substr(0, line_end);
        {
            std::istringstream iss(req_line);
            iss >> method >> target;
        }

        std::string route = target;
        std::string query_str;
        auto qpos = target.find('?');
        if (qpos != std::string::npos) {
            route = target.substr(0, qpos);
            query_str = target.substr(qpos + 1);
        }
        auto params = parse_query_params(query_str);

        try {
            if (route == "/status" || route == "/health" || route == "/") {
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"status\""
                    << ", \"status\": \"running\""
                    << ", \"session\": \"global\""
                    << ", \"auth_required\": false"
                    << ", \"clean_start\": true"
                    << ", \"weights_only_disk_cache\": " << (idx->weights_only_disk_cache ? "true" : "false")
                    << ", \"model_id\": " << dslvm::json_escape(m->model_id)
                    << ", \"model_cache_path\": " << dslvm::json_escape(m->cache_path)
                    << ", \"lazy_candidates\": " << idx->lazy_candidates.size()
                    << ", \"lazy_embedded_docs\": " << idx->lazy_embedded_paths.size()
                    << ", \"ntotal\": " << idx->entries.size()
                    << ", \"cached_broad_hits\": " << idx->cached_broad_hits.size()
                    << ", \"cached_query\": " << dslvm::json_escape(idx->cached_query_text)
                    << ", \"filter\": " << build_filter_state_json(idx) << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
            } else if (route == "/paths") {
                int64_t added = 0;
                auto it_add = params.find("add");
                if (it_add != params.end() && !it_add->second.empty()) {
                    added = idx->register_lazy_path(it_add->second);
                }
                auto it_path = params.find("path");
                if (it_path != params.end() && !it_path->second.empty()) {
                    added += idx->register_lazy_path(it_path->second);
                }
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"paths\""
                    << ", \"lazy_candidates_added\": " << added
                    << ", \"lazy_candidates\": " << idx->lazy_candidates.size()
                    << ", \"lazy_embedded_docs\": " << idx->lazy_embedded_paths.size() << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
            } else if (route == "/query" || route == "/search") {
                apply_http_filter_params(host, idx, params);
                std::string qtext;
                auto it_q = params.find("q");
                if (it_q != params.end()) qtext = it_q->second;
                auto it_query = params.find("query");
                if (qtext.empty() && it_query != params.end()) qtext = it_query->second;
                int64_t k = 10;
                auto it_k = params.find("k");
                if (it_k != params.end() && !it_k->second.empty()) k = std::stoll(it_k->second);

                lmsearch::SearchResults* res = idx->run_lazy_query(qtext, k);
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"query\""
                    << ", \"session\": \"global\""
                    << ", \"query\": " << dslvm::json_escape(qtext)
                    << ", \"k\": " << k
                    << ", \"from_memory_cache\": false"
                    << ", \"lazy_candidates\": " << idx->lazy_candidates.size()
                    << ", \"lazy_newly_embedded\": " << idx->last_newly_embedded
                    << ", \"lazy_embedded_docs\": " << idx->lazy_embedded_paths.size()
                    << ", \"cached_broad_hits\": " << idx->cached_broad_hits.size()
                    << ", \"filter\": " << build_filter_state_json(idx)
                    << ", \"hits\": " << format_hits_array_json(res) << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
            } else if (route == "/filter") {
                apply_http_filter_params(host, idx, params);
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"filter\""
                    << ", \"session\": \"global\""
                    << ", \"cached_broad_hits\": " << idx->cached_broad_hits.size()
                    << ", \"cached_query\": " << dslvm::json_escape(idx->cached_query_text)
                    << ", \"filter\": " << build_filter_state_json(idx) << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
            } else if (route == "/requery" || route == "/view") {
                apply_http_filter_params(host, idx, params);
                int64_t k = 10;
                auto it_k = params.find("k");
                if (it_k != params.end() && !it_k->second.empty()) k = std::stoll(it_k->second);

                lmsearch::SearchResults* res = idx->run_refined_view(k);
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"requery\""
                    << ", \"session\": \"global\""
                    << ", \"query\": " << dslvm::json_escape(idx->cached_query_text)
                    << ", \"k\": " << k
                    << ", \"from_memory_cache\": true"
                    << ", \"lazy_newly_embedded\": 0"
                    << ", \"lazy_embedded_docs\": " << idx->lazy_embedded_paths.size()
                    << ", \"cached_broad_hits\": " << idx->cached_broad_hits.size()
                    << ", \"filter\": " << build_filter_state_json(idx)
                    << ", \"hits\": " << format_hits_array_json(res) << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
            } else if (route == "/shutdown") {
                std::ostringstream out;
                out << "{\"ok\": true, \"result\": {\"operation\": \"shutdown\""
                    << ", \"status\": \"stopping\""
                    << ", \"requests_served\": " << requests_served << "}}";
                send_http_json(client_fd, 200, "OK", out.str());
                shutdown_reason = "client_shutdown";
                running = false;
            } else {
                send_http_json(
                    client_fd,
                    404,
                    "Not Found",
                    "{\"ok\": false, \"error\": {\"code\": \"NOT_FOUND\", \"message\": \"unknown route\"}}");
            }
        } catch (const std::exception& e) {
            std::ostringstream err;
            err << "{\"ok\": false, \"error\": {\"code\": \"SERVER_ERROR\", \"message\": "
                << dslvm::json_escape(e.what()) << "}}";
            send_http_json(client_fd, 500, "Internal Server Error", err.str());
        }

#ifdef _WIN32
        closesocket(client_fd);
#else
        close(client_fd);
#endif
    }

#ifdef _WIN32
    closesocket(server_fd);
#else
    close(server_fd);
#endif

    std::ostringstream out;
    out << "{\"operation\": \"serve\""
        << ", \"status\": \"stopped\""
        << ", \"shutdown_reason\": " << dslvm::json_escape(shutdown_reason)
        << ", \"port\": " << port
        << ", \"requests_served\": " << requests_served
        << ", \"clean_start\": true"
        << ", \"weights_only_disk_cache\": " << (idx->weights_only_disk_cache ? "true" : "false")
        << ", \"lazy_candidates\": " << idx->lazy_candidates.size()
        << ", \"lazy_embedded_docs\": " << idx->lazy_embedded_paths.size()
        << ", \"cached_broad_hits\": " << idx->cached_broad_hits.size() << "}";
    return dslvm::report_ok(out.str());
}

int run_client_cli(lmsearch::Host& host, const std::vector<std::string>& args) {
    std::string base_url = "http://127.0.0.1:8080";
    std::string action = "status";
    bool action_set = false;
    std::string query_text;
    int64_t k = 10;
    bool reset_filter = false;
    std::vector<std::string> add_paths;
    CliFilterConfig fcfg;

    try {
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            if (a == "--json") {
            } else if ((a == "--url" || a == "-u") && i + 1 < args.size()) {
                base_url = args[++i];
            } else if ((a == "--port" || a == "-p") && i + 1 < args.size()) {
                base_url = "http://127.0.0.1:" + args[++i];
            } else if ((a == "--query" || a == "-q") && i + 1 < args.size()) {
                query_text = args[++i];
            } else if ((a == "-k" || a == "--top-k") && i + 1 < args.size()) {
                k = std::stoll(args[++i]);
            } else if (a == "--reset" || a == "--clear") {
                reset_filter = true;
            } else if (a == "--add-path" && i + 1 < args.size()) {
                add_paths.push_back(args[++i]);
            } else if (fcfg.parse_opt(args, i)) {
            } else if (!a.empty() && a[0] != '-') {
                if (!action_set &&
                    (a == "status" || a == "health" || a == "paths" || a == "query" ||
                     a == "search" || a == "filter" || a == "requery" || a == "view" || a == "shutdown")) {
                    action = a;
                    action_set = true;
                } else if (action == "paths") {
                    add_paths.push_back(a);
                } else if ((action == "query" || action == "search") && query_text.empty()) {
                    query_text = a;
                } else {
                    return dslvm::report_error(
                        "cli",
                        dslvm::Error("CLI_USAGE", "unexpected client argument: " + a));
                }
            } else {
                return dslvm::report_error(
                    "cli",
                    dslvm::Error("CLI_USAGE", "unknown or incomplete client option: " + a));
            }
        }
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("cli", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("cli", dslvm::Error("CLI_USAGE", e.what()));
    }

    auto append_filter_qs = [&](std::ostringstream& qs, bool& first) {
        auto add_kv = [&](const std::string& key, const std::string& val) {
            qs << (first ? "?" : "&") << key << "=" << lmsearch::url_encode(val);
            first = false;
        };
        if (reset_filter) add_kv("reset", "1");
        if (!fcfg.dsl_path.empty()) add_kv("dsl", fcfg.dsl_path);
        if (!fcfg.ext_filter.empty()) add_kv("ext", fcfg.ext_filter);
        if (!fcfg.path_prefix.empty()) add_kv("path_prefix", fcfg.path_prefix);
        if (!fcfg.path_contains.empty()) add_kv("path_contains", fcfg.path_contains);
        if (!fcfg.exclude_path.empty()) add_kv("exclude_path", fcfg.exclude_path);
        if (!fcfg.text_contains.empty()) add_kv("contains", fcfg.text_contains);
        for (const auto& kv : fcfg.required_fields) {
            add_kv("field", kv.first + "=" + kv.second);
        }
        if (fcfg.min_size >= 0) add_kv("min_size", std::to_string(fcfg.min_size));
        if (fcfg.max_size >= 0) add_kv("max_size", std::to_string(fcfg.max_size));
        if (fcfg.has_min_score) add_kv("min_score", std::to_string(fcfg.min_score));
        if (fcfg.has_max_distance) add_kv("max_distance", std::to_string(fcfg.max_distance));
        if (!fcfg.filter_rule.empty()) add_kv("filter_rule", fcfg.filter_rule);
        if (!fcfg.rerank_rule.empty()) add_kv("rerank_rule", fcfg.rerank_rule);
    };

    try {
        lmsearch::RestClient* client = host.alloc_client(base_url);
        std::string resp;
        if (action == "status" || action == "health") {
            resp = client->get("/status");
        } else if (action == "paths") {
            for (const auto& p : add_paths) {
                resp = client->get("/paths?add=" + lmsearch::url_encode(p));
            }
            if (add_paths.empty()) resp = client->get("/paths");
        } else if (action == "query" || action == "search") {
            std::ostringstream qs;
            qs << "/query?q=" << lmsearch::url_encode(query_text) << "&k=" << k;
            bool first = false;
            append_filter_qs(qs, first);
            resp = client->get(qs.str());
        } else if (action == "filter") {
            std::ostringstream qs;
            qs << "/filter";
            bool first = true;
            append_filter_qs(qs, first);
            resp = client->get(qs.str());
        } else if (action == "requery" || action == "view") {
            std::ostringstream qs;
            qs << "/requery?k=" << k;
            bool first = false;
            append_filter_qs(qs, first);
            resp = client->get(qs.str());
        } else if (action == "shutdown") {
            resp = client->get("/shutdown");
        }
        if (resp.empty()) {
            return dslvm::report_error(
                "client",
                dslvm::Error("HTTP_CONNECTION_ERROR", "failed to connect to REST server at " + base_url));
        }
        std::cout << resp << "\n";
        return 0;
    } catch (const dslvm::Error& e) {
        return dslvm::report_error("client", e);
    } catch (const std::exception& e) {
        return dslvm::report_error("client", dslvm::Error("CLIENT_ERROR", e.what()));
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif
    lmsearch::Host host;
    const std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() == 1 && args[0] == "--version") {
        std::cout << "lmsearch " << LMSEARCH_VERSION << "\n";
        return 0;
    }

    if ((args.size() == 1 && args[0] == "--lmcode-schema") ||
        (args.size() == 2 && args[0] == "--lmcode-schema" && args[1] == "--json")) {
        std::cout << lmsearch::schema;
        return 0;
    }

    if (args.size() == 1 && args[0] == "--lmcode-signatures") {
        for (const auto& sig : host.registry.signatures()) {
            std::cout << sig << "\n";
        }
        return 0;
    }

    if ((args.size() == 1 && (args[0] == "--schema" || args[0] == "--cli-schema")) ||
        (args.size() == 2 && (args[0] == "--schema" || args[0] == "--cli-schema") && args[1] == "--json")) {
        std::cout << lmsearch::cli_schema;
        return 0;
    }

    if (args.size() == 2 && args[0] == "--lmcode") {
        return run_bytecode(host, args[1], false, false);
    }

    if (args.size() == 4 && args[0] == "--save-dslb" && args[2] == "--lmcode") {
        try {
            std::string bytes = read_binary_file(args[3]);
            std::ofstream out_f(args[1], std::ios::binary);
            out_f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        } catch (const dslvm::Error& e) {
            return dslvm::report_error("io", e);
        }
        return run_bytecode(host, args[3], false, false);
    }

    if (!args.empty() && args[0] == "index") {
        return run_index_cli(host, args);
    }

    if (!args.empty() && args[0] == "search") {
        return run_search_cli(host, args);
    }

    if (!args.empty() && args[0] == "serve") {
        return run_serve_cli(host, args);
    }

    if (!args.empty() && args[0] == "client") {
        return run_client_cli(host, args);
    }

    if (!args.empty() && args[0] == "run") {
        std::string program_path;
        bool dryrun = false;
        bool readonly = false;

        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            if (a == "--dryrun") {
                dryrun = true;
            } else if (a == "--readonly") {
                readonly = true;
            } else if (a == "--json") {
                // Output is always structured JSON envelope
            } else if ((a == "--program" || a == "--lmcode") && i + 1 < args.size()) {
                program_path = args[++i];
            } else if (a == "--input-json" && i + 1 < args.size()) {
                program_path = extract_json_string_field(args[++i], "program");
            } else if (a == "--input-file" && i + 1 < args.size()) {
                try {
                    std::string content = read_text_file(args[++i]);
                    program_path = extract_json_string_field(content, "program");
                } catch (const dslvm::Error& e) {
                    return dslvm::report_error("io", e);
                }
            } else {
                return dslvm::report_error(
                    "cli",
                    dslvm::Error("CLI_USAGE", "unknown or incomplete run option: " + a));
            }
        }

        if (program_path.empty()) {
            return dslvm::report_error(
                "cli",
                dslvm::Error("CLI_USAGE", "run requires --program <FILE>"));
        }
        return run_bytecode(host, program_path, dryrun, readonly);
    }

    return dslvm::report_error(
        "cli",
        dslvm::Error(
            "CLI_USAGE",
            "usage: lmsearch [--version | --schema [--json] | --lmcode-schema [--json] | --lmcode-signatures | --lmcode <FILE> | index <PATHS...> --output <FILE> | search --index <FILE> --query <TEXT> | serve <PATHS...> [--port <PORT>] [--timeout <SEC>] [--weights-only-cache] | client [--url <URL>] <status|query|filter|requery|shutdown> | run --program <FILE> [--dryrun] [--readonly]]"));
}
