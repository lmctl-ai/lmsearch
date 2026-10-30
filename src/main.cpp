// SPDX-License-Identifier: MIT
#include "lmsearch.hpp"
#include "lmsearch_schema.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
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

    return dslvm::report_ok(host.build_result_json());
}

struct CliFilterConfig {
    std::string dsl_path;
    std::string config_rule;
    std::string filter_rule;
    std::string rerank_rule;
    std::string file_filter_rule;
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
    int64_t nprobe = -1;
    int64_t ef_search = -1;
    std::string params;

    bool parse_opt(const std::vector<std::string>& args, size_t& i) {
        const std::string& a = args[i];
        if ((a == "--dsl" || a == "--lmcode") && i + 1 < args.size()) {
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
        } else if (a == "--embed-rule" && i + 1 < args.size()) {
            embed_rule = args[++i];
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
        if (!embed_rule.empty() && idx->embedder) idx->embedder->transform_rule = embed_rule;

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

    if ((args.size() == 1 && args[0] == "--dsl-schema") ||
        (args.size() == 2 && args[0] == "--dsl-schema" && args[1] == "--json")) {
        std::cout << lmsearch::schema;
        return 0;
    }

    if (args.size() == 1 && args[0] == "--dsl-signatures") {
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

    if (args.size() == 2 && args[0] == "--dsl") {
        return run_bytecode(host, args[1], false, false);
    }

    if (!args.empty() && args[0] == "index") {
        return run_index_cli(host, args);
    }

    if (!args.empty() && args[0] == "search") {
        return run_search_cli(host, args);
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
            } else if ((a == "--program" || a == "--dsl") && i + 1 < args.size()) {
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
            "usage: lmsearch [--version | --schema [--json] | --dsl-schema [--json] | --dsl-signatures | --dsl <FILE> | index <PATHS...> --output <FILE> | search --index <FILE> --query <TEXT> | run --program <FILE> [--dryrun] [--readonly]]"));
}
