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
            "usage: lmsearch [--version | --schema [--json] | --dsl-schema [--json] | --dsl-signatures | --dsl <FILE> | run --program <FILE> [--dryrun] [--readonly]]"));
}
