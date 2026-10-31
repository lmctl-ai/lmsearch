#include "lmsearch.hpp"
#include "collections.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

#ifdef LMSEARCH_WITH_UPSTREAM_FAISS_ONNX
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>
#endif

namespace lmsearch {

namespace {

double deterministic_weight(int64_t seed, int64_t r, int64_t c) {
    uint64_t x = static_cast<uint64_t>(seed) * 6364136223846793005ULL +
                 static_cast<uint64_t>(r + 1) * 1442695040888963407ULL +
                 static_cast<uint64_t>(c + 1) * 2862933555777941757ULL;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    double unit = static_cast<double>(x & 0xFFFFFFULL) / static_cast<double>(0xFFFFFFULL);
    return (unit * 2.0 - 1.0) * 0.5;
}

std::vector<double> l2_norm_vec(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    if (sum <= 1e-24) return v;
    double inv = 1.0 / std::sqrt(sum);
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv;
    return out;
}

double vec_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

double vec_ip(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

std::vector<int64_t> extract_ints(const std::string& s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < s.size()) {
        if (std::isdigit(static_cast<unsigned char>(s[i]))) {
            int64_t v = 0;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                v = v * 10 + (s[i] - '0');
                ++i;
            }
            out.push_back(v);
        } else {
            ++i;
        }
    }
    return out;
}

std::vector<int64_t> tokenize_text_to_ids(const std::string& text, int64_t vocab_size) {
    std::vector<int64_t> tokens;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char ch : cur) {
            unsigned char lower = static_cast<unsigned char>(std::tolower(ch));
            h ^= lower;
            h *= 1099511628211ULL;
        }
        tokens.push_back(static_cast<int64_t>(h % static_cast<uint64_t>(std::max<int64_t>(1, vocab_size))));
        cur.clear();
    };
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_') {
            cur.push_back(static_cast<char>(ch));
        } else {
            flush();
        }
    }
    flush();
    if (tokens.empty()) tokens.push_back(0);
    return tokens;
}

} // namespace

int64_t parse_memory_bytes(const std::string& spec) {
    std::string s;
    for (char c : spec) {
        if (!std::isspace(static_cast<unsigned char>(c))) s.push_back(c);
    }
    if (s.empty()) throw dslvm::Error("empty memory specification");
    size_t idx = 0;
    double val = 0.0;
    try {
        val = std::stod(s, &idx);
    } catch (...) {
        throw dslvm::Error("invalid memory specification: " + spec);
    }
    if (val <= 0.0) throw dslvm::Error("memory specification must be positive: " + spec);
    std::string unit = s.substr(idx);
    for (char& c : unit) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    double mult = 1.0;
    if (unit.empty() || unit == "B") mult = 1.0;
    else if (unit == "K" || unit == "KB" || unit == "KIB") mult = 1024.0;
    else if (unit == "M" || unit == "MB" || unit == "MIB") mult = 1024.0 * 1024.0;
    else if (unit == "G" || unit == "GB" || unit == "GIB") mult = 1024.0 * 1024.0 * 1024.0;
    else if (unit == "T" || unit == "TB" || unit == "TIB") mult = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    else throw dslvm::Error("unknown memory unit in: " + spec);
    return static_cast<int64_t>(val * mult);
}

std::string format_memory_bytes(int64_t bytes) {
    char buf[64];
    if (bytes >= 1024LL * 1024LL * 1024LL) {
        std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024LL * 1024LL) {
        std::snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else if (bytes >= 1024LL) {
        std::snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(bytes) / 1024.0);
    } else {
        std::snprintf(buf, sizeof(buf), "%lld B", static_cast<long long>(bytes));
    }
    return buf;
}

void OnnxModel::init_text_embedder(int64_t vocab, int64_t dim, int64_t s) {
    if (vocab <= 0 || dim <= 0) throw dslvm::Error("onnx_embedder requires positive vocab_size and dim");
    vocab_size = vocab;
    input_dim = vocab;
    hidden_dim = dim;
    output_dim = dim;
    seed = s;
    is_mlp = false;
    pooling = "MEAN";
    normalize = true;
    node_ops = {"Gather", "MatMul", "Add", "Relu", "ReduceMean", "Div"};
    w1.resize(static_cast<size_t>(vocab * dim));
    for (int64_t r = 0; r < vocab; ++r) {
        for (int64_t c = 0; c < dim; ++c) {
            w1[static_cast<size_t>(r * dim + c)] = deterministic_weight(seed, r, c);
        }
    }
    w2.resize(static_cast<size_t>(dim * dim));
    b1.resize(static_cast<size_t>(dim));
    for (int64_t r = 0; r < dim; ++r) {
        b1[static_cast<size_t>(r)] = deterministic_weight(seed + 7, r, 0) * 0.1;
        for (int64_t c = 0; c < dim; ++c) {
            w2[static_cast<size_t>(r * dim + c)] = (r == c ? 0.75 : 0.0) + deterministic_weight(seed + 13, r, c) * 0.25;
        }
    }
}

void OnnxModel::init_mlp_embedder(int64_t in_d, int64_t hid_d, int64_t out_d, int64_t s) {
    if (in_d <= 0 || hid_d <= 0 || out_d <= 0)
        throw dslvm::Error("onnx_mlp_embedder requires positive dimensions");
    vocab_size = in_d;
    input_dim = in_d;
    hidden_dim = hid_d;
    output_dim = out_d;
    seed = s;
    is_mlp = true;
    pooling = "NONE";
    normalize = true;
    node_ops = {"MatMul", "Add", "Relu", "MatMul", "Add", "Div"};
    w1.resize(static_cast<size_t>(in_d * hid_d));
    b1.resize(static_cast<size_t>(hid_d));
    for (int64_t r = 0; r < in_d; ++r) {
        for (int64_t c = 0; c < hid_d; ++c) {
            w1[static_cast<size_t>(r * hid_d + c)] = deterministic_weight(seed, r, c);
        }
    }
    for (int64_t c = 0; c < hid_d; ++c) {
        b1[static_cast<size_t>(c)] = deterministic_weight(seed + 3, c, 0) * 0.1;
    }
    w2.resize(static_cast<size_t>(hid_d * out_d));
    b2.resize(static_cast<size_t>(out_d));
    for (int64_t r = 0; r < hid_d; ++r) {
        for (int64_t c = 0; c < out_d; ++c) {
            w2[static_cast<size_t>(r * out_d + c)] = deterministic_weight(seed + 11, r, c);
        }
    }
    for (int64_t c = 0; c < out_d; ++c) {
        b2[static_cast<size_t>(c)] = deterministic_weight(seed + 19, c, 0) * 0.1;
    }
}

bool OnnxModel::check_valid() const {
    if (output_dim <= 0 || input_dim <= 0 || node_ops.empty()) return false;
#ifdef LMSEARCH_WITH_UPSTREAM_FAISS_ONNX
    onnx::ModelProto model;
    model.set_ir_version(ir_version);
    auto* opset_id = model.add_opset_import();
    opset_id->set_domain("");
    opset_id->set_version(opset);
    auto* graph = model.mutable_graph();
    graph->set_name(is_mlp ? "lmsearch_mlp_embedder" : "lmsearch_text_embedder");

    auto* inp = graph->add_input();
    inp->set_name("input");
    auto* inp_t = inp->mutable_type()->mutable_tensor_type();
    inp_t->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    inp_t->mutable_shape()->add_dim()->set_dim_value(1);
    inp_t->mutable_shape()->add_dim()->set_dim_value(is_mlp ? input_dim : output_dim);

    auto* w_init = graph->add_initializer();
    w_init->set_name("proj_weight");
    w_init->set_data_type(onnx::TensorProto_DataType_FLOAT);
    int64_t r_dim = is_mlp ? input_dim : output_dim;
    int64_t c_dim = output_dim;
    w_init->add_dims(r_dim);
    w_init->add_dims(c_dim);
    for (int64_t i = 0; i < r_dim * c_dim; ++i) {
        w_init->add_float_data(0.1f);
    }

    auto* w_in = graph->add_input();
    w_in->set_name("proj_weight");
    auto* w_t = w_in->mutable_type()->mutable_tensor_type();
    w_t->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    w_t->mutable_shape()->add_dim()->set_dim_value(r_dim);
    w_t->mutable_shape()->add_dim()->set_dim_value(c_dim);

    auto* node = graph->add_node();
    node->set_op_type("MatMul");
    node->set_name("matmul_0");
    node->add_input("input");
    node->add_input("proj_weight");
    node->add_output("hidden");

    auto* relu = graph->add_node();
    relu->set_op_type("Relu");
    relu->set_name("relu_0");
    relu->add_input("hidden");
    relu->add_output("embedding");

    auto* out = graph->add_output();
    out->set_name("embedding");
    auto* out_t = out->mutable_type()->mutable_tensor_type();
    out_t->set_elem_type(onnx::TensorProto_DataType_FLOAT);

    onnx::shape_inference::InferShapes(model);
    onnx::checker::check_model(model);
#endif
    return true;
}

std::string OnnxModel::summary_text() const {
    std::ostringstream oss;
    oss << "onnx(ir=" << ir_version << ",opset=" << opset
        << ",kind=" << (is_mlp ? "mlp" : "text")
        << ",in=" << input_dim << ",out=" << output_dim
        << ",pooling=" << pooling
        << ",norm=" << (normalize ? "true" : "false")
        << ",nodes=" << node_ops.size();
    if (!transform_rule.empty()) oss << ",rule=" << transform_rule;
    oss << ")";
    return oss.str();
}

static std::vector<double> apply_transform_rule_if_any(
    const OnnxModel& model,
    const std::vector<double>& vec) {
    if (model.transform_rule.empty() || !model.host) return vec;
    std::unique_ptr<dslvm::Registry> cb_reg;
    dslvm::Registry* active_reg = nullptr;
    cb_reg = std::make_unique<dslvm::Registry>(dslvm::StdMode::Disabled);
    model.host->bind_all(*cb_reg, true);
    active_reg = cb_reg.get();
    dslvm::stdlib::FloatList local_vec{vec};
    dslvm::Value arg_vec = dslvm::Value::object("FloatList", &local_vec);
    dslvm::Value arg_dim = dslvm::Value::integer(model.output_dim);
    dslvm::Value res = model.host->invoke_rule(
        model.transform_rule,
        "interface EmbeddingTransform(FloatList, int) -> FloatList",
        {arg_vec, arg_dim},
        &active_reg,
        &cb_reg);
    auto out = static_cast<dslvm::stdlib::FloatList*>(res.obj)->items;
    if (static_cast<int64_t>(out.size()) != model.output_dim) {
        throw dslvm::Error("EmbeddingTransform rule '" + model.transform_rule +
                           "' returned length " + std::to_string(out.size()) +
                           "; expected " + std::to_string(model.output_dim));
    }
    return out;
}

std::vector<double> OnnxModel::run_tokens(const std::vector<int64_t>& tokens) const {
    if (tokens.empty()) throw dslvm::Error("embed_tokens requires at least one token");
    size_t d = static_cast<size_t>(output_dim);
    std::vector<double> pooled(d, 0.0);
    if (pooling == "MAX") {
        std::fill(pooled.begin(), pooled.end(), -1e30);
    }
    size_t limit = (pooling == "CLS") ? 1 : tokens.size();
    for (size_t t = 0; t < limit; ++t) {
        int64_t tok = tokens[t] % vocab_size;
        if (tok < 0) tok += vocab_size;
        std::vector<double> emb(d, 0.0);
        for (size_t c = 0; c < d; ++c) {
            double v = w1[static_cast<size_t>(tok) * d + c];
            emb[c] = v;
        }
        // Linear projection + bias + ReLU
        std::vector<double> act(d, 0.0);
        for (size_t c = 0; c < d; ++c) {
            double sum = b1[c];
            for (size_t k = 0; k < d; ++k) {
                sum += emb[k] * w2[k * d + c];
            }
            act[c] = sum > 0.0 ? sum : sum * 0.1; // leaky ReLU to preserve rich signal
        }
        if (pooling == "MAX") {
            for (size_t c = 0; c < d; ++c) pooled[c] = std::max(pooled[c], act[c]);
        } else {
            for (size_t c = 0; c < d; ++c) pooled[c] += act[c];
        }
    }
    if (pooling == "MEAN" && !tokens.empty()) {
        for (size_t c = 0; c < d; ++c) pooled[c] /= static_cast<double>(tokens.size());
    }
    pooled = apply_transform_rule_if_any(*this, pooled);
    if (normalize) pooled = l2_norm_vec(pooled);
    return pooled;
}

std::vector<double> OnnxModel::run_text(const std::string& text) const {
    return run_tokens(tokenize_text_to_ids(text, vocab_size));
}

std::vector<double> OnnxModel::run_vector(const std::vector<double>& input) const {
    if (static_cast<int64_t>(input.size()) != input_dim) {
        throw dslvm::Error("embed_vector input length " + std::to_string(input.size()) +
                           " does not match model input_dim " + std::to_string(input_dim));
    }
    std::vector<double> out(static_cast<size_t>(output_dim), 0.0);
    if (is_mlp) {
        std::vector<double> hid(static_cast<size_t>(hidden_dim), 0.0);
        for (int64_t j = 0; j < hidden_dim; ++j) {
            double s = b1[static_cast<size_t>(j)];
            for (int64_t i = 0; i < input_dim; ++i) {
                s += input[static_cast<size_t>(i)] * w1[static_cast<size_t>(i * hidden_dim + j)];
            }
            hid[static_cast<size_t>(j)] = s > 0.0 ? s : 0.0;
        }
        for (int64_t k = 0; k < output_dim; ++k) {
            double s = b2[static_cast<size_t>(k)];
            for (int64_t j = 0; j < hidden_dim; ++j) {
                s += hid[static_cast<size_t>(j)] * w2[static_cast<size_t>(j * output_dim + k)];
            }
            out[static_cast<size_t>(k)] = s;
        }
    } else {
        for (int64_t c = 0; c < output_dim; ++c) {
            double s = b1[static_cast<size_t>(c)];
            for (int64_t k = 0; k < output_dim; ++k) {
                s += input[static_cast<size_t>(k)] * w2[static_cast<size_t>(k * output_dim + c)];
            }
            out[static_cast<size_t>(c)] = s;
        }
    }
    out = apply_transform_rule_if_any(*this, out);
    if (normalize) out = l2_norm_vec(out);
    return out;
}

void OnnxModel::save_to_file(const std::string& path) const {
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out) throw dslvm::Error("onnx save: cannot open " + path);
    out << "LMSEARCH_ONNX_V1\n";
    out << opset << " " << ir_version << " " << vocab_size << " " << input_dim << " "
        << hidden_dim << " " << output_dim << " " << seed << " " << (is_mlp ? 1 : 0) << " "
        << pooling << " " << (normalize ? 1 : 0) << "\n";
    out << transform_rule << "\n";
    if (!out) throw dslvm::Error("onnx save: write failed for " + path);
}

OnnxModel OnnxModel::load_from_file(Host* host, const std::string& path) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) throw dslvm::Error("onnx_load: cannot open " + path);
    std::string magic;
    std::getline(in, magic);
    if (!magic.empty() && magic.back() == '\r') magic.pop_back();
    if (magic != "LMSEARCH_ONNX_V1") throw dslvm::Error("onnx_load: invalid model file " + path);
    OnnxModel m;
    m.host = host;
    int mlp_flag = 0, norm_flag = 1;
    in >> m.opset >> m.ir_version >> m.vocab_size >> m.input_dim >> m.hidden_dim >> m.output_dim >> m.seed >> mlp_flag >> m.pooling >> norm_flag;
    std::string dummy;
    std::getline(in, dummy);
    std::getline(in, m.transform_rule);
    if (!m.transform_rule.empty() && m.transform_rule.back() == '\r') m.transform_rule.pop_back();
    if (mlp_flag) {
        m.init_mlp_embedder(m.input_dim, m.hidden_dim, m.output_dim, m.seed);
    } else {
        m.init_text_embedder(m.vocab_size, m.output_dim, m.seed);
    }
    m.pooling = m.pooling.empty() ? "MEAN" : m.pooling;
    m.normalize = (norm_flag != 0);
    return m;
}

static std::string resolve_model_cache_root(const std::string& custom_cache_dir) {
    if (!custom_cache_dir.empty()) return custom_cache_dir;
    if (const char* env_dir = std::getenv("LMSEARCH_CACHE_DIR")) {
        if (*env_dir) return std::string(env_dir);
    }
#ifdef _WIN32
    if (const char* local_app = std::getenv("LOCALAPPDATA")) {
        if (*local_app) return std::string(local_app) + "/lmsearch/models";
    }
    if (const char* up = std::getenv("USERPROFILE")) {
        if (*up) return std::string(up) + "/.cache/lmsearch/models";
    }
#else
    if (const char* home = std::getenv("HOME")) {
        if (*home) return std::string(home) + "/.cache/lmsearch/models";
    }
#endif
    return "build/.cache/lmsearch/models";
}

static std::string sanitize_model_slug(const std::string& id) {
    std::string out;
    for (unsigned char c : id) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "default_model" : out;
}

OnnxModel OnnxModel::load_or_download_cached(
    Host* host,
    const std::string& model_id_or_url,
    int64_t requested_dim,
    const std::string& custom_cache_dir) {
    if (model_id_or_url.empty()) {
        throw dslvm::Error("onnx_pretrained requires a non-empty model name or URL");
    }

    int64_t dim = requested_dim > 0 ? requested_dim : 16;
    if (requested_dim <= 0) {
        if (model_id_or_url.find("384") != std::string::npos) dim = 24;
        else if (model_id_or_url.find("large") != std::string::npos) dim = 32;
        else if (model_id_or_url.find("small") != std::string::npos || model_id_or_url.find("MiniLM") != std::string::npos) dim = 16;
    }

    std::string cache_root = resolve_model_cache_root(custom_cache_dir);
    std::string slug = sanitize_model_slug(model_id_or_url) + "_d" + std::to_string(dim);
    std::filesystem::path model_dir = std::filesystem::u8path(cache_root) / slug;
    std::filesystem::create_directories(model_dir);
    std::filesystem::path cached_onnx = model_dir / "model.onnx";
    std::string cached_path_str = cached_onnx.generic_string();

    if (std::filesystem::exists(cached_onnx) && std::filesystem::file_size(cached_onnx) > 0) {
        try {
            OnnxModel m = load_from_file(host, cached_path_str);
            m.model_id = model_id_or_url;
            m.cache_path = cached_path_str;
            m.from_cache = true;
            return m;
        } catch (...) {
            // Re-materialize if corrupt
        }
    }

    // Check if model_id_or_url is an existing local ONNX file
    std::filesystem::path local_candidate = std::filesystem::u8path(model_id_or_url);
    if (std::filesystem::exists(local_candidate) && std::filesystem::is_regular_file(local_candidate)) {
        OnnxModel m = load_from_file(host, model_id_or_url);
        m.model_id = model_id_or_url;
        m.save_to_file(cached_path_str);
        m.cache_path = cached_path_str;
        m.from_cache = true;
        return m;
    }

    //Deterministic seed from model_id_or_url so weights for a given model name are consistent across runs
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : model_id_or_url) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    int64_t seed = static_cast<int64_t>((h % 999983ULL) + 1ULL);

    OnnxModel m;
    m.host = host;
    m.model_id = model_id_or_url;
    m.init_text_embedder(4096, dim, seed);
    m.pooling = "MEAN";
    m.normalize = true;
    m.save_to_file(cached_path_str);
    m.cache_path = cached_path_str;
    m.from_cache = true;
    return m;
}

// ============================================================================
// AutoFaiss C++ Engine (AutoTuner & AutoPlan)
// ============================================================================

static std::vector<int64_t> optimal_nb_clusters(int64_t nb_vectors) {
    std::vector<int64_t> out;
    if (nb_vectors < 1000000) {
        double x_init = 4.0 * std::sqrt(static_cast<double>(std::max<int64_t>(1, nb_vectors)));
        int64_t p2 = 1LL << static_cast<int64_t>(std::llround(std::log2(std::max(1.0, x_init))));
        p2 = std::max<int64_t>(4, p2);
        out.push_back(p2);
        out.push_back(2 * p2);
        out.push_back(std::max<int64_t>(4, static_cast<int64_t>(std::llround(x_init))));
    } else if (nb_vectors < 10000000) {
        out.push_back(16384);
        out.push_back(65536);
    } else {
        out.push_back(65536);
        out.push_back(131072);
    }
    return out;
}

static AutoPlan compute_plan_for_key(
    const std::string& key,
    int64_t nb_vectors,
    int64_t dim,
    const std::string& metric,
    int64_t max_mem_bytes,
    bool direct_map,
    double max_query_ms) {
    AutoPlan p;
    p.index_key = key;
    p.metric = metric;
    p.nb_vectors = nb_vectors;
    p.dim = dim;
    p.max_memory_bytes = max_mem_bytes;
    auto nums = extract_ints(key);

    if (key == "Flat") {
        p.index_type = "FLAT";
        p.nb_clusters = 1;
        p.train_size = 0;
        p.estimated_bytes = nb_vectors * dim * 4;
        p.training_memory_bytes = 0;
        p.recommended_nprobe = 1;
        p.recommended_ef_search = 0;
        p.recommended_params = "";
        p.description = "Exact exhaustive search index (Flat) with zero compression.";
    } else if (key.find("HNSW") == 0) {
        p.index_type = "HNSW";
        int64_t m = nums.empty() ? 32 : nums[0];
        p.nb_clusters = 1;
        p.train_size = 0;
        p.estimated_bytes = nb_vectors * dim * 4 + nb_vectors * m * 2 * 4;
        p.training_memory_bytes = 0;
        p.recommended_nprobe = 1;
        p.recommended_ef_search = std::max<int64_t>(16, std::min<int64_t>(128, static_cast<int64_t>(max_query_ms * 6.0)));
        p.recommended_params = "efSearch=" + std::to_string(p.recommended_ef_search);
        p.description = "Hierarchical Navigable Small World (HNSW) graph index.";
    } else if (key.find("OPQ") == 0 && key.find("IVF") != std::string::npos && key.find("PQ") != std::string::npos) {
        bool has_hnsw = key.find("_HNSW") != std::string::npos;
        p.index_type = has_hnsw ? "OPQ_IVF_HNSW_PQ" : "OPQ_IVF_PQ";
        int64_t m_opq = nums.size() > 0 ? nums[0] : 8;
        int64_t out_d = nums.size() > 1 ? nums[1] : dim;
        int64_t ncent = nums.size() > 2 ? nums[2] : 64;
        int64_t pq_m = has_hnsw ? (nums.size() > 4 ? nums[4] : m_opq) : (nums.size() > 3 ? nums[3] : m_opq);
        int64_t m_hnsw = has_hnsw && nums.size() > 3 ? nums[3] : 32;
        p.nb_clusters = ncent;
        p.train_size = std::min<int64_t>(nb_vectors, ncent * 64);
        int64_t dm_overhead = direct_map ? 8 * nb_vectors : 0;
        int64_t vec_bytes = nb_vectors * (pq_m + 8);
        int64_t cent_bytes = ncent * dim * 4;
        int64_t opq_bytes = m_opq * out_d * 4;
        int64_t hnsw_bytes = has_hnsw ? ncent * m_hnsw * 2 * 4 : 0;
        p.estimated_bytes = dm_overhead + vec_bytes + cent_bytes + opq_bytes + hnsw_bytes;
        p.training_memory_bytes = p.train_size * dim * 6 + cent_bytes + opq_bytes;
        p.recommended_nprobe = std::max<int64_t>(1, std::min<int64_t>(ncent, static_cast<int64_t>(std::ceil(ncent * 0.125))));
        p.recommended_ef_search = has_hnsw ? p.recommended_nprobe * 2 : 0;
        p.recommended_params = has_hnsw
            ? ("nprobe=" + std::to_string(p.recommended_nprobe) + ",efSearch=" + std::to_string(p.recommended_ef_search) + ",ht=2048")
            : ("nprobe=" + std::to_string(p.recommended_nprobe) + ",ht=2048");
        p.description = "Inverted file index with Optimized Product Quantization (OPQ+IVF+PQ).";
    } else if (key.find("IVF") == 0 && key.find(",Flat") != std::string::npos) {
        p.index_type = "IVF_FLAT";
        int64_t ncent = nums.empty() ? 32 : nums[0];
        p.nb_clusters = ncent;
        p.train_size = std::min<int64_t>(nb_vectors, ncent * 64);
        int64_t dm_overhead = direct_map ? 8 * nb_vectors : 0;
        p.estimated_bytes = dm_overhead + nb_vectors * dim * 4 + ncent * dim * 4;
        p.training_memory_bytes = p.train_size * dim * 6 + ncent * dim * 4;
        p.recommended_nprobe = std::max<int64_t>(1, std::min<int64_t>(ncent, static_cast<int64_t>(std::ceil(ncent * 0.15))));
        p.recommended_ef_search = 0;
        p.recommended_params = "nprobe=" + std::to_string(p.recommended_nprobe);
        p.description = "Inverted file index with exact flat cluster storage (IVF,Flat).";
    } else if (key.find("IVF") == 0 && key.find("PQ") != std::string::npos) {
        p.index_type = "IVF_PQ";
        int64_t ncent = nums.size() > 0 ? nums[0] : 32;
        int64_t pq_m = nums.size() > 1 ? nums[1] : std::max<int64_t>(4, dim / 2);
        p.nb_clusters = ncent;
        p.train_size = std::min<int64_t>(nb_vectors, ncent * 64);
        int64_t dm_overhead = direct_map ? 8 * nb_vectors : 0;
        p.estimated_bytes = dm_overhead + nb_vectors * (pq_m + 8) + ncent * dim * 4;
        p.training_memory_bytes = p.train_size * dim * 6 + ncent * dim * 4;
        p.recommended_nprobe = std::max<int64_t>(1, std::min<int64_t>(ncent, static_cast<int64_t>(std::ceil(ncent * 0.2))));
        p.recommended_ef_search = 0;
        p.recommended_params = "nprobe=" + std::to_string(p.recommended_nprobe) + ",ht=2048";
        p.description = "Inverted file index with Product Quantization (IVF,PQ).";
    } else {
        p.index_type = "CUSTOM";
        p.nb_clusters = nums.empty() ? 1 : nums[0];
        p.train_size = nb_vectors;
        p.estimated_bytes = nb_vectors * dim * 4;
        p.training_memory_bytes = nb_vectors * dim * 4;
        p.recommended_params = "";
        p.description = "Custom FAISS index key: " + key;
    }
    double raw_bytes = static_cast<double>(nb_vectors * dim * 4);
    p.compression_ratio = p.estimated_bytes > 0 ? (raw_bytes / static_cast<double>(p.estimated_bytes)) : 1.0;
    return p;
}

std::vector<std::string> AutoTuner::compute_candidates() const {
    if (nb_vectors <= 0 || dim <= 0) throw dslvm::Error("autotuner requires positive nb_vectors and dim");
    int64_t max_bytes = parse_memory_bytes(max_memory);
    std::vector<std::string> keys;

    if (force_pq > 0) {
        int64_t pq = force_pq;
        auto clusters = optimal_nb_clusters(nb_vectors);
        for (int64_t c : clusters) {
            int64_t y = std::max<int64_t>(dim, pq * ((dim + pq - 1) / pq));
            std::string cl = c < 1000 ? ("IVF" + std::to_string(c)) : ("IVF" + std::to_string(c) + "_HNSW32");
            keys.push_back("OPQ" + std::to_string(pq) + "_" + std::to_string(y) + "," + cl + ",PQ" + std::to_string(pq) + "x8");
        }
        return keys;
    }

    if (nb_vectors < 1000) {
        keys.push_back(memory_mappable ? "IVF1,Flat" : "Flat");
        return keys;
    }
    if (nb_vectors < 10000) {
        if (memory_mappable) {
            for (int64_t c : optimal_nb_clusters(nb_vectors)) {
                keys.push_back("IVF" + std::to_string(c) + ",Flat");
            }
            return keys;
        }
        keys.push_back("HNSW15");
        for (int64_t c : optimal_nb_clusters(nb_vectors)) {
            keys.push_back("IVF" + std::to_string(c) + ",Flat");
        }
        return keys;
    }

    if (!memory_mappable) {
        int64_t m_hnsw = static_cast<int64_t>(std::floor((static_cast<double>(max_bytes) / (4.0 * nb_vectors) - dim) / 2.0));
        if (m_hnsw >= 8) {
            keys.push_back("HNSW" + std::to_string(std::min<int64_t>(m_hnsw, 32)));
        }
    }
    for (int64_t c : optimal_nb_clusters(nb_vectors)) {
        std::string ivf_key = "IVF" + std::to_string(c) + ",Flat";
        auto meta = compute_plan_for_key(ivf_key, nb_vectors, dim, metric, max_bytes, direct_map, max_query_ms);
        if (meta.estimated_bytes <= max_bytes) {
            keys.push_back(ivf_key);
        }
    }
    std::vector<int64_t> pq_vals = {64, 32, 16, 8, 4};
    for (int64_t pq : pq_vals) {
        if (pq < dim) {
            for (int64_t c : optimal_nb_clusters(nb_vectors)) {
                int64_t y = std::max<int64_t>(dim, pq * ((dim + pq - 1) / pq));
                std::string cl = c < 1000 ? ("IVF" + std::to_string(c)) : ("IVF" + std::to_string(c) + "_HNSW32");
                std::string k = "OPQ" + std::to_string(pq) + "_" + std::to_string(y) + "," + cl + ",PQ" + std::to_string(pq) + "x8";
                auto meta = compute_plan_for_key(k, nb_vectors, dim, metric, max_bytes, direct_map, max_query_ms);
                if (meta.estimated_bytes <= max_bytes || keys.empty()) {
                    keys.push_back(k);
                }
            }
        }
    }
    if (keys.empty()) keys.push_back("Flat");
    return keys;
}

AutoPlan AutoTuner::build_plan() const {
    auto candidates = compute_candidates();
    int64_t max_bytes = parse_memory_bytes(max_memory);
    AutoPlan base = compute_plan_for_key(candidates.front(), nb_vectors, dim, metric, max_bytes, direct_map, max_query_ms);
    if (!policy_rule.empty() && host) {
        AutoPlan* plan_handle = host->alloc_plan(base);
        dslvm::Value res = host->invoke_rule(
            policy_rule,
            "interface IndexPolicy(AutoPlan) -> string",
            {dslvm::Value::object("AutoPlan", plan_handle)});
        if (res.s.empty()) {
            throw dslvm::Error("IndexPolicy rule '" + policy_rule + "' returned an empty index_key");
        }
        base = compute_plan_for_key(res.s, nb_vectors, dim, metric, max_bytes, direct_map, max_query_ms);
    }
    return base;
}

// ============================================================================
// FAISS Index Engine + SAM Extensions
// ============================================================================

void FaissIndex::parse_key() {
    if (dim <= 0) throw dslvm::Error("faiss_index requires positive dim");
    auto nums = extract_ints(index_key);
    nlist = 1;
    m_hnsw = 32;
    m_pq = 0;
    nbits_pq = 8;
    opq_m = 0;
    opq_d = dim;
    is_trained = true;

    if (index_key == "Flat") {
        is_trained = true;
    } else if (index_key.find("HNSW") == 0) {
        m_hnsw = nums.empty() ? 32 : nums[0];
        if (m_hnsw <= 0) throw dslvm::Error("invalid HNSW M parameter in " + index_key);
        is_trained = true;
        param_str = "efSearch=" + std::to_string(cur_ef_search);
    } else if (index_key.find("OPQ") == 0 && index_key.find("IVF") != std::string::npos && index_key.find("PQ") != std::string::npos) {
        bool has_hnsw = index_key.find("_HNSW") != std::string::npos;
        opq_m = nums.size() > 0 ? nums[0] : 4;
        opq_d = nums.size() > 1 ? nums[1] : dim;
        nlist = nums.size() > 2 ? nums[2] : 4;
        m_pq = has_hnsw ? (nums.size() > 4 ? nums[4] : opq_m) : (nums.size() > 3 ? nums[3] : opq_m);
        if (m_pq <= 0 || dim % m_pq != 0) m_pq = std::gcd(dim, std::max<int64_t>(1, m_pq));
        is_trained = false;
        param_str = "nprobe=" + std::to_string(cur_nprobe);
    } else if (index_key.find("IVF") == 0 && index_key.find(",Flat") != std::string::npos) {
        nlist = nums.empty() ? 4 : nums[0];
        if (nlist <= 0) throw dslvm::Error("invalid IVF nlist in " + index_key);
        is_trained = false;
        param_str = "nprobe=" + std::to_string(cur_nprobe);
    } else if (index_key.find("IVF") == 0 && index_key.find("PQ") != std::string::npos) {
        nlist = nums.size() > 0 ? nums[0] : 4;
        m_pq = nums.size() > 1 ? nums[1] : std::max<int64_t>(1, dim / 2);
        nbits_pq = nums.size() > 2 ? nums[2] : 8;
        if (nlist <= 0 || m_pq <= 0 || dim % m_pq != 0) {
            throw dslvm::Error("invalid IVF,PQ parameters in " + index_key + " for dim " + std::to_string(dim));
        }
        is_trained = false;
        param_str = "nprobe=" + std::to_string(cur_nprobe);
    } else {
        throw dslvm::Error("unsupported FAISS index factory key: " + index_key);
    }

#ifdef LMSEARCH_WITH_UPSTREAM_FAISS_ONNX
    // Also validate non-OPQ simple keys through upstream faiss::index_factory
    if (index_key == "Flat" || index_key.find("HNSW") == 0 ||
        (index_key.find("IVF") == 0 && index_key.find(",Flat") != std::string::npos)) {
        faiss::MetricType mt = (metric == "L2") ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        std::unique_ptr<faiss::Index> test_idx(faiss::index_factory(static_cast<int>(dim), index_key.c_str(), mt));
        (void)test_idx;
    }
#endif
}

void FaissIndex::apply_param_string(const std::string& s) {
    param_str = s;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        auto eq = item.find('=');
        if (eq == std::string::npos) throw dslvm::Error("invalid parameter assignment: " + item);
        std::string key = item.substr(0, eq);
        std::string val_str = item.substr(eq + 1);
        int64_t val = 0;
        try {
            val = std::stoll(val_str);
        } catch (...) {
            throw dslvm::Error("invalid numeric parameter in: " + item);
        }
        if (val <= 0) throw dslvm::Error("parameter value must be positive: " + item);
        if (key == "nprobe") cur_nprobe = val;
        else if (key == "efSearch") cur_ef_search = val;
        else if (key == "efConstruction") cur_ef_construction = val;
        else if (key == "ht") cur_ht = val;
        else throw dslvm::Error("unknown index parameter: " + key);
    }
}

std::string FaissIndex::effective_params() const {
    if (!param_str.empty()) return param_str;
    if (index_key.find("HNSW") == 0) return "efSearch=" + std::to_string(cur_ef_search);
    if (index_key.find("IVF") != std::string::npos) return "nprobe=" + std::to_string(cur_nprobe);
    return "";
}

void FaissIndex::train_index() {
    if (entries.empty()) {
        // Initialize deterministic default centroids if trained before adding vectors
        centroids.assign(static_cast<size_t>(nlist), std::vector<double>(static_cast<size_t>(dim), 0.0));
        for (int64_t c = 0; c < nlist; ++c) {
            for (int64_t j = 0; j < dim; ++j) {
                centroids[static_cast<size_t>(c)][static_cast<size_t>(j)] = deterministic_weight(101, c, j);
            }
            if (metric == "COSINE") centroids[static_cast<size_t>(c)] = l2_norm_vec(centroids[static_cast<size_t>(c)]);
        }
        if (m_pq > 0) {
            int64_t dsub = dim / m_pq;
            int64_t ksub = 16;
            pq_centroids.assign(static_cast<size_t>(m_pq),
                std::vector<std::vector<double>>(static_cast<size_t>(ksub), std::vector<double>(static_cast<size_t>(dsub), 0.0)));
            for (int64_t m = 0; m < m_pq; ++m) {
                for (int64_t k = 0; k < ksub; ++k) {
                    for (int64_t j = 0; j < dsub; ++j) {
                        pq_centroids[static_cast<size_t>(m)][static_cast<size_t>(k)][static_cast<size_t>(j)] =
                            deterministic_weight(202 + m, k, j);
                    }
                }
            }
        }
        is_trained = true;
        return;
    }

    // K-means training on current entries for IVF coarse centroids
    int64_t actual_nlist = std::min<int64_t>(nlist, static_cast<int64_t>(entries.size()));
    centroids.assign(static_cast<size_t>(nlist), std::vector<double>(static_cast<size_t>(dim), 0.0));
    for (int64_t c = 0; c < nlist; ++c) {
        const auto& src = entries[static_cast<size_t>(c % actual_nlist)].norm_vec;
        centroids[static_cast<size_t>(c)] = src;
    }
    // 5 iterations of Lloyd's k-means
    for (int iter = 0; iter < 5; ++iter) {
        std::vector<std::vector<double>> sums(static_cast<size_t>(nlist), std::vector<double>(static_cast<size_t>(dim), 0.0));
        std::vector<int64_t> counts(static_cast<size_t>(nlist), 0);
        for (auto& e : entries) {
            int64_t best_c = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < nlist; ++c) {
                double d = vec_l2(e.norm_vec, centroids[static_cast<size_t>(c)]);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            e.list_id = best_c;
            counts[static_cast<size_t>(best_c)]++;
            for (int64_t j = 0; j < dim; ++j) {
                sums[static_cast<size_t>(best_c)][static_cast<size_t>(j)] += e.norm_vec[static_cast<size_t>(j)];
            }
        }
        for (int64_t c = 0; c < nlist; ++c) {
            if (counts[static_cast<size_t>(c)] > 0) {
                for (int64_t j = 0; j < dim; ++j) {
                    centroids[static_cast<size_t>(c)][static_cast<size_t>(j)] =
                        sums[static_cast<size_t>(c)][static_cast<size_t>(j)] / static_cast<double>(counts[static_cast<size_t>(c)]);
                }
                if (metric == "COSINE") {
                    centroids[static_cast<size_t>(c)] = l2_norm_vec(centroids[static_cast<size_t>(c)]);
                }
            }
        }
    }

    // Train Product Quantizer sub-codebooks if PQ is enabled
    if (m_pq > 0) {
        int64_t dsub = dim / m_pq;
        int64_t ksub = std::min<int64_t>(32, std::max<int64_t>(4, static_cast<int64_t>(entries.size())));
        pq_centroids.assign(static_cast<size_t>(m_pq),
            std::vector<std::vector<double>>(static_cast<size_t>(ksub), std::vector<double>(static_cast<size_t>(dsub), 0.0)));
        for (int64_t m = 0; m < m_pq; ++m) {
            for (int64_t k = 0; k < ksub; ++k) {
                const auto& src = entries[static_cast<size_t>(k % entries.size())].norm_vec;
                for (int64_t j = 0; j < dsub; ++j) {
                    pq_centroids[static_cast<size_t>(m)][static_cast<size_t>(k)][static_cast<size_t>(j)] =
                        src[static_cast<size_t>(m * dsub + j)];
                }
            }
            // Refine sub-centroids
            for (int iter = 0; iter < 3; ++iter) {
                std::vector<std::vector<double>> sums(static_cast<size_t>(ksub), std::vector<double>(static_cast<size_t>(dsub), 0.0));
                std::vector<int64_t> counts(static_cast<size_t>(ksub), 0);
                for (auto& e : entries) {
                    int64_t best_k = 0;
                    double best_d = std::numeric_limits<double>::infinity();
                    for (int64_t k = 0; k < ksub; ++k) {
                        double d = 0.0;
                        for (int64_t j = 0; j < dsub; ++j) {
                            double diff = e.norm_vec[static_cast<size_t>(m * dsub + j)] -
                                          pq_centroids[static_cast<size_t>(m)][static_cast<size_t>(k)][static_cast<size_t>(j)];
                            d += diff * diff;
                        }
                        if (d < best_d) {
                            best_d = d;
                            best_k = k;
                        }
                    }
                    counts[static_cast<size_t>(best_k)]++;
                    for (int64_t j = 0; j < dsub; ++j) {
                        sums[static_cast<size_t>(best_k)][static_cast<size_t>(j)] += e.norm_vec[static_cast<size_t>(m * dsub + j)];
                    }
                }
                for (int64_t k = 0; k < ksub; ++k) {
                    if (counts[static_cast<size_t>(k)] > 0) {
                        for (int64_t j = 0; j < dsub; ++j) {
                            pq_centroids[static_cast<size_t>(m)][static_cast<size_t>(k)][static_cast<size_t>(j)] =
                                sums[static_cast<size_t>(k)][static_cast<size_t>(j)] / static_cast<double>(counts[static_cast<size_t>(k)]);
                        }
                    }
                }
            }
        }
        // Encode PQ codes for all entries
        for (auto& e : entries) {
            e.pq_codes.resize(static_cast<size_t>(m_pq));
            for (int64_t m = 0; m < m_pq; ++m) {
                int64_t best_k = 0;
                double best_d = std::numeric_limits<double>::infinity();
                for (size_t k = 0; k < pq_centroids[static_cast<size_t>(m)].size(); ++k) {
                    double d = 0.0;
                    for (int64_t j = 0; j < dsub; ++j) {
                        double diff = e.norm_vec[static_cast<size_t>(m * dsub + j)] -
                                      pq_centroids[static_cast<size_t>(m)][k][static_cast<size_t>(j)];
                        d += diff * diff;
                    }
                    if (d < best_d) {
                        best_d = d;
                        best_k = static_cast<int64_t>(k);
                    }
                }
                e.pq_codes[static_cast<size_t>(m)] = static_cast<uint8_t>(best_k);
            }
        }
    }
    is_trained = true;
}

void FaissIndex::add_vector(int64_t id, const std::vector<double>& vec, Document* doc) {
    if (static_cast<int64_t>(vec.size()) != dim) {
        throw dslvm::Error("add vector dimension mismatch: got " + std::to_string(vec.size()) +
                           ", expected " + std::to_string(dim));
    }
    Entry e;
    e.id = id;
    e.raw_vec = vec;
    e.norm_vec = (metric == "COSINE") ? l2_norm_vec(vec) : vec;
    e.doc = doc ? doc : host->alloc_doc(id, "");
    if (e.doc->vec.empty()) e.doc->vec = vec;

    if (!centroids.empty()) {
        int64_t best_c = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < centroids.size(); ++c) {
            double d = vec_l2(e.norm_vec, centroids[c]);
            if (d < best_d) {
                best_d = d;
                best_c = static_cast<int64_t>(c);
            }
        }
        e.list_id = best_c;
    }
    if (m_pq > 0 && !pq_centroids.empty()) {
        int64_t dsub = dim / m_pq;
        e.pq_codes.resize(static_cast<size_t>(m_pq));
        for (int64_t m = 0; m < m_pq; ++m) {
            int64_t best_k = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < pq_centroids[static_cast<size_t>(m)].size(); ++k) {
                double d = 0.0;
                for (int64_t j = 0; j < dsub; ++j) {
                    double diff = e.norm_vec[static_cast<size_t>(m * dsub + j)] -
                                  pq_centroids[static_cast<size_t>(m)][k][static_cast<size_t>(j)];
                    d += diff * diff;
                }
                if (d < best_d) {
                    best_d = d;
                    best_k = static_cast<int64_t>(k);
                }
            }
            e.pq_codes[static_cast<size_t>(m)] = static_cast<uint8_t>(best_k);
        }
    }
    entries.push_back(std::move(e));
}

int64_t FaissIndex::remove_by_id(int64_t id) {
    size_t before = entries.size();
    entries.erase(
        std::remove_if(entries.begin(), entries.end(), [id](const Entry& e) { return e.id == id; }),
        entries.end());
    return static_cast<int64_t>(before - entries.size());
}

static bool has_allowed_extension(const std::filesystem::path& p, const std::string& ext_filter) {
    if (ext_filter.empty() || ext_filter == "*") return true;
    std::string ext = p.extension().generic_string();
    std::stringstream ss(ext_filter);
    std::string token;
    while (std::getline(ss, token, ',')) {
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) token.erase(token.begin());
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.pop_back();
        if (token.empty() || token == "*") return true;
        if (token[0] != '.') token = "." + token;
        if (ext == token) return true;
    }
    return false;
}

void SearchFilterOpts::clear() {
    ext_filter.clear();
    path_prefix.clear();
    path_contains.clear();
    exclude_path.clear();
    text_contains.clear();
    required_fields.clear();
    min_score = -1e30;
    max_distance = 1e30;
    min_size = -1;
    max_size = -1;
    respect_gitignore = true;
    include_hidden = false;
    ignore_patterns.clear();
}

void SearchFilterOpts::load_ignore_file(const std::string& path) {
    std::ifstream in(std::filesystem::u8path(path));
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        size_t first = 0;
        while (first < line.size() && (line[first] == ' ' || line[first] == '\t')) ++first;
        line = line.substr(first);
        if (line.empty() || line[0] == '#') continue;
        ignore_patterns.push_back(line);
    }
}

bool SearchFilterOpts::matches_doc(const Document* doc) const {
    bool has_doc_constraint =
        !ext_filter.empty() || !path_prefix.empty() || !path_contains.empty() ||
        !exclude_path.empty() || !text_contains.empty() || !required_fields.empty() ||
        min_size >= 0 || max_size >= 0;
    if (!has_doc_constraint) return true;
    if (!doc) return false;

    std::string dpath;
    auto it_p = doc->fields.find("path");
    if (it_p != doc->fields.end()) dpath = it_p->second;

    if (!ext_filter.empty() && ext_filter != "*") {
        if (!has_allowed_extension(std::filesystem::u8path(dpath), ext_filter)) return false;
    }
    if (!path_prefix.empty()) {
        if (dpath.compare(0, path_prefix.size(), path_prefix) != 0) return false;
    }
    if (!path_contains.empty()) {
        if (dpath.find(path_contains) == std::string::npos) return false;
    }
    if (!exclude_path.empty()) {
        std::stringstream ss(exclude_path);
        std::string pat;
        while (std::getline(ss, pat, ',')) {
            if (!pat.empty() && dpath.find(pat) != std::string::npos) return false;
        }
    }
    if (!text_contains.empty()) {
        if (doc->text.find(text_contains) == std::string::npos) return false;
    }
    for (const auto& kv : required_fields) {
        auto it = doc->fields.find(kv.first);
        if (it == doc->fields.end() || it->second != kv.second) return false;
    }
    if (min_size >= 0 || max_size >= 0) {
        auto it_s = doc->numbers.find("size");
        int64_t sz = (it_s != doc->numbers.end()) ? it_s->second : static_cast<int64_t>(doc->text.size());
        if (min_size >= 0 && sz < min_size) return false;
        if (max_size >= 0 && sz > max_size) return false;
    }
    return true;
}

static bool is_skipped_dir_name(const std::string& name) {
    return name == ".git" || name == ".hg" || name == ".svn" ||
           name == "build" || name == "dist" || name == "target" || name == "out" ||
           name == "node_modules" || name == ".venv" || name == "venv" ||
           name == "third_party" || name == "__pycache__" || name == "CMakeFiles" ||
           name == ".cache" || name == ".npm" || name == ".cargo" || name == ".rustup" ||
           name == ".nvm" || name == ".gradle";
}

static bool is_binary_or_artifact_ext(const std::string& ext) {
    return ext == ".onnx" || ext == ".faiss" || ext == ".dslb" ||
           ext == ".o" || ext == ".obj" || ext == ".a" || ext == ".lib" ||
           ext == ".so" || ext == ".dylib" || ext == ".dll" || ext == ".exe" ||
           ext == ".bin" || ext == ".dat" || ext == ".zip" || ext == ".tar" ||
           ext == ".gz" || ext == ".xz" || ext == ".7z" || ext == ".png" ||
           ext == ".jpg" || ext == ".jpeg" || ext == ".gif" || ext == ".webp" ||
           ext == ".mp4" || ext == ".mp3" || ext == ".wav" || ext == ".pdf" ||
           ext == ".pyc" || ext == ".class" || ext == ".jar" || ext == ".wasm";
}

static bool glob_match_str(const char* pat, const char* str) {
    while (*pat) {
        if (*pat == '*') {
            if (pat[1] == '*') {
                while (*pat == '*') ++pat;
                if (*pat == '/') ++pat;
                if (!*pat) return true;
                while (*str) {
                    if (glob_match_str(pat, str)) return true;
                    ++str;
                }
                return glob_match_str(pat, str);
            }
            ++pat;
            if (!*pat) return (*str == '\0' || std::strchr(str, '/') == nullptr);
            while (*str && *str != '/') {
                if (glob_match_str(pat, str)) return true;
                ++str;
            }
            return glob_match_str(pat, str);
        } else if (*pat == '?') {
            if (!*str || *str == '/') return false;
            ++pat;
            ++str;
        } else {
            if (*pat != *str) return false;
            ++pat;
            ++str;
        }
    }
    return *str == '\0';
}

static bool matches_single_gitignore_rule(
    std::string rule,
    const std::string& rel_path,
    const std::string& base_name,
    bool is_dir) {
    if (rule.empty() || rule[0] == '#') return false;
    bool dir_only = false;
    if (rule.size() > 1 && rule.back() == '/') {
        dir_only = true;
        rule.pop_back();
    }
    bool anchored = false;
    if (!rule.empty() && rule.front() == '/') {
        anchored = true;
        rule.erase(rule.begin());
    }
    if (rule.empty()) return false;

    if (dir_only && !is_dir) {
        std::string prefix = rule + "/";
        if (rel_path.compare(0, prefix.size(), prefix) == 0) return true;
        if (!anchored && rel_path.find("/" + prefix) != std::string::npos) return true;
        return false;
    }

    bool has_slash = (rule.find('/') != std::string::npos);
    if (!anchored && !has_slash) {
        if (glob_match_str(rule.c_str(), base_name.c_str())) return true;
    }
    if (glob_match_str(rule.c_str(), rel_path.c_str())) return true;
    if (!anchored && glob_match_str(("**/" + rule).c_str(), rel_path.c_str())) return true;
    return false;
}

static bool matches_ignore_list(
    const std::vector<std::string>& patterns,
    const std::string& rel_path,
    const std::string& base_name,
    bool is_dir) {
    bool ignored = false;
    for (const auto& raw : patterns) {
        if (raw.empty() || raw[0] == '#') continue;
        if (raw[0] == '!') {
            std::string neg = raw.substr(1);
            if (matches_single_gitignore_rule(neg, rel_path, base_name, is_dir)) {
                ignored = false;
            }
        } else {
            if (matches_single_gitignore_rule(raw, rel_path, base_name, is_dir)) {
                ignored = true;
            }
        }
    }
    return ignored;
}

static bool is_path_in_git_repo(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::path cur = std::filesystem::absolute(p, ec);
    if (ec) cur = p;
    if (!std::filesystem::is_directory(cur, ec)) {
        cur = cur.parent_path();
    }
    for (int depth = 0; depth < 16 && !cur.empty(); ++depth) {
        if (std::filesystem::exists(cur / ".git", ec)) return true;
        auto parent = cur.parent_path();
        if (parent == cur) break;
        cur = parent;
    }
    return false;
}

static void collect_dir_ignore_files(const std::filesystem::path& dir, std::vector<std::string>& out_patterns) {
    static const char* kIgnoreFiles[] = {".gitignore", ".ignore", ".lmsearchignore"};
    std::error_code ec;
    for (const char* fname : kIgnoreFiles) {
        std::filesystem::path f = dir / fname;
        if (std::filesystem::is_regular_file(f, ec)) {
            SearchFilterOpts tmp;
            tmp.load_ignore_file(f.generic_string());
            for (auto& pat : tmp.ignore_patterns) {
                out_patterns.push_back(std::move(pat));
            }
        }
    }
}

int64_t FaissIndex::add_file_path(const std::string& file_path) {
    std::filesystem::path fpath = std::filesystem::u8path(file_path);
    if (!std::filesystem::exists(fpath) || !std::filesystem::is_regular_file(fpath)) {
        throw dslvm::Error("add_file: file does not exist: " + file_path);
    }
    std::string fname = fpath.filename().generic_string();
    std::string ext = fpath.extension().generic_string();
    if (!filter_opts.include_hidden && !fname.empty() && fname[0] == '.') {
        return 0;
    }
    if (!filter_opts.ext_filter.empty()) {
        if (!has_allowed_extension(fpath, filter_opts.ext_filter)) return 0;
    } else if (is_binary_or_artifact_ext(ext)) {
        return 0;
    }
    std::string norm_path = fpath.generic_string();
    if (!filter_opts.exclude_path.empty()) {
        std::stringstream ss(filter_opts.exclude_path);
        std::string pat;
        while (std::getline(ss, pat, ',')) {
            if (!pat.empty() && norm_path.find(pat) != std::string::npos) return 0;
        }
    }
    if (!filter_opts.ignore_patterns.empty()) {
        if (matches_ignore_list(filter_opts.ignore_patterns, norm_path, fname, false)) {
            return 0;
        }
    }
    int64_t fsize = static_cast<int64_t>(std::filesystem::file_size(fpath));
    if (filter_opts.min_size >= 0 && fsize < filter_opts.min_size) return 0;
    if (filter_opts.max_size >= 0 && fsize > filter_opts.max_size) return 0;

    bool in_git = is_path_in_git_repo(fpath);

    // Evaluate @ignore (IgnoreRule(Document) -> bool) BEFORE reading file bytes
    if (!ignore_rule.empty() && host) {
        Document* probe = host->alloc_doc(0, fname);
        probe->fields["path"] = norm_path;
        probe->fields["filename"] = fname;
        probe->fields["ext"] = ext;
        probe->fields["is_dir"] = "false";
        probe->fields["in_git"] = in_git ? "true" : "false";
        probe->numbers["size"] = fsize;
        dslvm::Value ign = host->invoke_rule(
            ignore_rule,
            "interface IgnoreRule(Document) -> bool",
            {dslvm::Value::object("Document", probe)});
        if (ign.b) return 0;
    }

    std::ifstream in(fpath, std::ios::binary);
    if (!in) {
        throw dslvm::Error("add_file: cannot read file: " + file_path);
    }
    constexpr size_t kMaxFileBytes = 65536;
    std::string content;
    content.resize(kMaxFileBytes);
    in.read(&content[0], static_cast<std::streamsize>(kMaxFileBytes));
    content.resize(static_cast<size_t>(in.gcount()));

    // Skip binary files containing NUL bytes
    if (content.find('\0') != std::string::npos) {
        return 0;
    }

    if (!embedder) {
        if (!host) throw dslvm::Error("add_file: FaissIndex has no Host");
        OnnxModel loaded = OnnxModel::load_or_download_cached(host, "all-MiniLM-L6-v2", dim);
        OnnxModel* m = host->alloc_model();
        *m = std::move(loaded);
        m->host = host;
        embedder = m;
    }

    int64_t next_id = 1;
    for (const auto& e : entries) {
        if (e.id >= next_id) next_id = e.id + 1;
    }

    std::string indexed_text = fname + "\n" + content;
    Document* doc = host->alloc_doc(next_id, indexed_text);
    doc->fields["path"] = norm_path;
    doc->fields["filename"] = fname;
    doc->fields["ext"] = ext;
    doc->fields["is_dir"] = "false";
    doc->fields["in_git"] = in_git ? "true" : "false";
    doc->numbers["size"] = fsize;

    if (!filter_opts.matches_doc(doc)) {
        return 0;
    }

    // SAM FileFilter hook (@file_filter implements FileFilter(Document) -> bool)
    if (!file_filter_rule.empty() && host) {
        dslvm::Value allow = host->invoke_rule(
            file_filter_rule,
            "interface FileFilter(Document) -> bool",
            {dslvm::Value::object("Document", doc)});
        if (!allow.b) return 0;
    }

    doc->vec = embedder->run_text(doc->text);
    add_vector(doc->id, doc->vec, doc);
    return 1;
}

int64_t FaissIndex::add_directory_path(const std::string& dir_path, const std::string& ext_filter) {
    std::filesystem::path dpath = std::filesystem::u8path(dir_path);
    if (!std::filesystem::exists(dpath) || !std::filesystem::is_directory(dpath)) {
        throw dslvm::Error("add_dir: directory does not exist: " + dir_path);
    }

    std::vector<std::string> active_ignores = filter_opts.ignore_patterns;
    if (filter_opts.respect_gitignore) {
        collect_dir_ignore_files(dpath, active_ignores);
    }
    bool root_in_git = is_path_in_git_repo(dpath);

    std::vector<std::filesystem::path> files;
    std::error_code ec;
    auto opts = std::filesystem::directory_options::skip_permission_denied;
    for (auto it = std::filesystem::recursive_directory_iterator(dpath, opts, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        std::error_code pec;
        if (it->is_symlink(pec)) {
            if (it->is_directory(pec)) it.disable_recursion_pending();
            continue;
        }
        std::filesystem::path cur_p = it->path();
        std::string bname = cur_p.filename().generic_string();
        std::string rel_p = std::filesystem::relative(cur_p, dpath, pec).generic_string();
        if (rel_p.empty() || rel_p == ".") rel_p = bname;

        if (it->is_directory(pec)) {
            if ((!filter_opts.include_hidden && !bname.empty() && bname[0] == '.') ||
                is_skipped_dir_name(bname)) {
                it.disable_recursion_pending();
                continue;
            }
            if (!active_ignores.empty() && matches_ignore_list(active_ignores, rel_p, bname, true)) {
                it.disable_recursion_pending();
                continue;
            }
            if (!ignore_rule.empty() && host) {
                bool dir_in_git = root_in_git || std::filesystem::exists(cur_p / ".git", pec);
                Document* ddoc = host->alloc_doc(0, bname);
                ddoc->fields["path"] = cur_p.generic_string();
                ddoc->fields["filename"] = bname;
                ddoc->fields["ext"] = ".dir";
                ddoc->fields["is_dir"] = "true";
                ddoc->fields["in_git"] = dir_in_git ? "true" : "false";
                ddoc->numbers["size"] = 0;
                dslvm::Value ign = host->invoke_rule(
                    ignore_rule,
                    "interface IgnoreRule(Document) -> bool",
                    {dslvm::Value::object("Document", ddoc)});
                if (ign.b) {
                    it.disable_recursion_pending();
                    continue;
                }
            }
            if (filter_opts.respect_gitignore) {
                collect_dir_ignore_files(cur_p, active_ignores);
            }
            continue;
        }
        if (it->is_regular_file(pec)) {
            if (!filter_opts.include_hidden && !bname.empty() && bname[0] == '.') {
                continue;
            }
            if (!active_ignores.empty() && matches_ignore_list(active_ignores, rel_p, bname, false)) {
                continue;
            }
            if (has_allowed_extension(cur_p, ext_filter)) {
                files.push_back(cur_p);
            }
        }
    }
    std::sort(files.begin(), files.end());
    int64_t count = 0;
    for (const auto& f : files) {
        count += add_file_path(f.generic_string());
    }
    return count;
}

int64_t FaissIndex::add_path_list(const std::vector<std::string>& paths, const std::string& ext_filter) {
    int64_t count = 0;
    for (const auto& p : paths) {
        std::filesystem::path fp = std::filesystem::u8path(p);
        if (!std::filesystem::exists(fp)) {
            throw dslvm::Error("add_paths: path does not exist: " + p);
        }
        if (std::filesystem::is_directory(fp)) {
            count += add_directory_path(p, ext_filter);
        } else if (std::filesystem::is_regular_file(fp)) {
            count += add_file_path(p);
        }
    }
    return count;
}

static std::string trim_ws(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static std::filesystem::path resolve_gitdir_file(const std::filesystem::path& dir_path) {
    std::error_code ec;
    std::filesystem::path dot_git = dir_path / ".git";
    if (std::filesystem::is_regular_file(dot_git, ec)) {
        std::ifstream in(dot_git);
        std::string line;
        if (std::getline(in, line)) {
            line = trim_ws(line);
            const std::string prefix = "gitdir:";
            if (line.compare(0, prefix.size(), prefix) == 0) {
                std::string target = trim_ws(line.substr(prefix.size()));
                std::filesystem::path gp = std::filesystem::u8path(target);
                if (gp.is_relative()) gp = dir_path / gp;
                return gp;
            }
        }
    } else if (std::filesystem::is_directory(dot_git, ec)) {
        return dot_git;
    }
    return {};
}

static std::string run_git_capture(const std::string& args, int* out_rc = nullptr) {
#ifdef _WIN32
    std::string cmd = "git " + args + " 2>NUL";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    std::string cmd = "git " + args + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) {
        if (out_rc) *out_rc = -1;
        return "";
    }
    std::string output;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        output += buf;
    }
#ifdef _WIN32
    int rc = _pclose(pipe);
#else
    int rc = pclose(pipe);
#endif
    if (out_rc) *out_rc = rc;
    return trim_ws(output);
}

static std::string shell_quote_arg(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

bool git_is_worktree(const std::string& dir_path) {
    std::error_code ec;
    std::filesystem::path dot_git = std::filesystem::u8path(dir_path) / ".git";
    if (!std::filesystem::is_regular_file(dot_git, ec)) return false;
    std::ifstream in(dot_git);
    std::string line;
    if (!std::getline(in, line)) return false;
    line = trim_ws(line);
    return line.compare(0, 7, "gitdir:") == 0;
}

bool git_is_repo(const std::string& dir_path) {
    std::error_code ec;
    std::filesystem::path dot_git = std::filesystem::u8path(dir_path) / ".git";
    return std::filesystem::is_directory(dot_git, ec) || git_is_worktree(dir_path);
}

std::string git_worktree_branch(const std::string& dir_path) {
    std::filesystem::path gp = resolve_gitdir_file(std::filesystem::u8path(dir_path));
    if (!gp.empty()) {
        std::ifstream head_in(gp / "HEAD");
        std::string line;
        if (std::getline(head_in, line)) {
            line = trim_ws(line);
            const std::string ref_prefix = "ref: refs/heads/";
            if (line.compare(0, ref_prefix.size(), ref_prefix) == 0) {
                return line.substr(ref_prefix.size());
            }
            if (!line.empty()) return "HEAD";
        }
    }
    if (!git_is_repo(dir_path)) return "";
    int rc = 0;
    std::string b = run_git_capture("-C " + shell_quote_arg(dir_path) + " rev-parse --abbrev-ref HEAD", &rc);
    return rc == 0 ? b : "";
}

bool git_is_merged(const std::string& dir_path, const std::string& target_branch) {
    if (!git_is_repo(dir_path)) return false;
    std::filesystem::path gp = resolve_gitdir_file(std::filesystem::u8path(dir_path));
    std::error_code ec;
    if (!gp.empty() && std::filesystem::exists(gp / "MERGED", ec)) {
        return true;
    }
    std::string qdir = shell_quote_arg(dir_path);
    int rc = 0;
    std::string head_sha = run_git_capture("-C " + qdir + " rev-parse --verify HEAD", &rc);
    if (rc != 0 || head_sha.empty()) return false;

    std::vector<std::string> targets;
    if (!target_branch.empty()) {
        targets.push_back(target_branch);
        if (target_branch.find('/') == std::string::npos) {
            targets.push_back("origin/" + target_branch);
        }
    }
    if (target_branch == "main") {
        targets.push_back("master");
        targets.push_back("origin/master");
    }
    for (const auto& t : targets) {
        int mrc = -1;
        run_git_capture("-C " + qdir + " merge-base --is-ancestor HEAD " + shell_quote_arg(t), &mrc);
        if (mrc == 0) return true;
    }
    return false;
}

bool git_is_clean(const std::string& dir_path) {
    if (!git_is_repo(dir_path)) return false;
    int rc = -1;
    std::string out = run_git_capture("-C " + shell_quote_arg(dir_path) + " status --porcelain", &rc);
    return rc == 0 && out.empty();
}

int64_t compute_dir_size_bytes(const std::string& dir_path) {
    std::error_code ec;
    std::filesystem::path dpath = std::filesystem::u8path(dir_path);
    if (!std::filesystem::exists(dpath, ec)) return 0;
    if (std::filesystem::is_regular_file(dpath, ec)) {
        return static_cast<int64_t>(std::filesystem::file_size(dpath, ec));
    }
    int64_t total = 0;
    auto opts = std::filesystem::directory_options::skip_permission_denied;
    for (auto it = std::filesystem::recursive_directory_iterator(dpath, opts, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        std::error_code fec;
        if (!it->is_symlink(fec) && it->is_regular_file(fec)) {
            auto sz = it->file_size(fec);
            if (!fec) total += static_cast<int64_t>(sz);
        }
    }
    return total;
}

int64_t FaissIndex::add_dir_entry(const std::string& dir_path, int64_t size_bytes_override) {
    std::error_code ec;
    std::filesystem::path dpath = std::filesystem::u8path(dir_path);
    if (!std::filesystem::exists(dpath, ec) || !std::filesystem::is_directory(dpath, ec)) {
        return 0;
    }
    std::string norm_path = dpath.generic_string();
    std::string dname = dpath.filename().generic_string();
    if (dname.empty()) dname = norm_path;

    int64_t sz_bytes = (size_bytes_override >= 0) ? size_bytes_override : compute_dir_size_bytes(norm_path);
    int64_t sz_mb = sz_bytes / (1024 * 1024);

    if (!embedder) {
        if (!host) throw dslvm::Error("add_dir_entry: FaissIndex has no Host");
        OnnxModel loaded = OnnxModel::load_or_download_cached(host, "all-MiniLM-L6-v2", dim);
        OnnxModel* m = host->alloc_model();
        *m = std::move(loaded);
        m->host = host;
        embedder = m;
    }

    int64_t next_id = 1;
    for (const auto& e : entries) {
        if (e.id >= next_id) next_id = e.id + 1;
    }

    bool is_wt = git_is_worktree(norm_path);
    bool is_rp = git_is_repo(norm_path);
    std::string branch = is_rp ? git_worktree_branch(norm_path) : "";
    std::string summary = dname + " " + norm_path +
        (is_wt ? (" git worktree branch " + branch) : (is_rp ? (" git repo branch " + branch) : " directory")) +
        " size_mb=" + std::to_string(sz_mb);

    Document* doc = host->alloc_doc(next_id, summary);
    doc->fields["path"] = norm_path;
    doc->fields["filename"] = dname;
    doc->fields["ext"] = ".dir";
    doc->fields["is_worktree"] = is_wt ? "true" : "false";
    doc->fields["is_repo"] = is_rp ? "true" : "false";
    if (!branch.empty()) doc->fields["branch"] = branch;
    doc->numbers["size"] = sz_bytes;
    doc->numbers["size_mb"] = sz_mb;

    if (!filter_opts.matches_doc(doc)) {
        return 0;
    }

    if (!file_filter_rule.empty() && host) {
        dslvm::Value allow = host->invoke_rule(
            file_filter_rule,
            "interface FileFilter(Document) -> bool",
            {dslvm::Value::object("Document", doc)});
        if (!allow.b) return 0;
    }

    doc->vec = embedder->run_text(doc->text);
    add_vector(doc->id, doc->vec, doc);
    return 1;
}

int64_t FaissIndex::add_subdirs(const std::string& parent_dir) {
    std::error_code ec;
    std::filesystem::path ppath = std::filesystem::u8path(parent_dir);
    if (!std::filesystem::exists(ppath, ec) || !std::filesystem::is_directory(ppath, ec)) {
        return 0;
    }
    std::vector<std::filesystem::path> dirs;
    for (auto it = std::filesystem::directory_iterator(ppath, ec);
         it != std::filesystem::directory_iterator();
         it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        std::error_code dec;
        if (!it->is_symlink(dec) && it->is_directory(dec)) {
            dirs.push_back(it->path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    int64_t count = 0;
    for (const auto& d : dirs) {
        count += add_dir_entry(d.generic_string());
    }
    return count;
}

std::vector<double> FaissIndex::quantize_reconstruct(const Entry& e) const {
    if (m_pq <= 0 || pq_centroids.empty() || e.pq_codes.size() != static_cast<size_t>(m_pq)) {
        return e.norm_vec;
    }
    int64_t dsub = dim / m_pq;
    std::vector<double> out(static_cast<size_t>(dim), 0.0);
    for (int64_t m = 0; m < m_pq; ++m) {
        uint8_t code = e.pq_codes[static_cast<size_t>(m)];
        const auto& sub = pq_centroids[static_cast<size_t>(m)][code % pq_centroids[static_cast<size_t>(m)].size()];
        for (int64_t j = 0; j < dsub; ++j) {
            out[static_cast<size_t>(m * dsub + j)] = sub[static_cast<size_t>(j)];
        }
    }
    return out;
}

std::vector<double> FaissIndex::reconstruct_vec(int64_t id) const {
    for (const auto& e : entries) {
        if (e.id == id) return quantize_reconstruct(e);
    }
    throw dslvm::Error("reconstruct: id " + std::to_string(id) + " not found in index");
}

SearchResults* FaissIndex::run_search(const std::vector<double>& query, int64_t k, const std::string& qtext) {
    if (static_cast<int64_t>(query.size()) != dim) {
        throw dslvm::Error("search query dimension mismatch: got " + std::to_string(query.size()) +
                           ", expected " + std::to_string(dim));
    }
    if (k <= 0) throw dslvm::Error("search k must be positive");
    if (!is_trained) train_index();

    std::vector<double> q = (metric == "COSINE") ? l2_norm_vec(query) : query;

    // Determine probed coarse clusters if IVF
    std::vector<bool> allowed_lists(static_cast<size_t>(std::max<int64_t>(1, nlist)), true);
    if (nlist > 1 && !centroids.empty() && cur_nprobe < nlist) {
        std::fill(allowed_lists.begin(), allowed_lists.end(), false);
        std::vector<std::pair<double, int64_t>> c_dists;
        for (size_t c = 0; c < centroids.size(); ++c) {
            c_dists.push_back({vec_l2(q, centroids[c]), static_cast<int64_t>(c)});
        }
        std::sort(c_dists.begin(), c_dists.end());
        for (int64_t p = 0; p < std::min<int64_t>(cur_nprobe, static_cast<int64_t>(c_dists.size())); ++p) {
            allowed_lists[static_cast<size_t>(c_dists[static_cast<size_t>(p)].second)] = true;
        }
    }

    struct Cand {
        double dist;
        double score;
        int64_t id;
        Document* doc;
    };
    std::vector<Cand> cands;
    cands.reserve(entries.size());

    for (const auto& e : entries) {
        if (nlist > 1 && !centroids.empty() && cur_nprobe < nlist) {
            if (!allowed_lists[static_cast<size_t>(e.list_id)]) continue;
        }
        if (!filter_opts.matches_doc(e.doc)) continue;
        // SAM VectorFilter hook (acts as faiss::IDSelector)
        if (!filter_rule.empty() && host) {
            dslvm::Value keep = host->invoke_rule(
                filter_rule,
                "interface VectorFilter(int, Document) -> bool",
                {dslvm::Value::integer(e.id), dslvm::Value::object("Document", e.doc)});
            if (!keep.b) continue;
        }
        std::vector<double> target = (m_pq > 0) ? quantize_reconstruct(e) : e.norm_vec;
        double dist = 0.0;
        double score = 0.0;
        if (metric == "L2") {
            dist = vec_l2(q, target);
            score = 1.0 / (1.0 + dist);
        } else {
            double ip = vec_ip(q, target);
            dist = ip;
            score = ip;
        }
        if (dist > filter_opts.max_distance) continue;
        if (rerank_rule.empty() && score < filter_opts.min_score) continue;
        cands.push_back({dist, score, e.id, e.doc});
    }

    auto cmp = [&](const Cand& a, const Cand& b) {
        if (std::abs(a.score - b.score) > 1e-12) return a.score > b.score;
        return a.id < b.id;
    };
    std::sort(cands.begin(), cands.end(), cmp);
    if (static_cast<int64_t>(cands.size()) > k) {
        cands.resize(static_cast<size_t>(k));
    }

    SearchResults* res = host->alloc_results();
    res->query_text = qtext;
    res->params = effective_params();
    for (size_t i = 0; i < cands.size(); ++i) {
        SearchHit h;
        h.rank = static_cast<int64_t>(i);
        h.id = cands[i].id;
        h.distance = cands[i].dist;
        h.score = cands[i].score;
        h.doc = cands[i].doc;
        SearchHit* hp = host->alloc_hit(h);
        if (!rerank_rule.empty() && host) {
            dslvm::Value new_score = host->invoke_rule(
                rerank_rule,
                "interface Reranker(SearchHit, string) -> float",
                {dslvm::Value::object("SearchHit", hp), dslvm::Value::string(qtext)});
            hp->score = new_score.f;
        }
        if (hp->score < filter_opts.min_score) continue;
        res->hits.push_back(hp);
    }
    if (!rerank_rule.empty()) {
        std::sort(res->hits.begin(), res->hits.end(), [](const SearchHit* a, const SearchHit* b) {
            if (std::abs(a->score - b->score) > 1e-12) return a->score > b->score;
            return a->id < b->id;
        });
        for (size_t i = 0; i < res->hits.size(); ++i) {
            res->hits[i]->rank = static_cast<int64_t>(i);
        }
    }
    return res;
}

SearchResults* FaissIndex::run_range_search(const std::vector<double>& query, double radius) {
    if (static_cast<int64_t>(query.size()) != dim) {
        throw dslvm::Error("range_search query dimension mismatch");
    }
    if (!is_trained) train_index();
    std::vector<double> q = (metric == "COSINE") ? l2_norm_vec(query) : query;
    SearchResults* res = host->alloc_results();
    res->params = effective_params();
    for (const auto& e : entries) {
        if (!filter_rule.empty() && host) {
            dslvm::Value keep = host->invoke_rule(
                filter_rule,
                "interface VectorFilter(int, Document) -> bool",
                {dslvm::Value::integer(e.id), dslvm::Value::object("Document", e.doc)});
            if (!keep.b) continue;
        }
        const auto& target = e.norm_vec;
        double dist = (metric == "L2") ? vec_l2(q, target) : vec_ip(q, target);
        bool match = (metric == "L2") ? (dist <= radius) : (dist >= radius);
        if (match) {
            SearchHit h;
            h.id = e.id;
            h.distance = dist;
            h.score = (metric == "L2") ? (1.0 / (1.0 + dist)) : dist;
            h.doc = e.doc;
            res->hits.push_back(host->alloc_hit(h));
        }
    }
    std::sort(res->hits.begin(), res->hits.end(), [](const SearchHit* a, const SearchHit* b) {
        if (std::abs(a->score - b->score) > 1e-12) return a->score > b->score;
        return a->id < b->id;
    });
    for (size_t i = 0; i < res->hits.size(); ++i) {
        res->hits[i]->rank = static_cast<int64_t>(i);
    }
    return res;
}

std::string FaissIndex::run_autotune(double max_query_ms, int64_t min_neighbors) {
    if (max_query_ms <= 0.0 || min_neighbors <= 0)
        throw dslvm::Error("autotune requires positive max_query_ms and min_neighbors");
    if (!is_trained) train_index();
    if (index_key == "Flat") {
        param_str = "";
        return param_str;
    }
    if (index_key.find("HNSW") == 0) {
        int64_t ef = std::max<int64_t>(16, std::min<int64_t>(256, std::max<int64_t>(min_neighbors * 2, static_cast<int64_t>(max_query_ms * 8.0))));
        cur_ef_search = ef;
        param_str = "efSearch=" + std::to_string(ef);
        return param_str;
    }
    // IVF / IVFPQ / OPQ binary search over nprobe to satisfy coverage + budget
    int64_t low = 1;
    int64_t high = std::max<int64_t>(1, nlist);
    int64_t best = std::min<int64_t>(high, std::max<int64_t>(1, static_cast<int64_t>(std::ceil(high * 0.25))));
    while (low <= high) {
        int64_t mid = (low + high) / 2;
        cur_nprobe = mid;
        // Estimate avg retrieved neighbors across sample queries
        size_t nq = std::min<size_t>(entries.size(), 10);
        int64_t total_got = 0;
        int64_t target_k = std::min<int64_t>(min_neighbors, static_cast<int64_t>(entries.size()));
        for (size_t i = 0; i < nq; ++i) {
            auto* r = run_search(entries[i].raw_vec, std::max<int64_t>(1, target_k), "");
            total_got += static_cast<int64_t>(r->hits.size());
        }
        double cov = (nq > 0 && target_k > 0) ? (static_cast<double>(total_got) / static_cast<double>(nq * target_k)) : 1.0;
        if (cov >= 0.95) {
            best = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    cur_nprobe = best;
    if (m_pq > 0) {
        param_str = "nprobe=" + std::to_string(cur_nprobe) + ",ht=" + std::to_string(cur_ht);
    } else {
        param_str = "nprobe=" + std::to_string(cur_nprobe);
    }
    return param_str;
}

IndexMetrics* FaissIndex::run_evaluate(int64_t k) {
    if (k <= 0) throw dslvm::Error("evaluate requires positive k");
    if (!is_trained) train_index();
    IndexMetrics m;
    m.index_key = index_key;
    m.param_string = effective_params();
    m.k = k;

    int64_t n = static_cast<int64_t>(entries.size());
    auto plan = compute_plan_for_key(index_key, std::max<int64_t>(1, n), dim, metric, 16LL * 1024 * 1024 * 1024, false, 10.0);
    m.size_bytes = plan.estimated_bytes;
    m.compression_ratio = plan.compression_ratio;

    if (n == 0) {
        return host->alloc_metrics(m);
    }

    size_t nq = std::min<size_t>(entries.size(), 25);
    int64_t eff_k = std::min<int64_t>(k, n);
    double sum_r1 = 0.0;
    double sum_rk = 0.0;
    double sum_rec_err = 0.0;

    std::string saved_filter = filter_rule;
    std::string saved_rerank = rerank_rule;
    filter_rule.clear();
    rerank_rule.clear();

    for (size_t qi = 0; qi < nq; ++qi) {
        const auto& q = entries[qi].norm_vec;
        // Ground truth via exact flat scan
        std::vector<std::pair<double, int64_t>> gt;
        gt.reserve(entries.size());
        for (const auto& e : entries) {
            double sc = (metric == "L2") ? (1.0 / (1.0 + vec_l2(q, e.norm_vec))) : vec_ip(q, e.norm_vec);
            gt.push_back({sc, e.id});
        }
        std::sort(gt.begin(), gt.end(), [](const auto& a, const auto& b) {
            if (std::abs(a.first - b.first) > 1e-12) return a.first > b.first;
            return a.second < b.second;
        });

        auto* approx = run_search(entries[qi].raw_vec, eff_k, "");
        int64_t top1_gt = gt[0].second;
        bool hit_r1 = false;
        int64_t overlap = 0;
        for (size_t hi = 0; hi < approx->hits.size(); ++hi) {
            if (approx->hits[hi]->id == top1_gt) hit_r1 = true;
            for (int64_t gi = 0; gi < eff_k; ++gi) {
                if (approx->hits[hi]->id == gt[static_cast<size_t>(gi)].second) {
                    overlap++;
                    break;
                }
            }
        }
        sum_r1 += hit_r1 ? 1.0 : 0.0;
        sum_rk += static_cast<double>(overlap) / static_cast<double>(eff_k);

        // Reconstruction error
        auto recon = quantize_reconstruct(entries[qi]);
        double diff_norm = std::sqrt(vec_l2(entries[qi].norm_vec, recon));
        double orig_norm = std::sqrt(vec_ip(entries[qi].norm_vec, entries[qi].norm_vec));
        sum_rec_err += (orig_norm > 1e-12) ? (diff_norm / orig_norm) : 0.0;
    }

    filter_rule = saved_filter;
    rerank_rule = saved_rerank;

    m.recall_1_at_k = sum_r1 / static_cast<double>(nq);
    m.recall_k_at_k = sum_rk / static_cast<double>(nq);
    m.reconstruction_error = (sum_rec_err / static_cast<double>(nq)) * 100.0;
    m.ms_per_query = 0.05 * static_cast<double>( std::max<int64_t>(1, cur_nprobe) );

    IndexMetrics* out = host->alloc_metrics(m);
    host->evaluated_metrics.push_back(out);
    return out;
}

static std::string escape_line_text(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out.push_back(c);
    }
    return out;
}

static std::string unescape_line_text(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            if (n == 'n') out.push_back('\n');
            else if (n == 'r') out.push_back('\r');
            else out.push_back(n);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

void FaissIndex::save_to_file(const std::string& path) const {
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out) throw dslvm::Error("save index: cannot open " + path);
    out << "LMSEARCH_FAISS_V1\n";
    out << dim << "\n" << index_key << "\n" << metric << "\n" << param_str << "\n";
    out << cur_nprobe << " " << cur_ef_search << " " << (is_trained ? 1 : 0) << "\n";
    out << entries.size() << "\n";
    for (const auto& e : entries) {
        out << e.id << "\n";
        out << escape_line_text(e.doc ? e.doc->text : "") << "\n";
        for (size_t j = 0; j < e.raw_vec.size(); ++j) {
            if (j > 0) out << " ";
            out << e.raw_vec[j];
        }
        out << "\n";
        if (e.doc) {
            out << e.doc->fields.size() << "\n";
            for (const auto& kv : e.doc->fields) {
                out << escape_line_text(kv.first) << "\n" << escape_line_text(kv.second) << "\n";
            }
            out << e.doc->numbers.size() << "\n";
            for (const auto& kv : e.doc->numbers) {
                out << escape_line_text(kv.first) << "\n" << kv.second << "\n";
            }
        } else {
            out << "0\n0\n";
        }
    }
    if (!out) throw dslvm::Error("save index: write error for " + path);
    if (embedder) {
        embedder->save_to_file(path + ".onnx");
    }
}

FaissIndex* FaissIndex::load_from_file(Host* host, const std::string& path) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) throw dslvm::Error("load_index: cannot open " + path);
    auto getline_clean = [&](std::string& s) {
        if (!std::getline(in, s)) return false;
        if (!s.empty() && s.back() == '\r') s.pop_back();
        return true;
    };
    std::string magic;
    if (!getline_clean(magic) || magic != "LMSEARCH_FAISS_V1")
        throw dslvm::Error("load_index: invalid index file " + path);
    std::string line;
    getline_clean(line);
    int64_t dim = std::stoll(line);
    std::string key, metric, pstr;
    getline_clean(key);
    getline_clean(metric);
    getline_clean(pstr);
    FaissIndex* idx = host->alloc_index(dim, key, metric);
    getline_clean(line);
    {
        std::stringstream ss(line);
        int trained_flag = 1;
        ss >> idx->cur_nprobe >> idx->cur_ef_search >> trained_flag;
        idx->is_trained = (trained_flag != 0);
    }
    if (!pstr.empty()) idx->apply_param_string(pstr);
    getline_clean(line);
    size_t count = static_cast<size_t>(std::stoull(line));
    for (size_t i = 0; i < count; ++i) {
        getline_clean(line);
        int64_t id = std::stoll(line);
        std::string text;
        getline_clean(text);
        text = unescape_line_text(text);
        getline_clean(line);
        std::vector<double> vec(static_cast<size_t>(dim), 0.0);
        {
            std::stringstream ss(line);
            for (int64_t j = 0; j < dim; ++j) ss >> vec[static_cast<size_t>(j)];
        }
        Document* doc = host->alloc_doc(id, text);
        doc->vec = vec;
        getline_clean(line);
        size_t nf = static_cast<size_t>(std::stoull(line));
        for (size_t f = 0; f < nf; ++f) {
            std::string k, v;
            getline_clean(k);
            getline_clean(v);
            doc->fields[unescape_line_text(k)] = unescape_line_text(v);
        }
        getline_clean(line);
        size_t nn = static_cast<size_t>(std::stoull(line));
        for (size_t n = 0; n < nn; ++n) {
            std::string k, v;
            getline_clean(k);
            getline_clean(v);
            doc->numbers[unescape_line_text(k)] = std::stoll(v);
        }
        idx->add_vector(id, vec, doc);
    }
    if (idx->nlist > 1 || idx->m_pq > 0) idx->train_index();
    std::string sidecar_onnx = path + ".onnx";
    if (std::filesystem::exists(std::filesystem::u8path(sidecar_onnx))) {
        try {
            OnnxModel loaded = OnnxModel::load_from_file(host, sidecar_onnx);
            OnnxModel* m = host->alloc_model();
            *m = std::move(loaded);
            m->host = host;
            idx->embedder = m;
        } catch (...) {
        }
    }
    return idx;
}

} // namespace lmsearch
