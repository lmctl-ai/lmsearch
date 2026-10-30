#include "lmsearch.hpp"
#include "collections.hpp"
#include "testing.hpp"
#include <cmath>
#include <cstdio>
#include <sstream>

namespace lmsearch {

namespace {

std::string fmt_num(double v) {
    if (!std::isfinite(v)) return "0";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

OnnxModel& model_ref(void* p) {
    if (!p) throw dslvm::Error("null OnnxModel handle");
    return *static_cast<OnnxModel*>(p);
}

Document& doc_ref(void* p) {
    if (!p) throw dslvm::Error("null Document handle");
    return *static_cast<Document*>(p);
}

AutoTuner& tuner_ref(void* p) {
    if (!p) throw dslvm::Error("null AutoTuner handle");
    return *static_cast<AutoTuner*>(p);
}

AutoPlan& plan_ref(void* p) {
    if (!p) throw dslvm::Error("null AutoPlan handle");
    return *static_cast<AutoPlan*>(p);
}

FaissIndex& index_ref(void* p) {
    if (!p) throw dslvm::Error("null FaissIndex handle");
    return *static_cast<FaissIndex*>(p);
}

SearchResults& results_ref(void* p) {
    if (!p) throw dslvm::Error("null SearchResults handle");
    return *static_cast<SearchResults*>(p);
}

SearchHit& hit_ref(void* p) {
    if (!p) throw dslvm::Error("null SearchHit handle");
    return *static_cast<SearchHit*>(p);
}

IndexMetrics& metrics_ref(void* p) {
    if (!p) throw dslvm::Error("null IndexMetrics handle");
    return *static_cast<IndexMetrics*>(p);
}

} // namespace

Host::Host() {
    bind_all(registry, false);
}

dslvm::Context Host::context() {
    return dslvm::Context{};
}

OnnxModel* Host::alloc_model() {
    auto m = std::make_unique<OnnxModel>();
    m->host = this;
    models.push_back(std::move(m));
    return models.back().get();
}

Document* Host::alloc_doc(int64_t id, const std::string& text) {
    auto d = std::make_unique<Document>();
    d->id = id;
    d->text = text;
    docs.push_back(std::move(d));
    return docs.back().get();
}

AutoTuner* Host::alloc_tuner(int64_t nb_vectors, int64_t dim) {
    auto t = std::make_unique<AutoTuner>();
    t->host = this;
    t->nb_vectors = nb_vectors;
    t->dim = dim;
    tuners.push_back(std::move(t));
    return tuners.back().get();
}

AutoPlan* Host::alloc_plan(const AutoPlan& p) {
    auto ptr = std::make_unique<AutoPlan>(p);
    plans.push_back(std::move(ptr));
    return plans.back().get();
}

FaissIndex* Host::alloc_index(int64_t dim, const std::string& key, const std::string& metric) {
    auto idx = std::make_unique<FaissIndex>();
    idx->host = this;
    idx->dim = dim;
    idx->index_key = key;
    idx->metric = metric;
    idx->parse_key();
    indexes.push_back(std::move(idx));
    return indexes.back().get();
}

SearchHit* Host::alloc_hit(const SearchHit& h) {
    auto ptr = std::make_unique<SearchHit>(h);
    hits.push_back(std::move(ptr));
    return hits.back().get();
}

SearchResults* Host::alloc_results() {
    auto ptr = std::make_unique<SearchResults>();
    results_arena.push_back(std::move(ptr));
    return results_arena.back().get();
}

IndexMetrics* Host::alloc_metrics(const IndexMetrics& m) {
    auto ptr = std::make_unique<IndexMetrics>(m);
    metrics_arena.push_back(std::move(ptr));
    return metrics_arena.back().get();
}

dslvm::Value Host::invoke_rule(
    const std::string& rule_name,
    const std::string& expected_iface_prefix,
    std::vector<dslvm::Value> args,
    dslvm::Registry** out_cb_reg,
    std::unique_ptr<dslvm::Registry>* owned_cb_reg) {
    if (in_callback) {
        throw dslvm::Error("recursive SAM callback invocation is not allowed: " + rule_name);
    }
    if (bytecode_bytes.empty() || schema_bytes.empty()) {
        throw dslvm::Error("cannot invoke rule '" + rule_name + "' without loaded program bytes");
    }

    std::unique_ptr<dslvm::Registry> local_reg;
    dslvm::Registry* cb_reg = nullptr;
    if (out_cb_reg && *out_cb_reg) {
        cb_reg = *out_cb_reg;
    } else {
        local_reg = std::make_unique<dslvm::Registry>(dslvm::StdMode::Disabled);
        bind_all(*local_reg, true);
        cb_reg = local_reg.get();
        if (owned_cb_reg) *owned_cb_reg = std::move(local_reg);
        if (out_cb_reg) *out_cb_reg = cb_reg;
    }

    dslvm::Program cb_prog = dslvm::Program::load(bytecode_bytes, schema_bytes, *cb_reg);
    size_t found_idx = cb_prog.entries().size();
    for (size_t i = 0; i < cb_prog.entries().size(); ++i) {
        if (cb_prog.entries()[i].name == rule_name) {
            found_idx = i;
            break;
        }
    }
    if (found_idx == cb_prog.entries().size()) {
        throw dslvm::Error("unknown rule entry '" + rule_name + "'");
    }
    const auto& entry = cb_prog.entries()[found_idx];
    if (entry.iface != expected_iface_prefix) {
        throw dslvm::Error("rule '" + rule_name + "' implements '" + entry.iface +
                           "'; expected '" + expected_iface_prefix + "'");
    }

    in_callback = true;
    struct Guard {
        bool& flag;
        ~Guard() { flag = false; }
    } guard{in_callback};

    dslvm::Context cb_ctx;
    dslvm::Value res = cb_prog.call(found_idx, cb_ctx, *cb_reg, std::move(args));
    ++entries_called;
    return res;
}

void Host::bind_all(dslvm::Registry& reg, bool callback_mode) {
    using dslvm::Args;
    using dslvm::Value;

    dslvm::stdlib::Collections& colls = callback_mode ? cb_collections : reg.std_collections();
    if (callback_mode && !reg.has_std()) {
        dslvm::stdlib::register_string(reg);
        dslvm::stdlib::register_encoding(reg);
        dslvm::stdlib::register_math(reg);
        dslvm::stdlib::register_collections(reg, colls);
        dslvm::stdlib::register_file(reg, colls);
        dslvm::stdlib::register_containers(reg, colls);
        dslvm::stdlib::register_json(reg, colls);
    }
    dslvm::stdlib::register_testing(reg);

    auto guard_not_callback = [callback_mode](const char* op) {
        if (callback_mode) {
            throw dslvm::Error(std::string("host operation '") + op + "' is not permitted inside a SAM rule callback");
        }
    };

    // Functions
    reg.function("onnx_embedder", {"int", "int", "int"}, "OnnxModel", [this, guard_not_callback](Args& a) {
        guard_not_callback("onnx_embedder");
        OnnxModel* m = alloc_model();
        m->init_text_embedder(a[0].i, a[1].i, a[2].i);
        return Value::object("OnnxModel", m);
    });

    reg.function("onnx_mlp_embedder", {"int", "int", "int", "int"}, "OnnxModel", [this, guard_not_callback](Args& a) {
        guard_not_callback("onnx_mlp_embedder");
        OnnxModel* m = alloc_model();
        m->init_mlp_embedder(a[0].i, a[1].i, a[2].i, a[3].i);
        return Value::object("OnnxModel", m);
    });

    reg.function("onnx_load", {"string"}, "OnnxModel", [this, guard_not_callback](Args& a) {
        guard_not_callback("onnx_load");
        OnnxModel loaded = OnnxModel::load_from_file(this, a[0].s);
        OnnxModel* m = alloc_model();
        *m = std::move(loaded);
        m->host = this;
        return Value::object("OnnxModel", m);
    });

    reg.function("document", {"int"}, "Document", [this](Args& a) {
        return Value::object("Document", alloc_doc(a[0].i, ""));
    });

    reg.function("document", {"int", "string"}, "Document", [this](Args& a) {
        return Value::object("Document", alloc_doc(a[0].i, a[1].s));
    });

    reg.function("autotuner", {"int", "int"}, "AutoTuner", [this, guard_not_callback](Args& a) {
        guard_not_callback("autotuner");
        if (a[0].i <= 0 || a[1].i <= 0) throw dslvm::Error("autotuner requires positive nb_vectors and dim");
        return Value::object("AutoTuner", alloc_tuner(a[0].i, a[1].i));
    });

    reg.function("faiss_index", {"int", "string", "Metric"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("faiss_index");
        return Value::object("FaissIndex", alloc_index(a[0].i, a[1].s, a[2].s));
    });

    reg.function("auto_index", {"AutoPlan"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("auto_index");
        auto& p = plan_ref(a[0].obj);
        FaissIndex* idx = alloc_index(p.dim, p.index_key, p.metric);
        if (!p.recommended_params.empty()) idx->apply_param_string(p.recommended_params);
        return Value::object("FaissIndex", idx);
    });

    reg.function("flat_index", {"int", "Metric"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("flat_index");
        return Value::object("FaissIndex", alloc_index(a[0].i, "Flat", a[1].s));
    });

    reg.function("hnsw_index", {"int", "int", "Metric"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("hnsw_index");
        if (a[1].i <= 0) throw dslvm::Error("hnsw_index M must be positive");
        return Value::object("FaissIndex", alloc_index(a[0].i, "HNSW" + std::to_string(a[1].i), a[2].s));
    });

    reg.function("ivf_flat_index", {"int", "int", "Metric"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("ivf_flat_index");
        if (a[1].i <= 0) throw dslvm::Error("ivf_flat_index nlist must be positive");
        return Value::object("FaissIndex", alloc_index(a[0].i, "IVF" + std::to_string(a[1].i) + ",Flat", a[2].s));
    });

    reg.function("ivf_pq_index", {"int", "int", "int", "int", "Metric"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("ivf_pq_index");
        if (a[1].i <= 0 || a[2].i <= 0 || a[3].i <= 0)
            throw dslvm::Error("ivf_pq_index parameters must be positive");
        std::string key = "IVF" + std::to_string(a[1].i) + ",PQ" + std::to_string(a[2].i) + "x" + std::to_string(a[3].i);
        return Value::object("FaissIndex", alloc_index(a[0].i, key, a[4].s));
    });

    reg.function("load_index", {"string"}, "FaissIndex", [this, guard_not_callback](Args& a) {
        guard_not_callback("load_index");
        return Value::object("FaissIndex", FaissIndex::load_from_file(this, a[0].s));
    });

    reg.function("synthetic_vector", {"int", "int", "int", "float"}, "FloatList", [&colls](Args& a) {
        int64_t dim = a[0].i;
        int64_t cluster = a[1].i;
        int64_t item = a[2].i;
        double noise = a[3].f;
        if (dim <= 0) throw dslvm::Error("synthetic_vector requires positive dim");
        std::vector<double> v(static_cast<size_t>(dim), 0.0);
        for (int64_t j = 0; j < dim; ++j) {
            double base = std::sin(static_cast<double>((cluster + 1) * 17 + (j + 1) * 31));
            double delta = std::cos(static_cast<double>((item + 1) * 43 + (j + 1) * 13)) * noise;
            v[static_cast<size_t>(j)] = base + delta;
        }
        return Value::object("FloatList", colls.make_float_list(std::move(v)));
    });

    reg.function("l2_distance", {"FloatList", "FloatList"}, "float", [](Args& a) {
        const auto& x = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        const auto& y = static_cast<dslvm::stdlib::FloatList*>(a[1].obj)->items;
        if (x.size() != y.size()) throw dslvm::Error("l2_distance requires equal-length vectors");
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double d = x[i] - y[i];
            s += d * d;
        }
        return Value::real(s);
    });

    reg.function("inner_product", {"FloatList", "FloatList"}, "float", [](Args& a) {
        const auto& x = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        const auto& y = static_cast<dslvm::stdlib::FloatList*>(a[1].obj)->items;
        if (x.size() != y.size()) throw dslvm::Error("inner_product requires equal-length vectors");
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
        return Value::real(s);
    });

    reg.function("cosine_similarity", {"FloatList", "FloatList"}, "float", [](Args& a) {
        const auto& x = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        const auto& y = static_cast<dslvm::stdlib::FloatList*>(a[1].obj)->items;
        if (x.size() != y.size() || x.empty())
            throw dslvm::Error("cosine_similarity requires non-empty equal-length vectors");
        double dot = 0.0, nx = 0.0, ny = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            dot += x[i] * y[i];
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }
        double denom = std::sqrt(nx) * std::sqrt(ny);
        return Value::real(denom > 1e-24 ? (dot / denom) : 0.0);
    });

    reg.function("l2_normalize", {"FloatList"}, "FloatList", [&colls](Args& a) {
        const auto& x = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        double s = 0.0;
        for (double v : x) s += v * v;
        std::vector<double> out = x;
        if (s > 1e-24) {
            double inv = 1.0 / std::sqrt(s);
            for (double& v : out) v *= inv;
        }
        return Value::object("FloatList", colls.make_float_list(std::move(out)));
    });

    reg.function("assert_true", {"bool", "string"}, "void", [](Args& a) {
        if (!a[0].b) throw dslvm::Error("assertion failed: " + a[1].s);
        return Value::none();
    });

    reg.function("report", {"string", "string"}, "void", [this](Args& a) {
        reports.push_back({a[0].s, a[1].s});
        return Value::none();
    });

    // OnnxModel methods
    reg.method("OnnxModel", "opset", {}, "int", [](void* self, Args&) {
        return Value::integer(model_ref(self).opset);
    });
    reg.method("OnnxModel", "ir_version", {}, "int", [](void* self, Args&) {
        return Value::integer(model_ref(self).ir_version);
    });
    reg.method("OnnxModel", "input_dim", {}, "int", [](void* self, Args&) {
        return Value::integer(model_ref(self).input_dim);
    });
    reg.method("OnnxModel", "output_dim", {}, "int", [](void* self, Args&) {
        return Value::integer(model_ref(self).output_dim);
    });
    reg.method("OnnxModel", "node_count", {}, "int", [](void* self, Args&) {
        return Value::integer(static_cast<int64_t>(model_ref(self).node_ops.size()));
    });
    reg.method("OnnxModel", "node_op", {"int"}, "string", [](void* self, Args& a) {
        auto& ops = model_ref(self).node_ops;
        if (a[0].i < 0 || static_cast<size_t>(a[0].i) >= ops.size())
            throw dslvm::Error("OnnxModel.node_op: index out of range");
        return Value::string(ops[static_cast<size_t>(a[0].i)]);
    });
    reg.method("OnnxModel", "pooling", {"Pooling"}, "OnnxModel", [](void* self, Args& a) {
        model_ref(self).pooling = a[0].s;
        return Value::object("OnnxModel", self);
    });
    reg.method("OnnxModel", "normalize", {"bool"}, "OnnxModel", [](void* self, Args& a) {
        model_ref(self).normalize = a[0].b;
        return Value::object("OnnxModel", self);
    });
    reg.method("OnnxModel", "transform_rule", {"string"}, "OnnxModel", [](void* self, Args& a) {
        model_ref(self).transform_rule = a[0].s;
        return Value::object("OnnxModel", self);
    });
    reg.method("OnnxModel", "check", {}, "bool", [](void* self, Args&) {
        return Value::boolean(model_ref(self).check_valid());
    });
    reg.method("OnnxModel", "summary", {}, "string", [](void* self, Args&) {
        return Value::string(model_ref(self).summary_text());
    });
    reg.method("OnnxModel", "save", {"string"}, "OnnxModel", [](void* self, Args& a) {
        model_ref(self).save_to_file(a[0].s);
        return Value::object("OnnxModel", self);
    });
    reg.method("OnnxModel", "embed_text", {"string"}, "FloatList", [&colls](void* self, Args& a) {
        return Value::object("FloatList", colls.make_float_list(model_ref(self).run_text(a[0].s)));
    });
    reg.method("OnnxModel", "embed_tokens", {"IntList"}, "FloatList", [&colls](void* self, Args& a) {
        const auto& toks = static_cast<dslvm::stdlib::IntList*>(a[0].obj)->items;
        return Value::object("FloatList", colls.make_float_list(model_ref(self).run_tokens(toks)));
    });
    reg.method("OnnxModel", "embed_vector", {"FloatList"}, "FloatList", [&colls](void* self, Args& a) {
        const auto& vec = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        return Value::object("FloatList", colls.make_float_list(model_ref(self).run_vector(vec)));
    });
    reg.method("OnnxModel", "emit", {"string"}, "OnnxModel", [this](void* self, Args& a) {
        auto& m = model_ref(self);
        std::ostringstream oss;
        oss << "{\"label\": " << dslvm::json_escape(a[0].s)
            << ", \"opset\": " << m.opset
            << ", \"ir_version\": " << m.ir_version
            << ", \"input_dim\": " << m.input_dim
            << ", \"output_dim\": " << m.output_dim
            << ", \"nodes\": " << m.node_ops.size()
            << ", \"pooling\": " << dslvm::json_escape(m.pooling)
            << ", \"summary\": " << dslvm::json_escape(m.summary_text()) << "}";
        emitted_models.push_back(oss.str());
        return Value::object("OnnxModel", self);
    });

    // Document methods
    reg.method("Document", "id", {}, "int", [](void* self, Args&) {
        return Value::integer(doc_ref(self).id);
    });
    reg.method("Document", "text", {}, "string", [](void* self, Args&) {
        return Value::string(doc_ref(self).text);
    });
    reg.method("Document", "set_text", {"string"}, "Document", [](void* self, Args& a) {
        doc_ref(self).text = a[0].s;
        return Value::object("Document", self);
    });
    reg.method("Document", "field", {"string", "string"}, "Document", [](void* self, Args& a) {
        doc_ref(self).fields[a[0].s] = a[1].s;
        return Value::object("Document", self);
    });
    reg.method("Document", "get", {"string"}, "string", [](void* self, Args& a) {
        auto& f = doc_ref(self).fields;
        auto it = f.find(a[0].s);
        if (it == f.end()) throw dslvm::Error("Document has no field '" + a[0].s + "'");
        return Value::string(it->second);
    });
    reg.method("Document", "has", {"string"}, "bool", [](void* self, Args& a) {
        auto& d = doc_ref(self);
        return Value::boolean(d.fields.count(a[0].s) > 0 || d.numbers.count(a[0].s) > 0);
    });
    reg.method("Document", "number", {"string", "int"}, "Document", [](void* self, Args& a) {
        doc_ref(self).numbers[a[0].s] = a[1].i;
        return Value::object("Document", self);
    });
    reg.method("Document", "get_number", {"string"}, "int", [](void* self, Args& a) {
        auto& n = doc_ref(self).numbers;
        auto it = n.find(a[0].s);
        if (it == n.end()) throw dslvm::Error("Document has no numeric field '" + a[0].s + "'");
        return Value::integer(it->second);
    });
    reg.method("Document", "vector", {"FloatList"}, "Document", [](void* self, Args& a) {
        doc_ref(self).vec = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        return Value::object("Document", self);
    });
    reg.method("Document", "values", {}, "FloatList", [&colls](void* self, Args&) {
        return Value::object("FloatList", colls.make_float_list(doc_ref(self).vec));
    });

    // AutoTuner methods
    reg.method("AutoTuner", "metric", {"Metric"}, "AutoTuner", [](void* self, Args& a) {
        tuner_ref(self).metric = a[0].s;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "max_memory", {"string"}, "AutoTuner", [](void* self, Args& a) {
        (void)parse_memory_bytes(a[0].s);
        tuner_ref(self).max_memory = a[0].s;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "max_query_ms", {"float"}, "AutoTuner", [](void* self, Args& a) {
        if (a[0].f <= 0.0) throw dslvm::Error("max_query_ms must be positive");
        tuner_ref(self).max_query_ms = a[0].f;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "min_neighbors", {"int"}, "AutoTuner", [](void* self, Args& a) {
        if (a[0].i <= 0) throw dslvm::Error("min_neighbors must be positive");
        tuner_ref(self).min_neighbors = a[0].i;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "direct_map", {"bool"}, "AutoTuner", [](void* self, Args& a) {
        tuner_ref(self).direct_map = a[0].b;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "memory_mappable", {"bool"}, "AutoTuner", [](void* self, Args& a) {
        tuner_ref(self).memory_mappable = a[0].b;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "force_pq", {"int"}, "AutoTuner", [](void* self, Args& a) {
        if (a[0].i <= 0) throw dslvm::Error("force_pq must be positive");
        tuner_ref(self).force_pq = a[0].i;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "policy_rule", {"string"}, "AutoTuner", [](void* self, Args& a) {
        tuner_ref(self).policy_rule = a[0].s;
        return Value::object("AutoTuner", self);
    });
    reg.method("AutoTuner", "plan", {}, "AutoPlan", [this](void* self, Args&) {
        AutoPlan p = tuner_ref(self).build_plan();
        return Value::object("AutoPlan", alloc_plan(p));
    });
    reg.method("AutoTuner", "candidate_keys", {}, "StrList", [&colls](void* self, Args&) {
        return Value::object("StrList", colls.make_str_list(tuner_ref(self).compute_candidates()));
    });

    // AutoPlan methods
    reg.method("AutoPlan", "index_key", {}, "string", [](void* self, Args&) {
        return Value::string(plan_ref(self).index_key);
    });
    reg.method("AutoPlan", "index_type", {}, "string", [](void* self, Args&) {
        return Value::string(plan_ref(self).index_type);
    });
    reg.method("AutoPlan", "metric", {}, "Metric", [](void* self, Args&) {
        return Value::string(plan_ref(self).metric);
    });
    reg.method("AutoPlan", "nb_vectors", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).nb_vectors);
    });
    reg.method("AutoPlan", "dim", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).dim);
    });
    reg.method("AutoPlan", "nb_clusters", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).nb_clusters);
    });
    reg.method("AutoPlan", "train_size", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).train_size);
    });
    reg.method("AutoPlan", "estimated_bytes", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).estimated_bytes);
    });
    reg.method("AutoPlan", "training_memory_bytes", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).training_memory_bytes);
    });
    reg.method("AutoPlan", "max_memory_bytes", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).max_memory_bytes);
    });
    reg.method("AutoPlan", "compression_ratio", {}, "float", [](void* self, Args&) {
        return Value::real(plan_ref(self).compression_ratio);
    });
    reg.method("AutoPlan", "recommended_params", {}, "string", [](void* self, Args&) {
        return Value::string(plan_ref(self).recommended_params);
    });
    reg.method("AutoPlan", "recommended_nprobe", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).recommended_nprobe);
    });
    reg.method("AutoPlan", "recommended_ef_search", {}, "int", [](void* self, Args&) {
        return Value::integer(plan_ref(self).recommended_ef_search);
    });
    reg.method("AutoPlan", "description", {}, "string", [](void* self, Args&) {
        return Value::string(plan_ref(self).description);
    });
    reg.method("AutoPlan", "emit", {"string"}, "AutoPlan", [this](void* self, Args& a) {
        auto& p = plan_ref(self);
        std::ostringstream oss;
        oss << "{\"label\": " << dslvm::json_escape(a[0].s)
            << ", \"index_key\": " << dslvm::json_escape(p.index_key)
            << ", \"index_type\": " << dslvm::json_escape(p.index_type)
            << ", \"metric\": " << dslvm::json_escape(p.metric)
            << ", \"nb_vectors\": " << p.nb_vectors
            << ", \"dim\": " << p.dim
            << ", \"nb_clusters\": " << p.nb_clusters
            << ", \"train_size\": " << p.train_size
            << ", \"estimated_bytes\": " << p.estimated_bytes
            << ", \"compression_ratio\": " << fmt_num(p.compression_ratio)
            << ", \"recommended_params\": " << dslvm::json_escape(p.recommended_params) << "}";
        emitted_plans.push_back(oss.str());
        return Value::object("AutoPlan", self);
    });

    // FaissIndex methods
    reg.method("FaissIndex", "dim", {}, "int", [](void* self, Args&) {
        return Value::integer(index_ref(self).dim);
    });
    reg.method("FaissIndex", "ntotal", {}, "int", [](void* self, Args&) {
        return Value::integer(static_cast<int64_t>(index_ref(self).entries.size()));
    });
    reg.method("FaissIndex", "is_trained", {}, "bool", [](void* self, Args&) {
        return Value::boolean(index_ref(self).is_trained);
    });
    reg.method("FaissIndex", "index_key", {}, "string", [](void* self, Args&) {
        return Value::string(index_ref(self).index_key);
    });
    reg.method("FaissIndex", "metric", {}, "Metric", [](void* self, Args&) {
        return Value::string(index_ref(self).metric);
    });
    reg.method("FaissIndex", "embedder", {"OnnxModel"}, "FaissIndex", [](void* self, Args& a) {
        auto& idx = index_ref(self);
        auto& m = model_ref(a[0].obj);
        if (m.output_dim != idx.dim) {
            throw dslvm::Error("OnnxModel output_dim " + std::to_string(m.output_dim) +
                               " does not match FaissIndex dim " + std::to_string(idx.dim));
        }
        idx.embedder = &m;
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "set_params", {"string"}, "FaissIndex", [](void* self, Args& a) {
        index_ref(self).apply_param_string(a[0].s);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "nprobe", {"int"}, "FaissIndex", [](void* self, Args& a) {
        if (a[0].i <= 0) throw dslvm::Error("nprobe must be positive");
        auto& idx = index_ref(self);
        idx.cur_nprobe = a[0].i;
        idx.param_str = "nprobe=" + std::to_string(a[0].i);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "ef_search", {"int"}, "FaissIndex", [](void* self, Args& a) {
        if (a[0].i <= 0) throw dslvm::Error("ef_search must be positive");
        auto& idx = index_ref(self);
        idx.cur_ef_search = a[0].i;
        idx.param_str = "efSearch=" + std::to_string(a[0].i);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "ef_construction", {"int"}, "FaissIndex", [](void* self, Args& a) {
        if (a[0].i <= 0) throw dslvm::Error("ef_construction must be positive");
        index_ref(self).cur_ef_construction = a[0].i;
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "filter_rule", {"string"}, "FaissIndex", [](void* self, Args& a) {
        index_ref(self).filter_rule = a[0].s;
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "rerank_rule", {"string"}, "FaissIndex", [](void* self, Args& a) {
        index_ref(self).rerank_rule = a[0].s;
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "clear_rules", {}, "FaissIndex", [](void* self, Args&) {
        auto& idx = index_ref(self);
        idx.filter_rule.clear();
        idx.rerank_rule.clear();
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "train", {}, "FaissIndex", [](void* self, Args&) {
        index_ref(self).train_index();
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "add", {"int", "FloatList"}, "FaissIndex", [](void* self, Args& a) {
        const auto& vec = static_cast<dslvm::stdlib::FloatList*>(a[1].obj)->items;
        index_ref(self).add_vector(a[0].i, vec, nullptr);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "add_doc", {"Document"}, "FaissIndex", [](void* self, Args& a) {
        auto& idx = index_ref(self);
        auto& doc = doc_ref(a[0].obj);
        if (doc.vec.empty()) {
            if (!idx.embedder) {
                throw dslvm::Error("add_doc: Document has no vector and FaissIndex has no attached OnnxModel embedder");
            }
            doc.vec = idx.embedder->run_text(doc.text);
        }
        idx.add_vector(doc.id, doc.vec, &doc);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "add_text", {"int", "string"}, "FaissIndex", [this](void* self, Args& a) {
        auto& idx = index_ref(self);
        if (!idx.embedder) {
            throw dslvm::Error("add_text requires an OnnxModel attached via index.embedder(model)");
        }
        Document* doc = alloc_doc(a[0].i, a[1].s);
        doc->vec = idx.embedder->run_text(a[1].s);
        idx.add_vector(doc->id, doc->vec, doc);
        return Value::object("FaissIndex", self);
    });
    reg.method("FaissIndex", "remove_id", {"int"}, "int", [](void* self, Args& a) {
        return Value::integer(index_ref(self).remove_by_id(a[0].i));
    });
    reg.method("FaissIndex", "reconstruct", {"int"}, "FloatList", [&colls](void* self, Args& a) {
        return Value::object("FloatList", colls.make_float_list(index_ref(self).reconstruct_vec(a[0].i)));
    });
    reg.method("FaissIndex", "search", {"FloatList", "int"}, "SearchResults", [](void* self, Args& a) {
        const auto& q = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        return Value::object("SearchResults", index_ref(self).run_search(q, a[1].i, ""));
    });
    reg.method("FaissIndex", "search_text", {"string", "int"}, "SearchResults", [](void* self, Args& a) {
        auto& idx = index_ref(self);
        if (!idx.embedder) {
            throw dslvm::Error("search_text requires an OnnxModel attached via index.embedder(model)");
        }
        auto q = idx.embedder->run_text(a[0].s);
        return Value::object("SearchResults", idx.run_search(q, a[1].i, a[0].s));
    });
    reg.method("FaissIndex", "range_search", {"FloatList", "float"}, "SearchResults", [](void* self, Args& a) {
        const auto& q = static_cast<dslvm::stdlib::FloatList*>(a[0].obj)->items;
        return Value::object("SearchResults", index_ref(self).run_range_search(q, a[1].f));
    });
    reg.method("FaissIndex", "autotune", {"float", "int"}, "string", [](void* self, Args& a) {
        return Value::string(index_ref(self).run_autotune(a[0].f, a[1].i));
    });
    reg.method("FaissIndex", "evaluate", {"int"}, "IndexMetrics", [](void* self, Args& a) {
        return Value::object("IndexMetrics", index_ref(self).run_evaluate(a[0].i));
    });
    reg.method("FaissIndex", "save", {"string"}, "FaissIndex", [](void* self, Args& a) {
        index_ref(self).save_to_file(a[0].s);
        return Value::object("FaissIndex", self);
    });

    // SearchResults methods
    reg.method("SearchResults", "len", {}, "int", [](void* self, Args&) {
        return Value::integer(static_cast<int64_t>(results_ref(self).hits.size()));
    });
    reg.method("SearchResults", "at", {"int"}, "SearchHit", [](void* self, Args& a) {
        auto& h = results_ref(self).hits;
        if (a[0].i < 0 || static_cast<size_t>(a[0].i) >= h.size())
            throw dslvm::Error("SearchResults.at: index out of range");
        return Value::object("SearchHit", h[static_cast<size_t>(a[0].i)]);
    });
    reg.method("SearchResults", "query_text", {}, "string", [](void* self, Args&) {
        return Value::string(results_ref(self).query_text);
    });
    reg.method("SearchResults", "params", {}, "string", [](void* self, Args&) {
        return Value::string(results_ref(self).params);
    });
    reg.method("SearchResults", "ids", {}, "IntList", [&colls](void* self, Args&) {
        std::vector<int64_t> ids;
        for (const auto* h : results_ref(self).hits) ids.push_back(h->id);
        return Value::object("IntList", colls.make_int_list(std::move(ids)));
    });
    reg.method("SearchResults", "distances", {}, "FloatList", [&colls](void* self, Args&) {
        std::vector<double> dists;
        for (const auto* h : results_ref(self).hits) dists.push_back(h->distance);
        return Value::object("FloatList", colls.make_float_list(std::move(dists)));
    });
    reg.method("SearchResults", "scores", {}, "FloatList", [&colls](void* self, Args&) {
        std::vector<double> sc;
        for (const auto* h : results_ref(self).hits) sc.push_back(h->score);
        return Value::object("FloatList", colls.make_float_list(std::move(sc)));
    });
    reg.method("SearchResults", "emit", {"string"}, "SearchResults", [this](void* self, Args& a) {
        auto& r = results_ref(self);
        std::ostringstream oss;
        oss << "{\"label\": " << dslvm::json_escape(a[0].s)
            << ", \"query_text\": " << dslvm::json_escape(r.query_text)
            << ", \"params\": " << dslvm::json_escape(r.params)
            << ", \"hits\": [";
        for (size_t i = 0; i < r.hits.size(); ++i) {
            if (i > 0) oss << ", ";
            const auto* h = r.hits[i];
            oss << "{\"rank\": " << h->rank
                << ", \"id\": " << h->id
                << ", \"distance\": " << fmt_num(h->distance)
                << ", \"score\": " << fmt_num(h->score)
                << ", \"text\": " << dslvm::json_escape(h->doc ? h->doc->text : "") << "}";
        }
        oss << "]}";
        emitted_searches.push_back(oss.str());
        return Value::object("SearchResults", self);
    });

    // SearchHit methods
    reg.method("SearchHit", "rank", {}, "int", [](void* self, Args&) {
        return Value::integer(hit_ref(self).rank);
    });
    reg.method("SearchHit", "id", {}, "int", [](void* self, Args&) {
        return Value::integer(hit_ref(self).id);
    });
    reg.method("SearchHit", "distance", {}, "float", [](void* self, Args&) {
        return Value::real(hit_ref(self).distance);
    });
    reg.method("SearchHit", "score", {}, "float", [](void* self, Args&) {
        return Value::real(hit_ref(self).score);
    });
    reg.method("SearchHit", "text", {}, "string", [](void* self, Args&) {
        auto& h = hit_ref(self);
        return Value::string(h.doc ? h.doc->text : "");
    });
    reg.method("SearchHit", "document", {}, "Document", [](void* self, Args&) {
        auto& h = hit_ref(self);
        if (!h.doc) throw dslvm::Error("SearchHit has no attached Document");
        return Value::object("Document", h.doc);
    });

    // IndexMetrics methods
    reg.method("IndexMetrics", "recall_1_at_k", {}, "float", [](void* self, Args&) {
        return Value::real(metrics_ref(self).recall_1_at_k);
    });
    reg.method("IndexMetrics", "recall_k_at_k", {}, "float", [](void* self, Args&) {
        return Value::real(metrics_ref(self).recall_k_at_k);
    });
    reg.method("IndexMetrics", "reconstruction_error", {}, "float", [](void* self, Args&) {
        return Value::real(metrics_ref(self).reconstruction_error);
    });
    reg.method("IndexMetrics", "size_bytes", {}, "int", [](void* self, Args&) {
        return Value::integer(metrics_ref(self).size_bytes);
    });
    reg.method("IndexMetrics", "compression_ratio", {}, "float", [](void* self, Args&) {
        return Value::real(metrics_ref(self).compression_ratio);
    });
    reg.method("IndexMetrics", "ms_per_query", {}, "float", [](void* self, Args&) {
        return Value::real(metrics_ref(self).ms_per_query);
    });
    reg.method("IndexMetrics", "param_string", {}, "string", [](void* self, Args&) {
        return Value::string(metrics_ref(self).param_string);
    });
    reg.method("IndexMetrics", "emit", {"string"}, "IndexMetrics", [this](void* self, Args& a) {
        auto& m = metrics_ref(self);
        std::ostringstream oss;
        oss << "{\"label\": " << dslvm::json_escape(a[0].s)
            << ", \"index_key\": " << dslvm::json_escape(m.index_key)
            << ", \"params\": " << dslvm::json_escape(m.param_string)
            << ", \"k\": " << m.k
            << ", \"recall_1_at_k\": " << fmt_num(m.recall_1_at_k)
            << ", \"recall_k_at_k\": " << fmt_num(m.recall_k_at_k)
            << ", \"reconstruction_error\": " << fmt_num(m.reconstruction_error)
            << ", \"size_bytes\": " << m.size_bytes
            << ", \"compression_ratio\": " << fmt_num(m.compression_ratio) << "}";
        emitted_metrics.push_back(oss.str());
        return Value::object("IndexMetrics", self);
    });
}

std::string Host::build_result_json() const {
    auto join_vec = [](const std::vector<std::string>& v) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << v[i];
        }
        oss << "]";
        return oss.str();
    };
    std::ostringstream rep;
    rep << "{";
    for (size_t i = 0; i < reports.size(); ++i) {
        if (i > 0) rep << ", ";
        rep << dslvm::json_escape(reports[i].first) << ": " << dslvm::json_escape(reports[i].second);
    }
    rep << "}";

    std::ostringstream oss;
    oss << "{\"models\": " << join_vec(emitted_models)
        << ", \"plans\": " << join_vec(emitted_plans)
        << ", \"searches\": " << join_vec(emitted_searches)
        << ", \"metrics\": " << join_vec(emitted_metrics)
        << ", \"reports\": " << rep.str()
        << ", \"entries_called\": " << entries_called << "}";
    return oss.str();
}

} // namespace lmsearch
