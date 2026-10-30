#pragma once

#include "dslvm.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lmsearch {

struct Host;

struct OnnxModel {
    Host* host = nullptr;
    int64_t opset = 19;
    int64_t ir_version = 9;
    int64_t vocab_size = 256;
    int64_t input_dim = 16;
    int64_t hidden_dim = 32;
    int64_t output_dim = 16;
    int64_t seed = 42;
    bool is_mlp = false;
    std::string pooling = "MEAN";
    bool normalize = true;
    std::string transform_rule;
    std::string model_id = "custom";
    std::string cache_path;
    bool from_cache = false;
    std::vector<std::string> node_ops;
    // Deterministic weights:
    // For text/token embedder: token_table [vocab_size * output_dim], proj [output_dim * output_dim], bias [output_dim]
    // For MLP embedder: w1 [input_dim * hidden_dim], b1 [hidden_dim], w2 [hidden_dim * output_dim], b2 [output_dim]
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    std::vector<double> b2;

    void init_text_embedder(int64_t vocab, int64_t dim, int64_t s);
    void init_mlp_embedder(int64_t in_d, int64_t hid_d, int64_t out_d, int64_t s);
    bool check_valid() const;
    std::string summary_text() const;
    std::vector<double> run_tokens(const std::vector<int64_t>& tokens) const;
    std::vector<double> run_text(const std::string& text) const;
    std::vector<double> run_vector(const std::vector<double>& input) const;
    void save_to_file(const std::string& path) const;
    static OnnxModel load_from_file(Host* host, const std::string& path);
    static OnnxModel load_or_download_cached(
        Host* host,
        const std::string& model_id_or_url,
        int64_t requested_dim = 0,
        const std::string& custom_cache_dir = "");
};

struct Document {
    int64_t id = 0;
    std::string text;
    std::map<std::string, std::string> fields;
    std::map<std::string, int64_t> numbers;
    std::vector<double> vec;
};

struct AutoPlan {
    std::string index_key = "Flat";
    std::string index_type = "FLAT";
    std::string metric = "L2";
    int64_t nb_vectors = 0;
    int64_t dim = 0;
    int64_t nb_clusters = 1;
    int64_t train_size = 0;
    int64_t estimated_bytes = 0;
    int64_t training_memory_bytes = 0;
    int64_t max_memory_bytes = 0;
    double compression_ratio = 1.0;
    std::string recommended_params;
    int64_t recommended_nprobe = 1;
    int64_t recommended_ef_search = 32;
    std::string description;
};

struct AutoTuner {
    Host* host = nullptr;
    int64_t nb_vectors = 0;
    int64_t dim = 0;
    std::string metric = "L2";
    std::string max_memory = "16G";
    double max_query_ms = 10.0;
    int64_t min_neighbors = 20;
    bool direct_map = false;
    bool memory_mappable = false;
    int64_t force_pq = 0;
    std::string policy_rule;

    std::vector<std::string> compute_candidates() const;
    AutoPlan build_plan() const;
};

struct SearchHit {
    int64_t rank = 0;
    int64_t id = -1;
    double distance = 0.0;
    double score = 0.0;
    Document* doc = nullptr;
};

struct SearchResults {
    std::string query_text;
    std::string params;
    std::vector<SearchHit*> hits;
};

struct IndexMetrics {
    std::string index_key;
    std::string param_string;
    int64_t k = 10;
    double recall_1_at_k = 1.0;
    double recall_k_at_k = 1.0;
    double reconstruction_error = 0.0;
    int64_t size_bytes = 0;
    double compression_ratio = 1.0;
    double ms_per_query = 0.0;
};

struct FaissIndex {
    Host* host = nullptr;
    int64_t dim = 0;
    std::string index_key = "Flat";
    std::string metric = "L2";
    bool is_trained = true;
    int64_t nlist = 1;
    int64_t m_hnsw = 32;
    int64_t m_pq = 0;
    int64_t nbits_pq = 8;
    int64_t opq_m = 0;
    int64_t opq_d = 0;
    int64_t cur_nprobe = 1;
    int64_t cur_ef_search = 32;
    int64_t cur_ef_construction = 40;
    int64_t cur_ht = 2048;
    std::string param_str;

    OnnxModel* embedder = nullptr;
    std::string filter_rule;
    std::string rerank_rule;

    struct Entry {
        int64_t id = 0;
        std::vector<double> raw_vec;
        std::vector<double> norm_vec;
        std::vector<uint8_t> pq_codes;
        int64_t list_id = 0;
        Document* doc = nullptr;
    };
    std::vector<Entry> entries;
    // Centroids for IVF: [nlist * dim]
    std::vector<std::vector<double>> centroids;
    // PQ codebooks: [m_pq][ksub][dsub]
    std::vector<std::vector<std::vector<double>>> pq_centroids;

    void parse_key();
    void apply_param_string(const std::string& s);
    std::string effective_params() const;
    void train_index();
    void add_vector(int64_t id, const std::vector<double>& vec, Document* doc);
    int64_t add_file_path(const std::string& file_path);
    int64_t add_directory_path(const std::string& dir_path, const std::string& ext_filter = "");
    int64_t add_path_list(const std::vector<std::string>& paths, const std::string& ext_filter = "");
    int64_t remove_by_id(int64_t id);
    std::vector<double> reconstruct_vec(int64_t id) const;
    std::vector<double> quantize_reconstruct(const Entry& e) const;
    SearchResults* run_search(const std::vector<double>& query, int64_t k, const std::string& qtext);
    SearchResults* run_range_search(const std::vector<double>& query, double radius);
    std::string run_autotune(double max_query_ms, int64_t min_neighbors);
    IndexMetrics* run_evaluate(int64_t k);
    void save_to_file(const std::string& path) const;
    static FaissIndex* load_from_file(Host* host, const std::string& path);
};

struct Host {
    dslvm::Registry registry;
    dslvm::stdlib::Collections cb_collections;
    std::string bytecode_bytes;
    std::string schema_bytes;
    bool in_callback = false;
    size_t entries_called = 0;

    std::vector<std::unique_ptr<OnnxModel>> models;
    std::vector<std::unique_ptr<Document>> docs;
    std::vector<std::unique_ptr<AutoTuner>> tuners;
    std::vector<std::unique_ptr<AutoPlan>> plans;
    std::vector<std::unique_ptr<FaissIndex>> indexes;
    std::vector<std::unique_ptr<SearchHit>> hits;
    std::vector<std::unique_ptr<SearchResults>> results_arena;
    std::vector<std::unique_ptr<IndexMetrics>> metrics_arena;

    std::vector<std::string> emitted_models;
    std::vector<std::string> emitted_plans;
    std::vector<std::string> emitted_searches;
    std::vector<std::string> emitted_metrics;
    std::vector<std::pair<std::string, std::string>> reports;
    std::vector<IndexMetrics*> evaluated_metrics;

    Host();
    void bind_all(dslvm::Registry& reg, bool callback_mode);
    dslvm::Context context();

    OnnxModel* alloc_model();
    Document* alloc_doc(int64_t id, const std::string& text = "");
    AutoTuner* alloc_tuner(int64_t nb_vectors, int64_t dim);
    AutoPlan* alloc_plan(const AutoPlan& p);
    FaissIndex* alloc_index(int64_t dim, const std::string& key, const std::string& metric);
    SearchHit* alloc_hit(const SearchHit& h);
    SearchResults* alloc_results();
    IndexMetrics* alloc_metrics(const IndexMetrics& m);

    // Invoke an annotated rule entry by function name in an isolated callback registry/program.
    dslvm::Value invoke_rule(
        const std::string& rule_name,
        const std::string& expected_iface_prefix,
        std::vector<dslvm::Value> args,
        dslvm::Registry** out_cb_reg = nullptr,
        std::unique_ptr<dslvm::Registry>* owned_cb_reg = nullptr);

    std::string build_result_json() const;
};

int64_t parse_memory_bytes(const std::string& spec);
std::string format_memory_bytes(int64_t bytes);

} // namespace lmsearch
