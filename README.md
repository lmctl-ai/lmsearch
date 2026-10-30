# lmsearch

`lmsearch` is a native C++ vector & neural search CLI that integrates and extends **ONNX** (`onnx/onnx`), **FAISS** (`facebookresearch/faiss`), and **AutoFaiss** (`criteo/autofaiss`) with **`lmcode`** (`_lmcode_` / DSLB v6) as its primary extension and programmable rule mechanism.

## Architecture Overview

`lmsearch` embeds the `lmcode-cpp` stack VM (`vendor/lmcode/include/dslvm.hpp`) and exposes a domain-specific schema (`schema/lmsearch.dsl.json`) that unifies three vector search layers under a single `_lmcode_` block inside `lmscript`:

1. **ONNX Embedding & Graph Layer (`OnnxModel`)**:
   - Construct, inspect, validate (IR version, opset schemas, shape inference), persist, and execute text/token/vector embedding graphs (`onnx_text_embedder`, `onnx_mlp_embedder`, `onnx_load`).
   - Supports `MEAN`, `CLS`, and `MAX` pooling, L2 normalization, and custom `@embed` SAM rules (`implements EmbeddingTransform`).
2. **AutoFaiss Index Advisor & Hyperparameter Tuner (`AutoTuner`, `AutoPlan`, `IndexMetrics`)**:
   - Native C++ implementation of AutoFaiss's index selection heuristics (`compute_candidates`, `build_plan`), memory footprint estimator (`Flat`, `HNSW`, `IVF,Flat`, `IVF,PQ`, `OPQ,IVF,PQ`), cluster sizing, query-time parameter autotuning (`autotune`), and ground-truth recall/compression evaluation (`evaluate`).
   - Extensible via `@policy` (`implements IndexPolicy`) and `@audit` (`implements AuditRule`) SAM rules.
3. **FAISS Vector Indexing & Search Engine (`FaissIndex`, `SearchResults`, `SearchHit`, `Document`)**:
   - Supports `Flat`, `HNSW`, `IVF`, `IVFPQ`, and `OPQ` index factories across `L2`, `IP`, and `COSINE` metrics, plus vector reconstruction (`reconstruct`), range search (`range_search`), and persistence (`save` / `load_index`).
   - Extensible at query time via `@filter` (`implements VectorFilter`, acting as a programmable `faiss::IDSelector` over document metadata) and `@rerank` (`implements Reranker`).

## SAM Extension Interfaces (`dsl.v1`)

| Annotation | SAM Interface | Signature | Role |
|---|---|---|---|
| `@embed` | `EmbeddingTransform` | `(FloatList, int) -> FloatList` | Post-processes ONNX embeddings before indexing or querying |
| `@policy` | `IndexPolicy` | `(AutoPlan) -> string` | Selects or overrides candidate AutoFaiss index keys |
| `@file_filter` | `FileFilter` | `(Document) -> bool` | Ingest-time file gate & metadata enricher before embedding |
| `@config` | `SearchConfig` | `(FaissIndex) -> void` | Programmatic `lmcode` configuration of index parameters, common filters, and rules |
| `@filter` | `VectorFilter` | `(int, Document) -> bool` | Programmable `faiss::IDSelector` predicate during FAISS search |
| `@rerank` | `Reranker` | `(SearchHit, string) -> float` | Rescores candidate FAISS hits using document fields and query text |
| `@audit` | `AuditRule` | `(IndexMetrics) -> void` | Post-run quality gate invoked on every evaluated `IndexMetrics` |

## Quick Start

### Build

```bash
make            # Portable C++17 binary at build/lmsearch
make test       # Verify SDK lock, schema parity, and all .lms suites
```

To also link against the locally built upstream `libfaiss.a` and `libonnx.a` static libraries in `third_party/{faiss,onnx}`:

```bash
make with-upstream   # Produces build/lmsearch-upstream
```

### Run an `lmscript` Program

```bash
lmscript examples/basic_search.lms
lmscript examples/file_and_dir_search.lms
lmscript examples/hairy_search_filters.lms
lmscript examples/autofaiss_tuning.lms
lmscript examples/sam_rules.lms
lmscript examples/persistence_and_quantization.lms
```

### Direct CLI: Index Files/Directories, Common Filter Options & `lmcode` Callbacks

```bash
# 1. Index a list of files and directories (downloads & caches ONNX model weights in ~/.cache/lmsearch/models)
./build/lmsearch index README.md durable-memory examples \
    --output build/docs.faiss \
    --model all-MiniLM-L6-v2 \
    --ext .md,.lms \
    --exclude-path draft,legacy

# 2. Search text with common CLI filter options
./build/lmsearch search \
    --index build/docs.faiss \
    --query "AutoFaiss memory budget and ONNX embedding" \
    --ext .md \
    --path-prefix durable-memory \
    --contains AutoFaiss \
    --min-score 0.10 \
    -k 3

# 3. Index or search with hairy lmcode callbacks (@config, @file_filter, @filter, @rerank)
./build/lmsearch search \
    --index build/hairy_corpus.faiss \
    --query "faiss hnsw vector search" \
    --dsl build/hairy_search_filters.dslb \
    -k 5
```

### CLI Discovery Flags

```bash
./build/lmsearch --version
./build/lmsearch --dsl-schema
./build/lmsearch --dsl-signatures
./build/lmsearch --schema --json
./build/lmsearch run --program program.dslb --dryrun
```

## Releases

Pushes to `main` automatically build, test, and publish native executables for:
- `lmsearch-linux-x64`
- `lmsearch-linux-arm64`
- `lmsearch-macos-x64`
- `lmsearch-macos-arm64`
- `lmsearch-windows-x64.exe`
