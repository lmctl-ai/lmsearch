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
| `@policy` | `IndexPolicy` | `(AutoPlan) -> bool` | Accepts or rejects candidate AutoFaiss index plans |
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
lmscript examples/autofaiss_tuning.lms
lmscript examples/sam_rules.lms
lmscript examples/persistence_and_quantization.lms
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
