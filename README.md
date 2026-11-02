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
| `@ignore` | `IgnoreRule` | `(Document) -> bool` | Directory subtree & file prune predicate (alongside `.gitignore` / `.lmsearchignore`) |
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
    --lmcode build/hairy_search_filters.dslb \
    -k 5
```

### Single-User REST Web Server (`serve`) & Client (`client`) with Lazy In-Memory Index

`lmsearch serve` launches a single-user, global-session REST server (no user auth or session ID required) designed for interactive query -> filter refinement -> requery workflows:

1. **Weights-Only Disk Cache (`--weights-only-cache` / `--clean-start`)**: Only ONNX model weights (`model.onnx`) are cached on disk (`~/.cache/lmsearch/models/...`). Every server start begins with a clean in-memory index and result cache.
2. **Memory as a Lazy Index**: File/directory paths passed to `lmsearch serve` (or added via `/paths?add=...`) are cataloged lazily in memory without reading or embedding files upfront.
3. **Broad Query Result Cache + Fast Refined Filter View**:
   - `GET /query?q=<TEXT>&k=<K>` (or `lmsearch client query`) lazily embeds only candidate files matching the broad filter that are not yet in RAM, runs FAISS similarity search, and caches the full broad result set in server memory (`cached_broad_hits`).
   - `GET /filter?...` (or `lmsearch client filter`) modifies the active filter (`--ext`, `--path-prefix`, `--path-contains`, `--exclude-path`, `--contains`, `--field`, `--min-score`, `--max-distance`, `--min-size`, `--max-size`, `--lmcode`, or `--reset`) on the global session.
   - `GET /requery?k=<K>` / `GET /view?k=<K>` (or `lmsearch client requery`) treats the cached broad search results in server memory as a fast view (`from_memory_cache: true`, `lazy_newly_embedded: 0`) and returns refined hits immediately without touching disk or running ONNX embeddings.
4. **Client Shutdown & Idle Timeout Auto-Shutdown**:
   - Shut down explicitly via `lmsearch client shutdown` (`GET /shutdown`), or automatically when idle for `--timeout <SECONDS>` (or `--timeout-ms <MS>`).

```bash
# 1. Launch single-user REST server in background (clean start, weights-only disk cache, 60s idle timeout)
./build/lmsearch serve README.md durable-memory examples \
    --port 8080 --timeout 60 --weights-only-cache &

# 2. Broad query (lazily embeds matching candidate files in RAM & caches broad results)
./build/lmsearch client --port 8080 query --query "ONNX and FAISS vector search" -k 5

# 3. Refine filter on the global session
./build/lmsearch client --port 8080 filter --ext .md --path-prefix README --contains lmsearch

# 4. Requery fast refined view directly from server memory cache (0 new embeddings)
./build/lmsearch client --port 8080 requery -k 5

# 5. Shut down server explicitly (or let --timeout shut it down automatically)
./build/lmsearch client --port 8080 shutdown
```

### CLI Discovery Flags

```bash
./build/lmsearch --version
./build/lmsearch --lmcode-schema
./build/lmsearch --lmcode-signatures
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
