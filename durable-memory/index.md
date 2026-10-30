# lmsearch Durable Memory Index

Start here before changing `lmsearch`.

## 1. Purpose and Scope

`lmsearch` is a C++ host binary for `lmscript` (`_lmcode_` blocks, DSLB v6) that combines:
- **ONNX** (`onnx/onnx`) — neural graph construction, IR/opset validation, shape inference, and text/token/vector embedding execution (`OnnxModel`).
- **FAISS** (`facebookresearch/faiss`) — vector similarity indexing (`Flat`, `HNSW`, `IVF`, `IVFPQ`, `OPQ`), k-NN search, range search, vector reconstruction, and index persistence (`FaissIndex`).
- **AutoFaiss** (`criteo/autofaiss`) — automated FAISS index factory selection under memory and latency constraints, hyperparameter tuning, and recall/compression evaluation (`AutoTuner`, `AutoPlan`, `IndexMetrics`).
- **`lmcode-cpp`** (`vendor/lmcode/`) — embedded stack VM (`dslvm.hpp`) and standard collections (`lmcode-std`), pinned via `sdk.lock`.

## 2. Repository Map

| Path | Role |
|---|---|
| `schema/lmsearch.dsl.json` | Merged `dsl.v1` schema (`lmcode-std` + `lmsearch` domain types, functions, methods, enums, and 5 SAM interfaces) |
| `cli.schema.latest.json` | `cli.schema.v1` operational contract (`run`, `--dryrun`, `--readonly`) |
| `src/lmsearch.hpp` | Host domain structs (`OnnxModel`, `Document`, `AutoTuner`, `AutoPlan`, `FaissIndex`, `SearchResults`, `SearchHit`, `IndexMetrics`, `Host`) |
| `src/engine.cpp` | ONNX embedding runtime, AutoFaiss C++ planner/autotuner, and FAISS index engine (with optional `LMSEARCH_WITH_UPSTREAM_FAISS_ONNX` static-link bridge) |
| `src/bindings.cpp` | `dslvm::Registry` bindings for every function/method in `schema/lmsearch.dsl.json` + isolated callback VM execution (`Host::invoke_rule`) |
| `src/main.cpp` | CLI entry point (`--version`, `--dsl-schema`, `--dsl-signatures`, `--schema`, `--dsl`, `run --program`) and post-body `@audit` rule dispatch |
| `scripts/sync.py` | Vendor sync & lock verification against `lmcode-cpp` |
| `scripts/embed-schema.py` | Generates `build/lmsearch_schema.hpp` from `schema/lmsearch.dsl.json` and `cli.schema.latest.json` |
| `examples/*.lms` | Executable `lmscript` examples covering ONNX, FAISS, AutoFaiss, and SAM rules |
| `test.lms`, `test.sh` | End-to-end verification suite run locally and in GitHub Actions |
| `.github/workflows/release.yml` | 5-OS matrix build, test, and GitHub Release publisher |

## 3. Invariants

1. **Schema-Registry Parity**: `diff -u <(lmscript --dsl-signatures schema/lmsearch.dsl.json) <(build/lmsearch --dsl-signatures)` must always be empty.
2. **SDK Lock**: `python3 scripts/sync.py check` must pass before every build/release.
3. **SAM Callback Isolation**: Inline SAM callbacks (`@embed`, `@policy`, `@filter`, `@rerank`) execute in an isolated callback `dslvm::Program` + `dslvm::Registry` so `lmcode-std`'s single-active-run guard is respected and main-body collection handles remain valid.
