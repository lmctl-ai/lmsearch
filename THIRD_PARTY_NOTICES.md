# Third-Party Notices

`lmsearch` incorporates or extends designs, schemas, and components from the following projects:

## 1. lmcode-cpp
- **Repository**: https://github.com/lmctl-ai/lmcode-cpp
- **License**: MIT License
- **Files**: `vendor/lmcode/` (pinned via `sdk.lock`)

## 2. ONNX (Open Neural Network Exchange)
- **Repository**: https://github.com/onnx/onnx
- **License**: Apache License 2.0
- **Usage**: Neural network IR, operator schemas, shape inference, checker, and embedding execution (`OnnxModel`).

## 3. FAISS (Facebook AI Similarity Search)
- **Repository**: https://github.com/facebookresearch/faiss
- **License**: MIT License
- **Usage**: Dense vector similarity search, Flat / HNSW / IVF / PQ / OPQ indexing, vector reconstruction, and range search (`FaissIndex`).

## 4. AutoFaiss
- **Repository**: https://github.com/criteo/autofaiss
- **License**: Apache License 2.0
- **Usage**: Automated index factory key selection, memory-budget estimation, cluster heuristics, hyperparameter autotuning, and recall/compression evaluation (`AutoTuner`, `AutoPlan`, `IndexMetrics`).
