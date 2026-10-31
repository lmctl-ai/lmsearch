#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v lmscript >/dev/null 2>&1; then
    if [ -x "/home/ubuntu/.nvm/versions/node/v24.15.0/bin/lmscript" ]; then
        export PATH="/home/ubuntu/.nvm/versions/node/v24.15.0/bin:$PATH"
    fi
fi

BIN="${BIN:-./build/lmsearch}"
export LMSEARCH_BIN="$BIN"

echo "==> 1. Verifying SDK lock"
python3 scripts/sync.py check
python3 tests/check-host-flags.py schema/lmsearch.dsl.json "$BIN"
python3 tests/check-callback-flags.py "$BIN"

echo "==> 2. Checking CLI discovery flags (--version, --lmcode-schema, --schema, --lmcode-signatures)"
"$BIN" --version | tr -d '\r' | grep -q "^lmsearch "
cmp -s <("$BIN" --lmcode-schema | tr -d '\r') <(tr -d '\r' < schema/lmsearch.dsl.json)
cmp -s <("$BIN" --lmcode-schema --json | tr -d '\r') <(tr -d '\r' < schema/lmsearch.dsl.json)
cmp -s <("$BIN" --schema --json | tr -d '\r') <(tr -d '\r' < cli.schema.latest.json)
diff -u <(lmscript --lmcode-signatures schema/lmsearch.dsl.json | tr -d '\r') <("$BIN" --lmcode-signatures | tr -d '\r')

echo "==> 3. Running test.lms and examples/*.lms via lmscript"
lmscript --no-preflight test.lms > build/test_out.json
grep -q '"ok": true' build/test_out.json
grep -q '"all_assertions_passed"' build/test_out.json

for ex in examples/*.lms; do
    echo "    running $ex"
    lmscript --no-preflight "$ex" > "build/$(basename "$ex" .lms).out.json"
    grep -q '"ok": true' "build/$(basename "$ex" .lms).out.json"
done

echo "==> 4. Verifying negative controls (corrupt bytecode)"
printf 'NOT_DSLB' > build/bad.dslb
set +e
BAD_OUT=$("$BIN" --lmcode build/bad.dslb 2>&1)
BAD_RC=$?
set -e
if [ "$BAD_RC" -eq 0 ]; then
    echo "FAIL: expected non-zero exit code for corrupt bytecode" >&2
    exit 1
fi
echo "$BAD_OUT" | grep -q '"LOAD_BAD_BYTECODE"'

echo "==> 5. Verifying readonly policy refusal"
set +e
RO_OUT=$("$BIN" run --program build/bad.dslb --readonly 2>&1)
RO_RC=$?
set -e
if [ "$RO_RC" -eq 0 ]; then
    echo "FAIL: expected non-zero exit code for readonly refusal" >&2
    exit 1
fi
echo "$RO_OUT" | grep -q '"POLICY_READONLY"'

echo "==> 6. Verifying CLI 'index' (files + directories + model weight cache) and 'search' (text query)"
"$BIN" index README.md durable-memory --output build/cli_files.faiss --model all-MiniLM-L6-v2 --cache-dir build/cli_model_cache > build/cli_index_out.json
grep -q '"ok": true' build/cli_index_out.json
grep -q '"cached": true' build/cli_index_out.json
"$BIN" search --index build/cli_files.faiss --query "ONNX and FAISS vector similarity search" -k 2 > build/cli_search_out.json
grep -q '"ok": true' build/cli_search_out.json
grep -q '"operation": "search"' build/cli_search_out.json
grep -q '"path":' build/cli_search_out.json

echo "==> 7. Verifying CLI common filter flags AND lmcode (--lmcode) config/callbacks on index and search"
"$BIN" search --index build/cli_files.faiss --query "ONNX and FAISS vector search" --ext .md --path-prefix README --contains lmsearch -k 5 > build/cli_filtered_search.json
grep -q '"ok": true' build/cli_filtered_search.json
grep -q '"filename": "README.md"' build/cli_filtered_search.json

"$BIN" index build/hairy_corpus_ann.md build/hairy_corpus_engine.cpp build/hairy_corpus_draft.md build/hairy_corpus_legacy.md \
    --output build/cli_hairy.faiss --model all-MiniLM-L6-v2 --dim 16 --cache-dir build/cli_model_cache \
    --lmcode build/hairy_search_filters.dslb > build/cli_hairy_index.json
grep -q '"ok": true' build/cli_hairy_index.json
grep -q '"files_indexed": 2' build/cli_hairy_index.json
grep -q '"file_filter_rule": "ingest_gate"' build/cli_hairy_index.json

"$BIN" search --index build/cli_hairy.faiss --query "faiss hnsw vector search" \
    --lmcode build/hairy_search_filters.dslb -k 5 > build/cli_hairy_search.json
grep -q '"ok": true' build/cli_hairy_search.json
grep -q '"filter_rule": "complex_doc_predicate"' build/cli_hairy_search.json
grep -q '"rerank_rule": "multi_signal_rerank"' build/cli_hairy_search.json

if [ -x "./build/lmsearch-upstream" ]; then
    echo "==> 6. Verifying upstream FAISS + ONNX static-linked binary (build/lmsearch-upstream)"
    ./build/lmsearch-upstream --version | grep -q "^lmsearch "
    diff -u <(lmscript --lmcode-signatures schema/lmsearch.dsl.json) <(./build/lmsearch-upstream --lmcode-signatures)
fi

echo "PASS: all lmsearch tests succeeded"
