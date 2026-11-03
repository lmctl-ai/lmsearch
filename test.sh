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
lmscript test.lms > build/test_out.json
grep -qE '"ok": ?true' build/test_out.json
grep -q '"all_assertions_passed"' build/test_out.json

for ex in examples/*.lms; do
    echo "    running $ex"
    lmscript "$ex" > "build/$(basename "$ex" .lms).out.json"
    grep -qE '"ok": ?true' "build/$(basename "$ex" .lms).out.json"
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

echo "==> 8. Verifying REST server (single-user global session, lazy memory index, refined filter view, client shutdown & idle timeout)"
rm -f build/serve.port build/serve_out.json build/serve_timeout.port build/serve_timeout_out.json
"$BIN" serve README.md durable-memory \
    --port 0 --port-file build/serve.port --timeout 30 --weights-only-cache \
    --model all-MiniLM-L6-v2 --cache-dir build/cli_model_cache > build/serve_out.json &
SERVE_PID=$!

for _ in $(seq 1 50); do
    if [ -s build/serve.port ]; then
        break
    fi
    sleep 0.1
done
SERVE_PORT=$(tr -d '\r\n' < build/serve.port)

"$BIN" client --port "$SERVE_PORT" status > build/client_status.json
grep -q '"session": "global"' build/client_status.json
grep -q '"auth_required": false' build/client_status.json
grep -q '"clean_start": true' build/client_status.json
grep -q '"weights_only_disk_cache": true' build/client_status.json
grep -q '"lazy_embedded_docs": 0' build/client_status.json

"$BIN" client --port "$SERVE_PORT" query --query "ONNX and FAISS vector search" -k 5 > build/client_query.json
grep -q '"operation": "query"' build/client_query.json
grep -q '"from_memory_cache": false' build/client_query.json
grep -q '"cached_broad_hits":' build/client_query.json

"$BIN" client --port "$SERVE_PORT" filter --ext .md --path-prefix README --contains lmsearch > build/client_filter.json
grep -q '"operation": "filter"' build/client_filter.json
grep -q '"path_prefix": "README"' build/client_filter.json

"$BIN" client --port "$SERVE_PORT" requery -k 5 > build/client_requery.json
grep -q '"operation": "requery"' build/client_requery.json
grep -q '"from_memory_cache": true' build/client_requery.json
grep -q '"lazy_newly_embedded": 0' build/client_requery.json
grep -q '"filename": "README.md"' build/client_requery.json

cat > build/rest_client_dsl.lms <<EOF
p = _lmcode_ schema="../schema/lmsearch.dsl.json";
let c = rest_client("http://127.0.0.1:${SERVE_PORT}");
assert_true(str_contains(c.status(), "\"session\": \"global\""), "RestClient.status must report global session");
c.clear_filter();
let qres = c.query("ONNX and FAISS vector search", 5);
assert_true(!c.from_cache(), "RestClient.query must not be from cache");
assert_true(qres.len() >= 2, "RestClient.query must return >= 2 hits");
c.filter_ext(".md").filter_path_prefix("README").filter_text_contains("lmsearch");
let vres = c.requery(5);
assert_true(c.from_cache(), "RestClient.requery must come from server memory cache");
assert_true(c.newly_embedded() == 0, "RestClient.requery must embed 0 new files");
assert_true(vres.len() == 1, "RestClient.requery must return 1 filtered hit");
assert_true(vres.at(0).document().filename() == "README.md", "RestClient.requery hit must be README.md");
assert_true(c.shutdown(), "RestClient.shutdown must return true");
report("rest_client_dsl", "passed");
end _lmcode_
bin = env("LMSEARCH_BIN", "./build/lmsearch");
r = run_lmcode(p, {command: [bin, "--lmcode", "\$DSLFILE"]});
assert(r.ok, "rest_client_dsl failed");
println(json_stringify(r));
EOF
lmscript build/rest_client_dsl.lms > build/rest_client_dsl.out.json
grep -q '"passed"' build/rest_client_dsl.out.json
wait "$SERVE_PID"
grep -q '"shutdown_reason": "client_shutdown"' build/serve_out.json

"$BIN" serve README.md --port 0 --port-file build/serve_timeout.port --timeout-ms 400 \
    --weights-only-cache --model all-MiniLM-L6-v2 --cache-dir build/cli_model_cache > build/serve_timeout_out.json &
TIMEOUT_PID=$!
wait "$TIMEOUT_PID"
grep -q '"shutdown_reason": "idle_timeout"' build/serve_timeout_out.json

if [ -x "./build/lmsearch-upstream" ]; then
    echo "==> 6. Verifying upstream FAISS + ONNX static-linked binary (build/lmsearch-upstream)"
    ./build/lmsearch-upstream --version | grep -q "^lmsearch "
    diff -u <(lmscript --lmcode-signatures schema/lmsearch.dsl.json) <(./build/lmsearch-upstream --lmcode-signatures)
fi

echo "PASS: all lmsearch tests succeeded"
