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

echo "==> 2. Checking CLI discovery flags (--version, --dsl-schema, --schema, --dsl-signatures)"
"$BIN" --version | tr -d '\r' | grep -q "^lmsearch "
cmp -s <("$BIN" --dsl-schema | tr -d '\r') <(tr -d '\r' < schema/lmsearch.dsl.json)
cmp -s <("$BIN" --dsl-schema --json | tr -d '\r') <(tr -d '\r' < schema/lmsearch.dsl.json)
cmp -s <("$BIN" --schema --json | tr -d '\r') <(tr -d '\r' < cli.schema.latest.json)
diff -u <(lmscript --dsl-signatures schema/lmsearch.dsl.json | tr -d '\r') <("$BIN" --dsl-signatures | tr -d '\r')

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
BAD_OUT=$("$BIN" --dsl build/bad.dslb 2>&1)
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

if [ -x "./build/lmsearch-upstream" ]; then
    echo "==> 6. Verifying upstream FAISS + ONNX static-linked binary (build/lmsearch-upstream)"
    ./build/lmsearch-upstream --version | grep -q "^lmsearch "
    diff -u <(lmscript --dsl-signatures schema/lmsearch.dsl.json) <(./build/lmsearch-upstream --dsl-signatures)
fi

echo "PASS: all lmsearch tests succeeded"
