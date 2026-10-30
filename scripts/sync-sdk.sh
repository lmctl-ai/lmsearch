#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/sync.py sdk "${1:-../lmcode-cpp}"
