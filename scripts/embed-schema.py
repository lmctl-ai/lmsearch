#!/usr/bin/env python3
"""Embed exact DSL and CLI schema bytes: no runtime path discovery or JSON dependency."""
from pathlib import Path
import sys

dsl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
cli_path = Path(sys.argv[3]) if len(sys.argv) > 3 else dsl_path.parent.parent / "cli.schema.latest.json"

dsl_text = dsl_path.read_bytes().decode("utf-8")
cli_text = cli_path.read_bytes().decode("utf-8")
assert ')LMSEARCH_DSL"' not in dsl_text
assert ')LMSEARCH_CLI"' not in cli_text

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    '#pragma once\n'
    'namespace lmsearch {\n'
    'inline constexpr char schema[] = R"LMSEARCH_DSL(' + dsl_text + ')LMSEARCH_DSL";\n'
    'inline constexpr char cli_schema[] = R"LMSEARCH_CLI(' + cli_text + ')LMSEARCH_CLI";\n'
    '}\n',
    encoding="utf-8",
    newline="\n",
)
