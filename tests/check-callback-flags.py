#!/usr/bin/env python3
"""Retired callback options must fail as usage, before index/cache/file effects."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='search-callback-flags-') as tmp:
    root = Path(tmp)
    source = root / 'input.txt'
    source.write_text('offline callback control')
    missing = root / 'must-not-open.dslb'
    output = root / 'must-not-create.faiss'
    cache = root / 'cache'
    env = dict(os.environ, HOME=str(root))
    routes = [ ['index', str(source), '--output', str(output), '--cache-dir', str(cache)],
               ['search', '--index', str(output), '--query', 'offline'], ['run'] ]
    old = '--' + 'dsl'
    failures = []
    for args in routes:
        r = subprocess.run([str(binary), *args, old, str(missing)], env=env, capture_output=True)
        if r.returncode != 3 or b'CLI_USAGE' not in r.stdout + r.stderr:
            failures.append((args[0], r.returncode, 'retired callback accepted'))
        if output.exists() or cache.exists():
            failures.append((args[0], 'retired callback caused file effects'))
    assert not failures, failures
    schema = subprocess.run([str(binary), '--schema', '--json'], capture_output=True, check=True)
    for op in json.loads(schema.stdout)['operations']:
        names = {p['name'] for p in op.get('params', [])}
        assert 'dsl' not in names, (op['name'], names)
        if op['name'] in ['index', 'search']:
            assert 'lmcode' in names, names
    # Search loads its index before callbacks: create a real offline fixture so
    # a missing-index error cannot stand in for testing the new callback route.
    fixture = subprocess.run([str(binary), *routes[0]], env=env, capture_output=True)
    assert fixture.returncode == 0, fixture
    for args in routes:
        r = subprocess.run([str(binary), *args, '--lmcode', str(missing)], env=env, capture_output=True)
        assert r.returncode in (3, 4) and str(missing).encode() in r.stdout + r.stderr, r
print('PASS index/search/run callback routes and published parameter names; no retired aliases')
