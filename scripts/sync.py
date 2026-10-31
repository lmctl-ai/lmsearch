#!/usr/bin/env python3
"""Copy reviewed SDK bytes and pin every file; no network or dependencies."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def files(directory):
    return sorted(
        p for p in directory.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
    )


def hashes(directory):
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files(directory)
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if mode not in ('sdk', 'check'):
        sys.exit('usage: sync.py sdk [SDK_DIR] [REF] | check')
    if mode == 'check':
        directory = ROOT / 'vendor/lmcode'
        lock = json.loads((ROOT / 'sdk.lock').read_text(encoding='utf-8'))
        if hashes(directory) != lock['sha256']:
            sys.exit('sdk drift: sync reviewed upstream bytes, do not edit vendored files')
        print('PASS SDK lock')
        return
    source = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT.parent / 'lmcode-cpp'
    ref = sys.argv[3] if len(sys.argv) > 3 else 'HEAD'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', ref], text=True).strip()
    paths = ['include', 'stdlib', 'std', 'LICENSE']
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', *paths], text=True)
    if dirty:
        sys.exit('commit the source SDK before syncing')
    destination = ROOT / 'vendor/lmcode'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    selected = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '--name-only', commit,
                                        '--', 'include/dslvm.hpp', 'stdlib', 'std/lmcode-std.schema.json', 'LICENSE'], text=True).splitlines()
    for path in selected:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', commit + ':' + path]))
    (ROOT / 'sdk.lock').write_text(
        json.dumps({'repo': 'lmctl-ai/lmcode-cpp', 'commit': commit, 'sha256': hashes(destination)}, indent=2) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    print(f'Pinned sdk at {commit}')


if __name__ == '__main__':
    main()
