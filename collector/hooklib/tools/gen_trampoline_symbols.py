#!/usr/bin/env python3

import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} [libcuda.so.1] [Output cuda_symbols.def] [Output cuda_trampolines.inc]", file=sys.stderr)
    sys.exit(1)

real_libcuda = sys.argv[1]
out_symbols = sys.argv[2]
out_trampolines = sys.argv[3]

out = subprocess.check_output(
    ["readelf", "-Ws", "--wide", real_libcuda],
    text=True,
)

symbols = set()

for line in out.splitlines():
    parts = line.split()
    if len(parts) < 8:
        continue

    sym_type = parts[3]
    ndx = parts[6]
    name = parts[7]

    if sym_type != "FUNC":
        continue
    if ndx == "UND":
        continue

    name = name.split("@", 1)[0]

    if not name.startswith("cu"):
        continue

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        continue

    symbols.add(name)

symbols = sorted(symbols)

Path(out_symbols).write_text(
    "".join(f"CUDA_SYMBOL({s})\n" for s in symbols)
)

Path(out_trampolines).write_text(
    "".join(f"CUDA_TRAMP {s}\n" for s in symbols)
)

print(f"Generated for {len(symbols)} CUDA symbols")
