#!/usr/bin/env bash
# Re-run the Stage A and Stage B analysis from the saved draws and require the committed results.
#   ./reproduce.sh            uses python3 from PATH
#   PY="uv run --managed-python --python 3.12 --with torch --with numpy --with scipy --with safetensors --with transformers==5.18.0 python" ./reproduce.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cp -R "$HERE/code" "$HERE/full" "$HERE/data" "$W/"
rm -f "$W"/full/stagea/data/ANALYSIS_*
cd "$W/full"
$PY stagea_analyze.py stagea/data > "$W/a.log"
$PY stagea_analyze.py stagea/data --stage-b > "$W/b.log"
python3 - "$HERE/full/stagea/data" "$W/full/stagea/data" <<'PYEOF'
import json, math, sys
ref, new = sys.argv[1:3]


def diff(a, b, path, out):
    """Append every difference to out. Floats compare to 1e-9 relative (p-values can differ in the last digit
    across numerical libraries); everything else compares exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            out.append(f"{path}: keys differ")
        for k in a.keys() & b.keys():
            diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or math.isclose(a, b, rel_tol=1e-9)):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: {a!r} != {b!r}")


# Fields the analysis code gained after the committed Stage A run; only these may be new, and no planned control may be
# missing.
ADDED = {"ANALYSIS_A.json": {"controls_planned", "controls_missing"}}
bad = 0
for name in ("ANALYSIS_A.json", "ANALYSIS_B.json"):
    out = []
    old, cur = json.load(open(f"{ref}/{name}")), json.load(open(f"{new}/{name}"))
    for k in ADDED.get(name, ()):
        if k not in old:
            if k not in cur:
                out.append(f"{name}.{k}: missing from the re-run")
                continue
            v = cur.pop(k)
            if k == "controls_missing" and v != []:
                out.append(f"{name}.{k}: {v!r}")
    diff(old, cur, name, out)
    print(f"ok {name}" if not out else f"MISMATCH {name} ({len(out)} differences)", *out[:10], sep="\n  ")
    bad += bool(out)
a = json.load(open(f"{new}/ANALYSIS_A.json"))
b = json.load(open(f"{new}/ANALYSIS_B.json"))
print("Stage A: grid rejected", a["grid_rejected"], "| nulls rejected", a["nulls_rejected"])
print("Stage B: confirmed", len(b["confirmed"]), "cells | nulls rejected", b["nulls_rejected"])
sys.exit(1 if bad else 0)
PYEOF
