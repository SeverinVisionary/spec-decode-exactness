#!/usr/bin/env bash
# reproduce.sh must fail when a saved draw is altered: move one draw between two outcomes in a Stage B null chunk.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
"$HERE/reproduce.sh" > "$T/clean.log" 2>&1 || { cat "$T/clean.log"; echo "reproduce.sh fails on the unaltered data"; exit 1; }
cp -R "$HERE" "$T/repo"
python3 - "$T/repo/full/stagea/data/stageB_trtllm_fa01/chunks" <<'PYEOF'
import gzip, json, os, sys
d = sys.argv[1]
cell = sorted(c for c in os.listdir(d) if "__NULL__" in c)[0]
p = os.path.join(d, cell, "c00.json.gz")
r = json.load(gzip.open(p, "rt"))
i = max(range(len(r["counts"])), key=r["counts"].__getitem__)
j = min(range(len(r["counts"])), key=r["counts"].__getitem__)
r["counts"][i] -= 1
r["counts"][j] += 1
with gzip.open(p, "wt") as f:
    json.dump(r, f)
print("tampered", cell, "c00: moved one draw from outcome", i, "to", j)
PYEOF
if "$T/repo/reproduce.sh" > "$T/tampered.log" 2>&1; then
  echo "NEGATIVE CONTROL FAILED: reproduce.sh passed on altered data"
  exit 1
fi
grep -q "^MISMATCH ANALYSIS_B.json" "$T/tampered.log" || { cat "$T/tampered.log"; echo "reproduce.sh failed for another reason"; exit 1; }
echo "negative control ok: reproduce.sh passes on the saved data and reports a mismatch on the altered data"
