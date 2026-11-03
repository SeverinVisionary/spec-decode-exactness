"""FR22 E5 analysis.

    python e5_analyze.py DIR        # DIR/deep (spec.json, spec_rs.json) and DIR/drift/drift.json

- deep: FR21's calibration test (`calibration.py`, with the FR22 E1 fix), Holm at 0.005; emissions per position.
- drift: per prefix, TV between the speculative path's target row (the mean of its recorded rows; their spread is
  reported) and the decode path's row, and each against the fp64 row. A prefix whose spec-vs-decode TV exceeds the
  bf16 spread of FR20 (0.068) is a candidate defect (FR22 E5).
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CAL = HERE.parents[1] / "drafts" / "trtllm_v14" / "analysis" / "calibration.py"
BF16_SPREAD = 0.068


def tv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def main(d):
    d = pathlib.Path(d)
    res = {}
    if (d / "deep").exists():
        r = subprocess.run([sys.executable, str(CAL), str(d / "deep")], capture_output=True, text=True)
        print(r.stdout[-4000:], r.stderr[-1000:])
        c = json.loads((d / "deep" / "CALIBRATION.json").read_text())
        res["deep"] = dict(verdict=c["verdict"], emissions_by_position={
            f"{t['arm']}/{t['setting']}/{t['position']}": t["n"] for t in c["tests"]},
            min_p_holm=min(t["p_holm"] for t in c["tests"]))
    f = d / "drift" / "drift.json"
    if f.exists():
        dr = json.loads(f.read_text())
        rows = []
        for x in dr["prefixes"]:
            if not isinstance(x["decode_row"], list):
                continue
            sp = np.mean(x["spec_rows"], 0)
            rows.append(dict(prefix=x["prefix"], n=x["n"], tv_spec_decode=tv(sp, x["decode_row"]),
                             tv_spec_fp64=tv(sp, x["fp64_row"]), tv_decode_fp64=tv(x["decode_row"], x["fp64_row"]),
                             spec_row_spread=max(tv(r, sp) for r in x["spec_rows"])))
        a = np.array([r["tv_spec_decode"] for r in rows]) if rows else np.array([np.nan])
        res["drift"] = dict(prefixes=len(rows), verification_mismatch=dr["verification_mismatch"],
                            tv_spec_decode_median=float(np.median(a)), tv_spec_decode_max=float(a.max()),
                            tv_spec_fp64_max=max((r["tv_spec_fp64"] for r in rows), default=None),
                            tv_decode_fp64_max=max((r["tv_decode_fp64"] for r in rows), default=None),
                            over_bf16_spread=[r for r in rows if r["tv_spec_decode"] > BF16_SPREAD],
                            bf16_spread=BF16_SPREAD)
    (d / "E5.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "over_bf16_spread"} for k, v in res.items()},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
