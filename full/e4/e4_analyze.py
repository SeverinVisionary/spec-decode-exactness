"""FR22 E4 analysis: per ablation arm and setting, the FR18 endpoints plus the calibration test.

    python e4_analyze.py OUT        # reads OUT/base_base.json.gz and OUT/spec_*.json.gz; writes OUT/E4.json, E4.txt

- vs base: `repro_draft_target.compare` (TV, chi-square with sparse cells pooled), level 0.005; base split in halves
  is the noise floor.
- R < K, accepted beyond R, accepted tokens with target probability 0: from the recorder, as FR18.
- Calibration: FR21's test (`drafts/trtllm_v14/analysis/calibration.py`, with the FR22 E1 fix) on the emitted tokens
  against the engine's target rows at their positions; Holm over all arms, settings and positions, level 0.005.
"""
import gzip
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "drafts" / "trtllm_repro"))
sys.path.insert(0, str(HERE.parents[1] / "drafts" / "trtllm_v14" / "analysis"))

import numpy as np  # noqa: E402
from scipy.stats import chi2 as chi2_dist  # noqa: E402

import calibration as C  # noqa: E402
import repro_draft_target as R  # noqa: E402

LEVEL = 0.005


def calib_tests(cal):
    """FR21's per-position test on one arm x setting: rows grouped (rounded 1e-3), groups of >= 50 draws tested,
    statistics summed; any emitted token with probability 0 in its row sets p = 0."""
    out = []
    if not cal:
        return out
    V = len(cal[0][1][0])
    for j in range(max(len(e) for e, _ in cal)):
        groups, impossible = {}, 0
        for emitted, rows in cal:
            if len(emitted) > j:
                row = np.asarray(rows[j], float)
                row = row / max(row.sum(), 1e-12)
                impossible += row[emitted[j]] <= 0
                g = groups.setdefault(tuple(np.round(row, 3)), [np.zeros(V), np.zeros(V)])
                g[0][emitted[j]] += 1
                g[1] += row
        stat, dof, n = 0.0, 0, 0
        for obs, exp in groups.values():
            if obs.sum() < C.MIN_GROUP:
                continue
            r = C.gof(obs, exp)
            if r:
                stat, dof, n = stat + r["chi2"], dof + r["dof"], n + r["n"]
        if dof or impossible:
            out.append(dict(position=j, chi2=round(stat, 2), dof=dof, n=n, impossible_draws=int(impossible),
                            p=0.0 if impossible else float(chi2_dist.sf(stat, dof))))
    return out


def main(out):
    out = pathlib.Path(out)
    load = lambda p: json.load(gzip.open(p, "rt"))  # noqa: E731
    base = load(out / "base_base.json.gz")
    res, lines, all_tests = {}, [], []
    for name in R.SETTINGS:
        b = base["settings"][name]["seqs"]
        h = len(b) // 2
        res[name] = {"base_half_vs_half": R.compare(b[:h], b[h:])}
        lines.append(f"[{name}] base halves: {json.dumps(res[name]['base_half_vs_half'])}")
    for p in sorted(out.glob("spec_*.json.gz")):
        d = load(p)
        tag = d["tag"]
        for name in R.SETTINGS:
            s = d["settings"].get(name)
            if s is None:
                res[name][tag] = dict(status="MISSING")
                continue
            rows = s["verification_rows"]
            t = calib_tests(s.get("calibration", []))
            for x in t:
                all_tests.append((name, tag, x))
            res[name][tag] = dict(
                vs_base=R.compare(s["seqs"], base["settings"][name]["seqs"]), n=len(s["seqs"]),
                verifications=len(rows), rows_fewer_than_K=sum(1 for k, r, _, _ in rows if r < k),
                accepted_beyond_R=sum(max(0, min(n, k) - r) for k, r, n, _ in rows),
                accepted_zero_target_prob=sum(z for *_, z in rows), calibration_entries=len(s.get("calibration", [])),
                calibration=t)
    adj = C.holm([x["p"] for _, _, x in all_tests]) if all_tests else []
    for (name, tag, x), a in zip(all_tests, adj):
        x["p_holm"], x["reject"] = a, a < LEVEL
    for name in R.SETTINGS:
        for tag, r in sorted(res[name].items()):
            if tag == "base_half_vs_half" or "vs_base" not in r:
                continue
            vb, cal = r["vs_base"], r["calibration"]
            r["passes_vs_base"] = vb["p"] >= LEVEL
            r["calibrated"] = bool(cal) and not any(x["reject"] for x in cal)
            lines.append(f"[{name}] {tag:9s} TV {vb['tv']:.4f} p {vb['p']:.3g} | R<K {r['rows_fewer_than_K']}/"
                         f"{r['verifications']} beyond-R {r['accepted_beyond_R']} zero-p {r['accepted_zero_target_prob']}"
                         f" | calibration {'ok' if r['calibrated'] else 'FAIL' if cal else 'no data'} "
                         f"(min p_holm {min((x['p_holm'] for x in cal), default=float('nan')):.3g})")
    (out / "E4.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "E4.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
