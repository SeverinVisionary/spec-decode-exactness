"""FR22 E2 analysis: the detector x mutation x engine matrix, next to E1's exact N90 for the same class.

    python e2_analyze.py DIR [DIR ...]     # reads hf_*_c*.json.gz and vllm_*.json.gz; writes E2.json, E2.txt in DIR[0]

Detectors (FR22 E2): (i) the frozen F-A sequence chi-square against the engine's Stage A R_eng table, detected at
p < 0.01/118; (ii) greedy, 1,000 runs, detected if any run differs from the fp64 greedy chain (vLLM only; HF's greedy
path never reaches the verifier); (iii) the trace oracle on the first 1,000 verifications, detected if any invariant
is violated; (iv) the FR21 calibration test with E1's fix (groups of >= 50 draws, Holm over positions, level 0.005).
A mutation cell whose effect count is 0 is INVALID (§5.3 rule 1). M5 is NOT_APPLICABLE in both engines.
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist

HERE = pathlib.Path(__file__).resolve().parent
FULL = HERE.parent
sys.path[:0] = [str(HERE), str(FULL), str(FULL / "e1"), str(FULL.parent / "code"),
                str(FULL.parent / "drafts" / "trtllm_v14" / "analysis")]

import calibration as CAL  # noqa: E402
import p0b_common as C  # noqa: E402
import stagea_analyze as S  # noqa: E402
import trace_oracle as T  # noqa: E402

LEVEL_SEQ = 0.01 / 118
LEVEL_CAL = 0.005
CLASS = dict(M1="D1", M2="D2", M3="D3", M4="D4", M6="D8")
MODES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


def load(dirs):
    cells = collections.defaultdict(lambda: dict(seqs=[], verifs=[], calib=[], impossible=collections.Counter(),
                                                 effect=0, n_verif=0, chunks=0))
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("*.json.gz")):
            r = json.load(gzip.open(p, "rt"))
            c = cells[(r["engine"], r["mode"], r["setting"])]
            c["seqs"] += [tuple(s) for s in r["seqs"]]
            t = r["trace"]
            c["verifs"] += t["verifications"][: max(0, 1000 - len(c["verifs"]))]
            c["calib"] += t["calibration"]
            c["impossible"].update({int(k): v for k, v in t["impossible"].items()})
            c["effect"] += t["effect"]
            c["n_verif"] += t["n_verifications"]
            c["chunks"] += 1
    return cells


def calib_test(calib, impossible):
    groups = collections.defaultdict(lambda: [None, None])
    for j, key, obs, exp in calib:  # merge chunks: same position and rounded row
        g = groups[(j, tuple(key))]
        g[0] = np.asarray(obs, float) if g[0] is None else g[0] + obs
        g[1] = np.asarray(exp, float) if g[1] is None else g[1] + exp
    per = collections.defaultdict(lambda: [0.0, 0, 0])
    for (j, _), (obs, exp) in groups.items():
        if obs.sum() < CAL.MIN_GROUP:
            continue
        r = CAL.gof(obs, exp)
        if r:
            per[j][0] += r["chi2"]
            per[j][1] += r["dof"]
            per[j][2] += r["n"]
    tests = []
    for j in sorted(set(per) | set(impossible)):
        stat, dof, n = per.get(j, (0.0, 0, 0))
        p = 0.0 if impossible.get(j) or not np.isfinite(stat) else (float(chi2_dist.sf(stat, dof)) if dof else 1.0)
        tests.append(dict(position=j, chi2=stat if np.isfinite(stat) else "inf", dof=dof, n=n,
                          impossible=int(impossible.get(j, 0)), p=p))
    adj = CAL.holm([t["p"] for t in tests]) if tests else []
    for t, a in zip(tests, adj):
        t["p_holm"] = a
    return dict(detected=any(a < LEVEL_CAL for a in adj), tests=tests)


def seq_test(engine, setting, seqs, refs):
    name = S.table_name(engine, "FA", "VG2" if engine == "vllm" else "H1", setting, "cell")
    probs = refs.fa_seq(engine, name)
    r = S.fa_test(collections.Counter(seqs), probs, C.FA)
    return dict(table=name, p=r["p"], chi2=r["chi2"], dof=r["dof"], out_of_space=r["out_of_space"],
                tv_empirical=round(r["tv_empirical"], 4), detected=r["p"] < LEVEL_SEQ)


def greedy_test(seqs):
    rows = T._rows_for("S0")
    chain, margin = S.greedy_chain(lambda g: rows.p(tuple(g)), C.FA)
    bad = sum(1 for s in seqs if tuple(C.FA.cut(s)) != tuple(chain))
    return dict(chain=list(chain), min_margin=round(margin, 4), runs=len(seqs), differing=bad, detected=bad > 0)


def e1_n90():
    p = FULL / "e1" / "out" / "e1_power.json"
    if not p.exists():
        return {}
    return {(r["setting"], r["cls"]): r["n90"] for r in json.load(open(p))["rows"] if r["lam"] == 1.0}


def main(dirs):
    cells = load(dirs)
    refs = S.Refs(FULL / "stagea" / "data")
    n90 = e1_n90()
    res, lines = {}, []
    for engine in ("hf", "vllm"):
        for mode in MODES:
            for setting in ("S0", "S2", "G"):
                k = f"{engine}/{mode}/{setting}"
                if mode == "M5":
                    res[k] = dict(status="NOT_APPLICABLE", reason="no draft row at the bonus position in this engine")
                    continue
                if engine == "hf" and setting == "G":
                    res[k] = dict(status="NOT_APPLICABLE", reason="HF greedy verification does not call the verifier")
                    continue
                c = cells.get((engine, mode, setting))
                if c is None:
                    res[k] = dict(status="MISSING")
                    continue
                r = dict(n=len(c["seqs"]), chunks=c["chunks"], verifications=c["n_verif"], effect=c["effect"])
                if mode != "M0" and c["effect"] == 0:
                    r["status"] = "INVALID"
                    res[k] = r
                    continue
                r["status"] = "OK"
                r["trace"] = T.check(c["verifs"], setting)
                r["trace_detected"] = any(r["trace"]["violations"].values())
                if setting == "G":
                    r["greedy"] = greedy_test(c["seqs"])
                else:
                    r["sequence_chi2"] = seq_test(engine, setting, c["seqs"], refs)
                    r["calibration"] = calib_test(c["calib"], c["impossible"])
                    r["e1_n90"] = n90.get((setting, CLASS.get(mode)))
                res[k] = r
                det = []
                if "sequence_chi2" in r:
                    det += [f"chi2 {'Y' if r['sequence_chi2']['detected'] else 'n'} (p {r['sequence_chi2']['p']:.2g})",
                            f"calib {'Y' if r['calibration']['detected'] else 'n'}"]
                if "greedy" in r:
                    det.append(f"greedy {'Y' if r['greedy']['detected'] else 'n'} ({r['greedy']['differing']}/{r['greedy']['runs']})")
                det.append(f"trace {'Y' if r['trace_detected'] else 'n'} {r['trace']['violations']}")
                lines.append(f"{k:16s} n {r['n']:6d} effect {r['effect']:6d} | " + " | ".join(det)
                             + (f" | E1 N90 {r['e1_n90']}" if r.get("e1_n90") else ""))
    for k, r in res.items():
        if r.get("status") != "OK":
            lines.append(f"{k:16s} {r['status']} {r.get('reason', '')}")
    out = pathlib.Path(dirs[0])
    (out / "E2.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / "E2.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
