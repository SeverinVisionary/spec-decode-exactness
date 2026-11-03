"""FR22 E1 null calibration of the frozen grid tests, per running Stage A cell, plus the support bound.

For each of the 53 running grid cells (`stagea/data/ANALYSIS_A.json`, kind == "cell"):
- F-A: SIMS multinomial data sets of the cell's N from its own R_eng sequence distribution (the table Stage A tested
  against, via `stagea_analyze.Refs`), each tested with `stats.pooled_chi2` as Stage A did.
- F-B: SIMS repetitions of the frozen procedure under the null, as F5 simulates it (`f5_mde_full.null_tree` for the
  independent null's prefix set, `alt_counts` at eps = 0 for the cell, `fisher_p`), on the fp64 rows of the engine's
  documented sampler order (Stage A's R_eng tables cover only the scoring set, not every prefix a simulation visits).
Reported per cell: the empirical rejection rate at alpha/118, 0.01 and 0.05 with Clopper-Pearson 99% intervals, the
KS distance of the simulated p-values from uniform, and the support bound: with zero draws on R_eng-zero outcomes in
N draws, the exact one-sided 95% upper bound 1 - 0.05^(1/N) on that event's per-draw probability.
Family: the share of simulations in which Holm at 0.01 over m = 118 (cells not run at p = 1) rejects any cell.

    python null_calib.py [--sims 2000] [--only FA|FB] [--shard I/N] [--out e1_null.json]
    python null_calib.py --family e1_null_*.json          # merges shards, computes the family rate
"""
import argparse
import json
import pathlib
import sys
import zlib

HERE = pathlib.Path(__file__).resolve().parent
FULL = HERE.parent
sys.path.insert(0, str(FULL))
sys.path.insert(0, str(FULL.parent / "code"))

import numpy as np  # noqa: E402
from scipy.stats import beta, kstest  # noqa: E402

import f5_mde_full as f5  # noqa: E402
import stagea_analyze as S  # noqa: E402
import stats  # noqa: E402

LEVELS = {"alpha_118": stats.ALPHA / stats.M_FULL, "0.01": 0.01, "0.05": 0.05}
FB_ENGINE_ORDER = dict(vllm="vllm", trtllm="trtllm", llamacpp="lcpp")


def cp99(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.005, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.995, k + 1, n - k))
    return lo, hi


def fa_null_ps(probs, n, sims, rng):
    return np.array([stats.pooled_chi2(d, probs)[1] for d in rng.multinomial(n, probs, size=sims)])


def fb_null_ps(engine, setting, n, sims, rng):
    ref = f5.FBRef(FB_ENGINE_ORDER[engine], f5.FB_SETTINGS[setting])
    ps = []
    for _ in range(sims):
        sel = f5.null_tree(ref, n, rng)
        counts = f5.alt_counts(ref, n, 0.0, sel, rng)
        keys = sorted(counts)
        ps.append(f5.fisher_p([counts[g] for g in keys], [ref.step(g) for g in keys]))
    return np.array(ps)


def main(sims, only, shard, out):
    rows_a = json.load(open(FULL / "stagea" / "data" / "ANALYSIS_A.json"))["rows"]
    cells = [r for r in rows_a if r["kind"] == "cell" and (only is None or r["fixture"] == only)]
    i, n = shard
    cells = cells[i::n]
    refs = S.Refs(FULL / "stagea" / "data")
    res = []
    for r in cells:
        rng = np.random.default_rng(zlib.crc32(r["key"].encode()))
        if r["fixture"] == "FA":
            probs = refs.fa_seq(r["engine"], S.table_name(r["engine"], "FA", r["method"], r["setting"], "cell"))
            ps = fa_null_ps(probs, r["N"], sims, rng)
        else:
            ps = fb_null_ps(r["engine"], r["setting"], r["N"], sims, rng)
        row = dict(key=r["key"], N=r["N"], sims=sims, observed_p=r["p"], ps=[float(x) for x in ps])
        for name, lv in LEVELS.items():
            k = int((ps < lv).sum())
            lo, hi = cp99(k, sims)
            row[f"rate_{name}"], row[f"ci99_{name}"] = k / sims, [lo, hi]
            row[f"miscalibrated_{name}"] = ("anticonservative" if lv < lo else "conservative" if lv > hi else None)
        row["ks_uniform"] = float(kstest(ps, "uniform").statistic)
        if r.get("zero_prob_hits", 0) == 0 and r.get("out_of_space", 0) == 0:
            row["support_bound_95"] = float(1 - 0.05 ** (1 / r["N"]))
        res.append(row)
        print({k: v for k, v in row.items() if k != "ps"}, flush=True)
        json.dump(dict(sims=sims, levels=LEVELS, cells=res), open(out, "w"))  # per cell, so a cut run keeps its cells


def family(paths, out):
    cells = [c for p in paths for c in json.load(open(p))["cells"]]
    assert len({c["key"] for c in cells}) == len(cells) == 53, len(cells)
    pm = np.array([c["ps"] for c in cells])  # cells x sims; the other 118 - 53 cells enter Holm at p = 1
    sims = pm.shape[1]
    rej = [stats.holm(np.concatenate([pm[:, i], np.ones(stats.M_FULL - len(pm))])).any() for i in range(sims)]
    k = int(np.sum(rej))
    fam = dict(fwer=k / sims, ci99=cp99(k, sims), nominal=stats.ALPHA, sims=sims)
    print("family:", fam)
    for c in cells:
        c.pop("ps")
    json.dump(dict(levels=LEVELS, cells=cells, family=fam), open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--only", choices=("FA", "FB"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--family", nargs="+")
    ap.add_argument("--out", default=str(HERE / "e1_null.json"))
    a = ap.parse_args()
    if a.family:
        family(a.family, a.out)
    else:
        main(a.sims, a.only, tuple(int(x) for x in a.shard.split("/")), a.out)
