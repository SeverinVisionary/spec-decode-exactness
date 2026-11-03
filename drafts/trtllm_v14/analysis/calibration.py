"""FR21 primary analysis: is each emitted token distributed as the engine's own target probabilities?

In both one-model modes, a position j that is emitted (the first j draft tokens were accepted) carries a token drawn
from the target distribution at that prefix (strict acceptance: it is the target's own sample; rejection sampling:
accepted draft or residual sample, which together give exactly p). The recorder stores, per verification, the emitted
tokens and the engine's target-probability row at each emitted position, so this test does not depend on how close
those probabilities are to fp64.

Per arm x setting x position j: verifications are grouped by their probability row (rounded to 1e-3; rows repeat
because prefixes repeat). Each group with at least MIN_GROUP draws gets a Pearson goodness of fit of its emitted
tokens against its row (cells with expected count < 5 pooled); the group statistics and degrees of freedom are
summed. Holm over all tests; a test rejects at adjusted p < 0.005. Any emitted token with probability 0 in its own
row sets that test's p to 0, whatever its group size (FR22 E1 fix; the first version dropped such draws).

    python calibration.py <out dir with spec.json[.gz] spec_rs.json[.gz]>
"""
import gzip
import json
import pathlib
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist

ALPHA = 0.005
MIN_GROUP = 50  # verifications per probability row needed to test that row


def load(out, arm):
    for name in (f"{arm}.json.gz", f"{arm}.json"):
        p = out / name
        if p.exists():
            return json.loads(gzip.open(p).read() if name.endswith(".gz") else p.read_text())
    return None


def gof(obs, exp):
    big = exp >= 5
    o = np.append(obs[big], obs[~big].sum())
    e = np.append(exp[big], exp[~big].sum())
    keep = e > 0
    if (o[~keep] > 0).any():  # a draw on a zero-probability token: impossible under the row (FR22 E1 fix)
        return dict(chi2=float("inf"), dof=max(int(keep.sum()) - 1, 1), p=0.0, n=int(o.sum()),
                    max_abs_freq_diff=1.0)
    o, e = o[keep], e[keep]
    if len(e) < 2:
        return None
    stat = float(((o - e) ** 2 / e).sum())
    dof = len(e) - 1
    return dict(chi2=round(stat, 2), dof=dof, p=float(chi2_dist.sf(stat, dof)), n=int(o.sum()),
                max_abs_freq_diff=round(float(np.abs(o - e).max() / o.sum()), 4))


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def main(out):
    out = pathlib.Path(out)
    tests = []
    for arm in ("spec", "spec_rs"):
        d = load(out, arm)
        if d is None:
            print(arm, "missing")
            continue
        for setting, s in d["settings"].items():
            cal = s.get("calibration", [])
            print(f"{arm} {setting}: {len(cal)} verifications recorded, {s.get('calibration_missing', 0)} without "
                  f"target probabilities", flush=True)
            if not cal:
                continue
            V = len(cal[0][1][0])
            for j in range(max(len(e) for e, _ in cal)):
                # stratify by the probability row itself (rounded): rows repeat because prefixes repeat on this
                # fixture, and an aggregate over different rows can hide miscalibration that averages out
                groups, impossible = {}, 0
                for emitted, rows in cal:
                    if len(emitted) > j:
                        row = np.asarray(rows[j], float)
                        row = row / max(row.sum(), 1e-12)
                        impossible += row[emitted[j]] <= 0  # counted in every group, small or not (FR22 E1)
                        g = groups.setdefault(tuple(np.round(row, 3)), [np.zeros(V), np.zeros(V)])
                        g[0][emitted[j]] += 1
                        g[1] += row
                stat, dof, n, used, worst = 0.0, 0, 0, 0, 0.0
                for obs, exp in groups.values():
                    if obs.sum() < MIN_GROUP:
                        continue
                    r = gof(obs, exp)
                    if r:
                        stat, dof, n, used = stat + r["chi2"], dof + r["dof"], n + r["n"], used + 1
                        worst = max(worst, r["max_abs_freq_diff"])
                if dof or impossible:
                    tests.append(dict(arm=arm, setting=setting, position=j, chi2=round(stat, 2), dof=dof,
                                      p=0.0 if impossible else float(chi2_dist.sf(stat, dof)), n=n, groups=used,
                                      groups_total=len(groups), max_abs_freq_diff=worst,
                                      impossible_draws=int(impossible)))
    adj = holm([t["p"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = a
        t["reject"] = a < ALPHA
        print(t)
    verdict = ("CALIBRATION FAILS" if any(t["reject"] for t in tests) else "calibrated") if tests else "NO DATA"
    print("verdict:", verdict)
    (out / "CALIBRATION.json").write_text(json.dumps(dict(alpha=ALPHA, tests=tests, verdict=verdict), indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
