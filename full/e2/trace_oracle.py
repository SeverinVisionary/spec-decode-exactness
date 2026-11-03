"""FR22 E2 trace oracle: invariants I1-I5 over the verifications a cell's trace kept (the first 1,000).

- I1 one draft row per draft token (`rows_used` == number of drafts);
- I2 every accepted draft has p > 0 in its true target row (greedy: it is that row's argmax);
- I3 a residual token lies in the support of (p - q)+ of its true rows, or of p when (p - q)+ is empty (greedy: it is
  the row's argmax);
- I4 the bonus token has p > 0 in the row after the last draft: the engine's own row when it is known (HF), else the
  fp64 reference row at the prefix (greedy: the argmax);
- I5 where the prefix is known, each target row used at a reached position, and the bonus row used, equals the fp64
  reference row (`e1/specmodel.Rows`, the setting's processing) at its own prefix within DELTA (max abs difference on
  the fixture's V tokens; padding mass counts as difference).
`check(verifs, setting)` returns per-invariant violation counts, the coverage of I4/I5, and the largest I5 difference.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "e1"), str(HERE.parents[1] / "code")]

DELTA = 1e-3  # fp32 engines; M0 reports the observed floor next to it


def _rows_for(setting):
    import specmodel
    return specmodel.Rows("S0" if setting == "G" else setting)


def _diff(used, ref):
    u = np.asarray(used, float)
    V = len(ref)
    return float(max(np.abs(u[:V] - ref).max(), u[V:].sum() if len(u) > V else 0.0))


def _valid_prefix(pre, V, max_new):
    return pre is not None and len(pre) <= max_new and all(0 <= t < V for t in pre)


def check(verifs, setting, V=8, max_new=4):
    R = _rows_for(setting)
    v = dict(I1=0, I2=0, I3=0, I4=0, I5=0)
    cov = dict(verifications=len(verifs), I4_checked=0, I5_checked=0, I5_max_diff=0.0)
    for r in verifs:
        d, P, Q, K = r["d"], r["P"], r["Q"], len(r["d"])
        greedy, n_acc, em = r["greedy"], r["n_acc"], r["emitted"]
        pre = r.get("prefix")
        known = _valid_prefix(pre, V, max_new)
        if r["rows_used"] != K:
            v["I1"] += 1
        for j in range(min(n_acc, K)):
            if (greedy and d[j] != int(np.argmax(P[j]))) or (not greedy and P[j][d[j]] <= 0):
                v["I2"] += 1
                break
        ra = r["residual_at"]
        if ra is not None:
            t, p, q = em[ra], np.asarray(P[ra]), np.asarray(Q[ra])
            res = np.clip(p - q, 0, None)
            sup = res if res.sum() > 0 else p
            if (greedy and t != int(np.argmax(p))) or (not greedy and sup[t] <= 0):
                v["I3"] += 1
        elif len(em) > K:
            b = em[K]
            row = r["bonus_row"]
            if row is None and known and len(pre) + K < max_new and fx_eos_free(d):
                row = list(R.p(tuple(pre) + tuple(d)))
            if row is not None:
                cov["I4_checked"] += 1
                if (greedy and b != int(np.argmax(row[:V]))) or (not greedy and row[b] <= 0):
                    v["I4"] += 1
        if known:
            reach = range(K) if ra is None else range(ra + 1)
            worst = 0.0
            for j in reach:
                g = tuple(pre) + tuple(d[:j])
                if len(g) >= max_new or 7 in d[:j]:
                    break
                worst = max(worst, _diff(r["P_used"][j], R.p(g)))
            bu = r.get("bonus_row_used")
            if ra is None and bu is not None and len(pre) + K < max_new and fx_eos_free(d):
                worst = max(worst, _diff(bu, R.p(tuple(pre) + tuple(d))))
            cov["I5_checked"] += 1
            cov["I5_max_diff"] = max(cov["I5_max_diff"], worst)
            if worst > DELTA:
                v["I5"] += 1
    return dict(violations=v, coverage=cov)


def fx_eos_free(d, eos=7):
    return eos not in d
