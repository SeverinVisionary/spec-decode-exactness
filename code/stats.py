"""χ² test, Holm, and MDE simulation (PREREGISTRATION §5.4). Sees only counts and probabilities."""
import numpy as np
from scipy.stats import chi2

ALPHA, M_PLANNED = 0.01, 84  # 12 methods x S0..S6; G is deterministic
M_FULL = 118  # D-40 full study, FULL FR9 item 2; pass level=ALPHA / M_FULL explicitly there
N_A_DEFAULT = 200_000  # F5 tabulates the MDE at this N


def pooled_chi2(counts, probs, min_expected=5.0):
    counts, probs = np.asarray(counts, float), np.asarray(probs, float)
    probs = probs / probs.sum()  # engine float32 rows sum to 1 +- 1e-7
    n = counts.sum()
    exp = probs * n
    big = exp >= min_expected
    obs_b, exp_b = list(counts[big]), list(exp[big])
    if (~big).any():
        obs_b.append(counts[~big].sum())
        exp_b.append(exp[~big].sum())
    obs_b, exp_b = np.array(obs_b), np.array(exp_b)
    keep = exp_b > 0
    if (obs_b[~keep] > 0).any():
        return np.inf, 0.0, int(keep.sum() - 1)  # mass on a zero-probability outcome
    stat = float(((obs_b[keep] - exp_b[keep]) ** 2 / exp_b[keep]).sum())
    dof = int(keep.sum() - 1)
    return stat, float(chi2.sf(stat, dof)), dof


def holm(pvals, alpha=ALPHA):
    order = np.argsort(pvals)
    m, reject = len(pvals), np.zeros(len(pvals), bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def rejection_rate(probs, eps, n, level, sims, rng):
    probs = probs / probs.sum()
    supp = probs > 0
    alt = (1 - eps) * probs + eps * supp / supp.sum()
    draws = rng.multinomial(n, alt, size=sims)
    return float(np.mean([pooled_chi2(d, probs)[1] < level for d in draws]))


def mde(probs, n, level=ALPHA / M_PLANNED, sims=1000, power=0.9, seed=0, iters=12):
    rng = np.random.default_rng(seed)
    lo, hi = 0.0, 0.2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if rejection_rate(probs, mid, n, level, sims, rng) >= power:
            hi = mid
        else:
            lo = mid
    return hi
