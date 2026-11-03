"""D-40 full study, Phase 0b F5: MDE per cell at alpha/118 (FR9 item 7), from the fp64 reference only.

No sample from any engine is used. Every level is passed explicitly as `stats.ALPHA / stats.M_FULL`.

F-A (§5.4): the pilot's pooled chi^2 over the 2,801 outcomes, `stats.mde`, against the R1
perturbation (1 - eps) R + eps U(support of R), at N_A = 2e5 and at the floor 5e4, for each engine's
documented sampler order (`rnp.DISTS`) and setting.

F-B (§5.4): per visited prefix with at least 200 visits, a pooled chi^2 of the next-token counts against
the reference, combined by Fisher's method. The perturbation is R1 applied at every step. Each simulation follows the frozen procedure exactly: an independent null control is drawn
down the prefix trie (each child's visits are its parent's count of that token) and fixes the prefix set
(>= 200 visits); the cell is drawn the same way under the alternative and tested at those prefixes. F-A
applies R1 to the sequence distribution (`stats.mde`); F-B applies it per step. An MDE above the 0.2
search bound is recorded as None (censored).

    python f5_mde_full.py fa|fb [--sims 1000]      # writes phase0b/f5_mde_{fa,fb}.json
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import fb_fixture as fb  # noqa: E402
import rnp  # noqa: E402
import rnp_full  # noqa: E402,F401  (registers the full study's engine orders)
import stats  # noqa: E402

LEVEL = stats.ALPHA / stats.M_FULL
N_A, N_FLOOR, MIN_VISITS = 200_000, 50_000, 200
FA_SETTINGS = {
    "S0": dict(temperature=1.0), "S1": dict(temperature=0.7), "S2": dict(top_k=3), "S3": dict(top_p=0.8),
    "S4": dict(min_p=0.1), "S5": dict(repetition_penalty=1.3),
    "S6": dict(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
}
FB_SETTINGS = {s: FA_SETTINGS[s] for s in ("S0", "S2", "S6")}
# engine x setting combinations that are BLOCKED before Phase 0b (FR9 item 1, FR12): no MDE is computed
# SGLang S5: no prompt-scored reference (FR12); sglang_torch: only the S4/S6 rescue path (FR9 item 1, FR12)
BLOCKED = {"vllm": {"S4", "S6"}, "trtllm": {"S4", "S5", "S6"}, "sglang": {"S5"},
           "sglang_torch": {"S0", "S1", "S2", "S3", "S5"}}


def running(eng, settings):
    return {s: kw for s, kw in settings.items() if s not in BLOCKED.get(eng, set())}


def fa(sims):
    out = {}
    for eng in rnp.DISTS:
        for s, kw in running(eng, FA_SETTINGS).items():
            probs = rnp.sequence_probs(rnp.step_table(eng, kw))
            supp = probs > 0
            tv_u = 0.5 * np.abs(probs - supp / supp.sum()).sum()
            row = dict(support=int(supp.sum()), tv_ref_to_uniform_support=float(tv_u), level=LEVEL)
            for tag, n in (("NA", N_A), ("Nfloor", N_FLOOR)):
                rng = np.random.default_rng(2)
                if stats.rejection_rate(probs, 0.2, n, LEVEL, sims, rng) < 0.9:
                    e = None  # MDE above the 0.2 search bound (censored)
                else:
                    e = stats.mde(probs, n, level=LEVEL, sims=sims)
                row[f"mde_eps_{tag}"], row[f"mde_tv_{tag}"] = e, None if e is None else float(e * tv_u)
            out[f"{eng}_{s}"] = row
            print(eng, s, row, flush=True)
    return out


class FBRef:
    """Lazy per-prefix fp64 reference on F-B for one engine order and setting."""

    def __init__(self, eng, kw):
        self.m = rnp.Llama64(fb.NAMES[0], n_layers=4, n_heads=2)
        self.f, self.kw, self.cache = rnp.DISTS[eng], kw, {}

    def step(self, g):
        if g not in self.cache:
            ids = fb.PROMPT + list(g)
            self.cache[g] = self.f(self.m.logits(ids), ids, self.kw)
        return self.cache[g]


def perturb(p, eps):
    supp = p > 0
    return (1 - eps) * p + eps * supp / supp.sum()


def reach(ref, n, eps, floor):
    """Prefixes whose expected visit count at N under the (perturbed) reference is >= floor (R_eng scoring sets)."""
    out, frontier = {}, [((), 1.0)]
    while frontier:
        nxt = []
        for g, r in frontier:
            if n * r < floor:
                continue
            out[g] = r
            if len(g) == fb.MAX_NEW - 1:
                continue
            p = perturb(ref.step(g), eps)
            for t in range(fb.V):
                if t != fb.EOS and p[t] > 0:
                    nxt.append((g + (t,), r * p[t]))
        frontier = nxt
    return out


def null_tree(ref, n, rng):
    """One null draw of N sequences, simulated exactly down the prefix trie: each prefix's next-token counts are a
    multinomial of its visits, and each non-EOS child's visits are its parent's count of that token. Only
    prefixes with >= MIN_VISITS visits are followed (descendants cannot have more). Returns the selected set."""
    sel, stack = [], [((), n)]
    while stack:
        g, v = stack.pop()
        if v < MIN_VISITS:
            continue
        sel.append(g)
        if len(g) == fb.MAX_NEW - 1:
            continue
        c = rng.multinomial(v, ref.step(g))
        stack.extend((g + (t,), int(c[t])) for t in range(fb.V) if t != fb.EOS and c[t] > 0)
    return set(sel)


def alt_counts(ref, n, eps, sel, rng):
    """One alternative draw on the trie, restricted to the null-selected set (closed under ancestors): returns
    {prefix: next-token counts} for every selected prefix the alternative visits."""
    out, stack = {}, [((), n)]
    while stack:
        g, v = stack.pop()
        if v == 0 or g not in sel:
            continue
        c = rng.multinomial(v, perturb(ref.step(g), eps))
        out[g] = c
        if len(g) < fb.MAX_NEW - 1:
            stack.extend((g + (t,), int(c[t])) for t in range(fb.V) if t != fb.EOS and c[t] > 0)
    return out


def fb_power(ref, n, eps, sims, rng):
    """Frozen procedure (§5.4): the prefix set comes from an independent null control's realised visits (>= 200);
    the cell's counts at those prefixes are tested against the reference, per-prefix chi^2 combined by Fisher."""
    rej = 0
    for _ in range(sims):
        sel = null_tree(ref, n, rng)
        counts = alt_counts(ref, n, eps, sel, rng)
        keys = sorted(counts)
        if fisher_p([counts[g] for g in keys], [ref.step(g) for g in keys]) < LEVEL:
            rej += 1
    return rej / sims


def fisher_p(counts_list, probs_list):
    """Fisher's method over the per-prefix pooled chi^2. A count on a zero-probability token rejects outright
    (p = 0). A prefix whose support pools into one bin has no stochastic component (chi2.sf(0, 0) is NaN) and
    adds nothing; with no informative prefix the p-value is 1."""
    ps = []
    for c, p in zip(counts_list, probs_list):
        stat, pv, dof = stats.pooled_chi2(c, p)
        if np.isinf(stat):
            return 0.0
        if dof >= 1:
            assert np.isfinite(pv), (stat, pv, dof)
            ps.append(pv)
    if not ps:
        return 1.0
    x = -2 * np.sum(np.log(np.clip(ps, 1e-300, None)))
    return float(chi2.sf(x, 2 * len(ps)))


def bisect_mde(power_at, power=0.9, iters=10, hi=0.2):
    """The smallest eps with power >= 0.9, or None when power at the bound is below 0.9 (MDE above the bound:
    a censored value, not a detection threshold)."""
    if power_at(hi) < power:
        return None
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if power_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def fb_mde(ref, n, sims, seed=0):
    rng = np.random.default_rng(seed)
    return bisect_mde(lambda e: fb_power(ref, n, e, sims, rng))


def fbrun(sims):
    out = {}
    for eng in rnp.DISTS:
        for s, kw in running(eng, FB_SETTINGS).items():
            ref = FBRef(eng, kw)
            row = dict(level=LEVEL, min_visits=MIN_VISITS, expected_prefixes_NA=len(reach(ref, N_A, 0.0, MIN_VISITS)))
            for tag, n in (("NA", N_A), ("Nfloor", N_FLOOR)):
                row[f"mde_eps_{tag}"] = fb_mde(ref, n, sims)
            out[f"{eng}_{s}"] = row
            print(eng, s, row, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", choices=("fa", "fb"))
    ap.add_argument("--sims", type=int, default=1000)
    a = ap.parse_args()
    res = fa(a.sims) if a.fixture == "fa" else fbrun(a.sims)
    path = os.environ.get("F5_OUT", str(HERE / "phase0b" / f"f5_mde_{a.fixture}.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(level=LEVEL, m=stats.M_FULL, sims=a.sims, cells=res), open(path, "w"), indent=1)
