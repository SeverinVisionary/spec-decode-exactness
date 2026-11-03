"""FR9 power check: the rejected adaptive Stage A look vs fixed N, R1 perturbation, m = 118.

Usage: D40_WORK=<pilot work dir with refs/> python fr9_power.py S0 S3   (400 replicates per eps)"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "code"))
import fixture as fx, rnp, stats
M, A = 118, 0.01
lvl = A / M
def sim(probs, eps, sims, rng):
    probs = probs / probs.sum(); supp = probs > 0
    alt = (1 - eps) * probs + eps * supp / supp.sum()
    fix2, fix05, ad, ext = 0, 0, 0, 0
    for _ in range(sims):
        a = rng.multinomial(50_000, alt); b = rng.multinomial(150_000, alt)
        p1 = stats.pooled_chi2(a, probs)[1]; p2 = stats.pooled_chi2(a + b, probs)[1]
        fix2 += p2 < lvl; fix05 += p1 < lvl
        if 2 * p1 < lvl: ad += 1
        elif p1 < 0.05:
            ext += 1; ad += 2 * p2 < lvl
    return fix2 / sims, fix05 / sims, ad / sims, ext / sims
rng = np.random.default_rng(0)
for s in sys.argv[1:]:
    probs = rnp.sequence_probs(np.load(fx.WORK / "refs" / f"hf_{s}.npz")["eng"])
    for eps in (0.0, 0.003, 0.005, 0.009, 0.018):
        print(s, eps, "fixed2e5 %.3f fixed5e4 %.3f adaptive %.3f extended %.3f" % sim(probs, eps, 400, rng), flush=True)
