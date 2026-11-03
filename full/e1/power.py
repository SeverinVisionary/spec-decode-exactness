"""FR22 E1 detection limits: exact TV and N90 of the frozen F-A test for each defect class D1-D8.

For every setting (S0, S2), class and firing rate lam, the alternative sequence distribution comes from
`specmodel.sequence_dist` (exact). N90 is the smallest N (on a log grid, bisected) at which the frozen F-A statistic
(`stats.pooled_chi2` against the correct distribution) rejects at alpha/118 in at least 90% of 1,000 multinomial
simulations. Classes with TV 0 are inert at that setting (N90 = None). N90 above 1e10 is reported as censored.

    python power.py [--sims 1000] [--out e1_power.json]
"""
import argparse
import json
import zlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import specmodel as sm  # noqa: E402
import stats  # noqa: E402  (code/ is on the path via specmodel)

LEVEL = stats.ALPHA / stats.M_FULL
LAMS = (1.0, 0.3, 0.1, 0.03, 0.01)
N_LO, N_HI = 1e2, 1e10


def power(alt, ref, n, sims, rng):
    draws = rng.multinomial(int(n), alt, size=sims)
    return float(np.mean([stats.pooled_chi2(d, ref)[1] < LEVEL for d in draws]))


def n90(alt, ref, sims, rng, iters=14):
    if power(alt, ref, N_HI, sims, rng) < 0.9:
        return None
    lo, hi = np.log10(N_LO), np.log10(N_HI)
    if power(alt, ref, 10 ** lo, sims, rng) >= 0.9:
        return N_LO
    for _ in range(iters):
        mid = (lo + hi) / 2
        if power(alt, ref, 10 ** mid, sims, rng) >= 0.9:
            hi = mid
        else:
            lo = mid
    return float(round(10 ** hi))


def main(sims, out):
    rows = []
    for s in sm.SETTINGS:
        R = sm.Rows(s)
        ref = sm.reference(s)
        ok = sm.sequence_dist(R, "ok")
        assert np.abs(ok - ref).max() < 1e-12
        for c in sm.CLASSES:
            for lam in LAMS:
                alt = sm.sequence_dist(R, c, lam)
                tv = float(0.5 * np.abs(alt - ref).sum())
                zero_mass = float(alt[ref == 0].sum())
                rng = np.random.default_rng(zlib.crc32(f"{s}{c}{lam}".encode()))
                n = n90(alt, ref, sims, rng) if tv > 1e-12 else None
                row = dict(setting=s, cls=c, lam=lam, tv=tv, mass_on_ref_zero=zero_mass, n90=n,
                           censored=tv > 1e-12 and n is None, power_at_NA=power(alt, ref, 2e5, sims, rng))
                rows.append(row)
                print(row, flush=True)
    json.dump(dict(level=LEVEL, sims=sims, n_range=[N_LO, N_HI], rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE / "e1_power.json"))
    a = ap.parse_args()
    main(a.sims, a.out)
