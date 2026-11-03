"""FR22 E1: exact model of chain speculative sampling on F-A, correct and with defect classes D1-D8.

Everything is computed by enumeration from fp64 rows (`rnp.Llama64`), never sampled. One speculative round from a
generated prefix g drafts K = 3 tokens (drafting stops after an EOS; tokens past the 4-token budget are not
enumerated, since they could not be emitted: the model is an engine that always drafts K and cuts the output at the
budget, so D1's unchecked d_{K-1} and D2/D3's unchecked last position fall past the budget in a shortened round),
verifies them in order, and on full acceptance adds a bonus token. Emitted tokens are cut at the first EOS and at
the budget. Rows are processed with `rnp.vllm_dist` (S0: none; S2: top-k 3)
for both target (p) and draft (q).

Defect classes (FR22 E1), each replacing the round with probability lam:
- D1  last draft row missing: d_{K-1} is accepted without a check when d_0..d_{K-2} pass;
- D2  rows shifted by one: d_j is checked against q(g + d[:j+1]) (the row d_{j+1} is drawn from); d_{K-1} unchecked;
      0/0 (p = 0 and used q = 0) rejects;
- D3  D2 with 0/0 accepted (the NaN ratio of #20852);
- D4  residual drawn from p instead of norm((p - q)+);
- D5  bonus token drawn from q instead of p;
- D6  target processors skipped at drafted positions: acceptance and residual use the raw softmax (bonus processed);
- D7  greedy draft (argmax q) whose soft q is used in the acceptance test and residual as if sampled;
- D8  stale context: the target row at draft position j >= 1 (and the bonus row) is computed one token short.

    python specmodel.py --check      # the correct model reproduces R_np's sequence distribution
"""
import collections
import functools
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "code"))

import numpy as np  # noqa: E402

try:
    import torch  # noqa: F401  fixture.py needs it only to build models
    import transformers  # noqa: F401
except ImportError:  # the exact model needs only the committed weights: stub the builder imports
    import types
    sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))
    sys.modules.setdefault("transformers", types.SimpleNamespace(
        GenerationConfig=None, LlamaConfig=None, LlamaForCausalLM=None))
import fixture as fx  # noqa: E402
import rnp  # noqa: E402

K = 3
SETTINGS = {"S0": dict(temperature=1.0), "S2": dict(top_k=3)}
CLASSES = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
OUT_INDEX = {tuple(o): i for i, o in enumerate(fx.outcomes())}


class Rows:
    """fp64 rows at any generated prefix (tuples), lazily."""

    def __init__(self, setting):
        self.t, self.d, self.s = rnp.Llama64("target"), rnp.Llama64("draft"), SETTINGS[setting]

    @functools.lru_cache(maxsize=None)
    def logits(self, which, g):
        ids = fx.PROMPT + list(g)
        return (self.t if which == "t" else self.d).logits(ids)

    @functools.lru_cache(maxsize=None)
    def p(self, g):
        return rnp.vllm_dist(self.logits("t", g), fx.PROMPT + list(g), self.s)

    @functools.lru_cache(maxsize=None)
    def p_raw(self, g):
        return rnp._softmax(self.logits("t", g))

    @functools.lru_cache(maxsize=None)
    def q(self, g):
        return rnp.vllm_dist(self.logits("d", g), fx.PROMPT + list(g), self.s)


def _accept(p, q, zero_zero_accepts):
    if q > 0:
        return min(1.0, p / q)
    if p > 0:
        return 1.0
    return 1.0 if zero_zero_accepts else 0.0


def round_dist(R, g, mode):
    """{emitted tuple: probability} for one speculative round from prefix g under `mode` ("ok" or a class)."""
    r = fx.MAX_NEW - len(g)
    kmax = min(K, r)
    out = collections.defaultdict(float)

    def emit(toks, w):
        toks = list(toks)[:r]
        if fx.EOS in toks:
            toks = toks[: toks.index(fx.EOS) + 1]
        out[tuple(toks)] += w

    def rec(d, w):
        j = len(d)
        if j == kmax or (d and d[-1] == fx.EOS):  # every drafted token accepted
            if (d and d[-1] == fx.EOS) or j >= r:
                emit(d, w)
                return
            gd = g + tuple(d)
            row = R.q(gd) if mode == "D5" else R.p(g + tuple(d[:-1])) if mode == "D8" and j >= 1 else R.p(gd)
            for t in np.nonzero(row)[0]:
                emit(d + [int(t)], w * row[t])
            return
        gd = g + tuple(d)
        q_soft = R.q(gd)
        if mode == "D7":
            q_samp = np.zeros_like(q_soft)
            q_samp[int(np.argmax(q_soft))] = 1.0
        else:
            q_samp = q_soft
        p_used = R.p_raw(gd) if mode == "D6" else R.p(g + tuple(d[:-1])) if mode == "D8" and j >= 1 else R.p(gd)
        for t in np.nonzero(q_samp)[0]:
            t = int(t)
            wt = w * q_samp[t]
            if mode in ("D2", "D3"):
                q_used = R.q(gd + (t,)) if j < K - 1 else None
            else:
                q_used = q_soft
            unchecked = (mode == "D1" and j == K - 1) or (mode in ("D2", "D3") and j == K - 1)
            acc = 1.0 if unchecked else _accept(p_used[t], q_used[t], mode == "D3")
            if acc > 0:
                rec(d + [t], wt * acc)
            if acc < 1:
                res = p_used if mode == "D4" else np.maximum(p_used - q_used, 0.0)
                if res.sum() <= 0:
                    res = p_used
                res = res / res.sum()
                for u in np.nonzero(res)[0]:
                    emit(d + [int(u)], wt * (1 - acc) * res[u])

    rec([], 1.0)
    return out


def sequence_dist(R, mode, lam=1.0):
    """Exact distribution over the 2,801 F-A outcomes when each round is `mode` with probability lam, else correct."""

    @functools.lru_cache(maxsize=None)
    def from_prefix(g):
        rd = collections.defaultdict(float)
        for e, w in round_dist(R, g, "ok").items():
            rd[e] += (1 - lam) * w if mode != "ok" else w
        if mode != "ok" and lam > 0:
            for e, w in round_dist(R, g, mode).items():
                rd[e] += lam * w
        vec = np.zeros(len(OUT_INDEX))
        for e, w in rd.items():
            s = g + e
            if (s and s[-1] == fx.EOS) or len(s) == fx.MAX_NEW:
                vec[OUT_INDEX[s]] += w
            else:
                vec += w * from_prefix(s)
        return vec

    return from_prefix(())


def reference(setting):
    return rnp.sequence_probs(rnp.step_table("vllm", SETTINGS[setting]))


if __name__ == "__main__":
    if "--check" in sys.argv:
        for s in SETTINGS:
            R = Rows(s)
            ok, ref = sequence_dist(R, "ok"), reference(s)
            err = float(np.abs(ok - ref).max())
            print(s, "max |correct - R_np| =", err, "sum", ok.sum())
            assert err < 1e-12, err
            for c in CLASSES:
                alt = sequence_dist(R, c)
                print(" ", c, "TV at lam=1:", round(float(0.5 * np.abs(alt - ref).sum()), 5), "sum", round(alt.sum(), 12))
