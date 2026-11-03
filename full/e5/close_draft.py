"""FR22 E5: a draft close to the v1.4 fixture's target, and the choice of its noise level on CPU.

draft weights = target weights (repro_v14.make_model, seed 1) + sigma * std(w) * N(0, 1) per tensor (numpy seed 7),
rounded to bf16 like the target. The fp64 forward (`exact_ref.Model`) gives, along 300 target-sampled sequences
(S0), the mean per-token acceptance sum_x min(p, q). SIGMA is the largest grid value with acceptance >= 0.8.

    python close_draft.py                 # prints the acceptance per sigma
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "drafts" / "trtllm_v14" / "analysis"))
import exact_ref as X  # noqa: E402

GRID = (0.02, 0.05, 0.1, 0.2, 0.3)
SIGMA = 0.02  # FR22 run note 2: acceptance 0.886 (0.05 gives 0.750 < 0.8)


def draft_weights(sigma, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in X.weights(1).items():
        if k.endswith("norm.weight"):
            out[k] = v
        else:
            out[k] = X.bf16(v + sigma * v.std() * rng.standard_normal(v.shape))
    return out


class M(X.rnp.Llama64):
    def __init__(self, w):
        self.w, self.L, self.H, self.eps, self.theta = w, 2, 2, 1e-6, 10000.0


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def acceptance(sigma, n=300, seed=3):
    t, d = M(X.weights(1)), M(draft_weights(sigma))
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n):
        g = []
        while len(g) < X.MAX_NEW:
            ids = X.PROMPT + g
            p, q = softmax(t.logits(ids)), softmax(d.logits(ids))
            acc.append(np.minimum(p, q).sum())
            tok = int(rng.choice(X.V, p=p))
            g.append(tok)
            if tok == X.EOS:
                break
    return float(np.mean(acc))


if __name__ == "__main__":
    for s in GRID:
        print(s, round(acceptance(s), 4), flush=True)
