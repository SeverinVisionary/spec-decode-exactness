"""FR20 follow-up: exact fp64 target distribution for the repro_v14 fixture, and each arm's TV against it.

Rebuilds the target weights exactly as repro_v14.make_model (numpy rng, seed 1, rounded to bf16) and runs code/rnp.py's
Llama64 forward in fp64. Enumerates all continuations of PROMPT (MAX_NEW tokens, stopping at EOS).
    python exact_ref.py <out dir with base.json spec.json spec_rs.json>
"""
import collections
import gzip
import itertools
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "code"))
import rnp  # noqa: E402

V, EOS, MAX_NEW = 8, 7, 4
PROMPT = [1, 2, 3, 1, 2]


def bf16(a):
    """Round float32 to bfloat16 (round to nearest even), returned as float64."""
    b = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def weights(seed):
    rng = np.random.default_rng(seed)
    hidden, heads, head_dim, inter, layers = 64, 2, 32, 128, 2

    def w(*shape, std=0.3):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    sd = {"model.embed_tokens.weight": w(V, hidden, std=1.0), "model.norm.weight": np.ones(hidden, np.float32),
          "lm_head.weight": w(V, hidden, std=1.5)}
    for i in range(layers):
        p = f"model.layers.{i}."
        sd |= {p + "input_layernorm.weight": np.ones(hidden, np.float32),
               p + "post_attention_layernorm.weight": np.ones(hidden, np.float32),
               p + "self_attn.q_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.k_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.v_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.o_proj.weight": w(hidden, heads * head_dim),
               p + "mlp.gate_proj.weight": w(inter, hidden), p + "mlp.up_proj.weight": w(inter, hidden),
               p + "mlp.down_proj.weight": w(hidden, inter)}
    return {k: bf16(v) for k, v in sd.items()}


class Model(rnp.Llama64):
    def __init__(self, seed):
        self.w, self.L, self.H, self.eps, self.theta = weights(seed), 2, 2, 1e-6, 10000.0


def step_probs(z, top_k):
    z = np.asarray(z, float)
    if top_k:
        kth = np.sort(z)[-top_k]
        z = np.where(z >= kth, z, -np.inf)
    e = np.where(np.isfinite(z), np.exp(z - z[np.isfinite(z)].max()), 0.0)
    return e / e.sum()


def exact(model, top_k):
    dist, cache = {}, {}

    def p(hist):
        k = tuple(hist)
        if k not in cache:
            cache[k] = step_probs(model.logits(PROMPT + list(hist)), top_k)
        return cache[k]

    def walk(hist, prob):
        if len(hist) == MAX_NEW or (hist and hist[-1] == EOS):
            dist[tuple(hist)] = dist.get(tuple(hist), 0.0) + prob
            return
        q = p(hist)
        for t in range(V):
            if q[t] > 0:
                walk(hist + [t], prob * q[t])
    walk([], 1.0)
    return dist


def tv(counts, dist):
    n = sum(counts.values())
    keys = set(counts) | set(dist)
    return 0.5 * sum(abs(counts.get(k, 0) / n - dist.get(k, 0.0)) for k in keys)


def main(out):
    out = pathlib.Path(out)
    target = Model(1)
    arms = {a: json.loads(gzip.open(out / f"{a}.json.gz").read()) for a in ("base", "spec", "spec_rs")
            if (out / f"{a}.json.gz").exists()}
    rng = np.random.default_rng(0)
    res = {}
    for name, top_k in (("T1", 0), ("topk3", 3)):
        dist = exact(target, top_k)
        keys = list(dist)
        pr = np.array([dist[k] for k in keys])
        res[name] = {}
        for a, d in arms.items():
            seqs = [tuple(s) for s in d["settings"][name]["seqs"]]
            c = collections.Counter(seqs)
            n = len(seqs)
            # sampling floor: TV of n exact draws against the exact distribution (200 simulations)
            sims = [0.5 * np.abs(rng.multinomial(n, pr) / n - pr).sum() for _ in range(200)]
            outside = sum(v for k, v in c.items() if dist.get(k, 0.0) == 0.0)
            res[name][a] = dict(n=n, tv_vs_exact=round(tv(c, dist), 4), floor_mean=round(float(np.mean(sims)), 4),
                                floor_p99=round(float(np.quantile(sims, 0.99)), 4), draws_outside_support=outside)
            print(name, a, res[name][a], flush=True)
        # per-step marginal of the second token for a quick look at where mass moves
        for a, d in arms.items():
            seqs = [tuple(s) for s in d["settings"][name]["seqs"]]
            m = collections.Counter(s[1] for s in seqs if len(s) > 1)
            n = sum(m.values())
            print("   ", a, "token2", {k: round(v / n, 3) for k, v in sorted(m.items())})
        m2 = collections.defaultdict(float)
        for k, v in dist.items():
            if len(k) > 1:
                m2[k[1]] += v
        print("    exact token2", {k: round(v, 3) for k, v in sorted(m2.items())})
    (out / "EXACT_REF.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
