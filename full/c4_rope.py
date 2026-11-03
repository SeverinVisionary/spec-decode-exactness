"""RoPE-variant check for the recorded TensorRT-LLM draft-logits rows (follow-up to `c4_search.py`). CPU only.
Not a preregistered test: it changes no FR16 flag.

    D40_WORK=$PWD/../data python c4_rope.py stagea/data/repro_trtllm      # writes C4_ROPE.{txt,json} there

The engine runs the head_dim-32 re-layout (`code/vllm_export.py`): q/k/v padded from 8 to 32 rows per head, q
scaled by 2, `partial_rotary_factor = 0.25`. The Stage A/B gates validated this layout for the target model only.
This script recomputes the fp64 draft model under three readings of the re-layout's RoPE, and compares each
recorded row (centred logits, max |Δ|) at the contexts pre + d[:j] and pre[:-1] + d[:j]:
- `exact`: the intended rotation (first 8 dims, inv_freq = θ^(-2i/8), neox pairing (i, i+4)), the same as rnp;
- `freq32`: first 8 dims rotated, but inv_freq = θ^(-2i/32) (rotary dim taken from head_dim);
- `full32`: partial_rotary_factor ignored: all 32 dims rotated, inv_freq = θ^(-2i/32), neox pairing (i, i+16),
  so the padded zero rows pick up q_i·sin and k_i·sin terms.
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import rnp  # noqa: E402

N_PER_RUN = None  # every event (review: a prefix cap left later row shapes untested)
MATCH = 1e-3
PAD = 32


def rope_nd(x, pos, rot_dim, base_dim, theta=10000.0):
    """neox rotation of the first rot_dim dims of x (..., T, D), inv_freq = θ^(-2i/base_dim), i < rot_dim/2."""
    half = rot_dim // 2
    inv = 1.0 / theta ** (np.arange(0, rot_dim, 2, dtype=np.float64) / base_dim)
    ang = np.outer(pos, inv)
    cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
    r, rest = x[..., :rot_dim], x[..., rot_dim:]
    r1, r2 = r[..., :half], r[..., half:]
    return np.concatenate([r * cos + np.concatenate([-r2, r1], -1) * sin, rest], -1)


class Variant(rnp.Llama64):
    def logits_v(self, ids, variant):
        w, T, H = self.w, len(ids), self.H
        x = w["model.embed_tokens.weight"][list(ids)]
        pos = np.arange(T, dtype=np.float64)
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q, k, v = (h @ w[p + f"self_attn.{n}_proj.weight"].T for n in "qkv")
            d = q.shape[-1] // H
            q, k, v = (t.reshape(T, H, d).transpose(1, 0, 2) for t in (q, k, v))
            if variant == "exact":
                q, k, scale = rope_nd(q, pos, d, d), rope_nd(k, pos, d, d), 1 / np.sqrt(d)
            elif variant == "freq32":
                q, k, scale = rope_nd(q, pos, d, PAD), rope_nd(k, pos, d, PAD), 1 / np.sqrt(d)
            elif variant == "full32":
                qp, kp = np.zeros((H, T, PAD)), np.zeros((H, T, PAD))
                qp[..., :d], kp[..., :d] = q * 2.0, k  # the re-layout's q scale, exact in fp32
                q, k, scale = rope_nd(qp, pos, PAD, PAD), rope_nd(kp, pos, PAD, PAD), 1 / np.sqrt(PAD)
            a = q @ k.transpose(0, 2, 1) * scale + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            o = (a @ v).transpose(1, 0, 2).reshape(T, -1)
            x = x + o @ w[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g, u = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + ((g / (1 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T
        x = self.rms(x, w["model.norm.weight"])
        return x[-1] @ w["lm_head.weight"].T


def centred(v):
    v = np.asarray(v, float)
    return v - v.mean()


def main(folder):
    folder = pathlib.Path(folder)
    m = Variant("draft")
    ids = [1, 2, 3, 1, 2, 5]
    assert np.allclose(m.logits_v(ids, "exact"), rnp.Llama64("draft").logits(ids), atol=1e-12, rtol=0)
    cache, out, lines = {}, {}, []

    def ref(ctx, var):
        k = (tuple(ctx), var)
        if k not in cache:
            cache[k] = centred(m.logits_v(ctx, var))
        return cache[k]

    for arm in ("A1", "A2", "A3"):
        a = json.load(gzip.open(folder / f"{arm}.json.gz", "rt"))
        for run in a["runs"]:
            if "events" not in run:
                continue
            key = f"{arm}/{run['setting']}/u{run['u']}"
            evs = [e for e in run["events"] if e["path"] == "rejection_sampling" and e["draft_logits"]][:N_PER_RUN]
            tally, dists = collections.defaultdict(collections.Counter), collections.defaultdict(list)
            for e in evs:
                K, pre, d = e["draft_token_length"], e["tokens_before"], e["draft_tokens"][:e["draft_token_length"]]
                for r, row in enumerate(e["draft_logits"]):
                    row = centred(row)
                    cands = sorted((float(np.abs(row - ref(base + d[:j], var)).max()), f"{var}|{bn}+d[:{j}]")
                                   for var in ("exact", "freq32", "full32")
                                   for bn, base in (("pre", pre), ("pre[:-1]", pre[:-1]))
                                   for j in range(K + 1))
                    diff, label = cands[0]
                    tally[r][label if diff <= MATCH else "none<=1e-3"] += 1
                    dists[r].append(diff)
            out[key] = {f"row{r}": dict(best=dict(tally[r]), diff_median=float(np.median(dists[r])),
                                        diff_max=float(np.max(dists[r]))) for r in sorted(tally)}
            lines.append(f"{key} events={len(evs)}")
            for r, v in out[key].items():
                lines.append(f"  {r}: median {v['diff_median']:.2e} max {v['diff_max']:.2e} best {v['best']}")
    (folder / "C4_ROPE.json").write_text(json.dumps(out, indent=1) + "\n")
    (folder / "C4_ROPE.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
