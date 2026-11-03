"""Exploratory follow-up to FR17 Q2 (not preregistered; changes no flag). CPU only.

    D40_WORK=$PWD/../data python c4_probe_followup.py stagea/data/c4_probe      # writes C4_FOLLOWUP.{json,txt} there

FR17 Q2 found no draft forward equal to the fp64 draft at its nominal input (the round's first draft request tokens
plus the tokens drafted before it). The probe also showed the round's context forward computing only `chunk` tokens
from `context_current_position`, which ends before the newest target token. This script asks which input each forward
did condition on. For forward k of a round, with base = that round's first draft request tokens and m trailing base
tokens taken as missing from the draft KV (m = 0..3), it compares the forward's output row (softmax, max |Δp|) with
the fp64 draft at four layouts:
- `nominal`: base + d[:k] at positions 0, 1, ... (m is ignored);
- `drop`: base[:-m] + d[:k], contiguous positions;
- `gap`: base[:-m] at 0.., d[:k] after the m missing positions (those slots not attended);
- `zero`: as `gap`, but the m missing slots attended with K = V = 0 (unwritten cache).
A layout matches if max |Δp| ≤ TOL. Unmatched forwards mean none of these inputs (for example stale KV with unknown
contents).
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import rnp  # noqa: E402

TOL = 1e-4
MAX_M = 3


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Slots(rnp.Llama64):
    def logits_slots(self, slots):
        """slots: list of (token or None, position). A None slot has K = V = 0 at every layer and is attended."""
        w, T = self.w, len(slots)
        ids = [0 if t is None else t for t, _ in slots]
        zero = np.array([t is None for t, _ in slots])
        pos = np.array([p for _, p in slots], dtype=np.float64)
        x = w["model.embed_tokens.weight"][ids]
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q, k, v = (h @ w[p + f"self_attn.{n}_proj.weight"].T for n in "qkv")
            d = q.shape[-1] // self.H
            q = self.rope(q.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            k = self.rope(k.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            v = v.reshape(T, self.H, d).transpose(1, 0, 2)
            k[:, zero, :], v[:, zero, :] = 0.0, 0.0
            a = q @ k.transpose(0, 2, 1) / np.sqrt(d) + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            o = (a @ v).transpose(1, 0, 2).reshape(T, -1)
            x = x + o @ w[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g, u = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + ((g / (1 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T
        x = self.rms(x, w["model.norm.weight"])
        return x[-1] @ w["lm_head.weight"].T


def layouts(base, dk):
    n = len(base)
    yield "nominal", 0, [(t, j) for j, t in enumerate(base + dk)]
    for m in range(1, min(MAX_M, n - 1) + 1):
        kept = [(t, j) for j, t in enumerate(base[:n - m])]
        yield "drop", m, kept + [(t, n - m + j) for j, t in enumerate(dk)]
        if dk:
            yield "gap", m, kept + [(t, n + j) for j, t in enumerate(dk)]
            yield "zero", m, kept + [(None, n - m + j) for j in range(m)] + [(t, n + j) for j, t in enumerate(dk)]


def main(folder):
    folder = pathlib.Path(folder)
    model = Slots("draft")
    ids = [1, 2, 3, 1, 2, 5]
    assert np.allclose(model.logits_slots([(t, j) for j, t in enumerate(ids)]), rnp.Llama64("draft").logits(ids),
                       atol=1e-12, rtol=0)
    cache = {}

    def ref(slots):
        k = tuple(slots)
        if k not in cache:
            cache[k] = softmax(model.logits_slots(slots))
        return cache[k]

    p1 = json.load(gzip.open(folder / "P1.json.gz", "rt"))
    tally = collections.defaultdict(collections.Counter)
    best_dev = collections.defaultdict(list)
    examples = []
    for req in p1["requests"]:
        fwd = collections.defaultdict(list)
        for f in req["forwards"]:
            if len(f["reqs"]) == 1:
                fwd[f["round"]].append((f["reqs"][0], np.asarray(f["logits"], float).reshape(-1, 8)[-1]))
        for v in req["verifications"]:
            K, rnd = v["draft_token_length"], v["round"]
            if K == 0 or rnd not in fwd:
                continue
            d, fw = v["draft_tokens"][:K], fwd[rnd]
            start = fw[0][0]["kind"]
            base = fw[0][0]["tokens"]
            for k, (r, row) in enumerate(fw[:K]):
                p = softmax(row)
                cands = sorted((float(np.abs(p - ref(s)).max()), lay, m) for lay, m, s in layouts(base, d[:k]))
                dev, lay, m = cands[0]
                key = f"{start}-round/forward{k}"
                tally[key][f"{lay}/m{m}" if dev <= TOL else "none"] += 1
                best_dev[key].append(dev)
                if len(examples) < 6 and k == 0 or (len(examples) < 12 and dev > TOL):
                    examples.append(dict(request=req["i"], round=rnd, key=key, base=base, draft=d,
                                         ccp=r["context_current_position"], chunk=r["context_chunk_size"],
                                         best=[dict(dev=c[0], layout=c[1], m=c[2]) for c in cands[:3]]))
    out = {k: dict(n=len(best_dev[k]), best=dict(tally[k]), dev_median=float(np.median(best_dev[k])),
                   dev_max=float(np.max(best_dev[k]))) for k in sorted(tally)}
    (folder / "C4_FOLLOWUP.json").write_text(json.dumps(dict(summary=out, examples=examples), indent=1) + "\n")
    lines = [f"{k}: n={v['n']} best {v['best']} dev median {v['dev_median']:.2e} max {v['dev_max']:.2e}"
             for k, v in out.items()]
    (folder / "C4_FOLLOWUP.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
