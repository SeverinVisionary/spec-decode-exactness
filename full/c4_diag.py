"""Diagnostic for FR16's C4 (row alignment), which matched no row within 1e-4. CPU only.
Not a preregistered test: it does not change any FR16 flag.

    D40_WORK=$PWD/../data python c4_diag.py stagea/data/repro_trtllm      # writes C4_DIAG.{txt,json} there

For each recorded `py_draft_logits` row, search the candidate contexts and report the closest one:
- model: fp64 draft or target (`rnp.Llama64`);
- base: the request's tokens at verification (`tokens_before`), or the same minus its last token;
- j: the first j draft tokens appended (j = 0..K);
- pos0: the position of the first token (0, the reference; +1 or -1 tests a RoPE offset).
Rows are compared as centred logits (logits minus their mean, which is invariant to a constant shift), by max |Δ|.
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import rnp  # noqa: E402

N_PER_RUN = 60
MATCH = 1e-3


class Model(rnp.Llama64):
    """rnp.Llama64 with the first token's position as a parameter (the body is rnp.Llama64.logits)."""

    def logits_at(self, ids, pos0=0):
        w, T = self.w, len(ids)
        x = w["model.embed_tokens.weight"][list(ids)]
        pos = np.arange(T, dtype=np.float64) + pos0
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q, k, v = (h @ w[p + f"self_attn.{n}_proj.weight"].T for n in "qkv")
            d = q.shape[-1] // self.H
            q = self.rope(q.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            k = self.rope(k.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            v = v.reshape(T, self.H, d).transpose(1, 0, 2)
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


def centred(v):
    v = np.asarray(v, float)
    return v - v.mean()


def main(folder):
    folder = pathlib.Path(folder)
    models = {m: Model(m) for m in ("draft", "target")}
    cache = {}

    def ref(m, ids, pos0):
        k = (m, tuple(ids), pos0)
        if k not in cache:
            cache[k] = centred(models[m].logits_at(ids, pos0))
        return cache[k]

    # self-check: pos0 = 0 reproduces rnp.Llama64.logits exactly
    ids = [1, 2, 3, 1, 2, 5]
    assert np.allclose(models["draft"].logits_at(ids), rnp.Llama64("draft").logits(ids), atol=0, rtol=0)

    out, lines = {}, []
    for arm in ("A1", "A2", "A3"):
        a = json.load(gzip.open(folder / f"{arm}.json.gz", "rt"))
        for run in a["runs"]:
            if "events" not in run:
                continue
            key = f"{arm}/{run['setting']}/u{run['u']}"
            evs = [e for e in run["events"] if e["path"] == "rejection_sampling" and e["draft_logits"]][:N_PER_RUN]
            best_by_row = collections.defaultdict(collections.Counter)
            dists = collections.defaultdict(list)
            examples = []
            for e in evs:
                K, pre, d = e["draft_token_length"], e["tokens_before"], e["draft_tokens"][:e["draft_token_length"]]
                for r, row in enumerate(e["draft_logits"]):
                    row = centred(row)
                    cands = []
                    for m in models:
                        for bname, base in (("pre", pre), ("pre[:-1]", pre[:-1])):
                            for j in range(K + 1):
                                for pos0 in (0, 1, -1):
                                    diff = float(np.abs(row - ref(m, base + d[:j], pos0)).max())
                                    cands.append((diff, f"{m}|{bname}+d[:{j}]|pos0={pos0:+d}"))
                    cands.sort()
                    diff, label = cands[0]
                    best_by_row[r][label if diff <= MATCH else "none<=1e-3"] += 1
                    dists[r].append(diff)
                    if len(examples) < 3:
                        examples.append(dict(tokens_before=pre, draft=d, row=r, best=cands[:3]))
            res = {f"row{r}": dict(best=dict(best_by_row[r]),
                                   diff_median=float(np.median(dists[r])), diff_max=float(np.max(dists[r])))
                   for r in sorted(best_by_row)}
            out[key] = dict(events=len(evs), rows=res, examples=examples)
            lines.append(f"{key} events={len(evs)}")
            for r, v in res.items():
                lines.append(f"  {r}: median {v['diff_median']:.2e} max {v['diff_max']:.2e} best {v['best']}")
    (folder / "C4_DIAG.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (folder / "C4_DIAG.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
