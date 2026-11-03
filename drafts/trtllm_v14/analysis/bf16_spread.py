"""How far apart can two bf16 executions of the same fixture be? Emulates bf16 activations by rounding at different
points of the forward (two plausible kernel orders) and reports the TV between the resulting exact sequence
distributions and fp64. Scale check for FR20, not a model of TRT-LLM's kernels."""
import sys
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exact_ref import Model, bf16, exact, PROMPT  # noqa: E402


class B16(Model):
    def __init__(self, seed, where):
        super().__init__(seed)
        self.where = where

    def logits(self, ids, n_layers=None):
        r = bf16 if True else (lambda a: a)
        w, T = self.w, len(ids)
        x = r(w["model.embed_tokens.weight"][ids])
        pos = np.arange(T, dtype=np.float64)
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = r(self.rms(x, w[p + "input_layernorm.weight"]))
            q, k, v = (r(h @ w[p + f"self_attn.{n}_proj.weight"].T) for n in "qkv")
            d = q.shape[-1] // self.H
            q = self.rope(q.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            k = self.rope(k.reshape(T, self.H, d).transpose(1, 0, 2), pos)
            if "rope" in self.where:
                q, k = r(q), r(k)
            v = v.reshape(T, self.H, d).transpose(1, 0, 2)
            a = q @ k.transpose(0, 2, 1) / np.sqrt(d) + mask
            a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            if "probs" in self.where:
                a = r(a)
            o = r((a @ v).transpose(1, 0, 2).reshape(T, -1))
            x = r(x + r(o @ w[p + "self_attn.o_proj.weight"].T))
            h = r(self.rms(x, w[p + "post_attention_layernorm.weight"]))
            g, u = r(h @ w[p + "mlp.gate_proj.weight"].T), r(h @ w[p + "mlp.up_proj.weight"].T)
            x = r(x + r(r((g / (1 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T))
        x = r(self.rms(x, w["model.norm.weight"]))
        z = x[-1] @ w["lm_head.weight"].T
        return r(z) if "logits" in self.where else z


def tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


for name, k in (("T1", 0), ("topk3", 3)):
    ref = exact(Model(1), k)
    vs = {w: exact(B16(1, w), k) for w in ("", "rope", "rope+probs", "rope+probs+logits")}
    print(name, {w or "plain": round(tvd(ref, d), 4) for w, d in vs.items()},
          "max pairwise", round(max(tvd(a, b) for a in vs.values() for b in vs.values()), 4))
