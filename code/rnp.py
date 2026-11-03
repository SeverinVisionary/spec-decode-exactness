"""R_np: independent fp64 reference (PREREGISTRATION §5.1). Imports no inference engine.

Llama forward written from the architecture definition; sampling stages written per engine
from each engine's documented chain order. Weights are read with safetensors only."""
import numpy as np
from safetensors.numpy import load_file

import fixture as fx


class Llama64:
    def __init__(self, name, n_layers=2, n_heads=2, eps=1e-6, theta=10000.0):
        self.w = {k: v.astype(np.float64) for k, v in load_file(fx.MODELS / name / "model.safetensors").items()}
        self.L, self.H, self.eps, self.theta = n_layers, n_heads, eps, theta

    def rms(self, x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * g

    def rope(self, x, pos):
        d = x.shape[-1]
        inv = 1.0 / self.theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        ang = np.outer(pos, inv)
        cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def logits(self, ids, n_layers=None):
        w, T = self.w, len(ids)
        x = w["model.embed_tokens.weight"][ids]
        pos = np.arange(T, dtype=np.float64)
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(n_layers or self.L):
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


def _softmax(z):
    z = z - z[np.isfinite(z)].max()
    e = np.where(np.isfinite(z), np.exp(z), 0.0)
    return e / e.sum()


def hf_dist(logits, hist, s):
    """HF order: repetition penalty -> temperature -> top-k -> top-p -> min-p -> softmax."""
    z = logits.copy()
    if s.get("repetition_penalty", 1.0) != 1.0:
        r = s["repetition_penalty"]
        for t in set(hist):
            z[t] = z[t] * r if z[t] < 0 else z[t] / r
    if s.get("temperature", 1.0) != 1.0:
        z = z / s["temperature"]
    k = s.get("top_k", 0)
    if k:
        kth = np.sort(z)[-k]
        z = np.where(z < kth, -np.inf, z)
    tp = s.get("top_p", 1.0)
    if tp < 1.0:
        order = np.argsort(z)
        cum = np.cumsum(_softmax(z)[order])
        drop = cum <= 1 - tp
        drop[-1] = False
        z[order[drop]] = -np.inf
    mp = s.get("min_p")
    if mp is not None:
        p = _softmax(z)
        z = np.where(p < mp * p.max(), -np.inf, z)
    return _softmax(z)


def lcpp_dist(logits, hist, s, last_n=64):
    """llama.cpp default chain order: penalties -> top-k -> top-p -> min-p -> temperature -> dist."""
    z = logits.copy()
    if s.get("repetition_penalty", 1.0) != 1.0:
        r = s["repetition_penalty"]
        for t in set(hist[-last_n:]):
            z[t] = z[t] * r if z[t] <= 0 else z[t] / r
    alive = np.ones(len(z), bool)
    k = s.get("top_k", 0)
    if k:
        alive[np.argsort(-z, kind="stable")[k:]] = False
    tp = s.get("top_p", 1.0)
    if tp < 1.0:
        zz = np.where(alive, z, -np.inf)
        order = np.argsort(-zz, kind="stable")
        cum = np.cumsum(_softmax(zz)[order])
        last = int(np.argmax(cum >= tp)) + 1
        alive[order[last:]] = False
    mp = s.get("min_p", 0.0) or 0.0
    if mp > 0:
        zmax = z[alive].max()
        alive &= z >= zmax + np.log(mp)
    t = s.get("temperature", 1.0)
    return _softmax(np.where(alive, z / t, -np.inf))


def vllm_dist(logits, hist, s):
    """vLLM v0.30.0 V1 sampler order (v1/sample/sampler.py docstring and forward/sample):
    repetition penalty over prompt+output tokens (divide if logit > 0, else multiply;
    _custom_ops.apply_repetition_penalties_torch) -> temperature -> min-p (prob < min_p * max prob,
    builtin MinPLogitsProcessor, argmax-invariant stage) -> top-k (logit < k-th largest) -> top-p
    (ascending sort, drop cumsum <= 1 - p, keep the last; ops/topk_topp_sampler.py) -> softmax."""
    z = logits.copy()
    if s.get("repetition_penalty", 1.0) != 1.0:
        r = s["repetition_penalty"]
        for t in set(hist):
            z[t] = z[t] / r if z[t] > 0 else z[t] * r
    if s.get("temperature", 1.0) != 1.0:
        z = z / s["temperature"]
    mp = s.get("min_p", 0.0) or 0.0
    if mp > 0:
        p = _softmax(z)
        z = np.where(p < mp * p.max(), -np.inf, z)
    k = s.get("top_k", 0)
    if k:
        kth = np.sort(z)[-k]
        z = np.where(z < kth, -np.inf, z)
    tp = s.get("top_p", 1.0)
    if tp < 1.0:
        order = np.argsort(z, kind="stable")
        cum = np.cumsum(_softmax(z)[order])
        drop = cum <= 1 - tp
        drop[-1] = False
        z[order[drop]] = -np.inf
    return _softmax(z)


DISTS = {"hf": hf_dist, "lcpp": lcpp_dist, "vllm": vllm_dist}  # the full study adds its engines (Phase 0b)


def step_table(engine, setting, name="target", n_layers=None):
    """Per-prefix next-token distribution over the 400 nonterminal prefixes."""
    m = Llama64(name)
    f = DISTS[engine]
    rows = []
    for g in fx.nonterminal_prefixes():
        ids = fx.PROMPT + g
        rows.append(f(m.logits(ids, n_layers), ids, setting))
    return np.stack(rows)


def sequence_probs(step):
    """Exact probability of each of the 2,801 outcomes from a per-prefix step table."""
    idx = {tuple(g): i for i, g in enumerate(fx.nonterminal_prefixes())}
    out = []
    for seq in fx.outcomes():
        p = 1.0
        for j, tok in enumerate(seq):
            p *= step[idx[tuple(seq[:j])], tok]
        out.append(p)
    return np.array(out)
