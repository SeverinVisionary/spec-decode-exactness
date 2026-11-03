"""D-40 fixture (PREREGISTRATION §3): tiny Llama target/draft pair over V=8, EOS=7."""
import itertools
import json
import os
import pathlib

import numpy as np
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

V, EOS, MAX_NEW = 8, 7, 4
PROMPT = [1, 2, 3, 1, 2]
WORK = pathlib.Path(os.environ.get("D40_WORK", pathlib.Path.home() / ".cache/d40"))
MODELS = WORK / "models"


def config(**over):
    kw = dict(
        vocab_size=V, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, max_position_embeddings=64,
        bos_token_id=None, eos_token_id=EOS, pad_token_id=EOS, tie_word_embeddings=False,
        torch_dtype="float32",
    )
    kw.update(over)
    return LlamaConfig(**kw)


def nonterminal_prefixes():
    """The 400 generated prefixes (length 0..3, no EOS) at which the model is queried."""
    out = []
    for n in range(MAX_NEW):
        out.extend(list(p) for p in itertools.product(range(EOS), repeat=n))
    return out


def outcomes():
    """The 2,801 terminal sequences: stop at EOS or after MAX_NEW tokens."""
    out = []
    for n in range(1, MAX_NEW + 1):
        for p in itertools.product(range(EOS), repeat=n - 1):
            out.append(list(p) + [EOS])
    out.extend(list(p) for p in itertools.product(range(EOS), repeat=MAX_NEW))
    return out


@torch.no_grad()
def step_logits(model, prefixes):
    """Raw next-token logits (no cache, full prefix) for each generated prefix, float64 copy."""
    rows = []
    for g in prefixes:
        ids = torch.tensor([PROMPT + g])
        rows.append(model(input_ids=ids, use_cache=False).logits[0, -1].double().numpy())
    return np.stack(rows)


def softmax(x):
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def build_target(scale, seed=0, **over):
    torch.manual_seed(seed)
    m = LlamaForCausalLM(config(**over)).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(scale)
    return m


def build_draft(target, sigma, seed=1):
    d = LlamaForCausalLM(target.config).eval()
    d.load_state_dict(target.state_dict())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in d.parameters():
            p.add_(torch.randn(p.shape, generator=g) * sigma)
    return d


def entropy_stats(p):
    h = -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)
    return float(np.median(h)), float(h.min()), float(h.max())


def reach_weights(p, prefixes):
    """Probability the target (S0) reaches each nonterminal prefix."""
    idx = {tuple(g): i for i, g in enumerate(prefixes)}
    w = np.zeros(len(prefixes))
    for i, g in enumerate(prefixes):
        r = 1.0
        for k in range(len(g)):
            r *= p[idx[tuple(g[:k])], g[k]]
        w[i] = r
    return w


def save(model, name):
    path = MODELS / name
    model.save_pretrained(path, safe_serialization=True)
    GenerationConfig(eos_token_id=EOS, pad_token_id=EOS, do_sample=False).save_pretrained(path)
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    pre = nonterminal_prefixes()
    t = build_target(a.scale)
    d = build_draft(t, a.sigma)
    p, q = softmax(step_logits(t, pre)), softmax(step_logits(d, pre))
    w = reach_weights(p, pre)
    acc = np.minimum(p, q).sum(-1)
    rep = dict(scale=a.scale, sigma=a.sigma, entropy_median_min_max=entropy_stats(p),
               acceptance_reach_weighted=float((w * acc).sum() / w.sum()),
               acceptance_unweighted=float(acc.mean()))
    print(json.dumps(rep))
    if a.save:
        save(t, "target"), save(d, "draft")
