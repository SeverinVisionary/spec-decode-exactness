"""D-40 full study, Phase 0b F2: the F-B fixture (PREREGISTRATION.md §3), built on CPU.

F-B is frozen as V = 32 (EOS = 31), 6 new tokens, 4 layers, hidden 64, random init with seed 2.
Choices the frozen text leaves open, fixed here before any F-B model exists:
- prompt [1, 2, 3, 1, 2], as F-A;
- 2 attention heads (head_dim 32, which every engine's attention kernels accept: no re-layout),
  2 KV heads, intermediate 128, rms_norm_eps 1e-6, rope theta 10^4, untied embeddings, as F-A;
- draft = target + Gaussian noise of scale sigma on every parameter, generator seed 3 (F-A: target
  seed 0, draft seed 1);
- the pilot's §3 rules set the constants: median next-token entropy in [0.8, 1.8] nats (goal 1.3)
  and mean acceptance sum_x min(p, q) in [0.5, 0.8] (goal 0.65), each bisected to its goal as in
  `code/calibrate.py`.

F-B's sequence space (about 10^9 outcomes) cannot be enumerated, so both statistics are taken over
the prefixes the target visits: `N_CAL` target sequences at S0, drawn with a fixed torch seed. The
entropy median is over the distinct visited prefixes; the acceptance mean is over visits, which is
the reach weighting `calibrate.py` uses for F-A.

    python fb_fixture.py            # calibrate, save to $D40_WORK/models/fb_{target,draft}, print report
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM  # noqa: E402

import fixture as fx  # noqa: E402

V, EOS, MAX_NEW = 32, 31, 6
PROMPT = [1, 2, 3, 1, 2]
SEED_TARGET, SEED_DRAFT, SEED_CAL = 2, 3, 4
N_CAL = 2000
ENTROPY_GOAL, ACCEPT_GOAL = 1.3, 0.65
NAMES = ("fb_target", "fb_draft")


def config():
    return LlamaConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                       num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64,
                       bos_token_id=None, eos_token_id=EOS, pad_token_id=EOS, tie_word_embeddings=False,
                       rms_norm_eps=1e-6, torch_dtype="float32")


def build_target(scale):
    torch.manual_seed(SEED_TARGET)
    m = LlamaForCausalLM(config()).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(scale)
    return m


def build_draft(target, sigma):
    d = LlamaForCausalLM(target.config).eval()
    d.load_state_dict(target.state_dict())
    g = torch.Generator().manual_seed(SEED_DRAFT)
    with torch.no_grad():
        for p in d.parameters():
            p.add_(torch.randn(p.shape, generator=g) * sigma)
    return d


@torch.no_grad()
def visited_prefixes(target):
    """Generated prefixes (before EOS, shorter than MAX_NEW) visited by N_CAL S0 target sequences."""
    g = torch.Generator().manual_seed(SEED_CAL)
    visits = {}
    for _ in range(N_CAL):
        gen = []
        while len(gen) < MAX_NEW:
            visits[tuple(gen)] = visits.get(tuple(gen), 0) + 1
            logits = target(input_ids=torch.tensor([PROMPT + gen]), use_cache=False).logits[0, -1].double()
            tok = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
            gen.append(tok)
            if tok == EOS:
                break
    return visits


@torch.no_grad()
def probs(model, prefixes):
    return np.stack([torch.softmax(model(input_ids=torch.tensor([PROMPT + list(g)]), use_cache=False)
                                   .logits[0, -1].double(), -1).numpy() for g in prefixes])


def entropy_median(p):
    return float(np.median(-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)))


def bisect(f, lo, hi, goal, decreasing, iters=14):
    """Geometric bisection; the endpoints must bracket the goal (checked), and every value must be finite."""
    flo, fhi = f(lo), f(hi)
    assert np.isfinite(flo) and np.isfinite(fhi), (flo, fhi)
    assert (flo > goal) == decreasing and (fhi > goal) != decreasing, f"endpoints do not bracket {goal}: {flo}, {fhi}"
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if (f(mid) > goal) == decreasing:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def calibrate():
    # entropy over a fixed prefix set from the unscaled target's visits: the set must not move with the
    # bisection variable, or the bisection chases a moving target
    base = visited_prefixes(build_target(1.0))
    pre0 = sorted(base)
    scale = bisect(lambda s: entropy_median(probs(build_target(s), pre0)), 0.5, 200.0, ENTROPY_GOAL, decreasing=True)
    t = build_target(scale)
    visits = visited_prefixes(t)
    pre = sorted(visits)
    w = np.array([visits[g] for g in pre], float)
    p = probs(t, pre)

    def acc(sig):
        q = probs(build_draft(t, sig), pre)
        return float((w * np.minimum(p, q).sum(-1)).sum() / w.sum())

    sigma = bisect(acc, 0.002, 0.5, ACCEPT_GOAL, decreasing=True)
    d = build_draft(t, sigma)
    a = np.minimum(p, probs(d, pre)).sum(-1)
    ent = entropy_median(p)
    acc_w = float((w * a).sum() / w.sum())
    rep = dict(scale=scale, sigma=sigma, n_cal_sequences=N_CAL, distinct_visited_prefixes=len(pre),
               entropy_median_visited=ent, acceptance_visit_weighted=acc_w,
               acceptance_min_max=[float(a.min()), float(a.max())],
               eos_rate_per_step=float((w * p[:, EOS]).sum() / w.sum()),
               in_band=dict(entropy=0.8 <= ent <= 1.8, acceptance=0.5 <= acc_w <= 0.8))
    return t, d, rep


def save(model, name):
    path = fx.MODELS / name
    model.save_pretrained(path, safe_serialization=True)
    GenerationConfig(eos_token_id=EOS, pad_token_id=EOS, do_sample=False).save_pretrained(path)
    return path


if __name__ == "__main__":
    t, d, rep = calibrate()
    print(json.dumps(rep, indent=1))
    # fail closed: nothing is saved, copied or hashed unless the final fixture meets both §3 bands
    assert all(rep["in_band"].values()), f"F-B outside the §3 bands: {rep}"
    save(t, NAMES[0]), save(d, NAMES[1])
    out = os.environ.get("FB_REPORT", str(HERE / "phase0b" / "f2_fb_fixture.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
