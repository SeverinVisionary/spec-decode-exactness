"""FR22 E3: EAGLE-3 heads fitted to the fixture target by distillation (CPU), in `heads.py`'s checkpoint format.

The head's forward follows vLLM 0.30.0 and SGLang 0.5.21 `llama_eagle3.py` (identical math): h = fc(aux) (the
identity for one captured layer), e = embed(token) (the target's embedding, frozen); in the one decoder layer,
x = cat[input_layernorm(e), hidden_norm(h)] feeds q/k/v (input 2H), residual h; then post-attention norm, MLP,
residual; logits = lm_head(norm(out)); the pre-norm `out` is the hidden state of the next draft step.

Position semantics (EAGLE): the step at position p reads (embed(x_{p+1}), hidden_p) and predicts x_{p+2}. In a
drafting round from committed position p, step 0 reads the target's own hidden h_p and attends to the committed
positions <= p; step k >= 1 at position p + k reads the previous step's `out` and attends to the committed positions
<= p and the round's earlier steps. Training is teacher-forced on target-sampled sequences (accepted drafts follow
the target): loss = sum over steps k = 0..2 of KL(p_target(. | x_{<= p+k+1}) || q_k), on the S0 rows.

    python fit_heads.py FA|FB [--steps 3000]     # writes $D40_WORK/models/eagle3_{FA,FB}_fit and prints acceptance
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("D40_WORK", str(HERE.parent.parent / "data"))  # before fixture computes MODELS
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import fixture as fx  # noqa: E402
import heads  # noqa: E402

FIX = {"FA": dict(tdir="target", layers=2, V=8, EOS=7, max_new=4, aux=[0]),
       "FB": dict(tdir="fb_target", layers=4, V=32, EOS=31, max_new=6, aux=[0, 1, 2])}
K = 3
VARIANTS = {"fit": 1.0, "fitflat": 0.4}  # lm_head scale; flatter q lowers acceptance


class Target(torch.nn.Module):
    """fp32 Llama forward from the committed weights; returns logits and the outputs of the captured layers."""

    def __init__(self, fix):
        super().__init__()
        c = FIX[fix]
        self.w = {k: v.float() for k, v in load_file(str(fx.MODELS / c["tdir"] / "model.safetensors")).items()}
        self.L, self.H = c["layers"], 2
        self.aux = c["aux"]

    @staticmethod
    def rms(x, g, eps=1e-6):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g

    @staticmethod
    def rope(x, pos, theta=10000.0):
        d = x.shape[-1]
        inv = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        ang = pos[:, None].float() * inv[None]
        cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
        return x * cos + torch.cat([-x[..., d // 2:], x[..., : d // 2]], -1) * sin

    @torch.no_grad()
    def forward(self, ids):
        w, T = self.w, len(ids)
        x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        pos = torch.arange(T)
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        caps = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q, k, v = (h @ w[p + f"self_attn.{n}_proj.weight"].T for n in "qkv")
            d = q.shape[-1] // self.H
            q, k = (self.rope(t.view(T, self.H, d).transpose(0, 1), pos) for t in (q, k))
            v = v.view(T, self.H, d).transpose(0, 1)
            a = torch.softmax(q @ k.transpose(1, 2) / d ** 0.5 + mask, -1) @ v
            x = x + a.transpose(0, 1).reshape(T, -1) @ w[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g, u = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T
            if i in self.aux:
                caps.append(x)
        logits = self.rms(x, w["model.norm.weight"]) @ w["lm_head.weight"].T
        return logits, torch.cat(caps, -1)


class Head(torch.nn.Module):
    def __init__(self, init, embed):
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace(".", "__"): torch.nn.Parameter(v.float().clone())
                                         for k, v in init.items()})
        self.embed = embed
        self.nh, self.hd = heads.N_HEADS, heads.HEAD_DIM

    def __getitem__(self, k):
        return self.p[k.replace(".", "__")]

    def step(self, tok_ids, hid, pos, cache):
        """One draft step for a batch of positions (tensors of shape [B]); cache: list of (k, v, pos) entries the
        step may attend to (each [B, nh, hd]), extended in place with this step's entry. Returns logits, out."""
        m = "midlayer."
        e = Target.rms(self.embed[tok_ids], self[m + "input_layernorm.weight"])
        # fc only on the captured target features when their width differs (the engines' rule); for one captured
        # layer it never runs, so F-A's fc stays the identity heads.py wrote (vLLM applies it, SGLang skips it)
        h = hid @ self["fc.weight"].T if hid.shape[-1] != e.shape[-1] else hid
        x = torch.cat([e, Target.rms(h, self[m + "hidden_norm.weight"])], -1)
        B = x.shape[0]
        q, k, v = (x @ self[m + f"self_attn.{n}_proj.weight"].T for n in "qkv")
        q, k, v = (t.view(B, self.nh, self.hd) for t in (q, k, v))
        q = torch.stack([Target.rope(q[b], pos[b:b + 1].repeat(self.nh)) for b in range(B)])
        k = torch.stack([Target.rope(k[b], pos[b:b + 1].repeat(self.nh)) for b in range(B)])
        cache.append((k, v))
        ks = torch.stack([c[0] for c in cache], 2)  # B, nh, S, hd
        vs = torch.stack([c[1] for c in cache], 2)
        att = torch.softmax((q[:, :, None, :] * ks).sum(-1) / self.hd ** 0.5, -1)
        a = (att[..., None] * vs).sum(2).reshape(B, -1) @ self[m + "self_attn.o_proj.weight"].T
        x2 = a + h
        y = Target.rms(x2, self[m + "post_attention_layernorm.weight"])
        y = (torch.nn.functional.silu(y @ self[m + "mlp.gate_proj.weight"].T) * (y @ self[m + "mlp.up_proj.weight"].T)) \
            @ self[m + "mlp.down_proj.weight"].T
        out = y + x2
        return Target.rms(out, self["norm.weight"]) @ self["lm_head.weight"].T, out


def sample_sequences(tgt, fix, n, rng):
    c = FIX[fix]
    seqs = []
    for _ in range(n):
        g = []
        while len(g) < c["max_new"]:
            lg, _ = tgt(fx.PROMPT + g)
            t = int(rng.choice(c["V"], p=torch.softmax(lg[-1].double(), -1).numpy()))
            g.append(t)
            if t == c["EOS"]:
                break
        seqs.append(fx.PROMPT + g)
    return seqs


def rounds(tgt, seqs, fix):
    """Training examples: one per (sequence, committed position p) with p + 2 a generated position. Each holds the
    token ids, captured hidden states, target probability rows and the round's start p."""
    P = len(fx.PROMPT)
    ex = []
    for s in seqs:
        lg, cap = tgt(s)
        probs = torch.softmax(lg, -1)
        for p in range(P - 2, len(s) - 2):
            ex.append((s, cap, probs, p))
    return ex


def train(head, ex, steps, lr, rng):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for it in range(steps):
        batch = [ex[i] for i in rng.integers(len(ex), size=32)]
        loss = 0.0
        for s, cap, probs, p in batch:
            cache = []
            # committed entries 0..p-1 (teacher), then the round: step 0 at p (teacher hidden), steps 1.. own hidden
            for r in range(p):
                head.step(torch.tensor([s[r + 1]]), cap[r:r + 1], torch.tensor([r]), cache)
            hid = cap[p:p + 1]
            for k in range(K):
                if p + k + 1 >= len(s):
                    break
                logits, out = head.step(torch.tensor([s[p + k + 1]]), hid, torch.tensor([p + k]), cache)
                tgt_row = probs[p + k + 1]
                loss = loss + (tgt_row * (torch.log(tgt_row + 1e-12) - torch.log_softmax(logits[0], -1))).sum()
                hid = out
        opt.zero_grad()
        (loss / len(batch)).backward()
        opt.step()
        if it % 250 == 0:
            print(it, float(loss) / len(batch), flush=True)


@torch.no_grad()
def acceptance(head, tgt, fix, n, rng):
    """Mean per-token acceptance sum_x min(p, q) along target-sampled sequences, per draft step (S0)."""
    c = FIX[fix]
    seqs = sample_sequences(tgt, fix, n, rng)
    acc = [[] for _ in range(K)]
    for s in seqs:
        _, cap = tgt(s)
        for p in range(len(fx.PROMPT) - 2, len(s) - 2):
            cache = []
            for r in range(p):
                head.step(torch.tensor([s[r + 1]]), cap[r:r + 1], torch.tensor([r]), cache)
            hid = cap[p:p + 1]
            for k in range(K):
                if p + k + 1 >= len(s):
                    break
                logits, out = head.step(torch.tensor([s[p + k + 1]]), hid, torch.tensor([p + k]), cache)
                lg, _ = tgt(s[: p + k + 2])
                pt = torch.softmax(lg[-1], -1)
                acc[k].append(float(torch.minimum(pt, torch.softmax(logits[0], -1)).sum()))
                hid = out
    return [float(np.mean(a)) if a else None for a in acc]


def main(fix, steps, lr):
    rng = np.random.default_rng({"FA": 21, "FB": 22}[fix])
    torch.manual_seed({"FA": 21, "FB": 22}[fix])
    tgt = Target(fix)
    init_dir = heads.eagle3(fix, "tgt")  # random init in heads.py's format; only its weights are replaced
    init = load_file(str(init_dir / "model.safetensors"))
    head = Head(init, tgt.w["model.embed_tokens.weight"])
    print("acceptance before:", acceptance(head, tgt, fix, 100, np.random.default_rng(5)), flush=True)
    ex = rounds(tgt, sample_sequences(tgt, fix, 3000, rng), fix)
    train(head, ex, steps, lr, rng)
    report = {}
    lm = head["lm_head.weight"].detach().clone()
    for name, scale in VARIANTS.items():  # the GPU gate keeps the variant whose engine acceptance is in band (FR12)
        with torch.no_grad():
            head["lm_head.weight"].copy_(lm * scale)
        acc = acceptance(head, tgt, fix, 150, np.random.default_rng(6))
        chain = float(np.mean([np.prod(acc[:k + 1]) for k in range(K)]))
        report[name] = dict(lm_head_scale=scale, cpu_acceptance_per_step=acc, cpu_chain_rate_estimate=chain)
        print(name, report[name], flush=True)
        out = fx.MODELS / f"eagle3_{fix}_{name}"
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(init_dir, out)
        save_file({k: head[k].detach().contiguous().float() for k in init}, str(out / "model.safetensors"),
                  metadata={"format": "pt"})
    json.dump(dict(fixture=fix, steps=steps, lr=lr, variants=report), open(HERE / f"fit_{fix}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fix", choices=("FA", "FB"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    main(a.fix, a.steps, a.lr)
