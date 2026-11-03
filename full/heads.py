"""D-40 full study, Phase 0b F2: drafter heads (PREREGISTRATION.md §3, FR12), built on CPU.

A head must be lossless, not good: random init, only model surgery, no engine change.

EAGLE-3 head (VG4 vLLM, SG1/SG2 SGLang, TG2 TensorRT-LLM), one checkpoint per fixture and variant, in the
format all three loaders accept (source notes in FR12):
- `config.json`: `architectures: ["LlamaForCausalLMEagle3"]`, `model_type: llama`, one decoder layer,
  `hidden_size` = the target's (the target embedding is shared), `head_dim` 32 (every attention kernel in
  use accepts it), `draft_vocab_size` = V (no `d2t`: TensorRT-LLM refuses non-greedy drafting with one);
- auxiliary target layers, one per fixture, named per engine because the engines number them differently:
  vLLM reads top-level `eagle_aux_hidden_state_layer_ids` (k = residual stream after k layers), SGLang reads
  `eagle_config.eagle_aux_hidden_state_layer_ids` (i = output of layer i, at most L - 2), TensorRT-LLM takes
  `eagle3_layers_to_capture` from the API (`p0b_trtllm.EAGLE3_CAPTURE`). F-A (2 layers) captures the output
  of layer 0 only (SGLang cannot capture the last layer of a 2-layer target); F-B (4 layers) captures the
  outputs of layers 0, 1 and 2;
- unprefixed weight names (`midlayer.*`, `fc.weight`, `norm.weight`, `lm_head.weight`); no `embed_tokens`.

Variants: acceptance depends on the engine's own drafting, which cannot be reproduced on CPU, so each
fixture gets two variants, `tgt` (lm_head copied from the target) and `flat` (that copy times 0.1: a near
uniform draft). Phase 0b measures acceptance at S0 from the spec counters (counters only, no sample is
compared with a reference) and the post-Phase-0b revision keeps the variant inside [0.3, 0.9] (§3).

MTP (VG6, vLLM): no Llama MTP exists in vLLM 0.30.0; the simplest MTP target is MiMo (Qwen2 plus one MTP
layer). `mimo_FA` is F-A's head_dim-32 re-layout widened to hidden 64 (vLLM's Qwen2 derives head_dim from
hidden / heads) as a `MiMoForCausalLM` with all-zero q/k/v biases, so it computes exactly F-A's function
(checked in fp64 by `check_mimo`) and F-A's R_np applies unchanged; only the MTP layer (random init) is new.

    python heads.py          # writes $D40_WORK/models/{eagle3_FA_tgt, eagle3_FA_flat, eagle3_FB_tgt, eagle3_FB_flat, mimo_FA}
"""
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import numpy as np  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402

import fixture as fx  # noqa: E402

HEAD_DIM, N_HEADS = 32, 2
SEED = {"FA": 11, "FB": 12, "mimo": 13}
AUX = {"FA": dict(vllm=[1], sglang=[0], trtllm=[0]), "FB": dict(vllm=[1, 2, 3], sglang=[0, 1, 2], trtllm=[0, 1, 2])}
TARGET = {"FA": ("target", 16, 8, 32), "FB": ("fb_target", 64, 32, 128)}  # dir, H, V, intermediate
VARIANT_SCALE = {"tgt": 1.0, "flat": 0.1}


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def eagle3(fix, variant):
    tdir, H, V, inter = TARGET[fix]
    tw = load_file(fx.MODELS / tdir / "model.safetensors")
    rng = np.random.default_rng(SEED[fix])
    std = 0.02
    nd = N_HEADS * HEAD_DIM
    A = len(AUX[fix]["vllm"])
    w = {
        "midlayer.input_layernorm.weight": f32(np.ones(H)),
        "midlayer.hidden_norm.weight": f32(np.ones(H)),
        "midlayer.post_attention_layernorm.weight": f32(np.ones(H)),
        "midlayer.self_attn.q_proj.weight": f32(rng.normal(0, std, (nd, 2 * H))),
        "midlayer.self_attn.k_proj.weight": f32(rng.normal(0, std, (nd, 2 * H))),
        "midlayer.self_attn.v_proj.weight": f32(rng.normal(0, std, (nd, 2 * H))),
        "midlayer.self_attn.o_proj.weight": f32(rng.normal(0, std, (H, nd))),
        "midlayer.mlp.gate_proj.weight": f32(rng.normal(0, std, (inter, H))),
        "midlayer.mlp.up_proj.weight": f32(rng.normal(0, std, (inter, H))),
        "midlayer.mlp.down_proj.weight": f32(rng.normal(0, std, (H, inter))),
        "norm.weight": f32(np.ones(H)),
        # A = 1: exactly the identity, so the three engines run the same head (TensorRT-LLM builds no fc for one
        # captured layer and SGLang skips fc when the widths match; only vLLM applies it)
        "fc.weight": f32(np.eye(H) if A == 1 else np.concatenate([np.eye(H)] * A, axis=1) / A + rng.normal(0, std, (H, A * H))),
        "lm_head.weight": f32(tw["lm_head.weight"] * VARIANT_SCALE[variant]),
    }
    cfg = dict(architectures=["LlamaForCausalLMEagle3"], model_type="llama", hidden_size=H, intermediate_size=inter,
               num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM, num_hidden_layers=1,
               vocab_size=V, draft_vocab_size=V, target_hidden_size=H, num_aux_hidden_states=A,
               eagle_aux_hidden_state_layer_ids=AUX[fix]["vllm"],
               eagle_config=dict(eagle_aux_hidden_state_layer_ids=AUX[fix]["sglang"], use_aux_hidden_state=True),
               hidden_act="silu", rms_norm_eps=1e-6, max_position_embeddings=64, rope_theta=10000.0,
               rope_parameters=dict(rope_type="default", rope_theta=10000.0), attention_bias=False, mlp_bias=False,
               tie_word_embeddings=False, torch_dtype="float32", dtype="float32", bos_token_id=None,
               eos_token_id=V - 1, pad_token_id=V - 1)
    out = fx.MODELS / f"eagle3_{fix}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, out / "model.safetensors", metadata={"format": "pt"})
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    return out


def mimo():
    """F-A as MiMo for VG6, function-preserving (FR12). vLLM's Qwen2 attention ignores `head_dim` and uses
    hidden_size / num_heads (`models/qwen2.py:150`), so F-A's head_dim-32 re-layout (`code/vllm_export.py`) is
    widened from hidden 16 to 64 with zero padding: every padded residual coordinate stays exactly 0, the
    RMSNorm mean over 64 coordinates is a quarter of the mean over 16, and halving every norm weight with
    eps / 4 gives the same normalised values (both factors are powers of two, exact in fp32). q/k/v biases
    are zero. Only the MTP layer (random init) is new."""
    src = fx.MODELS / "target_vllm"
    cfg = json.load(open(src / "config.json"))
    sd = load_file(src / "model.safetensors")
    H0, H = cfg["hidden_size"], 64
    nh, hd, inter = cfg["num_attention_heads"], cfg["head_dim"], cfg["intermediate_size"]
    assert H // nh == hd, (H, nh, hd)
    nd = nh * hd

    def pad_cols(w):  # input side: hidden H0 -> H
        out = np.zeros((w.shape[0], H), np.float32)
        out[:, :H0] = w
        return out

    def pad_rows(w):  # output side: hidden H0 -> H
        out = np.zeros((H, w.shape[1]), np.float32)
        out[:H0] = w
        return out

    def norm(g):
        out = np.zeros(H, np.float32)
        out[:H0] = g * np.float32(0.5)
        return out

    w = {}
    for k, v in sd.items():
        if k.endswith("norm.weight"):
            w[k] = norm(v)
        elif k in ("model.embed_tokens.weight", "lm_head.weight") or any(k.endswith(f"{p}_proj.weight") for p in ("q", "k", "v", "gate", "up")):
            w[k] = pad_cols(v)
        elif k.endswith("o_proj.weight") or k.endswith("down_proj.weight"):
            w[k] = pad_rows(v)
        else:
            raise KeyError(k)
    for i in range(cfg["num_hidden_layers"]):
        for p in "qkv":
            w[f"model.layers.{i}.self_attn.{p}_proj.bias"] = f32(np.zeros(nd))
    rng = np.random.default_rng(SEED["mimo"])
    std = 0.02
    m = "model.mtp_layers.0."
    w.update({
        m + "token_layernorm.weight": f32(np.ones(H)), m + "hidden_layernorm.weight": f32(np.ones(H)),
        m + "input_proj.weight": f32(rng.normal(0, std, (H, 2 * H))),
        m + "input_layernorm.weight": f32(np.ones(H)), m + "post_attention_layernorm.weight": f32(np.ones(H)),
        m + "final_layernorm.weight": f32(np.ones(H)),
        m + "self_attn.q_proj.weight": f32(rng.normal(0, std, (nd, H))), m + "self_attn.q_proj.bias": f32(np.zeros(nd)),
        m + "self_attn.k_proj.weight": f32(rng.normal(0, std, (nd, H))), m + "self_attn.k_proj.bias": f32(np.zeros(nd)),
        m + "self_attn.v_proj.weight": f32(rng.normal(0, std, (nd, H))), m + "self_attn.v_proj.bias": f32(np.zeros(nd)),
        m + "self_attn.o_proj.weight": f32(rng.normal(0, std, (H, nd))),
        m + "mlp.gate_proj.weight": f32(rng.normal(0, std, (inter, H))),
        m + "mlp.up_proj.weight": f32(rng.normal(0, std, (inter, H))),
        m + "mlp.down_proj.weight": f32(rng.normal(0, std, (H, inter))),
    })
    cfg.update(architectures=["MiMoForCausalLM"], model_type="qwen2", hidden_size=H, rms_norm_eps=cfg["rms_norm_eps"] / 4,
               num_nextn_predict_layers=1, use_sliding_window=False, sliding_window=None,
               max_window_layers=cfg["num_hidden_layers"])
    out = fx.MODELS / "mimo_FA"
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, out / "model.safetensors", metadata={"format": "pt"})
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    shutil.copy(src / "generation_config.json", out / "generation_config.json")
    return out


def check_mimo():
    """CPU check that mimo_FA (as a plain Qwen2 forward in fp64, MTP layer unused) equals F-A's R_np logits."""
    import rnp
    ref = rnp.Llama64("target")
    m = Qwen64("mimo_FA")
    worst = 0.0
    for g in fx.nonterminal_prefixes():
        ids = fx.PROMPT + g
        worst = max(worst, float(np.abs(ref.logits(ids) - m.logits(ids)).max()))
    return worst


class Qwen64:
    """Minimal fp64 Qwen2 forward (Llama plus q/k/v bias; partial rotary as in the re-layout), for check_mimo only."""

    def __init__(self, name):
        self.cfg = json.load(open(fx.MODELS / name / "config.json"))
        self.w = {k: v.astype(np.float64) for k, v in load_file(fx.MODELS / name / "model.safetensors").items()}

    def logits(self, ids):
        c, w = self.cfg, self.w
        H, nh = c["hidden_size"], c["num_attention_heads"]
        d = H // nh
        rd = int(d * c["rope_parameters"].get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        eps, T = c["rms_norm_eps"], len(ids)
        rms = lambda x, g: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g  # noqa: E731
        inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rd, 2) / rd)
        ang = np.outer(np.arange(T), inv)
        cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))

        def rope(x):
            r, rest = x[..., :rd], x[..., rd:]
            x1, x2 = r[..., : rd // 2], r[..., rd // 2:]
            return np.concatenate([r * cos + np.concatenate([-x2, x1], -1) * sin, rest], -1)

        x = w["model.embed_tokens.weight"][ids]
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}."
            h = rms(x, w[p + "input_layernorm.weight"])
            q, k, v = (h @ w[p + f"self_attn.{n}_proj.weight"].T + w[p + f"self_attn.{n}_proj.bias"] for n in "qkv")
            q, k, v = (t.reshape(T, nh, d).transpose(1, 0, 2) for t in (q, k, v))
            a = rope(q) @ rope(k).transpose(0, 2, 1) / np.sqrt(d) + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            x = x + (a @ v).transpose(1, 0, 2).reshape(T, -1) @ w[p + "self_attn.o_proj.weight"].T
            h = rms(x, w[p + "post_attention_layernorm.weight"])
            gt, up = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + ((gt / (1 + np.exp(-gt))) * up) @ w[p + "mlp.down_proj.weight"].T
        return rms(x, w["model.norm.weight"])[-1] @ w["lm_head.weight"].T


if __name__ == "__main__":
    for f in ("FA", "FB"):
        for v in VARIANT_SCALE:
            print(eagle3(f, v))
    if os.path.exists(fx.MODELS / "target_vllm"):
        print(mimo())
        worst = check_mimo()
        print("mimo_FA vs F-A R_np, max |logit diff| over the 400 prefixes:", worst)
        assert worst < 1e-9, worst
    else:
        print("mimo_FA skipped: run code/vllm_export.py first (the notebooks do)")
