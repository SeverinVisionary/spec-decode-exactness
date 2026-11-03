"""Function-preserving re-layout of the fixture for vLLM's CPU backend (model surgery, §7 F3; R8).

vLLM v0.30.0 on CPU always selects the CPU_ATTN backend, whose kernels support head sizes
{32, 64, 80, ...} (v1/attention/backends/cpu_attn.py L64); the frozen fixture has head_dim 8.
The re-layout below computes the same function with head_dim 32:

- each head's 8 q/k/v rows are placed in the first 8 of 32 rows, the other 24 rows are zero;
- o_proj columns are mapped the same way (zero columns for the padding);
- RoPE: `partial_rotary_factor = 0.25` rotates only the first 8 dims of each head with
  inv_freq = theta^(-2i/8), i < 4, neox pairing (i, i+4): exactly the original rotation;
- the softmax scale is head_dim^-0.5 = 1/sqrt(32) instead of 1/sqrt(8), so q rows are scaled by
  sqrt(32/8) = 2 (a power of two, exact in fp32).

Every other tensor is copied byte for byte. The output is checked numerically: vLLM's R_eng
against R_np (built from the *original* weights) must meet the same δ_num bound as HF.
"""
import json
import shutil

import numpy as np
from safetensors.numpy import load_file, save_file

import fixture as fx

PAD = 32


def export(name):
    src, dst = fx.MODELS / name, fx.MODELS / f"{name}_vllm"
    cfg = json.load(open(src / "config.json"))
    H, D = cfg["num_attention_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
    assert cfg["num_key_value_heads"] == H and D == 8
    sd = load_file(src / "model.safetensors")
    out = {}
    for k, w in sd.items():
        if any(k.endswith(f"self_attn.{p}_proj.weight") for p in "qkv"):
            w3 = w.reshape(H, D, -1)
            big = np.zeros((H, PAD, w.shape[1]), w.dtype)
            big[:, :D] = w3 * (np.float32(2.0) if ".q_proj." in k else np.float32(1.0))
            out[k] = big.reshape(H * PAD, -1)
        elif k.endswith("self_attn.o_proj.weight"):
            w3 = w.reshape(w.shape[0], H, D)
            big = np.zeros((w.shape[0], H, PAD), w.dtype)
            big[:, :, :D] = w3
            out[k] = big.reshape(w.shape[0], H * PAD)
        else:
            out[k] = w
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, dst / "model.safetensors", metadata={"format": "pt"})
    rope = dict(cfg.get("rope_parameters") or {"rope_type": "default", "rope_theta": cfg.get("rope_theta", 10000.0)})
    rope["partial_rotary_factor"] = D / PAD
    cfg.update(head_dim=PAD, rope_parameters=rope, partial_rotary_factor=D / PAD)
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    shutil.copy(src / "generation_config.json", dst / "generation_config.json")
    return dst


if __name__ == "__main__":
    for n in ("target", "draft"):
        print(export(n))
