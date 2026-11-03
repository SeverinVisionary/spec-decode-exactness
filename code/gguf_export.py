"""Export the HF fixture to F32 GGUF for llama.cpp (model surgery only; §7 F3).

Mirrors llama.cpp conversion/llama.py: Q/K rows permuted for llama.cpp's RoPE layout.
Vocabulary is a stub: 8 single-character tokens, no merges; prompts are passed as token ids."""
import string
import sys

import numpy as np
from safetensors.numpy import load_file

import fixture as fx

sys.path.insert(0, str(fx.WORK / "llama.cpp/gguf-py"))
import gguf  # noqa: E402


def permute(w, n_head):
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def export(name, cfg=None, vocab=None, eos=None):
    """Defaults reproduce the pilot (F-A). The full study passes F-B's config, V and EOS (Phase 0b F2)."""
    cfg, vocab, eos = cfg or fx.config(), vocab or fx.V, fx.EOS if eos is None else eos
    sd = load_file(fx.MODELS / name / "model.safetensors")
    out = fx.MODELS / f"{name}.gguf"
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name(f"d40-{name}")
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_block_count(cfg.num_hidden_layers)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_rope_dimension_count(cfg.hidden_size // cfg.num_attention_heads)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_vocab_size(vocab)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    # single ASCII letters: a-h for the pilot's V = 8, a-z then A-F for F-B's V = 32
    w.add_token_list([(string.ascii_lowercase + string.ascii_uppercase)[i] for i in range(vocab)])
    w.add_token_types([gguf.TokenType.NORMAL] * vocab)
    w.add_token_merges(["a b"])  # dummy; text tokenization is never used
    w.add_eos_token_id(eos)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    nh = cfg.num_attention_heads
    w.add_tensor("token_embd.weight", sd["model.embed_tokens.weight"])
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        w.add_tensor(f"blk.{i}.attn_norm.weight", sd[p + "input_layernorm.weight"])
        w.add_tensor(f"blk.{i}.attn_q.weight", permute(sd[p + "self_attn.q_proj.weight"], nh))
        w.add_tensor(f"blk.{i}.attn_k.weight", permute(sd[p + "self_attn.k_proj.weight"], nh))
        w.add_tensor(f"blk.{i}.attn_v.weight", sd[p + "self_attn.v_proj.weight"])
        w.add_tensor(f"blk.{i}.attn_output.weight", sd[p + "self_attn.o_proj.weight"])
        w.add_tensor(f"blk.{i}.ffn_norm.weight", sd[p + "post_attention_layernorm.weight"])
        w.add_tensor(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"])
        w.add_tensor(f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"])
        w.add_tensor(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"])
    w.add_tensor("output_norm.weight", sd["model.norm.weight"])
    w.add_tensor("output.weight", sd["lm_head.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


if __name__ == "__main__":
    for n in ("target", "draft"):
        print(export(n))
    assert np.float32  # keep numpy import explicit for safetensors.numpy
