"""Standalone reproduction: TensorRT-LLM 1.2.1 two-model draft-target speculative decoding with sampling does not
preserve the target's output distribution.

    python repro_draft_target.py --out ./repro_out            # one GPU; about 10 min after the install

Needs only tensorrt-llm==1.2.1 (with the torch it pins), numpy, scipy and safetensors. The script:
1. writes two tiny random Llama checkpoints (V = 8, 2 layers, head_dim 32) with fixed seeds: a target and a draft;
2. samples N sequences from the prompt [1, 2, 3, 1, 2] (4 new tokens, EOS = 7) at temperature 1 and at top-k 3,
   in three arms, each in its own process (2N draws for `base`):
   - `base`: no speculation;
   - `spec`: `DraftTargetDecodingConfig(max_draft_len=3)`;
   - `spec_keeplast`: as `spec`, with `ModelDrafter._create_draft_request` passing
     `exclude_last_generation_logits=False`;
3. in the speculative arms, wraps `TorchSampler._process_draft_tokens_rejection_sampling` to record, per
   verification, the number of draft tokens K, the number of `py_draft_logits` rows R, the number accepted, and how
   many accepted draft tokens have probability 0 in their own `py_target_probs` row;
4. compares each speculative arm's sequence distribution with `base` (total variation and a chi-square test),
   next to `base` split in two halves as the noise floor.

The only change to TensorRT-LLM besides the `spec_keeplast` patch is a workaround for a separate crash: in 1.2.1,
`ModelDrafter.process_decoded_tokens` calls `generation_logits.squeeze_(0)` on an inference tensor outside inference
mode, which kills the executor thread when sampling with VANILLA attention. The script runs that method under
`torch.inference_mode()`. Nothing else is patched; the run is bf16 with stock kernels.
"""
import argparse
import collections
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

V, EOS, MAX_NEW = 8, 7, 4
PROMPT = [1, 2, 3, 1, 2]
SETTINGS = {"T1": dict(temperature=1.0), "topk3": dict(temperature=1.0, top_k=3)}
N = 20_000
BATCH = 2_000
ARM_SEED = {"base": 1000, "spec": 2000, "spec_keeplast": 3000}  # TorchSampler seed, per arm and setting
FATAL = "[TRT-LLM] [E] Error in event loop"


def make_model(path, seed):
    """A tiny Llama with random weights; lm_head scaled up so the next-token distributions are not near uniform.
    Stored in bf16 with torch_dtype bfloat16: an fp32 checkpoint keeps fp32 activations even with LLM(dtype="bfloat16"),
    and FlashInfer's RMSNorm dispatches fp16/bf16 only (FR18 run 2)."""
    import numpy as np
    import torch
    from safetensors.torch import save_file
    rng = np.random.default_rng(seed)
    hidden, heads, head_dim, inter, layers = 64, 2, 32, 128, 2

    def w(*shape, std=0.3):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    sd = {"model.embed_tokens.weight": w(V, hidden, std=1.0), "model.norm.weight": np.ones(hidden, np.float32),
          "lm_head.weight": w(V, hidden, std=1.5)}
    for i in range(layers):
        p = f"model.layers.{i}."
        sd |= {p + "input_layernorm.weight": np.ones(hidden, np.float32),
               p + "post_attention_layernorm.weight": np.ones(hidden, np.float32),
               p + "self_attn.q_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.k_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.v_proj.weight": w(heads * head_dim, hidden),
               p + "self_attn.o_proj.weight": w(hidden, heads * head_dim),
               p + "mlp.gate_proj.weight": w(inter, hidden), p + "mlp.up_proj.weight": w(inter, hidden),
               p + "mlp.down_proj.weight": w(hidden, inter)}
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.from_numpy(v).to(torch.bfloat16) for k, v in sd.items()}, str(path / "model.safetensors"),
              metadata={"format": "pt"})
    cfg = dict(architectures=["LlamaForCausalLM"], model_type="llama", vocab_size=V, hidden_size=hidden,
               intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=heads,
               num_key_value_heads=heads, head_dim=head_dim, hidden_act="silu", max_position_embeddings=64,
               rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False, mlp_bias=False,
               tie_word_embeddings=False, bos_token_id=None, eos_token_id=EOS, pad_token_id=EOS,
               torch_dtype="bfloat16", dtype="bfloat16")
    (path / "config.json").write_text(json.dumps(cfg, indent=1))
    (path / "generation_config.json").write_text(json.dumps(dict(eos_token_id=EOS, pad_token_id=EOS)))


def workaround_squeeze():
    """Report B workaround: run `process_decoded_tokens` under inference mode (its `squeeze_(0)` is unchanged)."""
    import functools
    import torch
    from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter
    orig = ModelDrafter.process_decoded_tokens

    @functools.wraps(orig)
    def process_decoded_tokens(*a, **k):
        with torch.inference_mode():
            return orig(*a, **k)

    ModelDrafter.process_decoded_tokens = process_decoded_tokens


def keep_last():
    """`_create_draft_request` as in v1.2.1 (model_drafter.py 112-129) with exclude_last_generation_logits=False."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter

    def _create_draft_request(self, request, input_tokens):
        return LlmRequest(
            input_tokens=input_tokens, request_id=request.py_request_id, max_new_tokens=request.py_max_new_tokens,
            sampling_config=request.sampling_config, guided_decoding_params=request.guided_decoding_params,
            target_seq_slot=request.py_seq_slot, return_perf_metrics=request.return_perf_metrics, is_streaming=False,
            exclude_last_generation_logits=False,  # the only change
            is_draft=True, return_generation_logits=self.sampler.should_provide_draft_probs(request))

    ModelDrafter._create_draft_request = _create_draft_request


class Verifications:
    """Records K, R, accepted and accepted-with-zero-target-probability for every rejection-sampling call."""

    def __init__(self):
        from tensorrt_llm._torch.pyexecutor import sampler as S
        from tensorrt_llm._torch.pyexecutor.llm_request import get_draft_token_length
        self.rows = []
        orig = S.TorchSampler._process_draft_tokens_rejection_sampling
        rec = self

        def wrapped(self_, request, *a, **k):
            K = int(get_draft_token_length(request))
            draft = list(request.py_draft_tokens or [])[:K]
            dl, tp = request.py_draft_logits, request.py_target_probs
            R = 0 if dl is None else int(dl.reshape(-1, dl.shape[-1]).shape[0])
            tp = None if tp is None else tp.detach().float().reshape(-1, tp.shape[-1]).cpu()
            n = orig(self_, request, *a, **k)
            acc = min(int(n), K)
            zero = 0 if tp is None else sum(1 for i in range(min(acc, tp.shape[0])) if float(tp[i, draft[i]]) == 0.0)
            rec.rows.append((K, R, int(n), zero))
            return n

        S.TorchSampler._process_draft_tokens_rejection_sampling = wrapped


def run_arm(arm, out):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"  # keep the executor in this process, so the patches apply
    # FlashInfer JIT-builds its bf16 kernels with the `ninja` installed next to this interpreter (a venv's bin/),
    # which is not on PATH unless the venv is activated
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig
    torch.manual_seed(0)
    kw = dict(model=str(out / "target"), skip_tokenizer_init=True, dtype="bfloat16", attn_backend="VANILLA",
              cuda_graph_config=None, disable_overlap_scheduler=True, disable_flashinfer_sampling=True,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3), max_num_tokens=1024, max_batch_size=128)
    rec = None
    if arm != "base":
        if not os.environ.get("REPRO_NO_WORKAROUND"):  # set when testing a patched build that fixes the squeeze_
            workaround_squeeze()
        if arm == "spec_keeplast":
            keep_last()
        rec = Verifications()
        kw["speculative_config"] = DraftTargetDecodingConfig(max_draft_len=3, speculative_model=str(out / "draft"))
    m = LLM(**kw)
    res = dict(arm=arm, settings={})
    try:
        for name, sp in SETTINGS.items():
            if rec:
                rec.rows = []
            # TorchSampler draws from a private generator seeded lazily from `_global_seed` (default 42); give every
            # arm and setting its own seed so the compared samples are independent
            sampler = m._executor.engine.sampler
            sampler._generator, sampler._global_seed = None, ARM_SEED[arm] + list(SETTINGS).index(name)
            seqs = []
            for _ in range(0, 2 * N if arm == "base" else N, BATCH):
                params = SamplingParams(max_tokens=MAX_NEW, end_id=EOS, pad_id=EOS, **sp)
                o = m.generate([PROMPT] * BATCH, params)
                seqs += [list(x.outputs[0].token_ids) for x in o]
            kr = collections.Counter(f"K{k}/R{r}" for k, r, _, _ in rec.rows) if rec else {}
            res["settings"][name] = dict(
                seqs=seqs, KR=dict(kr), verification_rows=[list(x) for x in rec.rows] if rec else [],  # (K, R, n, zero)
                verifications=len(rec.rows) if rec else 0,
                # drafts accepted past the last draft-logits row, i.e. accepted without any check
                accepted_beyond_R=sum(max(0, min(n, k) - r) for k, r, n, _ in rec.rows) if rec else 0,
                verifications_accepting_beyond_R=sum(1 for k, r, n, _ in rec.rows if min(n, k) > r) if rec else 0,
                rows_fewer_than_K=sum(1 for k, r, _, _ in rec.rows if 0 < k and r < k) if rec else 0,
                accepted_with_zero_target_prob=sum(z for *_, z in rec.rows) if rec else 0,
                verifications_accepting_zero_prob=sum(1 for *_, z in rec.rows if z) if rec else 0)
            print(arm, name, "done", flush=True)
    finally:
        m.shutdown()
    import tensorrt_llm
    res["versions"] = dict(tensorrt_llm=tensorrt_llm.__version__, torch=torch.__version__,
                           gpu=torch.cuda.get_device_name(0))
    (out / f"{arm}.json").write_text(json.dumps(res))


def compare(a, b):
    import numpy as np
    from scipy.stats import chi2_contingency
    ca, cb = collections.Counter(map(tuple, a)), collections.Counter(map(tuple, b))
    keys = sorted(set(ca) | set(cb))
    ta, tb = np.array([ca[k] for k in keys], float), np.array([cb[k] for k in keys], float)
    tv = 0.5 * np.abs(ta / ta.sum() - tb / tb.sum()).sum()
    # pool sparse outcomes into one cell so every expected count is at least 5 in both rows (unequal sample sizes)
    need = 5 * (ta.sum() + tb.sum()) / min(ta.sum(), tb.sum())
    big = (ta + tb) >= need
    table = np.stack([np.append(ta[big], ta[~big].sum()), np.append(tb[big], tb[~big].sum())])
    if 0 < table[:, -1].sum() < need and table.shape[1] > 2:  # pooled cell still sparse: merge it into the smallest
        j = int(np.argmin(table[:, :-1].sum(0)))
        table[:, j] += table[:, -1]
        table = table[:, :-1]
    table = table[:, table.sum(0) > 0]
    chi2, p, dof, _ = chi2_contingency(table)
    return dict(tv=float(tv), chi2=float(chi2), dof=int(dof), p=float(p))


def summarize(out):
    arms = {a: json.loads((out / f"{a}.json").read_text()) for a in ("base", "spec", "spec_keeplast")
            if (out / f"{a}.json").exists()}
    lines, summary = [], {}
    if "base" in arms:
        lines.append(f"versions: {json.dumps(arms['base']['versions'])}")
    for name in SETTINGS:
        if "base" not in arms:
            break
        base = arms["base"]["settings"][name]["seqs"]
        h = len(base) // 2
        summary[name] = {"base_half_vs_half": compare(base[:h], base[h:])}
        lines.append(f"[{name}] base half vs half, {h} vs {len(base) - h} (noise floor; each speculative arm is "
                     f"{N} vs all {len(base)} base draws, so its noise is smaller): "
                     f"{json.dumps(summary[name]['base_half_vs_half'])}")
        for arm in ("spec", "spec_keeplast"):
            if arm not in arms:
                continue
            s = arms[arm]["settings"][name]
            summary[name][arm] = dict(vs_base=compare(s["seqs"], base),
                                      **{k: s[k] for k in ("KR", "verifications", "rows_fewer_than_K",
                                                           "accepted_beyond_R", "verifications_accepting_beyond_R",
                                                           "accepted_with_zero_target_prob",
                                                           "verifications_accepting_zero_prob")})
            lines.append(f"[{name}] {arm} vs base: {json.dumps(summary[name][arm]['vs_base'])}")
            lines.append(f"[{name}] {arm} verifications: {s['verifications']}, with fewer draft-logits rows than "
                         f"draft tokens: {s['rows_fewer_than_K']}, K/R: {json.dumps(s['KR'])}, drafts accepted with no "
                         f"draft-logits row: {s['accepted_beyond_R']} (in {s['verifications_accepting_beyond_R']} "
                         f"verifications), accepted draft "
                         f"tokens with target probability 0: {s['accepted_with_zero_target_prob']} "
                         f"(in {s['verifications_accepting_zero_prob']} verifications)")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["base", "spec", "spec_keeplast", "all"], default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.arm != "all":
        return run_arm(a.arm, out)
    make_model(out / "target", seed=1)
    make_model(out / "draft", seed=2)
    failed = []
    for arm in ("base", "spec", "spec_keeplast"):
        if (out / f"{arm}.json").exists():
            print(arm, "already done")
            continue
        t = time.time()
        log = out / f"{arm}.log"
        with open(log, "w") as f:
            # a dead executor thread leaves generate() hanging: kill the arm 60 s after the fatal log line
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--out", str(out), "--arm", arm],
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            fatal_at, rc = None, None
            while p.poll() is None:
                time.sleep(10)
                if fatal_at is None and FATAL in open(log, errors="replace").read():
                    fatal_at = time.time()
                if (fatal_at and time.time() - fatal_at > 60) or time.time() - t > 3600:
                    rc = "fatal" if fatal_at else "timeout"
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                    break
            rc = rc or p.returncode
        print(arm, "exit", rc, f"{time.time() - t:.0f}s", flush=True)
        if rc != 0 or not (out / f"{arm}.json").exists():
            failed.append(arm)
            print(open(out / f"{arm}.log", errors="replace").read()[-3000:], flush=True)
    summarize(out)
    print("failed:", failed or "none", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
