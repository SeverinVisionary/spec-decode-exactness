"""Standalone reproduction ported to TensorRT-LLM v1.4.0rc1: tests whether one-model draft-target speculative
decoding (`DraftTargetDecodingConfig` -> `DraftTargetOneModelWorker`) with sampling preserves the target's output
distribution.

    python repro_v14.py --out ./repro_out_v14        # one GPU; about 10 min after the install

Needs tensorrt-llm==1.4.0rc1 (with the torch/flashinfer/transformers it pins), numpy, scipy and safetensors.
The script:
1. writes two tiny random Llama checkpoints (V = 8, 2 layers, head_dim 32) with fixed seeds: a target and a draft;
2. samples N sequences from the prompt [1, 2, 3, 1, 2] (4 new tokens, EOS = 7) at temperature 1 and at top-k 3,
   in three arms, each in its own process (2N draws for `base`):
   - `base`: no speculation;
   - `spec`: `DraftTargetDecodingConfig(max_draft_len=3, speculative_model=...)` (default strict acceptance);
   - `spec_rs`: as `spec` with `use_rejection_sampling=True`;
3. in the speculative arm, wraps `SpecWorkerBase._accept_draft_tokens` (failing soft with a warning if the hook is
   absent) to record, per generation-request verification, the number of draft tokens K, the number of `draft_probs`
   rows R (-1 when `use_rejection_sampling=False` and no draft probs exist), the number of accepted draft tokens,
   and how many accepted draft tokens have probability 0 in their own filtered target probability row;
4. compares the speculative arm's sequence distribution with `base` (total variation and a chi-square test),
   next to `base` split in two halves as the noise floor.
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
ARM_SEED = {"base": 1000, "spec": 2000, "spec_rs": 3000}  # per-arm and per-setting RNG seed
ARMS = ("base", "spec", "spec_rs")  # spec_rs: use_rejection_sampling=True
FATAL = "[TRT-LLM] [E] Error in event loop"


def make_model(path, seed):
    """A tiny Llama with random weights; lm_head scaled up so the next-token distributions are not near uniform.
    Stored in bf16 with both `torch_dtype` and `dtype` set to bfloat16 in config.json: unchanged from v1.2.1 because
    writing model.safetensors + config.json directly needs no transformers==5.5.4 API changes."""
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


class Verifications:
    """Records (K, R_or_-1, n_accepted, zero_target) per generation request in SpecWorkerBase._accept_draft_tokens.
    Fails soft with a warning if the hook target is missing or changes signature."""

    def __init__(self):
        self.rows = []
        self.calib, self.calib_missing = [], 0
        self.active = False
        try:
            from tensorrt_llm._torch.speculative import interface as spec_interface
            worker_cls = getattr(spec_interface, "SpecWorkerBase", None)
            orig = getattr(worker_cls, "_accept_draft_tokens", None) if worker_cls is not None else None
            if orig is None:
                print("WARNING: SpecWorkerBase._accept_draft_tokens not found; verification recorder disabled",
                      flush=True)
                return
        except Exception as exc:
            print(f"WARNING: could not import SpecWorkerBase._accept_draft_tokens ({exc}); "
                  f"verification recorder disabled", flush=True)
            return

        rec = self

        def wrapped(self_, logits, draft_tokens, num_contexts, batch_size, spec_metadata, *a, **k):
            out = orig(self_, logits, draft_tokens, num_contexts, batch_size, spec_metadata, *a, **k)
            try:
                accepted_tokens, num_accepted_tokens = out
                num_gens = int(batch_size) - int(num_contexts)
                if num_gens <= 0 or draft_tokens is None or draft_tokens.dim() < 2 or draft_tokens.shape[1] <= 0:
                    return out
                K = int(draft_tokens.shape[1])

                # Determine R (number of draft-prob rows per request, or -1 when strict acceptance uses no draft probs)
                R = -1
                if self_._can_use_rejection_sampling(spec_metadata):
                    stored_vocab = (spec_metadata.draft_probs_last_dim
                                    if spec_metadata.draft_probs_last_dim > 0
                                    else spec_metadata.draft_probs_vocab_size)
                    if self_._rejection_buffers_valid(draft_tokens, K, stored_vocab, num_contexts, batch_size,
                                                      logits, spec_metadata):
                        gen_slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
                        dp = spec_metadata.draft_probs[gen_slot_ids, :K, :stored_vocab]
                        R = int(dp.shape[1])

                # Compute filtered target probabilities for the generation slice using the worker's own kernel
                tp = None
                try:
                    penalized = self_._apply_occurrence_penalties(logits, draft_tokens, num_contexts,
                                                                  batch_size, spec_metadata)
                    num_gen_logits = num_gens * (K + 1)
                    gen_start = int(num_contexts)
                    gen_end = gen_start + num_gen_logits
                    gen_logits = penalized[gen_start:gen_end]
                    if (not getattr(spec_metadata, "is_all_greedy_sample", True)
                            and getattr(spec_metadata, "temperatures", None) is not None):
                        min_ps = (spec_metadata.min_ps[gen_start:gen_end]
                                  if getattr(spec_metadata, "min_ps", None) is not None else None)
                        tp_flat = self_._compute_probs_from_logits(
                            spec_metadata.advanced_sampling_mode,
                            gen_logits,
                            spec_metadata.temperatures[gen_start:gen_end],
                            spec_metadata.top_ks[gen_start:gen_end],
                            spec_metadata.top_ps[gen_start:gen_end],
                            min_ps,
                        )
                        tp = tp_flat.detach().float().reshape(num_gens, K + 1, -1).cpu()
                except Exception:
                    tp = None

                draft_cpu = draft_tokens.detach().cpu()
                # num_accepted_tokens includes the 1 bonus/resampled token; subtract 1 for accepted draft tokens
                n_emitted_cpu = num_accepted_tokens[num_contexts:batch_size].detach().cpu()
                for g in range(num_gens):
                    n_acc = max(0, int(n_emitted_cpu[g].item()) - 1)
                    acc = min(n_acc, K)
                    zero = (0 if tp is None else
                            sum(1 for i in range(min(acc, tp.shape[1]))
                                if float(tp[g, i, int(draft_cpu[g, i].item())]) == 0.0))
                    rec.rows.append((K, R, n_acc, zero))
                    # FR21 calibration: the emitted tokens and the engine's own target-probability row at each
                    # emitted position (position j is emitted iff the first j draft tokens were accepted)
                    n_emit = min(int(n_emitted_cpu[g].item()), K + 1)
                    if tp is None:
                        rec.calib_missing += 1
                    else:
                        emitted = accepted_tokens[num_contexts + g, :n_emit].detach().cpu().tolist()
                        rec.calib.append((emitted, [[round(float(x), 6) for x in tp[g, j]] for j in range(n_emit)]))
            except Exception as exc:
                if rec.active:
                    print(f"WARNING: verification recorder step failed ({exc}); disabling recorder", flush=True)
                    rec.active = False
            return out

        worker_cls._accept_draft_tokens = wrapped
        self.active = True


def _set_arm_rng_seed(m, arm_seed):
    """Give each arm and setting its own Philox seed in v1.4.0rc1.
    - Base (`TorchSampler`) reads `sampler._seed_manager._global_seed` (seed_manager.py:55-127).
    - One-model spec (`SpecMetadata`) reads `interface.DEFAULT_SAMPLING_SEED` (interface.py:444, 838-842)."""
    sampler = getattr( getattr(m, "_executor", None), "engine", None)
    sampler = getattr(sampler, "sampler", None)
    if sampler is not None:
        sampler._generator, sampler._global_seed = None, arm_seed
        sm = getattr(sampler, "_seed_manager", None)
        if sm is not None:
            sm._global_seed = arm_seed
            if hasattr(sm, "_seeds") and sm._seeds is not None:
                sm._seeds.fill_(arm_seed)
            if hasattr(sm, "_slot_owner"):
                sm._slot_owner = [None] * len(sm._slot_owner)
    try:
        from tensorrt_llm._torch.speculative import interface as spec_interface
        spec_interface.DEFAULT_SAMPLING_SEED = arm_seed
    except Exception:
        pass


def run_arm(arm, out):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"  # keep the executor in this process, so the patches apply
    # FlashInfer JIT-builds its bf16 kernels with the `ninja` installed next to this interpreter (a venv's bin/),
    # which is not on PATH unless the venv is activated
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig
    torch.manual_seed(0)
    # Note on `disable_flashinfer_sampling`: removed in v1.4.0rc1 (`TorchLlmArgs` sets `extra="forbid"` at
    # llm_args.py:5559 and `BaseLLM.__init__` raises `ValueError: LLM got invalid argument: disable_flashinfer_sampling`
    # at llm.py:406-416). In v1.4.0rc1, both `TorchSampler` and `SpecWorkerBase` (whose default
    # `advanced_sampling_mode` is `AdvancedSamplingMode.FULL`, llm_args.py:2369-2375) use TRT-LLM's own fused CUDA
    # sampling kernel (`torch.ops.trtllm.fused_sample_from_logits*`) rather than FlashInfer's sampling ops by default.
    kw = dict(model=str(out / "target"), skip_tokenizer_init=True, dtype="bfloat16", attn_backend=os.environ.get("REPRO_ATTN", "VANILLA"),  # FR20 run 2: TRTLLM
              cuda_graph_config=None, disable_overlap_scheduler=True,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3), max_num_tokens=1024, max_batch_size=128)
    rec = None
    if arm != "base":
        rec = Verifications()
        use_rej = arm == "spec_rs"  # default strict acceptance vs opt-in chain speculative sampling
        kw["speculative_config"] = DraftTargetDecodingConfig(
            max_draft_len=3, speculative_model=str(out / "draft"), use_rejection_sampling=use_rej
        )
    m = LLM(**kw)
    res = dict(arm=arm, settings={})
    try:
        for name, sp in SETTINGS.items():
            if rec:
                rec.rows, rec.calib, rec.calib_missing = [], [], 0
            _set_arm_rng_seed(m, ARM_SEED[arm] + list(SETTINGS).index(name))
            seqs = []
            for _ in range(0, 2 * N if arm == "base" else N, BATCH):
                params = SamplingParams(max_tokens=MAX_NEW, end_id=EOS, pad_id=EOS, **sp)
                o = m.generate([PROMPT] * BATCH, params)
                seqs += [list(x.outputs[0].token_ids) for x in o]
            kr = collections.Counter(f"K{k}/R{r}" for k, r, _, _ in rec.rows) if rec else {}
            # In v1.4.0rc1's default strict-acceptance mode (use_rejection_sampling=False), no draft_probs tensor
            # exists, so the recorder logs R = -1. Skip R == -1 rows when counting draft-prob row shortfalls
            # (rows_fewer_than_K, accepted_beyond_R, verifications_accepting_beyond_R).
            res["settings"][name] = dict(
                seqs=seqs, KR=dict(kr), verification_rows=[list(x) for x in rec.rows] if rec else [],  # (K, R, n, zero)
                calibration=rec.calib if rec else [], calibration_missing=rec.calib_missing if rec else 0,
                verifications=len(rec.rows) if rec else 0,
                accepted_beyond_R=sum(max(0, min(n, k) - r) for k, r, n, _ in rec.rows if r != -1) if rec else 0,
                verifications_accepting_beyond_R=sum(
                    1 for k, r, n, _ in rec.rows if r != -1 and min(n, k) > r
                ) if rec else 0,
                rows_fewer_than_K=sum(1 for k, r, _, _ in rec.rows if r != -1 and 0 < k and r < k) if rec else 0,
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
    arms = {a: json.loads((out / f"{a}.json").read_text()) for a in ARMS
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
        for arm in ARMS[1:]:
            if arm not in arms:
                continue
            s = arms[arm]["settings"][name]
            rows = s.get("verification_rows", [])
            # Recompute skipping R == -1 rows in case summarize() is run on raw verification_rows
            rows_fewer_than_K = (sum(1 for k, r, _, _ in rows if r != -1 and 0 < k and r < k)
                                 if rows else s["rows_fewer_than_K"])
            accepted_beyond_R = (sum(max(0, min(n, k) - r) for k, r, n, _ in rows if r != -1)
                                 if rows else s["accepted_beyond_R"])
            verifications_accepting_beyond_R = (sum(1 for k, r, n, _ in rows if r != -1 and min(n, k) > r)
                                                if rows else s["verifications_accepting_beyond_R"])
            summary[name][arm] = dict(
                vs_base=compare(s["seqs"], base),
                KR=s["KR"],
                verifications=s["verifications"],
                rows_fewer_than_K=rows_fewer_than_K,
                accepted_beyond_R=accepted_beyond_R,
                verifications_accepting_beyond_R=verifications_accepting_beyond_R,
                accepted_with_zero_target_prob=s["accepted_with_zero_target_prob"],
                verifications_accepting_zero_prob=s["verifications_accepting_zero_prob"],
            )
            lines.append(f"[{name}] {arm} vs base: {json.dumps(summary[name][arm]['vs_base'])}")
            lines.append(f"[{name}] {arm} verifications: {s['verifications']}, with fewer draft-logits rows than "
                         f"draft tokens: {rows_fewer_than_K}, K/R: {json.dumps(s['KR'])}, drafts accepted with no "
                         f"draft-logits row: {accepted_beyond_R} (in {verifications_accepting_beyond_R} "
                         f"verifications), accepted draft "
                         f"tokens with target probability 0: {s['accepted_with_zero_target_prob']} "
                         f"(in {s['verifications_accepting_zero_prob']} verifications)")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.arm != "all":
        return run_arm(a.arm, out)
    make_model(out / "target", seed=1)
    make_model(out / "draft", seed=2)
    failed = []
    # FR21 runs only the speculative arms: REPRO_ARMS=spec,spec_rs
    for arm in [x for x in os.environ.get("REPRO_ARMS", ",".join(ARMS)).split(",") if x]:
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
