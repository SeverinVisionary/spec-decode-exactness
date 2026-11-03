"""Phase 0b probes for TensorRT-LLM 1.2.1 on an L4 (FR8, FR9 items 1/3/7, FR11, FR12).

Engine options are Phase 0's (FR7, FR8): VANILLA attention, TP1 worker in process, CUDA graphs and the
overlap scheduler off, FlashInfer sampling off, fp32 hook (`d40_trtllm_fp32.py`, with FR11's drafter hook).
TG1t (TRTLLM attention) is BLOCKED (FR11) and is not run.

- smoke: F-A NULL, TG1v (draft-target) and TG2 (two-model EAGLE-3, FR12) x S0-S6, G; F-B NULL and TG1v x
  S0, S2, S6, G. At most 100 sequences each, discarded unread. Each cell records `type(sampler)`, the spec
  counters and the sampling strategy TorchSampler resolves: S4 and S5 resolve to temperature 1 alone, because
  TorchSampler applies neither min_p nor repetition_penalty (FR12);
- trtllm_sampler: FR9 item 1's `sampler_type="TRTLLMSampler"` smoke for NULL and TG1v at S4-S6;
- seed_check: FR9 item 3's `_global_seed` hook: one seed repeats exactly, another changes the draws;
- reference: R_eng from the raw generation logits passed through the sampler's own
  `top_k_top_p_sampling_batch` at the production batch (10^4 rows), δ_num against `rnp_full.trtllm_dist`;
- throughput on PROMPT_TP, one generate call of 10^4 requests.
"""
import json
import os
import time

import numpy as np

import p0b_common as C

LAYOUT = "_vllm"  # F-A head_dim-32 re-layout (Phase 0: both layouts ran NULL on VANILLA; FR9 item 1)


def llm(fix, method, sampler_type=None, layout=None):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, EagleDecodingConfig, KvCacheConfig
    lay = LAYOUT if (fix.name == "FA" and layout is None) else (layout or "")
    kw = dict(model=fix.path("target", lay), skip_tokenizer_init=True, dtype="float32", attn_backend="VANILLA",
              cuda_graph_config=None, disable_overlap_scheduler=True, disable_flashinfer_sampling=True,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3), max_num_tokens=1024,
              max_batch_size=128, enable_iter_perf_stats=True)
    if sampler_type:
        kw["sampler_type"] = sampler_type
    if method == "TG1v":
        kw["speculative_config"] = DraftTargetDecodingConfig(max_draft_len=3, speculative_model=fix.path("draft", lay))
    elif method.startswith("TG2"):  # TG2@tgt / TG2@flat: the two head variants (heads.py)
        # two-model EAGLE-3 (FR12): one-model drafts greedily and keeps bf16 hidden states; no d2t (draft vocab = V)
        kw["speculative_config"] = EagleDecodingConfig(
            max_draft_len=3, speculative_model=str(C.fx.MODELS / f"eagle3_{fix.name}_{method.split('@')[1]}"),
            eagle3_one_model=False,
            eagle3_layers_to_capture=set(EAGLE3_CAPTURE[fix.name]))
    return LLM(**kw)


EAGLE3_CAPTURE = {"FA": (0,), "FB": (0, 1, 2)}  # heads.AUX["trtllm"]


def sampler_of(m):
    return m._executor.engine.sampler


def params(fix, setting, n, seed0=0, max_tokens=None, **over):
    from tensorrt_llm import SamplingParams
    sp = C.SETTINGS[setting]
    kw = dict(max_tokens=max_tokens or fix.MAX_NEW, end_id=fix.EOS, pad_id=fix.EOS,
              temperature=sp.get("temperature", 1.0), top_k=sp.get("top_k"), top_p=sp.get("top_p"),
              min_p=sp.get("min_p"), repetition_penalty=sp.get("repetition_penalty"))
    kw.update(over)
    return [SamplingParams(seed=seed0 + i, **kw) for i in range(n)]


class Stats:
    """Executor spec-decoding stats, plus their provenance.

    `PyExecutor._update_iter_stats` (py_executor.py 828-837) takes `req.num_draft_tokens`, and when that is 0 falls
    back to counting the non-zero ids in `py_draft_tokens`; id 0 is a real token (FR8), so a fallback count is not
    exact. The wrapper counts, per iteration, the requests that took the fallback; FR12 treats acceptance as
    unavailable unless `fallback_requests` is 0."""

    def __init__(self, m):
        self.m, self.c = m, dict(draft_tokens=0, accepted_tokens=0, stat_iterations=0, fallback_requests=0)
        eng = m._executor.engine
        orig = eng._update_iter_stats

        def wrapped(stats, iter_latency_ms, num_completed_requests, scheduled_batch, micro_batch_id):
            for req in getattr(scheduled_batch, "generation_requests", []):
                if getattr(req, "num_draft_tokens", 0) == 0 and getattr(req, "py_draft_tokens", None):
                    self.c["fallback_requests"] += 1
            return orig(stats, iter_latency_ms, num_completed_requests, scheduled_batch, micro_batch_id)

        eng._update_iter_stats = wrapped

    def drain(self):
        try:
            ex = self.m._executor
            stats = [json.loads(ex._stats_serializer(st)) for st in ex.fetch_stats()]
            for st in stats:
                sd = st.get("specDecodingStats") or {}
                self.c["draft_tokens"] += int(sd.get("numDraftTokens") or 0)
                self.c["accepted_tokens"] += int(sd.get("numAcceptedTokens") or 0)
                self.c["stat_iterations"] += 1
        except Exception as e:  # noqa: BLE001
            self.c["error"] = f"{type(e).__name__}: {e}"[:200]
        return dict(self.c)


def seed_hook(m, seed):
    """FR9 item 3: the sampler's generator is created lazily from `_global_seed`; set it before any draw."""
    s = sampler_of(m)
    assert s._generator is None, "sampler generator already created"
    s._global_seed = int(seed)
    return type(s).__name__


def greedy_flags(fix, setting):
    """The strategy TorchSampler resolves for this setting (`sampling_utils.resolve_sampling_strategy`), recorded."""
    p = params(fix, setting, 1)[0]
    try:
        from tensorrt_llm._torch.pyexecutor.sampling_utils import UtilsSamplingParams, resolve_sampling_strategy
        return str(resolve_sampling_strategy(UtilsSamplingParams(temperature=p.temperature, top_p=p.top_p,
                                                                 top_k=p.top_k, use_beam_search=False),
                                         vocab_size=fix.V))
    except Exception as e:  # noqa: BLE001
        return f"unknown: {type(e).__name__}: {e}"[:120]


ORDER = ["S0", "S1", "S2", "S3", "G", "S4", "S5", "S6"]  # S4-S6 BLOCKED (FR12), smoked last


def drain_until_quiet(st, tries=5):
    """Per-iteration stats arrive after the responses; drain until two drains in a row add nothing."""
    quiet = 0
    for _ in range(tries * 4):
        before = dict(st.c)
        st.drain()
        quiet = quiet + 1 if st.c == before else 0
        if quiet >= 2:
            break
        time.sleep(0.25)
    return dict(st.c)


def smoke_one(fname, method, settings, sampler_type=None, prefix=""):
    fix = C.fixture(fname)
    out = {}
    try:
        m = llm(fix, method, sampler_type)
    except Exception as e:  # noqa: BLE001
        return dict(smoke={f"{prefix}{fname}/{method}/load": dict(error=f"{type(e).__name__}: {e}"[:400])})
    try:
        st = Stats(m)
        for s in settings:
            key = f"{prefix}{fname}/{method}/{s}"
            before = drain_until_quiet(st)  # nothing in flight: the generator can be replaced
            seed = C.seed_base("P0b", f"{fname}/{method}/{s}")
            try:
                # §3: every cell starts its own stream at h("P0b", cell); the generator is fetched through
                # get_generator (sampler.py 967-981), so clearing it makes the next draw create it from this seed
                if hasattr(sampler_of(m), "_generator"):
                    sampler_of(m)._generator = None
                sampler = seed_hook(m, seed)
            except Exception as e:  # noqa: BLE001
                sampler = dict(error=f"seed hook failed: {type(e).__name__}: {e}"[:200])  # unseeded: a harness failure
            if isinstance(sampler, dict):  # unseeded: the cell does not draw
                out[key] = dict(sampler)
                C.phase0_checkpoint(out)
                continue
            try:
                t = time.time()
                o = m.generate([fix.PROMPT] * C.N_SMOKE, params(fix, s, C.N_SMOKE))
                out[key] = dict(seq_per_s=round(C.N_SMOKE / (time.time() - t), 1), strategy=greedy_flags(fix, s),
                                stream_seed=seed, sampler=sampler,
                                **fix.shape_check([x.outputs[0].token_ids for x in o], expected=C.N_SMOKE))
            except Exception as e:  # noqa: BLE001
                out[key] = dict(error=f"{type(e).__name__}: {e}"[:300])
            after = drain_until_quiet(st)
            # draft_tokens is exact only when no request took the executor's non-zero-id fallback (Stats docstring)
            sc = {k: after[k] - before.get(k, 0) for k in ("draft_tokens", "accepted_tokens", "stat_iterations",
                                                           "fallback_requests")}
            sc["acceptance_available"] = sc["fallback_requests"] == 0 and "error" not in after
            out[key]["spec_counters"] = sc
            if "error" in after:  # a stats failure must not read as zero drafts
                out[key]["spec_counters"]["error"] = after["error"]
            C.phase0_checkpoint(out)
    finally:
        m.shutdown()
    return dict(smoke=out, hooks=hooks())


def smoke_probes():
    out = []
    for fname, methods, settings in (("FA", ("NULL", "TG1v", "TG2@tgt", "TG2@flat"), ORDER),
                                     ("FB", ("NULL", "TG1v"), ["S0", "S2", "G", "S6"])):
        for m in methods:
            out.append((f"smoke_{fname}_{m}", lambda f=fname, m=m, ss=settings: smoke_one(f, m, ss)))
    for m in ("NULL", "TG1v"):  # FR9 item 1
        out.append((f"smoke_FA_{m}_TRTLLMSampler",
                    lambda m=m: smoke_one("FA", m, ["S4", "S5", "S6"], sampler_type="TRTLLMSampler", prefix="TRTLLMSampler:")))
    return out


def hooks():
    import d40_trtllm_fp32
    from tensorrt_llm._torch.modules import rms_norm, rotary_embedding
    from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter
    return dict(flashinfer_rmsnorm=rms_norm.IS_FLASHINFER_AVAILABLE, flashinfer_rope=rotary_embedding.IS_FLASHINFER_AVAILABLE,
                drafter_inference_mode=getattr(ModelDrafter.process_decoded_tokens, "_d40_inference_mode", False),
                fp32_hook_patched=sorted(d40_trtllm_fp32.PATCHED))


def seed_check():
    out = {}
    for method in ("NULL", "TG1v"):
        cell = f"FA/{method}/S0"
        b = C.seed_base("P0b", cell)
        m = llm(C.FA, method)
        try:
            draws = []
            for seed in (b, b, b + 1):
                s = sampler_of(m)
                s._generator = None  # reseed: the generator is rebuilt from _global_seed at the next draw
                s._global_seed = seed
                o = m.generate([C.PROMPT_TP] * 20, params(C.FA, "S0", 20))  # no grid-cell draws
                draws.append([tuple(x.outputs[0].token_ids) for x in o])
            out[cell] = dict(same_seed_repeats=C.same_draws(draws[0], draws[1]),
                             other_seed_differs=not C.same_draws(draws[0], draws[2]))
            del draws
        finally:
            m.shutdown()
    return dict(seed_check=out)


class LiveCapture:
    """Wraps `sampling_utils.sample`, which TorchSampler's grouped-strategy sampler calls as a module global
    (`sampling_utils.py` 598), and keeps the processed probabilities it returns: the distribution the live
    executor samples from, at the batch the executor actually forms (worker in process, FR8)."""

    def __enter__(self):
        import tensorrt_llm._torch.pyexecutor.sampling_utils as su
        self.su, self.orig, self.rows, self.strategies, self.batches = su, su.sample, [], set(), []

        def sample(strategy, logits, *a, **k):
            tokens, probs = self.orig(strategy, logits, *a, **k)
            if probs is not None:
                self.rows.append(probs.detach().double().reshape(-1, probs.shape[-1]).cpu().numpy())
                self.strategies.add(str(strategy))
                self.batches.append(int(probs.shape[0]))
            return tokens, probs

        su.sample = sample
        return self

    def __exit__(self, *exc):
        self.su.sample = self.orig


COPIES = 128  # = max_batch_size: each scoring call fills one executor batch with copies of one prefix


def step_table(m, fix, setting, prefixes):
    """R_eng from the live sampler: per prefix, one generate call of COPIES identical one-token requests; every
    captured row belongs to that prefix. Returns (all rows, row -> prefix mapping, strategies, batch sizes)."""
    rows, mapping, strategies, batches = [], [], set(), []
    for j, g in enumerate(prefixes):
        with LiveCapture() as cap:
            m.generate([fix.PROMPT + list(g)] * COPIES, params(fix, setting, COPIES, max_tokens=1))
        if not cap.rows:
            raise RuntimeError(f"no processed probabilities captured for prefix {list(g)} at {setting}")
        r = np.concatenate(cap.rows)
        rows.append(r)
        mapping += [j] * len(r)
        strategies |= cap.strategies
        batches += cap.batches
    return np.concatenate(rows), mapping, sorted(strategies), batches


def reference():
    import rnp
    import rnp_full  # noqa: F401
    tables = os.environ.get("D40_P0B_OUT", "/tmp/p0b") + "_tables"
    out = {}
    for fname, settings in (("FA", ["S0", "S1", "S2", "S3"]), ("FB", ["S0", "S2"])):
        fix = C.fixture(fname)
        try:
            m = llm(fix, "NULL")
            seed_hook(m, 1)
        except Exception as e:  # noqa: BLE001
            out[f"{fname}/load"] = dict(error=f"{type(e).__name__}: {e}"[:300])
            continue
        try:
            for s in settings:
                try:
                    if fname == "FA":
                        pre, ref = C.fa_prefixes(), rnp.step_table("trtllm", C.SETTINGS[s])
                    else:
                        pre, fref = C.fb_prefixes("trtllm", s)
                        ref = np.stack([fref.step(tuple(g)) for g in pre])
                    rows, mapping, strategies, batches = step_table(m, fix, s, pre)
                    first, dev = C.first_rows(rows, mapping, len(pre))
                    out[f"{fname}/{s}"] = dict(C.delta_report(rows, ref, mapping), copy_dev=dev, strategies=strategies,
                                               sampler_batch_rows=[min(batches), max(batches)])
                    C.save_npz(tables, f"trtllm_{fname}_{s}.npz", eng=first, np=ref)
                except Exception as e:  # noqa: BLE001
                    out[f"{fname}/{s}"] = dict(error=f"{type(e).__name__}: {e}"[:300])
                C.phase0_checkpoint(out)
        finally:
            m.shutdown()
    return dict(reference=out, note="R_eng captured from the live TorchSampler (sampling_utils.sample); "
                                    "S4-S6 BLOCKED (FR12); G is checked by argmax in Stage A")


def throughput():
    out = {}
    for method in ("NULL", "TG1v", "TG2@tgt"):
        try:
            m = llm(C.FA, method)
        except Exception as e:  # noqa: BLE001
            out[method] = dict(error=f"load: {type(e).__name__}: {e}"[:200])
            continue
        try:
            seed_hook(m, 1)
            t = time.time()
            m.generate([C.PROMPT_TP] * C.CHUNK, params(C.FA, "S0", C.CHUNK))
            out[f"FA/{method}/S0"] = round(C.CHUNK / (time.time() - t), 1)
        finally:
            m.shutdown()
    return dict(throughput_seq_per_s=out)


def probes():
    return [*smoke_probes(), ("seed_check", seed_check), ("reference", reference), ("throughput", throughput)]
