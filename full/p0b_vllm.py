"""Phase 0b probes for vLLM 0.30.0 on an L4 (FR4, FR9 items 1/3/5/7, FR12).

vLLM 0.30.0 runs two model runners (FR12, `vllm/config/vllm.py` 675-723, 2838-2850): `draft_model` and
`ngram` (VG1, VG2, VG3, VG5) fall back to the V1 runner and its V1 sampler / rejection sampler, while the
non-speculative engine and `eagle3` / `mtp` (VG4, VG6) use Model Runner V2. A cell's null and R_eng must
come from the sampler that draws its samples (§5.1), so every V1 cell is paired with a V1 null
(`VLLM_USE_V2_MODEL_RUNNER=0`) and every V2 cell with a V2 null. The runner is fixed per probe (one child
process each). All probes run with `VLLM_USE_FLASHINFER_SAMPLER=0` (FR12: the FlashInfer sampler is used
only for some batch compositions and never for processed logprobs, so with it on, null draws and R_eng
could take different top-k/top-p kernels) and `VLLM_ENABLE_V1_MULTIPROCESSING=0` (the engine core in this
process, so the P7/P8 patches reach it).

F-A uses the head_dim-32 re-layout (FR4). Methods: NULL; VG1 draft_model greedy draft; VG2 draft_model
probabilistic; VG3 ngram; VG4@tgt / VG4@flat EAGLE-3 head variants (heads.py); VG5 draft_model with
`rejection_sample_method="block"`; VG6 MTP on `mimo_FA` (heads.py), k = 1 (one MTP layer).
"""
import os
import time

import numpy as np

import p0b_common as C

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

LAYOUT = "_vllm"
V1_METHODS = ("NULL", "VG1", "VG2", "VG3", "VG5")
V2_METHODS = ("NULL", "VG4@tgt", "VG4@flat", "VG6")


def runner(v):
    """Must run before vllm is imported in this process (one probe per child)."""
    if v == 2:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)  # V2 is the default
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


def runner_used(m):
    """What the engine actually chose (recorded, not assumed)."""
    try:
        return bool(m.llm_engine.vllm_config.use_v2_model_runner)
    except Exception as e:  # noqa: BLE001
        return f"unknown: {type(e).__name__}"


def lay(fix):
    return LAYOUT if fix.name == "FA" else ""


def spec(fix, method):
    dft = fix.path("draft", lay(fix))
    base = method.split("@")[0]
    if base == "NULL":
        return None
    if base == "VG1":
        return dict(method="draft_model", model=dft, num_speculative_tokens=3)
    if base == "VG2":
        return dict(method="draft_model", model=dft, num_speculative_tokens=3, draft_sample_method="probabilistic")
    if base == "VG3":
        return dict(method="ngram", num_speculative_tokens=3, prompt_lookup_min=1, prompt_lookup_max=2)
    if base == "VG4":
        return dict(method="eagle3", model=str(C.fx.MODELS / f"eagle3_{fix.name}_{method.split('@')[1]}"),
                    num_speculative_tokens=3)
    if base == "VG5":
        return dict(method="draft_model", model=dft, num_speculative_tokens=3, rejection_sample_method="block")
    if base == "VG5p":  # FR9 item 5 preflight: block with probabilistic drafts, counters only
        return dict(method="draft_model", model=dft, num_speculative_tokens=3, rejection_sample_method="block",
                    draft_sample_method="probabilistic")
    if base == "VG6":
        return dict(method="mtp", num_speculative_tokens=1)
    raise KeyError(method)


def target(fix, method):
    if method == "VG6" or method == "NULL_mimo":
        return str(C.fx.MODELS / "mimo_FA")
    return fix.path("target", lay(fix))


def llm(fix, method, **over):
    from vllm import LLM
    kw = dict(model=target(fix, method), skip_tokenizer_init=True, dtype="float32", max_model_len=64, seed=0,
              enforce_eager=True, disable_log_stats=False, gpu_memory_utilization=0.3, max_logprobs=-1,
              logprobs_mode="processed_logprobs")
    s = spec(fix, method) if method != "NULL_mimo" else None
    if s:
        kw["speculative_config"] = s
    kw.update(over)
    return LLM(**kw)


def params(fix, setting, seeds, **over):
    from vllm import SamplingParams
    return [SamplingParams(max_tokens=fix.MAX_NEW, seed=int(s), detokenize=False, **C.SETTINGS[setting], **over)
            for s in seeds]


SPEC_KEYS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")


def counters(m):
    """Numeric spec counters, or None when they cannot be read (never an error string that could compare equal)."""
    try:
        got = {x.name: getattr(x, "value", None) for x in m.get_metrics() if x.name in SPEC_KEYS}
    except Exception:  # noqa: BLE001
        return None
    if set(got) != set(SPEC_KEYS) or not all(isinstance(v, (int, float)) for v in got.values()):
        return None
    return {k.split(":")[1]: int(v) for k, v in got.items()}


def delta(after, before):
    if after is None or before is None:
        return None
    return {k: after[k] - before[k] for k in after}


def generate(m, fix, setting, seeds, prompt=None):
    o = m.generate([{"prompt_token_ids": prompt or fix.PROMPT}] * len(seeds), params(fix, setting, seeds), use_tqdm=False)
    return [x.outputs[0].token_ids for x in o]


def free(m):
    """Shut the engine down before the next one is built (two 0.3-memory engines in one process otherwise)."""
    try:
        m.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001
        pass
    import gc
    import torch
    gc.collect()
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()
    except Exception:  # noqa: BLE001
        torch.cuda.empty_cache()


ORDER = ["S0", "S1", "S2", "S3", "S5", "G", "S4", "S6"]  # refusals expected at S4 and S6 come last


def smoke_one(fname, v, method, settings=ORDER):
    runner(v)
    fix = C.fixture(fname)
    out = {}
    try:
        m = llm(fix, method)
    except Exception as e:  # noqa: BLE001
        return dict(smoke={f"{fname}/{method}/load": dict(runner=f"V{v}", error=f"{type(e).__name__}: {e}"[:400])})
    try:
        out[f"{fname}/{method}/use_v2_model_runner"] = runner_used(m)
        for s in settings:
            key = f"{fname}/{method}/{s}"
            before = counters(m)
            try:
                t = time.time()
                seqs = generate(m, fix, s, C.seeds("P0b", key, C.N_SMOKE))
                out[key] = dict(runner=f"V{v}", seq_per_s=round(C.N_SMOKE / (time.time() - t), 1),
                                **fix.shape_check(seqs, expected=C.N_SMOKE))
            except Exception as e:  # noqa: BLE001
                out[key] = dict(runner=f"V{v}", error=f"{type(e).__name__}: {e}"[:300])
            out[key]["spec_counters"] = delta(counters(m), before)  # this setting only
            C.phase0_checkpoint(out)
    finally:
        free(m)
    return dict(smoke=out)


def smoke_probes():
    out = []
    for fname, v, methods, settings in (
            ("FA", 1, V1_METHODS, ORDER), ("FA", 2, V2_METHODS + ("NULL_mimo",), ORDER),
            ("FB", 1, ("NULL", "VG1", "VG2"), ["S0", "S2", "G", "S6"]),
            ("FB", 2, ("NULL", "VG4@tgt", "VG4@flat"), ["S0", "S2", "G", "S6"])):
        for method in methods:
            out.append((f"smoke_{fname}_{method}_V{v}", lambda f=fname, v=v, mm=method, ss=settings: smoke_one(f, v, mm, ss)))
    return out


def drafter_flag(m):
    """The V1 proposer's `_enable_probabilistic_draft_probs`, through the in-process engine core (best effort)."""
    for path in ("llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.drafter",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.drafter"):
        obj = m
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return bool(obj._enable_probabilistic_draft_probs)
        except AttributeError:
            continue
    return "unknown"


def vg5_preflight():
    """FR9 item 5: VG5 with probabilistic drafts against VG2, as frozen, plus VG1 (FR12): the V1 proposer samples
    drafts only for "standard" + "probabilistic" (`v1/spec_decode/llm_base_proposer.py` 261-264), so VG5p drafts
    greedily like VG1, and V1 probabilistic drafting ignores seeds (same file, 1877). Counters only, on PROMPT_TP
    (no grid-cell draws); each engine is fresh, so its counters are this run's. The fold is decided only from
    complete numeric counters with drafts proposed in every run; anything else is UNRESOLVED."""
    runner(1)
    out = {}
    seeds = C.seeds("P0b", "FA/VG5/preflight", C.N_SMOKE)
    for method in ("VG2", "VG5p", "VG1"):
        m = llm(C.FA, method)
        try:
            generate(m, C.FA, "S0", seeds, prompt=C.PROMPT_TP)
            out[method] = counters(m)
            out[f"{method}_probabilistic_draft_probs"] = drafter_flag(m)
        finally:
            free(m)
            del m
    complete = all(out[k] is not None and out[k]["spec_decode_num_draft_tokens"] > 0 for k in ("VG2", "VG5p", "VG1"))
    out["complete"] = complete
    out["VG5p_equals_VG1"] = (out["VG5p"] == out["VG1"]) if complete else "UNRESOLVED"
    out["VG5p_equals_VG2"] = (out["VG5p"] == out["VG2"]) if complete else "UNRESOLVED"
    return dict(vg5_preflight=out)


def seed_check_v(v, methods):
    runner(v)
    out = {}
    for method in methods:
        m = llm(C.FA, method)
        try:
            cell = f"FA/{method}/S0"
            s = C.seeds("P0b", cell, 50)  # on PROMPT_TP: seed checks draw no sequence of a grid cell (FR12)
            x, y = generate(m, C.FA, "S0", s, prompt=C.PROMPT_TP), generate(m, C.FA, "S0", s, prompt=C.PROMPT_TP)
            z = generate(m, C.FA, "S0", [a + 50 for a in s], prompt=C.PROMPT_TP)
            out[cell] = dict(runner=f"V{v}", same_seed_repeats=C.same_draws(x, y), next_seeds_differ=not C.same_draws(x, z))
            del x, y, z
        finally:
            free(m)
    return out


def seed_check_v1():
    return dict(seed_check=seed_check_v(1, ("NULL", "VG1", "VG2")),
                note="V1 probabilistic drafting ignores seeds (llm_base_proposer.py, TODO consider seeds)")


def seed_check_v2():
    return dict(seed_check=seed_check_v(2, ("NULL", "VG4@tgt")))


# ------------------------------------------------------------------ P7 / P8 effect assertions (V1 draft_model)

def _maxdiff(a, b):
    import torch
    big = 1e30
    a = torch.nan_to_num(a.float(), nan=0.0, posinf=big, neginf=-big)
    b = torch.nan_to_num(b.float(), nan=0.0, posinf=big, neginf=-big)
    return float((a - b).abs().max()) if a.numel() else 0.0


def effect_p7():
    """P7 (vllm#42744 class): target sampling constraints and logits processors skipped at drafted positions.
    Patched quantity: the target logits the rejection sampler verifies against (processed -> raw)."""
    runner(1)
    import vllm.v1.sample.rejection_sampler as rs
    rec = dict(calls=0, max_abs_diff=0.0)
    orig_c, orig_p = rs.apply_sampling_constraints, rs.RejectionSampler.apply_logits_processors

    def constraints(logits, *a, **k):
        proc = orig_c(logits.clone(), *a, **k)
        rec["calls"] += 1
        rec["max_abs_diff"] = max(rec["max_abs_diff"], _maxdiff(proc, logits))
        return logits

    rs.apply_sampling_constraints = constraints
    rs.RejectionSampler.apply_logits_processors = lambda self, logits, *a, **k: logits
    try:
        m = llm(C.FA, "VG1")
        try:
            generate(m, C.FA, "S2", C.seeds("P0b", "FA/P7/S2", 20))
        finally:
            free(m)
    finally:
        rs.apply_sampling_constraints, rs.RejectionSampler.apply_logits_processors = orig_c, orig_p
    return dict(effect=dict(P7_S2=dict(rec, changed=rec["calls"] > 0 and rec["max_abs_diff"] > 0)))


def effect_p8():
    """P8 (pilot L3 behaviour 1): a greedy draft whose q is used as if sampled. Patched quantity: the draft
    token ids (sampled -> argmax), with q still the full softmax handed to the verifier."""
    runner(1)
    import torch
    import vllm.v1.spec_decode.llm_base_proposer as lbp
    rec = dict(calls=0, tokens_changed=0, q_nondegenerate=False)
    orig = lbp.compute_probs_and_sample_next_token

    def p8(logits, *a, **k):
        ids, probs = orig(logits, *a, **k)
        am = probs.argmax(dim=-1)
        rec["calls"] += 1
        rec["tokens_changed"] += int((am != ids).sum())
        rec["q_nondegenerate"] |= bool((probs.max(-1).values < 1).any())
        return am.to(ids.dtype), probs

    lbp.compute_probs_and_sample_next_token = p8
    try:
        m = llm(C.FA, "VG2")
        try:
            generate(m, C.FA, "S0", C.seeds("P0b", "FA/P8/S0", 20))
        finally:
            free(m)
    finally:
        lbp.compute_probs_and_sample_next_token = orig
    torch.cuda.synchronize()
    return dict(effect=dict(P8_S0=dict(rec, changed=rec["tokens_changed"] > 0 and rec["q_nondegenerate"])))


# ------------------------------------------------------------------ R_eng (§5.1) and δ_num

def step_table(m, fix, setting, prefixes):
    """Returns (rows for all 10^4 requests, request -> prefix mapping)."""
    from vllm import SamplingParams
    reqs, mapping = C.batch_copies(prefixes)
    o = m.generate([{"prompt_token_ids": fix.PROMPT + list(g)} for g in reqs],
                   [SamplingParams(max_tokens=1, seed=1, detokenize=False, logprobs=fix.V, **C.SETTINGS[setting])
                    for _ in reqs], use_tqdm=False)
    rows = []
    for x in o:
        row = np.zeros(fix.V)
        for tok, lp in x.outputs[0].logprobs[0].items():
            row[tok] = np.exp(lp.logprob)
        rows.append(row)
    return np.stack(rows), mapping


RUNNING = ["S0", "S1", "S2", "S3", "S5"]  # S4, S6 BLOCKED (FR9 item 1)


def reference_v(v, targets):
    runner(v)
    import rnp
    tables = os.environ.get("D40_P0B_OUT", "/tmp/p0b") + "_tables"
    out = {}
    for fname, method in targets:
        fix = C.fixture(fname)
        try:
            m = llm(fix, method)
        except Exception as e:  # noqa: BLE001
            out[f"{fname}/{method}/load/V{v}"] = dict(error=f"{type(e).__name__}: {e}"[:300])
            C.phase0_checkpoint(out)
            continue
        try:
            for s in (RUNNING if fname == "FA" else ["S0", "S2"]):
                try:
                    if fname == "FA":
                        pre, ref = C.fa_prefixes(), rnp.step_table("vllm", C.SETTINGS[s])
                    else:
                        pre, fref = C.fb_prefixes("vllm", s)
                        ref = np.stack([fref.step(tuple(g)) for g in pre])
                    rows, mapping = step_table(m, fix, s, pre)
                    eng, dev = C.first_rows(rows, mapping, len(pre))
                    out[f"{fname}/{method}/{s}/V{v}"] = dict(C.delta_report(rows, ref, mapping), copy_dev=dev)
                    C.save_npz(tables, f"vllm_{fname}_{method}_{s}_V{v}.npz", eng=eng, np=ref)
                except Exception as e:  # noqa: BLE001
                    out[f"{fname}/{method}/{s}/V{v}"] = dict(error=f"{type(e).__name__}: {e}"[:300])
                C.phase0_checkpoint(out)
        finally:
            free(m)
    return out


def reference_v1():
    return dict(reference=reference_v(1, [("FA", "NULL"), ("FB", "NULL")]))


def reference_v2():
    return dict(reference=reference_v(2, [("FA", "NULL"), ("FB", "NULL"), ("FA", "NULL_mimo")]),
                note="NULL_mimo: mimo_FA (VG6 target), zero q/k/v biases, scored against F-A's R_np")


def throughput_v(v, methods):
    runner(v)
    out = {}
    for method in methods:
        try:
            m = llm(C.FA, method)
        except Exception as e:  # noqa: BLE001
            out[method] = dict(error=f"load: {type(e).__name__}: {e}"[:200])
            continue
        try:
            t = time.time()
            generate(m, C.FA, "S0", range(1, C.CHUNK + 1), prompt=C.PROMPT_TP)
            out[f"FA/{method}/S0/V{v}"] = round(C.CHUNK / (time.time() - t), 1)
        finally:
            free(m)
    return out


def throughput_v1():
    return dict(throughput_seq_per_s=throughput_v(1, ("NULL", "VG1", "VG2")))


def throughput_v2():
    return dict(throughput_seq_per_s=throughput_v(2, ("NULL", "VG4@tgt")))


def probes():
    return [*smoke_probes(), ("vg5_preflight", vg5_preflight), ("seed_check_v1", seed_check_v1),
            ("seed_check_v2", seed_check_v2), ("effect_p7", effect_p7), ("effect_p8", effect_p8),
            ("reference_v1", reference_v1), ("reference_v2", reference_v2), ("throughput_v1", throughput_v1),
            ("throughput_v2", throughput_v2)]
