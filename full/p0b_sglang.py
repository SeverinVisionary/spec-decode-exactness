"""Phase 0b probes for SGLang 0.5.21 on an L4 (FR5, FR6, FR9 items 1/3/7, FR12).

Engine options are FR5/FR6's: Triton attention for target and draft, CUDA graphs off, fused ops in torch.
F-A uses the head_dim-32 re-layout (FR9 item 1). Methods: SG1 (EAGLE-3 head, chain, topk 1), SG2 (EAGLE-3
head, tree, topk 2), SG3s (standalone draft), SG3n (n-gram).

- smoke: F-A NULL, SG1, SG2, SG3s, SG3n x S0-S6, G; F-B NULL and SG1 x S0, S2, S6, G. At most 100 sequences
  each, discarded unread, with the spec fields of `meta_info` summed;
- backend: FR9 item 1's `sampling_backend="pytorch"` smoke at S4 and S6 for NULL and SG1;
- seed_check: FR9 item 3: per-request seeds are ignored, so one engine instance per cell with
  `random_seed = h`: two instances with the same seed repeat, another seed differs;
- reference: R_eng on the non-speculative engine: S0, S1, S5 from `return_logprob` (softmax of the
  penalised, temperature-scaled logits, `sampler.py` 248-280); S2, S3, S4, S6 from
  `return_sampling_mask` with `sampling_logprobs_mode="support"` (the sampler's own post-filter weights,
  `sampler.py` 530-584), which needs a finite top_k, so settings without one pass top_k = V; δ_num against
  `rnp_full.sglang_dist`. S5 is recorded but cannot be reached by prompt scoring (FR12);
- throughput on PROMPT_TP, one generate call of 10^4 requests.
"""
import os
import time

import numpy as np

import p0b_common as C

LAYOUT = "_vllm"
SPEC = {
    "NULL": lambda fix, v: {},
    "SG1": lambda fix, v: dict(speculative_algorithm="EAGLE3", speculative_draft_model_path=str(C.fx.MODELS / f"eagle3_{fix.name}_{v}"),
                               speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4),
    "SG2": lambda fix, v: dict(speculative_algorithm="EAGLE3", speculative_draft_model_path=str(C.fx.MODELS / f"eagle3_{fix.name}_{v}"),
                               speculative_num_steps=3, speculative_eagle_topk=2, speculative_num_draft_tokens=6),
    "SG3s": lambda fix, v: dict(speculative_algorithm="STANDALONE", speculative_draft_model_path=fix.path("draft", lay(fix)),
                             speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4),
    "SG3n": lambda fix, v: dict(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4),
}


def lay(fix):
    return LAYOUT if fix.name == "FA" else ""


def engine(fix, method, seed, **over):
    os.environ["SGLANG_FORCE_FUSED_OP_BACKEND"] = "torch"  # FR6, inherited by the scheduler subprocess
    import sglang as sgl
    kw = dict(model_path=fix.path("target", lay(fix)), skip_tokenizer_init=True, dtype="float32", context_length=64,
              mem_fraction_static=0.3, random_seed=int(seed), attention_backend="triton",
              cuda_graph_backend_prefill="disabled", cuda_graph_backend_decode="disabled",
              **SPEC[method.split("@")[0]](fix, method.split("@")[1] if "@" in method else None))
    if method.split("@")[0] in ("SG1", "SG2", "SG3s"):  # drafts with a model (FR5)
        kw["speculative_draft_attention_backend"] = "triton"
    kw.update(over)
    return sgl.Engine(**kw)


def sp(fix, setting, **over):
    s = C.SETTINGS[setting]
    p = dict(max_new_tokens=fix.MAX_NEW, temperature=s.get("temperature", 1.0), top_k=s.get("top_k", -1),
             top_p=s.get("top_p", 1.0), min_p=s.get("min_p", 0.0), repetition_penalty=s.get("repetition_penalty", 1.0))
    p.update(over)
    return p


SPEC_FIELDS = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")


def counters(outs):
    c = {f: 0 for f in SPEC_FIELDS}
    for o in outs:
        mi = o.get("meta_info", {})
        for f in SPEC_FIELDS:
            c[f] += int(mi.get(f) or 0)
    return c


SAFE = ["S0", "S1", "S2", "S3", "S5", "G"]
HAZARD = ["S4", "S6"]  # the min_p branch: S6 killed the scheduler in Phase 0, so each runs in its own child


def smoke_one(fname, method, settings, prefix="", **engine_over):
    """§3: SGLang ignores per-request seeds, so each cell gets its own engine with random_seed = h("P0b", cell)."""
    fix = C.fixture(fname)
    out = {}
    for s in settings:
        key = f"{prefix}{fname}/{method}/{s}"
        seed = C.seed_base("P0b", f"{fname}/{method}/{s}")
        try:
            eng = engine(fix, method, seed, **engine_over)
        except Exception as e:  # noqa: BLE001
            out[f"{prefix}{fname}/{method}/load"] = dict(error=f"{type(e).__name__}: {e}"[:400])
            return dict(smoke=out)
        try:
            t = time.time()
            o = eng.generate(input_ids=[fix.PROMPT] * C.N_SMOKE, sampling_params=sp(fix, s))
            out[key] = dict(seq_per_s=round(C.N_SMOKE / (time.time() - t), 1), spec=counters(o), random_seed=seed,
                            **fix.shape_check([x["output_ids"] for x in o], expected=C.N_SMOKE))
        except Exception as e:  # noqa: BLE001
            out[key] = dict(error=f"{type(e).__name__}: {e}"[:300])
        finally:
            eng.shutdown()
        C.phase0_checkpoint(out)
    return dict(smoke=out, engine_options=dict(options(), **engine_over))


def options():
    return dict(attention_backend="triton", cuda_graphs="disabled", fused_op_backend="torch")


FA_METHODS = ("NULL", "SG1@tgt", "SG1@flat", "SG2@tgt", "SG2@flat", "SG3s", "SG3n")
FB_METHODS = ("NULL", "SG1@tgt", "SG1@flat")


def smoke_probes():
    out = []
    for fname, methods, safe, hazard in (("FA", FA_METHODS, SAFE, HAZARD), ("FB", FB_METHODS, ["S0", "S2", "G"], ["S6"])):
        for m in methods:
            out.append((f"smoke_{fname}_{m}", lambda f=fname, m=m, ss=safe: smoke_one(f, m, ss)))
        for m in methods:
            for h in hazard:
                out.append((f"smoke_{fname}_{m}_{h}", lambda f=fname, m=m, h=h: smoke_one(f, m, [h])))
    # FR9 item 1: the shipped sampling_backend="pytorch" at S4 and S6, one child each
    for m in ("NULL", "SG1@tgt"):
        for h in HAZARD:
            out.append((f"smoke_FA_{m}_{h}_pytorch",
                        lambda m=m, h=h: smoke_one("FA", m, [h], prefix="pytorch:", sampling_backend="pytorch")))
    return out


def seed_check():
    out = {}
    for method in ("NULL", "SG1@tgt"):
        cell = f"FA/{method}/S0"
        b = C.seed_base("P0b", cell)
        draws = []
        for seed in (b, b, b + 1):
            eng = engine(C.FA, method, seed)
            try:
                o = eng.generate(input_ids=[C.PROMPT_TP] * 20, sampling_params=sp(C.FA, "S0"))  # no grid draws
                draws.append([tuple(x["output_ids"]) for x in o])
            finally:
                eng.shutdown()
        out[cell] = dict(same_seed_repeats=C.same_draws(draws[0], draws[1]),
                         other_seed_differs=not C.same_draws(draws[0], draws[2]))
        del draws
    return dict(seed_check=out)


MASKED = {"S2", "S3", "S4", "S6"}


def step_table(eng, fix, setting, prefixes):
    reqs, mapping = C.batch_copies(prefixes)
    if setting in MASKED:
        p = sp(fix, setting, max_new_tokens=1)
        if p["top_k"] in (-1, None):
            p["top_k"] = fix.V  # finite top_k needed by the sampling mask (scheduler.py 2957-2967); same set
        # Engine.generate() has no sampling-mask arguments (srt/entrypoints/engine.py 411-463): build the request
        # object it builds and run it the same way (engine.py 473-517)
        from sglang.srt.managers.io_struct import GenerateReqInput
        obj = GenerateReqInput(input_ids=[fix.PROMPT + list(g) for g in reqs], sampling_params=p,
                               return_sampling_mask=True, sampling_logprobs_mode="support")
        o = eng.loop.run_until_complete(eng.tokenizer_manager.generate_request(obj, None).__anext__())
        rows = []
        for x in o:
            row = np.zeros(fix.V)
            mi = x["meta_info"]
            mask, lps = mi["output_token_sampling_mask"][0], mi["output_token_sampling_logprobs"][0]
            for tok, lp in zip(mask, lps):
                row[int(tok)] = np.exp(lp)
            rows.append(row)
    else:
        o = eng.generate(input_ids=[fix.PROMPT + list(g) for g in reqs], sampling_params=sp(fix, setting, max_new_tokens=1),
                         return_logprob=True, top_logprobs_num=fix.V)
        rows = []
        for x in o:
            row = np.zeros(fix.V)
            for lp, tok, *_ in x["meta_info"]["output_top_logprobs"][0]:
                row[int(tok)] = np.exp(lp)
            rows.append(row)
    return np.stack(rows), mapping


def penalised_reference(prefixes, setting):
    """fp64 S5 distribution as SGLang samples it after generating the prefix: the repetition penalty on the
    generated tokens only (positive logits divided, negative multiplied), then softmax(z / T)."""
    import rnp
    m = rnp.Llama64("target")
    rp, T = setting.get("repetition_penalty", 1.0), setting.get("temperature", 1.0)
    rows = []
    for g in prefixes:
        z = m.logits(C.FA.PROMPT + list(g)).astype(np.float64)
        for t in set(g):
            z[t] = z[t] / rp if z[t] > 0 else z[t] * rp
        rows.append(rnp._softmax(z / T))
    return np.stack(rows)


def reference_for(backend, settings_fa=None, settings_fb=None):
    """R_eng on the non-speculative engine for one sampling backend: the default (FlashInfer) for every running
    setting, and the PyTorch backend for S4 and S6, the settings FR9 item 1 may rescue with it (FR12). Stage A's
    gate (FR13) passes the settings its running set needs."""
    import rnp
    import rnp_full  # noqa: F401
    dist = "sglang" if backend is None else "sglang_torch"
    if settings_fa is None:
        settings_fa = list(C.STOCHASTIC) if backend is None else ["S4", "S6"]
    if settings_fb is None:
        settings_fb = list(C.FB_SETTINGS) if backend is None else ["S6"]
    tables = os.environ.get("D40_P0B_OUT", "/tmp/p0b") + "_tables"
    tag = backend or "default"
    out = {}
    for fname, settings in (("FA", settings_fa), ("FB", settings_fb)):
        if not settings:
            continue
        fix = C.fixture(fname)
        try:
            eng = engine(fix, "NULL", 0, **({} if backend is None else dict(sampling_backend=backend)))
        except Exception as e:  # noqa: BLE001
            out[f"{fname}/{tag}/load"] = dict(error=f"{type(e).__name__}: {e}"[:300])
            continue
        try:
            for s in settings:
                key = f"{fname}/{s}/{tag}"
                try:
                    if fname == "FA":
                        pre, ref = C.fa_prefixes(), rnp.step_table(dist, C.SETTINGS[s])
                    else:
                        pre, fref = C.fb_prefixes(dist, s)
                        ref = np.stack([fref.step(tuple(g)) for g in pre])
                    rows, mapping = step_table(eng, fix, s, pre)
                    first, dev = C.first_rows(rows, mapping, len(pre))
                    out[key] = dict(C.delta_report(rows, ref, mapping), copy_dev=dev,
                                    path="sampling_mask" if s in MASKED else "logprob")
                    if s == "S5":
                        # a prefix scored as a prompt is never penalised (penalties see output tokens only), so the
                        # row above only checks the unpenalised path; the distance to the distribution S5 sampling
                        # actually uses (penalty on the generated prefix) is what makes S5 REFERENCE_UNRESOLVED (FR12)
                        pen = penalised_reference(pre, C.SETTINGS[s])
                        out[key]["vs_output_penalised"] = C.delta_report(rows, pen, mapping)
                        out[key]["note"] = "prompt-scored: the penalty is not applied; see vs_output_penalised"
                    C.save_npz(tables, f"sglang_{fname}_{s}_{tag}.npz", eng=first, np=ref)
                except Exception as e:  # noqa: BLE001
                    out[key] = dict(error=f"{type(e).__name__}: {e}"[:300])
                C.phase0_checkpoint(out)
        finally:
            eng.shutdown()
    return dict(reference=out)


def reference():
    return reference_for(None)


def reference_pytorch():
    return reference_for("pytorch")


def throughput():
    out = {}
    for method in ("NULL", "SG1@tgt", "SG3s"):
        try:
            eng = engine(C.FA, method, 1)
        except Exception as e:  # noqa: BLE001
            out[method] = dict(error=f"load: {type(e).__name__}: {e}"[:200])
            continue
        try:
            t = time.time()
            eng.generate(input_ids=[C.PROMPT_TP] * C.CHUNK, sampling_params=sp(C.FA, "S0"))
            out[f"FA/{method}/S0"] = round(C.CHUNK / (time.time() - t), 1)
        finally:
            eng.shutdown()
    return dict(throughput_seq_per_s=out)


def probes():
    return [*smoke_probes(), ("seed_check", seed_check), ("reference", reference),
            ("reference_pytorch", reference_pytorch), ("throughput", throughput)]
