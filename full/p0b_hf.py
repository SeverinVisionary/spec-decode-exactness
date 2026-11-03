"""Phase 0b probes for HF transformers 5.18.0 on CUDA (HG1 = pilot H1; FR9 item 7).

Runs on the T4 (where Stage A draws HF samples, FR9 item 5) and on the L4 (F1 install record).
- smoke: HG1 and NULL x S0-S6, G, at most 100 sequences each, discarded unread;
- seed_check: HF seeds are chunk-level (FR9 item 3): one stream seed repeats a chunk exactly, the next
  seed changes it;
- effect: P1-P6 (§5.3: P1, P2, P5 at S2; P3 at S2 and S4; P4 at S6; P6 at S0) each change the quantity
  they patch on the first batch, against the unpatched run with the same stream seed;
- reference: R_eng (generate's processed scores, one new token per prefix) at batch 1 and at a batch
  of 10^3 copies (the null's production batch), with δ_num against R_np;
- throughput on PROMPT_TP.
Assisted generation in 5.18.0 is batch 1 only (`generation/utils.py:3999`, "assisted generate is only
supported for batch_size = 1"), so HG1 cells are drawn one sequence per call; the null batches 10^3.
"""
import os
import time

import p0b_common as C

DEV = None
N_EFFECT = 20  # sequences per effect-assertion run; P5 needs a rejection, hence a crop, among them
NULL_BATCH = 1000


def device():
    global DEV
    if DEV is None:
        import torch
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
    return DEV


def arm(method, setting, prompt=None):
    import torch
    import hf_arms as H
    a = H.Arm(method, setting)
    a.target.to(device())
    if a.assistant is not None:
        a.assistant.to(device())
    a.prompt = torch.tensor([prompt or C.FA.PROMPT], device=device())
    return a


def smoke():
    out = {}
    for method in ("HG1", "NULL"):
        for s in C.SETTINGS:
            cell = f"FA/{method}/{s}"
            try:
                a = arm("H1" if method == "HG1" else "NULL", s)
                t = time.time()
                seed = C.seed_base("P0b", cell)
                seqs = a.draw(C.N_SMOKE, seed) if method == "HG1" else a.draw_null_batched(C.N_SMOKE, seed, batch=C.N_SMOKE)
                out[cell] = dict(seq_per_s=round(C.N_SMOKE / (time.time() - t), 1), stream_seed=seed,
                                 **C.FA.shape_check(seqs))
            except Exception as e:  # noqa: BLE001
                out[cell] = dict(error=f"{type(e).__name__}: {e}"[:300])
            C.phase0_checkpoint(out)
    return dict(smoke=out, device=device())


def seed_check():
    out = {}
    for method in ("HG1", "NULL"):
        cell = f"FA/{method}/S0"
        a = arm("H1" if method == "HG1" else "NULL", "S0", prompt=C.PROMPT_TP)  # no grid-cell draws
        b = C.seed_base("P0b", cell)
        draw = (lambda s: a.draw(20, s)) if method == "HG1" else (lambda s: a.draw_null_batched(20, s, batch=20))
        x, y, z = draw(b), draw(b), draw(b + 1)
        out[cell] = dict(same_seed_repeats=C.same_draws(x, y), next_seed_differs=not C.same_draws(x, z))
        del x, y, z
    return dict(seed_check=out)


# ------------------------------------------------------------------ effect assertions (§5.3)

class Capture:
    """Wraps generation.utils._speculative_sampling and keeps (q logits, p logits, acceptance ratio) per call.
    Installed before a control, so a control that wraps the same function sees it as `orig`, and the
    capture holds what the verifier actually received."""

    def __enter__(self):
        import transformers.generation.utils as U
        self.U, self.orig, self.calls = U, U._speculative_sampling, []

        def cap(cand_ids, cand_logits, gamma, new_logits, **kw):
            import torch
            q = cand_logits.softmax(-1)
            p = new_logits.softmax(-1)
            idx = cand_ids[:, -gamma:]
            qi = q[:, torch.arange(gamma), idx].squeeze(0, 1)
            pi = p[:, torch.arange(gamma), idx].squeeze(0, 1)
            w = kw.get("assistant_ensemble_weight")
            ratio = pi / qi if w is None else 1.0 - w + w * (pi / qi)
            self.calls.append(dict(ids=idx.detach().cpu().clone(), q=cand_logits.detach().double().cpu().clone(),
                                   p=new_logits.detach().double().cpu().clone(),
                                   ratio=ratio.detach().double().cpu().clone()))
            return self.orig(cand_ids, cand_logits, gamma, new_logits, **kw)

        U._speculative_sampling = cap
        return self

    def __exit__(self, *exc):
        self.U._speculative_sampling = self.orig


def maxdiff(a, b):
    import torch
    if a.shape != b.shape:
        return float("inf")
    big = 1e30
    fa = torch.nan_to_num(a, nan=0.0, posinf=big, neginf=-big)
    fb = torch.nan_to_num(b, nan=0.0, posinf=big, neginf=-big)
    return float((fa - fb).abs().max())


def compare(base, pat):
    """Matched comparison on the FIRST verifier call only: same prompt and stream seed, so it is the same invocation
    in both runs. P1 compares the draft q at the first drafted position (its prefix is the prompt in both runs), P2
whether the drafted ids changed;
    P3/P4/P6 need the same candidate ids (the draft is unpatched), else the comparison is an alignment failure."""
    if not base or not pat:
        return dict(aligned=False, why="no verifier call", q=None, p=None, ratio=None)
    x, y = base[0], pat[0]
    same_ids = x["ids"].shape == y["ids"].shape and bool((x["ids"] == y["ids"]).all())
    r = dict(aligned=True, same_candidate_ids=same_ids, candidate_ids_changed=int(not same_ids),
             q_first_position=maxdiff(x["q"][:, 0], y["q"][:, 0]))
    if same_ids and x["p"].shape == y["p"].shape:
        r.update(p=maxdiff(x["p"], y["p"]), ratio=maxdiff(x["ratio"], y["ratio"]))
    else:
        r.update(p=None, ratio=None)
    return r


def control_ctx(control, a, extra):
    """The control's patch, active while `a` draws (P6 is the shipped ensemble option on the arm itself)."""
    import controls as K
    if control == "P1":
        return K.p1_raw_q()
    if control == "P2":
        return K.p2_greedy_draft_q_as_sampled()
    if control == "P3":
        return K.p3_processors_bonus_only(a)
    if control == "P4":
        return K.p4_topp_before_topk(a)
    if control == "P5":
        return _p5_counting(K, extra)
    return _null_ctx()


def control_arm(control, setting):
    return arm("P6" if control == "P6" else "H1", setting)  # P6 is H1 with the shipped ensemble option


def effect(control, setting, seed=None, a=None):
    """Stage A (FR13) passes each control chunk's stream seed and its drawing arm."""
    seed = C.seed_base("P0b", f"FA/{control}/{setting}") if seed is None else seed
    a = control_arm(control, setting) if a is None else a
    if control == "P2":
        return _effect_p2(setting, seed, a)
    base_arm = arm("H1", setting)
    with Capture() as c0:
        base_arm.draw(N_EFFECT, seed)
    with Capture() as cs:  # sham: the unpatched run again; it must match exactly, or "changed" means nothing
        base_arm.draw(N_EFFECT, seed)
    sham = compare(c0.calls, cs.calls)  # must be aligned, same ids, all differences 0
    extra = {}
    with Capture() as c1:
        with control_ctx(control, a, extra):
            a.draw(N_EFFECT, seed)
    d = compare(c0.calls, c1.calls)
    # P2 (greedy draft, q used as if sampled) leaves the draft logits unchanged; it changes which ids are drafted
    quantity = {"P1": "q_first_position", "P2": "candidate_ids_changed", "P3": "p", "P4": "p", "P5": "altered_crops",
                "P6": "ratio"}[control]
    sham_clean = bool(sham.get("aligned") and sham.get("same_candidate_ids")
                      and all(sham[k] == 0 for k in ("q_first_position", "p", "ratio")))
    if control == "P5":  # the patched quantity is the target cache length after a crop, counted at the crop itself
        changed = sham_clean and extra.get("altered_crops", 0) > 0
    else:
        v = d.get(quantity)
        changed = sham_clean and d.get("aligned") and v is not None and v > 0
    return dict(control=control, setting=setting, stream_seed=seed, patched_quantity=quantity,
                max_abs_diff=d, sham=sham, sham_clean=sham_clean, changed=bool(changed), **extra)


def _effect_p2(setting, seed, a):
    """P2 (greedy draft, q used as if sampled) changes the drafted ids only when the sampled draft differs from the
    argmax one, which a single seeded first call misses with probability about 0.18 at S2 (FR13). So each of N_EFFECT
    sequences is drawn alone with seed + i, and its first verifier call is compared: the sham must match on every one,
    and at least one must draft different ids (failure probability about 0.18^20)."""
    base_arm = arm("H1", setting)
    per, sham_clean, changed = [], True, 0
    for i in range(N_EFFECT):
        with Capture() as c0:
            base_arm.draw(1, seed + i)
        with Capture() as cs:
            base_arm.draw(1, seed + i)
        with Capture() as c1:
            with control_ctx("P2", a, {}):
                a.draw(1, seed + i)
        sham = compare(c0.calls, cs.calls)
        clean = bool(sham.get("aligned") and sham.get("same_candidate_ids")
                     and all(sham[k] == 0 for k in ("q_first_position", "p", "ratio")))
        d = compare(c0.calls, c1.calls)
        sham_clean &= clean
        changed += int(bool(d.get("aligned")) and d.get("candidate_ids_changed") == 1)
        per.append(dict(seed=seed + i, sham_clean=clean, candidate_ids_changed=d.get("candidate_ids_changed")))
    return dict(control="P2", setting=setting, stream_seed=seed, patched_quantity="candidate_ids_changed",
                per_sequence=per, sequences_changed=changed, sham_clean=sham_clean,
                changed=bool(sham_clean and changed > 0))


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _p5_counting:
    """P5 wrapped so the run also counts the crops it altered (the patched quantity is the cache length)."""

    def __init__(self, K, extra):
        self.K, self.extra = K, extra

    def __enter__(self):
        import transformers.cache_utils as CU
        self.ctx = self.K.p5_crop_one_too_few()
        self.ctx.__enter__()
        self.CU, self.patched = CU, CU.DynamicCache.crop
        extra = self.extra
        extra["altered_crops"] = 0
        inner = self.patched

        def crop(cache, max_length):
            before = cache.get_seq_length()
            r = inner(cache, max_length)
            if max_length < 0 and cache.get_seq_length() != before + max_length:
                extra["altered_crops"] += 1
            return r

        CU.DynamicCache.crop = crop
        return self

    def __exit__(self, *exc):
        self.CU.DynamicCache.crop = self.patched
        return self.ctx.__exit__(*exc)


EFFECTS = [("P1", "S2"), ("P2", "S2"), ("P3", "S2"), ("P3", "S4"), ("P4", "S6"), ("P5", "S2"), ("P6", "S0")]


def effects():
    out = {}
    for control, s in EFFECTS:
        try:
            out[f"{control}_{s}"] = effect(control, s)
        except Exception as e:  # noqa: BLE001
            out[f"{control}_{s}"] = dict(error=f"{type(e).__name__}: {e}"[:300], changed=False)
        C.phase0_checkpoint(out)
    return dict(effect=out, all_changed=all(v.get("changed") for v in out.values()))


# ------------------------------------------------------------------ R_eng (§5.1) and δ_num

def step_table(setting, batch):
    """Every row of every batch (batch copies of one prefix per generate call), with its prefix index."""
    import numpy as np
    import torch
    import hf_arms as H
    m = H.load("target").to(device())
    kw = dict(H.SETTINGS[setting])
    rows, mapping = [], []
    with torch.no_grad():
        for j, g in enumerate(C.fa_prefixes()):
            ids = torch.tensor([C.FA.PROMPT + g] * batch, device=device())
            o = m.generate(input_ids=ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, **kw)
            rows.append(torch.softmax(o.scores[0].double(), -1).cpu().numpy())
            mapping += [j] * batch
    return np.concatenate(rows), mapping


def reference():
    import rnp
    out = {}
    tables = os.environ.get("D40_P0B_OUT", "/tmp/p0b") + "_tables"
    for s in C.STOCHASTIC:
        ref = rnp.step_table("hf", {k: v for k, v in C.SETTINGS[s].items()})
        row = {}
        for tag, batch in (("batch1", 1), ("batch1000", NULL_BATCH)):
            rows, mapping = step_table(s, batch)
            first, dev = C.first_rows(rows, mapping, len(ref))
            row[tag] = dict(C.delta_report(rows, ref, mapping), within_batch_row_dev=dev)
            C.save_npz(tables, f"hf_FA_{s}_{tag}.npz", eng=first, np=ref)
        out[s] = row
        C.phase0_checkpoint(out)
    return dict(reference=out)


def throughput():
    out = {}
    a = arm("H1", "S0", prompt=C.PROMPT_TP)
    t = time.time()
    a.draw(200, 1)
    out["HG1_S0_batch1"] = round(200 / (time.time() - t), 1)
    n = arm("NULL", "S0", prompt=C.PROMPT_TP)
    t = time.time()
    n.draw_null_batched(10_000, 1, batch=NULL_BATCH)
    out["NULL_S0_batch1000"] = round(10_000 / (time.time() - t), 1)
    return dict(throughput_seq_per_s=out)


def probes():
    return [("smoke", smoke), ("seed_check", seed_check), ("effects", effects), ("reference", reference),
            ("throughput", throughput)]
