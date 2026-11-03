"""FR22 E2, engine (b): vLLM 0.30.0 `draft_model` with probabilistic drafts (VG2's path), F-A, L4.

`vllm.v1.sample.rejection_sampler.rejection_sample` (a module global called from `RejectionSampler.forward` after the
target logits are processed) is replaced by a loop that runs `verifier.verify` per request on CPU copies of the draft
rows and the processed target rows; the engine's own bonus token is used unless the mode redraws it (M6). The engine
core runs in this process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set by p0b_vllm), so the patch reaches it. One engine
serves every mode. Cells: M0, M1-M4, M6 at S0 and S2 (N = 5·10⁴ each, chunks of 5,000) and at G (1,000 runs each).

    python vllm_e2.py --out DIR [--n 50000] [--n-greedy 1000]
"""
import argparse
import gzip
import json
import os
import pathlib
import sys
import time
import zlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "code")]

os.environ["TRITON_F32_DEFAULT"] = "ieee"  # FR13: Triton fp32 dots at IEEE precision, as Stage A
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.environ.get("D40_WORK", "/tmp"), "triton_cache_ieee"))
import p0b_vllm as E  # noqa: E402  (sets the vLLM environment before vllm is imported)
import p0b_common as C  # noqa: E402

MODES = ("M0", "M1", "M2", "M3", "M4", "M6")
CHUNK = 5000
STATE = dict(mode="M0", trace=None, gen=None)


def seed_of(mode, setting, chunk):
    return 8_000_000 + zlib.crc32(f"E2/vllm/{mode}/{setting}".encode()) % 10**6 * 100 + chunk


def install():
    import inspect
    import torch
    import vllm.v1.sample.rejection_sampler as rs
    import verifier as V
    PH = -1  # rs.PLACEHOLDER_TOKEN_ID is a triton constexpr
    orig = rs.rejection_sample
    sig = inspect.signature(orig)

    def rejection_sample(*args, **kwargs):
        a = sig.bind(*args, **kwargs).arguments  # by name, whatever the call's argument order
        draft_token_ids, num_draft_tokens, max_spec_len = a["draft_token_ids"], a["num_draft_tokens"], a["max_spec_len"]
        cu_num_draft_tokens, draft_probs, target_logits = a["cu_num_draft_tokens"], a["draft_probs"], a["target_logits"]
        bonus_token_ids, sampling_metadata = a["bonus_token_ids"], a["sampling_metadata"]
        if not (sampling_metadata.all_greedy or sampling_metadata.all_random):
            return orig(*args, **kwargs)  # engine-internal dummy batches only; E2 runs one setting per batch
        greedy = bool(sampling_metadata.all_greedy)
        if not greedy:
            assert draft_probs is not None, "VG2 must draft probabilistically"
        P_all = target_logits.float().softmax(-1).cpu()
        Q_all = draft_probs.float().cpu() if draft_probs is not None else None
        d_all = draft_token_ids.cpu().tolist()
        bonus = bonus_token_ids.view(-1).cpu().tolist()
        ends = cu_num_draft_tokens.cpu().tolist()
        outs = getattr(sampling_metadata, "output_token_ids", None)
        out = torch.full((len(num_draft_tokens), max_spec_len + 1), PH, dtype=torch.int32)
        for i, n in enumerate(num_draft_tokens):
            e = ends[i]
            s = e - n
            if n == 0:
                out[i, 0] = bonus[i]
                continue
            Q = Q_all[s:e] if Q_all is not None else torch.nn.functional.one_hot(
                torch.tensor(d_all[s:e]), P_all.shape[-1]).float()
            prefix = list(outs[i]) if outs is not None and i < len(outs) else None
            emitted, _ = V.verify(d_all[s:e], Q, P_all[s:e], STATE["mode"], STATE["gen"], bonus_token=bonus[i],
                                  trace=STATE["trace"], prefix=prefix, greedy=greedy)
            out[i, :len(emitted)] = torch.tensor(emitted, dtype=torch.int32)
        return out.to(target_logits.device)

    rs.rejection_sample = rejection_sample


def run_cell(m, fix, mode, setting, n, out):
    import torch
    import verifier as V
    path = pathlib.Path(out) / f"vllm_{mode}_{setting}.json.gz"
    if path.exists():
        print("have", path.name, flush=True)
        return
    STATE["mode"] = mode
    STATE["trace"] = V.Trace(keep=1000)
    STATE["gen"] = torch.Generator().manual_seed(seed_of(mode, setting, 99))
    seqs, t0, counters = [], time.time(), []
    for c in range((n + CHUNK - 1) // CHUNK):
        k = min(CHUNK, n - c * CHUNK)
        seeds = [seed_of(mode, setting, c) * 10_000 + j for j in range(k)]
        before = E.counters(m)
        seqs += [list(s) for s in E.generate(m, fix, setting, seeds)]
        counters.append(E.delta(E.counters(m), before))
        if len(STATE["trace"].verifs) >= 1000:
            STATE["trace"].keep = 0  # verifications kept from the first chunks only
    tr = STATE["trace"].dump()
    rec = dict(engine="vllm", mode=mode, setting=setting, n=len(seqs), seconds=round(time.time() - t0, 1),
               spec_counters=counters, seqs=seqs, trace=tr,
               effect_ok=(mode == "M0") or tr["effect"] > 0)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    print(mode, setting, "n", len(seqs), "verifs", tr["n_verifications"], "effect", tr["effect"],
          f"{rec['seconds'] / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--n-greedy", type=int, default=1000)
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    E.runner(1)
    fix = C.fixture("FA")
    m = E.llm(fix, "VG2")
    install()  # after the engine is built: its profiling run uses mixed dummy sampling metadata
    assert E.runner_used(m) is False, "VG2 must run on the V1 runner"
    try:
        for setting, n in (("S0", a.n), ("S2", a.n), ("G", a.n_greedy)):
            for mode in MODES:
                try:
                    run_cell(m, fix, mode, setting, n, a.out)
                except Exception as e:  # noqa: BLE001  (a failed cell is recorded; the rest still run)
                    print("CELL FAILED", mode, setting, type(e).__name__, str(e)[:500], flush=True)
                    (pathlib.Path(a.out) / f"vllm_{mode}_{setting}.FAILED").write_text(f"{type(e).__name__}: {e}\n")
    finally:
        E.free(m)


if __name__ == "__main__":
    main()
