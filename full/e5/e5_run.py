"""FR22 E5 on TensorRT-LLM v1.4.0rc1 (one-model draft-target, TRTLLM attention, bf16, L4).

    python e5_run.py --out DIR --part deep|drift

- deep: `repro_v14.run_arm` for `spec` and `spec_rs` (each in its own process, as FR21), with the draft replaced by
  `close_draft.SIGMA`'s close draft, so that emitted positions 1 and 2 are well populated. Analysis: FR21's
  calibration test (with the FR22 E1 fix) on DIR/deep.
- drift: one process with the speculative engine (`spec_rs`, T = 1), sequences drawn one request at a time so each
  verification's committed prefix is known (the recorder's entries are matched to the request's output tokens, and
  checked against them). Then the non-speculative engine gives full-vocabulary logprobs at the same prefixes
  (max_tokens = 1, logprobs = V). Writes DIR/drift.json with, per prefix, the speculative row, the decode row and the
  fp64 row of `exact_ref`.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
V14 = HERE.parents[1] / "drafts" / "trtllm_v14"
sys.path.insert(0, str(V14))
sys.path.insert(0, str(V14 / "analysis"))
import repro_v14 as R  # noqa: E402

N_DRIFT = 1000


def make_models(out):
    import numpy as np  # noqa: F401
    import torch
    from safetensors.torch import save_file
    import close_draft as CD
    R.make_model(out / "target", seed=1)
    d = out / "draft"
    d.mkdir(parents=True, exist_ok=True)
    w = CD.draft_weights(CD.SIGMA)
    save_file({k: torch.from_numpy(v.astype("float32")).to(torch.bfloat16) for k, v in w.items()},
              str(d / "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "generation_config.json"):
        (d / f).write_text((out / "target" / f).read_text())


def drift(out):
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    import numpy as np
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig
    kw = dict(model=str(out / "target"), skip_tokenizer_init=True, dtype="bfloat16", attn_backend="TRTLLM",
              cuda_graph_config=None, disable_overlap_scheduler=True,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3), max_num_tokens=1024, max_batch_size=128)
    rec = R.Verifications()
    m = LLM(**kw, speculative_config=DraftTargetDecodingConfig(max_draft_len=3, speculative_model=str(out / "draft"),
                                                               use_rejection_sampling=True))
    R._set_arm_rng_seed(m, 7000)
    rows, mismatch = {}, 0
    try:
        for _ in range(N_DRIFT):
            k0 = len(rec.calib)
            o = m.generate([R.PROMPT], SamplingParams(max_tokens=R.MAX_NEW, end_id=R.EOS, pad_id=R.EOS, temperature=1.0))
            seq = list(o[0].outputs[0].token_ids)
            pos = 1  # the first output token comes from the context step
            for emitted, tp in rec.calib[k0:]:
                if seq[pos:pos + len(emitted)] != list(emitted)[: max(0, len(seq) - pos)]:
                    mismatch += 1
                    break
                for j in range(len(emitted)):
                    if pos + j > len(seq):
                        break
                    rows.setdefault(tuple(seq[:pos + j]), []).append(tp[j])
                pos += len(emitted)
    finally:
        m.shutdown()
    (out / "drift_spec.json").write_text(json.dumps(dict(mismatch=mismatch, rows={",".join(map(str, g)): r
                                                                                for g, r in rows.items()})))
    print("spec rows at", len(rows), "prefixes; mismatches", mismatch, flush=True)
    base = LLM(**kw)
    prefixes = sorted(rows)
    dec = {}
    try:
        for g in prefixes:
            o = base.generate([R.PROMPT + list(g)], SamplingParams(max_tokens=1, end_id=R.EOS, pad_id=R.EOS,
                                                                   temperature=1.0, logprobs=R.V))
            try:
                lp = o[0].outputs[0].logprobs[0]
                row = np.zeros(R.V)
                for t, v in lp.items():
                    row[int(t)] = np.exp(getattr(v, "logprob", v))
                dec[g] = row.tolist()
            except Exception as e:  # noqa: BLE001  recorded, never fatal: the spec rows are already saved
                dec[g] = f"{type(e).__name__}: {e}; logprobs={o[0].outputs[0].logprobs!r}"[:400]
    finally:
        base.shutdown()
    import exact_ref as X
    t64 = X.Model(1)
    res = []
    for g in prefixes:
        z = t64.logits(R.PROMPT + list(g))
        p64 = np.exp(z - z.max())
        res.append(dict(prefix=list(g), n=len(rows[g]), spec_rows=rows[g][:20], decode_row=dec[g],
                        fp64_row=(p64 / p64.sum()).tolist()))
    (out / "drift.json").write_text(json.dumps(dict(verification_mismatch=mismatch, prefixes=res)))
    print("drift prefixes", len(res), "mismatch", mismatch, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("deep", "drift"), required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve() / a.part
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "draft" / "model.safetensors").exists():
        make_models(out)
    if a.part == "drift":
        return drift(out)
    env = dict(os.environ, REPRO_ATTN="TRTLLM")
    for arm in ("spec", "spec_rs"):
        if (out / f"{arm}.json").exists():
            continue
        with open(out / f"{arm}.log", "w") as f:
            rc = subprocess.run([sys.executable, str(V14 / "repro_v14.py"), "--out", str(out), "--arm", arm],
                                stdout=f, stderr=subprocess.STDOUT, env=env, timeout=3600).returncode
        print(arm, "exit", rc, flush=True)


if __name__ == "__main__":
    main()
