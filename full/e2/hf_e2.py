"""FR22 E2, engine (a): HF transformers 5.18.0 assisted generation (H1's path), CPU.

Every cell is H1 on F-A at S0 or S2 with `transformers.generation.utils._speculative_sampling` replaced by
`verifier.verify` in one mode (M0 clean, M1-M4, M6). N = 5·10⁴ per cell, in chunks of 5,000 drawn by a spawn pool.
The trace keeps the first 1,000 verifications of chunk 0; calibration data and the effect count cover every chunk.
The greedy detector is `NOT_APPLICABLE` here: with do_sample=False HF compares argmaxes inline in `_assisted_decoding`
and never calls `_speculative_sampling`, so no mutation reaches greedy runs without copying that method.

    python hf_e2.py --out DIR [--workers N] [--n 50000]
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
import zlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "code")]

SETTINGS = ("S0", "S2")
CHUNK = 5000


def seed_of(mode, setting, chunk):
    return 7_000_000 + zlib.crc32(f"E2/hf/{mode}/{setting}".encode()) % 10**6 * 100 + chunk


def task(t):
    mode, setting, chunk, n, out = t
    import torch
    torch.set_num_threads(1)
    import transformers.generation.utils as U
    import p0b_hf as E
    import p0b_common as C
    import verifier as V
    path = pathlib.Path(out) / f"hf_{mode}_{setting}_c{chunk:02d}.json.gz"
    if path.exists():
        return str(path)
    a = E.arm("H1", setting)
    L = len(C.FA.PROMPT)
    seed = seed_of(mode, setting, chunk)
    gen = torch.Generator().manual_seed(seed + 1)
    tr = V.Trace(keep=1000 if chunk == 0 else 0)

    def ss(candidate_input_ids, candidate_logits, candidate_length, new_logits, assistant_ensemble_weight=None):
        assert assistant_ensemble_weight is None
        K = int(candidate_length)
        ids = candidate_input_ids[0]
        d = ids[-K:].tolist()
        Q = candidate_logits[0, :K].float().softmax(-1)
        P = new_logits[0, :K].float().softmax(-1)
        b = new_logits[0, K].float().softmax(-1)
        emitted, n_acc = V.verify(d, Q, P, mode, gen, bonus_row=b, trace=tr, prefix=ids[L:-K].tolist())
        return torch.tensor([emitted], dtype=ids.dtype, device=ids.device), torch.tensor(n_acc)

    orig, U._speculative_sampling = U._speculative_sampling, ss
    t0 = time.time()
    try:
        seqs = a.draw(n, seed)
    finally:
        U._speculative_sampling = orig
    rec = dict(engine="hf", mode=mode, setting=setting, chunk=chunk, n=len(seqs), seed=seed,
               seconds=round(time.time() - t0, 1), seqs=[list(s) for s in seqs], trace=tr.dump())
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--modes", default=",".join(("M0", "M1", "M2", "M3", "M4", "M6")))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(m, s, i, min(CHUNK, a.n - i * CHUNK), a.out)
             for i in range((a.n + CHUNK - 1) // CHUNK) for m in a.modes.split(",") for s in SETTINGS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, len(tasks))) as pool:
        for k, p in enumerate(pool.imap_unordered(task, tasks), 1):
            print(f"{k}/{len(tasks)} {p} {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
