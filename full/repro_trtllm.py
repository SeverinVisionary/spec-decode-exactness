"""FR16: coin-injected minimal reproduction of the TensorRT-LLM 1.2.1 TG1v (two-model draft-target, VANILLA) defect.

    python repro_trtllm.py --out out/d40_full/repro_trtllm      # on an L4 (Colab)

Same engine options, fixture, fp32 hook and drafter hook as Stage A/B (`p0b_trtllm.llm`, FR8, FR11). Every arm wraps
`TorchSampler._process_draft_tokens_rejection_sampling` and `_process_draft_tokens_greedy` to record, per verification
call, the request's tokens, its draft tokens, the shapes and values of `py_draft_logits` and `py_target_probs`, the
path taken and the number accepted. The wrapper only copies tensors to the host; it changes no value.

Arms (FR16):
- A1 coins: `sampling_utils.get_rejected_indices` as in v1.2.1, but its uniform draw replaced by a constant u
  (u = 0: every verified draft is accepted; u = 1: a verified draft is accepted iff p_target >= q_draft). S0 and S2,
  200 requests each.
- A2 hook swap: as A1 at S2, u = 1, with FR11's inference-mode wrapper replaced by v1.2.1's
  `process_decoded_tokens` with the in-place `squeeze_(0)` made out of place. Tests whether the harness hook matters.
- A3 fix: as A1 at S2, u = 1, with `_create_draft_request` passing `exclude_last_generation_logits=False`; then,
  with real coins, F-A draws at N = 10^5 for S0 and S2 (seeds h("R", key)), tested by `repro_analyze.py`.
"""
import argparse
import gzip
import json
import os
import pathlib
import sys
import time

import numpy as np

import p0b_common as C
import p0b_trtllm as E

N_EVENTS = 200
N_FIX = 100_000
CHUNK = 10_000


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") if path.suffix == ".gz" else open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def tolist(t):
    return None if t is None else t.detach().float().cpu().numpy().round(9).tolist()


class Recorder:
    """Wraps the two verification paths of TorchSampler; records inputs and outcome of every call."""

    def __init__(self):
        from tensorrt_llm._torch.pyexecutor import sampler as S
        from tensorrt_llm._torch.pyexecutor.llm_request import get_draft_token_length
        self.S, self.events, self.on = S, [], False
        cls = S.TorchSampler
        self.orig_rs, self.orig_gr = cls._process_draft_tokens_rejection_sampling, cls._process_draft_tokens_greedy
        rec = self

        def rs(self_, request, *args, **kwargs):
            ev = rec.before(request, "rejection_sampling", get_draft_token_length(request))
            return rec.after(ev, request, rec.orig_rs(self_, request, *args, **kwargs))

        def gr(self_, request, *args, **kwargs):
            ev = rec.before(request, "greedy", get_draft_token_length(request))
            return rec.after(ev, request, rec.orig_gr(self_, request, *args, **kwargs))

        cls._process_draft_tokens_rejection_sampling, cls._process_draft_tokens_greedy = rs, gr

    def before(self, req, path, k):
        if not self.on:
            return None
        dl, tp = req.py_draft_logits, req.py_target_probs
        return dict(path=path, tokens_before=list(req.get_tokens(0)), draft_tokens=list(req.py_draft_tokens or []),
                    draft_token_length=int(k), draft_logits_shape=None if dl is None else list(dl.shape),
                    target_probs_shape=None if tp is None else list(tp.shape), draft_logits=tolist(dl),
                    target_probs=tolist(tp))

    def after(self, ev, req, n):
        if ev is not None:
            ev.update(num_accepted=int(n), tokens_after=list(req.get_tokens(0)))
            self.events.append(ev)
        return n


_ORIG = {}


def real_coin():
    """Restore v1.2.1's own `get_rejected_indices` (real uniform draws)."""
    from tensorrt_llm._torch.pyexecutor import sampler as S
    S.get_rejected_indices = _ORIG.setdefault("gri", S.get_rejected_indices)


def inject_coin(u):
    """v1.2.1 `get_rejected_indices` (sampling_utils.py 428-452) with `torch.rand(...)` replaced by the constant u."""
    import torch
    from tensorrt_llm._torch.pyexecutor import sampler as S
    _ORIG.setdefault("gri", S.get_rejected_indices)

    def get_rejected_indices(draft_probs, target_probs, generator, draft_tokens):
        num_draft_tokens = draft_probs.size(0)
        draft_tokens = draft_tokens[:num_draft_tokens]
        token_idx = torch.arange(num_draft_tokens, dtype=torch.int32, device=generator.device)
        draft_tokens_cuda = torch.tensor(draft_tokens, dtype=torch.int32, pin_memory=True).to(
            device=generator.device, non_blocking=True)
        p = draft_probs[token_idx, draft_tokens_cuda]
        q = target_probs.squeeze(0)[token_idx, draft_tokens_cuda]
        accept_probs = torch.minimum(torch.ones((), device=generator.device, dtype=q.dtype), q / p)
        return (torch.full_like(accept_probs, float(u)) > accept_probs).nonzero()

    S.get_rejected_indices = get_rejected_indices


def swap_hook():
    """A2: v1.2.1 `ModelDrafter.process_decoded_tokens` (model_drafter.py 449-489) without FR11's inference-mode
    wrapper, with `generation_logits.squeeze_(0)` replaced by `generation_logits = generation_logits.squeeze(0)`."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter

    def process_decoded_tokens(self, draft_batch, draft_position, cleanup_resources=True):
        new_requests = []
        for req in draft_batch.all_requests():
            target_model_req = self.req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                self.draft_seq_slot_manager.free_resources(req)
                continue
            if req.py_request_id not in self.draft_tokens_accumulator:
                self.draft_tokens_accumulator[req.py_request_id] = [0] * self.max_total_draft_tokens
            self.draft_tokens_accumulator[req.py_request_id][draft_position - 1] = req.get_last_tokens(0)
            generation_logits = req.py_result.generation_logits
            if generation_logits is not None:
                beam_width = generation_logits.size(0)
                assert beam_width == 1, f"expected beam_width=1, got {beam_width}"
                generation_logits = generation_logits.squeeze(0)  # the one change (A2)
                if generation_logits.device.type == 'cpu':
                    generation_logits = generation_logits.to('cuda', non_blocking=True)
            target_model_req.py_draft_logits = generation_logits
            if req.state != LlmRequestState.GENERATION_COMPLETE and draft_position < target_model_req.py_draft_pages_allocated:
                new_requests.append(req)
            else:
                if cleanup_resources:
                    self.draft_seq_slot_manager.free_resources(req)
        return new_requests

    ModelDrafter.process_decoded_tokens = process_decoded_tokens


def apply_fix():
    """A3: v1.2.1 `ModelDrafter._create_draft_request` (model_drafter.py 112-129) with
    `exclude_last_generation_logits=False`: the two-model loop with the overlap scheduler off appends no extra
    iteration, so no last logits chunk should be dropped."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter

    def _create_draft_request(self, request, input_tokens):
        return LlmRequest(
            input_tokens=input_tokens, request_id=request.py_request_id, max_new_tokens=request.py_max_new_tokens,
            sampling_config=request.sampling_config, guided_decoding_params=request.guided_decoding_params,
            target_seq_slot=request.py_seq_slot, return_perf_metrics=request.return_perf_metrics, is_streaming=False,
            exclude_last_generation_logits=False,  # the one change (A3)
            is_draft=True, return_generation_logits=self.sampler.should_provide_draft_probs(request))

    ModelDrafter._create_draft_request = _create_draft_request


def run_events(m, rec, setting, u, seed):
    fix = C.FA
    inject_coin(u)
    E.sampler_of(m)._generator = None
    E.seed_hook(m, seed)
    rec.events, rec.on = [], True
    o = m.generate([fix.PROMPT] * N_EVENTS, E.params(fix, setting, N_EVENTS))
    rec.on = False
    return dict(setting=setting, u=u, seed=seed, events=rec.events, outputs=[list(x.outputs[0].token_ids) for x in o])


def run_draws(m, setting, seed, part_dir):
    """Chunks of 10^4 written as they finish; a rerun keeps finished chunks and redraws the rest from the cell's
    own stream (the stream restarts at h("R", key) + j0, so a chunk's draws do not depend on earlier chunks)."""
    fix, seqs = C.FA, []
    real_coin()
    for j0 in range(0, N_FIX, CHUNK):
        cp = part_dir / f"draws_{setting}_{j0:06d}.json.gz"
        if cp.exists():
            seqs += json.load(gzip.open(cp, "rt"))["seqs"]
            continue
        E.sampler_of(m)._generator = None
        E.seed_hook(m, seed + j0)
        o = m.generate([fix.PROMPT] * CHUNK, E.params(fix, setting, CHUNK))
        chunk = [list(x.outputs[0].token_ids) for x in o]
        write(cp, dict(setting=setting, seed=seed + j0, seqs=chunk))
        seqs += chunk
    return dict(setting=setting, seed=seed, n=len(seqs), seqs=seqs)


def arm(name, out, setup, jobs):
    path = out / f"{name}.json.gz"
    if path.exists():
        print(name, "already done:", path)
        return
    part_dir = out / f"{name}_parts"
    t = time.time()
    setup()
    m = E.llm(C.FA, "TG1v")
    rec = Recorder()
    try:
        import phase0
        res = dict(arm=name, hooks=E.hooks(), environment=phase0.environment(), runs=[])
        for i, (kind, *a) in enumerate(jobs):
            rp = part_dir / f"run{i}.json.gz"
            if rp.exists():
                r = json.load(gzip.open(rp, "rt"))
            else:
                r = run_events(m, rec, *a) if kind == "events" else run_draws(m, *a, part_dir)
                write(rp, r)
            res["runs"].append(r)
            print(name, kind, a, "done", f"{time.time() - t:.0f}s", flush=True)
        res["wall_s"] = time.time() - t
        write(path, res)
    finally:
        m.shutdown()


def run_all(out, arm_timeout=3600, grace=120):
    """Each arm in its own process group (patches are global), with phase0's fatal-marker watchdog: a TensorRT-LLM
    executor that logs a fatal error and stops returning is killed after `grace` s, not at the arm timeout."""
    import signal
    import subprocess
    import phase0
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in ("A1", "A2", "A3"):
        if (out / f"{name}.json.gz").exists():
            print(name, "already done")
            continue
        log = out / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}.log"  # a fresh log per attempt: no stale fatal marker
        with open(log, "w") as f:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--out", str(out), "--arm", name],
                                 start_new_session=True, stdout=f, stderr=subprocess.STDOUT)
            t0, fatal_at, status = time.monotonic(), None, None
            while p.poll() is None:
                time.sleep(10)
                if fatal_at is None and phase0.fatal_marker("trtllm", str(log)):
                    fatal_at = time.monotonic()
                if fatal_at is not None and time.monotonic() - fatal_at > grace:
                    status = "FATAL"
                elif time.monotonic() - t0 > arm_timeout:
                    status = "TIMEOUT"
                if status:
                    phase0.kill_group(p)
                    break
        print(name, "exit", p.returncode, status or "", flush=True)
        print(open(log, errors="replace").read()[-2000:], flush=True)
        if status or p.returncode or not (out / f"{name}.json.gz").exists():
            failed.append(name)
    print("arms failed or incomplete:", failed or "none", flush=True)
    return 1 if failed else 0


def seed(key):
    return C.seed_base("R", key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["A1", "A2", "A3", "all"], default="all")  # one arm per process: patches are global
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.arm == "all":
        return run_all(out)
    elif a.arm == "A1":
        arm("A1", out, lambda: None, [("events", s, u, seed(f"A1/{s}/u{u}")) for s in ("S0", "S2") for u in (0, 1)])
    elif a.arm == "A2":
        arm("A2", out, swap_hook, [("events", "S2", 1, seed("A2/S2/u1"))])
    else:
        arm("A3", out, apply_fix, [("events", "S2", 1, seed("A3/S2/u1")),
                                   ("draws", "S0", seed("A3/S0/draws")), ("draws", "S2", seed("A3/S2/draws"))])


if __name__ == "__main__":
    sys.exit(main())
