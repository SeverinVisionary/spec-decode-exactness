"""FR17: trace TensorRT-LLM 1.2.1 TG1v's drafter to settle C4 (which drafting step each `py_draft_logits` row is).

    python c4_probe.py --out out/d40_full/c4_probe      # on an L4 (Colab)

Engine options, fixture, fp32 hook and drafter hook are Stage A/B's (`p0b_trtllm.llm`, FR8, FR11). Two arms, each
in its own process under phase0's fatal-marker watchdog:
- P1 trace: TG1v at S0 with real coins, 200 one-request `generate` calls (a draft batch then holds one request,
  so every row of the draft model's output logits belongs to it). Recorded per drafting round:
  - every `ModelDrafter.forward_draft_model` call: the draft request's kind (context/generation), its tokens,
    context position and chunk size, and the full output logits;
  - every `ModelDrafter.process_decoded_tokens` call: the draft position, the token just drafted, and the
    `generation_logits` the call then assigns to `py_draft_logits`;
  - every verification call (`repro_trtllm.Recorder`): draft tokens, `py_draft_logits`, `py_target_probs`,
    number accepted.
- P2 draft alone: the draft model loaded as an ordinary (non-speculative) TensorRT-LLM model, with the processed
  S0 probabilities captured live (`p0b_trtllm.step_table`) at the first 100 F-A prefixes. This checks the engine's
  draft model against the fp64 draft directly, which the Stage A/B gates did only for the target.
Nothing here compares draws with a reference. `c4_probe_analyze.py` does the analysis on CPU.
"""
import argparse
import os
import pathlib
import sys
import time

import p0b_common as C
import p0b_trtllm as E
import repro_trtllm as RT

N_REQ = 200
N_P2_PREFIXES = 100


def seed(key):
    return C.seed_base("C4", key)


def field(r, name):
    """An int field of an LlmRequest, or -1. The bindings raise RuntimeError, not AttributeError, for a field that
    does not apply in the request's phase (context_chunk_size on a generation request crashed P1's first run)."""
    try:
        return int(getattr(r, name))
    except Exception:  # noqa: BLE001
        return -1


class Trace:
    """Wraps three ModelDrafter methods; records their inputs and outputs while `on`. Copies to host only."""

    def __init__(self):
        from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter
        self.round, self.fwd, self.dec, self.on = -1, [], [], False
        o_prep, o_fwd, o_dec = (ModelDrafter.prepare_draft_tokens, ModelDrafter.forward_draft_model,
                                ModelDrafter.process_decoded_tokens)
        tr = self

        def prepare_draft_tokens(self_, *a, **k):
            tr.round += 1
            return o_prep(self_, *a, **k)

        def forward_draft_model(self_, draft_batch, *a, **k):
            reqs = []
            if tr.on:
                for kind, lst in (("context", draft_batch.context_requests),
                                  ("generation", draft_batch.generation_requests)):
                    for r in lst:
                        reqs.append(dict(kind=kind, rid=int(r.py_request_id), tokens=list(r.get_tokens(0)),
                                         context_current_position=field(r, "context_current_position"),
                                         context_chunk_size=field(r, "context_chunk_size")))
            out = o_fwd(self_, draft_batch, *a, **k)
            if tr.on:
                tr.fwd.append(dict(round=tr.round, is_first=bool(k.get("is_first_draft_token", a[1] if len(a) > 1
                                                                      else None)),
                                   reqs=reqs, logits=RT.tolist(out["logits"])))
            return out

        def process_decoded_tokens(self_, draft_batch, *a, **k):
            if tr.on:
                pos = k.get("draft_position", a[0] if a else None)
                for r in draft_batch.all_requests():
                    try:
                        gl = r.py_result.generation_logits
                        gl = None if gl is None else RT.tolist(gl.reshape(-1, gl.shape[-1]))
                    except Exception as e:  # noqa: BLE001
                        gl = f"error: {type(e).__name__}: {e}"[:200]
                    tr.dec.append(dict(round=tr.round, rid=int(r.py_request_id), draft_position=int(pos),
                                       last_token=int(r.get_last_tokens(0)), tokens=list(r.get_tokens(0)),
                                       generation_logits=gl))
            return o_dec(self_, draft_batch, *a, **k)

        for name, fn in (("prepare_draft_tokens", prepare_draft_tokens), ("forward_draft_model", forward_draft_model),
                         ("process_decoded_tokens", process_decoded_tokens)):
            fn._d40_inference_mode = getattr(getattr(ModelDrafter, name), "_d40_inference_mode", False)
            setattr(ModelDrafter, name, fn)


def p1(out):
    fix = C.FA
    m = E.llm(fix, "TG1v")
    rec, tr = RT.Recorder(), Trace()
    rec_before = rec.before

    def before(req, path, k):  # tag each verification with the drafting round that produced its draft tokens
        ev = rec_before(req, path, k)
        if ev is not None:
            ev["round"] = tr.round
        return ev

    rec.before = before
    try:
        import phase0
        RT.real_coin()
        res = dict(arm="P1", hooks=E.hooks(), environment=phase0.environment(), setting="S0", requests=[])
        t = time.time()
        for i in range(N_REQ):
            E.sampler_of(m)._generator = None
            E.seed_hook(m, seed("P1/S0") + i)
            rec.events, tr.fwd, tr.dec = [], [], []
            rec.on = tr.on = True
            o = m.generate([fix.PROMPT], E.params(fix, "S0", 1))
            rec.on = tr.on = False
            res["requests"].append(dict(i=i, output=list(o[0].outputs[0].token_ids), forwards=tr.fwd,
                                        decoded=tr.dec, verifications=rec.events))
        res["wall_s"] = time.time() - t
        RT.write(out / "P1.json.gz", res)
    finally:
        m.shutdown()


class DraftAsTarget:
    """F-A with the draft model in the target's place, so `p0b_trtllm.llm(..., "NULL")` loads the draft alone."""

    def __init__(self, fix):
        self._fix = fix

    def path(self, name, layout=""):
        return self._fix.path("draft" if name == "target" else name, layout)

    def __getattr__(self, k):
        return getattr(self._fix, k)


def p2(out):
    fix = C.FA
    m = E.llm(DraftAsTarget(fix), "NULL")
    try:
        import phase0
        E.seed_hook(m, seed("P2/S0"))
        pre = C.fa_prefixes()[:N_P2_PREFIXES]
        rows, mapping, strategies, batches = E.step_table(m, fix, "S0", pre)
        first, dev = C.first_rows(rows, mapping, len(pre))
        RT.write(out / "P2.json.gz", dict(arm="P2", hooks=E.hooks(), environment=phase0.environment(), setting="S0",
                                          prefixes=[list(g) for g in pre], probs=first.tolist(), copy_dev=dev,
                                          strategies=strategies, batch_rows=[min(batches), max(batches)]))
    finally:
        m.shutdown()


def run_all(out, arm_timeout=3600, grace=120):
    """As `repro_trtllm.run_all`: each arm in its own process group, killed `grace` s after a fatal log line."""
    import subprocess
    import phase0
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in ("P2", "P1"):  # P2 first: shorter, and a draft-load problem shows before the trace runs
        if (out / f"{name}.json.gz").exists():
            print(name, "already done")
            continue
        log = out / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}.log"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["P1", "P2", "all"], default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.arm == "all":
        return run_all(out)
    out.mkdir(parents=True, exist_ok=True)
    (p1 if a.arm == "P1" else p2)(out)


if __name__ == "__main__":
    sys.exit(main())
