"""D-40 full study, Phase 0 (non-evaluative; PREREGISTRATION.md §7): environment, feasibility, smoke.

Runs on a Colab GPU runtime, one engine per fresh runtime:

    python phase0.py --engine vllm|sglang|trtllm|llamacpp|hf --out out/d40_full

Each probe draws at most 100 sequences per arm and checks only:
- crash;
- output shape (outcomes in the F-A outcome space);
- that the speculative path fired (engine counters or logs);
- timing.

Outputs are counted only to check the shape, then discarded unread: no sample is compared with
any reference. Every probe is wrapped, so one failing arm never hides the others.

Each probe runs in its own child process (fresh CUDA state) with a wall-clock limit
(`--probe-timeout`, default 900 s). A probe that hangs is killed with its whole process group and
recorded as `TIMEOUT`, together with the settings it finished. The JSON is rewritten after every
probe, so a stalled or killed run still leaves a record (`"complete": false` until the end).
Some engines log a fatal error and then hang instead of exiting. TensorRT-LLM's executor thread can
die while `generate` waits forever (Phase 0, 2026-10-19). The parent watches the probe log for that
engine's fatal marker (`FATAL_LOG_MARKERS`). It gives the child `FATAL_GRACE_S` to exit on its own,
then kills the group and records `ERROR` with the marker, instead of waiting for the time limit.
Each probe's console output goes to `$D40_WORK/logs/probe_<arm>.log`; for a probe that does not run,
the last 4000 characters are copied into the record as `log_tail` and printed.
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
os.environ.setdefault("D40_WORK", "<runtime>/d40work")

import fixture as fx  # noqa: E402

N_SMOKE = 100
VALID = set(map(tuple, fx.outcomes()))
# G before S6: an engine that dies on S6 (seen with vLLM on CPU) must not take the greedy probe with it
SETTINGS = {
    "S0": dict(temperature=1.0), "S2": dict(temperature=1.0, top_k=3), "G": dict(temperature=0.0),
    "S6": dict(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def environment():
    env = dict(python=platform.python_version(), platform=platform.platform(),
               gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
               compute_cap=sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader"),
               cuda=sh("nvcc --version | tail -1"), harness_git=sh(f"git -C {HERE} rev-parse HEAD"))
    arch = os.path.join(os.environ["D40_WORK"], "llama.cpp", os.environ.get("D40_LCPP_BUILD", "build-cuda"), "d40_cuda_arch.txt")
    if os.path.exists(arch):
        env["llamacpp_cuda_arch"] = open(arch).read().strip()
    lcpp_src = os.path.join(os.environ["D40_WORK"], "llama.cpp")
    if os.path.isdir(lcpp_src):
        env["llamacpp_git"] = sh(f"git -C {lcpp_src} rev-parse HEAD")
    # installed versions from package metadata: importing an engine here would run outside any probe's time limit
    from importlib import metadata
    for mod, dist in (("torch", "torch"), ("transformers", "transformers"), ("vllm", "vllm"), ("sglang", "sglang"),
                      ("tensorrt_llm", "tensorrt-llm"), ("numpy", "numpy"), ("pyarrow", "pyarrow")):
        try:
            env[mod] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[mod] = "not installed"
    return env


def shape_check(seqs):
    seqs = [tuple(int(t) for t in s) for s in seqs]
    cut = []
    for s in seqs:
        if fx.EOS in s:
            s = s[: s.index(fx.EOS) + 1]
        cut.append(s[: fx.MAX_NEW])
    bad = sum(1 for s in cut if s not in VALID)
    n = len(cut)
    del seqs, cut  # discarded unread (PREREGISTRATION.md §7)
    return dict(n=n, out_of_space=bad)


def probe(name, fn):
    t0 = time.time()
    try:
        r = fn() or {}
        return dict(arm=name, status="RUNS", wall_s=round(time.time() - t0, 2), **r)
    except Exception as e:  # noqa: BLE001
        return dict(arm=name, status="ERROR", error=f"{type(e).__name__}: {e}"[:500],
                    traceback=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 2))


PARTIAL = None  # child mode: per-setting progress is written here, so a timeout keeps what finished


def checkpoint(res):
    if PARTIAL:
        json.dump(res, open(PARTIAL, "w"), default=str)


# ------------------------------------------------------------------ vLLM (GPU)

def vllm_probes():
    out = []
    for suffix in ("_vllm", ""):  # the pilot's head_dim-32 re-layout first, then the original head_dim 8
        tgt = str(fx.MODELS / f"target{suffix}")
        dft = str(fx.MODELS / f"draft{suffix}")
        specs = {
            "NULL": None,
            "VG1": dict(method="draft_model", model=dft, num_speculative_tokens=3),
            "VG2": dict(method="draft_model", model=dft, num_speculative_tokens=3, draft_sample_method="probabilistic"),
            "VG3": dict(method="ngram", num_speculative_tokens=3, prompt_lookup_min=1, prompt_lookup_max=2),
            "VG5": dict(method="draft_model", model=dft, num_speculative_tokens=3, rejection_sample_method="block"),
        }
        for arm, spec in specs.items():
            def run(arm=arm, spec=spec, tgt=tgt, suffix=suffix):
                from vllm import LLM, SamplingParams
                kw = dict(model=tgt, skip_tokenizer_init=True, dtype="float32", max_model_len=64, seed=0,
                          enforce_eager=True, disable_log_stats=False, gpu_memory_utilization=0.3)
                if spec:
                    kw["speculative_config"] = spec
                llm = LLM(**kw)
                res = {}
                try:
                    for s, sp in SETTINGS.items():
                        try:
                            params = [SamplingParams(max_tokens=fx.MAX_NEW, seed=i, detokenize=False, **sp) for i in range(N_SMOKE)]
                            t = time.time()
                            o = llm.generate([{"prompt_token_ids": fx.PROMPT}] * N_SMOKE, params, use_tqdm=False)
                            res[s] = dict(seq_per_s=round(N_SMOKE / (time.time() - t), 1),
                                          **shape_check([x.outputs[0].token_ids for x in o]))
                        except Exception as e:  # noqa: BLE001
                            res[s] = dict(error=f"{type(e).__name__}: {e}"[:300])
                        checkpoint(res)
                    try:
                        res["spec_counters"] = {m.name: getattr(m, "value", None) for m in llm.get_metrics() if "spec_decode" in m.name}
                    except Exception as e:  # noqa: BLE001
                        res["spec_counters"] = f"unavailable: {e}"
                finally:
                    del llm
                    import gc, torch
                    gc.collect(); torch.cuda.empty_cache()
                return dict(model=f"target{suffix}", settings=res)
            out.append((f"{arm}{suffix or '_headdim8'}", run))
    return out


# ------------------------------------------------------------------ SGLang

def sglang_probes():
    out = []
    for suffix in ("_vllm", ""):
        tgt, dft = str(fx.MODELS / f"target{suffix}"), str(fx.MODELS / f"draft{suffix}")
        specs = {
            "NULL": {},
            "SG3_standalone": dict(speculative_algorithm="STANDALONE", speculative_draft_model_path=dft,
                                   speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4,
                                   speculative_draft_attention_backend="triton"),
            "SG3_ngram": dict(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4),
        }
        for arm, spec in specs.items():
            def run(spec=spec, tgt=tgt, suffix=suffix):
                # FlashInfer (SGLang's default attention) has no fp32 kernels: KeyError torch.float32 while capturing
                # the prefill CUDA graph (Phase 0, L4). Triton attention, no CUDA graphs (FR5).
                # sgl_kernel's fused ops (rmsnorm first) do not dispatch fp32 either: every fused op takes its
                # pure-torch reference path (forward_native); the scheduler subprocesses inherit this (FR6).
                os.environ["SGLANG_FORCE_FUSED_OP_BACKEND"] = "torch"
                import sglang as sgl
                eng = sgl.Engine(model_path=tgt, skip_tokenizer_init=True, dtype="float32", context_length=64,
                                 mem_fraction_static=0.3, random_seed=0, attention_backend="triton",
                                 cuda_graph_backend_prefill="disabled", cuda_graph_backend_decode="disabled", **spec)
                res = {}
                try:
                    for s, sp in SETTINGS.items():
                        try:
                            p = dict(max_new_tokens=fx.MAX_NEW, temperature=sp.get("temperature", 1.0),
                                     top_k=sp.get("top_k", -1), top_p=sp.get("top_p", 1.0), min_p=sp.get("min_p", 0.0))
                            t = time.time()
                            o = eng.generate(input_ids=[fx.PROMPT] * N_SMOKE, sampling_params=[dict(p, sampling_seed=i) for i in range(N_SMOKE)])
                            res[s] = dict(seq_per_s=round(N_SMOKE / (time.time() - t), 1),
                                          **shape_check([x["output_ids"] for x in o]),
                                          meta_keys=sorted(o[0].get("meta_info", {}).keys())[:20])
                        except Exception as e:  # noqa: BLE001
                            res[s] = dict(error=f"{type(e).__name__}: {e}"[:300])
                        checkpoint(res)
                finally:
                    eng.shutdown()
                return dict(model=f"target{suffix}", settings=res,
                            engine_options=dict(attention_backend="triton", cuda_graphs="disabled",
                                                fused_op_backend="torch"))
            out.append((f"{arm}{suffix or '_headdim8'}", run))
    return out


# ------------------------------------------------------------------ TensorRT-LLM

def trtllm_probes():
    """fp32 needs these departures from TensorRT-LLM defaults (FR7, corrected by FR8):
    - FlashInfer stays installed (`modules/attention.py` imports it unconditionally), but `d40_trtllm_fp32.py`,
      loaded by a .pth in the trtllm venv, sets IS_FLASHINFER_AVAILABLE=False in `modules.rms_norm` and
      `modules.rotary_embedding`, whose FlashInfer kernels are fp16/bf16-only; both fall back to torch;
    - `disable_flashinfer_sampling=True` (the sampler's own switch) keeps sampling on torch strategies;
    - TP1 worker in-process (TLLM_WORKER_USE_SINGLE_PROCESS=1): no MPI spawn, and the recorded hook flags
      describe the process that computes.
    Arms run on VANILLA attention (torch SDPA) first, then on the default TRTLLM attention, whose fp32 support
    Phase 0 has to establish. The attention choice also switches the drafter: with TRTLLM attention, draft-target
    uses the chain drafter, which drafts greedily (argmax); with VANILLA it samples drafts with the request's
    strategy (`py_executor_creator.py`, `use_chain_drafter`). CUDA graphs and the overlap scheduler are off."""
    out = []
    for attn in ("VANILLA", "TRTLLM"):
        for suffix in ("_vllm", ""):
            tgt, dft = str(fx.MODELS / f"target{suffix}"), str(fx.MODELS / f"draft{suffix}")

            def make_spec(dft=dft):
                from tensorrt_llm.llmapi import DraftTargetDecodingConfig
                return DraftTargetDecodingConfig(max_draft_len=3, speculative_model=dft)

            for arm, spec_fn in (("NULL", None), ("TG1", make_spec)):
                def run(spec_fn=spec_fn, tgt=tgt, suffix=suffix, attn=attn):
                    # TP1 worker in this process (llmapi/utils.py enable_worker_single_process_for_tp1): no MPI spawn,
                    # and the flags recorded below describe the process that computes (FR8 review)
                    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
                    from tensorrt_llm import LLM, SamplingParams
                    from tensorrt_llm.llmapi import KvCacheConfig
                    kw = dict(model=tgt, skip_tokenizer_init=True, dtype="float32", attn_backend=attn,
                              cuda_graph_config=None, disable_overlap_scheduler=True, disable_flashinfer_sampling=True,
                              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
                              max_num_tokens=1024, max_batch_size=128, enable_iter_perf_stats=True)
                    if spec_fn:
                        kw["speculative_config"] = spec_fn()
                    llm = LLM(**kw)
                    res = {}
                    spec = dict(draft_tokens=0, accepted_tokens=0, stat_iterations=0)

                    def drain_stats():
                        # per-iteration specDecodingStats (py_executor.py). With the in-process TP1 worker, llm.get_stats()
                        # always returns [] (only the proxy fills its queue), so read the worker directly via fetch_stats()
                        # (base_worker.py) and fall back to get_stats() for a proxy executor (FR8 delta review)
                        try:
                            ex = llm._executor
                            if hasattr(ex, "fetch_stats"):
                                stats = [json.loads(ex._stats_serializer(st)) for st in ex.fetch_stats()]
                            else:
                                stats = [json.loads(st) if isinstance(st, str) else st for st in llm.get_stats(timeout=2)]
                            for st in stats:
                                sd = st.get("specDecodingStats") or {}
                                spec["draft_tokens"] += int(sd.get("numDraftTokens") or 0)
                                spec["accepted_tokens"] += int(sd.get("numAcceptedTokens") or 0)
                                spec["stat_iterations"] += 1
                        except Exception as e:  # noqa: BLE001
                            spec["error"] = f"{type(e).__name__}: {e}"[:200]

                    try:
                        for s, sp in SETTINGS.items():
                            try:
                                params = [SamplingParams(max_tokens=fx.MAX_NEW, seed=i, end_id=fx.EOS, pad_id=fx.EOS,
                                                         temperature=sp.get("temperature", 1.0), top_k=sp.get("top_k"),
                                                         top_p=sp.get("top_p"), min_p=sp.get("min_p")) for i in range(N_SMOKE)]
                                t = time.time()
                                o = llm.generate([fx.PROMPT] * N_SMOKE, params)
                                res[s] = dict(seq_per_s=round(N_SMOKE / (time.time() - t), 1),
                                              **shape_check([x.outputs[0].token_ids for x in o]))
                            except Exception as e:  # noqa: BLE001
                                res[s] = dict(error=f"{type(e).__name__}: {e}"[:300])
                            drain_stats()
                            checkpoint(res)
                        drain_stats()  # the last iteration's stats are appended after its responses go out
                        # draft_tokens counts non-zero draft ids when num_draft_tokens is 0, and id 0 is a real token
                        # at V=8: a fired / not-fired indicator, not an exact count; 0 iterations = unavailable
                        spec["available"] = spec["stat_iterations"] > 0
                        res["spec_counters"] = spec
                    finally:
                        llm.shutdown()
                    import d40_trtllm_fp32
                    from tensorrt_llm._torch.modules import rms_norm, rotary_embedding
                    return dict(model=f"target{suffix}", settings=res,
                                engine_options=dict(attn_backend=attn, cuda_graphs="disabled", overlap_scheduler="disabled",
                                                    flashinfer_sampling="disabled", worker="single-process",
                                                    kv_cache_free_gpu_memory_fraction=0.3, max_num_tokens=1024,
                                                    max_batch_size=128,
                                                    drafter=("chain, greedy argmax drafts" if attn == "TRTLLM"
                                                             else "model drafter, drafts sampled with the request strategy")
                                                    if spec_fn else None,
                                                    # TorchSampler: min_p is not applied (S6 = temp/top_k/top_p only) and
                                                    # per-request seed is not applied (one generator, seed 42)
                                                    sampler_notes="min_p ignored; per-request seed ignored",
                                                    flashinfer_rmsnorm=rms_norm.IS_FLASHINFER_AVAILABLE,
                                                    flashinfer_rope=rotary_embedding.IS_FLASHINFER_AVAILABLE,
                                                    fp32_hook_patched=sorted(d40_trtllm_fp32.PATCHED)))
                out.append((f"{arm}_{attn.lower()}{suffix or '_headdim8'}", run))
    return out


# ------------------------------------------------------------------ llama.cpp (CUDA)

def llamacpp_probes():
    """Needs D40_LCPP_BUILD pointing at a portable CUDA build (the notebook builds `build-cuda`)."""
    def lcpp():
        import lcpp_arms as L
        if "-ngl" not in L.KV:
            L.KV = L.KV + ["-ngl", "99"]  # offload all layers to the GPU
            L.SERVER_ARGS["LG2"] = L.SERVER_ARGS["L1"] + ["--spec-draft-sampling", "probabilistic"]
        return L
    out = []
    for arm, server_key, port in (("NULL", "NULL", 18300), ("LG1", "L1", 18301), ("LG2", "LG2", 18302),
                                  ("L2_ngram_server_extra", "L2", 18303)):
        def run(server_key=server_key, port=port):
            L = lcpp()
            srv = L.Server(server_key, port)
            res = {}
            try:
                for s in ("S0", "S2", "G", "S6"):
                    t = time.time()
                    seqs = L.draw(srv, s, N_SMOKE, 7)
                    res[s] = dict(seq_per_s=round(N_SMOKE / (time.time() - t), 1), **shape_check(seqs))
                    checkpoint(res)
            finally:
                srv.close()
            res["drafted_requests"] = open(srv.log.name).read().count("draft acceptance")
            return dict(settings=res)
        out.append((arm, run))

    def run_cli():
        L = lcpp()
        seqs = L.cli_draw("L3", "S0", 20, 11)
        return dict(settings={"S0": shape_check(seqs)}, cli_reruns=L.CLI_STATS["reruns"])
    out.append(("LG3_llama_speculative", run_cli))
    return out


# ------------------------------------------------------------------ HF (CUDA)

def hf_probes():
    out = []

    def run():
        import torch
        import hf_arms as H
        arm = H.Arm("H1", "S0")
        arm.target.to("cuda"); arm.assistant.to("cuda"); arm.prompt = arm.prompt.to("cuda")
        t = time.time()
        seqs = arm.draw(N_SMOKE, 5)
        return dict(seq_per_s=round(N_SMOKE / (time.time() - t), 1), **shape_check(seqs), device=torch.cuda.get_device_name(0))
    out.append(("HG1", run))
    return out


ENGINES = dict(vllm=vllm_probes, sglang=sglang_probes, trtllm=trtllm_probes, llamacpp=llamacpp_probes, hf=hf_probes)


# Log lines after which the engine cannot finish the probe, though its process may stay alive.
FATAL_LOG_MARKERS = {
    "trtllm": ("[TRT-LLM] [E] Error in event loop",),
}
FATAL_GRACE_S = 30
POLL_S = 5


def fatal_marker(engine, log_path):
    markers = FATAL_LOG_MARKERS.get(engine, ())
    if not markers:
        return None
    try:
        txt = open(log_path, errors="replace").read()
    except OSError:
        return None
    return next((m for m in markers if m in txt), None)


def kill_group(p):
    # the child may already have exited (and taken its group with it) between the poll and the kill
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def run_child(engine, arm, timeout, workdir, script=None):
    """Run one probe in a fresh process group; kill the whole group (engine workers too) on timeout.

    `script` is the harness the child runs with `--engine/--only/--result-file` (default: this file;
    Phase 0b passes `phase0b.py`)."""
    res_path = os.path.join(workdir, f"probe_{arm}.json")
    part_path = os.path.join(workdir, f"probe_{arm}.partial.json")
    for f in (res_path, part_path):
        if os.path.exists(f):
            os.remove(f)
    log_path = os.path.join(workdir, f"probe_{arm}.log")

    def log_tail():
        # engines that kill their own process (SGLang SIGKILLs its tree on a worker error) leave the cause only here
        try:
            txt = open(log_path, errors="replace").read()
        except OSError:
            return ""
        print(f"--- last lines of {log_path}:\n{txt[-3000:]}", flush=True)
        return txt[-4000:]

    t0 = time.time()  # wall time for the record
    m0 = time.monotonic()  # deadline and grace use the monotonic clock
    with open(log_path, "w") as log:
        p = subprocess.Popen([sys.executable, script or os.path.abspath(__file__), "--engine", engine, "--only", arm,
                              "--result-file", res_path], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        def killed(status, error):
            part = json.load(open(part_path)) if os.path.exists(part_path) else {}
            return dict(arm=arm, status=status, error=error, settings_done=part, log_tail=log_tail(),
                        wall_s=round(time.time() - t0, 2))

        try:
            rc, fatal_at = None, None
            while rc is None:
                remaining = m0 + timeout - time.monotonic()
                if remaining <= 0:
                    # freeze the log first, then classify what it held when the group was killed
                    kill_group(p)
                    m = fatal_at[1] if fatal_at else fatal_marker(engine, log_path)
                    if m:
                        return killed("ERROR", f"engine logged a fatal error ({m!r}) and had not exited at the "
                                               f"{timeout} s limit; process group killed")
                    return killed("TIMEOUT", f"no result within {timeout} s; process group killed")
                try:
                    rc = p.wait(timeout=min(POLL_S, remaining))
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if fatal_at is None:
                        m = fatal_marker(engine, log_path)
                        if m:
                            fatal_at = (now, m)
                    elif now - fatal_at[0] >= FATAL_GRACE_S:
                        kill_group(p)
                        return killed("ERROR", f"engine logged a fatal error ({fatal_at[1]!r}) and did not exit "
                                               f"within {FATAL_GRACE_S} s; process group killed")
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)  # stray engine workers left behind by a finished probe
            except ProcessLookupError:
                pass
    if os.path.exists(res_path):
        r = json.load(open(res_path))
        if r.get("status") != "RUNS":
            r["log_tail"] = log_tail()
            if os.path.exists(part_path):  # what the probe finished before it raised
                r["settings_done"] = json.load(open(part_path))
        return r
    part = json.load(open(part_path)) if os.path.exists(part_path) else {}
    return dict(arm=arm, status="ERROR", error=f"child exited with code {rc} and no result", settings_done=part,
                log_tail=log_tail(), wall_s=round(time.time() - t0, 2))


def main():
    global PARTIAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--out", default="out/d40_full")
    ap.add_argument("--probe-timeout", type=int, default=900, help="wall-clock seconds per probe")
    ap.add_argument("--only", help=argparse.SUPPRESS)          # child mode: run this one probe
    ap.add_argument("--result-file", help=argparse.SUPPRESS)   # child mode: where to write its result
    a = ap.parse_args()
    arms = ENGINES[a.engine]()

    if a.only:  # child
        PARTIAL = a.result_file.replace(".json", ".partial.json")
        r = probe(a.only, dict(arms)[a.only])
        json.dump(r, open(a.result_file, "w"), indent=1, default=str)
        return

    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(os.environ["D40_WORK"], "logs")
    os.makedirs(work, exist_ok=True)
    rec = dict(phase="0", engine=a.engine, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), environment=environment(),
               probe_timeout_s=a.probe_timeout, complete=False,
               fixture_sha256={n: sh(f"sha256sum {fx.MODELS / n}").split()[0] if os.path.exists(fx.MODELS / n) else None
                               for n in ("target/model.safetensors", "draft/model.safetensors", "target.gguf", "draft.gguf")},
               probes=[])
    gpu = rec["environment"]["gpu"].split(",")[0].replace(" ", "_") or "nogpu"
    path = os.path.join(a.out, f"phase0_{a.engine}_{gpu}_{int(time.time())}.json")
    print("recording to", path, flush=True)
    for name, _ in arms:
        print(f"=== probe {name} (limit {a.probe_timeout} s)", flush=True)
        r = run_child(a.engine, name, a.probe_timeout, work)
        rec["probes"].append(r)
        print(f"=== {name}: {r['status']} {r.get('error', '')}", flush=True)
        json.dump(rec, open(path, "w"), indent=1, default=str)  # rewritten after every probe
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["complete"] = True
    json.dump(rec, open(path, "w"), indent=1, default=str)
    print("wrote", path)
    for p in rec["probes"]:
        print(p["arm"], p["status"], p.get("error", ""))


if __name__ == "__main__":
    main()
