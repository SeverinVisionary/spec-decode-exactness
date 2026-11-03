"""D-40 full study, Stage A draws (PREREGISTRATION.md §5.4, FR9 items 3-6, FR13).

One engine per fresh Colab runtime:

    python stagea.py --engine vllm|sglang|trtllm|llamacpp|hf --out out/d40_full

Everything is written under `<out>/stageA_<engine>/`, which is the same folder on every runtime, so a restart
resumes:
1. **Gate** (FR13). `R_eng` and δ_num are rebuilt on this GPU for every fixture × setting the running set needs,
   with the Phase 0b reference code, and each passes only if it is complete and 6·δ_num < MDE/10 (F5 at N_A). The
   gate record `gate.json` is written before any draw and reused on the same GPU type.
2. **Draws**, in FR13's order: controls, nulls, cells, then greedy G. Each chunk of `--chunk` draws is written to
   `chunks/<key>/cNN.json.gz` with its seeds, counters and the environment; a chunk file exists only once it is
   complete. vLLM, llama.cpp and HF resume at the first missing chunk. SGLang and TensorRT-LLM have one seed per
   cell (FR9 item 3), so an incomplete cell is moved to `partial/` (kept, never read) and redrawn whole.
3. `stageA_<engine>.json` records every job and is rewritten after each one.

Draws are only counted here. Nothing in this file compares a draw with a reference; the statistics run later, on
CPU, from the chunk files (§8). Each job runs in its own child process group with a wall-clock limit (Phase 0's
`run_child`), so a crash or hang loses at most the chunk in flight.
"""
import argparse
import collections
import gzip
import importlib
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("D40_WORK", "<runtime>/d40work")

import phase0  # noqa: E402
import p0b_common as C  # noqa: E402

ENGINES = ("vllm", "sglang", "trtllm", "llamacpp", "hf")
STAGE = "A"
N_G = 1000  # §5.4: greedy runs per method
MDE_KEY = dict(vllm="vllm", sglang="sglang", trtllm="trtllm", llamacpp="lcpp", hf="hf")
TRITON_ENGINES = ("vllm", "sglang")  # FR13: Triton fp32 dots at IEEE precision

# FR13's running set: fixture -> settings, per engine. Methods per fixture below.
FA_ALL = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
RUNNING = {
    "vllm": dict(FA=dict(settings=["S0", "S1", "S2", "S3", "S5"], methods=["VG1", "VG2", "VG3"], null="NULL@V1"),
                 FB=dict(settings=["S0", "S2"], methods=["VG1", "VG2"], null="NULL@V1"),
                 controls=[("P7", "S2"), ("P8", "S0")]),
    "sglang": dict(FA=dict(settings=["S0", "S1", "S2", "S3"], methods=["SG3s", "SG3n"], null="NULL"),
                   controls=[]),
    "llamacpp": dict(FA=dict(settings=FA_ALL, methods=["LG1", "LG2"], null="NULL"),
                     FB=dict(settings=["S0", "S2", "S6"], methods=["LG2"], null="NULL"),
                     controls=[]),
    "trtllm": dict(FA=dict(settings=["S0", "S1", "S2", "S3"], methods=["TG1v"], null="NULL"),
                   FB=dict(settings=["S0", "S2"], methods=["TG1v"], null="NULL"),
                   controls=[]),
    "hf": dict(FA=dict(settings=FA_ALL, methods=["HG1"], null="NULL"),
               controls=[("P1", "S2"), ("P2", "S2"), ("P3", "S2"), ("P3", "S4"), ("P4", "S6"), ("P5", "S2"),
                         ("P6", "S0")]),
}
PER_CELL_SEED = ("sglang", "trtllm")  # one stream per cell: redraw whole cells (FR9 item 3)


def key(engine, fname, method, setting):
    return f"{engine}/{fname}/{method}/{setting}"


# ------------------------------------------------------------------ paths and chunk files

def root():
    return os.environ["D40_STAGEA_OUT"]


def n_a():
    return int(os.environ.get("D40_STAGEA_NA", 200_000))


def chunk_size():
    return int(os.environ.get("D40_STAGEA_CHUNK", C.CHUNK))


def cell_dir(k):
    return os.path.join(root(), "chunks", k.replace("/", "__"))


def chunk_path(k, i):
    return os.path.join(cell_dir(k), f"c{i:02d}.json.gz")


def chunk_plan(n):
    """FR9 item 3: chunk k covers j in [J_k, J_k + n_k); a short last chunk reuses no index."""
    c = chunk_size()
    return [(i, j0, min(c, n - j0)) for i, j0 in enumerate(range(0, n, c))]


def have(k, i, j0, n):
    """A chunk counts only if its file reads back whole and covers exactly this plan's indices."""
    if not os.path.exists(chunk_path(k, i)):
        return False
    try:
        c = json.load(gzip.open(chunk_path(k, i), "rt"))
    except (OSError, EOFError, ValueError):
        return False
    return c.get("key") == k and c.get("j0") == j0 and c.get("n") == n and sum(c.get("counts", [])) == n


def cell_complete(engine, k, kind):
    """From the chunks themselves; done.json is metadata, except that a one-stream-per-cell engine needs it (a cell
    without it was interrupted and is redrawn whole)."""
    if engine in PER_CELL_SEED and not os.path.exists(done_path(k)):
        return False
    return all(have(k, i, j0, n) for i, j0, n in chunk_plan(n_of(kind)))


def write_json(path, obj):
    """Same-folder temporary file, then rename: an interrupted write never truncates the previous copy."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def env_record():
    p = os.path.join(root(), "env.json")
    return json.load(open(p)) if os.path.exists(p) else {}


_T = [time.time()]


def tick():
    """Start of a chunk's draw: write_chunk records the seconds since (throughput per chunk)."""
    _T[0] = time.time()


def write_chunk(k, i, j0, seqs, **meta):
    """Atomic on the local side: written to a temporary name in the same folder, then renamed."""
    meta.setdefault("draw_seconds", round(time.time() - _T[0], 2))
    counts = collections.Counter(tuple(int(t) for t in s) for s in seqs)
    rec = dict(key=k, chunk=i, j0=j0, n=len(seqs), seqs=[list(s) for s in counts], counts=list(counts.values()),
               written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), environment=env_record(), **meta)
    os.makedirs(cell_dir(k), exist_ok=True)
    tmp = chunk_path(k, i) + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f, default=str)
    os.replace(tmp, chunk_path(k, i))
    del counts, seqs  # counted only (§8)
    return dict(chunk=i, n=rec["n"], distinct=len(rec["seqs"]))


def done_path(k):
    return os.path.join(cell_dir(k), "done.json")


def mark_cell_done(k, info):
    write_json(done_path(k), dict(key=k, **info))


def restart_whole_cell(k):
    """SGLang / TensorRT-LLM: an incomplete cell is moved aside whole (kept, never read) before it is redrawn.
    Called only for cells that are not complete."""
    d = cell_dir(k)
    if os.path.isdir(d) and os.listdir(d):
        dst = os.path.join(root(), "partial", f"{k.replace('/', '__')}_{int(time.time())}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(d, dst)
        return dst
    return None


def shape(fix, seqs):
    s = fix.shape_check(seqs)
    s.pop("n", None)
    return s


# ------------------------------------------------------------------ gate (FR13)

MDE_AT = {200_000: "NA", 50_000: "Nfloor"}  # F5 computed the MDE at N_A and at the §7 floor only


def mde(engine, fname, setting):
    """F5's MDE at this run's N: N_A, or the 5·10^4 floor if §7's budget rule lowered N (FR13). Other N (the CPU dry
    run) use N_A's value."""
    f = os.path.join(HERE, "phase0b", f"f5_mde_{fname.lower()}.json")
    cells = json.load(open(f))["cells"]
    c = cells.get(f"{MDE_KEY[engine]}_{setting}")
    if c is None:
        return None
    at = MDE_AT.get(n_a(), "NA")
    return c[f"mde_tv_{at}"] if fname == "FA" else c[f"mde_eps_{at}"]


def gate_reports(engine, raw):
    """Map an engine's reference output onto {(fixture, setting): [delta reports]}."""
    out = collections.defaultdict(list)
    if engine == "hf":
        for s, row in raw.get("reference", {}).items():
            for tag in ("batch1", "batch1000"):  # HG1 draws at batch 1, its null at 10^3 (both must pass)
                out[("FA", s)].append(row.get(tag, dict(error="missing")))
        return out
    for k, v in raw.get("reference", {}).items():
        parts = k.split("/")
        fname = parts[0]
        s = next((p for p in parts if p in C.SETTINGS), None)
        if s is None:
            continue
        if engine == "llamacpp" and not k.endswith(f"/np{16}"):
            continue  # production shape: -np 16 (FR9 item 5)
        out[(fname, s)].append(v)
    return out


def gate_verdicts(engine, results):
    """results: list of child results (each may hold `reference`). Returns {"FA/S0": {...pass...}}."""
    merged = dict(reference={})
    for r in results:
        merged["reference"].update(r.get("reference") or {})
        merged["reference"].update(r.get("settings_done") or {})
    reps = gate_reports(engine, merged)
    run = RUNNING[engine]
    v = {}
    for fname in ("FA", "FB"):
        if fname not in run:
            continue
        for s in run[fname]["settings"]:
            got = reps.get((fname, s), [])
            m = mde(engine, fname, s)
            bad = [g for g in got if not isinstance(g, dict) or "error" in g or not g.get("finite")]
            delta = max((g["delta_num"] for g in got if isinstance(g, dict) and "delta_num" in g), default=None)
            ok = bool(got) and not bad and delta is not None and m is not None and 6 * delta < m / 10
            v[f"{fname}/{s}"] = dict(passed=ok, delta_num=delta, mde=m, bound=None if m is None else m / 60,
                                     reports=len(got), failed_reports=len(bad))
    return v


def gate_probes(engine):
    """Child jobs that rebuild R_eng with the Phase 0b code, restricted to the running set."""
    if engine == "vllm":
        import p0b_vllm as E
        return [("gate", lambda: dict(reference=E.reference_v(1, [("FA", "NULL"), ("FB", "NULL")])))]
    if engine == "sglang":
        import p0b_sglang as E
        # one child per setting: a scoring path that kills the scheduler loses only its own setting
        return [(f"gate_FA_{s}", lambda s=s: E.reference_for(None, settings_fa=[s], settings_fb=[]))
                for s in RUNNING["sglang"]["FA"]["settings"]]
    E = importlib.import_module(f"p0b_{engine}")
    return [("gate", E.reference)]


def gate_ok(gate, fname, setting):
    return bool(gate.get("verdicts", {}).get(f"{fname}/{setting}", {}).get("passed"))


# ------------------------------------------------------------------ the jobs

def cells_of(engine, gate):
    """[(kind, fixture, method, setting)] in FR13's order: controls, nulls, cells, greedy. Gate-failed settings
    are left out (REFERENCE_UNRESOLVED). Greedy runs where the fixture's S0 gate passed: its argmax chain is read
    from that R_eng (§5.4)."""
    run = RUNNING[engine]
    out = [("control", "FA", c, s) for c, s in run["controls"] if gate_ok(gate, "FA", s)]
    for fname in ("FA", "FB"):
        if fname in run:
            out += [("null", fname, run[fname]["null"], s) for s in run[fname]["settings"] if gate_ok(gate, fname, s)]
    for fname in ("FA", "FB"):
        if fname in run:
            out += [("cell", fname, m, s) for m in run[fname]["methods"] for s in run[fname]["settings"]
                    if gate_ok(gate, fname, s)]
    for fname in ("FA", "FB"):
        if fname in run:
            if gate_ok(gate, fname, "S0"):
                out += [("greedy", fname, m, "G") for m in run[fname]["methods"]]
    return out


def groups(engine, gate):
    """Jobs, each one child process: cells that share an engine instance are drawn in one job."""
    cells = cells_of(engine, gate)
    if engine in ("sglang", "hf"):
        if engine == "hf":  # one job per phase, so each phase finishes before the next starts (FR13 order)
            ph = [("draw_hf_controls", [c for c in cells if c[0] == "control"]),
                  ("draw_hf_nulls", [c for c in cells if c[0] == "null"]),
                  ("draw_hf_cells", [c for c in cells if c[0] in ("cell", "greedy")])]
            return [(n, cs) for n, cs in ph if cs]
        return [(f"draw_{f}_{m}_{s}".replace("@", "_"), [(k, f, m, s)]) for k, f, m, s in cells]
    g = collections.OrderedDict()
    for k, f, m, s in cells:
        base = m if k != "control" else {"P7": "VG1", "P8": "VG2"}[m]
        name = f"draw_{f}_{base}" + (f"_{m}" if k == "control" else "")
        g.setdefault(name.replace("@", "_"), []).append((k, f, m, s))
    return list(g.items())


def n_of(kind):
    return N_G if kind == "greedy" else n_a()


# ------------------------------------------------------------------ engine drawers (child side)

def draw_vllm(cells):
    import p0b_vllm as E
    E.runner(1)  # every running vLLM path is on the V1 runner (FR12, FR13)
    kind, fname, method, _ = cells[0]
    fix = C.fixture(fname)
    base = {"P7": "VG1", "P8": "VG2"}.get(method, method).replace("@V1", "")
    patch = _vllm_control(method) if kind == "control" else None
    m = E.llm(fix, base)
    out = {}
    try:
        v2 = E.runner_used(m)
        out["use_v2_model_runner"] = v2
        if v2 is not False:  # FR12/FR13: every running vLLM path, with its null and R_eng, is on the V1 runner
            raise RuntimeError(f"vLLM did not run on the V1 runner (use_v2_model_runner={v2!r})")
        for kind, fname, method, s in cells:
            k = key("vllm", fname, method, s)
            done = []
            for i, j0, n in chunk_plan(n_of(kind)):
                if have(k, i, j0, n):
                    continue
                seeds = [C.seed_base(STAGE, k) + j for j in range(j0, j0 + n)]
                before = E.counters(m)
                if patch:
                    patch["rec"].clear()
                tick()
                seqs = E.generate(m, fix, s, seeds)
                meta = dict(seed_range=[seeds[0], seeds[-1]], spec_counters=E.delta(E.counters(m), before),
                            shape=shape(fix, seqs), use_v2_model_runner=v2)
                if patch:
                    eff = dict(patch["rec"])
                    meta["effect"] = dict(eff, changed=patch["changed"](eff))
                    if not meta["effect"]["changed"]:
                        raise RuntimeError(f"{k} chunk {i}: effect assertion failed {eff}")  # §5.3: abort
                done.append(write_chunk(k, i, j0, seqs, **meta))
                C.phase0_checkpoint({k: done})
            mark_cell_done(k, dict(chunks=len(chunk_plan(n_of(kind)))))
            out[k] = done
    finally:
        E.free(m)
        if patch:
            patch["undo"]()
    return dict(draws=out)


def _vllm_control(control):
    """P7 / P8 as in Phase 0b's effect probes (p0b_vllm.effect_p7/p8), active for the whole control cell; `rec` is
    reset per chunk and must show a change in every chunk (§5.3)."""
    rec = {}
    if control == "P7":
        import vllm.v1.sample.rejection_sampler as rs
        orig_c, orig_p = rs.apply_sampling_constraints, rs.RejectionSampler.apply_logits_processors
        import p0b_vllm as E

        def constraints(logits, *a, **k):
            proc = orig_c(logits.clone(), *a, **k)
            rec["calls"] = rec.get("calls", 0) + 1
            rec["max_abs_diff"] = max(rec.get("max_abs_diff", 0.0), E._maxdiff(proc, logits))
            return logits

        rs.apply_sampling_constraints = constraints
        rs.RejectionSampler.apply_logits_processors = lambda self, logits, *a, **k: logits

        def undo():
            rs.apply_sampling_constraints, rs.RejectionSampler.apply_logits_processors = orig_c, orig_p

        return dict(rec=rec, undo=undo, changed=lambda e: e.get("calls", 0) > 0 and e.get("max_abs_diff", 0) > 0)
    import vllm.v1.spec_decode.llm_base_proposer as lbp
    orig = lbp.compute_probs_and_sample_next_token

    def p8(logits, *a, **k):
        ids, probs = orig(logits, *a, **k)
        am = probs.argmax(dim=-1)
        rec["calls"] = rec.get("calls", 0) + 1
        rec["tokens_changed"] = rec.get("tokens_changed", 0) + int((am != ids).sum())
        rec["q_nondegenerate"] = rec.get("q_nondegenerate", False) or bool((probs.max(-1).values < 1).any())
        return am.to(ids.dtype), probs

    lbp.compute_probs_and_sample_next_token = p8

    def undo():
        lbp.compute_probs_and_sample_next_token = orig

    return dict(rec=rec, undo=undo, changed=lambda e: e.get("tokens_changed", 0) > 0 and e.get("q_nondegenerate", False))


def draw_sglang(cells):
    import p0b_sglang as E
    (kind, fname, method, s), = cells
    fix = C.fixture(fname)
    k = key("sglang", fname, method, s)
    moved = restart_whole_cell(k)
    seed = C.seed_base(STAGE, k)
    eng = E.engine(fix, method, seed)
    done = []
    try:
        for i, j0, n in chunk_plan(n_of(kind)):
            tick()
            o = eng.generate(input_ids=[fix.PROMPT] * n, sampling_params=E.sp(fix, s))
            seqs = [x["output_ids"] for x in o]
            done.append(write_chunk(k, i, j0, seqs, engine_seed=seed, request_seeds="unused (FR9 item 3)",
                                    spec_counters=E.counters(o), shape=shape(fix, seqs)))
            del o
            C.phase0_checkpoint({k: done})
    finally:
        eng.shutdown()
    mark_cell_done(k, dict(chunks=len(done), engine_seed=seed))
    return dict(draws={k: done}, moved_partial=moved)


def draw_llamacpp(cells):
    import p0b_llamacpp as E
    kind, fname, method, _ = cells[0]
    fix = C.fixture(fname)
    srv = E.Server(fix, method, E.port())
    out = {}
    try:
        for kind, fname, method, s in cells:
            k = key("llamacpp", fname, method, s)
            done = []
            for i, j0, n in chunk_plan(n_of(kind)):
                if have(k, i, j0, n):
                    continue
                seeds = [C.seed_base(STAGE, k) + j for j in range(j0, j0 + n)]
                srv.log.flush()
                at = os.path.getsize(srv.log.name)
                tick()
                seqs = E.draw(srv, fix, s, seeds)
                time.sleep(0.5)  # let the server finish logging this chunk's requests
                done.append(write_chunk(k, i, j0, seqs, seed_range=[seeds[0], seeds[-1]], shape=shape(fix, seqs),
                                        spec_counters=_lcpp_counters(srv, at), slots=E.NP))
                C.phase0_checkpoint({k: done})
            mark_cell_done(k, dict(chunks=len(chunk_plan(n_of(kind)))))
            out[k] = done
    finally:
        srv.close()
    return dict(draws=out)


LCPP_ACC = re.compile(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")


def _lcpp_counters(srv, at):
    """Per-request lines llama-server logs at the end of each drafted request (server-context.cpp 714-716 at the
    Phase 0 commit): drafted requests, and accepted / generated draft tokens, for this chunk's log bytes only."""
    srv.log.flush()
    with open(srv.log.name, errors="replace") as f:
        f.seek(at)
        txt = f.read()
    acc = [(int(a), int(g)) for a, g in LCPP_ACC.findall(txt)]
    return dict(drafted_requests=txt.count("draft acceptance"), accepted_tokens=sum(a for a, _ in acc),
                draft_tokens=sum(g for _, g in acc))


def draw_trtllm(cells):
    import p0b_trtllm as E
    kind, fname, method, _ = cells[0]
    fix = C.fixture(fname)
    m = E.llm(fix, method)
    out = {}
    try:
        st = E.Stats(m)
        hooks = E.hooks()  # fp32 / drafter hooks, recorded in every chunk (FR8, FR11)
        for kind, fname, method, s in cells:
            k = key("trtllm", fname, method, s)
            if cell_complete("trtllm", k, kind):
                continue
            moved = restart_whole_cell(k)
            seed = C.seed_base(STAGE, k)
            E.drain_until_quiet(st)
            sampler = E.sampler_of(m)
            sampler._generator = None  # the cell's stream starts at h("A", key) (FR9 item 3)
            hooked = E.seed_hook(m, seed)
            done = []
            for i, j0, n in chunk_plan(n_of(kind)):
                before = E.drain_until_quiet(st)
                tick()
                o = m.generate([fix.PROMPT] * n, E.params(fix, s, n))
                seqs = [x.outputs[0].token_ids for x in o]
                after = E.drain_until_quiet(st)
                sc = {c: after[c] - before.get(c, 0) for c in ("draft_tokens", "accepted_tokens", "stat_iterations",
                                                              "fallback_requests")}
                if "error" in after:
                    sc["error"] = after["error"]
                done.append(write_chunk(k, i, j0, seqs, stream_seed=seed, sampler=hooked,
                                        request_seeds="unused (FR9 item 3)", strategy=E.greedy_flags(fix, s), hooks=hooks,
                                        spec_counters=sc, shape=shape(fix, seqs)))
                del o
                C.phase0_checkpoint({k: done})
            mark_cell_done(k, dict(chunks=len(done), stream_seed=seed, moved_partial=moved))
            out[k] = done
    finally:
        m.shutdown()
    return dict(draws=out)


# HF: the batch-1 chunks (HG1, P1-P6, greedy) are split over worker processes (FR13); each chunk seeds its own
# stream with h("A", key) + J_k, so the split does not change any draw. Draft/accept counters come from the verifier
# call itself (`_speculative_sampling`, which the controls also wrap), per chunk (§5.5 item 3).

class _HFCounters:
    def __enter__(self):
        import transformers.generation.utils as U
        self.U, self.orig = U, U._speculative_sampling
        self.c = dict(verifier_calls=0, draft_tokens=0, accepted_tokens=0)

        def counted(candidate_input_ids, candidate_logits, candidate_length, *a, **k):
            out = self.orig(candidate_input_ids, candidate_logits, candidate_length, *a, **k)
            self.c["verifier_calls"] += 1
            self.c["draft_tokens"] += int(candidate_length)
            self.c["accepted_tokens"] += int(out[1])  # n_matches
            return out

        U._speculative_sampling = counted
        return self

    def __exit__(self, *exc):
        self.U._speculative_sampling = self.orig


def _hf_task(task):
    kind, k, control, setting, i, j0, n = task
    import torch
    import p0b_hf as E
    if not _hf_task.__dict__.get("threads_set"):
        torch.set_num_threads(1)  # many workers share the runtime's CPUs (FR13)
        _hf_task.threads_set = True
    cache = _hf_task.__dict__.setdefault("arms", {})
    method = control if kind == "control" else "H1"
    if (method, setting) not in cache:
        cache[(method, setting)] = E.control_arm(control, setting) if kind == "control" else E.arm("H1", setting)
    a = cache[(method, setting)]
    seed = C.seed_base(STAGE, k) + j0
    tick()
    meta = dict(stream_seed=seed, batch=1, request_seeds="unused (FR9 item 3)")
    if kind == "control":
        eff = E.effect(control, setting, seed=seed, a=a)
        meta["effect"] = eff
        if not eff.get("changed"):
            return dict(key=k, chunk=i, error=f"effect assertion failed: {json.dumps(eff, default=str)[:300]}")
        extra = {}
        with _HFCounters() as cnt, E.control_ctx(control, a, extra):  # counters outside: they see what the verifier got
            seqs = a.draw(n, seed)
        meta["control_extra"] = extra
    else:
        with _HFCounters() as cnt:
            seqs = a.draw(n, seed)
    meta["spec_counters"] = dict(cnt.c)
    meta["shape"] = shape(C.FA, seqs)
    return dict(key=k, **write_chunk(k, i, j0, seqs, **meta))


def draw_hf(cells):
    import multiprocessing as mp
    import p0b_hf as E
    out = {}
    tasks = []
    for kind, fname, method, s in cells:
        k = key("hf", fname, method, s)
        if kind == "null":  # batched 10^3 (FR9 item 5): fast, drawn in this process
            a = E.arm("NULL", s)
            for i, j0, n in chunk_plan(n_a()):
                if have(k, i, j0, n):
                    continue
                seed = C.seed_base(STAGE, k) + j0
                tick()
                seqs = a.draw_null_batched(n, seed, batch=E.NULL_BATCH)
                out.setdefault(k, []).append(write_chunk(k, i, j0, seqs, stream_seed=seed, batch=E.NULL_BATCH,
                                                         request_seeds="unused (FR9 item 3)", shape=shape(C.FA, seqs)))
                C.phase0_checkpoint(out)
            del a
            continue
        for i, j0, n in chunk_plan(n_of(kind)):
            if not have(k, i, j0, n):
                tasks.append((kind, k, method, s, i, j0, n))
    if tasks:
        workers = min(int(os.environ.get("D40_HF_WORKERS") or os.cpu_count() or 1), len(tasks))
        out["workers"] = workers
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_hf_task, tasks):
                out.setdefault(r["key"], []).append(r)
                C.phase0_checkpoint(out)
                if "error" in r:  # §5.3: a failed assertion aborts; nothing queued after it may run
                    pool.terminate()
                    raise RuntimeError(f"{r['key']} chunk {r['chunk']}: {r['error']}")
    for kind, fname, method, s in cells:
        k = key("hf", fname, method, s)
        if cell_complete("hf", k, kind):
            mark_cell_done(k, dict(chunks=len(chunk_plan(n_of(kind)))))
    return dict(draws=out)


DRAW = dict(vllm=draw_vllm, sglang=draw_sglang, llamacpp=draw_llamacpp, trtllm=draw_trtllm, hf=draw_hf)


# ------------------------------------------------------------------ driver

def load_gate():
    p = os.path.join(root(), "gate.json")
    return json.load(open(p)) if os.path.exists(p) else None


def child_jobs(engine):
    gate = load_gate()
    if gate is None:
        return gate_probes(engine)
    return [(name, (lambda cs=cs: DRAW[engine](cs))) for name, cs in groups(engine, gate)]


def cell_status(engine, gate):
    """Every running cell's state, for the record."""
    st = {}
    for kind, fname, method, s in cells_of(engine, gate):
        k = key(engine, fname, method, s)
        plan = chunk_plan(n_of(kind))
        st[k] = dict(kind=kind, chunks_done=sum(have(k, i, j0, n) for i, j0, n in plan), chunks=len(plan),
                     complete=cell_complete(engine, k, kind))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--out", default="out/d40_full")
    ap.add_argument("--n-a", type=int, default=200_000, help="draws per cell (FR9 item 4)")
    ap.add_argument("--chunk", type=int, default=C.CHUNK)
    ap.add_argument("--job-timeout", type=int, default=6 * 3600, help="wall-clock seconds per job")
    ap.add_argument("--gate-timeout", type=int, default=2400)
    ap.add_argument("--max-hours", type=float, default=None, help="start no job after this many hours")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--tag", default="", help="output folder suffix, e.g. _t4 for HF's T4 fallback (FR13)")
    ap.add_argument("--only", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.engine in TRITON_ENGINES:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"  # FR13, inherited by every child
        # its own kernel cache, so no kernel compiled under another precision default is reused
        os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["D40_WORK"], "triton_cache_ieee")

    if a.only:  # child
        phase0.PARTIAL = a.result_file.replace(".json", ".partial.json")
        r = phase0.probe(a.only, dict(child_jobs(a.engine))[a.only])
        json.dump(r, open(a.result_file, "w"), indent=1, default=str)
        return

    os.environ["D40_STAGEA_OUT"] = os.path.join(a.out, f"stageA_{a.engine}{a.tag}")
    os.environ["D40_STAGEA_NA"], os.environ["D40_STAGEA_CHUNK"] = str(a.n_a), str(a.chunk)
    os.makedirs(root(), exist_ok=True)
    work = os.path.join(os.environ["D40_WORK"], "logs", "stageA")
    os.makedirs(work, exist_ok=True)
    env = phase0.environment()
    env["TRITON_F32_DEFAULT"] = os.environ.get("TRITON_F32_DEFAULT")
    gpu = env["gpu"].split(",")[0].strip() or "nogpu"
    if gpu != "nogpu" and a.n_a not in MDE_AT:
        sys.exit(f"--n-a {a.n_a}: production runs use N_A = 2e5 or the 5e4 floor (§7, FR9 item 4), the N F5 covers")
    # one chunk plan per output folder: a resume with another N_A or chunk size would mix index ranges
    spec_path = os.path.join(root(), "spec.json")
    spec = dict(n_a=a.n_a, chunk=a.chunk, gpu=gpu)
    if os.path.exists(spec_path):
        old = json.load(open(spec_path))
        if old != spec:
            sys.exit(f"{root()} was started with {old}; this run asks for {spec}. Use the same N_A, chunk size and "
                     f"GPU type to resume, or a new --out (§5.1 host rule: a GPU change needs its own references).")
    else:
        write_json(spec_path, spec)
    write_json(os.path.join(root(), "env.json"), env)
    t0 = time.time()
    rec_path = os.path.join(root(), f"stageA_{a.engine}.json")
    try:
        rec = json.load(open(rec_path))
    except FileNotFoundError:
        rec = dict(stage=STAGE, engine=a.engine, runs=[])
    except ValueError:  # an interrupted write of an older harness: keep it aside, the chunks are the data
        os.replace(rec_path, rec_path + f".corrupt_{int(time.time())}")
        rec = dict(stage=STAGE, engine=a.engine, runs=[])
    run = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), environment=env, n_a=a.n_a,
               chunk=a.chunk, jobs=[])
    rec["runs"].append(run)

    def save():
        write_json(rec_path, rec)

    gate = load_gate()  # same GPU type as the spec (checked above), so its references belong to every chunk here
    if gate is None:
        os.environ["D40_P0B_OUT"] = os.path.join(root(), "gate")  # R_eng tables go to gate_tables/
        results = []
        for name, _ in gate_probes(a.engine):
            print(f"=== gate {name} (limit {a.gate_timeout} s)", flush=True)
            r = phase0.run_child(a.engine, name, a.gate_timeout, work, script=os.path.abspath(__file__))
            results.append(r)
            run["jobs"].append(dict(r, log_tail=r.get("log_tail", "")[-1500:]))
            save()
        gate = dict(gpu=gpu, built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), environment=env,
                    verdicts=gate_verdicts(a.engine, results), results=results)
        write_json(os.path.join(root(), "gate.json"), gate)
    run["gate"] = gate["verdicts"]
    for fs, v in gate["verdicts"].items():
        print(f"gate {fs}: {'PASS' if v['passed'] else 'REFERENCE_UNRESOLVED'} δ_num={v['delta_num']} bound={v['bound']}")
    save()
    if a.gate_only:
        return
    for name, cs in groups(a.engine, gate):
        if a.max_hours is not None and time.time() - t0 > a.max_hours * 3600:
            run["stopped"] = f"--max-hours {a.max_hours} reached before {name}"
            print(run["stopped"], flush=True)
            break
        if all(cell_complete(a.engine, key(a.engine, f, m, s), kd) for kd, f, m, s in cs):
            continue
        print(f"=== {name}: {len(cs)} cells (limit {a.job_timeout} s)", flush=True)
        r = phase0.run_child(a.engine, name, a.job_timeout, work, script=os.path.abspath(__file__))
        run["jobs"].append(dict(arm=r.get("arm"), status=r.get("status"), error=r.get("error"),
                                wall_s=r.get("wall_s"), log_tail=r.get("log_tail", "")[-1500:]))
        print(f"=== {name}: {r['status']} {r.get('error', '')}", flush=True)
        save()
        if "effect assertion failed" in str(r.get("error")):
            # §5.3: a control that does not change its quantity invalidates the harness for this engine; stop spending
            run["stopped"] = f"{name}: {r['error']}"
            print("stopping:", run["stopped"], flush=True)
            break
        if any(kd in ("control", "null") for kd, *_ in cs) and \
                not all(cell_complete(a.engine, key(a.engine, f, m, s), kd) for kd, f, m, s in cs):
            # FR13 order: a control or null phase that did not finish stops the run, so a resume finishes it before any
            # later phase draws; a failed cell job does not stop the other cells
            run["stopped"] = f"{name} ended {r['status']} with its control/null cells incomplete: {r.get('error')}"
            print("stopping:", run["stopped"], flush=True)
            break
    run["cells"] = cell_status(a.engine, gate)
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run["complete"] = all(v["complete"] for v in run["cells"].values())
    save()
    left = [k for k, v in run["cells"].items() if not v["complete"]]
    print(f"Stage A {a.engine}: {len(run['cells']) - len(left)}/{len(run['cells'])} cells complete", flush=True)
    for k in left:
        print("  incomplete:", k, run["cells"][k])


if __name__ == "__main__":
    main()
