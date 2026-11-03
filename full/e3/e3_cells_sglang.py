"""FR22 E3 cells for SGLang's gate-passing arm (run note 8): SG1@fit (EAGLE-3 chain, fitted head), F-A at the
Stage-A-running settings whose gate passes on this GPU, N_A = 2·10⁵, and greedy G (1,000 runs).

As Stage A's `draw_sglang`: SGLang ignores per-request seeds, so each cell is one engine with
`random_seed = h("E3", key)` and an interrupted cell is redrawn whole. `--part gate` rebuilds R_eng with Phase 0b's
`reference_for` (one child per setting, as Stage A: a scoring path that kills the scheduler loses only its own
setting) into OUT/gate_tables and writes OUT/gate.json (Stage A's gate rule). `--part draw` draws each cell in its own
child. The SGLang nulls are Stage A's (same sampler, already passed; FR22 E3). Analysis runs off-GPU.

    python e3_cells_sglang.py --out OUT --part gate|draw
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "code")]

ARM = "SG1@fit"
SETTINGS = ["S0", "S1", "S2", "S3"]  # Stage A's SGLang F-A running settings; the gate decides which run
N_G = 1000


def child(out, args):
    p = subprocess.run([sys.executable, __file__, "--out", str(out), *args], capture_output=True, text=True,
                       timeout=3 * 3600)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    return json.loads(line[-1][7:]) if line else dict(error=(p.stdout + p.stderr)[-3000:])


def gate(out):
    import stagea as A
    results = []
    for s in SETTINGS:
        r = child(out, ["--one-gate", s])
        print("gate", s, json.dumps(r, default=str)[:600], flush=True)
        results.append(r)
    v = A.gate_verdicts("sglang", results)
    A.write_json(str(out / "gate.json"), dict(results=results, verdicts=v))
    print(json.dumps(v, indent=1), flush=True)


def one_gate(s):
    import p0b_sglang as E
    return E.reference_for(None, settings_fa=[s], settings_fb=[])


def one_draw(kind, s):
    import p0b_common as C
    import p0b_sglang as E
    import stagea as A
    fix = C.fixture("FA")
    k = A.key("sglang", "FA", ARM, s)
    n = N_G if kind == "greedy" else A.n_a()
    if A.cell_complete("sglang", k, kind):
        return dict(key=k, complete=True, skipped=True)
    moved = A.restart_whole_cell(k)
    seed = C.seed_base("E3", k)
    eng = E.engine(fix, ARM, seed)
    done = []
    try:
        for i, j0, c in A.chunk_plan(n):
            A.tick()
            o = eng.generate(input_ids=[fix.PROMPT] * c, sampling_params=E.sp(fix, s))
            seqs = [x["output_ids"] for x in o]
            done.append(A.write_chunk(k, i, j0, seqs, engine_seed=seed, request_seeds="unused (FR9 item 3)",
                                      spec_counters=E.counters(o), shape=A.shape(fix, seqs)))
            del o
    finally:
        eng.shutdown()
    A.mark_cell_done(k, dict(chunks=len(done), engine_seed=seed))
    return dict(key=k, chunks=len(done), moved_partial=moved)


def draw(out):
    gv = json.load(open(out / "gate.json"))["verdicts"]
    settings = [s for s in SETTINGS if gv.get(f"FA/{s}", {}).get("passed")]
    print("gate-passed settings", settings, flush=True)
    jobs = [("cell", s) for s in settings] + ([("greedy", "G")] if "S0" in settings else [])
    for kind, s in jobs:
        r = child(out, ["--one-draw", kind, s])
        print(kind, s, json.dumps(r, default=str)[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("gate", "draw"))
    ap.add_argument("--one-gate")
    ap.add_argument("--one-draw", nargs=2)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["D40_STAGEA_OUT"] = str(out)
    os.environ["D40_STAGE"] = "A"
    os.environ["D40_P0B_OUT"] = str(out / "gate")  # R_eng tables go to OUT/gate_tables
    os.environ["TRITON_F32_DEFAULT"] = "ieee"  # FR13
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.environ.get("D40_WORK", "/tmp"), "triton_cache_ieee"))
    if a.one_gate:
        print("RESULT", json.dumps(one_gate(a.one_gate), default=str))
    elif a.one_draw:
        print("RESULT", json.dumps(one_draw(*a.one_draw), default=str))
    elif a.part == "gate":
        gate(out)
    else:
        draw(out)


if __name__ == "__main__":
    main()
