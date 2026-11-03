"""FR22 E3 cells for the arm that passed the gate (run note 5): vLLM VG4p@fitflat (EAGLE-3, probabilistic drafts,
Model Runner V2), F-A S0-S3 and F-B S0/S2 at N_A = 2·10⁵, greedy G (1,000 runs per fixture).

VG4 runs on Model Runner V2, and Stage A drew vLLM's nulls and R_eng on V1 only, so this run rebuilds both on V2
(§5.1: a cell's null and R_eng come from the sampler that draws it): `--part gate` scores R_eng with Phase 0b's
`reference_v(2, ...)` into OUT/gate_tables and writes OUT/gate.json (Stage A's gate rule); `--part FA|FB` draws the
V2 null and the VG4p cell at each gate-passed setting, then greedy. Chunks and seeds follow Stage A's layout, with
h("E3", key) + j. Analysis: `e3_analyze.py OUT`.

    python e3_cells.py --out OUT --part gate|FA|FB
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "code")]

ARM = "VG4p@fitflat"
SETTINGS = dict(FA=["S0", "S1", "S2", "S3"], FB=["S0", "S2"])
N_G = 1000


def patch_spec(E):
    orig = E.spec

    def spec(fix, m):
        if m.startswith("VG4p"):  # FR22 run note 3: VG4 with probabilistic drafting
            return dict(orig(fix, m.replace("VG4p", "VG4")), draft_sample_method="probabilistic")
        return orig(fix, m)

    E.spec = spec


def gate(out):
    import p0b_vllm as E
    import stagea as A
    os.environ["D40_P0B_OUT"] = str(out / "gate")
    ref = E.reference_v(2, [("FA", "NULL"), ("FB", "NULL")])
    v = A.gate_verdicts("vllm", [dict(reference={k.replace("/V2", "/V1"): x for k, x in ref.items()})])
    A.write_json(str(out / "gate.json"), dict(runner="V2", reference=ref, verdicts=v))
    print(json.dumps(v, indent=1))


def draw(out, fname):
    import p0b_common as C
    import p0b_vllm as E
    import stagea as A
    patch_spec(E)
    gv = json.load(open(out / "gate.json"))["verdicts"]
    fix = C.fixture(fname)
    settings = [s for s in SETTINGS[fname] if gv.get(f"{fname}/{s}", {}).get("passed")]
    print(fname, "gate-passed settings", settings, flush=True)
    jobs = [("NULL@V2", "null", s) for s in settings] + [(ARM, "cell", s) for s in settings]
    if gv.get(f"{fname}/S0", {}).get("passed"):
        jobs.append((ARM, "greedy", "G"))
    for method in ("NULL@V2", ARM):
        m = E.llm(fix, method.replace("@V2", ""))
        try:
            v2 = E.runner_used(m)
            if v2 is not True:
                raise RuntimeError(f"expected Model Runner V2, got {v2!r}")
            for meth, kind, s in jobs:
                if meth != method:
                    continue
                k = A.key("vllm", fname, meth, s)
                n = N_G if kind == "greedy" else A.n_a()
                for i, j0, c in A.chunk_plan(n):
                    if A.have(k, i, j0, c):
                        continue
                    seeds = [C.seed_base("E3", k) + j for j in range(j0, j0 + c)]
                    before = E.counters(m)
                    A.tick()
                    seqs = E.generate(m, fix, s, seeds)
                    A.write_chunk(k, i, j0, seqs, seed_range=[seeds[0], seeds[-1]], use_v2_model_runner=v2,
                                  spec_counters=E.delta(E.counters(m), before), shape=A.shape(fix, seqs))
                A.mark_cell_done(k, dict(chunks=len(A.chunk_plan(n))))
                print("done", k, flush=True)
        finally:
            E.free(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("gate", "FA", "FB"), required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["D40_STAGEA_OUT"] = str(out)
    os.environ["D40_STAGE"] = "A"
    os.environ["TRITON_F32_DEFAULT"] = "ieee"  # FR13, as Stage A (run note 6: run 1 omitted it and failed the gate)
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.environ.get("D40_WORK", "/tmp"), "triton_cache_ieee"))
    import p0b_vllm as E
    E.runner(2)
    if a.part == "gate":
        gate(out)
    else:
        draw(out, a.part)


if __name__ == "__main__":
    main()
