"""D-40 full study, Phase 0b (non-evaluative; PREREGISTRATION.md §7, FR9 item 7, FR11, FR12).

One engine per fresh Colab runtime:

    python phase0b.py --engine vllm|sglang|trtllm|llamacpp|hf --out out/d40_full

Every probe runs in its own child process group with a wall-clock limit (the Phase 0 machinery,
`phase0.run_child`), and the record is rewritten after every probe. Engine probes live in
`p0b_<engine>.py`; each returns a list of (name, fn). A probe returns a dict, which may hold:
- `smoke`: per cell × setting, `shape_check` counts and timing (sequences discarded unread);
- `seed_check`: equality checks of repeated draws (never tabulated);
- `effect`: control effect assertions (max-abs difference of the patched quantity, §5.3);
- `reference`: R_eng δ_num reports; the R_eng/R_np tables are written next to the record as `.npz`;
- `throughput`: seq/s on `PROMPT_TP` at the production batch shape.

On a CPU host (the dry run) the same command runs unchanged: probes that need a GPU record `ERROR`,
and the harness, the records and the fp64 references are still exercised end to end.
"""
import argparse
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("D40_WORK", "<runtime>/d40work")

import phase0  # noqa: E402
import p0b_common as C  # noqa: E402

ENGINES = ("vllm", "sglang", "trtllm", "llamacpp", "hf")


def arms(engine):
    return importlib.import_module(f"p0b_{engine}").probes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--out", default="out/d40_full")
    ap.add_argument("--probe-timeout", type=int, default=1200, help="wall-clock seconds per probe")
    ap.add_argument("--only-probes", help="comma-separated probe names (default: all)")
    ap.add_argument("--only", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", help=argparse.SUPPRESS)
    a = ap.parse_args()
    probes = arms(a.engine)

    if a.only:  # child
        phase0.PARTIAL = a.result_file.replace(".json", ".partial.json")
        r = phase0.probe(a.only, dict(probes)[a.only])
        json.dump(r, open(a.result_file, "w"), indent=1, default=str)
        return

    selection = None
    if a.only_probes:
        keep = set(a.only_probes.split(","))
        unknown = keep - {n for n, _ in probes}
        if unknown:
            sys.exit(f"unknown probes: {sorted(unknown)}; known: {[n for n, _ in probes]}")
        probes = [p for p in probes if p[0] in keep]
        selection = [n for n, _ in probes]  # a subset: the record is never complete
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(os.environ["D40_WORK"], "logs")
    os.makedirs(work, exist_ok=True)
    env = phase0.environment()
    gpu = env["gpu"].split(",")[0].replace(" ", "_") or "nogpu"
    stamp = int(time.time())
    os.environ["D40_P0B_OUT"] = os.path.join(a.out, f"phase0b_{a.engine}_{gpu}_{stamp}")  # .npz tables go here
    rec = dict(phase="0b", engine=a.engine, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               environment=env, probe_timeout_s=a.probe_timeout, complete=False, n_smoke=C.N_SMOKE,
               chunk=C.CHUNK, prompt_throughput=C.PROMPT_TP, tables_dir=os.environ["D40_P0B_OUT"] + "_tables",
               fixture_sha256={n: phase0.sh(f"sha256sum {C.fx.MODELS / n}").split()[0]
                               if os.path.exists(C.fx.MODELS / n) else None
                               for n in ("target/model.safetensors", "draft/model.safetensors",
                                         "fb_target/model.safetensors", "fb_draft/model.safetensors",
                                         "target.gguf", "draft.gguf", "fb_target.gguf", "fb_draft.gguf",
                                         "eagle3_FA_tgt/model.safetensors", "eagle3_FA_flat/model.safetensors",
                                         "eagle3_FB_tgt/model.safetensors", "eagle3_FB_flat/model.safetensors",
                                         "mimo_FA/model.safetensors", "mimo_FA/config.json",
                                         # runtime re-layouts the engines load (built by the notebook, code/vllm_export.py)
                                         "target_vllm/model.safetensors", "target_vllm/config.json",
                                         "draft_vllm/model.safetensors", "draft_vllm/config.json")},
               probe_selection=selection, probes=[])
    path = os.environ["D40_P0B_OUT"] + ".json"
    print("recording to", path, flush=True)
    for name, _ in probes:
        print(f"=== probe {name} (limit {a.probe_timeout} s)", flush=True)
        r = phase0.run_child(a.engine, name, a.probe_timeout, work, script=os.path.abspath(__file__))
        rec["probes"].append(r)
        print(f"=== {name}: {r['status']} {r.get('error', '')}", flush=True)
        json.dump(rec, open(path, "w"), indent=1, default=str)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["complete"] = selection is None  # only a run of every probe is a complete Phase 0b record
    json.dump(rec, open(path, "w"), indent=1, default=str)
    print("wrote", path)
    for p in rec["probes"]:
        print(p["arm"], p["status"], p.get("error", ""))


if __name__ == "__main__":
    main()
