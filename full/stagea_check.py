"""Check a Stage A output folder for harness failures (FR13). Reads chunk metadata and counts only; it never
compares draws with a reference.

    python stagea_check.py <out>/stageA_<engine> [--require-clean]

Lists, per running cell: missing or extra chunks, wrong draw counts, seed ranges that do not follow FR9 item 3
(each chunk's exact range; FR9 does not require different cells' hash-derived ranges to be disjoint),
out-of-space or malformed sequences, speculative cells whose counters show no drafting, control chunks whose effect
assertion did not change the patched quantity, and environment fields missing from a chunk (§5.5). Exit 1 under
`--require-clean` if anything is listed.
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stagea as A  # noqa: E402
import p0b_common as C  # noqa: E402

DRAFT_FIELDS = ("spec_decode_num_drafts", "drafted_requests", "spec_num_proposed_drafts", "draft_tokens")
ENV_FIELDS = ("gpu", "harness_git")


def check(folder, allow_cpu=False):
    engine = os.path.basename(os.path.normpath(folder)).split("_", 1)[1]
    engine = next(e for e in A.ENGINES if engine == e or engine.startswith(e + "_"))  # e.g. hf_t4 (FR13 fallback)
    st = os.path.basename(os.path.normpath(folder))[5]  # stageA_* or stageB_* (FR14)
    rec = json.load(open(os.path.join(folder, f"stage{st}_{engine}.json")))
    run = rec["runs"][-1]
    os.environ["D40_STAGEA_OUT"] = folder
    if st == "B":
        spec = json.load(open(os.path.join(folder, "spec.json")))
        os.environ["D40_STAGE"], os.environ["D40_STAGEB_KEYS"] = "B", ",".join(spec["keys"])
    os.environ["D40_STAGEA_NA"], os.environ["D40_STAGEA_CHUNK"] = str(run["n_a"]), str(run["chunk"])
    gate = json.load(open(os.path.join(folder, "gate.json")))
    issues = []
    cells = A.cells_of(engine, gate)
    for kind, fname, method, s in cells:
        k = A.key(engine, fname, method, s)
        plan = A.chunk_plan(A.n_of(kind))
        base = A.seed0(k)
        files = sorted(f for f in os.listdir(A.cell_dir(k))) if os.path.isdir(A.cell_dir(k)) else []
        extra = [f for f in files if f.endswith(".json.gz") and f not in {os.path.basename(A.chunk_path(k, i))
                                                                      for i, _, _ in plan}]
        if extra:
            issues.append(f"{k}: unexpected files {extra}")
        if not A.cell_complete(engine, k, kind):
            issues.append(f"{k}: not complete")
        shas, hists = set(), {}
        for i, j0, n in plan:
            p = A.chunk_path(k, i)
            if not os.path.exists(p):
                issues.append(f"{k} c{i:02d}: missing")
                continue
            c = json.load(gzip.open(p, "rt"))
            if c["n"] != n or sum(c["counts"]) != n or c["j0"] != j0:
                issues.append(f"{k} c{i:02d}: n {c['n']} / sum {sum(c['counts'])} / j0 {c['j0']}, expected {n} / {j0}")
            sh = c.get("shape", {})
            if sh.get("out_of_space") or sh.get("raw_malformed"):
                issues.append(f"{k} c{i:02d}: shape {sh}")
            env = c.get("environment") or {}
            shas.add(env.get("harness_git"))
            hist = tuple(sorted(zip(map(tuple, c["seqs"]), c["counts"])))
            if kind != "greedy" and len(c["seqs"]) > 1 and hist in hists:  # a stochastic chunk repeated: seeds reused
                issues.append(f"{k} c{i:02d}: identical draws to c{hists[hist]:02d}")
            hists.setdefault(hist, i)
            if engine in A.TRITON_ENGINES and env.get("TRITON_F32_DEFAULT") != "ieee":
                issues.append(f"{k} c{i:02d}: TRITON_F32_DEFAULT is {env.get('TRITON_F32_DEFAULT')!r}, FR13 needs 'ieee'")
            if (env.get("gpu") or "").split(",")[0].strip() != (gate.get("gpu") if gate.get("gpu") != "nogpu" else ""):
                issues.append(f"{k} c{i:02d}: drawn on {env.get('gpu')!r}, gate built on {gate.get('gpu')!r}")
            if engine == "vllm" and c.get("use_v2_model_runner") is not False:
                issues.append(f"{k} c{i:02d}: use_v2_model_runner {c.get('use_v2_model_runner')!r}, FR12 needs V1")
            # "gpu" may be empty only in a CPU dry run (--allow-cpu); production chunks must name their GPU
            if not env.get("harness_git") or not (env.get("gpu") or (allow_cpu and "gpu" in env)):
                issues.append(f"{k} c{i:02d}: environment lacks {[f for f in ENV_FIELDS if not env.get(f)]}")
            if "seed_range" in c:
                if c["seed_range"] != [base + j0, base + j0 + n - 1]:
                    issues.append(f"{k} c{i:02d}: seed range {c['seed_range']}, expected {[base + j0, base + j0 + n - 1]}")
            elif "stream_seed" in c:
                want = base if engine == "trtllm" else base + j0
                if c["stream_seed"] != want:
                    issues.append(f"{k} c{i:02d}: stream seed {c['stream_seed']}, expected {want}")
            elif "engine_seed" in c:
                if c["engine_seed"] != base:
                    issues.append(f"{k} c{i:02d}: engine seed {c['engine_seed']}, expected {base}")
            else:
                issues.append(f"{k} c{i:02d}: no seed recorded")
            if kind in ("cell", "control") and "NULL" not in method:
                sc = c.get("spec_counters")
                if not isinstance(sc, dict) or "error" in sc or not any((sc.get(f) or 0) > 0 for f in DRAFT_FIELDS):
                    issues.append(f"{k} c{i:02d}: no drafting in counters {sc}")
            if kind == "control" and not (c.get("effect") or {}).get("changed"):
                issues.append(f"{k} c{i:02d}: effect assertion did not change the patched quantity")
        if len(shas) > 1:
            issues.append(f"{k}: chunks from different harness commits {sorted(map(str, shas))}")
    unresolved = [fs for fs, v in gate["verdicts"].items() if not v["passed"]]
    return engine, len(cells), unresolved, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--require-clean", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="CPU dry run: chunks may have an empty GPU name")
    a = ap.parse_args()
    engine, n, unresolved, issues = check(a.folder, a.allow_cpu)
    print(f"{engine}: {n} cells to draw; gate REFERENCE_UNRESOLVED: {unresolved or 'none'}; {len(issues)} listed")
    for x in issues:
        print("  ", x)
    if a.require_clean and issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
