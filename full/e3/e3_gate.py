"""FR22 E3 feasibility gate (non-evaluative, Phase-0 rules: counters only, sequences discarded unread).

    python e3_gate.py --engine vllm|sglang --out DIR

For each fitted head variant (`fit`, `fitflat`) and arm, a 100-sequence S0 smoke through the Phase 0b smoke code
(`p0b_vllm.smoke_one` / `p0b_sglang.smoke_one`, one child process per arm), recording the spec counters. Acceptance:
vLLM accepted / draft tokens; SGLang correct / proposed drafts. SG2 (tree) also records the accepted tree paths
(`d40_sgl_tree_hook.py`) for the tree-exercised check. Writes DIR/e3_gate_<engine>.json.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ARMS = {"vllm2": [("FA", "VG4p@fit"), ("FA", "VG4p@fitflat"), ("FB", "VG4p@fit"), ("FB", "VG4p@fitflat")],
        "vllm": [("FA", "VG4@fit"), ("FA", "VG4@fitflat"), ("FB", "VG4@fit"), ("FB", "VG4@fitflat")],
        "sglang": [("FA", "SG1@fit"), ("FA", "SG1@fitflat"), ("FA", "SG2@fit"), ("FA", "SG2@fitflat"),
                   ("FB", "SG1@fit"), ("FB", "SG1@fitflat")]}


def one(engine, fname, method):
    import p0b_common as C
    if engine in ("vllm", "vllm2"):
        import p0b_vllm as E
        if method.startswith("VG4p"):  # FR22 run note 3: VG4 with probabilistic drafting
            orig = E.spec
            E.spec = lambda fix, m: dict(orig(fix, m.replace("VG4p", "VG4")), draft_sample_method="probabilistic")
        r = E.smoke_one(fname, 2, method, ["S0"])
    else:
        import p0b_sglang as E
        r = E.smoke_one(fname, method, ["S0"])
    return r


def acceptance(engine, rec):
    for k, v in rec.get("smoke", {}).items():
        if not isinstance(v, dict):
            continue
        c = v.get("spec_counters") if engine.startswith("vllm") else v.get("spec")
        if not c:
            continue
        if engine.startswith("vllm") and c.get("spec_decode_num_draft_tokens"):
            return c["spec_decode_num_accepted_tokens"] / c["spec_decode_num_draft_tokens"]
        if engine == "sglang" and c.get("spec_num_proposed_drafts"):
            return c["spec_num_correct_drafts"] / c["spec_num_proposed_drafts"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ARMS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one", nargs=2)
    a = ap.parse_args()
    if a.one:
        print("RESULT", json.dumps(one(a.engine, *a.one), default=str))
        return
    res = []
    for fname, method in ARMS[a.engine]:
        env = dict(os.environ)
        trace = os.path.join(a.out, f"tree_{fname}_{method.replace('@', '_')}.jsonl")
        if method.startswith("SG2"):
            env["D40_TREE_TRACE"] = trace
        try:
            p = subprocess.run([sys.executable, __file__, "--engine", a.engine, "--out", a.out, "--one", fname, method],
                               capture_output=True, text=True, timeout=1800, env=env)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            rec = json.loads(line[-1][7:]) if line else dict(error=(p.stdout + p.stderr)[-3000:])
        except Exception as e:  # noqa: BLE001
            rec = dict(error=f"{type(e).__name__}: {e}")
        row = dict(fixture=fname, method=method, record=rec, acceptance=acceptance(a.engine, rec))
        if method.startswith("SG2"):
            paths = [json.loads(x) for x in open(trace)] if os.path.exists(trace) else []
            depth1 = sorted({ai[1] for p_ in paths for ai, n in zip(p_["accept_index"], p_["accept_lens"])
                             if n >= 2 and len(ai) > 1})
            row["tree"] = dict(verifies=len(paths), distinct_depth1_nodes=depth1, exercised=len(depth1) >= 2,
                               hook_error=open(trace + ".err").read() if os.path.exists(trace + ".err") else None)
        row["in_band"] = row["acceptance"] is not None and 0.3 <= row["acceptance"] <= 0.9
        res.append(row)
        print(fname, method, "acceptance", row["acceptance"], row.get("tree", ""), flush=True)
        json.dump(res, open(os.path.join(a.out, f"e3_gate_{a.engine}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
