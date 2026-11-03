"""Check Phase 0b records: list every failure a record holds, instead of trusting a probe's RUNS status.

    python p0b_check.py RECORD.json [...] [--require-clean]

A probe returns RUNS whenever its function returns, even if cells inside it failed, so this walks the whole
record and reports:
- probes not RUNS;
- any entry with an `error`;
- shape: `out_of_space`, `raw_malformed` > 0 or `count_ok` false;
- seed checks whose repeat or difference check failed;
- effect assertions that did not change their quantity (or whose sham run was not clean);
- speculative smoke cells that ran but proposed no draft, or whose counters are unavailable;
- reference rows without a finite `delta_num`;
- a probe section marked `complete: false` (the vLLM VG5 preflight when counters or drafts are missing);
- an incomplete record.
Lines marked OUTCOME (an effect assertion with a clean sham run that did not change its quantity) are results,
not harness faults; a sham run that is not clean is a harness failure. A run of a probe subset is incomplete.
With --require-clean the exit code is 1 when anything else is listed (the CPU dry run requires it for HF and
llama.cpp); otherwise the list is informational (GPU runs hold expected refusals, which the post-Phase-0b
revision classifies).
"""
import json
import sys


def walk(x, path=""):
    if isinstance(x, dict):
        yield path, x
        for k, v in x.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")


# per engine, the counter of drafts proposed in a smoke cell (this setting only): vLLM, TensorRT-LLM and llama.cpp
# under `spec_counters`, SGLang under `spec`. HF's HG1 is the study's own
# reference arm, which drafts by construction (its effect assertions read the verifier calls).
SETTINGS = {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "G"}
COUNTER_EXEMPT = {"HG1"}  # firing established by construction and by the effect assertions (above)
PROPOSED = ("spec_decode_num_drafts", "draft_tokens", "spec_num_proposed_drafts", "drafted_requests")


def not_drafting(path, d):
    """A speculative smoke cell (method not NULL) that ran without error but shows no proposed draft."""
    out = []
    method = path.split("/")[-2] if path.count("/") >= 2 else ""
    if method.startswith("NULL") or "error" in d:
        return out
    smoke_cell = "/smoke/" in path and path.rsplit("/", 1)[-1] in SETTINGS
    if smoke_cell and method not in COUNTER_EXEMPT and not any(k in d for k in ("spec_counters", "spec")):
        out.append(f"{path}: speculative smoke cell without draft counters")
    for key in ("spec_counters", "spec"):
        if key not in d:
            continue
        c = d[key]
        if not isinstance(c, dict):
            out.append(f"{path}: {key} unavailable ({c})")
        elif "error" in c:
            out.append(f"{path}: {key} error {str(c['error'])[:120]}")
        elif not any(isinstance(c.get(f), (int, float)) and c[f] > 0 for f in PROPOSED):
            out.append(f"{path}: speculative method did not draft {c}")
    return out


def failures(rec):
    out = []
    if not rec.get("complete"):
        out.append("record incomplete")
    for p in rec.get("probes", []):
        arm = p.get("arm")
        if p.get("status") != "RUNS":
            out.append(f"{arm}: {p.get('status')} {str(p.get('error', ''))[:160]}")
        for path, d in walk({k: v for k, v in p.items() if k not in ("traceback", "log_tail")}):
            if "error" in d and path:
                out.append(f"{arm}{path}: {str(d['error'])[:160]}")
            if d.get("out_of_space") or d.get("raw_malformed") or d.get("count_ok") is False:
                out.append(f"{arm}{path}: shape {({k: d.get(k) for k in ('n', 'out_of_space', 'raw_malformed', 'count_ok')})}")
            for k in ("same_seed_repeats", "next_seed_differs", "next_seeds_differ", "other_seed_differs"):
                if d.get(k) is False:
                    out.append(f"{arm}{path}: seed check {k} false")
            if "sham_clean" in d and not d["sham_clean"]:
                # the unpatched arm did not repeat itself: the control is invalid, a harness failure
                out.append(f"{arm}{path}: sham run not clean")
            elif "changed" in d and not d["changed"]:
                out.append(f"OUTCOME {arm}{path}: effect not shown {({k: d.get(k) for k in ('sham_clean', 'patched_quantity')})}")
            out.extend(f"{arm}{x}" for x in not_drafting(path, d))
            if d.get("complete") is False:  # a probe's own evidence incomplete, e.g. the vLLM VG5 preflight
                out.append(f"{arm}{path}: incomplete {[k for k, v in d.items() if v == 'UNRESOLVED']}")
            if "delta_num" in d and not (isinstance(d["delta_num"], (int, float)) and d["delta_num"] < float("inf")):
                out.append(f"{arm}{path}: delta_num {d['delta_num']}")
    return out


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    bad = False
    for path in paths:
        rec = json.load(open(path))
        f = failures(rec)
        print(f"{path}: {rec.get('engine')} {len(rec.get('probes', []))} probes, {len(f)} listed")
        for x in f:
            print("  ", x)
        # an effect that does not show is a Phase 0b outcome for the post-Phase-0b revision, not a harness failure
        bad |= any(not x.startswith("OUTCOME ") for x in f)
    if "--require-clean" in sys.argv and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
