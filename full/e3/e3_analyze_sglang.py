"""FR22 E3 analysis of `e3_cells_sglang.py` output (run note 8): each SG1@fit cell against its own rebuilt R_eng table
(Stage A's frozen F-A test), Holm at 0.01 over all E3 cells run (vLLM's from `e3_analyze.py`'s E3.json and these);
the nulls are Stage A's SGLang nulls, quoted from ANALYSIS_A.json. Greedy against the argmax chain of the rebuilt S0
R_eng, as Stage A.

    python e3_analyze_sglang.py SGL_OUT VLLM_OUT     # writes SGL_OUT/E3_sglang.json, SGL_OUT/E3_sglang.txt
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "code")]

import fixture as fx  # noqa: E402
import p0b_common as C  # noqa: E402
import stagea as A  # noqa: E402
import stagea_analyze as S  # noqa: E402
from e3_cells_sglang import ARM, SETTINGS  # noqa: E402

ALPHA = 0.01


def main(out, vllm_out):
    out, vllm_out = pathlib.Path(out), pathlib.Path(vllm_out)
    os.environ["D40_STAGEA_OUT"] = str(out)
    refs = S.Refs(out, sub=".")
    gv = json.load(open(out / "gate.json"))["verdicts"]
    stage_a = {r["key"]: r for r in json.load(open(HERE.parent / "stagea" / "data" / "ANALYSIS_A.json"))["rows"]
               if r.get("key", "").startswith("sglang/FA/NULL/")}
    fix = C.fixture("FA")
    rows = []
    for s in SETTINGS:
        k = A.key("sglang", "FA", ARM, s)
        if not gv.get(f"FA/{s}", {}).get("passed"):
            rows.append(dict(key=k, kind="cell", status="REFERENCE_UNRESOLVED (gate)", gate=gv.get(f"FA/{s}")))
            continue
        if not (out / "chunks" / k.replace("/", "__")).exists():
            rows.append(dict(key=k, kind="cell", status="MISSING"))
            continue
        rec = S.read_cell(out, k)
        r = dict(key=k, kind="cell", fixture="FA", setting=s, N=rec["n"], spec_counters=rec["spec_counters"],
                 null=stage_a.get(f"sglang/FA/NULL/{s}"))
        r.update(S.fa_test(rec["counts"], refs.fa_seq("sglang", f"sglang_FA_{s}_default.npz"), fix))
        r["mde_eps"] = S.mde_of("sglang", "FA", s)
        if rec["n"] != A.n_a():
            r.update(status=f"INCOMPLETE (N={rec['n']})", p=1.0)
        rows.append(r)
    k = A.key("sglang", "FA", ARM, "G")
    if (out / "chunks" / k.replace("/", "__")).exists():
        rec = S.read_cell(out, k)
        t = refs.load("sglang", "sglang_FA_S0_default.npz")["eng"]
        idx = {tuple(g): i for i, g in enumerate(fx.nonterminal_prefixes())}
        chain, margin = S.greedy_chain(lambda g: t[idx[g]] if g in idx else None, fix)
        hits = sum(m for sq, m in rec["counts"].items() if fix.cut(sq) == chain)
        rows.append(dict(key=k, kind="greedy", N=rec["n"], argmax_chain=chain, min_top2_margin=margin, matches=hits,
                         status="REFERENCE_UNRESOLVED" if chain is None else "PASS" if hits == rec["n"] else "FAIL"))
    # Holm over every E3 cell run: vLLM's (E3.json) and these
    vrows = [r for r in json.load(open(vllm_out / "E3.json"))["rows"] if r.get("kind") == "cell" and "p" in r]
    mine = [r for r in rows if "p" in r]
    rej = S.holm_over([r["p"] for r in vrows] + [r["p"] for r in mine], len(vrows) + len(mine), ALPHA)
    for r, x in zip(mine, rej[len(vrows):]):
        r["holm_reject"] = bool(x)
        r.setdefault("status", "REJECTED" if x else "NOT_REJECTED")
    vllm_changed = [r["key"] for r, x in zip(vrows, rej[: len(vrows)]) if bool(x) != bool(r.get("holm_reject"))]
    lines = []
    for r in rows:
        extra = ""
        if "p" in r:
            extra = f" N {r['N']} p {r['p']:.3g} mde {r.get('mde_eps')}"
            sc = r.get("spec_counters", {})
            if sc.get("spec_num_proposed_drafts"):
                extra += f" acc {sc['spec_num_correct_drafts'] / sc['spec_num_proposed_drafts']:.3f}"
            if r.get("null"):
                extra += f" | Stage A null p {r['null'].get('p'):.3g}"
        if r["kind"] == "greedy" and "matches" in r:
            extra = f" {r['matches']}/{r['N']} on chain {r['argmax_chain']} (margin {r['min_top2_margin']:.3g})"
        lines.append(f"{r['key']:36s} {r['kind']:6s} {r['status']}{extra}")
    lines.append(f"Holm at {ALPHA} over {len(vrows) + len(mine)} E3 cells; vLLM verdicts changed: {vllm_changed or 'none'}")
    (out / "E3_sglang.json").write_text(json.dumps(dict(alpha=ALPHA, gate=gv, rows=rows, holm_m=len(vrows) + len(mine),
                                                        vllm_changed=vllm_changed), indent=1, default=str) + "\n")
    (out / "E3_sglang.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
