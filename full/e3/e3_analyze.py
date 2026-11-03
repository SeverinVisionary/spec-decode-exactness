"""FR22 E3 analysis of `e3_cells.py` output: Stage A's frozen tests against the V2 R_eng, Holm at 0.01 over the E3
cells run (and, separately, over the E3 nulls); a cell whose null is rejected is INCONCLUSIVE.

    python e3_analyze.py OUT        # writes OUT/E3.json, OUT/E3.txt
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "code")]

import f5_mde_full as f5  # noqa: E402
import fixture as fx  # noqa: E402
import p0b_common as C  # noqa: E402
import stagea as A  # noqa: E402
import stagea_analyze as S  # noqa: E402
from e3_cells import ARM, SETTINGS  # noqa: E402

ALPHA = 0.01


def main(out):
    out = pathlib.Path(out)
    os.environ["D40_STAGEA_OUT"] = str(out)
    refs = S.Refs(out, sub=".")
    gv = json.load(open(out / "gate.json"))["verdicts"]
    rows, null_sel = [], {}
    for fname in ("FA", "FB"):
        fix = C.fixture(fname)
        for s in SETTINGS[fname]:
            if not gv.get(f"{fname}/{s}", {}).get("passed"):
                rows.append(dict(key=A.key("vllm", fname, ARM, s), kind="cell", status="REFERENCE_UNRESOLVED (gate)"))
                continue
            tname = f"vllm_{fname}_NULL_{s}_V2.npz"
            for kind, meth in (("null", "NULL@V2"), ("cell", ARM)):
                k = A.key("vllm", fname, meth, s)
                if not (out / "chunks" / k.replace("/", "__")).exists():
                    rows.append(dict(key=k, kind=kind, status="MISSING"))
                    continue
                rec = S.read_cell(out, k)
                r = dict(key=k, kind=kind, fixture=fname, setting=s, N=rec["n"], spec_counters=rec["spec_counters"])
                if fname == "FA":
                    r.update(S.fa_test(rec["counts"], refs.fa_seq("vllm", tname), fix))
                else:
                    fb = refs.fb_rows("vllm", s, tname)
                    if kind == "null":
                        v = S.visits(rec["counts"], fix)
                        null_sel[s] = {g for g, c in v.items() if c.sum() >= f5.MIN_VISITS}
                    r.update(S.fb_test(rec["counts"], null_sel[s], fb, fix))
                r["mde_eps"] = S.mde_of("vllm", fname, s)
                if rec["n"] != A.n_a():
                    r.update(status=f"INCOMPLETE (N={rec['n']})", p=1.0)
                rows.append(r)
        k = A.key("vllm", fname, ARM, "G")
        if (out / "chunks" / k.replace("/", "__")).exists():
            rec = S.read_cell(out, k)
            if fname == "FA":
                t = refs.load("vllm", "vllm_FA_NULL_S0_V2.npz")["eng"]
                idx = {tuple(g): i for i, g in enumerate(fx.nonterminal_prefixes())}
                chain, margin = S.greedy_chain(lambda g: t[idx[g]] if g in idx else None, fix)
            else:
                fb = refs.fb_rows("vllm", "S0", "vllm_FB_NULL_S0_V2.npz")
                chain, margin = S.greedy_chain(lambda g: fb.get(g), fix)
            hits = sum(m for sq, m in rec["counts"].items() if fix.cut(sq) == chain)
            rows.append(dict(key=k, kind="greedy", N=rec["n"], argmax_chain=chain, min_top2_margin=margin, matches=hits,
                             status="REFERENCE_UNRESOLVED" if chain is None else "PASS" if hits == rec["n"] else "FAIL"))
    tested = [r for r in rows if "p" in r]
    for kind in ("cell", "null"):
        grp = [r for r in tested if r["kind"] == kind]
        for r, x in zip(grp, S.holm_over([r["p"] for r in grp], len(grp), ALPHA)):
            r["holm_reject"] = bool(x)
    bad_null = {(r["fixture"], r["setting"]) for r in tested if r["kind"] == "null" and r.get("holm_reject")}
    for r in tested:
        if r["kind"] == "cell" and "status" not in r:
            r["status"] = ("INCONCLUSIVE (null rejected)" if (r["fixture"], r["setting"]) in bad_null else
                           "REJECTED" if r["holm_reject"] else "NOT_REJECTED")
        elif r["kind"] == "null" and "status" not in r:
            r["status"] = "REJECTED" if r["holm_reject"] else "NOT_REJECTED"
    lines = []
    for r in rows:
        extra = ""
        if "p" in r:
            extra = f" N {r['N']} p {r['p']:.3g} mde {r.get('mde_eps')}"
            if r.get("spec_counters", {}).get("spec_decode_num_draft_tokens"):
                sc = r["spec_counters"]
                extra += f" acc {sc['spec_decode_num_accepted_tokens'] / sc['spec_decode_num_draft_tokens']:.3f}"
        if r["kind"] == "greedy" and "matches" in r:
            extra = f" {r['matches']}/{r['N']} on chain {r['argmax_chain']} (margin {r['min_top2_margin']:.3g})"
        lines.append(f"{r['key']:36s} {r['kind']:6s} {r['status']}{extra}")
    (out / "E3.json").write_text(json.dumps(dict(alpha=ALPHA, gate=gv, rows=rows), indent=1, default=str) + "\n")
    (out / "E3.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
