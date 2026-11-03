"""D-40 full study, Stage A statistics (PREREGISTRATION.md §5.2-§5.4, §6; FR9 item 2, FR12 F5, FR13).

Runs on CPU from the committed chunk files and the gate's R_eng tables (§8); imports no inference engine.

    D40_WORK=../data python stagea_analyze.py stagea/data      # -> stagea/data/ANALYSIS_A.{json,txt}

- **F-A:** pooled Pearson χ² over the 2,801 outcomes against the sequence distribution of the engine's own R_eng
  (`stats.pooled_chi2`, as in the pilot).
- **F-B:** per prefix with ≥ 200 visits *in the matching null*, the pooled χ² of the cell's next-token counts against
  that prefix's R_eng row; Fisher-combined (`f5_mde_full.fisher_p`, the rule F5 used: no-dof prefixes skipped; a
  count on a zero-expectation pooled bin gives p = 0, as `stats.pooled_chi2` does for F-A).
- **Malformed draws.** A draw that fails `Fixture.raw_ok` (wrong length, id out of range, EOS not last) sets the
  cell's p to 0 on either fixture: the engine produced an output R_eng gives probability 0.
- **Completeness.** A cell, null or control whose merged N differs from its plan (N_A; 1,000 for greedy) is
  `INCOMPLETE` with p = 1; `stagea_check` should already have listed it.
- **Families.** Grid: Holm at α = 0.01 over m = 118 (FR9 item 2), every cell not run at p = 1. Nulls: Holm at 0.01
  over the 47 null cells of the §4 grid (F-A 5 engines × 7 settings, F-B 4 engines × 3), not-run nulls at p = 1
  (FR9 item 2), and also over the running nulls only. A null rejected under **either** makes the cells of the same
  engine × fixture × setting INCONCLUSIVE (§5.2): the stricter reading for any failure claim. Controls: each must reject at p < α/118 (§5.3, FR9).
- **Greedy G:** every run must equal the argmax chain of the fixture's S0 R_eng (§5.4).
- **MDE:** F5's value at N_A (`phase0b/f5_mde_{fa,fb}.json`); ε > 0.02, or above F5's search bound, is
  `UNDERPOWERED`.

R_eng per draw path: HF cells and controls draw at batch 1 and use the `batch1` table, the HF null the `batch1000`
table; llama.cpp uses the production `np16` tables; SGLang its `default`-backend tables.
"""
import collections
import gzip
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
os.environ.setdefault("D40_WORK", str(HERE.parent / "data"))

import f5_mde_full as f5  # noqa: E402
import fixture as fx  # noqa: E402
import p0b_common as C  # noqa: E402
import rnp  # noqa: E402
import stagea as A  # noqa: E402
import stats  # noqa: E402

ALPHA, M_GRID = stats.ALPHA, stats.M_FULL
assert not (HERE / "stagea" / "data" / "stageA_hf_t4").exists(), "HF T4 fallback present: analyse it instead (FR13)"
M_NULL = 5 * 7 + 4 * 3  # §4: F-A nulls of 5 engines x S0-S6, F-B nulls of vLLM, SGLang, llama.cpp, TRT-LLM x S0/S2/S6
UNDERPOWERED_EPS = 0.02
N_A = 200_000
F5 = {f: json.load(open(HERE / "phase0b" / f"f5_mde_{f.lower()}.json"))["cells"] for f in ("FA", "FB")}
FB_ORDER = dict(vllm="vllm", trtllm="trtllm", llamacpp="lcpp")


def table_name(engine, fname, method, setting, kind):
    if engine == "vllm":
        return f"vllm_{fname}_NULL_{setting}_V1.npz"
    if engine == "sglang":
        return f"sglang_{fname}_{setting}_default.npz"
    if engine == "trtllm":
        return f"trtllm_{fname}_{setting}.npz"
    if engine == "llamacpp":
        return f"lcpp_{fname}_{setting}_np16.npz"
    return f"hf_FA_{setting}_{'batch1000' if kind == 'null' else 'batch1'}.npz"


class Refs:
    def __init__(self, data, sub=None):
        self.data, self.sub, self.cache = data, sub, {}

    def load(self, engine, name):
        k = (engine, name)
        if k not in self.cache:
            z = np.load(self.data / (self.sub or f"stageA_{engine}") / "gate_tables" / name)
            self.cache[k] = {n: z[n] for n in z.files}
        return self.cache[k]

    def fa_seq(self, engine, name):
        k = ("seq", engine, name)
        if k not in self.cache:
            p = rnp.sequence_probs(self.load(engine, name)["eng"])
            self.cache[k] = p / p.sum()
        return self.cache[k]

    def fb_rows(self, engine, setting, name):
        """{prefix tuple: R_eng row} for the F-B scoring set. llama.cpp stores its prefixes; vLLM and TRT-LLM
        scored C.fb_prefixes(order, setting) in that order, rebuilt here and checked against the table length."""
        k = ("fb", engine, name)
        if k not in self.cache:
            t = self.load(engine, name)
            if "prefixes" in t:
                pre = [tuple(int(x) for x in row if x >= 0) for row in t["prefixes"]]
            else:
                pre = [tuple(g) for g in C.fb_prefixes(FB_ORDER[engine], setting)[0]]
            assert len(pre) == len(t["eng"]), (engine, setting, len(pre), t["eng"].shape)
            self.cache[k] = dict(zip(pre, t["eng"]))
        return self.cache[k]


def read_cell(folder, k):
    """Merged counts of a cell's chunks, with the chunk-level metadata the write-up needs."""
    d = folder / "chunks" / k.replace("/", "__")
    counts, n, secs, spec = collections.Counter(), 0, 0.0, collections.Counter()
    files = sorted(d.glob("c*.json.gz"))
    for f in files:
        c = json.load(gzip.open(f, "rt"))
        assert c["key"] == k, (f, c["key"])
        for s, m in zip(c["seqs"], c["counts"]):
            counts[tuple(s)] += m
        n += c["n"]
        secs += float(c.get("draw_seconds") or 0)
        for kk, v in (c.get("spec_counters") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                spec[kk] += v
    assert sum(counts.values()) == n
    return dict(counts=counts, n=n, chunks=len(files), draw_seconds=round(secs, 1), spec_counters=dict(spec))


def malformed(counts, fix):
    return sum(m for s, m in counts.items() if not fix.raw_ok(s))


def fa_test(counts, probs, fix):
    idx = {tuple(o): i for i, o in enumerate(fx.outcomes())}
    vec, bad = np.zeros(len(idx)), 0
    for s, m in counts.items():
        s = fix.cut(s)
        if s in idx:
            vec[idx[s]] += m
        else:
            bad += m
    n = vec.sum()
    stat, p, dof = stats.pooled_chi2(vec, probs)
    bad += malformed(counts, fix)
    if bad:
        p = 0.0  # a malformed or out-of-space draw has probability 0 under R_eng
    return dict(chi2=stat, dof=dof, p=p, out_of_space=bad, zero_prob_hits=int(vec[probs == 0].sum()),
                tv_empirical=float(0.5 * np.abs(vec / max(n, 1) - probs).sum()))


def visits(counts, fix):
    """Next-token counts at every visited prefix of the trie."""
    out = collections.defaultdict(lambda: np.zeros(fix.V))
    for s, m in counts.items():
        if not fix.raw_ok(s):
            continue  # counted by malformed(); the cell's p is 0
        s = fix.cut(s)
        for j, t in enumerate(s):
            out[s[:j]][t] += m
    return out


def fb_test(counts, sel, rows, fix):
    v = visits(counts, fix)
    missing = [g for g in sel if g not in rows]
    keys = sorted(g for g in sel if g in rows)
    p = f5.fisher_p([v[g] if g in v else np.zeros(fix.V) for g in keys], [rows[g] for g in keys])
    bad = malformed(counts, fix)
    if bad:
        p = 0.0
    informative = sum(1 for g in keys if stats.pooled_chi2(v[g] if g in v else np.zeros(fix.V), rows[g])[2] >= 1)
    return dict(p=p, prefixes=len(keys), informative_prefixes=informative, missing_reference=len(missing),
                out_of_space=bad)


def greedy_chain(rows_of, fix):
    seq, margins = (), []
    while len(seq) < fix.MAX_NEW and (not seq or seq[-1] != fix.EOS):
        row = rows_of(seq)
        if row is None:
            return None, None
        top2 = np.sort(row)[-2:]
        margins.append(float(top2[1] - top2[0]))
        seq += (int(np.argmax(row)),)
    return seq, min(margins)


def holm_over(pvals, m, alpha=ALPHA):
    """Holm at alpha over m hypotheses, the len(pvals) given plus m - len(pvals) at p = 1."""
    allp = np.concatenate([np.asarray(pvals, float), np.ones(m - len(pvals))])
    return stats.holm(allp, alpha)[: len(pvals)]


def mde_of(engine, fname, setting):
    c = F5[fname].get(f"{A.MDE_KEY[engine]}_{setting}")
    return None if c is None else c["mde_eps_NA"]


def analyze(data):
    refs = Refs(data)
    rows, gates = [], {}
    for engine in A.ENGINES:
        folder = data / f"stageA_{engine}"
        gate = json.load(open(folder / "gate.json"))
        gates[engine] = {k: v["passed"] for k, v in gate["verdicts"].items()}
        os.environ["D40_STAGEA_OUT"] = str(folder)
        cells = A.cells_of(engine, gate)
        null_sel = {}
        # nulls first: they fix the F-B prefix sets
        for kind, fname, method, s in sorted(cells, key=lambda c: c[0] != "null"):
            k = A.key(engine, fname, method, s)
            fix = C.fixture(fname)
            rec = read_cell(folder, k)
            r = dict(key=k, kind=kind, engine=engine, fixture=fname, method=method, setting=s, N=rec["n"],
                     chunks=rec["chunks"], draw_seconds=rec["draw_seconds"], spec_counters=rec["spec_counters"],
                     distinct=len(rec["counts"]))
            if kind == "greedy":
                tname = table_name(engine, fname, method, "S0", kind)
                if fname == "FA":
                    t = refs.load(engine, tname)["eng"]
                    idx = {tuple(g): i for i, g in enumerate(fx.nonterminal_prefixes())}
                    chain, margin = greedy_chain(lambda g: t[idx[g]] if g in idx else None, fix)
                else:
                    fb = refs.fb_rows(engine, "S0", tname)
                    chain, margin = greedy_chain(lambda g: fb.get(g), fix)
                hits = sum(m for sq, m in rec["counts"].items() if fix.cut(sq) == chain)
                r.update(argmax_chain=chain, min_top2_margin=margin, matches=hits, mismatches=rec["n"] - hits,
                         other_outputs={str(list(sq)): m for sq, m in rec["counts"].items() if fix.cut(sq) != chain},
                         status="REFERENCE_UNRESOLVED (argmax chain leaves the scored prefixes)" if chain is None
                         else "INCOMPLETE" if rec["n"] != A.n_of(kind) else "PASS" if hits == rec["n"] else "FAIL")
                rows.append(r)
                continue
            tname = table_name(engine, fname, method, s, kind)
            if fname == "FA":
                r.update(fa_test(rec["counts"], refs.fa_seq(engine, tname), fix))
                r["mde_eps"] = mde_of(engine, "FA", s)
            else:
                fb = refs.fb_rows(engine, s, tname)
                if kind == "null":
                    v = visits(rec["counts"], fix)
                    null_sel[s] = {g for g, c in v.items() if c.sum() >= f5.MIN_VISITS}
                r.update(fb_test(rec["counts"], null_sel[s], fb, fix))
                r["mde_eps"] = mde_of(engine, "FB", s)
                if r["missing_reference"]:
                    r["status"] = "REFERENCE_UNRESOLVED (null-selected prefix not scored)"
                    r["p"] = 1.0
            r["underpowered"] = r["mde_eps"] is None or r["mde_eps"] > UNDERPOWERED_EPS
            if rec["n"] != A.n_of(kind):  # F5's MDE is for N_A only; a short cell is not tested
                r.update(status=f"INCOMPLETE (N={rec['n']})", p=1.0, underpowered=True)
            rows.append(r)

    grid = [r for r in rows if r["kind"] == "cell"]
    nulls = [r for r in rows if r["kind"] == "null"]
    ctrls = [r for r in rows if r["kind"] == "control"]
    for r, x in zip(grid, holm_over([r["p"] for r in grid], M_GRID)):
        r["holm_reject"] = bool(x)
    for r, x in zip(nulls, holm_over([r["p"] for r in nulls], M_NULL)):
        r["holm_reject"] = bool(x)
    for r, x in zip(nulls, holm_over([r["p"] for r in nulls], len(nulls))):
        r["holm_reject_running_only"] = bool(x)
    level = ALPHA / M_GRID
    for r in ctrls:
        r["reject_at_alpha_over_m"] = bool(r["p"] < level)
    rejected_nulls = {(r["engine"], r["fixture"], r["setting"]) for r in nulls
                      if r["holm_reject"] or r["holm_reject_running_only"]}
    for r in grid:
        null_bad = (r["engine"], r["fixture"], r["setting"]) in rejected_nulls
        r["status"] = r.get("status") or ("INCONCLUSIVE (null rejected)" if null_bad else
                                          "REJECTED" if r["holm_reject"] else "NOT_REJECTED")
    greedy = [r for r in rows if r["kind"] == "greedy"]
    # §5.3: every planned control must run and reject; one dropped by its gate leaves the harness unvalidated
    planned_ctrls = [A.key(e, "FA", c, s) for e in A.ENGINES for c, s in A.RUNNING[e]["controls"]]
    summary = dict(
        stage="A", alpha=ALPHA, m_grid=M_GRID, m_null=M_NULL, control_level=level,
        grid_cells_run=len(grid), nulls_run=len(nulls), controls_run=len(ctrls), greedy_run=len(greedy),
        gates=gates,
        controls_planned=planned_ctrls,
        controls_missing=sorted(set(planned_ctrls) - {r["key"] for r in ctrls}),
        harness_valid=bool(ctrls) and set(planned_ctrls) <= {r["key"] for r in ctrls}
        and all(r["reject_at_alpha_over_m"] for r in ctrls),
        controls_failing=[r["key"] for r in ctrls if not r["reject_at_alpha_over_m"]],
        nulls_rejected=[r["key"] for r in nulls if r["holm_reject"] or r["holm_reject_running_only"]],
        incomplete=[r["key"] for r in rows if str(r.get("status", "")).startswith("INCOMPLETE")],
        nulls_rejected_running_only=[r["key"] for r in nulls if r["holm_reject_running_only"]],
        grid_rejected=[r["key"] for r in grid if r["holm_reject"]],
        greedy_failed=[r["key"] for r in greedy if r["status"] != "PASS"],
        underpowered=[r["key"] for r in grid if r["underpowered"]],
    )
    # §5.4: Stage B takes every Stage-A-rejected cell plus its null
    cand = [r for r in grid if r["holm_reject"]]
    summary["stageB_cells"] = [r["key"] for r in cand]
    summary["stageB_nulls"] = sorted({r["key"] for r in nulls for c in cand
                                      if (r["engine"], r["fixture"], r["setting"]) == (c["engine"], c["fixture"],
                                                                                       c["setting"])})
    # §5.2 says nothing on nulls of INCONCLUSIVE cells; Stage B redraws them anyway (more data, no gain toward pass)
    summary["rows"] = rows
    return summary


def show(s, out):
    lines = []
    for r in s["rows"]:
        if r["kind"] == "greedy":
            lines.append(f"{r['key']:28s} greedy N={r['N']} {r['status']} {r['matches']}/{r['N']} "
                         f"chain={r['argmax_chain']} margin={r['min_top2_margin']:.3g}")
            continue
        flag = r.get("reject_at_alpha_over_m", r.get("holm_reject"))
        extra = (f"chi2={r['chi2']:.1f}/{r['dof']} tv={r['tv_empirical']:.4f} zero={r['zero_prob_hits']} "
                 f"oos={r['out_of_space']}" if r["fixture"] == "FA" else
                 f"prefixes={r['prefixes']} informative={r['informative_prefixes']} missing={r['missing_reference']} "
                 f"oos={r['out_of_space']}")
        mde = "censored" if r["mde_eps"] is None else f"{r['mde_eps']:.4f}"
        lines.append(f"{r['key']:28s} {r['kind']:7s} N={r['N']} p={r['p']:.3g} {extra} mde={mde} "
                     f"reject={flag} {r.get('status', '')}")
    for k in ("m_grid", "m_null", "control_level", "grid_cells_run", "nulls_run", "controls_run", "greedy_run",
              "incomplete", "controls_missing", "harness_valid", "controls_failing", "nulls_rejected", "nulls_rejected_running_only", "grid_rejected",
              "greedy_failed", "underpowered", "stageB_cells", "stageB_nulls"):
        lines.append(f"{k}: {s[k]}")
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text)


FR14_KEYS = {f"trtllm/{f}/{m}/{st}" for f, sts in (("FA", ("S0", "S1", "S2", "S3")), ("FB", ("S0", "S2")))
             for m in ("TG1v", "NULL") for st in sts}
if os.environ.get("D40_STAGEB_EXPECTED"):  # CPU dry run only (stageb_cpu.sh): its own cell list
    FR14_KEYS = set(os.environ["D40_STAGEB_EXPECTED"].split(","))


def analyze_b(data):
    """FR14: every stageB_* folder; Stage A's tests at N_B, the F-B prefix set from the Stage B null, Bonferroni at
    0.01 per family (cells, nulls). A cell is confirmed if p < 0.01/k and its null is rejected in neither stage."""
    n_b = int(os.environ.get("D40_STAGEB_N", A.N_B))  # the CPU dry run sets a small N; production is N_B (§5.4)
    os.environ["D40_STAGE"], os.environ["D40_STAGEA_NA"] = "B", str(n_b)  # F-B scoring set at N_B (FR14)
    a_rows = {r["key"]: r for r in json.load(open(data / "ANALYSIS_A.json"))["rows"]}
    rows, gates = [], {}
    for folder in sorted(data.glob("stageB_*")):
        spec = json.load(open(folder / "spec.json"))
        assert spec.get("stage") == "B" and spec["n_a"] == n_b, (folder, spec)
        engine = spec["keys"][0].split("/")[0]
        os.environ["D40_STAGEB_KEYS"], os.environ["D40_STAGEA_OUT"] = ",".join(spec["keys"]), str(folder)
        gate = json.load(open(folder / "gate.json"))
        gates[folder.name] = {k: v["passed"] for k, v in gate["verdicts"].items()}
        refs, null_sel = Refs(data, folder.name), {}
        cells = A.cells_of(engine, gate)
        drawn = {A.key(engine, *c[1:]) for c in cells}
        for k in spec["keys"]:
            if k not in drawn:  # its setting failed this folder's gate
                kind = "null" if "/NULL" in k else "cell"
                rows.append(dict(key=k, kind=kind, folder=folder.name, p=1.0, status="REFERENCE_UNRESOLVED (Stage B gate)",
                                 engine=engine, fixture=k.split("/")[1], setting=k.split("/")[3]))
        for kind, fname, method, s in sorted(cells, key=lambda c: c[0] != "null"):
            k = A.key(engine, fname, method, s)
            fix = C.fixture(fname)
            rec = read_cell(folder, k)
            r = dict(key=k, kind=kind, folder=folder.name, engine=engine, fixture=fname, method=method, setting=s,
                     N=rec["n"], chunks=rec["chunks"], draw_seconds=rec["draw_seconds"],
                     spec_counters=rec["spec_counters"], distinct=len(rec["counts"]))
            tname = table_name(engine, fname, method, s, kind)
            if fname == "FA":
                r.update(fa_test(rec["counts"], refs.fa_seq(engine, tname), fix))
            else:
                fb = refs.fb_rows(engine, s, tname)
                if kind == "null":
                    v = visits(rec["counts"], fix)
                    null_sel[s] = {g for g, c in v.items() if c.sum() >= f5.MIN_VISITS}
                r.update(fb_test(rec["counts"], null_sel[s], fb, fix))
                if r["missing_reference"]:
                    r.update(status="REFERENCE_UNRESOLVED (null-selected prefix not scored)", p=1.0)
            if rec["n"] != n_b:
                r.update(status=f"INCOMPLETE (N={rec['n']})", p=1.0)
            rows.append(r)
    # FR14 fixes the families: 6 cells and 6 nulls, all four folders present
    assert {r["key"] for r in rows} == FR14_KEYS and len(rows) == len(FR14_KEYS), sorted({r["key"] for r in rows})
    fam = {f: [r for r in rows if r["kind"] == f] for f in ("cell", "null")}
    for f, rs in fam.items():
        for r in rs:
            r["bonferroni_level"] = ALPHA / len(rs)
            r["reject"] = bool(r["p"] < ALPHA / len(rs))
    null_bad = {(r["engine"], r["fixture"], r["setting"]) for r in fam["null"] if r["reject"]}
    null_bad |= {(r["engine"], r["fixture"], r["setting"]) for r in a_rows.values()
                 if r["kind"] == "null" and (r.get("holm_reject") or r.get("holm_reject_running_only"))}
    # §6: the null must pass both stages, i.e. be tested; an untested (incomplete or unresolved) null does not pass
    untested = {(r["engine"], r["fixture"], r["setting"]) for r in fam["null"] if "status" in r}
    for r in fam["cell"]:
        a_null = a_rows.get(f"{r['engine']}/{r['fixture']}/NULL/{r['setting']}")
        if (a_null is None or "status" in a_null) and "status" not in r:
            r["status"] = "INCONCLUSIVE (Stage A null untested)"
        if (r["engine"], r["fixture"], r["setting"]) in untested and "status" not in r:
            r["status"] = "INCONCLUSIVE (Stage B null untested)"
        nb = (r["engine"], r["fixture"], r["setting"]) in null_bad
        r["confirmed"] = r["reject"] and not nb and "status" not in r
        r["status"] = r.get("status") or ("CONFIRMED" if r["confirmed"] else
                                          "INCONCLUSIVE (null rejected)" if nb else "NOT_CONFIRMED")
        ra = a_rows.get(r["key"], {})
        r["stageA_p"], r["stageA_tv"] = ra.get("p"), ra.get("tv_empirical")
    return dict(stage="B", alpha=ALPHA, cells=len(fam["cell"]), nulls=len(fam["null"]), gates=gates,
                confirmed=[r["key"] for r in fam["cell"] if r["confirmed"]],
                nulls_rejected=[r["key"] for r in fam["null"] if r["reject"]], rows=rows)


def show_b(s, out):
    lines = []
    for r in s["rows"]:
        extra = (f"chi2={r.get('chi2', float('nan')):.1f}/{r.get('dof')} tv={r.get('tv_empirical', float('nan')):.4f} "
                 f"zero={r.get('zero_prob_hits')}" if r["fixture"] == "FA" else
                 f"prefixes={r.get('prefixes')} informative={r.get('informative_prefixes')} "
                 f"missing={r.get('missing_reference')}")
        lines.append(f"{r['key']:24s} {r['kind']:5s} N={r.get('N')} p={r['p']:.3g} {extra} "
                     f"level={r.get('bonferroni_level', float('nan')):.2g} {r.get('status', '')}")
    for k in ("cells", "nulls", "nulls_rejected", "confirmed"):
        lines.append(f"{k}: {s[k]}")
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text)


def dump(s, path):
    json.dump(s, open(path, "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else list(o) if isinstance(o, (tuple, set)) else str(o))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--stage-b"]
    data = pathlib.Path(args[0] if args else HERE / "stagea" / "data").resolve()
    if "--stage-b" in sys.argv:  # FR14:  python stagea_analyze.py stagea/data --stage-b
        s = analyze_b(data)
        dump(s, data / "ANALYSIS_B.json")
        show_b(s, data / "ANALYSIS_B.txt")
    else:
        s = analyze(data)
        dump(s, data / "ANALYSIS_A.json")
        show(s, data / "ANALYSIS_A.txt")
