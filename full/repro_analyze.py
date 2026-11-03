"""FR16 analysis of the TensorRT-LLM TG1v reproduction (`repro_trtllm.py` output). CPU only.

    D40_WORK=$PWD/../data python repro_analyze.py stagea/data/repro_trtllm      # writes REPRO.{json,txt} there

Per verification event (rejection-sampling path), from the engine's own tensors:
- C1 rows: draft-logits rows R against draft tokens K;
- C2 unverified acceptance: num_accepted > R (a draft token emitted without any verification row);
- C3 support violation: an accepted draft token whose own `py_target_probs` row gives it probability 0;
- C4 alignment: draft-logits row i matched to the fp64 draft model at prefix (tokens, d_<i) ("aligned") or
  (tokens, d_<=i) ("shifted"), at the setting's TorchSampler distribution, max abs difference <= 1e-4.
A3's draws are tested with Stage A's F-A χ² against R_eng (the Stage B L4 gate tables), Bonferroni 0.01/2.
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np

import p0b_common as C
import rnp_full
import stagea_analyze as SA

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import fixture as fx  # noqa: E402
import rnp  # noqa: E402

TOL = 1e-4
ALPHA = 0.01
MIN_RS = 20  # rejection-sampling verifications required per planned events run (coverage gate)
N_FIX = 100_000


def dist(logits, setting):
    return rnp_full.trtllm_dist(np.asarray(logits, float), None, C.SETTINGS[setting])


class Draft:
    def __init__(self):
        self.m, self.cache = rnp.Llama64("draft"), {}

    def row(self, ids, setting):
        k = (tuple(ids), setting)
        if k not in self.cache:
            self.cache[k] = dist(self.m.logits(list(ids)), setting)
        return self.cache[k]


def event_checks(ev, setting, draft):
    K, n = ev["draft_token_length"], ev["num_accepted"]
    d = ev["draft_tokens"][:K]
    R = ev["draft_logits_shape"][0] if ev["draft_logits_shape"] else 0
    tp = np.asarray(ev["target_probs"] or [[]], float)
    tp = tp.reshape(-1, tp.shape[-1]) if tp.size else tp
    out = dict(K=K, R=R, num_accepted=n, unverified=max(0, n - R))
    out["support_violation"] = [i for i in range(min(n, len(tp))) if tp[i][d[i]] == 0.0]
    align = []
    pre = ev["tokens_before"]
    for i in range(R):
        row = dist(ev["draft_logits"][i], setting)
        a = float(np.abs(row - draft.row(pre + d[:i], setting)).max())
        s = float(np.abs(row - draft.row(pre + d[:i + 1], setting)).max()) if i + 1 <= len(d) else float("inf")
        align.append("aligned" if a <= TOL else "shifted" if s <= TOL else f"neither({a:.2g},{s:.2g})")
    out["alignment"] = align
    return out


def summarize_events(run, draft):
    rows = [event_checks(ev, run["setting"], draft) for ev in run["events"] if ev["path"] == "rejection_sampling"]
    paths = collections.Counter(ev["path"] for ev in run["events"])
    return dict(setting=run["setting"], u=run["u"], events=len(run["events"]), paths=dict(paths),
                KR=dict(collections.Counter(f"K{r['K']}/R{r['R']}" for r in rows)),
                unverified_events=sum(r["unverified"] > 0 for r in rows),
                unverified_tokens=sum(r["unverified"] for r in rows),
                support_violation_events=sum(bool(r["support_violation"]) for r in rows),
                alignment=dict(collections.Counter(a for r in rows for a in r["alignment"])),
                example=next((dict(ev, **r) for ev, r in zip(
                    [e for e in run["events"] if e["path"] == "rejection_sampling"], rows)
                    if r["unverified"] or r["support_violation"]), None))


def draws_test(run, data):
    fix = C.FA
    counts = collections.Counter(tuple(s) for s in run["seqs"])
    probs = SA.Refs(data, sub="stageB_trtllm_fa01").fa_seq("trtllm", f"trtllm_FA_{run['setting']}.npz")
    r = SA.fa_test(counts, probs, fix)
    return dict(setting=run["setting"], n=run["n"], **{k: r[k] for k in ("p", "chi2", "dof", "tv_empirical",
                                                                         "zero_prob_hits", "out_of_space")},
                level=ALPHA / 2,
                reject=bool(r["p"] < ALPHA / 2 or r["zero_prob_hits"] or r["out_of_space"]))


def main(folder):
    folder = pathlib.Path(folder)
    data = folder.parent
    draft, res = Draft(), {}
    for arm in ("A1", "A2", "A3"):
        p = folder / f"{arm}.json.gz"
        if not p.exists():
            res[arm] = "MISSING"
            continue
        a = json.load(gzip.open(p, "rt"))
        res[arm] = dict(hooks=a["hooks"], runs=[summarize_events(r, draft) if "events" in r else draws_test(r, data)
                                                for r in a["runs"]])
    runs = {arm: res[arm]["runs"] if isinstance(res.get(arm), dict) else [] for arm in ("A1", "A2", "A3")}
    plan = dict(A1=[("S0", 0), ("S0", 1), ("S2", 0), ("S2", 1)], A2=[("S2", 1)], A3=[("S2", 1)])

    def ev_ok(arm):
        """every planned events run present, each with at least MIN_RS rejection-sampling verifications"""
        got = {(r["setting"], r["u"]): r for r in runs[arm] if "u" in r}
        return all(k in got and got[k]["paths"].get("rejection_sampling", 0) >= MIN_RS for k in plan[arm])

    draws = {r["setting"]: r for r in runs["A3"] if "n" in r}
    draws_ok = all(s in draws and draws[s]["n"] == N_FIX for s in ("S0", "S2"))
    a1, s2u1 = runs["A1"], [r for r in runs["A1"] if r.get("setting") == "S2" and r.get("u") == 1]
    res["coverage"] = dict(A1=ev_ok("A1"), A2=ev_ok("A2"), A3_events=ev_ok("A3"), A3_draws=draws_ok)
    covered = all(res["coverage"].values())  # FR16 amendment: every verdict needs all planned coverage
    reproduced = covered and any(r["unverified_events"] for r in a1) and \
        any(r["support_violation_events"] for r in s2u1)
    res["verdict"] = dict(
        reproduced=reproduced,
        hook_excluded=reproduced and all(r["unverified_events"] and r["support_violation_events"]
                                         for r in runs["A2"]),
        fix_clean_events=reproduced and ev_ok("A3") and all(
            not r["unverified_events"] and not r["support_violation_events"] for r in runs["A3"] if "u" in r),
        fix_draws_pass=reproduced and draws_ok and all(not r["reject"] for r in draws.values()),
    )
    (folder / "REPRO.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    lines = [f"{k}: {v}" for k, v in res["verdict"].items()]
    for arm in ("A1", "A2", "A3"):
        if isinstance(res.get(arm), dict):
            for r in res[arm]["runs"]:
                r = {k: v for k, v in r.items() if k != "example"}
                lines.append(f"{arm} {json.dumps(r)}")
    (folder / "REPRO.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
