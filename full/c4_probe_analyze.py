"""FR17 analysis of the drafter trace (`c4_probe.py` output). CPU only.

    D40_WORK=$PWD/../data python c4_probe_analyze.py stagea/data/c4_probe      # writes C4_PROBE.{json,txt} there

Q1 draft load (P2): max |Δp| between the engine's processed S0 probabilities for the draft model alone and the fp64
   draft's softmax, over the probed prefixes. Clean if ≤ TOL.
Q2 forward fidelity (P1): each draft forward's output row against the fp64 draft at the draft request's own tokens.
   Clean if ≤ TOL; otherwise the drafter conditioned on something other than its tokens (for example stale KV).
Q3 alignment (P1): each verification's `py_draft_logits` row i is located among that round's forward rows
   (|Δ| ≤ MATCH on raw logits), and labelled by the draft position whose token that forward produced: row i is
   "aligned" if it produced d_i, "shifted+s" if it produced d_{i+s}.
Every comparison is at S0 (T = 1), where the processed distribution is the softmax of the logits.
"""
import collections
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import rnp  # noqa: E402

TOL = 1e-4     # probability scale, as δ_num (§5.1)
MATCH = 1e-5   # raw-logit identity between a stored row and a forward row (same tensor, copied)
N_P2 = 100     # c4_probe.N_P2_PREFIXES


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def rows2d(x):
    a = np.asarray(x, float)
    return a.reshape(-1, a.shape[-1]) if a.size else a.reshape(0, 0)


def main(folder):
    folder = pathlib.Path(folder)
    draft = rnp.Llama64("draft")
    cache = {}

    def ref(ids):
        k = tuple(ids)
        if k not in cache:
            cache[k] = softmax(draft.logits(list(ids)))
        return cache[k]

    res = {}
    # Q1
    p2 = json.load(gzip.open(folder / "P2.json.gz", "rt"))
    fx_prompt = [1, 2, 3, 1, 2]
    probs = np.asarray(p2["probs"], float)
    refs = np.stack([ref(fx_prompt + list(g)) for g in p2["prefixes"]])
    complete = probs.shape == refs.shape == (N_P2, 8)
    finite = bool(np.isfinite(probs).all() and np.isfinite(refs).all())
    d1 = np.abs(probs - refs).max(1) if complete else np.array([np.inf])
    copies_ok = bool(np.isfinite(p2["copy_dev"]) and p2["copy_dev"] <= TOL)
    res["Q1_draft_load"] = dict(prefixes=int(len(p2["prefixes"])), complete=bool(complete), finite=finite,
                                max_abs=float(d1.max()), median_abs=float(np.median(d1)), copy_dev=p2["copy_dev"],
                                copies_ok=copies_ok, strategies=p2["strategies"],
                                clean=bool(complete and finite and copies_ok and d1.max() <= TOL))
    # Q2, Q3
    p1 = json.load(gzip.open(folder / "P1.json.gz", "rt"))
    q2 = collections.Counter()
    q2_by_kind = collections.defaultdict(list)
    q3 = collections.Counter()
    kr = collections.Counter()
    examples = []
    for req in p1["requests"]:
        fwd_by_round = collections.defaultdict(list)
        for f in req["forwards"]:
            L = rows2d(f["logits"])
            if len(f["reqs"]) != 1:
                q2["batch_not_single"] += 1
                continue
            r = f["reqs"][0]
            if not np.isfinite(L).all():
                q2["non_finite"] += 1
            # with one request and beam 1 the last output row is the one sampled from
            fwd_by_round[f["round"]].append(dict(row=L[-1], kind=r["kind"], tokens=r["tokens"], dev=None,
                                                 ccp=r["context_current_position"], chunk=r["context_chunk_size"]))
        dec_by_round = collections.defaultdict(list)
        for d in req["decoded"]:
            dec_by_round[d["round"]].append(d)
        # forward k of a round produced the k-th drafted token of that round
        for v in req["verifications"]:
            K, rnd = v["draft_token_length"], v["round"]
            if K == 0 or rnd not in fwd_by_round:
                q3["no-draft-round"] += 1
                continue
            d = v["draft_tokens"][:K]
            dl = rows2d(v["draft_logits"]) if v["draft_logits"] else np.zeros((0, 0))
            kr[f"K{K}/R{len(dl)}"] += 1
            fw = fwd_by_round[rnd]
            # Q2 (review): forward k's effective input is the round's first draft request's tokens plus the k tokens
            # drafted before it; later forwards read the previous token from device tensors before the request's
            # token list is updated, so their recorded tokens lag.
            base = fw[0]["tokens"]
            for k, f in enumerate(fw):
                if k > len(d):
                    q2["forward_beyond_K"] += 1
                    continue
                ctx = base + d[:k]
                f["dev"] = dev = float(np.abs(softmax(f["row"]) - ref(ctx)).max())
                f["ctx"] = ctx
                q2["clean" if dev <= TOL else "unclean"] += 1
                q2_by_kind["first:" + f["kind"] if k == 0 else "later"].append(dev)
            for i, row in enumerate(dl):
                hits = [k for k, f in enumerate(fw) if np.abs(f["row"] - row).max() <= MATCH]
                lab = ("no-forward-match" if not hits else f"ambiguous{hits}" if len(hits) > 1
                       else "aligned" if hits[0] == i else f"shifted{hits[0] - i:+d}")
                q3[f"row{i}:{lab}"] += 1
            if len(examples) < 5 and len(dl):
                examples.append(dict(request=req["i"], round=rnd, draft=d, R=len(dl),
                                     forwards=[dict(kind=f["kind"], tokens=f["tokens"], ctx=f.get("ctx"), dev=f["dev"],
                                                    ccp=f["ccp"], chunk=f["chunk"]) for f in fw],
                                     decoded=[dict(pos=x["draft_position"], tok=x["last_token"],
                                                   gl_rows=None if not isinstance(x["generation_logits"], list)
                                                   else len(x["generation_logits"]))
                                              for x in dec_by_round[rnd]]))
    res["Q2_forward_fidelity"] = dict(counts=dict(q2), by_kind={k: dict(n=len(v), median=float(np.median(v)),
                                                                       max=float(np.max(v)),
                                                                       clean=int(sum(x <= TOL for x in v)))
                                                               for k, v in q2_by_kind.items()})
    res["Q3_alignment"] = dict(counts=dict(q3), KR=dict(kr))
    res["examples"] = examples
    (folder / "C4_PROBE.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    lines = [f"Q1 draft load: {json.dumps(res['Q1_draft_load'])}",
             f"Q2 forward fidelity: {json.dumps(res['Q2_forward_fidelity'])}",
             f"Q3 alignment: {json.dumps(res['Q3_alignment'])}"]
    (folder / "C4_PROBE.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
