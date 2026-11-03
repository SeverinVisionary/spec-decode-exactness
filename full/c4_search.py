"""Exhaustive context search for the recorded TensorRT-LLM draft-logits rows (follow-up to `c4_diag.py`). CPU only.
Not a preregistered test: it changes no FR16 flag.

    D40_WORK=$PWD/../data python c4_search.py stagea/data/repro_trtllm      # writes C4_SEARCH.{txt,json} there

Each distinct recorded `py_draft_logits` row (rounded to 1e-4) is compared, as centred logits by max |Δ|, with the
fp64 draft and target models at every context: the F-A prompt [1,2,3,1,2] followed by any continuation of 0..6
tokens (V = 8; 299,593 contexts). RoPE is relative, so the absolute start position does not matter. A best match
within 1e-3 means that context is the one the drafter actually conditioned on. No match means the row is not a
clean forward pass of either fp64 model at any such context.
"""
import collections
import gzip
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import rnp  # noqa: E402

PROMPT = [1, 2, 3, 1, 2]
V, MAX_CONT, MATCH = 8, 6, 1e-3


def centred(v):
    v = np.asarray(v, float)
    return v - v.mean()


def main(folder):
    folder = pathlib.Path(folder)
    rows = collections.OrderedDict()  # rounded row -> (row, [(run key, tokens_before, draft, row index)])
    for arm in ("A1", "A2", "A3"):
        a = json.load(gzip.open(folder / f"{arm}.json.gz", "rt"))
        for run in a["runs"]:
            if "events" not in run:
                continue
            key = f"{arm}/{run['setting']}/u{run['u']}"
            for e in run["events"]:
                for r, row in enumerate(e["draft_logits"] or []):
                    k = tuple(np.round(centred(row), 4))
                    rows.setdefault(k, (centred(row), []))[1].append(
                        (key, e["tokens_before"], e["draft_tokens"][:e["draft_token_length"]], r))
    keys = list(rows)
    R = np.stack([rows[k][0] for k in keys])  # (n_rows, V)
    best = {m: (np.full(len(keys), np.inf), [None] * len(keys)) for m in ("draft", "target")}
    for m in best:
        model = rnp.Llama64(m)
        bd, bc = best[m]
        for n in range(MAX_CONT + 1):
            for cont in itertools.product(range(V), repeat=n):
                ctx = PROMPT + list(cont)
                d = np.abs(R - centred(model.logits(ctx))).max(1)
                better = d < bd
                for i in np.nonzero(better)[0]:
                    bd[i], bc[i] = d[i], ctx
    out, lines, summary = [], [], collections.Counter()
    for i, k in enumerate(keys):
        row, occ = rows[k]
        m = min(best, key=lambda mm: best[mm][0][i])
        diff, ctx = float(best[m][0][i]), best[m][1][i]
        key, pre, d, r = occ[0]
        # where the best context sits relative to the first occurrence: base + draft prefix?
        rel = "other"
        for j in range(len(d) + 1):
            if ctx == pre + d[:j]:
                rel = f"pre+d[:{j}]"
            elif ctx == pre[:-1] + d[:j]:
                rel = f"pre[:-1]+d[:{j}]"
        status = "match" if diff <= MATCH else "none"
        summary[(status, m if status == "match" else "-", rel if status == "match" else "-", r)] += len(occ)
        out.append(dict(row=row.tolist(), n_occurrences=len(occ), first=dict(run=key, tokens_before=pre, draft=d,
                                                                                row_index=r),
                        best_model=m, best_diff=diff, best_context=ctx, relation=rel,
                        draft_best_diff=float(best["draft"][0][i]), target_best_diff=float(best["target"][0][i])))
    for (status, m, rel, r), c in sorted(summary.items(), key=lambda kv: -kv[1]):
        lines.append(f"{status:5s} model={m:6s} relation={rel:14s} row{r}: {c} occurrences")
    diffs = np.array([o["best_diff"] for o in out])
    lines.append(f"distinct rows {len(out)}; best |Δ| median {np.median(diffs):.2e}, min {diffs.min():.2e}, "
                 f"max {diffs.max():.2e}")
    (folder / "C4_SEARCH.json").write_text(json.dumps(out, indent=1) + "\n")
    (folder / "C4_SEARCH.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
