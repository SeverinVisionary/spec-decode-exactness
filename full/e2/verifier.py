"""FR22 E2: one chain verification step, correct (M0) or with a frozen mutation, plus the trace it leaves.

Both engine adapters (`hf_e2.py`, `vllm_e2.py`) replace the engine's own verification function with `verify`, which
receives the engine's draft tokens, draft rows Q (the distributions the drafts were drawn from) and processed target
rows P, and returns the emitted tokens. Mutations (FR22 E2, = E1's classes):
- M1 = D1: the last draft row is missing, so the last draft token is accepted unchecked;
- M2 = D2: draft token j is checked against Q[j + 1]; the last one is unchecked; 0/0 rejects;
- M3 = D3: M2 with 0/0 accepted;
- M4 = D4: the residual is drawn from P[j] instead of norm((P[j] - Q[j])+);
- M6 = D8: the target row at position j >= 1 (and the bonus row) is the row of position j - 1.
M5 (bonus from q) is NOT_APPLICABLE in both engines: neither computes a draft row at the bonus position, so there is
no q to draw the bonus from without adding a drafter step (an engine change, not a monkeypatch).

Greedy requests (vLLM only; HF's greedy path does not call the verifier) accept d_j iff it is the argmax of the row
used, emit that argmax otherwise, and take the bonus as the argmax of the bonus row used (or the engine's bonus).

The trace keeps, per verification, the generated prefix (when the adapter knows it), the tokens, the rows the verifier
used and the true rows, so the trace oracle (`trace_oracle.py`) can check them. `effect` counts verifications where
the mutation changed what the verifier did at a position it actually reached (§5.3 rule 1).
"""
import torch

MODES = ("M0", "M1", "M2", "M3", "M4", "M6")


class Trace:
    """`calib` accumulates FR21's calibration data in place: per position j, rows grouped by their 1e-3 rounding,
    each group holding the emitted-token counts and the summed rows (the expected counts); `impossible` counts emitted
    tokens with probability 0 in their row."""

    def __init__(self, keep=1000):
        self.keep, self.verifs, self.calib, self.impossible, self.effect, self.n = keep, [], {}, {}, 0, 0

    def add(self, rec, calib, touched):
        self.n += 1
        self.effect += int(touched)
        if len(self.verifs) < self.keep:
            self.verifs.append(rec)
        if rec["greedy"]:
            return
        for j, (t, row) in enumerate(zip(*calib)):
            s = sum(row) or 1.0
            row = [v / s for v in row]
            g = self.calib.setdefault(j, {}).setdefault(tuple(round(v, 3) for v in row), [[0] * len(row), [0.0] * len(row)])
            g[0][t] += 1
            g[1] = [a + b for a, b in zip(g[1], row)]
            if row[t] <= 0:
                self.impossible[j] = self.impossible.get(j, 0) + 1

    def dump(self):
        return dict(n_verifications=self.n, effect=self.effect, verifications=self.verifs,
                    impossible={str(j): v for j, v in self.impossible.items()},
                    calibration=[[j, list(k), g[0], [round(x, 6) for x in g[1]]]
                                 for j, gs in self.calib.items() for k, g in gs.items()])


def _rows(x):
    return [[round(float(v), 7) for v in r] for r in x]


def _calib(emitted, P, K, bonus_row):
    rows = [P[j] if j < K else bonus_row for j in range(len(emitted)) if j < K or bonus_row is not None]
    return [emitted[:len(rows)], _rows(rows)]


def verify(d, Q, P, mode, gen, bonus_row=None, bonus_token=None, trace=None, prefix=None, greedy=False):
    """d: [K] draft tokens; Q: [K, V] draft rows; P: [K, V] target rows at the draft positions (true rows); all on
    `gen`'s device. The bonus is drawn from `bonus_row` (the target row after the last draft) when given, else the
    engine's own `bonus_token` is used. Returns (emitted tokens, number of accepted drafts)."""
    K = len(d)
    Qu, Pu = Q.clone(), P.clone()
    brow = bonus_row
    unchecked = set()
    if mode == "M1":
        unchecked.add(K - 1)
    if mode in ("M2", "M3"):
        if K > 1:
            Qu[:-1] = Q[1:]
        unchecked.add(K - 1)
    if mode == "M6" and K > 1:
        Pu[1:] = P[:-1]
        brow = P[-1]
    emitted, n_acc, residual_at, touched = [], 0, None, False
    for j in range(K):
        t = int(d[j])
        if j in unchecked:
            touched = True
        elif not torch.equal(Pu[j], P[j]) or (not greedy and not torch.equal(Qu[j], Q[j])):
            touched = True
        if greedy:
            acc = 1.0 if (j in unchecked or t == int(torch.argmax(Pu[j]))) else 0.0
        else:
            p, q = float(Pu[j, t]), float(Qu[j, t])
            if j in unchecked:
                acc = 1.0
            elif q > 0:
                acc = min(1.0, p / q)
            else:
                acc = 1.0 if p > 0 else (1.0 if mode == "M3" else 0.0)
        u = float(torch.rand((), generator=gen, device=gen.device)) if not greedy else 0.0
        if u < acc:
            emitted.append(t)
            n_acc += 1
            continue
        if greedy:
            r = int(torch.argmax(Pu[j]))
        else:
            res = Pu[j] if mode == "M4" else torch.clamp(Pu[j] - Qu[j], min=0)
            if float(res.sum()) <= 0:
                res = Pu[j]
            touched = touched or mode == "M4"
            r = int(torch.multinomial(res / res.sum(), 1, generator=gen))
        emitted.append(r)
        residual_at = j
        break
    else:
        if brow is not None:
            touched = touched or (bonus_row is None or not torch.equal(brow, bonus_row))
            emitted.append(int(torch.argmax(brow)) if greedy else int(torch.multinomial(brow / brow.sum(), 1, generator=gen)))
        elif bonus_token is not None:
            emitted.append(int(bonus_token))
    if trace is not None:
        touched = touched and mode != "M0"
        trace.add(dict(prefix=prefix, d=[int(x) for x in d], n_acc=n_acc, emitted=emitted, residual_at=residual_at,
                       rows_used=K - len(unchecked), greedy=greedy, Q=_rows(Q), P=_rows(P), Q_used=_rows(Qu),
                       P_used=_rows(Pu), bonus_row=_rows([bonus_row])[0] if bonus_row is not None else None,
                       bonus_row_used=_rows([brow])[0] if brow is not None else None),
                  # calibration: emitted token j against the true target row at its position; the bonus position
                  # only when its row is known (HF), so emitted is cut to the rows available
                  _calib(emitted, P, K, bonus_row), touched)
    return emitted, n_acc
