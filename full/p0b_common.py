"""D-40 full study, Phase 0b: definitions shared by every engine's probes (FR9 item 7, FR11, FR12).

Phase 0b is non-evaluative (PREREGISTRATION.md §7). A smoke draws at most `N_SMOKE` sequences of a
cell and checks only crash, shape, firing and timing; the sequences are discarded unread. Two
quantities are *not* samples of any grid cell and are therefore not capped:
- `R_eng` scoring: one new token per prefix, reading the processed distribution (§5.1);
- throughput (F6): drawn from `PROMPT_TP`, a prompt no grid cell uses, so no drawn sequence belongs
  to a cell's outcome distribution (FR12).
"""
import hashlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fixture as fx  # noqa: E402

N_SMOKE = 100
CHUNK = 10_000  # FR9 item 5: production batch, one generate call per chunk of 10^4 requests
PROMPT_TP = [4, 5, 6, 4, 5]  # throughput only (FR12): no grid cell uses this prompt

# §4 settings, as in the pilot (PREREGISTRATION §4). Each engine maps them onto its own API.
SETTINGS = {
    "S0": dict(temperature=1.0),
    "S1": dict(temperature=0.7),
    "S2": dict(temperature=1.0, top_k=3),
    "S3": dict(temperature=1.0, top_p=0.8),
    "S4": dict(temperature=1.0, min_p=0.1),
    "S5": dict(temperature=1.0, repetition_penalty=1.3),
    "S6": dict(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
    "G": dict(temperature=0.0),
}
STOCHASTIC = [s for s in SETTINGS if s != "G"]
FB_SETTINGS = ("S0", "S2", "S6")


def seed_base(stage, cell):
    """FR9 item 3: h(stage, cell). Phase 0b uses stage "P0b"."""
    return int.from_bytes(hashlib.sha256(f"d40full|{stage}|{cell}".encode()).digest()[:4], "big") % 2**30


def seeds(stage, cell, n, offset=0):
    b = seed_base(stage, cell)
    return [b + offset + j for j in range(n)]


class Fixture:
    def __init__(self, name):
        self.name = name
        if name == "FA":
            self.V, self.EOS, self.MAX_NEW, self.PROMPT = fx.V, fx.EOS, fx.MAX_NEW, fx.PROMPT
            self.target, self.draft, self.n_layers, self.n_heads = "target", "draft", 2, 2
        else:
            import fb_fixture as fb
            self.V, self.EOS, self.MAX_NEW, self.PROMPT = fb.V, fb.EOS, fb.MAX_NEW, fb.PROMPT
            (self.target, self.draft), self.n_layers, self.n_heads = fb.NAMES, 4, 2

    def path(self, which, layout=""):
        return str(fx.MODELS / f"{getattr(self, which)}{layout}")

    def cut(self, seq):
        s = tuple(int(t) for t in seq)
        if self.EOS in s:
            s = s[: s.index(self.EOS) + 1]
        return s[: self.MAX_NEW]

    def in_space(self, s):
        return (len(s) == self.MAX_NEW or (s and s[-1] == self.EOS)) and all(0 <= t < self.V for t in s) \
            and self.EOS not in s[:-1]

    def raw_ok(self, seq):
        """The engine's own output, unrepaired: length <= MAX_NEW, every id in range, EOS only as the last token."""
        s = tuple(int(t) for t in seq)
        return len(s) <= self.MAX_NEW and all(0 <= t < self.V for t in s) and self.EOS not in s[:-1] \
            and (len(s) == self.MAX_NEW or (len(s) > 0 and s[-1] == self.EOS))

    def shape_check(self, seqs, expected=None):
        seqs = list(seqs)
        cut = [self.cut(s) for s in seqs]
        r = dict(n=len(seqs), out_of_space=sum(1 for s in cut if not self.in_space(s)),
                 raw_malformed=sum(1 for s in seqs if not self.raw_ok(s)))
        if expected is not None:
            r["count_ok"] = len(seqs) == expected
        del cut, seqs  # discarded unread (§7)
        return r


FA, FB = Fixture("FA"), None


def fixture(name):
    global FB
    if name == "FA":
        return FA
    if FB is None:
        FB = Fixture("FB")
    return FB


def fa_prefixes():
    return fx.nonterminal_prefixes()


FB_SCORE_FLOOR = 100  # expected visits at N_A: half the test's 200, so prefixes a real null selects are covered


def fb_prefixes(engine, setting):
    """F-B prefixes scored for R_eng: expected visits >= 100 at N_A under the fp64 reference. The test keeps
    prefixes with >= 200 realised null visits (§5.4); the lower floor covers those a null reaches by chance."""
    import f5_mde_full as f5
    ref = f5.FBRef(engine, f5.FB_SETTINGS[setting])
    # at the run's N (FR14): Stage B's null selects prefixes at N_B = 10^6, which N_A's set would not cover
    n = int(os.environ.get("D40_STAGEA_NA", f5.N_A))
    return [list(g) for g in sorted(f5.reach(ref, n, 0.0, FB_SCORE_FLOOR))], ref


def same_draws(a, b):
    """Seed checks compare draws for equality only; neither side is tabulated or compared with a reference."""
    return [tuple(x) for x in a] == [tuple(x) for x in b]


def tv_rows(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(-1)


def delta_report(eng, ref, mapping=None):
    """δ_num over EVERY engine row: row i is compared with ref[mapping[i]] (default: row i). Non-finite values and
    malformed shapes are reported as failures, never dropped."""
    eng, ref = np.asarray(eng, float), np.asarray(ref, float)
    mapping = np.arange(len(eng)) if mapping is None else np.asarray(mapping)
    ok_shape = eng.ndim == 2 and ref.ndim == 2 and eng.shape[1] == ref.shape[1] and len(mapping) == len(eng)
    if not ok_shape:
        return dict(error=f"shape mismatch eng {eng.shape} ref {ref.shape} mapping {len(mapping)}")
    finite = bool(np.isfinite(eng).all() and np.isfinite(ref).all())
    tv = tv_rows(eng, ref[mapping])
    return dict(delta_num=float(np.nanmax(tv)) if finite else float("inf"), finite=finite,
                argmax_prefix=int(mapping[int(np.nanargmax(tv))]) if finite else None,
                support_mismatch=int(((eng > 0) != (ref[mapping] > 0)).sum()),
                row_sum_dev=float(np.abs(eng.sum(-1) - 1).max()), rows=int(len(eng)), prefixes=int(len(ref)))


def batch_copies(prefixes, total=CHUNK):
    """Exactly `total` requests (the production batch), request i scoring prefix i mod len(prefixes).
    Returns (requests, mapping)."""
    mapping = [i % len(prefixes) for i in range(max(total, len(prefixes)))]
    return [prefixes[j] for j in mapping], mapping


def first_rows(rows, mapping, n_prefix):
    """One table row per prefix (its first occurrence) and the largest disagreement between copies."""
    rows = np.asarray(rows, float)
    mapping = np.asarray(mapping)
    first = np.zeros((n_prefix, rows.shape[1]))
    seen = np.zeros(n_prefix, bool)
    dev = 0.0
    for i, j in enumerate(mapping):
        if not seen[j]:
            first[j], seen[j] = rows[i], True
        else:
            dev = max(dev, float(np.abs(rows[i] - first[j]).max()))
    return first, dev


def save_npz(out_dir, name, **arrays):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    np.savez_compressed(path, **arrays)
    return path


def phase0_checkpoint(res):
    """Per-step progress for a probe (child mode), so a timeout keeps what finished."""
    import phase0
    phase0.checkpoint(res)
