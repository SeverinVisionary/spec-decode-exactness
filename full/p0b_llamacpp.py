"""Phase 0b probes for llama.cpp CUDA at the Phase 0 commit (FR9 items 1, 5, 7; FR12), on a T4.

- smoke (one child probe per fixture x method, drafted requests counted per setting): F-A NULL, LG1, LG2 x S0-S6, G and LG3 (`llama-speculative -np 1`, a KNOWN reproduction outside
  Holm) x S0; F-B NULL and LG2 x S0, S2, S6, G. At most 100 sequences each, discarded unread. Draws use
  the production server shape: `-np 16` and 16 client threads (FR9 item 5);
- seed_check: per-request seeds (FR9 item 3) under 16 concurrent slots: a seed list repeats exactly,
  the next list differs;
- reference: R_eng from `n_probs` / `post_sampling_probs` (the pilot's F4 path) on the `-np 16` server
  with 16 concurrent clients, and on `-np 1`; δ_num against R_np (`rnp.lcpp_dist`), F-A and F-B;
- throughput on PROMPT_TP with `-np 16`.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import time
import urllib.request

import numpy as np

import p0b_common as C

NP = 16


def L():
    import lcpp_arms as L
    return L


class Server:
    def __init__(self, fix, method, port, slots=NP):
        L_ = L()
        self.port, self.fix = port, fix
        os.makedirs(C.fx.WORK / "logs", exist_ok=True)
        self.log = open(C.fx.WORK / "logs" / f"p0b_server_{fix.name}_{method}_{port}.log", "w")
        draft = ["-md", C.fx.MODELS.as_posix() + f"/{fix.draft}.gguf", "--spec-type", "draft-simple",
                 "--spec-draft-n-max", "3"]
        extra = {"NULL": [], "LG1": draft, "LG2": draft + ["--spec-draft-sampling", "probabilistic"]}[method]
        cmd = [str(L_.BIN / "llama-server"), "-m", C.fx.MODELS.as_posix() + f"/{fix.target}.gguf",
               "-c", str(64 * slots), "-np", str(slots), "-t", "2", "-ngl", "99", *L_.KV,
               "--port", str(port), "--host", "127.0.0.1", *extra]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited: see {self.log.name}")
                time.sleep(0.25)
        raise RuntimeError("llama-server did not become healthy")

    def complete(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))

    def many(self, bodies, threads=NP):
        with cf.ThreadPoolExecutor(threads) as ex:
            return list(ex.map(self.complete, bodies))

    def drafted(self):
        self.log.flush()
        return open(self.log.name, errors="replace").read().count("draft acceptance")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def body(fix, setting, prompt=None, **over):
    p = dict(L().params(setting), prompt=prompt or fix.PROMPT, n_predict=fix.MAX_NEW, return_tokens=True,
             cache_prompt=False)
    p.update(over)
    return p


def draw(srv, fix, setting, seeds, prompt=None):
    return [tuple(d["tokens"]) for d in srv.many([body(fix, setting, prompt, seed=int(s)) for s in seeds])]


PORT = [18400]


def port():
    PORT[0] += 1
    return PORT[0]


def smoke_one(fname, method):
    """One fixture x method per child probe, so a server that fails to load cannot take other methods with it."""
    fix = C.fixture(fname)
    out = {}
    settings = list(C.SETTINGS) if fname == "FA" else [*C.FB_SETTINGS, "G"]
    try:
        srv = Server(fix, method, port())
    except Exception as e:  # noqa: BLE001
        return dict(smoke={f"{fname}/{method}/load": dict(error=f"{type(e).__name__}: {e}"[:300])})
    try:
        for s in settings:
            cell = f"{fname}/{method}/{s}"
            before = srv.drafted()
            try:
                t = time.time()
                seqs = draw(srv, fix, s, C.seeds("P0b", cell, C.N_SMOKE))
                out[cell] = dict(seq_per_s=round(C.N_SMOKE / (time.time() - t), 1),
                                 **fix.shape_check(seqs, expected=C.N_SMOKE))
            except Exception as e:  # noqa: BLE001
                out[cell] = dict(error=f"{type(e).__name__}: {e}"[:300])
            time.sleep(0.5)  # let the server finish logging this setting's requests
            # this setting only: requests whose slot logged a draft acceptance line
            out[cell]["spec_counters"] = dict(drafted_requests=srv.drafted() - before)
            C.phase0_checkpoint(out)
    finally:
        srv.close()
    return dict(smoke=out)


def smoke_lg3():
    out = {}
    try:
        seqs = L().cli_draw("L3", "S0", 20, None, seeds=C.seeds("P0b", "FA/LG3/S0", 20))
        st = L().CLI_STATS
        # draft_tokens from llama-speculative's own n_drafted; unavailable (None) if no run printed it
        out["FA/LG3/S0"] = dict(C.FA.shape_check(seqs), cli_reruns=st["reruns"],
                                spec_counters=dict(draft_tokens=st["n_drafted"], runs_with_stat=st["runs_with_stat"],
                                                   runs=len(seqs)) if st["runs_with_stat"] else None)
    except Exception as e:  # noqa: BLE001
        out["FA/LG3/S0"] = dict(error=f"{type(e).__name__}: {e}"[:300])
    return dict(smoke=out)


def seed_check():
    out = {}
    for m in ("NULL", "LG2"):
        srv = Server(C.FA, m, port())
        try:
            cell = f"FA/{m}/S0"
            s = C.seeds("P0b", cell, 64)
            tp = C.PROMPT_TP  # no grid-cell draws
            x, y = draw(srv, C.FA, "S0", s, prompt=tp), draw(srv, C.FA, "S0", s, prompt=tp)
            z = draw(srv, C.FA, "S0", [v + 64 for v in s], prompt=tp)
            out[cell] = dict(same_seed_repeats=C.same_draws(x, y), next_seeds_differ=not C.same_draws(x, z),
                             concurrent_slots=NP)
            del x, y, z
        finally:
            srv.close()
    return dict(seed_check=out)


def step_table(srv, fix, setting, prefixes):
    bodies = [body(fix, setting, prompt=fix.PROMPT + list(g), n_predict=1, n_probs=fix.V, post_sampling_probs=True,
                   seed=1, return_tokens=False) for g in prefixes]
    rows = []
    for d in srv.many(bodies):
        row = np.zeros(fix.V)
        for t in d["completion_probabilities"][0]["top_probs"]:
            row[t["id"]] = t["prob"]
        rows.append(row)
    return np.stack(rows)


def rnp_keys(setting):
    return L().RNP_KEYS[setting]


def reference():
    import rnp
    import f5_mde_full as f5
    tables = os.environ.get("D40_P0B_OUT", "/tmp/p0b") + "_tables"
    out = {}
    for slots in (NP, 1):
        srv = Server(C.FA, "NULL", port(), slots=slots)
        try:
            for s in C.STOCHASTIC:
                ref = rnp.step_table("lcpp", rnp_keys(s))
                eng = step_table(srv, C.FA, s, C.fa_prefixes())
                out[f"FA/{s}/np{slots}"] = C.delta_report(eng, ref)
                C.save_npz(tables, f"lcpp_FA_{s}_np{slots}.npz", eng=eng, np=ref)
                C.phase0_checkpoint(out)
        finally:
            srv.close()
    srv = Server(C.fixture("FB"), "NULL", port())
    try:
        for s in C.FB_SETTINGS:
            pre, fref = C.fb_prefixes("lcpp", s)
            ref = np.stack([fref.step(tuple(g)) for g in pre])
            eng = step_table(srv, C.fixture("FB"), s, pre)
            out[f"FB/{s}/np{NP}"] = dict(C.delta_report(eng, ref), prefixes=len(pre), level=f5.LEVEL)
            C.save_npz(tables, f"lcpp_FB_{s}_np{NP}.npz", eng=eng, np=ref, prefixes=np.array(
                [list(g) + [-1] * (C.fixture("FB").MAX_NEW - len(g)) for g in pre]))
            C.phase0_checkpoint(out)
    finally:
        srv.close()
    return dict(reference=out)


def throughput():
    out = {}
    for m in ("NULL", "LG2"):
        srv = Server(C.FA, m, port())
        try:
            t = time.time()
            draw(srv, C.FA, "S0", range(1, 2001), prompt=C.PROMPT_TP)
            out[f"FA/{m}/S0/np{NP}"] = round(2000 / (time.time() - t), 1)
        finally:
            srv.close()
    return dict(throughput_seq_per_s=out)


def probes():
    smokes = [(f"smoke_{f}_{m}", (lambda f=f, m=m: smoke_one(f, m)))
              for f, ms in (("FA", ("NULL", "LG1", "LG2")), ("FB", ("NULL", "LG2"))) for m in ms]
    return [*smokes, ("smoke_FA_LG3", smoke_lg3), ("seed_check", seed_check), ("reference", reference),
            ("throughput", throughput)]
