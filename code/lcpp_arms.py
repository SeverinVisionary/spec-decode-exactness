"""llama.cpp v0.5.0 arms (PREREGISTRATION §4). Draws sequences; never scores them."""
import json
import os
import re
import subprocess
import time
import urllib.request

import fixture as fx

BIN = fx.WORK / "llama.cpp" / os.environ.get("D40_LCPP_BUILD", "build") / "bin"  # R10: build-hostC on the 2.80 GHz host
# f32 KV + no flash attention: default FA/f16 gives delta_num ~1.7e-4 vs R_np, above the §5.1 bound
KV = ["--cache-type-k", "f32", "--cache-type-v", "f32", "-fa", "off"]
CHAIN = ["penalties", "top_k", "top_p", "min_p", "temperature"]
NEUTRAL = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, repeat_last_n=64,
               frequency_penalty=0.0, presence_penalty=0.0, typical_p=1.0, xtc_probability=0.0,
               dry_multiplier=0.0, top_n_sigma=-1.0, mirostat=0)
SETTINGS = {
    "S0": {},
    "S1": dict(temperature=0.7),
    "S2": dict(top_k=3),
    "S3": dict(top_p=0.8),
    "S4": dict(min_p=0.1),
    "S5": dict(repeat_penalty=1.3),
    "S6": dict(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
}
# G (greedy): the temperature stage with temp <= 0 keeps only the max logit (src/llama-sampler.cpp);
# llama-speculative switches to greedy verification for temp <= 0 (speculative.cpp L263). Kept out of
# SETTINGS so the F4/F5 loops stay S0-S6.
ALL_SETTINGS = dict(SETTINGS, G=dict(temperature=0.0))
# rnp.lcpp_dist keys for each setting
RNP_KEYS = {
    "S0": {}, "S1": dict(temperature=0.7), "S2": dict(top_k=3), "S3": dict(top_p=0.8),
    "S4": dict(min_p=0.1), "S5": dict(repetition_penalty=1.3),
    "S6": dict(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
}
SERVER_ARGS = {
    "NULL": [],
    "L1": ["-md", str(fx.MODELS / "draft.gguf"), "--spec-type", "draft-simple", "--spec-draft-n-max", "3"],
    "L2": ["--spec-type", "ngram-simple", "--spec-draft-n-max", "3",
           "--spec-ngram-simple-size-n", "1", "--spec-ngram-simple-size-m", "2"],
}


def params(setting):
    p = dict(NEUTRAL, **ALL_SETTINGS[setting])
    p["samplers"] = CHAIN
    return p


class Server:
    def __init__(self, method, port, threads=1):
        self.port = port
        self.log = open(fx.WORK / "logs" / f"server_{method}_{port}.log", "w")
        cmd = [str(BIN / "llama-server"), "-m", str(fx.MODELS / "target.gguf"), "-c", "64", "-np", "1",
               "-t", str(threads), *KV, "--port", str(port), "--host", "127.0.0.1", *SERVER_ARGS[method]]
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited: see {self.log.name}")
                time.sleep(0.25)
        raise RuntimeError("llama-server did not become healthy")

    def complete(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()


def draw(server, setting, n, seed, seeds=None):
    """seeds: optional explicit per-request seeds (Stage A/B pass distinct hashed seeds per cell)."""
    out = []
    base = dict(params(setting), prompt=fx.PROMPT, n_predict=fx.MAX_NEW, return_tokens=True, cache_prompt=False)
    if seeds is None:
        seeds = [(seed * 1_000_003 + i) % 2**31 for i in range(n)]
    for s in seeds:
        d = server.complete(dict(base, seed=int(s)))
        out.append(tuple(d["tokens"]))
    return out


def step_table(server, setting):
    """R_eng: post-sampling next-token distribution at each nonterminal prefix (non-speculative)."""
    import numpy as np
    rows = []
    base = dict(params(setting), n_predict=1, n_probs=fx.V, post_sampling_probs=True, cache_prompt=False, seed=1)
    for g in fx.nonterminal_prefixes():
        d = server.complete(dict(base, prompt=fx.PROMPT + g))
        row = np.zeros(fx.V)
        for t in d["completion_probabilities"][0]["top_probs"]:
            row[t["id"]] = t["prob"]
        rows.append(row)
    return np.stack(rows)


# llama-speculative (L3 ratio test, L4 tree): one process per sequence.
CLI_ARGS = {"L3": ["-np", "1"], "L4": ["-np", "3", "--spec-draft-p-split", "0.1"]}
CLI_FLAGS = {"temperature": "--temp", "top_k": "--top-k", "top_p": "--top-p", "min_p": "--min-p",
             "repeat_penalty": "--repeat-penalty", "repeat_last_n": "--repeat-last-n"}
CLI_STATS = {"reruns": 0}
PROMPT_TEXT = "".join(chr(ord("a") + t) for t in fx.PROMPT)


def cli_seeds(seed, n):
    import numpy as np
    return np.random.default_rng(seed).integers(1, 2**31 - 1, size=n)


def cli_parse(stdout):
    """Generated ids after the echoed prompt; cut at first EOS, then at MAX_NEW (the example
    overshoots -n by up to the draft length)."""
    text = stdout.replace("\n", "").replace(" ", "")
    if not text.startswith(PROMPT_TEXT):
        raise ValueError(f"unexpected output {stdout!r}")
    ids = [ord(c) - ord("a") for c in text[len(PROMPT_TEXT):]]
    if fx.EOS in ids:
        ids = ids[: ids.index(fx.EOS) + 1]
    return tuple(ids[: fx.MAX_NEW])


def cli_draw(method, setting, n, seed, seeds=None):
    p = dict(NEUTRAL, **ALL_SETTINGS[setting])
    base = [str(BIN / "llama-speculative"), "-m", str(fx.MODELS / "target.gguf"),
            "-md", str(fx.MODELS / "draft.gguf"), "-p", PROMPT_TEXT, "-n", str(fx.MAX_NEW),
            "--spec-draft-n-max", "3", "-c", "64", "-t", "1", *KV, "--samplers", ";".join(CHAIN),
            *CLI_ARGS[method]]
    for k, flag in CLI_FLAGS.items():
        base += [flag, str(p[k])]
    out, CLI_STATS["reruns"], CLI_STATS["n_drafted"], CLI_STATS["runs_with_stat"] = [], 0, 0, 0
    for s in (cli_seeds(seed, n) if seeds is None else seeds):
        # Output is a deterministic function of the seed; stdout goes through llama.cpp's async
        # logger and can lose its tail at exit, so an incomplete parse re-runs the same seed.
        for attempt in range(25):  # R9
            r = subprocess.run(base + ["-s", str(int(s))], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                raise RuntimeError(f"seed {int(s)} rc {r.returncode}: " + r.stderr[-300:])
            if PROMPT_TEXT.startswith(r.stdout.replace("\n", "").replace(" ", "")):
                # rc 0 with empty or prompt-only stdout: the whole output was lost (seen on the
                # x86 run under load, R8); same-seed rerun as for a lost tail
                CLI_STATS["reruns"] += 1
                continue
            seq = cli_parse(r.stdout)
            if (seq and seq[-1] == fx.EOS) or len(seq) == fx.MAX_NEW:
                # llama-speculative's closing LOG_INF "n_drafted = %d" (speculative.cpp 644); the async logger can
                # drop it, so runs without the line are counted, never read as zero
                m = re.search(r"n_drafted\s*=\s*(\d+)", r.stderr + r.stdout)
                if m:
                    CLI_STATS["n_drafted"] = CLI_STATS.get("n_drafted", 0) + int(m.group(1))
                    CLI_STATS["runs_with_stat"] = CLI_STATS.get("runs_with_stat", 0) + 1
                break
            CLI_STATS["reruns"] += 1
        else:
            raise RuntimeError(f"seed {int(s)}: incomplete output after 25 runs")
        out.append(seq)
    return out
