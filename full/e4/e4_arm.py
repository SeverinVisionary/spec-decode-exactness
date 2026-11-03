"""FR22 E4: one ablation arm of the #20852 fix on TensorRT-LLM 1.2.1 (FR18 fixture, bf16, L4).

The notebook applies the factor patches of the arm (`F1.patch`, `F2.patch`, `F3.patch`, split from the upstream fix
branch's final diff) to the installed package, then runs this script in a fresh process:

    python e4_arm.py --models DIR --out DIR --arm spec|base --tag F12

It reuses `drafts/trtllm_repro/repro_draft_target.py` (models, settings, N, the squeeze_ workaround = F4, which is on
in every arm) and replaces its recorder with one that also keeps, per verification, the emitted tokens and the
engine's target-probability rows at the emitted positions, in FR21's calibration format.
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "drafts" / "trtllm_repro"))
import repro_draft_target as R  # noqa: E402


class Recorder:
    """(K, R, n, zero) as FR18, plus calibration entries (emitted tokens, target rows at those positions)."""

    def __init__(self):
        from tensorrt_llm._torch.pyexecutor import sampler as S
        from tensorrt_llm._torch.pyexecutor.llm_request import get_draft_token_length
        self.rows, self.calib = [], []
        orig = S.TorchSampler._process_draft_tokens_rejection_sampling
        rec = self

        def wrapped(self_, request, *a, **k):
            K = int(get_draft_token_length(request))
            draft = list(request.py_draft_tokens or [])[:K]
            dl, tp = request.py_draft_logits, request.py_target_probs
            Rn = 0 if dl is None else int(dl.reshape(-1, dl.shape[-1]).shape[0])
            tp = None if tp is None else tp.detach().float().reshape(-1, tp.shape[-1]).cpu()
            before = len(request.get_tokens(0))
            n = orig(self_, request, *a, **k)
            emitted = [int(t) for t in request.get_tokens(0)[before:]]
            acc = min(int(n), K)
            zero = 0 if tp is None else sum(1 for i in range(min(acc, tp.shape[0])) if float(tp[i, draft[i]]) == 0.0)
            rec.rows.append((K, Rn, int(n), zero))
            if tp is not None and emitted and len(emitted) <= tp.shape[0]:
                rec.calib.append([emitted, [[round(float(x), 6) for x in tp[j]] for j in range(len(emitted))]])
            return n

        S.TorchSampler._process_draft_tokens_rejection_sampling = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("base", "spec"), required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    holder = {}

    class Verifications(Recorder):  # run_arm instantiates R.Verifications; keep a handle for the calibration rows
        def __init__(self):
            super().__init__()
            holder["rec"] = self

    R.Verifications = Verifications
    R.ARM_SEED = dict(R.ARM_SEED, spec=5000 + 10 * sum(1 << (int(c) - 1) for c in a.tag if c in "123"), base=4000)
    tmp = out / f"_{a.tag}"
    tmp.mkdir(exist_ok=True)
    for m in ("target", "draft"):
        if not (tmp / m).exists():
            os.symlink(pathlib.Path(a.models) / m, tmp / m)
    # run_arm resets rec.rows per setting but not our calib list: split it per setting by wrapping generate
    import tensorrt_llm
    orig_generate = tensorrt_llm.LLM.generate
    state = {"setting": None, "by_setting": {}}

    def generate(self_, prompts, params, *k, **kw):
        name = "topk3" if getattr(params, "top_k", None) == 3 else "T1"
        if "rec" in holder:
            if state["setting"] != name:
                holder["rec"].calib = state["by_setting"].setdefault(name, [])
                state["setting"] = name
        return orig_generate(self_, prompts, params, *k, **kw)

    tensorrt_llm.LLM.generate = generate
    R.run_arm(a.arm, tmp)
    d = json.loads((tmp / f"{a.arm}.json").read_text())
    for name, cal in state["by_setting"].items():
        d["settings"][name]["calibration"] = cal
    d["tag"] = a.tag
    import gzip
    with gzip.open(out / f"{a.arm}_{a.tag}.json.gz", "wt") as f:
        json.dump(d, f)
    (tmp / f"{a.arm}.json").unlink()
    print("wrote", out / f"{a.arm}_{a.tag}.json.gz", flush=True)


if __name__ == "__main__":
    main()
