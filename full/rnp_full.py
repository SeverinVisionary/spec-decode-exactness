"""R_np sampler orders for the full study's engines that the pilot did not have (§5.1, Phase 0b F4).

Importing this module registers them in `rnp.DISTS`. Like `rnp`, it imports no inference engine; each
order is written from the pinned engine source, cited per function.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))

import numpy as np  # noqa: E402

import rnp  # noqa: E402

_softmax = rnp._softmax


def trtllm_dist(logits, hist, s):
    """TensorRT-LLM 1.2.1 TorchSampler (`_torch/pyexecutor/sampling_utils.py` 186-258,
    `top_k_top_p_sampling_batch`): temperature -> top-k (mask logits < k-th largest, ties kept) -> top-p on
    the masked, scaled logits (descending order; keep token i iff the mass before it is < p, so the
    crossing token is kept) -> renormalise. min_p and repetition_penalty are never applied (no reference
    to either in `_torch/pyexecutor`, FR8); a setting that has only those stages is sampled at its temperature with
    them ignored, so its cells are BLOCKED (FR12) and this function is not used for them."""
    z = logits / max(s.get("temperature", 1.0), 1e-5)
    k = s.get("top_k", 0)
    if k and k < len(z):
        kth = np.sort(z)[-k]
        z = np.where(z < kth, -np.inf, z)
    p = _softmax(z)
    tp = s.get("top_p", 1.0)
    if tp < 1.0:
        order = np.argsort(-z, kind="stable")
        before = np.concatenate([[0.0], np.cumsum(p[order])[:-1]])
        keep = np.zeros(len(z), bool)
        keep[order[before < tp]] = True
        p = np.where(keep, p, 0.0)
        p = p / p.sum()
    return p


def _keep_top_k(p, k):
    """FlashInfer top-k renorm semantics (in-tree mirror `kernels/ops/sampling/renorm_triton.py`): keep p >= k-th largest."""
    return p >= np.sort(p)[-k] if k and k < len(p) else np.ones(len(p), bool)


def _keep_top_p(p, tp):
    """FlashInfer top-p renorm semantics (same mirror, 122-130): ascending sort, pivot = first index with
    cdf >= 1 - p, keep every probability >= the pivot (the crossing token and its ties are kept)."""
    if tp >= 1.0:
        return np.ones(len(p), bool)
    srt = np.sort(p)
    cdf = np.cumsum(srt)
    pivot = srt[int(np.searchsorted(cdf, 1 - tp, side="left"))]
    return p >= pivot


def sglang_dist(logits, hist, s):
    """SGLang 0.5.21 non-speculative sampler, default FlashInfer backend (`srt/layers/sampler.py`):
    penalties are applied to the logits first (output tokens only, so a prefix scored as a prompt is not
    penalised: S5 is not reachable by prompt scoring, FR12), then softmax(z / T); then
    - without min_p: `top_k_top_p_sampling_from_probs(..., filter_apply_order="joint")`, the top-k and top-p
      sets each taken on the same probabilities and intersected (sampler.py 397-402);
    - with min_p: sequential top-k renorm -> top-p renorm -> keep p >= min_p * max p (sampler.py 364-384)."""
    T = s.get("temperature", 1.0)
    p = _softmax(logits / T)
    k, tp, mp = s.get("top_k", 0), s.get("top_p", 1.0), s.get("min_p", 0.0) or 0.0
    if mp > 0:
        p = np.where(_keep_top_k(p, k), p, 0.0)
        p = p / p.sum()
        p = np.where(_keep_top_p(p, tp), p, 0.0)
        p = p / p.sum()
        p = np.where(p >= mp * p.max(), p, 0.0)
    else:
        p = np.where(_keep_top_k(p, k) & _keep_top_p(p, tp), p, 0.0)
    return p / p.sum()


def sglang_torch_dist(logits, hist, s):
    """SGLang 0.5.21 non-speculative sampler with `sampling_backend="pytorch"` (`srt/layers/sampler.py` 766-803):
    softmax(z / T), sorted descending; the cumulative sum is taken on these probabilities before top-k masking;
    keep exactly the first k positions; drop a position when the mass strictly before it exceeds top_p (the
    crossing token is kept); drop probabilities below min_p times the largest; sample from what is left."""
    p = _softmax(logits / s.get("temperature", 1.0))
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    before = np.cumsum(ps) - ps
    keep = np.ones(len(p), bool)
    k = s.get("top_k", 0)
    if k and k < len(p):
        keep[k:] = False
    keep &= ~(before > s.get("top_p", 1.0))
    mp = s.get("min_p", 0.0) or 0.0
    keep &= ~(ps < ps[0] * mp)
    out = np.zeros(len(p))
    out[order] = np.where(keep, ps, 0.0)
    return out / out.sum()


rnp.DISTS["trtllm"] = trtllm_dist
rnp.DISTS["sglang"] = sglang_dist
rnp.DISTS["sglang_torch"] = sglang_torch_dist
