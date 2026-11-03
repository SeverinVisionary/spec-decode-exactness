"""Injected positive controls P1-P5 and the measure-zero boundary arm (PREREGISTRATION §5.3-5.4).

Each is a runtime monkeypatch of transformers v5.18.0 reproducing a historical defect class.
Use as a context manager around Arm("H1", setting).draw(...)."""
import contextlib

import torch
import transformers.cache_utils as CU
import transformers.generation.candidate_generator as C
import transformers.generation.utils as U
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

_STASH = {}


def _stash_target_logits(model):
    def hook(_m, _inp, out):
        # R11: clone. `.to(float32)` on float32 logits is an alias, and HF applies the logits
        # processors in place on it, so the Stage A stash already held processed logits.
        _STASH["raw"] = out.logits.detach().to(torch.float32).clone()
    return model.register_forward_hook(hook)


@contextlib.contextmanager
def p1_raw_q():
    """q = draft's raw logits while the draft samples from processed logits (transformers#47932)."""
    orig = C.AssistedCandidateGenerator._generate_candidates

    def gen(self, args):
        self.generation_config.output_logits = True
        out = self.assistant_model.generate(**args, **self.assistant_kwargs)
        self.assistant_kwargs["past_key_values"] = out.past_key_values
        return out.sequences, torch.stack(out.logits, dim=1)

    C.AssistedCandidateGenerator._generate_candidates = gen
    try:
        yield
    finally:
        C.AssistedCandidateGenerator._generate_candidates = orig


@contextlib.contextmanager
def p2_greedy_draft_q_as_sampled():
    """Draft decodes greedily but its scores are used as if sampled (transformers#32867)."""
    orig = C.AssistedCandidateGenerator._prepare_generation_args

    def prep(self, input_ids, mn, mx):
        args = orig(self, input_ids, mn, mx)
        self.generation_config.do_sample = False
        return args

    C.AssistedCandidateGenerator._prepare_generation_args = prep
    try:
        yield
    finally:
        C.AssistedCandidateGenerator._prepare_generation_args = orig


def _patched_verifier(transform):
    """Replace new_logits seen by _speculative_sampling with transform(raw, processed, gamma)."""
    orig = U._speculative_sampling

    def ss(cand_ids, cand_logits, gamma, new_logits, **kw):
        raw = _STASH["raw"][:, -gamma - 1:]
        return orig(cand_ids, cand_logits, gamma, transform(raw, new_logits, gamma), **kw)

    return orig, ss


@contextlib.contextmanager
def p3_processors_bonus_only(arm):
    """Target processors applied only at the bonus position (vllm#42744)."""
    def t(raw, proc, gamma):
        out = proc.clone()
        out[:, :gamma] = raw[:, :gamma]
        return out

    orig, ss = _patched_verifier(t)
    h = _stash_target_logits(arm.target)
    U._speculative_sampling = ss
    try:
        yield
    finally:
        U._speculative_sampling = orig
        h.remove()


@contextlib.contextmanager
def p4_topp_before_topk(arm):
    """Verifier alone applies top-p before top-k (sglang#41916). Meaningful only when both are set."""
    kw = arm.kw
    temp, k, p = kw.get("temperature", 1.0), kw.get("top_k", 0), kw.get("top_p", 1.0)
    mp = kw.get("min_p")
    from transformers.generation.logits_process import MinPLogitsWarper

    def t(raw, proc, gamma):
        out = raw.clone() / temp
        for i in range(out.shape[1]):
            row = out[:, i]
            row = TopPLogitsWarper(p)(None, row) if p < 1.0 else row
            row = TopKLogitsWarper(k)(None, row) if k else row
            row = MinPLogitsWarper(mp)(None, row) if mp is not None else row
            out[:, i] = row
        return out

    orig, ss = _patched_verifier(t)
    h = _stash_target_logits(arm.target)
    U._speculative_sampling = ss
    try:
        yield
    finally:
        U._speculative_sampling = orig
        h.remove()


@contextlib.contextmanager
def p5_crop_one_too_few():
    """After a rejection the target KV cache is cropped one token too few (rollback class)."""
    orig_crop, orig_ad = CU.DynamicCache.crop, U.GenerationMixin._assisted_decoding
    targets = set()

    def crop(self, max_length):
        if max_length < 0 and id(self) in targets:
            max_length += 1
            if max_length == 0:
                return None
        return orig_crop(self, max_length)

    def ad(self, *a, **k):
        targets.add(id(k["past_key_values"]))
        try:
            return orig_ad(self, *a, **k)
        finally:
            targets.discard(id(k["past_key_values"]))

    CU.DynamicCache.crop, U.GenerationMixin._assisted_decoding = crop, ad
    try:
        yield
    finally:
        CU.DynamicCache.crop, U.GenerationMixin._assisted_decoding = orig_crop, orig_ad


@contextlib.contextmanager
def boundary_coin_zero():
    """§5.4 boundary arm: the acceptance coin r is exactly 0."""
    orig = torch.rand_like
    torch.rand_like = lambda x, *a, **k: torch.zeros_like(x)
    try:
        yield
    finally:
        torch.rand_like = orig
