"""D-40 full study (PREREGISTRATION.md FR8): fp32 dispatch for TensorRT-LLM 1.2.1.

TensorRT-LLM calls FlashInfer's RMSNorm and RoPE kernels whenever FlashInfer imports, and both
dispatch fp16/bf16 only (`DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16` in flashinfer-python 0.6.4). FlashInfer
cannot simply be uninstalled: `_torch/modules/attention.py` imports `FlashInferAttentionMetadata`
unconditionally (Phase 0, 2026-10-07).

So FlashInfer stays installed and this hook sets `IS_FLASHINFER_AVAILABLE = False` in exactly two
modules, right after they load. Each then takes its own pure-torch branch:
- `tensorrt_llm._torch.modules.rms_norm`
- `tensorrt_llm._torch.modules.rotary_embedding`

FR11 adds one more module, with no change to any computation:
- `tensorrt_llm._torch.speculative.model_drafter`: `ModelDrafter.process_decoded_tokens` runs under
  `torch.inference_mode()`. The draft logits it receives were allocated by `HandleLogits` under
  inference mode, and its `generation_logits.squeeze_(0)` (model_drafter.py:476 in 1.2.1) is an
  in-place view change, so outside inference mode it raises and kills the executor thread (Phase 0,
  2026-10-19, TG1 on VANILLA attention). The squeeze and the copy to CUDA are unchanged.

Nothing else changes. The notebook loads this file from a `.pth` in the trtllm venv, so it also runs
in TensorRT-LLM's worker processes.
"""
import importlib.abc
import sys

FLASHINFER_TARGETS = frozenset({
    "tensorrt_llm._torch.modules.rms_norm",
    "tensorrt_llm._torch.modules.rotary_embedding",
})
DRAFTER_TARGET = "tensorrt_llm._torch.speculative.model_drafter"  # FR11
TARGETS = FLASHINFER_TARGETS | {DRAFTER_TARGET}
PATCHED = set()


def _patch_drafter(module):
    import functools
    import torch
    cls = module.ModelDrafter
    orig = cls.process_decoded_tokens
    if getattr(orig, "_d40_inference_mode", False):
        return

    @functools.wraps(orig)
    def process_decoded_tokens(*args, **kwargs):
        with torch.inference_mode():
            return orig(*args, **kwargs)

    process_decoded_tokens._d40_inference_mode = True
    cls.process_decoded_tokens = process_decoded_tokens


class _Fp32Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader_exec = spec.loader.exec_module

        def exec_module(module, _exec=loader_exec):
            _exec(module)
            if module.__name__ == DRAFTER_TARGET:
                _patch_drafter(module)
            else:
                module.IS_FLASHINFER_AVAILABLE = False
            PATCHED.add(module.__name__)

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _Fp32Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Fp32Finder())
