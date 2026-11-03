"""FR22 E3 tree evidence for SGLang 0.5.21 (loaded from a .pth, so it also runs in the scheduler subprocess).

With D40_TREE_TRACE=<file> set, wraps `eagle_worker_common._finalize_accept_tree_path` (called only for topk > 1)
to append each verify's accepted tree-node indices and lengths to <file> as JSON lines. Values are not changed.
The patch is applied lazily, when SGLang itself imports that module, so nothing is imported at startup.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

TARGET = "sglang.srt.speculative.eagle_worker_common"


def _patch(mod):
    orig = mod._finalize_accept_tree_path

    def _finalize_accept_tree_path(batch, accept_index, accept_lens, *a, **k):
        try:
            with open(os.environ["D40_TREE_TRACE"], "a") as f:
                f.write(json.dumps(dict(accept_index=accept_index.tolist(), accept_lens=accept_lens.tolist(),
                                        num_draft_tokens=k.get("num_draft_tokens"))) + "\n")
        except Exception:  # noqa: BLE001  the trace never changes the run
            pass
        return orig(batch, accept_index, accept_lens, *a, **k)

    mod._finalize_accept_tree_path = _finalize_accept_tree_path


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _patch(module)
            except Exception as e:  # noqa: BLE001
                with open(os.environ["D40_TREE_TRACE"] + ".err", "a") as f:
                    f.write(f"{type(e).__name__}: {e}\n")

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("D40_TREE_TRACE"):
    sys.meta_path.insert(0, _Finder())
