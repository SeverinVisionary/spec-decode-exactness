"""HF transformers v5.18.0 arms (PREREGISTRATION §4). Draws sequences; never scores them."""
import copy

import torch
from transformers import LlamaForCausalLM

import fixture as fx

SETTINGS = {
    "S0": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
    "S1": dict(do_sample=True, temperature=0.7, top_k=0, top_p=1.0),
    "S2": dict(do_sample=True, temperature=1.0, top_k=3, top_p=1.0),
    "S3": dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.8),
    "S4": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=0.1),
    "S5": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.3),
    "S6": dict(do_sample=True, temperature=0.7, top_k=5, top_p=0.9, min_p=0.05),
    "G": dict(do_sample=False),
}

# assistant generation_config overrides per method; None = no assistant model
ASSISTANT = {
    "H1": dict(num_assistant_tokens=3, num_assistant_tokens_schedule="constant",
               assistant_confidence_threshold=0.0),
    "H2": dict(num_assistant_tokens=3, num_assistant_tokens_schedule="heuristic",
               assistant_confidence_threshold=0.0),
    "H3": dict(num_assistant_tokens=3, num_assistant_tokens_schedule="constant",
               assistant_confidence_threshold=0.4),
}
CALL = {
    "H4": dict(prompt_lookup_num_tokens=3),
    "H5": dict(assistant_early_exit=1),
    "P6": dict(assistant_ensemble_weight=0.5),
    "NULL": dict(),
}


def load(name):
    return LlamaForCausalLM.from_pretrained(fx.MODELS / name, dtype=torch.float32).eval()


class Arm:
    def __init__(self, method, setting):
        self.method, self.setting = method, setting
        self.target = load("target")
        self.kw = dict(max_new_tokens=fx.MAX_NEW, **SETTINGS[setting])
        self.assistant = None
        base = "H1" if method in ("P6",) or method.startswith("P") else method
        if base in ASSISTANT:
            self.assistant = load("draft")
            for k, v in ASSISTANT[base].items():
                setattr(self.assistant.generation_config, k, v)
            self.kw["assistant_model"] = self.assistant
        self.kw.update(CALL.get(method, {}))
        self.prompt = torch.tensor([fx.PROMPT])

    @torch.no_grad()
    def draw(self, n, seed):
        torch.manual_seed(seed)
        out = []
        for _ in range(n):
            seq = self.target.generate(input_ids=self.prompt, **copy.copy(self.kw))
            out.append(tuple(seq[0, len(fx.PROMPT):].tolist()))
        return out

    @torch.no_grad()
    def draw_null_batched(self, n, seed, batch=500):
        """Non-speculative sampling may batch; same per-sequence distribution."""
        torch.manual_seed(seed)
        out = []
        kw = {k: v for k, v in self.kw.items() if k != "assistant_model"}
        while len(out) < n:
            b = min(batch, n - len(out))
            seq = self.target.generate(input_ids=self.prompt.repeat(b, 1), **kw)
            for row in seq[:, len(fx.PROMPT):].tolist():
                if fx.EOS in row:
                    row = row[: row.index(fx.EOS) + 1]
                out.append(tuple(row))
        return out
