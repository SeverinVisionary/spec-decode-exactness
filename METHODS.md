# Methods details

Details of the defect trace, the tests, the grid, the fix ablation and the EAGLE-3 supplementary cells for the paper
"Draft-Model Speculative Decoding Can Emit Zero-Probability Tokens" (https://doi.org/10.5281/zenodo.23908668).
The paper's appendix keeps the fixtures, the reference, the gate and the detection limit.

## Defect trace details

In the 192 rounds that start with a context forward, the verifier receives R = 1 draft row, and row 0 holds the
distribution d₁ was drawn from.
In the 38 rounds that start with a generation forward, R = 2; rows 0 and 1 are aligned and d₂'s row is missing.
The sampler sets the accepted count to K and checks only R rows, so when none of them rejects, K − R tokens are never
tested.
With the acceptance coin fixed at u = 1 at S2, 105 of 251 verifications emitted unchecked tokens.
For a token outside the top 3 of both the target and the misplaced draft row, the acceptance ratio p(d)/q(d) is 0/0,
and the comparison u > NaN is false, so the token is kept.

Replacing our inference-mode hook with an out-of-place squeeze still left unverified acceptances in 76 of 278
verifications and zero-probability acceptances in 75.
The bf16 reproduction runs without the fp32 hook.
Its one change to the engine works around a second defect, an in-place `squeeze_` on an inference tensor that crashes
outside `torch.inference_mode()` and came in with TensorRT-LLM pull request 9079
(https://github.com/NVIDIA/TensorRT-LLM/pull/9079).
There no draft is accepted past the last row, so the misplaced row alone is enough.

The drafter is built with `exclude_last_generation_logits=True`.
Keeping that last logits chunk adds one row but only moves the F-A distance at T = 1 from 0.497 to 0.432 (10⁵ draws).
In the ablation below, the patch's first part alone passes at T = 1 and top-k 3 (p = 0.56 and 0.85).

## Power and planted defects

In simulation at S0 and S2 against eight defect classes, each firing in at least 10% of rounds, the F-A test has power
at least 0.99 at N_A, with two exceptions at S0.
A bonus token drawn from the draft instead of the target is caught with power 0.64, and skipped logits processors
cannot be caught at all, because at T = 1 they do nothing.

We planted five verifier mutations inside real engines, each replacing the engine's verification function at run time
on its own draft tokens and rows.
They were a missing last row, shifted rows with and without the 0/0 case, a residual drawn from the wrong distribution
and rows from a stale prefix.
We ran them in Hugging Face Transformers on CPU and in vLLM, at S0 and S2 with 5×10⁴ draws per cell, and the χ² test
caught all five while passing the unmutated cells.

## Tests

F-A uses Pearson's χ² over the 2,801 outcomes.
We pool outcomes with expected count below 5, including those of probability zero, into one bin and take p from the
asymptotic distribution.
F-B also draws N_A = 2×10⁵ sequences per cell and fixes its prefixes from the null run: those visited at least 200
times.
At each, the next-token counts get the same pooled test, a prefix whose pooled table has no degree of freedom is
dropped, and the p-values are combined by Fisher's method.
We simulated nulls through this whole path, F-B included, and found no cell miscalibrated at 0.01 or at 0.01/118; the
simulations use the fp64 rows.
At the diagnostic level 0.05, one cell's simulated null, vLLM's F-B cell at S0, rejects in 6.8% of 2,000 simulations.

## The 118 cells

F-A has 100: vLLM 30, SGLang 28, TensorRT-LLM 21, llama.cpp 14, and Hugging Face Transformers 7.
F-B has 18, over S0, S2 and S6.
The preregistration's coverage count is 133: it adds 15 vLLM cells at S4 and S6, outside m, because vLLM refuses
min-p with speculation.
We list the 65 that did not run in the table below.

| Engine | Paths and reason | Cells |
|---|---|---:|
| vLLM | EAGLE-3 head and MTP layer: acceptance outside the required band | 12 |
| | block rejection sampling: identical counters to the default path | 5 |
| SGLang | EAGLE chain and tree heads: acceptance outside the required band | 17 |
| | S4, S6: the default sampler crashes | 4 |
| | S2, S3, S5: no usable reference | 6 |
| TensorRT-LLM | TRTLLM attention: no fp32 speculative decoding | 10 |
| | EAGLE-3: slot-manager error at start | 7 |
| | draft-target at S4–S6: settings ignored or sampler not seedable | 4 |
| **Total** | | **65** |

## Fix ablation

The patch has three parts.
The first part keeps the last draft row, also records the row from the draft's final context chunk, and makes the
logits buffer large enough to hold both.
We ran the bf16 reproduction with 2×10⁴ draws per arm: every arm with it passes against the non-speculative baseline
and every arm without it fails; two arms without it crashed before drawing.
Why this part passes while keeping the last logits chunk alone does not is not established.

## EAGLE-3 chain drafters

They come from the supplementary extensions (FR22 in the preregistration): the heads are ours, fitted to the
fixtures, and the eight cells form a Holm family of their own at 0.01.
We ran six vLLM cells on its V2 model runner with probabilistic drafting, drew their nulls on that runner, and got
p ≥ 0.147.
SGLang's F-A cells at S0 and S1 gave 0.742 and 0.994 against its Stage A nulls; its S2 and S3 references could not
be built.
SGLang's S0 cell ran at acceptance 0.294, just under the band its 100-sequence gate had passed.
