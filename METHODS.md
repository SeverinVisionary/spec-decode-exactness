# Methods details

Details of the tests, the grid, the fix ablation and the EAGLE-3 supplementary cells for the paper
"Draft-Model Speculative Decoding Can Emit Zero-Probability Tokens" (https://doi.org/10.5281/zenodo.23908668).
The paper's appendix keeps the fixtures, the reference, the gate and the detection limit.

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
