# Preregistration — D-40 full study (GPU engines)

**Frozen 2026-10-04, before any full-study run.** No full-study model has been built, no Colab session
has run, and no sample has been drawn. Revisions are appended to the revision log at the end,
*before* the run they affect. Each says which way it moves the verdict. Nothing above the log is
edited after freezing.

**Status.** `ADMITTED` by the owner on 2026-10-04 […]. The cap is
**$80**. The GPU provider is **Google Colab** on the owner's account. The owner runs the notebooks […], and their outputs return to the repository as files.

**Basis.** The admission is an owner decision on the CPU pilot's `INCONCLUSIVE` record
(`../RESULT.md`, `../CLOSEOUT.md`). It is not a pilot CONTINUE. The pilot's preregistration does not
carry over. Its design is reused where stated, and its lessons are written in as rules (§5.5).

---

## 1. Question, frozen

Do the speculative-decoding paths shipped by GPU serving engines emit the **same sequence
distribution** as the same engine's non-speculative sampling? Each path runs end to end:
- proposer;
- logits processors;
- verifier or tree;
- bonus token;
- KV rollback;
- adaptive draft length.

The test is against an exactly enumerated reference, on tiny fixtures whose distributions are known
exactly.

**Out of scope:**
- multi-GPU and tensor-parallel paths;
- quantized targets;
- real vocabularies;
- long contexts;
- throughput.

## 2. Engines and pins, frozen

| Engine | Pin | Install |
|---|---|---|
| vLLM | `0.30.0` (PyPI, CUDA wheel) | `pip install vllm==0.30.0` |
| SGLang | `0.5.21` (PyPI) | `pip install "sglang[all]==0.5.21"` |
| TensorRT-LLM | `1.2.1` (PyPI, NVIDIA index) | `pip install tensorrt-llm==1.2.1` |
| llama.cpp | `master` at the commit recorded in Phase 0, CUDA build, portable (`-DGGML_NATIVE=OFF`) | source |
| HF transformers | `5.18.0`, torch `2.14.1` | PyPI |

Each engine runs in its own fresh Colab runtime. The installed versions and the GPU name, driver and
CUDA version are recorded in every chunk.

## 3. Fixtures, frozen

- **F-A, the pilot fixture.** Target and draft as in the pilot (V = 8, EOS = 7, 4 new tokens, 2,801
  outcomes, prompt `[1,2,3,1,2]`). The committed weights and hashes are reused unchanged.
  - Engines whose attention kernels reject head_dim 8 get the pilot's function-preserving head_dim-32
    re-layout (`code/vllm_export.py`).
  - That re-layout must pass §5.1 on each engine.
- **F-B, a larger fixture.** V = 32 (EOS = 31), 6 new tokens, 4 layers, hidden 64, random init with
  seed 2.
  - The logit scale and draft σ are set by the pilot's §3 rules (median entropy in [0.8, 1.8] nats;
    mean acceptance in [0.5, 0.8]).
  - Its sequence space (about 10⁹ outcomes) is not enumerated. F-B is tested per step instead
    (§5.4).
- **Drafter heads.** These are tiny EAGLE-3 / EAGLE and MTP heads for F-A and F-B, built in each
  engine's documented checkpoint format.
  - They are random-init, with scales calibrated so acceptance falls in [0.3, 0.9]. A head needs to
    be lossless, not good.
  - Model surgery is allowed (weights, config, checkpoint format); engine-code changes are never
    allowed.
  - A head an engine will not load is `BLOCKED` for that engine.

## 4. Grid, frozen

**Settings.** S0–S6 and G exactly as in the pilot (`../PREREGISTRATION.md` §4), applied through each
engine's own API. Unlisted default sampler stages are disabled explicitly. An engine that refuses a
setting with speculation on makes that cell `BLOCKED`, as vLLM's `min_p` refusal did in the pilot.

**Methods**, in run-priority order:

| Id | Engine | Path |
|---|---|---|
| VG1 | vLLM | `draft_model`, k = 3, `draft_sample_method="greedy"` (default) |
| VG2 | vLLM | `draft_model`, k = 3, `draft_sample_method="probabilistic"` |
| VG3 | vLLM | `ngram`, k = 3, lookup 1–2 |
| VG4 | vLLM | EAGLE-3 head, k = 3 |
| VG5 | vLLM | `draft_model`, k = 3, `rejection_sample_method="block"` |
| VG6 | vLLM | MTP, if a tiny MTP-capable fixture loads |
| SG1 | SGLang | EAGLE / EAGLE-3 head, chain (topk 1) |
| SG2 | SGLang | EAGLE head, tree (topk > 1), if shipped for the fixture |
| SG3 | SGLang | n-gram / standalone draft, if shipped |
| LG1 | llama.cpp CUDA server | `draft-simple`, greedy draft (default) |
| LG2 | llama.cpp CUDA server | `draft-simple`, `--spec-draft-sampling probabilistic` (PR #27694) |
| LG3 | llama.cpp CUDA | `llama-speculative`, `-np 1` (pilot defect: KNOWN, reported as a reproduction, not counted) |
| TG1 | TensorRT-LLM | draft-target |
| TG2 | TensorRT-LLM | EAGLE |
| HG1 | HF on CUDA | pilot H1 (draft model, constant k = 3) |

- **F-A** runs every method × S0–S6, plus G.
- **F-B** runs VG1, VG2, VG4, SG1, LG2 and TG1 × {S0, S2, S6}, plus G.
- MTP and tree arms run only if Phase 0 shows they load on a fixture with an independent fp64
  reference. Otherwise they are `REFERENCE_UNRESOLVED` (the pilot's H6 rule).

## 5. Oracles, controls, statistics, frozen

### 5.1 Reference

- **`R_eng`.** Each engine's own **non-speculative** per-step processed distribution at every F-A
  prefix (400), and at every F-B prefix the tests visit. It is taken **on the exact sampler code path
  that draws that engine's samples**. This is the pilot's vLLM Triton-vs-PyTorch lesson.
- **`R_np`.** The pilot's fp64 implementation, extended to F-B and to each engine's documented
  sampler order.
- **δ_num.** The maximum per-step TV between `R_eng` and `R_np`, which must satisfy 6·δ_num <
  MDE/10. An engine × setting that fails is `REFERENCE_UNRESOLVED`.
- **Host verification.** References are rebuilt and re-verified on every GPU type that draws
  samples. The GPU type is recorded per chunk.

### 5.2 Null controls

- Each engine × setting × fixture gets its non-speculative sampler at the same N, under the same test.
- Holm is applied at α = 0.01 over all null cells.
- A rejected null makes the matching speculative cells `INCONCLUSIVE`.

### 5.3 Positive controls, with the pilot's lesson built in

- **HF on CUDA:** P1–P6 as in the pilot (H1; P3 at S2 and S4; P4 at S6; P6 at S0), with the stash
  cloned.
- **vLLM, injected at runtime** (monkeypatch, never committed engine changes):
  - **P7:** target processors skipped at drafted positions (vllm#42744 class);
  - **P8:** a greedy draft whose q is used as if sampled (the pilot's L3 behaviour 1).
- **Effect assertion.** Every control asserts, in Phase 0 *and* in every Stage A chunk, that its patch
  changes the patched quantity: a max-abs difference above 0 against the unpatched quantity on that
  chunk's first batch. A chunk whose assertion fails aborts.
- **Validity.** Every control must reject in Stage A at p < α/m, where m is the number of running
  grid cells. A miss caps the verdict at `INCONCLUSIVE`.

### 5.4 Statistic, N, stages

- **F-A.** Pooled Pearson χ² over the 2,801 outcomes, exactly as in the pilot.
- **F-B.** For each visited prefix with at least 200 visits, a pooled χ² of the next-token counts
  against `R_eng`. These are combined by Fisher's method into one p per cell, and the prefix set is
  fixed by the null control's visits.
- **Stage A:** N_A = 2·10⁵ per cell, with fixed seeds committed before the run. Holm at α = 0.01 over
  all grid cells of all engines and fixtures. Controls and nulls are separate families.
- **Stage B:** every Stage-A-rejected cell plus its null, at N_B = 10⁶, with seeds **disjoint from
  Stage A**. Bonferroni at 0.01 per family.
- **MDE.** Per cell at its actual N, against the pilot's perturbation (uniform over the support, R1),
  at α/m. A cell with ε > 0.02 is `UNDERPOWERED`. F-B's MDE is computed for its per-step test.
- **Greedy G.** 1,000 runs per method. Every one must match the `R_eng` argmax chain exactly.

### 5.5 Rules carried from the pilot's lessons

1. **Controls must change the computation** (§5.3). Stashes clone.
2. **Every chunk records** the GPU name, driver, CUDA, engine versions and the harness git SHA. That
   SHA must be committed on this branch before the chunk runs.
3. **Every arm records** draft and accept counters per chunk.
4. **Binaries are portable** (no `-march=native` or GPU-arch-only builds), or are rebuilt and
   re-verified per GPU type.
5. **Novelty searches are run and logged before any NEW call:** bare API names, in issues, PRs and
   Discussions, open and closed.
6. **A container or runtime restart resumes from cached chunks** with the same seeds. It is logged,
   never silent.

## 6. Decision rule, frozen (two-sided)

Classes are NEW, KNOWN and BOUNDARY, as in the pilot. **The pilot's llama.cpp L3 defect, and
vllm#58214, #59785 and #55005, are KNOWN to this study.**

- **CONTINUE** (to write-up […]: conformance study plus bug list) requires:
  - a valid harness;
  - at least one grid cell confirmed in Stage B and classed NEW;
  - that cell's null passing both stages;
  - a seeded or coin-injected minimal reproduction.
- **KILL** (on value) requires:
  - a valid harness;
  - no NEW confirmed failure;
  - at least 75% of grid cells run at N ≥ 5·10⁴, neither `UNDERPOWERED` nor
    `REFERENCE_UNRESOLVED`.
- **INCONCLUSIVE:** the harness is invalid or coverage is below 75%. Any rerun needs a new
  prospective revision.
- **STOPPED_SCOOPED:** before the verdict, a public source posts a cross-engine conformance test
  against an exact or enumerated reference. On the verdict day, check LosslessBench (lilyzh.ng and
  its dataset), vLLM #59566/#59602 and successors, and arXiv.
- **STOPPED (other):** a named blocker, including the $80 cap being reached.

**Scope sentence, required beside any verdict.** The result covers the single-GPU paths in §4, on
tiny fixtures, at the stated MDE. It says nothing about multi-GPU, quantized, long-context or
real-vocabulary behaviour. "Lossless" is never claimed.

**Prediction, recorded before any run.**
- vLLM VG1/VG3 and llama.cpp LG1/LG2 pass. They are exact-match or probabilistic designs, and the
  pilot found the CPU paths clean.
- The likeliest NEW failures are in tree verification (SG2) and the EAGLE paths.
- Most likely verdict: CONTINUE or KILL, roughly even.

## 7. Phase 0 (non-evaluative) and resources, frozen

**Phase 0 may install, build, calibrate and smoke-test.** Smoke tests are capped at 100 sequences per
arm, checked for crash, shape, firing and timing only, and discarded unread. It never compares a
speculative sample to a reference.

- **F1. Install** each engine on Colab and record the environment.
- **F2. Fixtures.** Build F-B and the drafter heads, and record hashes.
- **F3. Feasibility.** Classify each arm (runs / `BLOCKED` / `REFERENCE_UNRESOLVED`), and run the
  control effect-assertions.
- **F4. References.** Build `R_eng` per engine × setting × fixture on each GPU type, and compute
  δ_num.
- **F5. MDE** for every cell.
- **F6. Budget.** Measure throughput and fit the grid to the cap.

Before Stage A, a revision fixes the running cells, Holm's m and the seed plan.

**Budget.**
- **Cap:** $80 of Colab compute, measured in compute units at the owner's plan rate and recorded per
  session.
- **When it binds:** if the grid does not fit, N_A is lowered uniformly, but not below 5·10⁴. Cells
  beyond the cap are then `NOT_RUN` in §4's priority order, with controls and nulls first.
- **At the cap:** the study stops (`STOPPED`), and the owner is told before the last 10% is spent.

## 8. Producer/checker separation and claims, frozen

- **Module separation.** Sampling harness, `R_eng`, `R_np` and statistics are separate modules. The
  statistics see only counts and probabilities.
- **Wording.** Engine claims name the pins in §2. "Lossless" is never claimed; the strongest wording
  is "no deviation detected at MDE ε, at N, on this fixture".
- **Upstream.** Reports are drafted for the owner and filed only by the owner.
- **Review.** […]

---

## Revision log

Revisions are appended here, before the run they affect.

### FR1 (2026-10-05, before any Phase 0 run completed): SGLang install command

- **Trigger.** On Colab, `pip install "sglang[all]==0.5.21"` failed while building `outlines_core`.
  `outlines_core==0.1.26` (pinned through `outlines==0.1.11`) ships no CPython 3.13 wheel, so pip
  builds it from source, which needs a Rust toolchain. Colab's image does not include one.
- **Change.**
  - The notebooks install a minimal Rust toolchain (rustup) before pip.
  - They install `sglang==0.5.21` without the `[all]` extra. That extra adds only `diffusion`,
    `http2` and `tracing`, which this study does not use.
  - The SGLang pin stays `0.5.21`. Its core dependencies (torch 2.13.0, `sglang-kernel` 0.4.7,
    flashinfer 0.6.18, xgrammar 0.2.7) are unchanged.
- **Effect on design.** None. No sample had been drawn; Phase 0 is non-evaluative.

### FR2 (2026-10-05, before any TensorRT-LLM Phase 0 run completed): TensorRT-LLM interpreter

- **Trigger.** On Colab, `pip install tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com`
  failed in metadata generation.
  - `tensorrt_llm` 1.2.1 ships x86_64 wheels for CPython 3.10 and 3.12 only.
  - Colab's interpreter is 3.13, so pip fell back to the PyPI sdist, which does not build.
- **Change.**
  - For `trtllm` only, the notebooks create a CPython 3.12 virtualenv with `uv`. They install
    `tensorrt-llm==1.2.1`, plus `safetensors`, `scipy` and `numpy`, into it (`libopenmpi-dev` via apt).
  - They run `vllm_export.py` and `phase0.py` with that interpreter. The path is recorded in
    `<runtime>/d40_python`; other engines keep the system Python.
- **Pins.** The TensorRT-LLM pin stays `1.2.1`. A dry resolve on 2026-10-05 for CPython 3.12, x86_64
  manylinux_2_28 gives torch 2.9.1, tensorrt 10.14.1.48.post1 and mpi4py 4.1.2. The run records its
  actual versions in the Phase 0 JSON.
- **Effect on design.** None. Phase 0 is non-evaluative.

### FR3 (2026-10-05, before any Phase 0 record was written): probe isolation, GPU type, llama.cpp CUDA arch

- **Trigger 1: hangs.** On a Colab T4 (compute capability 7.5), the first vLLM probe stalled after
  engine start-up, and the run wrote nothing.
  - vLLM logged that FA2 needs compute capability ≥ 8 and fell back to `FLEX_ATTENTION`.
  - `phase0.py` had no per-probe time limit and wrote its JSON only at the end.
  - The SGLang run on the same account also left no file.
- **Trigger 2: build time.** The llama.cpp CUDA build compiled kernels for every architecture
  (`-DGGML_NATIVE=OFF`, no arch list) with output hidden. It produced no visible progress for over
  15 minutes on Colab's 2 vCPUs.
- **Change to `phase0.py`.**
  - Each probe runs in its own child process and process group, with a wall-clock limit
    (`--probe-timeout`, default 900 s).
  - A probe over the limit is killed with its whole group and recorded as `TIMEOUT`, together with
    the settings it finished.
  - The JSON is rewritten after every probe and carries `complete: false` until the end.
  - The environment block now records the GPU's `compute_cap` and the llama.cpp CUDA arch.
- **Change to the notebooks: GPU.** They ask for an L4 GPU, as the hardware row of the frozen design
  already says (1 × L4). They no longer say a T4 works.
- **Change to the notebooks: llama.cpp.**
  - The build compiles for the runtime GPU's compute capability only
    (`-DCMAKE_CUDA_ARCHITECTURES=<cc>`), records it in `build-cuda/d40_cuda_arch.txt`, uses all
    cores and shows progress.
  - §5.4 already permits this: binaries are "portable … or are rebuilt and re-verified per GPU
    type". Every chunk therefore records the arch, and a new GPU type means a rebuild.
  - Still `-DGGML_NATIVE=OFF`, so there is no host-CPU tuning.
- **Effect on design.** None. Phase 0 is non-evaluative, and no sample has been compared with a
  reference.

### FR4 (2026-10-06, after the llama.cpp and vLLM Phase 0 records, before any other engine's Phase 0 and before Stage A)

- **Trigger.**
  - `PHASE0_FULL.md`: vLLM timed out at 900 s on every head_dim-8 arm, on an L4, costing 75 min.
  - The llama.cpp record lacks the llama.cpp commit, which §2 says Phase 0 records.
- **Change to `phase0.py`.**
  - It records `llamacpp_git` (the `rev-parse HEAD` of the cloned source) in the environment block.
  - The vLLM, SGLang and TensorRT-LLM probes run the head_dim-32 re-layout arms first and the
    original head_dim-8 arms second. The set of arms is unchanged.
- **Change to the Stage A plan (to be fixed in the pre-Stage-A revision, §6).**
  - vLLM arms use the head_dim-32 re-layout; the vLLM head_dim-8 arms are `BLOCKED`.
  - V × S6 is `BLOCKED`, because min_p is refused with speculative decoding.
  - VG5 needs a counters-only preflight with `draft_sample_method="probabilistic"` before it is
    kept as a distinct arm.
- **Effect on design.** None for any verdict. Phase 0 remains non-evaluative.

### FR5 (2026-10-06, before any SGLang probe ran and before Stage A): failure logs, SGLang attention backend

- **Trigger.**
  - The first L4 SGLang record (`PHASE0_FULL.md`) shows every arm exiting with −9, and no cause in
    the JSON.
  - The owner's console output gives the cause: `KeyError: torch.float32` in FlashInfer's prefill
    module lookup. FlashInfer, SGLang's default attention backend, has no fp32 kernels. SGLang then
    SIGKILLs its own process tree.
- **Change to `phase0.py`.**
  - Each probe's console output goes to `$D40_WORK/logs/probe_<arm>.log`.
  - For a probe that does not run, the last 4000 characters are copied into the record (`log_tail`)
    and printed.
- **Change to the SGLang engine options** (Phase 0 now; Stage A uses the same options):
  `attention_backend="triton"`, `speculative_draft_attention_backend="triton"`,
  `cuda_graph_backend_prefill="disabled"`, `cuda_graph_backend_decode="disabled"`.
  - This keeps the frozen fp32 fixture and the SGLang pin.
  - The SGLang verifier and sampler code paths under test are unchanged; only the attention kernel
    and graph capture differ.
  - The configuration goes into every chunk, and claims about SGLang name it.
- **Effect on design.** None. Phase 0 is non-evaluative.

### FR6 (2026-10-06, before any SGLang probe ran and before Stage A): SGLang fused ops in torch

- **Trigger.** The second L4 SGLang attempt, with FR5's settings, got past attention. Every arm then
  failed in the first forward pass: `sgl_kernel` rmsnorm "failed to dispatch data type Float"
  (`PHASE0_FULL.md`). `sgl_kernel`'s fused CUDA ops are fp16/bf16-only.
- **Change.**
  - The SGLang probe sets `SGLANG_FORCE_FUSED_OP_BACKEND=torch` before importing SGLang, so the
    scheduler subprocesses inherit it.
  - With it, every SGLang `BaseFusedOp` (RMSNorm, activation, rotary and the like) runs its
    pure-torch reference `forward_native`. That handles fp32 and does the same arithmetic as the
    fused kernel.
  - Each probe records its engine options: attention backend, CUDA graphs and fused-op backend.
- **Scope of SGLang claims.** SGLang results are for this configuration: fp32, Triton attention, no
  CUDA graphs, torch fused ops. The speculative verifier and sampler code paths under test are
  SGLang's own and unchanged. Defects that live only in fp16/bf16 fused kernels or CUDA-graph replay
  are outside this study's SGLang coverage, and the §8 claims will say so.
- **Effect on design.** None. Phase 0 is non-evaluative.

### FR7 (2026-10-06, before any TensorRT-LLM probe ran and before Stage A): TensorRT-LLM fp32 path, source-checked

- **Trigger.** A source review of the pinned wheels, made before the TensorRT-LLM run so as not to
  spend another Colab session, found the same fp32 trap that stopped SGLang.
  - `tensorrt_llm/_torch/modules/rms_norm.py` (1.2.1) calls FlashInfer `rmsnorm` whenever
    `import flashinfer` succeeds.
  - In `flashinfer-python` 0.6.4, TensorRT-LLM's pin, `csrc/norm.cu` dispatches with
    `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16`, which is fp16/bf16 only.
  - Every TensorRT-LLM `_torch` use of FlashInfer outside `auto_deploy` is guarded by
    `IS_FLASHINFER_AVAILABLE` (RMSNorm, RoPE/activation custom ops, the sampler's FlashInfer
    strategy) or imported only when that flag is set (the FlashInfer attention backends).
- **Change to the notebook (trtllm only).** After installing `tensorrt-llm==1.2.1` into the 3.12
  venv, it uninstalls `flashinfer-python` and fails if `flashinfer` is still importable.
- **Change to `phase0.py` (TensorRT-LLM).**
  - Each arm runs twice: on `attn_backend="VANILLA"` (torch SDPA, dtype-agnostic; the generic
    two-model draft-target path) and on the default `"TRTLLM"`. The default's fp32 support is in C++
    and has to be established empirically.
  - `cuda_graph_config=None` and `disable_overlap_scheduler=True`.
  - Greedy is passed as `temperature=0.0`. The earlier `or None` was greedy only because TensorRT-LLM
    treats all-`None` sampling parameters as greedy.
  - The canonical `speculative_model` field replaces the `speculative_model_dir` alias.
  - Each probe records its engine options and whether FlashInfer was importable.
- **The same review of the SGLang path under FR5 and FR6** found no further fp32 blocker:
  - model forward: `forward_native` RMSNorm, SiLU and RoPE; `F.linear`; Triton attention, no dtype
    restriction;
  - verify: the `tree_speculative_sampling_target_only` JIT kernel, which is fp32-only;
  - sampling: FlashInfer sampling kernels, which take fp32 probabilities.
- **Seeds (Stage A note).** Outside deterministic-inference mode SGLang ignores per-request
  `sampling_seed` and samples from its global generator (`random_seed=0`). Phase 0 is unaffected.
  Stage A must either enable deterministic inference (which forces the torch sampler) or record that
  SGLang draws are seeded per engine, not per request.
- **Effect on design.** None. Phase 0 is non-evaluative.

### FR8 (2026-10-07, after the first TensorRT-LLM Phase 0 attempt, before Stage A): FR7 corrected; torch build for TensorRT-LLM

- **Trigger 1: the FR7 change broke the import.** The TensorRT-LLM record of 2026-10-07 shows every
  arm failing at import:
  `ImportError: cannot import name 'FlashInferAttentionMetadata' from 'tensorrt_llm._torch.attention_backend'`.
  - `tensorrt_llm/_torch/modules/attention.py:15` imports it unconditionally.
  - `attention_backend/__init__.py` exports it only when FlashInfer imports.
  - So TensorRT-LLM 1.2.1 cannot run without FlashInfer. FR7's source review checked only direct
    `import flashinfer` lines and missed this indirect dependency. No sample was drawn.
- **Trigger 2: wrong torch build.** A local install of the exact pins (no GPU) showed it:
  - TensorRT-LLM 1.2.1 is a CUDA 13 build (`cuda-python>=13`, `nvidia-nccl-cu13`, bindings linked
    to `libcublasLt.so.13`).
  - Plain-PyPI `torch==2.9.1` resolves to `2.9.1+cu128`; the Colab record shows `torch 2.9.1+cu128`.
  - NVIDIA's install instructions use torch's cu130 build.
- **Change: FlashInfer stays installed.**
  - `full/d40_trtllm_fp32.py`, loaded by a `.pth` in the trtllm venv, sets
    `IS_FLASHINFER_AVAILABLE = False` in exactly two modules as they load:
    `tensorrt_llm._torch.modules.rms_norm` and `tensorrt_llm._torch.modules.rotary_embedding`.
  - In the full installed 1.2.1 source, these are the only two call sites of FlashInfer kernels in
    `_torch` outside `auto_deploy` and the sampler. Both read the flag at call time and fall back to
    pure torch.
  - The sampler's FlashInfer strategy is switched off with its own option,
    `disable_flashinfer_sampling=True`.
  - The notebook asserts that the hook is active before any probe runs. Each probe records both flags
    and the patched-module list.
- **Change: torch build.** The notebook installs `torch==2.9.1+cu130` and `torchvision==0.24.1+cu130`
  from the PyTorch cu130 index before `tensorrt-llm==1.2.1`, and asserts CUDA 13. A fresh resolve
  gives `torch 2.9.1+cu130`, `flashinfer-python 0.6.4`, `transformers 4.57.3`, and no `-cu12`
  packages. Over the cu130 torch, the TensorRT-LLM install step makes no changes, and
  `uv pip check` passes.
- **Verified locally, without a GPU.**
  - The install and the resolve.
  - The hook in the real venv: with stand-in modules of the same names, exactly the two targets are
    switched, a control module is not, and the switch holds in `spawn` and `subprocess` children
    (TensorRT-LLM starts workers with `multiprocessing` spawn).
  - The rest of the fp32 path in the full source:
    - `trtllm::silu_and_mul` is a dtype-generic Triton kernel;
    - `float32` maps to a KV-cache `DataType.FLOAT`;
    - VANILLA attention handles chunked context over cached tokens, which is the two-model verify
      path for non-TRTLLM backends.
- **Not verifiable without a GPU.** A full import, because `attention_backend/flashinfer.py` calls
  `check_cuda_arch()` at import, and the TRTLLM C++ attention in fp32. The VANILLA arms do not
  depend on either.
- **Effect on design.** None. Phase 0 is non-evaluative.
- **Pre-run review.** Two independent reviews […] ran against the installed
  1.2.1 source. […]
  Neither review found a code blocker for the VANILLA arms. Their findings, now applied to
  `phase0.py`:
  - **TP1 worker in-process** (`TLLM_WORKER_USE_SINGLE_PROCESS=1`). Otherwise the worker is
    started with an MPI spawn, never exercised on Colab, and the recorded hook flags would come from
    the proxy process rather than the one that computes.
  - **Firing evidence for TG1:** `enable_iter_perf_stats=True`, and the per-iteration
    `specDecodingStats.numDraftTokens/numAcceptedTokens` are summed into `spec_counters`.
    - A second review pass of the fixes found that, with the in-process worker, `llm.get_stats()`
      always returns `[]`; only the proxy fills its queue.
    - The probe therefore reads the worker directly with `fetch_stats()` and `_stats_serializer`
      (`executor/base_worker.py`). This was checked locally on a real `IterationStats` binding object:
      `numDraftTokens`/`numAcceptedTokens` come through.
    - It drains once more before shutdown, because the last iteration's stats are appended after its
      responses.
    - `draft_tokens` counts non-zero draft ids when `num_draft_tokens` is 0, and id 0 is a real
      token at V = 8, so the counter is a fired/not-fired indicator, not an exact count.
      `stat_iterations == 0` is recorded as unavailable.
  - **Memory and ordering:** KV cache `free_gpu_memory_fraction=0.3`, `max_num_tokens=1024`,
    `max_batch_size=128` (VANILLA forces `tokens_per_block = max_num_tokens`). All VANILLA arms run
    before the TRTLLM arms.
  - **The attention choice also switches the drafter.** With `attn_backend="TRTLLM"`, draft-target
    uses the chain drafter, which drafts greedily by argmax; with VANILLA, drafts are sampled with
    the request's strategy (`py_executor_creator.py`, `use_chain_drafter`). So VANILLA and TRTLLM arms
    are different speculative paths, not attention-only variants. Each probe records its drafter.
  - **TorchSampler limits (Stage A must decide):** `min_p` is not applied, so TensorRT-LLM S6 is
    temperature/top-k/top-p only. Per-request `seed` is not applied: one generator per engine,
    seeded 42. Both are recorded per probe.
  - **Validation:** all 8 arm configurations validate against the real `TorchLlmArgs`. A stand-in
    run of the probe flow passed and caught an indentation slip before commit.

### FR9 (2026-10-18, after Phase 0 for llama.cpp, vLLM and SGLang, before Phase 0b and Stage A): cost plan, running cells, Holm's m, seeds, Phase 0b

This is the §7 revision that "fixes the running cells, Holm's m and the seed plan". It adds the
owner-approved cost plan of 2026-10-18. Nothing has been sampled against a reference. Two
independent reviews […] of a first draft were applied before this text was
committed […]. Every change states its direction on the verdict.

**1. Running cells (§4, applying its own `BLOCKED` rule to the Phase 0 record).**
- **Original head_dim-8 fixture:** `BLOCKED` on vLLM (all arms time out) and SGLang (Triton
  `K ≥ 16`). Both engines serve F-A through the head_dim-32 re-layout, which §3 allows and which
  must pass §5.1 per engine. TensorRT-LLM and llama.cpp use whichever layout passes Phase 0b;
  HF uses the original.
- **vLLM × {S4, S6}:** `BLOCKED` (15 cells: 12 F-A over VG1–VG6, plus 3 F-B).
  - vLLM refuses `min_p` with speculative decoding. Phase 0 shows the refusal at S6, and the pilot
    showed it at S4 and S6.
  - Phase 0b confirms the refusal at S4 on 0.30.0.
- **Split ids.** Phase 0 showed two §4 ids cover two distinct paths each. Each path is its own cell:
  - SG3 → **SG3s** (standalone draft model) and **SG3n** (n-gram).
  - TG1 → **TG1v** (`attn_backend="VANILLA"`, drafts sampled with the request's strategy) and
    **TG1t** (`attn_backend="TRTLLM"`, chain drafter, argmax drafts). They differ in the drafter, not
    only in attention (FR8).
  - The split adds 17 cells: SG3 +7, TG1 +7 in F-A and +3 in F-B. That raises m (toward pass) and
    adds cells to both sides of the coverage ratio (item 2).
- **Settings an engine does not apply.** An engine that silently ignores a setting has not run the
  §4 cell, so the cell is `BLOCKED`, as for a refusal. The same holds where the setting crashes the
  engine, unless Phase 0b finds a shipped option that applies it. In that case the option is also
  used for that engine's null and `R_eng`, and Phase 0b records it. The direction is toward
  INCONCLUSIVE, through coverage. The cases:
  - **TensorRT-LLM TG2 × {S4, S5, S6}: `BLOCKED` now.**
    - For EAGLE-3, `instantiate_sampler` returns `get_spec_decoder(...)`, which is a TorchSampler
      (`_torch/pyexecutor/_util.py:957`, `speculative/utils.py:170`), before `sampler_type` is read.
    - TorchSampler applies neither `min_p` nor `repetition_penalty`: there is no reference to either
      anywhere in `_torch/pyexecutor` (FR8).
  - **TensorRT-LLM TG1v/TG1t × {S4, S5, S6}** (8 cells: 6 F-A plus the 2 F-B S6 cells). Phase 0b smokes the shipped
    `sampler_type="TRTLLMSampler"` once.
    - It will probably fail: `TRTLLMSampler` declares `MAX_DECODING_TOKENS = 1`, with the comment
      "must be 1 when not in speculative decoding".
    - Each probe records `type(sampler)`. Without a working option, these cells are `BLOCKED`.
    - If the option works, S4–S6 claims name the C++ sampler path, which is not the S0–S3 path.
  - **SGLang × {S4, S6}** (9 cells with F-B SG1 S6). The `min_p` branch crashes at S6 (Phase 0);
    S4 has not been run. Phase 0b smokes S4 and tries the shipped `sampling_backend="pytorch"`.
- **Not added:** llama.cpp's server n-gram path, smoked in Phase 0 as an extra, is not in §4 and stays
  out of the grid.
- **LG3** stays a KNOWN reproduction outside Holm and outside coverage (§4).

**2. Holm's m and coverage.**
- Stochastic F-A cells, S0–S6 per method:
  - vLLM VG1–VG6, 30 cells after the S4/S6 blocks;
  - SGLang SG1, SG2, SG3s and SG3n, 28 cells;
  - llama.cpp LG1–LG2, 14 cells;
  - TensorRT-LLM TG1v, TG1t and TG2, 21 cells;
  - HF HG1, 7 cells.
  - That makes **100**.
- F-B cells over {S0, S2, S6}: VG1, VG2 and VG4 (S0 and S2 only); SG1, LG2, TG1v and TG1t. That makes
  **18**.
- **m = 118**, fixed now.
  - Later revisions may remove cells from the running set but never raise m.
  - A cell that is not run enters Holm with p = 1. TG2 × {S4, S5, S6} is already in that position.
  - Greedy G cells are exact-match checks outside Holm (§5.4).
  - Null cells not run enter the null-family Holm with p = 1.
  - Fixing m at 118, instead of the running count, makes the grid test and the α/m control-validity
    bar (§5.3) stricter than the frozen "running cells" definition. A larger α/m divisor also widens
    the MDE, so more cells can become `UNDERPOWERED`. The direction is toward pass and toward
    INCONCLUSIVE.
- **Coverage for KILL (§6) is counted, as frozen and as in the pilot, over every grid cell including
  `BLOCKED` ones.** That is 118 + the 15 vLLM `min_p` blocks = **133**. KILL therefore needs
  ⌈0.75 × 133⌉ = **100** cells run at N ≥ 5·10⁴ that are neither `UNDERPOWERED` nor
  `REFERENCE_UNRESOLVED`.
  - A cell made INCONCLUSIVE by a rejected null does not count as covered.
- **Consequence, recorded before any data.**
  - The 36 cells that need drafter heads are still unproven: VG4, VG6, SG1, SG2, TG2, plus F-B VG4
    and SG1.
  - Without them, at most 82 cells can run, so **KILL is impossible unless heads load**.
  - Even if every head loads, blocking all 20 setting cells in item 1 (TG2's 3 included) leaves
    at most 98, which is again short of KILL. VG5 folding (item 5) would lower these maxima by 5.
  - In that case the possible verdicts are CONTINUE (one confirmed NEW failure) or INCONCLUSIVE.

**3. Seed plan.**
- **The rule.**
  - Let `h(stage, cell) = int.from_bytes(sha256(f"d40full|{stage}|{cell}").digest()[:4], "big") mod 2³⁰`.
  - Request `j` of a cell has the seed `h(stage, cell) + j`.
  - `j` counts from 0 across the cell's chunks. Chunk `k`, of size `n_k`, covers
    `j ∈ [J_k, J_k + n_k)` with `J_k = n_0 + … + n_{k−1}`, so a short final chunk (N_A not a
    multiple of 10⁴) reuses no index.
  - Stage B uses `h("A", cell) + 10⁷ + j`, so the seed sets are **disjoint from Stage A by
    construction** (N_A < 10⁷), and seeds never repeat within a cell. Phase 0b uses
    `h("P0b", cell)`.
  - Every seed is below 2³¹, which fits llama.cpp's `uint32` seed.
  - The harness writes the seed range into every chunk record.
- **vLLM, llama.cpp server:** per-request seeds as above.
- **HF:** `generate` draws a whole batch from one RNG stream, so HF seeds are **chunk-level**, not
  per-request.
  - Chunk `k` seeds its stream once with `h(stage, cell) + J_k` (Stage B: `h("A", cell) + 10⁷ + J_k`)
    before its first `generate` call.
  - The chunk record stores that stream seed and the batch shape, and marks per-request seeds as
    unused. The draws depend on batch shape and request order, which are fixed per chunk.
  - Resume from cached chunks (§5.5.6) is unaffected, since each chunk reseeds.
- **SGLang:** per-request seeds are ignored outside deterministic-inference mode. That mode switches
  to batch-invariant kernels, so it is not used.
  - Each cell runs in one engine instance with `random_seed = h(stage, cell)`.
  - Chunk records mark per-request seeds as unused.
- **TensorRT-LLM:** TorchSampler seeds one generator with the constant 42 (`sampler.py:959`), so every
  instance would draw the same stream.
  - The harness reaches the sampler at `llm._executor.engine.sampler` (single-process worker, FR8).
    It asserts `sampler._generator is None`, then sets `sampler._global_seed = h(stage, cell)` before
    the first request.
  - The generator is created lazily from `_global_seed` (`sampler.py:976`), and the drafter shares
    this sampler (`model_drafter.py:127`). This is a seeding hook, recorded per chunk; it changes no
    sampling computation.
  - Phase 0b must show its effect: two seeds give different outputs and one seed repeats exactly.
  - Chunk records mark per-request seeds as unused.
  - If the hook has no effect, TensorRT-LLM's Stage B is declared **not seed-disjoint** from Stage A.
    Skipping draws cannot reach a disjoint point in the stream, because generator use per sequence
    varies with drafts and batch shape.
  - A TRTLLMSampler cell runs only if Phase 0b shows an equivalent working seed.
- **Departure from §5.5.6, declared.** For SGLang and TensorRT-LLM, which have no per-request seeds,
  a runtime restart cannot resume mid-cell: a reseeded instance would replay its stream. These two
  engines therefore **redraw the whole cell** after a restart. The partial chunks are logged, kept,
  and never read. The discard does not depend on the data. Other engines resume from cached chunks,
  as §5.5.6 says.

**4. N.**
- **N_A stays 2·10⁵ per cell (§5.4) and nulls are drawn at the same N (§5.2).** The approved cost plan
  had proposed an adaptive look at 5·10⁴ and a smaller null. A simulation done before this revision
  rejected the adaptive look (`full/fr9_power.py`, output in `phase0/fr9_power.txt`; R1
  perturbation, m = 118, 400 replicates).
  - The rule extends at p < 0.05 and Bonferroni-combines the two looks.
  - Power at S0, ε = 0.005: 0.55 adaptive against 1.00 at a fixed 2·10⁵.
  - Power at S3, ε = 0.009: 0.46 against 0.87.
  - The loss comes mainly from stopping cells with p ≥ 0.05 after the first look. The decision rests
    on power alone, not on a cost estimate.
  - Without the smaller null, a rejected null stays as likely as the frozen design makes it.
- **If Phase 0b's batched throughput projects an engine's Stage A over its allotment in item 6,** §7's
  existing rule applies: N_A is lowered uniformly for that engine, not below 5·10⁴.

**5. Cost levers** (procedure only, unless stated otherwise):
- **Batching.** Each `generate` call submits a whole chunk of 10⁴ requests (vLLM, SGLang,
  TensorRT-LLM, HF in batches of 10³). The llama.cpp server runs with `-np 16` and as many client
  threads.
  - Batched fp32 kernels can differ from batch-1 kernels, so δ_num (§5.1) is measured on the
    production batch shape (item 7, F4).
- **Pruned cells are never started:** the blocks in item 1. **VG5** runs only if the item-7
  preflight shows counters different from VG2. Otherwise VG5 is folded into VG1/VG2 and enters Holm
  as `NOT_RUN` (p = 1). That drops 5 cells from coverage, toward INCONCLUSIVE.
- **Cheapest GPU for HF and llama.cpp. This reverses FR3 for these two engines only.** FR3 required
  an L4 because vLLM stalled on a T4; HF and llama.cpp were not the problem.
  - They run on a Colab T4 (sm75). §5.1 references are built on the T4 (§5.1 host rule).
  - llama.cpp is rebuilt for sm75 at **the exact Phase 0 commit** (§2), not current `master`.
  - CPU is not used: §1 asks about the GPU paths.
  - vLLM, SGLang and TensorRT-LLM stay on the L4.
- **Exact `R_eng`.** `R_eng` is computed by scoring the 400 F-A prefixes, one new token each, and
  reading the processed per-step distribution from the sampler path (§5.1). It is never estimated
  by sampling. An engine whose sampler path exposes no processed distribution makes its cells
  `REFERENCE_UNRESOLVED` (§5.1), unless a revision before Stage A says otherwise.
- **Resumable runs.** There is one Stage A notebook per engine. Every chunk is written to Drive with
  the §5.5.2 metadata and its seed range. After a restart, a run resumes at the first missing chunk
  (SGLang and TensorRT-LLM: the first missing cell, per item 3).

**6. Budget allotment** inside the $80 cap. These are planning figures, not limits on the design;
§7's cap rule governs.
- Spending to date (Phase 0) is not yet recorded in compute units. The owner reports it from the
  Colab usage page, and it counts against the Phase 0b allotment.
- Phase 0 and Phase 0b together: up to $10.
- Stage A: up to $30.
- Stage B: up to $30.
- Reserve: $10.
- The owner is told when a phase passes 80% of its allotment.

**7. Phase 0b (non-evaluative, §7 F1–F6).** Same rules as Phase 0: at most 100 sequences per smoke,
and no sample compared with any reference. Before Stage A, it must produce:
- **F1:**
  - completed install records for TensorRT-LLM (FR8, including the CUDA-13 torch assertion) and HF on
    the L4;
  - HF and llama.cpp install records on the T4, with llama.cpp at the Phase 0 commit;
  - the llama.cpp commit itself (FR4).
- **F2:** F-B and the drafter heads, built on CPU and hashed.
- **F3:**
  - every running cell × setting smoked, including S1, S3, S4 and S5, which Phase 0 did not cover;
  - the vLLM S4 refusal, and the item-1 sampler options with `type(sampler)` recorded;
  - the item-3 seed checks;
  - the head and F-B loads;
  - the VG5 counters preflight (`draft_sample_method="probabilistic"`, counters only, against VG2);
  - the effect assertions for **every** §5.3 control, P1–P8.
- **F4:**
  - `R_eng` per engine × setting × fixture on each GPU type, and δ_num against `R_np` at the
    production batch shape;
  - for VG6 (MTP) and SG2 (tree), the independent fp64 reference that §4 requires (otherwise
    `REFERENCE_UNRESOLVED`).
- **F5:** the MDE per cell at α/118, computed with an explicit `level=stats.ALPHA / stats.M_FULL`
  (`code/stats.py`). The pilot scripts `code/f5_mde.py` and `code/analyze.py` keep the default
  α/`M_PLANNED` = α/84 because they reproduce the closed pilot; the full study does not use them
  for F5 or analysis.
- **F6:** batched throughput, and the per-engine N_A under item 4.

A further revision, before Stage A, records the Phase 0b outcomes: the final running set, any
lowered N_A, and the head/F-B status. It may only remove cells.

### FR10 (2026-10-18, after the second TensorRT-LLM Phase 0 attempt, before Phase 0b and Stage A): pyarrow pin

- **Trigger.** The TensorRT-LLM Phase 0 rerun of 2026-10-18 stopped in the install cell, at the FR8
  hook check: `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`, raised from
  `tensorrt_llm.llmapi.llm_utils` through `datasets==3.1.0`. No probe ran and no sample was drawn.
  - TensorRT-LLM 1.2.1 pins `numpy<2`, and its `datasets` pin leaves `pyarrow` unpinned.
  - `pyarrow` 26.0.0, released after FR8, is the first release whose `lib.pyx` refuses NumPy 1.
    25.0.1 has no such check and builds against `numpy>=1.25`.
- **Change.** The trtllm install line adds `"pyarrow<26"` to the `tensorrt-llm==1.2.1` install. Nothing
  else changes. The FR8 hook check, which imports TensorRT-LLM, stays the gate before any probe.
- **Direction.** None on the verdict: an install pin of a data-loading library that no sampling path
  calls. The probe records the resolved `pyarrow` version with the environment block.

### FR11 (2026-10-19, after the TensorRT-LLM Phase 0 record, before Phase 0b and Stage A): TG1t blocked; TG1v drafter hook

- **Trigger.** In the TensorRT-LLM Phase 0 record of 2026-10-19 (`phase0_trtllm_NVIDIA_L4_1792375777.json`,
  `PHASE0_FULL.md`), TG1 failed on every arm. No samples were drawn and nothing was compared with a
  reference. The diagnosis below comes from the TensorRT-LLM v1.2.1 source (tag `v1.2.1`, commit
  `376f7e1b`). No further GPU run was made.
- **TG1t (TRTLLM attention): fp32 speculative decoding is unsupported. `BLOCKED` now.**
  - `AttentionOp::initialize` (`cpp/tensorrt_llm/common/attentionOp.cpp`) enables XQA only for fp16
    and bf16. When speculative decoding is enabled and XQA is off, it fails
    `TLLM_CHECK_WITH_INFO(false, "Speculative decoding mode doesn't support the data type or cross
    attention.")` (line 2977). The Python backend sets `is_spec_decoding_enabled` on SM89
    (`_torch/attention_backend/trtllm.py:1306`).
  - In the record, NULL on TRTLLM attention at head_dim 32 runs, and TG1 on the same model aborts in
    `AttentionOp::initialize` within 17 s. Of the checks in `initialize`, the only ones that depend on
    speculative decoding are this one and the custom-mask/context-FMHA check (line 2899). Both
    follow from fp32. The log tail holds only the native backtrace, so it cannot tell the two apart.
  - No shipped option keeps fp32 and enables speculative decoding on this backend. fp32 is required
    by §5.1.
  - So all 10 TG1t cells are `BLOCKED` and enter Holm with p = 1: F-A S0–S6 (7) and F-B S0/S2/S6 (3).
    FR9 item 1 already counted S4–S6 and F-B S6 (4 cells) as probably blocked.
- **head_dim 8 on TRTLLM attention** fails `mmha_supported` ("Head size 8 is not supported by
  MMHA", `attentionOp.cpp:2647`). The supported sizes are 32, 48, …, 256
  (`decoderMaskedMultiheadAttention.cu:145`). Grid cells already use the head_dim-32 re-layout, so
  no cell changes.
- **TG1v (VANILLA attention): an upstream drafter defect, worked around with an execution-mode hook.**
  - Only the TRTLLM backend takes the chain drafter (`py_executor_creator.py:358`). On VANILLA, the
    dynamic draft loop runs `ModelDrafter.process_decoded_tokens`.
  - For any non-greedy request, `TorchSampler.should_provide_draft_probs` is true (`sampler.py:2790`),
    so the draft request returns generation logits.
  - Those logits live in a `LogitsStorage` buffer allocated lazily inside `HandleLogits.__call__`,
    which runs under `@torch.inference_mode()` (`handle_logits.py:14`; `llm_request.py:73`). The buffer
    is therefore an inference tensor.
  - `process_decoded_tokens` does not run under inference mode, and `generation_logits.squeeze_(0)`
    (`model_drafter.py:476`) changes the view in place. That raises, the executor thread exits
    (`py_executor.py:394`), and `generate` never returns. The same line is unchanged in `v1.3.0rc9`.
  - Greedy requests (the G setting) return no draft logits and would not reach this line. The probe
    never got to G, because every arm runs S0 first and S0 killed the executor.
- **Change (TG1v only).** `full/d40_trtllm_fp32.py`, already loaded from a `.pth` (FR8), now also
  wraps `ModelDrafter.process_decoded_tokens` in `torch.inference_mode()`.
  - The method body is unchanged: the same squeeze, the same CPU→CUDA copy and the same assignment
    to `py_draft_logits`. The rejection step that reads `py_draft_logits` already runs inside
    `TorchSampler`'s inference-mode methods.
  - The hook changes autograd bookkeeping only, not any value. Each probe records it in
    `fp32_hook_patched`.
  - Phase 0b (F3) must show the hook works: TG1v at S0 returns 100/100 in space, with spec counters
    showing drafts and acceptances, with the hook active.
  - If it does not work, the TG1v stochastic cells are `BLOCKED`. G is unaffected.
- **Direction.**
  - TG1t's 10 cells move toward INCONCLUSIVE through coverage. They do not raise m.
  - The largest running set falls from 115 to 105. KILL needs 100 (FR9 item 2).
  - If every head loads but all 20 setting cells of FR9 item 1 are blocked, the maximum falls from 98
    to 92. Without heads it falls from 82 (118 minus the 36 head cells) to 72. In both cases KILL stays impossible.
  - The TG1v hook changes no sampling computation. It can only let cells run that would otherwise
    produce nothing, so it has no direction on the verdict beyond coverage.

### FR12 (2026-10-18, before Phase 0b, before any F-B model or drafter head exists, before Stage A): Phase 0b design

- **Trigger.** Building Phase 0b (FR9 item 7). A source review of the three pinned engines (vLLM `ced6857a`,
  SGLang `e00930c5`, TensorRT-LLM `376f7e1b`) before any Phase 0b run found facts the earlier revisions did not
  have. No F-B model or head has been built, no Phase 0b probe has run, and no sample has been drawn.
- **llama.cpp Phase 0 commit (FR4 gap): `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`.**
  - The Phase 0 run printed the commit only to the notebook console. The notebook was opened from GitHub, so no
    output copy exists in Drive.
  - The commit is reconstructed from `master`'s history. The Phase 0 record started at 2026-10-06 02:32:47Z.
    The last commit before it is `5e03bdd8` (00:42Z), and the one before that is `50569eb8` (2026-10-05 22:34Z).
  - Both commits touch only `ggml/src/ggml-hexagon/` (the Qualcomm backend), which a CUDA build does not compile.
    So the CUDA build is the same whichever of the two Phase 0 cloned.
  - Phase 0b checks out this commit, asserts it, and rebuilds for the T4 (sm75).
- **F-B (§3) choices the frozen text leaves open** (`full/fb_fixture.py`):
  - prompt `[1, 2, 3, 1, 2]`;
  - 2 heads (head_dim 32, which every engine's attention accepts, so there is no re-layout); intermediate 128;
  - draft = target plus noise of scale σ, generator seed 3.
  - The §3 bands are computed over the prefixes 2,000 seeded S0 target sequences visit. The entropy median is
    over the distinct visited prefixes; the acceptance mean is over visits, which is the F-A reach weighting.
  - The GGUF pair uses the pilot's exporter with F-B's config and single-letter tokens.
- **Drafter heads (§3)** (`full/heads.py`).
  - **EAGLE-3.** One checkpoint per fixture, in the format the three loaders share: `LlamaForCausalLMEagle3`,
    one decoder layer, head_dim 32, `draft_vocab_size = V`, unprefixed weight names, the target's embedding
    shared.
    - There is no `d2t`. With one, TensorRT-LLM refuses non-greedy drafting (`sampler.py` 2131-2139) and drops
      the drafts silently.
    - On F-A (one auxiliary layer) `fc` is exactly the identity. TensorRT-LLM builds no `fc` for one captured
      layer, and SGLang skips it when the widths match, so all three engines run the same head.
    - The auxiliary layers are the output of layer 0 on F-A, and of layers 0–2 on F-B. Each engine numbers them
      differently (vLLM `[1]` / `[1,2,3]`, SGLang `[0]` / `[0,1,2]`, TensorRT-LLM `{0}` / `{0,1,2}`).
    - SGLang cannot capture the last layer of a 2-layer target.
  - **Head variants.** Acceptance depends on each engine's own drafting, which cannot be reproduced on CPU. So
    each fixture gets two variants: `tgt` (the target's lm_head) and `flat` (that lm_head times 0.1).
    - Phase 0b reads acceptance from the spec counters at S0. Counters only: no sample is compared with a
      reference.
    - The post-Phase-0b revision keeps, per engine and fixture, a variant with acceptance in [0.3, 0.9].
      Otherwise those cells are `BLOCKED`.
    - The rule is fixed now: `tgt` if its acceptance is in the band, otherwise `flat` if its acceptance is in
      the band. Acceptance is the S0 accepted/draft-token counter delta of that setting alone.
    - Where a counter cannot be read exactly, acceptance is "unavailable" and the variant does not qualify.
      That covers TensorRT-LLM's fallback draft count, which skips token id 0 (FR8).
  - **TG2** is TensorRT-LLM's two-model EAGLE-3 (`eagle3_one_model=False`).
    - The one-model path drafts by argmax, keeps bf16 hidden states for an fp32 model, and does not reset
      VANILLA attention's cached-token counts between draft steps.
    - The two-model path drafts with the request's strategy through the shared TorchSampler, like TG1v.
  - **VG6** (vLLM MTP). vLLM 0.30.0 has no Llama MTP. Its simplest MTP target is MiMo (Qwen2 plus one MTP
    layer).
    - `mimo_FA` is F-A's head_dim-32 re-layout as `MiMoForCausalLM`, with all-zero q/k/v biases. vLLM's Qwen2
      attention takes head_dim = hidden / heads (`models/qwen2.py:150`), so the re-layout is also widened from
      hidden 16 to 64. Every added coordinate is zero, every RMSNorm weight is halved and eps is divided by 4.
      All three factors are powers of two, so the function is unchanged in exact arithmetic.
    - F2 checks in fp64 that its logits equal F-A's at all 400 prefixes. F-A's fp64 reference therefore covers
      it, which is the independent reference §4 requires.
    - Only the MTP layer (random init) is new. It uses k = 1 (one MTP layer).
  - **SG2** is EAGLE-3 with `speculative_eagle_topk=2` and 6 draft tokens. Its reference is the target's own
    (§4); the tree changes only the proposal.
- **TensorRT-LLM S4 and S5 are `BLOCKED` (TG1v, TG2).** TorchSampler applies neither `min_p` nor
  `repetition_penalty`; it reads only temperature, top_p and top_k (`sampler.py` 341-357). S4 and S5 pass
  `temperature=1.0`, so they are sampled from S0's distribution with the stage silently ignored
  (`sampling_utils.resolve_sampling_strategy`). Each probe records the resolved strategy.
  - This is FR9 item 1's silently-ignored case. It adds TG1v × {S4, S5} and TG2 × S5 to TG2's existing
    S4–S6 block.
  - The shipped `sampler_type="TRTLLMSampler"` is still smoked once (FR9 item 1). Its `update_requests` appends
    one token per step and never sets accepted draft counts (`sampler.py` 2831, 3113-3153), so it is not expected
    to verify drafts.
- **vLLM runs two samplers.**
  - `draft_model` and `ngram` (VG1, VG2, VG3, VG5) fall back to the V1 model runner.
  - The non-speculative engine and `eagle3` / `mtp` (VG4, VG6) use Model Runner V2 (`config/vllm.py` 675-723,
    2838-2850).
  - §5.1 requires `R_eng` and the null on the sampler that draws the samples. So each V1 cell's null and `R_eng`
    run with `VLLM_USE_V2_MODEL_RUNNER=0`, and each V2 cell's with the default. The record states which runner
    the engine actually used.
- **VG5.** The V1 rejection sampler has no `block` method; it checks only `"synthetic"` (`rejection_sampler.py`
  84-95). VG5 is therefore expected to equal VG1, as Phase 0's identical counters already showed.
  - The V1 proposer samples drafts only when `rejection_sample_method == "standard"` and
    `draft_sample_method == "probabilistic"` (`llm_base_proposer.py` 261-264). So FR9's preflight variant
    (block plus probabilistic) drafts greedily, like VG1.
  - V1 probabilistic drafting ignores seeds (same file, 1877), so VG2's counters do not repeat even against
    themselves.
  - The preflight therefore runs VG2 as frozen and also VG1 on the same 100 seeds, and records the proposer
    flag. VG5 is folded if its counters equal VG1's.
- **vLLM FlashInfer sampler off** (`VLLM_USE_FLASHINFER_SAMPLER=0`) for every vLLM null, cell and `R_eng`.
  - vLLM uses FlashInfer top-k/top-p only for some batch compositions, and never when processed logprobs are
    requested (`v1/worker/gpu/sample/sampler.py` 300-308).
  - With it on, null draws and `R_eng` could run different kernels. This is the pilot's Triton-vs-PyTorch lesson.
- **`R_eng` paths.**
  - **vLLM:** processed logprobs, `max_logprobs=-1`, per runner.
  - **HF:** `generate` scores, at batch 1 (the HG1 path; assisted generation is batch-1 only,
    `generation/utils.py:3999`) and at 10³ copies (the null's batch).
  - **llama.cpp:** `post_sampling_probs` on the `-np 16` server.
  - **SGLang:** `return_sampling_mask` with `sampling_logprobs_mode="support"` (the sampler's own post-filter
    weights) for S2, S3, S4 and S6, with top_k = V where a setting has none; `return_logprob` for S0 and S1.
  - **TensorRT-LLM:** the processed probabilities captured from the live TorchSampler call
    (`sampling_utils.sample`, a module global in the executor's grouped sampler, `sampling_utils.py` 598) on the
    in-process worker.
    - Each prefix is scored by one call of 128 identical one-token requests (`max_batch_size`), so every
      captured row belongs to that prefix. TorchSampler's logprobs are raw by specification (`sampler.py`
      2040-2066).
  - **Production batch.** Scoring uses it: vLLM, SGLang and HF send exactly 10⁴ requests (vLLM, SGLang) or
    10³ copies (HF null), each mapped to its prefix.
  - δ_num is the maximum TV over every returned row against its prefix's reference, not over one copy. The
    largest disagreement between copies is recorded too.
  - **F-B scoring set.** The prefixes are those with expected visits ≥ 100 at N_A, half the test's 200, so
    that prefixes a real null reaches by chance are covered.
  - **SGLang `sampling_backend="pytorch"`** is FR9 item 1's rescue option for S4 and S6. Its semantics differ
    from the FlashInfer default: the cumulative mass is taken before top-k, and min_p is measured against the
    original maximum (`sampler.py` 766-803).
    - So `R_eng` for S4 and S6 is also scored on that backend, against its own fp64 order
      (`rnp_full.sglang_torch_dist`).
    - A rescued cell uses that backend for its draws, its null and its `R_eng`.
  - **SGLang S5 cannot be scored by prompt.** SGLang penalises output tokens only, and never prompt tokens
    (`repetition_penalty.py` 34-38), so a prefix passed as a prompt is not penalised. Phase 0b records the S5 row.
    Unless a revision finds an output-penalised path, SGLang S5 cells are `REFERENCE_UNRESOLVED`.
- **Throughput (F6)** is measured on the prompt `[4, 5, 6, 4, 5]`, which no grid cell uses, so no drawn sequence
  belongs to a cell's distribution. Like the scoring requests, these draws are not smoke samples of a cell and
  are not capped at 100. They are timed and discarded unread.
  - The seed checks of every engine, and vLLM's VG5 counters preflight, use the same prompt. They draw no
    sequence of any grid cell.
  - So each cell × setting draws at most its 100 smoke sequences. HF's control cells also draw their effect
    assertion runs, which §5.3 requires.
  - HF effect assertions add a sham run: the unpatched arm twice with one seed, which must agree exactly
    before a difference counts as an effect.
  - The comparison is matched. It uses the first verifier call, which has the same inputs in both runs:
    - P1: the draft q at the first drafted position;
    - P2: whether the drafted candidate ids changed (a greedy draft leaves q unchanged and changes the ids);
    - P3, P4 and P6: the full p or acceptance ratio, provided the candidate ids agree (otherwise an alignment
      failure, not an effect);
    - P5: the target cache length counted at the crop itself.
  - **Isolation.** Each engine × fixture × method smoke runs in its own child process, so a crash cannot take
    other methods with it.
    - Settings run safe first (S0, S1, S2, S3, S5, G), then the refused or ignored ones.
    - SGLang's S4 and S6 (the min_p branch that killed its scheduler in Phase 0) each get their own child.
  - Engine versions are read from package metadata, so the supervising process imports no engine outside a
    time limit.
  - **HF torch pin.** Phase 0's HF run used the runtime's preinstalled torch 2.11.0, not the §2 pin 2.14.1;
    the HF notebook now installs torch 2.14.1 and asserts both pins and a CUDA kernel on the T4 before any
    probe. This changes no cell; it restores the frozen configuration.
  - **TensorRT-LLM counter provenance.** Each smoke cell records `fallback_requests`: requests whose draft count
    came from the executor's non-zero-id fallback (`py_executor.py` 834-837). Acceptance is available only when it
    is 0.
- **F5 for F-B.** The R1 perturbation is applied per step, and the per-prefix χ² are combined by Fisher's method.
  - The simulation follows the frozen procedure on the prefix trie. A child's visits are its parent's count
    of that token. An independent simulated null control fixes the prefix set from its realised visits
    (≥ 200), and the cell drawn under the alternative is tested at those prefixes.
  - An MDE above the 0.2 search cap is recorded as such.
  - A visited prefix whose pooled χ² has no degree of freedom (its support pools into one bin, e.g. a single
    surviving token at S6) carries no test and is left out of the Fisher combination. Stage A's F-B statistic
    uses the same rule.
  - F-A applies R1 to the sequence distribution; F-B applies it per step.
- **Direction.**
  - The TensorRT-LLM S4/S5 blocks (3 cells) move toward INCONCLUSIVE through coverage; m stays 118.
  - SGLang S5 (`REFERENCE_UNRESOLVED` unless resolved) would remove up to 4 more F-A cells.
  - The head-variant choice uses counters only, never a conformance result. Proposals with different
    acceptance exercise the verifier's accept and reject branches in different proportions, though. So the
    choice, like any backend choice, can change which paths are covered and the power against some defects.
    That direction is unknown, and it is stated here before any counter is read.
  - Everything else is procedure, with no direction on the verdict.

### FR13 (2026-10-19, after the Phase 0b records, before Stage A): Phase 0b outcomes, running set, Stage A procedure

- **Trigger.** The five Phase 0b records of 2026-10-19 (`phase0b/records/`, summarised in `phase0b/COLAB_RESULTS.md`).
  Phase 0b is non-evaluative: smokes were checked for crash, shape, firing and timing only; no sample has been
  compared with a reference. This is the revision FR9 item 7 requires before Stage A. As to the running set, it only
  removes cells (m stays 118). It also fixes the Stage A procedure.
- **Removals, by the rules already frozen.**
  - **Drafter heads out of band (FR12 rule, §3).** S0 acceptance from the counter deltas, at both variants:
    vLLM VG4 F-A 0.016 (`tgt`) / 0.019 (`flat`), F-B 0.044 / 0.062; SGLang SG1 F-A 0.020 / 0.023, SG2 F-A
    0.026 / 0.027, SG1 F-B 0.044 / 0.050. No variant is in [0.3, 0.9], so every VG4, SG1 and SG2 cell is
    `BLOCKED`. VG6's MTP layer (one variant) has acceptance 28/252 = 0.11, outside §3's band: `BLOCKED`.
  - **TG2** fails to run at both variants (`ValueError: No free slots`, EAGLE-3 slot manager): `BLOCKED`.
  - **VG5** is folded: the preflight's counters equal VG1's exactly (171 / 513 / 121), with the proposer flag off.
    It enters Holm as `NOT_RUN`, as FR9 item 5 fixed.
  - **vLLM speculative S4/S6** refuse (`VLLMValidationError`); already `BLOCKED` (FR9).
  - **SGLang S4 and S6.** The default sampler kills the scheduler (`top_k_renorm_probs: probs must be contiguous`)
    at every S4/S6 cell. The `sampling_backend="pytorch"` option did run (NULL and SG1 smokes, `reference_pytorch`),
    so FR9 item 1 would allow a rescue; but no running method (SG3s, SG3n) was smoked with it, and its δ_num was
    measured without the IEEE setting. Blocking these 4 cells is therefore **a choice, not a forced removal**: it
    avoids adding an unsmoked backend to Stage A. Direction: toward INCONCLUSIVE through coverage (KILL is already
    impossible, below).
  - **SGLang S5** stays `REFERENCE_UNRESOLVED` (FR12): no output-penalised scoring path was found.
  - **TensorRT-LLM TG1v × S6** (F-A and F-B). TorchSampler ignores min_p; `TRTLLMSampler` has no `_generator`, so the
    seed hook cannot reach it and FR9 item 3 forbids running it unseeded: `BLOCKED`.
- **Reference precision (§5.1), and the fix that is tried once.**
  - vLLM's and SGLang's δ_num are about 2·10⁻⁴ on F-A and 1·10⁻³ on F-B. They agree to 3–4 significant digits
    across both engines and both vLLM runners, with the same worst prefix (359 at S0, 247 at S1), and the error is
    spread over every row: per-token log-ratios to `R_np` of about
    10⁻³ (`vllm_FA_NULL_S0_V1.npz`). The gate 6·δ_num < MDE/10 needs below about 2·10⁻⁵. llama.cpp, HF and
    TensorRT-LLM (VANILLA torch attention) are at 1–2·10⁻⁷ (F-A) and 1·10⁻⁶ (F-B).
  - Probable cause, from source (a diagnosis the gate's rebuild tests, not yet an established one): both engines run
    Triton attention (vLLM `triton_unified_attention.py` 545-589; SGLang
    `attention_backend="triton"`, FR5), whose `tl.dot` on fp32 inputs defaults to TF32 on sm80 and newer. vLLM's
    own matmul precision is `"highest"` (`envs.py` 91), so the matmuls are not the cause.
  - **Change.** Every vLLM and SGLang null, cell and `R_eng` runs with `TRITON_F32_DEFAULT=ieee` (Triton's documented
    default-precision knob, `knobs.py` 496). This is an environment setting, not an engine-code change, and it applies
    to the speculative and the non-speculative paths alike. Claims about these two engines name it.
  - Phase 0b's F4 for these two engines is therefore incomplete. It is completed as the **first step of each Stage A
    notebook**, before any Stage A draw: `R_eng` and δ_num are rebuilt with the setting on, for every engine × setting ×
    fixture the running set needs. This also re-verifies the host (§5.1), and it is non-evaluative.
  - **Gate rule, fixed now.** An engine × setting × fixture runs only if its rebuilt `R_eng` is complete (every
    prefix, finite) and 6·δ_num < MDE/10, with MDE the F5 value at N_A: `mde_tv_NA` for F-A and `mde_eps_NA` for F-B
    (`phase0b/f5_mde_*.json`). Otherwise its cells and its null are `REFERENCE_UNRESOLVED`. The rule is applied by
    the harness (`stagea.py`), and the gate record is written before any draw.
  - The same gate runs for llama.cpp, HF and TensorRT-LLM in their Stage A notebooks (host re-verification).
  - **SGLang S2 and S3 are expected to fail the gate for another reason.** Phase 0b's default-backend reference
    crashed at F-A S2 inside the sampling-mask capture (`top_k_renorm_prob`, "probs must be contiguous"), after S0 and
    S1; S3 uses the same masked path and was never reached. The IEEE setting does not touch that path. The gate runs
    each SGLang setting in its own child, so S0 and S1 are unaffected; if S2 and S3 crash again they are
    `REFERENCE_UNRESOLVED`, and SGLang keeps 4 cells.
  - For F-B the gate compares δ_num (a TV distance) with `mde_eps_NA` (the ε of F5's per-step search), as fixed above;
    this is up to about twice as lenient as a TV bound, and moot at the observed margins.
  - **HF on the L4 has no Phase 0b `R_eng` yet.** If its L4 gate fails, HF is rerun on the T4 (FR9 item 5's plan) into
    its own output folder (`d40_full_stagea_hf_t4.ipynb`, `stageA_hf_t4/`), rather than losing its controls.
  - The gate's MDE is F5's at the run's N: `*_NA` at N_A = 2·10⁵, or `*_Nfloor` if §7's budget rule lowers N to
    5·10⁴; production runs at any other N are refused, since F5 computed neither.
- **Seed checks.** Two cells did not repeat under one seed:
  - **vLLM VG2:** expected (FR12: V1 probabilistic drafting ignores seeds). Its target-side sampling is seeded per
    request; the draft stream is not.
  - **TensorRT-LLM TG1v:** the NULL engine repeats and another seed differs, so the hook works; the draft path adds
    nondeterminism. FR9 item 3 asked that one seed repeat exactly; for TG1v that condition is not met, and this is a
    prospective relaxation of it for TG1v only, not a pass.
  - For both, draws are not reproducible from their seed. Stage B seeds stay disjoint from Stage A within each cell by construction (FR9 item 3),
    and a minimal reproduction of a failure in these cells must be coin-injected (§6 allows it). No direction on the
    verdict.
- **The running set** (cells drawn at N_A, if their gate passes):

  | Engine | F-A cells | F-B cells | Nulls | Controls |
  |---|---|---|---|---|
  | vLLM (V1 runner) | VG1, VG2, VG3 × S0, S1, S2, S3, S5 (15) | VG1, VG2 × S0, S2 (4) | V1 × the same 7 settings | P7 (VG1, S2), P8 (VG2, S0) |
  | SGLang (default backend) | SG3s, SG3n × S0–S3 (8) | — | 4 | — |
  | llama.cpp | LG1, LG2 × S0–S6 (14) | LG2 × S0, S2, S6 (3) | 10 | — |
  | TensorRT-LLM | TG1v × S0–S3 (4) | TG1v × S0, S2 (2) | 6 | — |
  | HF | HG1 × S0–S6 (7) | — | 7 | P1, P2, P5 at S2; P3 at S2 and S4; P4 at S6; P6 at S0 (§5.3) |

  That is **at most 57 cells** of m = 118 (53 if SGLang S2/S3 fail the gate, as expected); everything else enters
  Holm with p = 1. Greedy G runs 1,000 times per running
  method (VG1, VG2, VG3, SG3s, SG3n, LG1, LG2, TG1v, HG1; F-B for VG1, VG2, LG2, TG1v), where that fixture's S0
  gate passes, since the argmax chain is read from that `R_eng`.
- **Consequence, recorded before any draw.** Coverage for KILL needs 100 of 133 (FR9 item 2); at most 57 can run.
  **KILL is impossible.** The possible verdicts are CONTINUE (one NEW failure confirmed in Stage B) and INCONCLUSIVE.
  Stage A still runs, because a confirmed NEW failure in the running set decides CONTINUE.
- **Stage A procedure.**
  - **Cell keys and seeds.** A cell's key is `<engine>/<fixture>/<method>/<setting>`, a null's
    `<engine>/<fixture>/NULL/<setting>` (vLLM: `NULL@V1`), a control's `<engine>/FA/<control>/<setting>`. Seeds
    follow FR9 item 3 with `h("A", key)`. The engine prefix keeps nulls of different engines apart.
    Ranges `[h, h + N_A)` of different cells can overlap (llama.cpp's F-A null S2 and S6 share 2,710 seed values at
    N_A = 2·10⁵). FR9 item 3 fixes the seeds by hash and requires uniqueness within a cell and Stage A/B
    disjointness **within a cell**, not disjoint ranges across cells (a cell's Stage A range can also meet another
    cell's Stage B range), and the hash is not changed now. Each cell's test reads only its
    own draws, so an overlap can make two cells' tests dependent; it biases neither.
  - **Chunks.** 10⁴ draws per chunk, 20 chunks per cell (N_A = 2·10⁵). A chunk file holds the distinct sequences and
    their counts, the seed range (or stream seed / engine seed), the spec counters, the environment and the harness
    SHA (§5.5). Counters (§5.5 item 3): vLLM's metrics; SGLang's `meta_info`; TensorRT-LLM's iteration stats; for
    llama.cpp, the per-request `draft acceptance` lines of the server log (accepted / generated draft tokens); for HF,
    a counter on the verifier call `_speculative_sampling` (draft tokens and matches). One output folder holds one
    chunk plan and one GPU type; a resume with another N_A, chunk size or GPU type is refused. Draws are counted, never compared with a reference, inside the notebook; the statistics run later, on
    CPU, from the chunk files (§8).
  - **Resume.** vLLM, llama.cpp and HF resume at the first missing chunk. SGLang and TensorRT-LLM redraw a whole cell
    after a restart (FR9 item 3); partial chunks are moved to `partial/`, kept and never read.
  - **Effect assertions (§5.3).** vLLM's P7/P8 are checked on every chunk's own draws; HF's P1–P6 run the Phase 0b
    matched assertion at the start of every chunk, with that chunk's stream seed. **P2 changes**: its quantity, the
    drafted ids at the first verifier call, is unchanged whenever the sampled draft equals the argmax draft, which
    happens with probability about 0.18 at S2 (fp64, committed weights), so a single seeded call would abort about
    one chunk in five at random. P2's assertion therefore draws each of its 20 sequences alone (seeds s, …, s + 19),
    requires the sham to match on every one, and requires the drafted ids to change on at least one (failure
    probability about 10⁻¹⁵). This is a property of the check, fixed before any Stage A draw, not of the data. A failed assertion aborts the control
    and stops that engine's notebook: an invalid control means an invalid harness (§6), so further spending is waste.
  - **Order** inside each notebook: gate, then controls, then nulls, then cells and greedy, each phase finishing
    before the next starts, so a short budget loses cells last, as §7 orders.
- **Cost plan (owner's request to save cost; procedure only).**
  - **HF moves to the L4**, reversing FR9 item 5 for HF only. HG1 and P1–P6 draw one sequence per `generate` call
    (assisted generation is batch-1 only), which is CPU-bound (27.6 seq/s on the T4). The draws are split by chunk
    over `os.cpu_count()` worker processes on one GPU; chunk-level seeds make the result independent of the split.
    The gate rebuilds HF's `R_eng` on the L4 (§5.1 host rule).
  - llama.cpp stays on the T4. vLLM, SGLang and TensorRT-LLM stay on the L4.
  - Every notebook starts with the gate and releases the runtime as soon as the gate leaves nothing to draw.
  - Projected Stage A use, from Phase 0b throughput: vLLM about 1.5 h, SGLang about 1.5 h, TensorRT-LLM about 5 h,
    HF about 3 h (all L4); llama.cpp about 3.5 h (T4). These are estimates; §7's cap rule governs, and the owner is
    told at 80% of the Stage A allotment.
- **Direction.**
  - The removals move toward INCONCLUSIVE through coverage; m stays 118. KILL is now impossible.
  - The precision setting is meant to make references resolvable; it changes no sampling rule. It does change the
    rounding of kernels that the speculative and non-speculative paths use **differently** (in SGLang, for example,
    the plain decode kernel reduces with `tl.sum`, while the extend/verify kernel uses `tl.dot`). A conformance
    difference that came only from TF32 rounding in one path could therefore disappear. Its effect on detecting
    failures is unknown, and claims for these engines cover the IEEE configuration only.
  - The gate is non-evaluative and its rule is fixed before it runs.
  - HF's GPU move, the worker split and the chunk procedure are procedure, with no direction on the verdict.

### FR14 (2026-10-20, after the Stage A analysis, before any Stage B draw): Stage B set and procedure

- **Trigger.** The Stage A analysis (`stagea/STAGEA_RESULT.md`, […]) found a valid harness: all 9 controls
  reject at p < α/118, no null is rejected, and all 13 greedy runs pass. Holm rejects 6 grid cells, all TensorRT-LLM
  TG1v: F-A S0–S3 and F-B S0, S2. No Stage B draw has been made.
- **Stage B set (§5.4, applied as frozen).** The 6 rejected cells and their 6 nulls: `trtllm/FA/{TG1v,NULL}/{S0..S3}`
  and `trtllm/FB/{TG1v,NULL}/{S0,S2}`.
  - Each is drawn at N_B = 10⁶, in chunks of 10⁴.
  - TensorRT-LLM has one stream seed per cell (FR9 item 3), set to `h("A", key) + 10⁷`. That is disjoint from Stage
    A within each cell.
  - §5.4 names only cells and nulls. Stage B therefore draws no controls and no greedy G.
- **Procedure.** `stagea.py --stage B` reuses the Stage A harness unchanged except for the stage, the cell list, the
  seed offset and N.
  - The 12 cells are split into four output folders, `stageB_trtllm_{fa01,fa23,fbs0,fbs2}`. Each runs as its own
    notebook on its own L4, in parallel. Every cell is in exactly one folder.
  - Each folder rebuilds its own gate on its L4 (§5.1 host rule), before any draw, with FR13's rule.
  - The gate's MDE is F5's at N_A, because F5 computed none at N_B. An MDE at 10⁶ would be smaller, by about √5 for
    a χ² test, and its bound stricter. Even then, Stage A's margins hold: at least 6 times on F-B S0 and 40 times
    on F-A S2. Stage A's δ_num values for TensorRT-LLM were 1.7–2.1·10⁻⁷ (F-A) and 1.1–1.2·10⁻⁶ (F-B), against
    bounds of 1.8·10⁻⁵–7.2·10⁻⁵ and 1.6·10⁻⁵–1.6·10⁻⁴.
  - A setting that fails its gate is `REFERENCE_UNRESOLVED` in Stage B, and its cells cannot be confirmed.
- **F-B scoring set at N_B.** FR12 scores the prefixes with expected visits of at least 100, half the test's 200, so
  that a null's chance visits are covered. At N_B a null selects prefixes that are rarer per draw than at N_A.
  - `C.fb_prefixes` therefore takes the run's N, so Stage B scores expected visits ≥ 100 at 10⁶.
  - At N_A the set is unchanged, and so is Stage A.
- **Statistics (§5.4).** These are Stage A's tests:
  - F-A is the pooled χ² against the sequence distribution of the gate's `R_eng`.
  - F-B is the per-prefix χ² with Fisher's method, at the prefixes with ≥ 200 visits in the Stage B null.
  - Bonferroni is applied at 0.01 per family: the 6 cells at 0.01/6, and the 6 nulls at 0.01/6.
  - A cell is **confirmed** if its p < 0.01/6 and its null is rejected in neither stage (§6). Stage A's nulls for
    these settings were not rejected.
  - The analysis script is committed before the Stage B chunks are read.
- **What a confirmation does and does not decide (§6).** CONTINUE needs, besides a confirmed cell:
  - a NEW classification after the logged novelty search (§5.5 item 5), including TensorRT-LLM issues and PRs on the
    draft-target and VANILLA paths;
  - a coin-injected minimal reproduction (FR13: TG1v draws do not repeat under one seed).
  The Stage A diagnostic (repeated-token outcomes over-drawn; `STAGEA_RESULT.md`) is a hypothesis for that
  reproduction, not a finding.
- **Cost.** About 20 L4-hours of draws plus four installs and gates, about 21–22 L4-hours in all. This is inside FR9
  item 6's Stage B allotment.
- **Direction.**
  - The set, N_B, the seeds and the tests are as frozen.
  - Omitting controls and greedy follows §5.4's wording. Stage A already validated the harness, so this has no
    direction.
  - Taking the gate's MDE at N_A leaves the gate no stricter than Stage A's. A setting that passes could be one a
    10⁶-level gate would fail, which leans toward confirming. The δ_num margins above make that moot.
  - The larger F-B scoring set adds coverage only, so the null's selected prefixes have references. Without it, a
    selected prefix would be `REFERENCE_UNRESOLVED`. No direction.
  - The four-way split is procedure.

### FR15 (2026-10-21, after the NEW/KNOWN search, before any reproduction run or design): KNOWN reading for TG1v

- **Owner's decision, recorded before the reproduction is designed or run.** The pilot's frozen §6 governs NEW/KNOWN:
  a confirmed cell is KNOWN only if "matched by such a report", i.e. a public report of that failure in the engine's
  issues, PRs or announcements. The search brief's wider clause ("a fix to the draft-target rejection sampling … even
  if it does not describe the symptom") does not amend §6 and is not applied.
- **Consequence.** TensorRT-LLM TG1v item (1) in `stagea/CLASSIFICATION.md` is **NEW**. The six confirmed TG1v cells count
  toward CONTINUE, subject to the coin-injected minimal reproduction, which is still required.
- **Disclosure.** Under the brief's clause the item would be KNOWN via TensorRT-LLM #2297/#2298 (a 2024 speculative-decoding
  accuracy fix in the C++ path). Any write-up states this alternative reading and its consequence (no CONTINUE).
- **Direction.** Toward CONTINUE: this reading makes CONTINUE reachable where the brief's clause would not. It is recorded
  here, with the direction stated, before the reproduction exists, so the reproduction's outcome cannot inform it.

### FR16 (2026-10-21, after FR15 and a source read of v1.2.1, before any reproduction run): coin-injected reproduction of TG1v

- **Source read (hypothesis, not a finding).** In v1.2.1 `ModelDrafter._create_draft_request` (`model_drafter.py:112-129`)
  always passes `exclude_last_generation_logits=True` ("prepare_draft_tokens uses overlap scheduling"), including on the
  overlap-off dynamic loop the harness uses. `LogitsStorage.get(exclude_last=True)` drops the last logits chunk, so
  `py_draft_logits` would have fewer rows (R) than draft tokens (K). `get_rejected_indices` verifies only R tokens, but
  `_process_draft_tokens_rejection_sampling` (`sampler.py:1366-1445`) accepts `get_draft_token_length` = K when none is
  rejected. The remaining draft tokens would be emitted unverified, from the draft distribution. If the context step's
  logits are also missing, the rows are shifted by one. The reproduction decides which, if either, holds.
- **Procedure** (`repro_trtllm.py`, `build_repro_notebook.py`, run once on a Colab L4; TensorRT-LLM 1.2.1 and every Stage
  A/B engine option, fixture F-A, fp32 hook; seeds `h("R", key)`). Each arm runs in its own process, and every
  verification call is recorded with the engine's own `py_draft_logits` and `py_target_probs`.
  - **A1:** v1.2.1's `get_rejected_indices` with its uniform draw replaced by a constant u ∈ {0, 1}, at S0 and S2, with
    200 requests per (setting, u).
  - **A2 (hook control):** as A1 at S2, u = 1. FR11's inference-mode wrapper is replaced by v1.2.1's
    `process_decoded_tokens` with `squeeze_(0)` made out of place.
  - **A3 (fix):** as A1 at S2, u = 1, with `_create_draft_request` passing `exclude_last_generation_logits=False`. Then
    real coins, 10⁵ F-A draws each at S0 and S2.
- **Checks** (`repro_analyze.py`, CPU, run before any write-up). C1 R vs K. C2: an event accepts more draft tokens than R.
  C3: an accepted draft token has probability 0 in its own `py_target_probs` row. C4: each draft-logits row is matched to
  the fp64 draft model at the aligned or the shifted prefix (|Δ| ≤ 10⁻⁴). A3's draws use Stage A's F-A χ² against the
  Stage B L4 `R_eng` tables (`stageB_trtllm_fa01`), at Bonferroni 0.01/2.
- **Decision.**
  - **Reproduced** (§6's minimal reproduction) if A1 has at least one C2 event and, at S2 with u = 1, at least one C3
    event. Both are deterministic violations of rejection sampling, computed from the engine's own tensors.
  - **Hook excluded** if A2 shows C2 and C3 as well.
  - **Cause confirmed** if A3's events show no C2 or C3 and neither A3 draw test rejects. If A3's events are clean but a
    test rejects, a second defect remains. That is reported, and the mechanism is called incomplete.
  - Without "reproduced", §6's reproduction requirement is unmet, and CONTINUE is not reached on TG1v.
- **Cost.** One L4 session, about 1 h (one install, three engine starts, 2 × 10⁵ draws at about 54 s per 10⁴): about
  2 CU. Inside FR9 item 6.
- **Direction.** None on the classification: FR15 fixed it. A1 can only reproduce or fail to reproduce. A2 and A3 are
  diagnostics. The reproduction uses the Stage B L4 gate tables rather than rebuilding R_eng, since the same GPU type and
  pins apply (§5.1).
- **FR16 amendment (same day, after a review […], before any run).**
  - Any A3 draw with probability 0 under R_eng, or out of space, rejects, whatever the χ² p.
  - Every verdict needs coverage: all planned runs present, each events run with ≥ 20 rejection-sampling
    verifications, and both A3 draw runs at exactly 10⁵. "Hook excluded" and both "cause" flags also require
    "reproduced".
  - A3's draws are chunked: chunk j0 is seeded `h("R", key) + j0` and saved as it finishes.
  - Arms run under phase0's fatal-marker watchdog (killed 120 s after a fatal log line, at most 1 h per arm).
  - Direction: stricter only.

### FR17 (2026-10-21, after FR16's result and three CPU C4 diagnostics, before any FR17 run): drafter trace for C4

- **Trigger.** FR16's C4 could not say which drafting step each `py_draft_logits` row comes from. Three CPU diagnostics
  found no fp64 draft forward pass at any tried context that matches a recorded row within 10⁻³:
  `c4_diag.py`, `c4_search.py` (299,593 contexts) and `c4_rope.py`.
  - A v1.2.1 source read (`model_drafter.py` 112-260, 787-846, 963-1040) gives two hypotheses, neither tested.
    - (a) After its first round, a two-model request is redrafted as a generation request that feeds only its
      last token. The draft KV cache may then hold positions that were never written, or were written by
      rejected drafts.
    - (b) `process_decoded_tokens` assigns `py_draft_logits` again at every draft position. The rows the verifier
      sees are the last request's `generation_logits` after `exclude_last`, and context-step logits may not be
      among them.
- **Procedure** (`c4_probe.py`, `build_c4_probe_notebook.py`, run once on a Colab L4). TensorRT-LLM 1.2.1 with every
  Stage A/B engine option, fixture F-A, the fp32 hook and FR11's drafter hook; seeds `h("C4", key)`.
  - **P2:** the draft model loaded alone, without speculation. Live S0 probabilities at the first 100 F-A prefixes
    (`p0b_trtllm.step_table`).
  - **P1:** TG1v at S0 with real coins, 200 one-request `generate` calls. Each records:
    - every `forward_draft_model` call: the draft request's kind, tokens, context position and output logits;
    - every `process_decoded_tokens` call: the draft position, the drafted token and the assigned
      `generation_logits`;
    - every verification (FR16's Recorder), tagged with its drafting round.
- **Analysis** (`c4_probe_analyze.py`, CPU […]). At S0 the processed distribution is softmax(logits).
  - **Q1, draft load:** P2 against the fp64 draft, clean if max |Δp| ≤ 10⁻⁴.
  - **Q2, forward fidelity:** each P1 forward's sampled row against the fp64 draft at that draft request's own
    tokens, clean if max |Δp| ≤ 10⁻⁴.
  - **Q3, alignment:** each verification row i is located among its round's forward rows (raw-logit |Δ| ≤ 10⁻⁵).
    It is labelled aligned if that forward produced d_i, shifted±s if it produced d_{i±s}, and no-forward-match
    otherwise.
- **Reading.**
  - Q3 settles C4: it names the drafting step behind each row.
  - Q1 clean with Q2 unclean means the drafter conditions on a state other than its request's tokens (hypothesis a).
  - Q1 unclean means the draft model as loaded differs from the fp64 draft. FR16's C4 reference is then invalid,
    but FR16's C2 and C3, which use only the engine's own tensors, are unaffected.
- **Not affected.** The §6 verdict, FR16's flags and every Stage A/B result. FR17 is a diagnostic of the mechanism
  only. Draws are not drawn and not tested.
- **Compute-unit check** […].
  - GPU needed: TensorRT-LLM has no CPU path.
  - Work cut to the minimum: S0 only, 200 requests, no draws, P2 at 100 prefixes rather than 400.
  - One runtime and one install for both arms.
  - L4, the card Stage A/B used, so the engine environment is the same.
  - Estimate about 0.4 L4-h (mostly the install), about 0.7 CU.
- **Direction.** None on any preregistered result.
- **FR17 amendment (same day, after a review […], before any run).**
  - Q2 compares forward k with the fp64 draft at the round's first draft-request tokens plus d_<k. Later forwards
    read the previous drafted token from device tensors before the request's token list is updated, so the
    recorded tokens lag by one.
  - Q3 counts a row that matches more than one forward as ambiguous. It does not take the first match.
    "Q3 settles C4" holds only for rows with exactly one match.
  - Q1 is clean only if all 100 prefixes are present, every value is finite, and the copies agree
    (copy_dev ≤ 10⁻⁴).
  - The notebook's run cell catches a timeout or a missing output folder and prints it, so it never raises
    before the release cell.
  - Direction: stricter only.
- **FR17 run note (2026-10-22, after the first full run, before any analysis).** P2 completed. P1 crashed at its
  first generation-phase draft forward, inside the probe: `getattr(r, "context_chunk_size", -1)` falls back only on
  AttributeError, and the bindings raise RuntimeError there. The probe now records -1 for any field the request's
  phase does not provide. Neither field enters Q1–Q3. P1 is rerun as is; P2 is kept (`run_all` skips a finished arm).
  Nothing was analysed before the fix.

### FR18 (2026-10-22, after FR17's result, before any FR18 run): standalone reproduction for the upstream report

- **Trigger.** Report A in `drafts/trtllm_REPORTS.md` cannot be filed without a script someone else can run. The
  harness version needs this repo, the fixture and the fp32 hook, and it has only run fp32.
- **Procedure** (`drafts/trtllm_repro/repro_draft_target.py`, run once on a Colab L4 through
  `drafts/trtllm_repro/build_notebook.py`).
  - The script needs only tensorrt-llm 1.2.1. It writes its own two random tiny Llama models (V = 8, head_dim 32,
    seeds 1 and 2) and runs in bf16 with stock kernels.
  - The one patch is Report B's workaround: `process_decoded_tokens` runs under inference mode. There is no fp32
    hook.
  - Three arms: `base` (no speculation), `spec` (`DraftTargetDecodingConfig(max_draft_len=3)`) and `spec_keeplast`
    (`exclude_last_generation_logits=False`).
  - Each speculative arm draws 2·10⁴ sequences at T = 1 and at top-k 3, and `base` draws 4·10⁴, so its halves
    give a 2·10⁴-vs-2·10⁴ noise floor. Every arm and setting gets its own TorchSampler seed.
- **Reading.**
  - Primary, from the engine's own tensors: R < K in a speculative arm's verifications, and, at top-k 3, accepted
    draft tokens whose own `py_target_probs` entry is 0. Also drafts accepted past the last draft-logits row
    (min(n, K) > R), which were accepted with no check. Any of these reproduces Report A's mechanism outside the
    harness.
  - Secondary: a chi-square test of each speculative arm against `base`, with sparse outcomes pooled until every
    expected count is at least 5, and with `base` split in half as the noise floor. Reported with its TV, at level 0.005.
  - If bf16 does not reproduce, the report says so. It stays at fp32 with the hook, and the claim is not widened.
- **Not affected.** The §6 verdict and every flag. These are new models and a new dtype, so the numbers are not
  comparable with FR16's and are not pooled with them.
- **Compute-unit check.** GPU needed: TensorRT-LLM has no CPU path. One runtime runs all three arms, at
  1.6·10⁵ draws in total. L4 is the card FR16 used, which keeps the environment the same. Estimate about 0.4 L4-h
  (mostly the install), about 0.7 CU.
- **Direction.** None.
- **FR18 run note (2026-10-22, after the first run, before any analysis).** All three arms of the first run died at the
  engine's first forward. FlashInfer JIT-builds its bf16 RMSNorm with `ninja`, and the venv's `bin/` (where pip
  put `ninja`) was not on PATH. No draws were made. The script now puts its interpreter's directory on PATH, and
  the install cell checks that `ninja` is there. The rerun is otherwise unchanged.
- **FR18 run note 2 (same day, before any analysis).** The second run died at the same point, with FlashInfer's
  RMSNorm rejecting fp32 input. With an fp32 checkpoint, `LLM(dtype="bfloat16")` still ran the activations in fp32.
  No draws were made. The script now writes both checkpoints in bf16 with `torch_dtype` bfloat16. Same models,
  same seeds, rounded to bf16.

### FR19 (2026-10-22, after filing NVIDIA/TensorRT-LLM#20852, before any FR19 run): GPU check of a candidate fix

- **Trigger.** The owner asked for an upstream fix PR for #20852. The fix
  (`drafts/trtllm_fix/fix.patch`, against `release/1.2`, whose patched files are byte-identical to v1.2.1) was drafted
  and revised […] after one review […].
  It has only been compiled; no test has run.
- **Procedure** (`drafts/trtllm_fix/build_notebook.py`, run once on a Colab L4). Install stock 1.2.1 as in FR18. Run the
  patched `test_torch_rejection_sampling.py` against the stock package, apply the patch to the installed package, and
  run the tests again. Then run FR18's `repro_draft_target.py` unchanged except `REPRO_NO_WORKAROUND=1`, so Report B's
  inference-mode workaround is off.
- **Reading.** The fix passes only if all of these hold:
  - the patched tests pass, and at least one new test fails on stock 1.2.1;
  - both speculative arms run to completion without the workaround;
  - R = K in every verification (`rows_fewer_than_K` = 0) and `accepted_beyond_R` = 0;
  - no accepted draft token has target probability 0;
  - at T = 1 and at top-k 3, the chi-square test of each speculative arm against `base` has p ≥ 0.005, with TV
    comparable to the base half-vs-half floor.
  Anything else means the fix is not ready, and no PR is opened.
- **Not affected.** The §6 verdict, every flag and FR18's numbers. This tests a patch, not the study's question.
- **Compute-unit check.** GPU needed, since TensorRT-LLM has no CPU path, and the same card as FR18. Estimate about
  0.5 L4-h, about 0.9 CU.
- **Direction.** None.
- **FR19 result (2026-10-22).** The first run died before the patch: the cell that locates `site-packages` also
  captured the `tensorrt_llm` import banner. It was fixed in the notebook, and the second run completed. On the L4,
  with stock TensorRT-LLM 1.2.1:
  - Stock 1.2.1: the new tests gave 5 failed, 1 passed. With the patch applied, all 6 passed.
  - All three arms exited 0 with the inference-mode workaround off.
  - Every speculative verification had R = K: 59,935, 59,918, 59,908 and 59,936 of them, all K3/R3.
  - Drafts accepted with no draft-logits row: 0. Accepted draft tokens with target probability 0: 0.
  - Against `base`, TV and chi-square p were, at T = 1: spec 0.0096, p = 0.23; spec_keeplast 0.0101, p = 0.38. At
    top-k 3: spec 0.0090, p = 0.76; spec_keeplast 0.0083, p = 0.77. The base half-vs-half floor was 0.011
    (p = 0.69) at T = 1 and 0.015 (p = 0.015) at top-k 3.
  - FR18 on stock 1.2.1 had top-k 3 TV 0.9999, with 19,997 accepted zero-probability tokens.
  Every FR19 criterion holds. Data: `drafts/trtllm_fix/out/`. Cost about 1.1 L4-h over the two runtimes (the first
  ran about 20 min before it died), roughly 2 CU.

### FR20 (2026-10-23, after FR19's result, before any FR20 run): the #20852 reproduction on TensorRT-LLM v1.4.0rc1

- **Trigger.** The fix (FR19, NVIDIA/TensorRT-LLM#20869) targets `release/1.2`, which has been dormant since April.
  In 1.3/1.4, `DraftTargetDecodingConfig` runs a different one-model path: `DraftTargetOneModelWorker`. No one has
  tested whether the distribution defect survives in the current release, v1.4.0rc1, published 2026-10-22.
- **Source reading** (`drafts/trtllm_v14/NOTES_V14.md`, […] spot-checked). With sampling,
  v1.4.0rc1 has two modes:
  - The default, `use_rejection_sampling=False`, drafts by argmax and accepts the longest prefix that equals the
    target's own sampled tokens. That is exact in principle.
  - The opt-in, `use_rejection_sampling=True`, drafts by sampling and verifies with `chain_speculative_sampling`.
  None of the four 1.2.1 code defects exists in this path, so a pass is the expected outcome.
- **Procedure** (`drafts/trtllm_v14/repro_v14.py`, run once on a Colab L4 through `drafts/trtllm_v14/build_notebook.py`).
  - Stock v1.4.0rc1, with torch 2.14.0+cu130 from its pin. Nothing is patched.
  - The same models, prompt, N and settings as FR18. `disable_flashinfer_sampling` was removed upstream and is
    dropped.
  - Three arms: `base` (2N draws), `spec` (default mode) and `spec_rs` (`use_rejection_sampling=True`).
  - A soft recorder on `SpecWorkerBase._accept_draft_tokens` logs K, R (−1 when no draft probabilities exist), the
    number accepted, and accepted tokens with target probability 0.
- **Reading.**
  - An arm reproduces the defect class if either:
    - its chi-square test against `base` gives p < 0.005, with TV clearly above the base half-vs-half floor; or
    - the recorder finds any accepted draft token with target probability 0.
  - Otherwise the wording is "no deviation detected at this N on this fixture", for that arm and mode.
  - If an arm fails to run (install, driver, VANILLA attention), it is recorded as `BLOCKED` with the error, and
    nothing is inferred from it.
  - The recorder is diagnostic only. If it fails soft, only the distribution test counts.
- **Not affected.** The §6 verdict, every flag, and FR18/FR19's numbers. The engine version is new and is not pooled.
- **Compute-unit check.** GPU needed. Estimate about 0.8 L4-h (install ~20 min, three arms ~25 min), about 1.5 CU,
  […]. One rerun is allowed for a notebook-side bug, as in FR18 and FR19.
- **Direction.** None. A pass narrows the claim to 1.2.x, and a failure is a new upstream report.
- **FR20 run note 1 (2026-10-23, after run 1, before any analysis).** In run 1, the `base` arm completed, and both
  speculative arms died in engine warmup with
  `AttributeError: 'VanillaAttentionMetadata' object has no attribute 'host_request_types'` at
  `_torch/speculative/draft_target.py:267`. The one-model draft-target path does not run with VANILLA attention, so
  under VANILLA, `spec` and `spec_rs` are `BLOCKED`. Run 1's statistics are not read.
- **Run 2.** All three arms run on the default TRTLLM attention backend (`REPRO_ATTN=TRTLLM`), so `base` and the
  speculative arms share the same kernels. Everything else in FR20 is unchanged. Output goes to
  `d40_full/v14_repro_trtllm`. This is the one rerun FR20 allows. About 1.5 CU more.
- **FR20 result (2026-10-23).** Run 2 completed on TRTLLM attention: all three arms exited 0, with v1.4.0rc1 and torch
  2.14.0+cu130.
  - **Distribution test as preregistered.** Every speculative comparison against `base` rejects:

    | setting | arm | TV vs base | χ² p-value | base half-vs-half TV |
    |---|---|---|---|---|
    | T = 1 | spec | 0.047 | 2e-32 | 0.011 |
    | T = 1 | spec_rs | 0.045 | 2e-28 | 0.011 |
    | top-k 3 | spec | 0.052 | 2e-41 | 0.011 |
    | top-k 3 | spec_rs | 0.051 | 7e-39 | 0.011 |

  - **Recorder.** R = K in every `spec_rs` verification. No accepted draft token had target probability 0.
  - **Checks after the run** (`drafts/trtllm_v14/analysis/`, exploratory, not preregistered):
    - There is no cross-batch RNG correlation, and lengths and first tokens are identical across arms.
    - Against the exact fp64 distribution of the bf16 weights (`exact_ref.py`), the arms are at TV 0.037 (`base`),
      0.062 (`spec`) and 0.057 (`spec_rs`) at T = 1. The sampling floor is about 0.009–0.013.
    - Emulating bf16 activations with different rounding points (`bf16_spread.py`) moves this fixture by TV up to
      0.068 between variants, and about 0.1 from fp64.
    - So a difference of about 0.05 between a single-token decode path (`base`) and a multi-token verification
      forward (`spec`, `spec_rs`) fits ordinary bf16 numerics. Both modes deviating by the same amount also points
      that way.
  - **Reading.** The preregistered rule fires, but it did not account for the base and speculative paths computing
    the target logits differently in bf16. This flagged doubt is not disposed. **No defect is claimed for
    v1.4.0rc1.** The one firm finding is that the one-model draft-target path crashes under VANILLA attention (run 1).
    The question needs a test that does not depend on numerics. That is FR21.
  - Cost about 1.6 L4-h over three runtimes: a token time-out, run 1 and run 2. Roughly 3 CU.

### FR21 (2026-10-23, after FR20's result, before any FR21 run): calibration test of v1.4.0rc1's one-model path

- **Trigger.** FR20's distribution test rejects for v1.4.0rc1, but bf16 numerics can explain it: the base and
  speculative paths compute the target logits with different forward shapes. A test is needed that does not depend
  on numerics.
- **Principle.** In both one-model modes, a position j that gets emitted carries a token drawn from the engine's own
  target distribution at that prefix. In strict acceptance it is the target's own sample. In rejection sampling, the
  accepted draft token and the residual sample together give exactly p. So, conditional on emission, the emitted
  tokens must follow the probability rows the engine itself computed, however far those rows are from fp64.
- **Procedure** (`drafts/trtllm_v14/repro_v14.py` with the calibration recorder; notebook
  `d40_trtllm_v14_calib.ipynb`; run once on an L4).
  - Stock v1.4.0rc1, TRTLLM attention, the FR20 models and settings.
  - Arms `spec` and `spec_rs` only (`REPRO_ARMS`), at N = 2·10⁴ each.
  - Per verification, the recorder stores the emitted tokens and the engine's target-probability row at each
    emitted position. The rows are recomputed with the worker's own `_compute_probs_from_logits` on the same logits.
- **Primary analysis** (`drafts/trtllm_v14/analysis/calibration.py`, written and tested on synthetic data before the
  run).
  - The test runs per arm × setting × position.
  - Verifications are grouped by their probability row, rounded to 1e-3.
  - Each group with at least 50 draws gets a Pearson goodness of fit against its row, and the group statistics are
    summed.
  - Holm correction across all tests. A test rejects at adjusted p < 0.005.
  - On synthetic data, the calibrated case passes; a 10% sharpening fails (p ≈ 1e-9); 3% wrong-row substitutions
    fail.
- **Reading.**
  - **Calibrated** (no rejection): FR20's deviation is attributed to numerics, and no sampling defect is claimed for
    v1.4.0rc1 at this N.
  - **Rejection:** a candidate defect. Before any claim, check the recorder's alignment (logits row ↔ position).
    If position 0 fails in both modes, suspect the recorder first. Only after that check is a new upstream report
    drafted.
  - **No data:** missing rows, or a recorder that fails soft. Recorded as `BLOCKED`, with nothing inferred.
  - Not addressed: whether the engine's target logits are correct up to numerics (wrong positions or KV would give
    calibrated sampling from wrong rows). FR20's checks after the run put the arms within the bf16 spread, which is
    weak evidence that they are.
- **Not affected.** The §6 verdict and every flag.
- **Compute-unit check.** About 0.6 L4-h (install ~20 min, two arms ~10 min), about 1 CU. […]
- **Direction.** None.
- **FR21 result (2026-10-23).** The first runtime timed out on the token prompt before any work. The second
  completed: both arms exited 0, with about 58,000 verifications per arm and setting. All had target-probability
  rows; none was missing.
  - **Calibration** (`analysis/calibration.py`, `out_fr21/CALIBRATION.json`): calibrated. All 12 tests (2 arms × 2
    settings × positions 0–2) pass. The smallest Holm-adjusted p is 0.85, and the smallest raw p is 0.071 (`spec_rs`,
    T = 1, position 1).
  - **Reading, as preregistered.** In both one-model modes of v1.4.0rc1, every emitted token follows the engine's
    own target probabilities. FR20's deviation from `base` is attributed to numerics: the speculative path's
    verification forward and the decode path give different bf16 target probabilities. No sampling defect is
    claimed for v1.4.0rc1 at this N on this fixture.
  - **Scope.** Not shown: that v1.4.0rc1's target logits are right up to numerics (see the FR21 entry). Separately,
    the one-model draft-target path crashes under VANILLA attention (FR20 run 1).
  - **So the #20852 distribution defect is confined to the 1.2.x two-model path**, among the versions tested.
  - Cost about 0.7 L4-h, roughly 1.3 CU. FR20 and FR21 together came to about 4.3 CU.

### FR22 (2026-10-28, after FR21 and the enrichment review, before any extension code or run): supplementary extensions E1–E5

- **Trigger.** The owner asked for ways to raise the study's value before the write-up. Two advisers were asked
  […]. The owner chose to do all five items (2026-10-28).
- **Status of everything below: supplementary.** The §6 verdict (CONTINUE), m = 118, the 53-cell running set and
  every `BLOCKED` / `REFERENCE_UNRESOLVED` label stay as recorded. No extension result enters §6's families, changes
  a flag, or is reported as part of the frozen grid. Each extension has its own family and level, fixed here.
- **Budget.** At most 23 compute units in total (E2 ≤ 3, E3 ≤ 12, E4 ≤ 4, E5 ≤ 4), […]. The $80 cap applies on top. […]

**E1. Oracle audit (CPU only, retrospective, no new draws).**
- *Null calibration.* For every running grid cell (§4 as fixed by FR13), simulate 2,000 null data sets at the cell's
  N from its own R_eng table, through the cell's own analysis path (`stats.pooled_chi2` for F-A; the F-B prefix
  selection from a simulated null, per-prefix χ², Fisher). Report the empirical rejection rate at α/118, 0.01 and
  0.05, with Clopper–Pearson 99% intervals, and a KS distance of the p-values from uniform. A cell whose rate at 0.05
  lies above its 99% interval is reported as **miscalibrated**, with the direction (anticonservative means the frozen
  test was easier to fail, conservative means it was easier to pass).
- *Detection limits.* An exact model of chain speculative sampling on the F-A fixture (fp64 R_np target rows, the
  fixture draft rows, k = 3, 4 new tokens) gives the exact sequence distribution of a correct sampler and of eight
  defect classes. The output distribution of each is computed exactly by enumeration, never sampled:
  - D1 last draft row missing: the last draft token is accepted without a check when the earlier ones pass;
  - D2 rows shifted by one: draft token j is checked against the draft row of position j + 1, and the last token is
    unchecked;
  - D3 = D2 with zero-over-zero acceptance (a token with p = 0 and used q = 0 is accepted);
  - D4 residual drawn from p instead of norm((p − q)₊);
  - D5 bonus token drawn from q instead of p;
  - D6 target processors skipped at drafted positions (raw softmax p in the acceptance test; vllm#42744 class);
  - D7 greedy draft whose q is used as if sampled (pilot L3);
  - D8 stale context: the target row at draft position j ≥ 1 is computed one accepted token short.
  Each class fires per speculative round with probability λ ∈ {1, 0.3, 0.1, 0.03, 0.01}, the round otherwise being
  correct. For S0 and S2, report the exact TV to the correct distribution and N₉₀, the draws needed for 90% power
  of the frozen F-A test at α/118 (bisection on simulated power, 1,000 simulations per point). The correct model
  must reproduce R_np's sequence distribution to 1e-12; otherwise E1's detection part is `BLOCKED`.
- *Support bound.* For each cell with zero draws on R_eng-zero outcomes, the exact one-sided 95% upper bound on
  that event's probability per draw (3/N rule replaced by the exact binomial bound).
- *FR21 code fix.* `drafts/trtllm_v14/analysis/calibration.py` drops pooled buckets of zero expected count without
  rejecting observed draws in them (found by a review […]). The fix rejects such a test outright, as
  `stats.pooled_chi2` already does. The FR21 data are re-analysed with the fix; its reading changes only if a
  test then rejects.
- *Output.* `full/e1/` (code, `E1_RESULT.md`, JSON). Direction: none on the verdict; it fixes what the paper may
  claim from non-rejections.

**E2. Trace oracle and mutation suite.**
- *Trace oracle* (`full/e2/trace_oracle.py`). Per verification it takes the draft tokens, the draft rows used, the
  target rows used, the accepted count and the emitted tokens, and checks: I1 one draft row per draft token;
  I2 every accepted token has p > 0; I3 a residual token lies in the support of (p − q)₊; I4 the bonus token has
  p > 0 in the row after the last accepted token; I5 (where a reference exists) each row used is the row at its own
  prefix, to within the engine's δ_num. Plus the FR21 calibration test (with E1's fix) over all verifications.
- *Mutations, frozen here before any is implemented.* Runtime monkeypatches only, no committed engine change:
  M1 = D1, M2 = D2, M3 = D3, M4 = D4, M5 = D5, M6 = D8, each at λ = 1, injected into (a) HF transformers 5.18.0
  assisted generation on CPU (H1's path […]) and (b) vLLM 0.30.0 `draft_model` with probabilistic draft
  (VG2's path, L4). A mutation whose engine has no matching code path is `NOT_APPLICABLE` for that engine, with the
  reason. Each mutation needs an effect assertion (§5.3 rule 1) or it is `INVALID`.
- *Detectors.* (i) the frozen F-A sequence χ² at N = 5·10⁴ at α/118; (ii) greedy G, 1,000 runs; (iii) the trace
  oracle on the first 1,000 verifications; (iv) the calibration test. Settings S0 and S2.
- *Endpoint.* A detector × mutation × engine matrix of detected / not detected, next to E1's exact N₉₀ for the same
  class. No pass/fail rule on the study; misses are reported as limits.
- *Cost.* vLLM: 12 mutation cells + 2 clean cells at 5·10⁴, about 1.5 L4-h, ≤ 3 CU.

**E3. Drafter heads and one tree path, with fitted heads.**
- *Departure from §3, stated.* §3 fixed random-init heads. Phase 0b put every random head's acceptance at 0.016–0.062,
  far below [0.3, 0.9], so VG4, SG1 and SG2 were `BLOCKED` (FR13). E3 replaces random init with heads **fitted on
  CPU** to the fixture target by distillation (KL to the target's next-token distribution over the fixture's
  enumerated prefixes, unrolled over the k = 3 draft steps the engines run). Only weights change; checkpoint formats
  are `heads.py`'s; no engine change. The fitted heads are hashed and committed before any GPU run.
- *Feasibility gate (≤ 2 CU, non-evaluative, Phase-0 rules: counters only, nothing compared with a reference).*
  S0 acceptance from the spec counters, 100 sequences per arm, for VG4 (vLLM EAGLE-3), SG1 (SGLang EAGLE chain), SG2
  (SGLang EAGLE tree, topk > 1) and TG2 (TensorRT-LLM EAGLE-3, with one retry of the slot-manager failure by
  configuration only). An arm runs only if its acceptance lies in [0.3, 0.9]. SG2 additionally needs tree evidence:
  the engine's accept-length counters must show accepted tokens on a non-first branch, or SGLang's own trace of the
  selected path; without it SG2 is reported as "tree not exercised".
- *Cells.* For each arm that passes: F-A S0–S3 and F-B S0/S2 where §4 lists them, at N_A = 2·10⁵, with the Stage A
  gate rebuilt on the drawing GPU, the engine's Stage A nulls (already passed) reused, and greedy G. The frozen
  statistic and seed rule (`h("E3", key) + j`). Family: Holm at 0.01 over the E3 cells run. A rejected cell is
  redrawn at 10⁶ with disjoint seeds (Bonferroni 0.01 per family) before it is called confirmed, and a confirmed
  cell goes through the §5.5 novelty search and a seeded or coin-injected reproduction before any upstream report.
- *Real-model bridge (E3b), only if E3 runs at least one arm and ≥ 4 CU remain in E3's allotment.* A small
  pretrained target with a published EAGLE-3 head that the engine loads, at top-k 3, 4 new tokens (at most 81
  outcomes), against the engine's own non-speculative per-prefix rows. Its own revision is written before it runs.
- *Prediction, recorded.* Chain EAGLE paths pass; the tree path is the likeliest place for a failure.

**E4. Ablation of the #20852 fix (TensorRT-LLM 1.2.1, FR18 fixture, L4).**
- *Factors.* F1 keep the last draft row (`exclude_last_generation_logits=False` plus the logits storage sizing);
  F2 collect one row per draft token at its own position (the accumulator); F3 reject a token with target
  probability 0 and check rows = tokens. F4 (out-of-place squeeze) is applied in every arm: without it the executor
  dies (FR18), which is a liveness defect reported separately.
- *Arms.* All 8 subsets of {F1, F2, F3}, implemented as environment flags in one patch of 1.2.1, at T = 1 and top-k 3,
  2·10⁴ draws each, plus `base` at 4·10⁴, as FR18/FR19.
- *Endpoints, per arm and setting.* TV and χ² p against `base` (level 0.005, `base` halves as the noise floor),
  verifications with R < K, accepted tokens with target probability 0, misaligned rows (the FR17 trace check), and
  the calibration test of E2.
- *Prediction.* Only {F1, F2, F3} passes at top-k 3; {F1, F2} passes at T = 1 (full support makes F3 inert);
  F1 alone fails at both (FR16's A3).
- *Cost.* 17 arms × 2 settings at about 2 min each plus install, ≤ 2 L4-h, ≤ 4 CU.

**E5. Target-row drift and deeper positions on v1.4.0rc1 (one-model path, TRTLLM attention, L4).**
- *Drift.* For 2,000 verifications per setting, record the speculative path's target rows at each emitted position,
  then compute the non-speculative target rows at the same prefixes through the engine's own decode path (full-vocab
  logprobs). Report per-prefix TV and max-abs logit difference, against the bf16 spread of FR20 (0.068). A drift
  larger than the bf16 spread at any prefix, reproduced at fp32 if the path accepts fp32, is a candidate defect.
- *Deeper positions.* FR21 had 68–93 draws at position 2. E5 reruns the FR21 calibration with a draft made close to
  the target (the target's weights plus noise, σ chosen on CPU for acceptance ≥ 0.7), N = 2·10⁴ per arm and
  setting, so that positions 1 and 2 each get ≥ 5,000 emissions. Same test, with E1's fix, Holm at 0.005.
- *Cost.* ≤ 1.5 L4-h, ≤ 4 CU.

- **Not affected.** The §6 verdict, every flag, every Stage A/B and FR16–FR21 number.
- **Direction.** None on the verdict. E1 and E2 can only narrow what non-rejections are allowed to mean; E3–E5 add
  supplementary cells with their own families.

- **FR22 run note 1 (2026-10-28, after fitting the heads on CPU, before any E3 GPU run).** `full/e3/fit_heads.py`
  fitted one head per fixture (F-A 150 steps, F-B 300 steps, KL distillation on S0 target rows) and saved two
  variants of each, as FR12 did with `tgt`/`flat`: `fit` (as trained) and `fitflat` (lm_head × 0.4). CPU
  estimates of the per-token acceptance (not the engines' counters) are in `full/e3/fit_{FA,FB}.json`.
  The gate keeps, per arm, the variant whose engine acceptance lies in [0.3, 0.9]; if both do, `fitflat`
  (further from the band's upper edge); if neither, the arm stays `BLOCKED`. Hashes (sha256, first 16 hex):
  - 910082a4ff69c79f  data/models/eagle3_FA_fit/model.safetensors
  - 33b1b0ed44d49855  data/models/eagle3_FA_fitflat/model.safetensors
  - fee8de83971e9ff4  data/models/eagle3_FB_fit/model.safetensors
  - 8f518c6033c3e292  data/models/eagle3_FB_fitflat/model.safetensors

- **FR22 run note 2 (2026-10-28, before any E5 GPU run).** The E5 close draft is the v1.4 fixture's target plus
  noise σ · std(w) · N(0, 1) per weight tensor (numpy seed 7, bf16), `full/e5/close_draft.py`. On the fp64 forward,
  per-token acceptance along target-sampled sequences is 0.886 at σ = 0.02 and 0.750 at σ = 0.05; σ = 0.02 is used
  (the largest grid value with acceptance ≥ 0.8). The drift part samples 1,000 sequences one request at a time with
  `use_rejection_sampling=True` at T = 1, so each verification's committed prefix is known and checked against the
  output; the decode rows come from the non-speculative engine's full-vocabulary logprobs at those prefixes.

- **FR22 run note 3 (2026-10-28, after the vLLM gate, before any E3 draw).** vLLM VG4 S0 acceptance from the counters
  (100 sequences each): F-A `fit` 0.273, `fitflat` 0.294; F-B `fit` 0.247, `fitflat` 0.240. All are below the
  [0.3, 0.9] band, so **VG4 stays `BLOCKED`**; the band is not moved. The two variants give the same acceptance
  because VG4 drafts greedily (vLLM's default `draft_sample_method`), so acceptance is bounded by the target's
  E[max p] whatever the head's calibration; the fitted heads raised it from 0.016–0.062 (random heads) to this
  bound. One arm is added, gated by the same rule: **VG4p**, the same heads with
  `draft_sample_method="probabilistic"` (the rejection-sampling path with a non-degenerate q). Direction: adds
  supplementary cells only if VG4p passes the gate; no effect on the verdict.

- **FR22 run note 4 (2026-10-28, E2 implementation, before any E2 draw).** Both engines' verification functions are
  replaced by one verifier (`e2/verifier.py`) that runs M0 or one mutation on the engine's own draft tokens, draft rows
  and processed target rows: HF `_speculative_sampling` (`e2/hf_e2.py`, […] 3 shards by mode) and
  vLLM `rejection_sample` (`e2/vllm_e2.py`, `d40_e2_vllm.ipynb`, L4). M0 through the same replacement is each
  engine's clean cell. Stated limits, fixed now: **M5 is `NOT_APPLICABLE`** in both engines (neither computes a draft
  row at the bonus position; giving it one is an engine change); **the greedy detector is `NOT_APPLICABLE` for HF**
  (with `do_sample=False` HF matches argmaxes inline and never calls the verifier). The effect assertion counts
  verifications where the mutation changed what the verifier did at a position it reached; 0 makes the cell
  `INVALID`. The trace oracle's I5 compares each target row used with the fp64 row at its prefix (`e1/specmodel.Rows`)
  at δ = 10⁻³; where the engine does not expose the prefix (vLLM `output_token_ids` may be empty), I5 and the
  reference form of I4 are reported as not covered for those verifications. The sequence χ² uses each engine's Stage A
  R_eng table (`hf_FA_*_batch1`, `vllm_FA_NULL_*_V1`) at α/118; calibration uses FR21's test with E1's fix, Holm over
  positions within a cell at 0.005. A CPU smoke (40 draws, M0 and M6) checked the HF path end to end: M0's rows match
  the fp64 rows to 2·10⁻⁷; no detector result from it is used.

- **FR22 run note 5 (2026-10-28, after the VG4p gate, before any E3 draw).** VG4p S0 acceptance (100 sequences
  each): F-A `fit` 0.911, `fitflat` 0.552; F-B `fit` 0.771, `fitflat` 0.405. By run note 1's rule (keep the in-band
  variant; both in band → `fitflat`), **VG4p@fitflat runs on F-A and F-B**; F-A `fit` is out of band (0.911 > 0.9) and
  not run. SGLang's gate has not run yet (no free GPU session); SG1/SG2 stay pending and are reported as not run if
  that persists. One departure, stated: VG4 runs on Model Runner V2, and Stage A drew vLLM's nulls and R_eng on V1
  only, so "the engine's Stage A nulls reused" cannot hold (§5.1 requires the null and R_eng from the sampler that
  draws the cell). The E3 run therefore rebuilds R_eng and the gate on V2 (`reference_v(2, ...)`, Stage A's gate
  rule) and draws a V2 null at every cell's setting at N_A, tested like Stage A's nulls (Holm at 0.01 over the E3
  nulls; a rejected null makes its cell INCONCLUSIVE). Cells: F-A S0–S3, F-B S0/S2, N_A = 2·10⁵, seeds h("E3", key) + j,
  Holm at 0.01 over the E3 cells run; greedy G, 1,000 runs per fixture (`e3/e3_cells.py`, `e3/e3_analyze.py`,
  `d40_e3_cells_vllm.ipynb`).

- **FR22 run note 6 (2026-10-28, after E3 cells run 1 and E4, before the reruns).** (1) E3 cells run 1 failed its
  V2 gate at every setting (F-A δ_num ≈ 1.7–2.2·10⁻⁴ against bounds 1.8–7.2·10⁻⁵; F-B ≈ 1.1–1.3·10⁻³) and drew
  nothing. The cause is mine: `e3_cells.py` did not set `TRITON_F32_DEFAULT=ieee`, which FR13 requires for every
  vLLM/SGLang run (Stage A's V1 gate passed at δ ≈ 1.4·10⁻⁷ with it). That is a harness omission, not an engine
  finding; the rerun sets it (and `vllm_e2.py` now sets it too, before any E2 vLLM draw), writes to a new folder
  (`e3_cells_r2`), and is otherwise unchanged. Run 1's gate output is kept. (2) E4's F0 arm died on a seed bug in
  `e4_arm.py` (a negative shift for the digit 0; fixed […]) before drawing; F0 is rerun alone into the same
  output folder (`d40_e4_f0.ipynb`; `base` is reused, its seed rule unchanged), and the analysis is rerun over all
  arms. F2 and F23 ended in TensorRT-LLM executor errors; those are their recorded outcomes.

- **FR22 run note 7 (2026-10-28, after E2 vLLM run 1, written while the reruns run and before any of their output was
  read).** (1) E2 vLLM run 1 drew nothing: `vllm_e2.py` replaced `rejection_sample` before the engine was built,
  and vLLM's start-up `profile_run` calls it with dummy sampling metadata that is neither all-greedy nor all-random,
  which the replacement rejected by assertion (log in `e2/out_vllm_r1/`). A harness error, not an engine finding.
  The rerun installs the replacement after the engine is built, binds the call's arguments by name through the
  original's signature (checked against the v0.30.0 source), and hands any mixed batch to the original; nothing
  else changes (same seeds, cells, detectors). (2) Colab's Google Drive sign-in could not be completed […], so the E2 vLLM notebook pushes its output to a results branch […] and the SGLang gate
  notebook prints its JSON; the run-once guard and release cells are unchanged. (3) The SGLang gate (SG1, SG2)
  now runs, under run note 5's rule and the same [0.3, 0.9] band. Direction: none on the verdict; E2's matrix and
  E3's arm set are filled in, not changed.

- **FR22 run note 8 (2026-10-28, after the SGLang gate, before any SGLang E3 draw).** SGLang S0 acceptance (correct /
  proposed drafts, 100 sequences each; `e3/out_gate/e3_gate_sglang.json`): F-A SG1 `fit` 0.315, `fitflat` 0.292;
  F-A SG2 `fit` 0.265, `fitflat` 0.229, with the tree exercised in both (accepted paths through 66–73 distinct
  depth-1 nodes); F-B SG1 `fit` 0.252, `fitflat` 0.246. By run note 1's rule **SG1@fit runs on F-A**; F-B SG1 and
  SG2 stay `BLOCKED` (below the band; the band is not moved), so the tree path gets no E3 cell. SG1@fit is drawn as
  Stage A drew SGLang (`stagea.draw_sglang`): one engine per cell with `random_seed = h("E3", key)`, Stage A's chunk
  plan, the gate rebuilt on the drawing GPU by Stage A's rule, one child per setting, at Stage A's SGLang F-A settings
  S0–S3 (S2 and S3 failed Stage A's gate on a FlashInfer scoring error and will likely fail again), N_A = 2·10⁵,
  greedy G with 1,000 runs if S0 passes. The nulls are Stage A's SGLang nulls (same sampler and engine options,
  already passed), as FR22 E3 states. Analysis: each cell against its own rebuilt R_eng table (`fa_test`), Holm at
  0.01 over all E3 cells run (the six vLLM cells and these), a rejected cell redrawn at 10⁶ before it is called
  confirmed. Code `e3/e3_cells_sglang.py`, notebook `d40_e3_cells_sglang.ipynb`, output pushed to
  a results branch […]. Cost about 1 L4-h (~1.7 CU), inside E3's allotment. Direction: supplementary cells only.
