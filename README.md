# Draft-model speculative decoding can emit zero-probability tokens

Data and code for the paper of the same title (Hanyu Yang).

The study tests whether the speculative-decoding paths of five inference engines sample from the same distribution
as the engine itself with speculation off. Two tiny random Llama fixtures make that distribution exact: F-A
(vocabulary 8, four new tokens, 2,801 outcomes, enumerated) and F-B (vocabulary 32, tested per prefix).

Result: in the preregistered grid, TensorRT-LLM 1.2.1's two-model draft path (K = 3) fails in all six cells taken to
Stage B (N = 10^6 draws each, p = 0 numerically). On F-A its total variation from the reference is 0.47 to 0.58,
and at top-k and top-p it draws sequences that the engine's own processed distribution gives probability zero:
449,762 and 376,375 of 10^6. No other grid cell that ran was rejected.

## Layout

| Path | Contents |
| --- | --- |
| `PREREGISTRATION.md` | The preregistration with its dated amendments (FR1 to FR22). See below for what was removed. |
| `PREREGISTRATION_REDACTIONS.md` | One line per removal in `PREREGISTRATION.md`, with its reason. |
| `code/` | Fixture, fp64 reference sampler, statistics and the HF and llama.cpp arms that the full study imports. |
| `full/` | The full-study harness: per-engine run code (`full/stagea.py`, `full/p0b_*.py`, `full/repro_trtllm.py`), the TensorRT-LLM fp32 hook (`full/d40_trtllm_fp32.py`) and the analysis (`full/stagea_analyze.py`, `full/stagea_check.py`). |
| `full/stagea/data/` | Every Stage A and Stage B draw as gzipped count chunks, the per-engine reference tables and gate records, and the analysis outputs `ANALYSIS_A.*`, `ANALYSIS_B.*`. |
| `full/phase0/`, `full/phase0b/` | Feasibility records, fixture checksums and the detection limits (MDE) used by the analysis. |
| `full/e1/` to `full/e5/` | Supplementary extensions: null calibration and simulated power (E1), verifier mutations (E2), EAGLE-3 chain heads (E3), fix ablation with patches `full/e4/F1.patch` to `full/e4/F3.patch` (E4), TensorRT-LLM 1.4.0rc1 one-model path (E5). |
| `drafts/trtllm_repro/` | Stand-alone bf16 reproduction of the TensorRT-LLM defect, outside the harness. |
| `drafts/trtllm_fix/` | The proposed fix (`drafts/trtllm_fix/fix.patch`), its unit test and the verification runs. |
| `drafts/trtllm_v14/` | Checks on TensorRT-LLM 1.4.0rc1. |
| `data/models/` | The fixture weights (safetensors and GGUF) and the EAGLE-3 heads. |
| `data/refs/` | Reference next-token tables for HF, llama.cpp and vLLM. |

The settings are S0 T = 1; S1 T = 0.7; S2 T = 1, top-k 3; S3 T = 1, top-p 0.8; S4 T = 1, min-p 0.1; S5 T = 1,
repetition penalty 1.3; S6 T = 0.7, top-k 5, top-p 0.9, min-p 0.05; G greedy (`full/p0b_common.py`).

## The preregistration

`PREREGISTRATION.md` is the preregistration as frozen on 2026-10-04, with the amendments appended before the runs
they affect. It was kept in a private working repository. This repository is a single new commit, so it cannot show
those dates; they rest on that private history. The `harness_git` fields in the run records name commits of the
same private repository: Stage A ran at `91c5ee1` and Stage B at `9943e75`. The Stage A and B draw code shipped in
`full/` and `code/` (`full/stagea.py`, `full/p0b_*.py`, `full/d40_trtllm_fp32.py` and the modules they import) is
byte-identical to the Stage B commit; the Phase 0 and extension records name other commits of the same repository. At the Stage A commit two files differed, `full/stagea.py` and
`full/p0b_common.py` (Stage B options added later); their Stage A versions are in `full/at_stageA/`.

Each `[…]` in it marks text removed from the public copy: names of the tools and advisers that reviewed or drafted
parts of it, internal pull-request, commit and branch names, internal tooling, and internal budget-approval rules.
`PREREGISTRATION_REDACTIONS.md` lists them. Nothing else is changed except paths (below). The document refers to
working notes, result write-ups and the pilot study's files, which are not part of this repository; the design
values it takes from the pilot are the settings above and the fixture code in `code/fixture.py`.

The runs were on Colab. In the preregistration, code, logs and run records, the Drive output folder is written as
a local `out` folder, other Colab working paths as `<runtime>/` (standard system paths such as Python's
`site-packages` are left as they were), and the checkout prefix in tracebacks is dropped. The
Colab notebooks and the scripts that built them are not included; each notebook ran the `.py` files shipped here.

## Reproduce

`reproduce.sh` re-runs the Stage A and Stage B statistics on CPU from the saved draws and checks that every verdict,
count and p-value (to a relative 10^-9) matches the committed `full/stagea/data/ANALYSIS_A.json` and
`full/stagea/data/ANALYSIS_B.json`. The one allowed difference is two fields the analysis gained after the committed
Stage A run (`controls_planned`, and `controls_missing`, which must be empty). It needs Python 3.12 with `torch`,
`numpy`, `scipy`, `safetensors` and `transformers` (CPU builds are enough), and takes a few minutes.

```bash
./reproduce.sh
```

It exits non-zero if any result differs. It checks the statistics, not the GPU side: the reference tables
(`gate_tables/`) and the gate verdicts (`gate.json`) are recorded outputs of the GPU runs and are taken as given.
`negative_control.sh` first requires `reproduce.sh` to pass, then moves one draw between two outcomes in a saved
Stage B null chunk and requires `reproduce.sh` to report a mismatch.

Redrawing the samples needs an NVIDIA GPU and the engine versions pinned in `PREREGISTRATION.md`: vLLM 0.30.0,
SGLang 0.5.21, TensorRT-LLM 1.2.1 (with torch 2.9.1 and transformers 4.57.3), llama.cpp at a fixed commit, and
transformers 5.18.0 for the HF arms. Each run's `env.json` records what it used.

## Not claimed

- That any engine is lossless. A cell that was not rejected is a cell where the test found no evidence of a
  difference.
- Power for every cell. Power was simulated only for F-A at S0 and S2, per modelled defect class (`full/e1/`), and
  some classes are not detectable at the sample sizes used: for example skipped logits processors at S0, which
  change nothing there. Defects that fire in a few percent of rounds or fewer are often missed: a bonus token drawn
  from the draft (D5) at S0 has power 0.64 when it fires in 10% of rounds and 0.002 at 3% (`full/e1/out/e1_power.json`).
- That every null cell is calibrated at every level. E1's simulated nulls flag `vllm/FB/VG1/S0` as anticonservative
  at the diagnostic 0.05 level (rejection rate 0.0675); it is not flagged at 0.01 (rate 0.015, 99% interval
  0.009 to 0.024) or at 0.01/118, and its observed p is 0.93 (`full/e1/out/e1_null.json`).
- Anything about real model sizes or vocabularies. The fixtures are tiny by design.
- Anything about the 65 grid cells that did not run, listed with reasons in the paper's appendix, or about tree
  drafting.
- Anything from the supplementary extensions about the grid. The E2 mutation cells reject by design, and in the E4
  ablation the arms F0 and F3 fail and F12 fails its calibration test at T = 1; none of these is a grid cell.

## Licence

- Code: MIT (`LICENSE`).
- Data, fixtures, run records and documents: CC BY 4.0 (`LICENSE-DATA`).
- Files derived from TensorRT-LLM, under Apache-2.0 (`LICENSE-APACHE`, `NOTICE`): the patches `full/e4/*.patch` and
  `drafts/trtllm_fix/fix.patch`, and `drafts/trtllm_fix/test_torch_rejection_sampling.py`, a modified copy of
  TensorRT-LLM's test of the same name.
