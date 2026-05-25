# CCO — Cuda-Compute-OSS

<p align="center">
  <img src="docs/assets/cco-readme-banner.png" alt="CCO Cuda-Compute-OSS banner" width="100%">
</p>

<p align="center">
  <a href="https://discord.gg/kEHZ3wJuHM"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Join the CCO Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/zeokin/Cuda-OSS/issues"><img src="https://img.shields.io/badge/Status-Early%20development-yellow" alt="Status: Early development"></a>
</p>

**A protocol-first framework for AI agents to optimize CUDA / Triton kernels on real hardware — and to grow a public knowledge base of what works.**

CCO does not ship a model. It ships *scaffolding*: a 10-step experiment protocol ([`program.md`](program.md)), a benchmark harness with 5-stage correctness verification ([`tools/bench.py`](tools/bench.py)), a Nsight Compute wrapper that emits greppable metrics ([`tools/ncu_profile.py`](tools/ncu_profile.py)), and a curated optimization knowledge base ([`CUDA_OPTIMIZATION.md`](CUDA_OPTIMIZATION.md)) that grows with every accepted experiment.

You bring the agent. Claude, Codex, Cursor, a local model — anything that can read `program.md` and edit one file. The framework supplies the discipline; the agent supplies the reasoning.

---

## Project Status

CCO is in early development. Some pieces are shipped and stable; others are actively being built. This README will only mark something as "ready" when it actually is.

| Area | Status | Notes |
|---|---|---|
| Experiment protocol (`program.md`) | **Ready** | 10-step loop, documented, stable |
| Benchmark harness (`tools/bench.py`) | **Ready** | 5-stage correctness + roofline analysis |
| NCU profiler (`tools/ncu_profile.py`) | **Ready** | 5 skill sets, greppable output |
| Run loop (`tools/run_loop.py`) | **Ready** | Commit, bench, decide, record |
| Knowledge base (`CUDA_OPTIMIZATION.md`) | **Growing** | 5 kernel sections + cross-kernel patterns documented |
| Reference implementations (`references/`) | **Ready** | 5 kernels |
| Kernel configs (`kernel_configs/`) | **Ready** | 5 kernels |
| Baseline kernels (`kernels/`) | **In flight — see [#1](https://github.com/zeokin/Cuda-OSS/issues/1)** | The 5 baselines are not yet shipped; Quick Start below shows the workaround |
| Benchmark data (`BENCHMARKS.md`) | **Awaiting submissions — see [#5](https://github.com/zeokin/Cuda-OSS/issues/5)** | All rows are placeholders until a contributor runs the loop on real hardware |
| CI workflow | **Ready** | Lint + import-check |

For the full roadmap, see [open issues](https://github.com/zeokin/Cuda-OSS/issues).

---

## Design Principles

Every architectural choice in this repo is downstream of these five bets. They are the lens to evaluate any proposed change.

1. **The knowledge base is the artifact.** `CUDA_OPTIMIZATION.md` is the long-term value the project is built to grow. Every other piece of the framework — the harness, the NCU wrapper, the protocol — exists to ensure entries that land in the KB are correct, reproducible, and useful to future runs. Anything that does not compound the KB has to justify its weight.

2. **Real hardware is the only ground truth.** Roofline analysis uses real peak FLOPs and HBM bandwidth for the detected GPU. Correctness is checked against PyTorch reference outputs on actual CUDA. We do not lower the bar to make the demo easier.

3. **Agent-agnostic protocol.** The agent is a replaceable component. `program.md` is written to be readable by any reasonable coding agent, not by one specific model or vendor. Tooling emits greppable `key=value` text, not vendor-specific structured output.

4. **Small surface, legible end-to-end.** The whole framework is around ten Python files. A new contributor can read it in an afternoon. Adding complexity (parallel orchestration, vector stores, model-replacement infrastructure) is a serious decision, not a default.

5. **Reproducible lineage.** Every accepted experiment writes a row to `workspace/results.tsv` with the hypothesis, metrics, decision, `git_sha`, and `parent_experiment_id`. Anyone can replay the chain of changes that produced a given speedup.

---

## How It Works

Defined in detail in [`program.md`](program.md). At a glance, one experiment iteration is:

1. **Benchmark** `kernel.py` for correctness + throughput + bandwidth
2. **Macro analysis** — roofline classifies the kernel as compute-bound or memory-bound
3. **Micro analysis** — NCU reveals stall reasons, occupancy, L1/L2 hit rates
4. **Hypothesize** an optimization grounded in the bottleneck + [`CUDA_OPTIMIZATION.md`](CUDA_OPTIMIZATION.md)
5. **Modify** the kernel (one focused change)
6. **Commit** with the hypothesis as the message
7. **Re-benchmark**
8. **Decide** keep (>1% gain, still correct) or revert
9. **Record** to [`workspace/results.tsv`](workspace/results.tsv) with parent-experiment lineage; update [`memory/<kernel_type>.md`](memory/) and [`CUDA_OPTIMIZATION.md`](CUDA_OPTIMIZATION.md)
10. **Repeat**

The agent does the optimizing. The framework enforces correctness, captures the result, and grows the KB.

---

## Quick Start

```bash
# Install dependencies (uv is the supported package manager)
uv sync

# Validate environment: CUDA, Triton, NCU, GPU detection
uv run tools/prepare.py
```

Once a baseline kernel exists in `kernels/` (see [#1](https://github.com/zeokin/Cuda-OSS/issues/1)):

```bash
# Pick a kernel to optimize
cp kernels/rms_norm.py kernel.py

# Baseline benchmark
uv run tools/bench.py

# Drive iterations manually:
#   "Read program.md and start optimizing kernel.py."
# or run the automated loop:
uv run tools/run_loop.py --hypothesis "increase tile size from 64 to 128" --ncu

# Visualize the run history (timeline + lineage)
uv run tools/visualize.py
```

Until baselines land, you can still use the framework with your own kernel: write a `kernel.py` that exports the [contract](#kernel-contract), make sure the matching `kernel_configs/<name>.toml` + `<name>.py` and `references/<name>.py` exist, and run `tools/bench.py` against it.

---

## Bundled Kernels

Each bundled kernel has a reference implementation, a benchmark config, and (when [#1](https://github.com/zeokin/Cuda-OSS/issues/1) lands) a baseline. The five chosen kernels span the bottleneck types worth documenting in the KB:

| Kernel | Description | Bottleneck |
|---|---|---|
| `rms_norm` | Per-row RMS normalization | Memory-bound |
| `qkv_part_rope` | QKV partial rotary positional embedding | Mixed |
| `swiglu_input_quant` | SwiGLU activation + FP8 blockwise quantization | Memory-bound, multi-output |
| `persistent_matmul` | GEMM with persistent CTA pattern | Compute-bound |
| `dsa_forward` | Dynamic sparse attention (GQA-aware) | Mixed |

The choice is deliberate: two memory-bound, one compute-bound, two mixed. That coverage is what lets `CUDA_OPTIMIZATION.md § Cross-Kernel Optimization Patterns` say something general — a pattern that helps a memory-bound kernel on one workload and a compute-bound kernel on another is the kind of pattern worth promoting.

---

## Targeted Hardware

The framework is designed to run on NVIDIA datacenter and workstation GPUs. Roofline numbers, peak FLOPs, and HBM bandwidth are wired in for these SKUs in [`tools/bench.py`](tools/bench.py):

H100 (SXM / PCIe), H800, H200, A100 (SXM / PCIe), L40S, L4, A10, RTX 4090, RTX 4080, RTX 3090, RTX 3080, B200, B100.

If your GPU is not in the list, the framework still runs (it falls back to a clock-rate-derived estimate), but the reported `pct_peak_*` numbers should be taken with appropriate skepticism.

**Important:** "designed for" is not the same as "verified on." `BENCHMARKS.md` is the source of truth for which hardware has actually been measured. Today most rows there are placeholders — see [#5](https://github.com/zeokin/Cuda-OSS/issues/5).

---

## Project Structure

```
Cuda-OSS/
├── program.md              # Agent workflow protocol (the "research org code")
├── CUDA_OPTIMIZATION.md    # Agent-maintained optimization knowledge base
├── BENCHMARKS.md           # Public benchmark results
├── CONTRIBUTING.md         # How to contribute (read first)
├── kernel.py               # The kernel currently being optimized (gitignored content)
├── kernels/                # Read-only baselines (one .py per kernel)
├── kernels_optimized/      # Agent-produced optimized versions
├── kernel_configs/         # <name>.toml + <name>.py — test cases (auto-discovered)
├── references/             # Pure-PyTorch correctness oracles
├── memory/                 # Per-kernel experiment log (memory/<kernel_type>.md)
├── workspace/              # Runtime outputs (results.tsv, MEMORY.md, ncu_reports/)
├── docs/                   # Reference: stall reasons, mem/compute opts, arch notes
├── tools/
│   ├── bench.py            # Benchmark harness + 5-stage correctness pipeline
│   ├── ncu_profile.py      # Nsight Compute wrapper with greppable output
│   ├── run_loop.py         # Automated experiment driver
│   ├── prepare.py          # Environment validation
│   ├── merge_results.py    # Merge results.tsv from multiple agent worktrees
│   └── visualize.py        # Render progress.png and lineage.txt from results.tsv
└── pyproject.toml
```

---

## Kernel Contract

Every kernel module under `kernels/` must export the following five names. See `references/<name>.py` and `kernel_configs/<name>.{toml,py}` for the matching pieces.

```python
KERNEL_TYPE: str                          # identifier matching a kernel_configs/<name>.* entry
def kernel_fn(**inputs) -> torch.Tensor: ...   # or tuple[torch.Tensor, ...] for multi-output
def get_inputs() -> dict: ...                  # one sample input dict for smoke tests
def get_flops() -> int: ...                    # for roofline
def get_bytes() -> int: ...                    # for roofline
```

For CUDA C kernels, the `.py` is a thin wrapper that compiles a sibling `.cu` file via `torch.utils.cpp_extension.load_inline()`.

**`kernel_fn` may not delegate the computation back to PyTorch.** Calling `torch.nn.functional.rms_norm`, `torch.matmul`, etc. inside the kernel body is *forbidden*: it would pass correctness, look like a legitimate optimization win, and poison `CUDA_OPTIMIZATION.md` with reasoning unrelated to anything we actually computed. See [#21](https://github.com/zeokin/Cuda-OSS/issues/21) for the AST-level guard.

---

## How to Contribute

CCO is open source under MIT, and contribution is broader than "write CUDA." Pick the path that fits.

| If you have... | Start here |
|---|---|
| An NVIDIA GPU + want to optimize | [#1](https://github.com/zeokin/Cuda-OSS/issues/1) (restore baselines) or [`good-first-issue`](https://github.com/zeokin/Cuda-OSS/labels/good-first-issue) labelled issues |
| An NVIDIA GPU + want to publish numbers | [#5](https://github.com/zeokin/Cuda-OSS/issues/5) (seed BENCHMARKS.md) or [#17](https://github.com/zeokin/Cuda-OSS/issues/17) (submit rms_norm row) |
| No GPU but want to help | [`type:docs`](https://github.com/zeokin/Cuda-OSS/labels/type%3Adocs) issues, issue triage, KB review, walkthrough writing |
| Tooling skills | [`type:tooling`](https://github.com/zeokin/Cuda-OSS/labels/type%3Atooling) issues — `tools/bench.py`, `tools/ncu_profile.py`, `tools/run_loop.py` are all open for improvement |
| Specific bottleneck expertise | Read [`CUDA_OPTIMIZATION.md`](CUDA_OPTIMIZATION.md), open a PR that adds or refines an entry under "Cross-Kernel Optimization Patterns" |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor's full reference: kernel contract details, KB curation rules, PR conventions, benchmark submission template, the no-torch-delegation rule.

---

## Community

Discord: **[discord.gg/kEHZ3wJuHM](https://discord.gg/kEHZ3wJuHM)** — `#help`, `#kernels`, `#benchmarks`, `#agent-prompts`, `#papers`, `#hardware`.

GitHub issues are the canonical place for design discussion, bug reports, and roadmap conversation. Use the appropriate `type:*` and `P0…P3` labels; see [`.github/`](.github) for templates.

---

## License

MIT — see [LICENSE](LICENSE). By contributing you agree your contribution is licensed under MIT as well.
