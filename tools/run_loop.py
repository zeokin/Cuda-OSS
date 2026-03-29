#!/usr/bin/env python3
"""
run_loop.py -- Automated experiment orchestrator for cuda-evolve.

Handles the mechanical parts of the experiment loop so the agent only needs
to provide the hypothesis and code change. This script:

1. Commits the current kernel.py state
2. Runs bench.py and captures structured output
3. Optionally runs ncu_profile.py
4. Applies the keep/revert decision automatically
5. Appends results to results.tsv with full metadata
6. Outputs a compact summary for the agent's context

Usage:
  uv run tools/run_loop.py --hypothesis "increase tile size from 64 to 128"
  uv run tools/run_loop.py --hypothesis "vectorize loads" --quick
  uv run tools/run_loop.py --hypothesis "reduce registers" --ncu
  uv run tools/run_loop.py --hypothesis "try num_warps=8" --parent-id exp_003
  uv run tools/run_loop.py --dry-run   # show what would happen without executing
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "workspace" / "results.tsv"
KERNEL_FILE = ROOT / "kernel.py"
KERNEL_CU_FILE = ROOT / "kernel.cu"


def _get_kernel_type() -> str:
    """Read KERNEL_TYPE from kernel.py without fully importing it."""
    if not KERNEL_FILE.exists():
        return ""
    try:
        spec = importlib.util.spec_from_file_location("_kernel_peek", str(KERNEL_FILE))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "KERNEL_TYPE", "")
    except Exception:
        return ""


def _run(cmd: list[str], capture: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=ROOT,
        timeout=timeout,
        check=False,
    )


def _git_sha() -> str:
    r = _run(["git", "rev-parse", "--short", "HEAD"])
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _git_commit(message: str) -> bool:
    _run(["git", "add", "kernel.py"])
    if KERNEL_CU_FILE.exists():
        _run(["git", "add", "kernel.cu"])
    r = _run(["git", "commit", "-m", message])
    return r.returncode == 0


def _git_revert() -> bool:
    r = _run(["git", "reset", "--hard", "HEAD~1"])
    return r.returncode == 0


def _get_experiment_count() -> int:
    if not RESULTS_FILE.exists():
        return 0
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    return max(0, len(lines) - 1)


def _get_last_experiment_id() -> str:
    if not RESULTS_FILE.exists():
        return ""
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return ""
    return lines[-1].split("\t")[0]


def _parse_bench_output(log: str) -> dict[str, str]:
    """Extract greppable key=value pairs from bench.py output."""
    metrics = {}
    patterns = [
        "correctness", "throughput_tflops", "speedup_vs_pytorch",
        "pct_peak_compute", "pct_peak_bandwidth", "bottleneck",
        "peak_vram_mb", "bench_time_seconds", "kernel_type",
        "latency_us", "latency_ms", "bandwidth_gb_s",
        "gpu_memory_gb",
    ]
    for line in log.split("\n"):
        line = line.strip()
        for pat in patterns:
            if line.startswith(f"{pat}:"):
                val = line.split(":", 1)[1].strip()
                metrics[pat] = val
                break
    return metrics


def _parse_ncu_output(log: str) -> dict[str, str]:
    """Extract greppable key=value pairs from ncu_profile.py output."""
    metrics = {}
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("ncu_"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                metrics[parts[0].strip()] = parts[1].strip()
    return metrics


def _append_result(
    experiment_id: str,
    hypothesis: str,
    bench_metrics: dict[str, str],
    ncu_metrics: dict[str, str],
    kept: bool,
    git_sha: str,
    parent_id: str,
) -> None:
    """Append a row to results.tsv with the extended schema."""
    correctness = bench_metrics.get("correctness", "UNKNOWN")
    latency = bench_metrics.get("latency_ms", "0")
    throughput = bench_metrics.get("throughput_tflops", "0")
    peak_vram = bench_metrics.get("peak_vram_mb", "0")
    pct_compute = bench_metrics.get("pct_peak_compute", "").rstrip("%")
    pct_bw = bench_metrics.get("pct_peak_bandwidth", "").rstrip("%")
    bottleneck = bench_metrics.get("bottleneck", "")

    ncu_stall = ncu_metrics.get("ncu_top_stall", "")
    ncu_occ = ncu_metrics.get("ncu_occupancy", "")
    ncu_l1 = ncu_metrics.get("ncu_l1_hit_rate", "")
    ncu_l2 = ncu_metrics.get("ncu_l2_hit_rate", "")

    kept_str = "yes" if kept else "no"

    cols = [
        experiment_id, hypothesis, correctness, latency, throughput, peak_vram,
        kept_str, pct_compute, pct_bw, bottleneck, git_sha, parent_id,
        ncu_stall, ncu_occ, ncu_l1, ncu_l2,
    ]
    row = "\t".join(cols)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def run_experiment(
    hypothesis: str,
    quick: bool = False,
    run_ncu: bool = False,
    parent_id: str = "",
    dry_run: bool = False,
) -> dict:
    """Run one experiment cycle: commit -> bench -> decide -> record."""
    exp_count = _get_experiment_count() + 1
    kernel_type = _get_kernel_type()
    if kernel_type:
        exp_id = f"{kernel_type}_exp_{exp_count:03d}"
    else:
        exp_id = f"exp_{exp_count:03d}"

    if not parent_id:
        parent_id = _get_last_experiment_id()

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Parent: {parent_id}")
    print(f"{'=' * 60}")

    if not KERNEL_FILE.exists():
        print("ERROR: kernel.py not found")
        return {"status": "error", "reason": "kernel.py not found"}

    if dry_run:
        print("[DRY RUN] Would commit, benchmark, and record")
        print(f"  experiment_id: {exp_id}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  quick: {quick}")
        print(f"  ncu: {run_ncu}")
        return {"status": "dry_run", "experiment_id": exp_id}

    # --- Step 1: Commit ---
    commit_msg = f"experiment: {hypothesis}"
    if not _git_commit(commit_msg):
        print("WARNING: git commit failed (no changes or git error)")

    git_sha = _git_sha()
    print(f"git_sha: {git_sha}")

    # --- Step 2: Benchmark ---
    print("\n--- Running benchmark ---")
    bench_cmd = ["uv", "run", "tools/bench.py"]
    if quick:
        bench_cmd.append("--quick")

    t0 = time.time()
    bench_result = _run(bench_cmd, timeout=600)
    bench_time = time.time() - t0

    bench_log = bench_result.stdout or ""
    if bench_result.stderr:
        bench_log += "\n" + bench_result.stderr

    log_path = ROOT / "run.log"
    log_path.write_text(bench_log, encoding="utf-8")
    print(f"bench_time: {bench_time:.1f}s")

    bench_metrics = _parse_bench_output(bench_log)
    correctness = bench_metrics.get("correctness", "UNKNOWN")
    throughput = bench_metrics.get("throughput_tflops", "0")
    print(f"correctness: {correctness}")
    print(f"throughput_tflops: {throughput}")

    # --- Step 3: NCU (optional) ---
    ncu_metrics: dict[str, str] = {}
    if run_ncu and correctness == "PASS":
        print("\n--- Running NCU profiling ---")
        ncu_result = _run(["uv", "run", "tools/ncu_profile.py"], timeout=600)
        ncu_log = ncu_result.stdout or ""
        if ncu_result.stderr:
            ncu_log += "\n" + ncu_result.stderr

        ncu_log_path = ROOT / "ncu.log"
        ncu_log_path.write_text(ncu_log, encoding="utf-8")
        ncu_metrics = _parse_ncu_output(ncu_log)

    # --- Step 4: Decide ---
    kept = False
    vram_exceeded = False
    try:
        peak_vram = float(bench_metrics.get("peak_vram_mb", "0"))
        gpu_mem_gb = float(bench_metrics.get("gpu_memory_gb", "0"))
        if gpu_mem_gb > 0 and peak_vram > 0:
            gpu_mem_mb = gpu_mem_gb * 1024
            vram_pct = peak_vram / gpu_mem_mb * 100
            if vram_pct > 80:
                vram_exceeded = True
                print(f"\nVRAM: {peak_vram:.0f} MB / {gpu_mem_mb:.0f} MB ({vram_pct:.1f}%) — exceeds 80% limit")
    except ValueError:
        pass

    if correctness == "FAIL":
        print("\nDECISION: REVERT (correctness failed)")
        _git_revert()
    elif vram_exceeded:
        print("DECISION: REVERT (VRAM exceeds 80% of GPU memory)")
        _git_revert()
    elif correctness == "PASS":
        # Compare against parent throughput
        tp_val = 0.0
        try:
            tp_val = float(throughput)
        except ValueError:
            pass

        parent_tp = _get_parent_throughput(parent_id)
        if parent_tp > 0 and tp_val > 0:
            improvement = (tp_val - parent_tp) / parent_tp * 100
            print(f"improvement: {improvement:.2f}% (parent: {parent_tp:.3f}, current: {tp_val:.3f})")
            if improvement > 1.0:
                print("DECISION: KEEP (>1% improvement)")
                kept = True
            else:
                print(f"DECISION: REVERT ({improvement:.2f}% < 1% threshold)")
                _git_revert()
        else:
            print("DECISION: KEEP (baseline experiment — no parent to compare against)")
            kept = True
    else:
        print(f"DECISION: REVERT (unknown correctness: {correctness})")
        _git_revert()

    # --- Step 5: Record ---
    _append_result(exp_id, hypothesis, bench_metrics, ncu_metrics, kept, git_sha, parent_id)
    print(f"\nresult_recorded: {exp_id} -> results.tsv")

    # --- Step 6: Compact summary ---
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"experiment_id: {exp_id}")
    print(f"hypothesis: {hypothesis}")
    print(f"correctness: {correctness}")
    print(f"throughput_tflops: {throughput}")
    print(f"kept: {'yes' if kept else 'no'}")
    print(f"git_sha: {git_sha}")
    if ncu_metrics:
        for k, v in sorted(ncu_metrics.items()):
            if k.startswith("ncu_"):
                print(f"{k}: {v}")
    print("=== END EXPERIMENT SUMMARY ===")

    return {
        "status": "completed",
        "experiment_id": exp_id,
        "correctness": correctness,
        "throughput": throughput,
        "kept": kept,
        "git_sha": git_sha,
        "bench_metrics": bench_metrics,
        "ncu_metrics": ncu_metrics,
    }


def _get_parent_throughput(parent_id: str) -> float:
    """Look up parent experiment's throughput from results.tsv."""
    if not parent_id or not RESULTS_FILE.exists():
        return 0.0

    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return 0.0

    headers = lines[0].split("\t")
    tp_idx = headers.index("throughput") if "throughput" in headers else 4
    id_idx = headers.index("experiment_id") if "experiment_id" in headers else 0

    for line in reversed(lines[1:]):
        cols = line.split("\t")
        if len(cols) > max(id_idx, tp_idx) and cols[id_idx] == parent_id:
            try:
                return float(cols[tp_idx])
            except ValueError:
                return 0.0

    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Automated experiment runner for cuda-evolve"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help="Description of the change being tested",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use --quick mode for bench.py",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Run NCU profiling after benchmark",
    )
    parser.add_argument(
        "--parent-id",
        type=str,
        default="",
        help="Parent experiment ID (default: last experiment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    args = parser.parse_args()

    result = run_experiment(
        hypothesis=args.hypothesis,
        quick=args.quick,
        run_ncu=args.ncu,
        parent_id=args.parent_id,
        dry_run=args.dry_run,
    )

    if result.get("correctness") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
