"""Merge results.tsv files from multiple agent worktrees into the main repo.

Usage:
    uv run tools/merge_results.py ../cuda-evolve-matmul ../cuda-evolve-rms-norm

This reads results.tsv from each worktree directory, deduplicates rows by
experiment_id, and writes the merged result to results.tsv in the current
working directory.
"""

import sys
from pathlib import Path

HEADER = (
    "experiment_id\thypothesis\tcorrectness\ttime_ms\tthroughput\tpeak_vram_mb\tkept"
    "\tpct_peak_compute\tpct_peak_bandwidth\tbottleneck\tgit_sha\tparent_experiment_id"
    "\tncu_top_stall\tncu_occupancy\tncu_l1_hit_rate\tncu_l2_hit_rate"
    "\n"
)

EXPECTED_COLS = len(HEADER.strip().split("\t"))


def load_rows(path: Path) -> list[str]:
    if not path.exists():
        print(f"  [skip] {path} does not exist")
        return []
    lines = path.read_text().strip().split("\n")
    if len(lines) <= 1:
        print(f"  [skip] {path} has no data rows")
        return []
    rows = lines[1:]
    print(f"  [load] {path}: {len(rows)} rows")
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run merge_results.py <worktree_dir> [<worktree_dir> ...]")
        print("Merges results.tsv from each worktree into ./results.tsv")
        sys.exit(1)

    worktree_dirs = [Path(d) for d in sys.argv[1:]]
    local_results = Path("workspace/results.tsv")

    all_rows: list[str] = []

    if local_results.exists():
        print("Loading local results.tsv:")
        all_rows.extend(load_rows(local_results))

    for wd in worktree_dirs:
        results_path = wd / "workspace" / "results.tsv"
        print(f"Loading from {wd}:")
        all_rows.extend(load_rows(results_path))

    seen_ids: set[str] = set()
    unique_rows: list[str] = []
    for row in all_rows:
        cols = row.split("\t")
        exp_id = cols[0] if cols else row
        if exp_id not in seen_ids:
            seen_ids.add(exp_id)
            if len(cols) < EXPECTED_COLS:
                cols.extend([""] * (EXPECTED_COLS - len(cols)))
            unique_rows.append("\t".join(cols[:EXPECTED_COLS]))

    local_results.write_text(HEADER + "\n".join(unique_rows) + "\n")
    print(f"\nMerged {len(unique_rows)} unique rows into {local_results}")


if __name__ == "__main__":
    main()
