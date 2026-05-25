#!/usr/bin/env python3
"""Render workspace/results.tsv into progress and lineage artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.rstrip("%")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_kept(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "keep", "kept", "pass"}


def infer_kernel(experiment_id: str) -> str:
    if "_exp_" in experiment_id:
        return experiment_id.split("_exp_", 1)[0]
    return ""


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, raw in enumerate(reader, start=1):
            exp_id = (raw.get("experiment_id") or "").strip()
            if not exp_id:
                continue
            throughput = parse_float(raw.get("throughput_tflops")) or parse_float(raw.get("throughput"))
            if throughput is None:
                continue
            rows.append(
                {
                    "idx": i,
                    "exp_id": exp_id,
                    "kernel": infer_kernel(exp_id),
                    "throughput": throughput,
                    "kept": parse_kept(raw.get("kept")),
                    "bottleneck": ((raw.get("bottleneck") or "").strip() or "unknown").lower(),
                    "parent": (raw.get("parent_experiment_id") or "").strip(),
                    "pct_compute": parse_float(raw.get("pct_peak_compute")),
                    "pct_bandwidth": parse_float(raw.get("pct_peak_bandwidth")),
                }
            )
    return rows


def kernel_roofline(rows: list[dict]) -> float:
    ceilings: list[float] = []
    for row in rows:
        pct = None
        if row["bottleneck"] == "compute_bound" and row["pct_compute"] and row["pct_compute"] > 0:
            pct = row["pct_compute"]
        elif row["bottleneck"] == "memory_bound" and row["pct_bandwidth"] and row["pct_bandwidth"] > 0:
            pct = row["pct_bandwidth"]
        elif row["pct_compute"] and row["pct_compute"] > 0:
            pct = row["pct_compute"]
        elif row["pct_bandwidth"] and row["pct_bandwidth"] > 0:
            pct = row["pct_bandwidth"]

        if pct:
            ceilings.append(row["throughput"] / (pct / 100.0))

    if ceilings:
        return float(median(ceilings))
    return max(r["throughput"] for r in rows)


def build_lineage(rows: list[dict]) -> str:
    order = {row["exp_id"]: row["idx"] for row in rows}
    by_id = {row["exp_id"]: row for row in rows}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []

    for row in rows:
        exp_id = row["exp_id"]
        parent = row["parent"]
        if parent and parent in by_id:
            children[parent].append(exp_id)
        else:
            roots.append(exp_id)

    for parent, child_ids in children.items():
        children[parent] = sorted(child_ids, key=lambda x: order[x])
    roots = sorted(roots, key=lambda x: order[x])

    lines: list[str] = []

    def label(exp_id: str) -> str:
        row = by_id[exp_id]
        status = "kept" if row["kept"] else "reverted"
        return f"{exp_id} [{status}] tflops={row['throughput']:.4f} bottleneck={row['bottleneck']}"

    def walk(node: str, prefix: str) -> None:
        kids = children.get(node, [])
        for i, child in enumerate(kids):
            is_last = i == len(kids) - 1
            connector = "\\-- " if is_last else "+-- "
            lines.append(prefix + connector + label(child))
            walk(child, prefix + ("    " if is_last else "|   "))

    for root in roots:
        lines.append(label(root))
        walk(root, "")

    return "\n".join(lines) + "\n"


def plot_progress(rows: list[dict], roofline: float, out_path: Path) -> None:
    colors = {"compute_bound": "#1565c0", "memory_bound": "#ef6c00", "unknown": "#6d6d6d"}
    x = list(range(1, len(rows) + 1))
    y = [row["throughput"] for row in rows]

    has_scatter = any(
        row["kept"] and row["pct_compute"] is not None and row["pct_bandwidth"] is not None for row in rows
    )

    if has_scatter:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
        ax2 = None

    ax1.plot(x, y, color="#9e9e9e", linewidth=1.0, alpha=0.5, label="all experiments")

    for i, row in enumerate(rows, start=1):
        color = colors.get(row["bottleneck"], colors["unknown"])
        style = "-" if row["kept"] else ":"
        alpha = 0.95 if row["kept"] else 0.45
        ax1.plot([i], [row["throughput"]], marker="o", linestyle=style, color=color, alpha=alpha)
        if i > 1:
            prev = rows[i - 2]
            segment_style = "-" if row["kept"] else ":"
            ax1.plot(
                [i - 1, i],
                [prev["throughput"], row["throughput"]],
                linestyle=segment_style,
                color=color,
                alpha=alpha,
                linewidth=1.4,
            )

    ax1.axhline(
        roofline,
        linestyle="--",
        color="#1f1f1f",
        linewidth=1.4,
        label=f"derived ceiling {roofline:.3f} TFLOPS",
    )
    ax1.set_xlabel("experiment ordinal")
    ax1.set_ylabel("throughput (TFLOPS)")
    ax1.set_title("throughput timeline by experiment")
    ax1.grid(alpha=0.2, linestyle=":")
    ax1.legend(loc="best", fontsize=9)

    if ax2 is not None:
        accepted = [
            (i, row)
            for i, row in enumerate(rows, start=1)
            if row["kept"] and row["pct_compute"] is not None and row["pct_bandwidth"] is not None
        ]
        xs = [row["pct_compute"] for _, row in accepted]
        ys = [row["pct_bandwidth"] for _, row in accepted]
        cs = [i for i, _ in accepted]
        sc = ax2.scatter(xs, ys, c=cs, cmap="viridis", s=45, alpha=0.9, edgecolors="none")
        ax2.set_xlabel("pct_peak_compute")
        ax2.set_ylabel("pct_peak_bandwidth")
        ax2.set_title("accepted roofline position")
        ax2.grid(alpha=0.2, linestyle=":")
        ax2.set_xlim(0, max(100, max(xs) * 1.1))
        ax2.set_ylim(0, max(100, max(ys) * 1.1))
        cbar = fig.colorbar(sc, ax=ax2)
        cbar.set_label("experiment ordinal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="render workspace/results.tsv into progress and lineage artifacts")
    parser.add_argument("--kernel", type=str, default="", help="filter rows by kernel type prefix in experiment_id")
    parser.add_argument("--out", type=Path, default=Path("workspace/progress.png"), help="output image path")
    parser.add_argument("--lineage-out", type=Path, default=Path("workspace/lineage.txt"), help="lineage text path")
    parser.add_argument("--input", type=Path, default=Path("workspace/results.tsv"), help="results tsv path")
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.kernel:
        rows = [row for row in rows if row["kernel"] == args.kernel]
    else:
        unknown_ids = [row["exp_id"] for row in rows if not row["kernel"]]
        kernels = sorted({row["kernel"] for row in rows if row["kernel"]})
        if unknown_ids and (kernels or len(unknown_ids) > 1):
            sample = ", ".join(unknown_ids[:3])
            if len(unknown_ids) > 3:
                sample += ", ..."
            raise SystemExit(
                f"cannot infer kernel from experiment_id for rows: {sample}. "
                "Use --kernel <name>."
            )
        if len(kernels) > 1:
            raise SystemExit(
                f"multiple kernels detected: {', '.join(kernels)}. "
                "Use --kernel <name>."
            )

    if not rows:
        raise SystemExit("no rows with throughput found for selected filter")

    roofline = kernel_roofline(rows)
    plot_progress(rows, roofline, args.out)
    lineage = build_lineage(rows)
    args.lineage_out.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_out.write_text(lineage, encoding="utf-8")

    kept = sum(1 for row in rows if row["kept"])
    reverted = len(rows) - kept
    kernel_name = args.kernel or (rows[0]["kernel"] if rows[0]["kernel"] else "mixed")
    print(f"input={args.input}")
    print(f"kernel={kernel_name}")
    print(f"rows_selected={len(rows)}")
    print(f"kept_count={kept}")
    print(f"reverted_count={reverted}")
    print(f"roofline_tflops={roofline:.6f}")
    print(f"progress_png={args.out}")
    print(f"lineage_txt={args.lineage_out}")


if __name__ == "__main__":
    main()
