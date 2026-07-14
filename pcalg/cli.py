"""Command-line interface for py_pcalg."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from pcalg.pc import demo_pc_algorithm, skeleton
from pcalg.utils import Matrix, independence_test


def _discover_from_csv(
    csv_path: Path,
    alpha: float,
    output: Path | None,
    no_plot: bool,
) -> int:
    data = pd.read_csv(csv_path)
    corr = data.corr()
    labels = list(range(data.shape[1]))
    label_dict = {i: str(col) for i, col in enumerate(data.columns)}

    graph = skeleton(
        suff_stat=[corr.to_numpy(), len(data)],
        indep_test=independence_test,
        alpha=alpha,
        labels=labels,
    )

    edge_count = int(graph.M.sum() // 2)
    print(f"Variables: {data.shape[1]}")
    print(f"Samples:   {len(data)}")
    print(f"Alpha:     {alpha}")
    print(f"Edges:     {edge_count}")
    print("Adjacency matrix:")
    print(graph.M)

    if not no_plot:
        from pcalg.graph import visualize_graph

        save_path = str(output) if output else None
        visualize_graph(graph, label_dict, title=f"PC skeleton ({csv_path.name})", save_path=save_path)
        if save_path:
            print(f"Plot saved: {save_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcalg",
        description="Discover causal skeletons with the PC algorithm (Fisher-Z partial correlation).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Run the bundled demo on test_data.csv")

    run = sub.add_parser("run", help="Run skeleton discovery on a CSV file")
    run.add_argument("csv", type=Path, help="CSV file (numeric columns only)")
    run.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    run.add_argument("-o", "--output", type=Path, help="Save plot to this path (PNG)")
    run.add_argument("--no-plot", action="store_true", help="Skip graph visualization")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        demo_pc_algorithm()
        return 0

    if args.command == "run":
        if not args.csv.exists():
            print(f"File not found: {args.csv}", file=sys.stderr)
            return 1
        return _discover_from_csv(args.csv, args.alpha, args.output, args.no_plot)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
