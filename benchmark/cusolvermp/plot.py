"""Plot warm-median timings for one routine, dtype, and process grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark.cusolvermp.model import ROUTINES, ProcessGrid


def _arguments() -> argparse.Namespace:
    """Read the completed solver, dtype, and grid to plot.

    Returns:
        Parsed result root, case filters, and optional output PNG path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--routine", choices=ROUTINES, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64", "complex64", "complex128"), required=True)
    parser.add_argument("--grid", type=ProcessGrid.parse, required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    """Plot warm median timing against matrix dimension for every tile size."""
    args = _arguments()
    result_dir = (
        Path(args.output_root) / "cases" / args.routine / args.dtype / str(args.grid)
    )
    records = []
    for path in sorted(result_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "passed":
            records.append(record)
    if not records:
        raise SystemExit(f"no passing records found under {result_dir}")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for tile_size in sorted({record["tile_size"] for record in records}):
        selected = sorted(
            (record for record in records if record["tile_size"] == tile_size),
            key=lambda record: record["matrix_size"],
        )
        axis.plot(
            [record["matrix_size"] for record in selected],
            [record["warm_median_seconds"] for record in selected],
            marker="o",
            label=rf"$T_A={tile_size}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Matrix dimension $N$")
    axis.set_ylabel("Warm median solve time [s]")
    axis.set_title(
        f"JAXMg {args.routine}: {args.dtype}, {args.grid} process grid"
    )
    axis.grid(which="both", alpha=0.25)
    axis.legend()

    output = Path(args.output or (
        Path(args.output_root) / "plots"
        / f"{args.routine}__{args.dtype}__g{args.grid}.png"
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(output)


if __name__ == "__main__":
    main()
