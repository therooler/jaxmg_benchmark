"""Print or submit the Isambard benchmark jobs.

Dry-run is the default. Pass ``--submit`` only after reviewing the printed
commands and case counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from benchmark.cusolvermp.model import (
    BACKENDS,
    DEFAULT_BACKEND,
    NATIVE_TILE_SIZE,
    ROUTINES,
    BenchmarkConfig,
    ProcessGrid,
    grids_for_backend,
    node_shape,
    planned_sizes,
)


def _arguments() -> argparse.Namespace:
    """Read submission filters and the explicit ``--submit`` confirmation.

    Returns:
        Parsed configuration, script, output root, optional case filters, and
        whether Slurm submission was requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--script", default="slurm/suite.sbatch")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--routine", action="append", choices=ROUTINES)
    parser.add_argument("--dtype", action="append", choices=("float32", "float64", "complex64", "complex128"))
    parser.add_argument("--grid", action="append", type=ProcessGrid.parse)
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def _submission_command(
    *,
    config: BenchmarkConfig,
    config_path: Path,
    script_path: Path,
    output_root: Path,
    routine: str,
    dtype: str,
    grid: ProcessGrid,
    backend: str = DEFAULT_BACKEND,
) -> list[str]:
    """Build one outer Slurm job that will run a complete benchmark suite.

    The allocation always reserves whole nodes, including for the single-GPU
    reference: sites commonly enforce a minimum GRES per job (Slurm rejects a
    smaller request with ``QOSMinGRES``), and a node to itself also keeps the
    reference timings free of contention. ``run_suite.py`` then starts a
    separate ``srun`` per case inside it, sized to the case's own grid, so the
    reference still runs on exactly one GPU.

    Args:
        config: Hardware, CPU, memory, and wall-time settings.
        config_path: Absolute benchmark TOML path.
        script_path: Slurm wrapper that activates the runtime environment.
        output_root: Root directory for case JSON files and logs.
        routine: Selected solver.
        dtype: Selected matrix dtype.
        grid: Selected GPU process grid.
        backend: Selected solver implementation.

    Returns:
        Argument vector suitable for ``subprocess.run``.
    """
    nodes, _ = node_shape(grid, config)
    short_dtype = dtype.replace("complex", "c").replace("float", "f")
    tag = "" if backend == DEFAULT_BACKEND else f"_{backend}"
    exported = (
        "ALL,JAXMG_BENCHMARK_CONFIG="
        + str(config_path)
        + ",JAXMG_BENCHMARK_ROUTINE="
        + routine
        + ",JAXMG_BENCHMARK_DTYPE="
        + dtype
        + ",JAXMG_BENCHMARK_GRID="
        + str(grid)
        + ",JAXMG_BENCHMARK_BACKEND="
        + backend
        + ",JAXMG_BENCHMARK_OUTPUT_ROOT="
        + str(output_root)
    )
    return [
        "sbatch",
        "--job-name",
        f"jaxmg{tag}_{routine}_{short_dtype}_{grid}",
        "--nodes",
        str(nodes),
        "--ntasks",
        str(nodes),
        "--ntasks-per-node",
        "1",
        "--cpus-per-task",
        str(config.gpus_per_node * config.cpus_per_gpu),
        "--gres",
        f"gpu:{config.gpus_per_node}",
        "--mem",
        config.suite_memory,
        "--time",
        config.suite_walltime,
        "--output",
        "slurm-logs/%x-%j.out",
        "--error",
        "slurm-logs/%x-%j.err",
        "--export",
        exported,
        *config.slurm_args,
        str(script_path),
    ]


def main() -> None:
    """Print the selected Slurm jobs and optionally submit each one."""
    args = _arguments()
    config_path = Path(args.config).resolve()
    script_path = Path(args.script).resolve()
    config = BenchmarkConfig.load(config_path)
    routines = tuple(args.routine or config.routines)
    dtypes = tuple(args.dtype or config.dtypes)
    backends = tuple(args.backend or config.backends)

    # run_suite.py rejects a routine, dtype, or backend the configuration does
    # not enable.
    if set(routines) - set(config.routines):
        raise ValueError(
            f"routines not enabled by the configuration: "
            f"{sorted(set(routines) - set(config.routines))}"
        )
    if set(dtypes) - set(config.dtypes):
        raise ValueError(
            f"dtypes not enabled by the configuration: "
            f"{sorted(set(dtypes) - set(config.dtypes))}"
        )
    if set(backends) - set(config.backends):
        raise ValueError(
            f"backends not enabled by the configuration: "
            f"{sorted(set(backends) - set(config.backends))}"
        )

    submitted = 0
    total_cases = 0
    suites = 0
    for backend in backends:
        # The single-GPU reference has one grid of its own; an explicit --grid
        # filter only applies to the distributed backend it belongs to.
        available = grids_for_backend(config, backend)
        grids = tuple(grid for grid in (args.grid or available) if grid in available)
        if not grids:
            raise ValueError(
                f"backend {backend} runs on "
                f"{', '.join(str(grid) for grid in available)}, "
                f"but --grid selected {', '.join(str(grid) for grid in args.grid)}"
            )
        # A native suite runs each dimension once, because the tile is not used.
        tiles = (NATIVE_TILE_SIZE,) if backend == "jax" else config.tiles
        for routine in routines:
            for dtype in dtypes:
                for grid in grids:
                    suites += 1
                    case_count = sum(
                        len(
                            planned_sizes(
                                config,
                                routine=routine,
                                dtype=dtype,
                                grid=grid,
                                tile_size=tile_size,
                                backend=backend,
                            )
                        )
                        for tile_size in tiles
                    )
                    total_cases += case_count
                    command = _submission_command(
                        config=config,
                        config_path=config_path,
                        script_path=script_path,
                        output_root=Path(args.output_root).resolve(),
                        routine=routine,
                        dtype=dtype,
                        grid=grid,
                        backend=backend,
                    )
                    print(f"cases={case_count:3d} {' '.join(command)}")
                    if args.submit:
                        Path("slurm-logs").mkdir(parents=True, exist_ok=True)
                        subprocess.run(command, check=True)
                        submitted += 1
    print(
        f"logical_suites={suites} "
        f"fresh_process_cases={total_cases} submitted={submitted}"
    )


if __name__ == "__main__":
    main()
