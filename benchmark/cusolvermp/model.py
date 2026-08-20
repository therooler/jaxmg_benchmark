"""Describe the benchmark cases before they are submitted.

The benchmark only uses matrix sizes that fit the selected process grid and
tile size without padding. This module also records the memory that JAXMg can
calculate in advance. cuSOLVERMp allocates its own routine-specific workspace,
so that part is intentionally left out of the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, lcm, sqrt
from pathlib import Path
import tomllib


DTYPE_BYTES = {
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}

ROUTINES = ("potrs", "lu_solve", "gesvd", "syevd")

# Peak number of matrix-sized buffers one process holds at any point in a case,
ROUTINE_MATRIX_COPIES = {"potrs": 1, "lu_solve": 1, "gesvd": 3, "syevd": 2}

# Matrix-sized results returned alongside the factorized input.
ROUTINE_VECTOR_OUTPUTS = {"potrs": 0, "lu_solve": 0, "gesvd": 2, "syevd": 1}


@dataclass(frozen=True, order=True)
class ProcessGrid:
    """Two-dimensional arrangement of the GPU processes used by cuSOLVERMp.

    Attributes:
        rows: Number of process rows.
        cols: Number of process columns.
    """

    rows: int
    cols: int

    @classmethod
    def parse(cls, value: str) -> "ProcessGrid":
        """Parse a grid written as ``ROWSxCOLS``.

        Args:
            value: Text such as ``"4x2"``.

        Returns:
            The validated process grid.

        Raises:
            ValueError: If the text does not contain two positive dimensions.
        """
        try:
            rows, cols = (int(part) for part in value.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid process grid {value!r}; expected ROWSxCOLS") from exc
        if rows < 1 or cols < 1:
            raise ValueError("process-grid dimensions must be positive")
        return cls(rows, cols)

    @property
    def processes(self) -> int:
        """Return the total number of participating GPU processes."""
        return self.rows * self.cols

    def __str__(self) -> str:
        """Return the grid in the command-line ``ROWSxCOLS`` form."""
        return f"{self.rows}x{self.cols}"


def factor_grids(processes: int) -> tuple[ProcessGrid, ...]:
    """Return the non-transposed rectangular grids for a process count.

    The long dimension is placed first, matching the benchmark's historical
    ``Nx1``, ``(N/2)x2``, ... convention. Transposed layouts can still be
    requested explicitly with ``sweep.grids``.

    Args:
        processes: Total number of one-GPU Python processes.

    Returns:
        Factor grids ordered from the most elongated to the most square.

    Raises:
        ValueError: If ``processes`` is not positive.
    """
    if processes < 1:
        raise ValueError("process count must be positive")
    return tuple(
        ProcessGrid(processes // cols, cols)
        for cols in range(1, int(sqrt(processes)) + 1)
        if processes % cols == 0
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Hardware, case-selection, execution, and Slurm settings from TOML.

    The configuration is shared by the planner, the per-case runner, and the
    submission command so the requested resource shape matches the benchmark
    assumptions.
    """

    gpu_name: str
    visible_memory_mib: int
    allocator_fraction: float
    gpus_per_node: int
    cpus_per_gpu: int
    tiles: tuple[int, ...]
    dtypes: tuple[str, ...]
    routines: tuple[str, ...]
    baseline_sizes: tuple[int, ...]
    frontier_fractions: tuple[float, ...]
    grids: tuple[ProcessGrid, ...]
    node_counts: tuple[int, ...]
    cold_runs: int
    warm_runs: int
    case_timeout_seconds: int
    suite_walltime: str
    suite_memory: str
    slurm_args: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load and validate a benchmark configuration file.

        Args:
            path: TOML file containing ``hardware``, ``sweep``, ``execution``,
                and ``slurm`` sections.

        Returns:
            A validated immutable configuration.

        Raises:
            KeyError: If a required configuration section or key is absent.
            ValueError: If the loaded values are inconsistent.
        """
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        hardware, sweep = raw["hardware"], raw["sweep"]
        execution, slurm = raw["execution"], raw["slurm"]
        node_counts = tuple(int(value) for value in sweep.get("node_counts", ()))
        explicit_grids = tuple(
            ProcessGrid.parse(value) for value in sweep.get("grids", ())
        )
        if explicit_grids:
            grids = explicit_grids
        else:
            grids = tuple(
                grid
                for nodes in node_counts
                for grid in factor_grids(nodes * int(hardware["gpus_per_node"]))
            )
        config = cls(
            gpu_name=str(hardware["gpu_name"]),
            visible_memory_mib=int(hardware["visible_memory_mib"]),
            allocator_fraction=float(hardware["allocator_fraction"]),
            gpus_per_node=int(hardware["gpus_per_node"]),
            cpus_per_gpu=int(hardware["cpus_per_gpu"]),
            tiles=tuple(int(value) for value in sweep["tiles"]),
            dtypes=tuple(str(value) for value in sweep["dtypes"]),
            routines=tuple(str(value) for value in sweep["routines"]),
            baseline_sizes=tuple(int(value) for value in sweep["baseline_sizes"]),
            frontier_fractions=tuple(float(value) for value in sweep["frontier_fractions"]),
            grids=grids,
            node_counts=node_counts,
            cold_runs=int(execution["cold_runs"]),
            warm_runs=int(execution["warm_runs"]),
            case_timeout_seconds=int(execution["case_timeout_seconds"]),
            suite_walltime=str(slurm["suite_walltime"]),
            suite_memory=str(slurm["suite_memory"]),
            slurm_args=tuple(str(value) for value in slurm.get("submit_args", ())),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject values that cannot describe a supported benchmark suite.

        Raises:
            ValueError: If hardware resources, dtypes, solvers, run counts, or
                process grids are incompatible with the benchmark runner.
        """
        if (
            self.visible_memory_mib < 1
            or self.gpus_per_node < 1
            or self.cpus_per_gpu < 1
        ):
            raise ValueError("hardware memory, GPU count, and CPUs per GPU must be positive")
        if not 0 < self.allocator_fraction <= 1:
            raise ValueError("allocator_fraction must be in (0, 1]")
        if set(self.dtypes) - set(DTYPE_BYTES):
            raise ValueError(f"unsupported dtypes: {set(self.dtypes) - set(DTYPE_BYTES)}")
        if set(self.routines) - set(ROUTINES):
            raise ValueError(f"routines must be drawn from {', '.join(ROUTINES)}")
        if any(tile < 1 for tile in self.tiles):
            raise ValueError("tile sizes must be positive")
        if not self.grids:
            raise ValueError("configure sweep.grids or sweep.node_counts")
        if any(nodes < 1 for nodes in self.node_counts):
            raise ValueError("node counts must be positive")
        if self.cold_runs != 1 or self.warm_runs < 1:
            raise ValueError("the suite requires one cold run and at least one warm run")
        if not self.suite_walltime or not self.suite_memory:
            raise ValueError("Slurm wall time and memory must be configured")
        for grid in self.grids:
            if grid.processes % self.gpus_per_node:
                raise ValueError(f"grid {grid} does not occupy whole nodes")

    @property
    def visible_bytes_per_gpu(self) -> int:
        """Return the GPU memory reported by the scheduler, in bytes."""
        return self.visible_memory_mib * 1024**2

    @property
    def allocator_budget_per_gpu(self) -> int:
        """Return the configured JAX allocator budget per GPU, in bytes."""
        return floor(self.visible_bytes_per_gpu * self.allocator_fraction)


@dataclass(frozen=True)
class MemoryEstimate:
    """Known per-GPU allocations for one benchmark case.

    cuSOLVERMp workspace is excluded because the library chooses that size at
    runtime. The fields describe only allocations that the benchmark can
    calculate before execution.

    Attributes:
        local_matrix_bytes: Local shard of the input matrix.
        rhs_capacity_bytes: Tile-aligned right-hand-side work capacity.
        redistribution_scratch_bytes: Three native redistribution buffers.
        pivot_bytes: LU pivot indices, or zero for Cholesky.
        vector_output_bytes: Local shards of the GESVD singular-vector matrices,
            or zero for the solves.
        allocator_budget_bytes: VMM budget available to the process.
    """

    local_matrix_bytes: int
    rhs_capacity_bytes: int
    redistribution_scratch_bytes: int
    pivot_bytes: int
    vector_output_bytes: int
    allocator_budget_bytes: int

    @property
    def known_total_bytes(self) -> int:
        """Return the sum of all allocations known before the solver runs."""
        return sum(
            (
                self.local_matrix_bytes,
                self.rhs_capacity_bytes,
                self.redistribution_scratch_bytes,
                self.pivot_bytes,
                self.vector_output_bytes,
            )
        )

    @property
    def known_budget_fraction(self) -> float:
        """Return the known allocation as a fraction of the VMM budget."""
        return self.known_total_bytes / self.allocator_budget_bytes


@dataclass(frozen=True)
class BenchmarkCase:
    """One solver, dtype, grid, matrix-size, and tile-size measurement.

    A case is the unit executed in a fresh ``srun`` process group. Its matrix
    dimension is required to be tile-aligned so the input matrix needs no
    JAXMg padding.
    """

    routine: str
    dtype: str
    grid: ProcessGrid
    matrix_size: int
    tile_size: int

    @property
    def case_id(self) -> str:
        """Return the stable filename-safe identifier used for logs and JSON."""
        return (
            f"{self.routine}__{self.dtype}__g{self.grid}__"
            f"n{self.matrix_size}__t{self.tile_size}"
        )

    @property
    def alignment_quantum(self) -> int:
        """Return the smallest valid increment of ``N`` for this tile and grid."""
        return self.tile_size * lcm(self.grid.rows, self.grid.cols)

    @property
    def needs_matrix_padding(self) -> bool:
        """Return whether this case would require input matrix padding."""
        return self.matrix_size % self.alignment_quantum != 0

    def estimate_memory(self, config: BenchmarkConfig) -> MemoryEstimate:
        """Estimate the known per-process memory required by this case.

        Args:
            config: Hardware and allocator settings for the target machine.

        Returns:
            Matrix, RHS, redistribution, pivot, and allocator-budget values.

        Raises:
            ValueError: If the matrix dimension requires padding.
        """
        if self.needs_matrix_padding:
            raise ValueError(f"case {self.case_id} requires matrix padding")
        itemsize = DTYPE_BYTES[self.dtype]
        local_rows = self.matrix_size // self.grid.rows
        local_cols = self.matrix_size // self.grid.cols
        local_matrix_bytes = local_rows * local_cols * itemsize
        # Only the solves take a right-hand side; the decompositions instead
        # return matrix-sized results counted by ROUTINE_VECTOR_OUTPUTS.
        takes_rhs = self.routine in ("potrs", "lu_solve")
        # The native redistribution uses three equally sized tile buffers:
        # receive, send, and saved local data. It is sized for whichever
        # process-grid direction has the longer local slab.
        return MemoryEstimate(
            local_matrix_bytes=local_matrix_bytes,
            rhs_capacity_bytes=(
                local_rows * self.tile_size * itemsize if takes_rhs else 0
            ),
            redistribution_scratch_bytes=(
                3 * self.tile_size * max(local_rows, local_cols) * itemsize
            ),
            pivot_bytes=local_cols * 8 if self.routine == "lu_solve" else 0,
            vector_output_bytes=(
                ROUTINE_VECTOR_OUTPUTS[self.routine] * local_matrix_bytes
            ),
            allocator_budget_bytes=config.allocator_budget_per_gpu,
        )

    def to_dict(self, config: BenchmarkConfig) -> dict[str, object]:
        """Return JSON-ready metadata for this case and its memory estimate.

        Args:
            config: Hardware and allocator settings for the target machine.

        Returns:
            A flat dictionary suitable for a benchmark result record.
        """
        memory = self.estimate_memory(config)
        return {
            "case_id": self.case_id,
            "routine": self.routine,
            "dtype": self.dtype,
            "process_rows": self.grid.rows,
            "process_cols": self.grid.cols,
            "processes": self.grid.processes,
            "matrix_size": self.matrix_size,
            "tile_size": self.tile_size,
            "alignment_quantum": self.alignment_quantum,
            "matrix_padding": False,
            "local_matrix_bytes": memory.local_matrix_bytes,
            "global_matrix_bytes": self.matrix_size**2 * DTYPE_BYTES[self.dtype],
            "local_matrix_budget_fraction": (
                memory.local_matrix_bytes / memory.allocator_budget_bytes
            ),
            "rhs_capacity_bytes": memory.rhs_capacity_bytes,
            "redistribution_scratch_bytes": memory.redistribution_scratch_bytes,
            "pivot_bytes": memory.pivot_bytes,
            "vector_output_bytes": memory.vector_output_bytes,
            "known_total_bytes": memory.known_total_bytes,
            "allocator_budget_bytes": memory.allocator_budget_bytes,
            "known_budget_fraction": memory.known_budget_fraction,
            "solver_workspace_bytes": None,
        }


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round a baseline dimension upwards to a valid no-padding dimension.

    Args:
        value: Requested matrix dimension.
        multiple: Required positive alignment increment.

    Returns:
        The smallest multiple of ``multiple`` not smaller than ``value``.
    """
    return (value + multiple - 1) // multiple * multiple


def _near_limit_sizes(
    config: BenchmarkConfig,
    *,
    routine: str,
    dtype: str,
    grid: ProcessGrid,
    tile_size: int,
) -> set[int]:
    """Return aligned dimensions near the matrix-only allocator limit.

    This is a guide for the sweep, not a promise that a case fits. The native
    cuSOLVERMp workspace is routine-specific and cannot be known in advance.

    Args:
        config: Hardware and allocator settings.
        routine: Solver whose concurrent matrix-sized buffers set the limit.
        dtype: Element dtype used by the matrix.
        grid: Distributed process grid.
        tile_size: cuSOLVERMp tile width.

    Returns:
        Aligned dimensions at the requested fractions of the matrix-only
        allocator limit.

    Raises:
        KeyError: If the routine has no recorded matrix-copy count.
    """
    budget = config.allocator_budget_per_gpu / ROUTINE_MATRIX_COPIES[routine]
    matrix_limit = floor(sqrt(budget * grid.processes / DTYPE_BYTES[dtype]))
    quantum = tile_size * lcm(grid.rows, grid.cols)
    return {
        floor(sqrt(fraction) * matrix_limit) // quantum * quantum
        for fraction in config.frontier_fractions
    }


def planned_sizes(
    config: BenchmarkConfig,
    *,
    routine: str,
    dtype: str,
    grid: ProcessGrid,
    tile_size: int,
) -> tuple[int, ...]:
    """Return small-to-large aligned dimensions for one configuration.

    The shared baseline covers small and medium problems. Near the expected
    memory limit, the configured fractions give a more detailed sweep.

    Args:
        config: Hardware and allocator settings.
        routine: Solver being planned. GESVD holds more matrix-sized buffers
            than the solves, so it reaches a smaller maximum dimension.
        dtype: Element dtype used by the matrix.
        grid: Distributed process grid.
        tile_size: cuSOLVERMp tile width.

    Returns:
        Sorted no-padding matrix dimensions for this configuration.
    """
    quantum = tile_size * lcm(grid.rows, grid.cols)
    near_limit = {
        size
        for size in _near_limit_sizes(
            config, routine=routine, dtype=dtype, grid=grid, tile_size=tile_size
        )
        if size >= quantum
    }
    if not near_limit:
        return ()

    first_near_limit = min(near_limit)
    baseline = {
        aligned
        for size in config.baseline_sizes
        if (aligned := _round_up_to_multiple(size, quantum)) < first_near_limit
    }
    return tuple(sorted(baseline | near_limit))
