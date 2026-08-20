"""Run one JAXMg benchmark case.

``run_suite.py`` starts this module once per case with ``srun``. Each Python
process owns one GPU. Input creation and validation are not timed. The first
call includes compilation; the remaining calls use the same compiled solver.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

# XLA reads allocator settings when the GPU client is initialized.  Set these
# defaults before importing JAX; an explicit launcher override remains valid.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "vmm")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark.cusolvermp.model import (
    ROUTINES,
    BenchmarkCase,
    BenchmarkConfig,
    ProcessGrid,
)


def _arguments() -> argparse.Namespace:
    """Read the command-line description of one fresh benchmark case.

    Returns:
        Parsed configuration path, solver, dtype, grid, matrix size, tile size,
        and destination JSON path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routine", choices=ROUTINES, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "complex64", "complex128"),
        required=True,
    )
    parser.add_argument("--grid", type=ProcessGrid.parse, required=True)
    parser.add_argument("--matrix-size", type=int, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write a complete JSON result without exposing a partial file.

    Args:
        path: Final result path written by rank zero.
        payload: JSON-serialisable benchmark result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _global_numpy(value: jax.Array) -> np.ndarray:
    """Materialize a small distributed result as one NumPy array.

    Args:
        value: A JAX array that may be distributed across hosts.

    Returns:
        The complete global value on every host when gathering is required.
    """
    if not value.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value)


def _global_scalar(value: jax.Array) -> float:
    """Return a replicated distributed scalar as one Python float.

    Args:
        value: A rank-zero JAX array produced by a global reduction.

    Returns:
        The scalar value, gathering first if it is not fully addressable.
    """
    return float(np.asarray(_global_numpy(value)).reshape(-1)[0])


def _global_max_seconds(elapsed: float) -> float:
    """Return the largest elapsed time reported by any process.

    Args:
        elapsed: Local elapsed wall-clock seconds.

    Returns:
        The slowest rank's duration, used as the distributed solve time.
    """
    gathered = multihost_utils.process_allgather(jnp.asarray(elapsed), tiled=False)
    return float(np.max(np.asarray(gathered)))


def _make_mesh(grid: ProcessGrid) -> Mesh:
    """Construct the row-major JAX mesh used by cuSOLVERMp.

    Args:
        grid: Requested process-grid shape.

    Returns:
        A mesh with ``pr`` and ``pc`` axes.

    Raises:
        RuntimeError: If the visible GPU count does not match the grid.
    """
    devices = np.asarray(jax.devices("gpu"), dtype=object)
    if devices.size != grid.processes:
        raise RuntimeError(
            f"grid {grid} requires {grid.processes} GPUs, but JAX sees {devices.size}"
        )
    return Mesh(devices.reshape(grid.rows, grid.cols), ("pr", "pc"))


def _make_input_factory(*, matrix_size: int, dtype: jnp.dtype, mesh: Mesh):
    """Build a compiled factory for the diagonal system used by a case.

    Args:
        matrix_size: Global square matrix dimension.
        dtype: JAX dtype for both matrix and right-hand side.
        mesh: Mesh determining the output sharding.

    Returns:
        A zero-argument jitted function returning sharded ``(A, b)`` arrays.
    """
    a_sharding = NamedSharding(mesh, P("pr", "pc"))
    b_sharding = NamedSharding(mesh, P("pr", None))

    def make_inputs():
        """Return ``diag(1, ..., N)`` and an all-ones single-column RHS."""
        diagonal = jnp.arange(1, matrix_size + 1, dtype=dtype)
        a = jnp.diag(diagonal)
        b = jnp.ones((matrix_size, 1), dtype=dtype)
        return a, b

    return jax.jit(make_inputs, out_shardings=(a_sharding, b_sharding))


def _make_gesvd_input_factory(
    *, matrix_size: int, dtype: jnp.dtype, mesh: Mesh, seed: int = 0
):
    """Build a compiled factory for the dense matrix decomposed by GESVD.

    Args:
        matrix_size: Global square matrix dimension.
        dtype: JAX dtype for the generated matrix.
        mesh: Mesh determining the output sharding.
        seed: Fixed PRNG seed, so a case is reproducible across runs.

    Returns:
        A zero-argument jitted function returning the sharded matrix.
    """
    a_sharding = NamedSharding(mesh, P("pr", "pc"))

    def make_inputs():
        """Return a dense standard-normal matrix of the configured dtype."""
        return jax.random.normal(
            jax.random.key(seed), (matrix_size, matrix_size), dtype=dtype
        )

    return jax.jit(make_inputs, out_shardings=a_sharding)


def _rank_status_codes(status: jax.Array, status_size: int) -> list[int]:
    """Return the leading native status value reported by each rank.

    Args:
        status: Concatenated native status values.
        status_size: Number of status values emitted by one rank.

    Returns:
        One native return code per participating rank.

    Raises:
        AssertionError: If the status shape is wrong or any code is non-zero.
    """
    words = _global_numpy(status).reshape(-1)
    if words.size % status_size:
        raise AssertionError(f"unexpected native status size {words.size}")
    rank_codes = [int(value) for value in words[::status_size]]
    if any(rank_codes):
        raise AssertionError(f"non-zero native status codes: {rank_codes}")
    return rank_codes


def _validate(
    *, out: jax.Array, status: jax.Array, status_size: int, dtype_name: str
) -> dict[str, object]:
    """Validate the diagonal solution and native status from every rank.

    Args:
        out: Returned solution vector or matrix.
        status: Concatenated native status values.
        status_size: Number of status values emitted by one rank.
        dtype_name: Benchmark dtype, used to select numerical tolerance.

    Returns:
        Result fields recording the solution error and per-rank return codes.

    Raises:
        AssertionError: If the solution, status shape, or native status fails.
    """
    solution = _global_numpy(out).reshape(-1)
    expected = 1.0 / np.arange(1, solution.size + 1, dtype=np.float64)
    maximum_error = float(np.max(np.abs(solution - expected)))
    tolerance = 5e-4 if dtype_name in ("float32", "complex64") else 1e-10
    if not np.allclose(solution, expected, rtol=tolerance, atol=tolerance):
        raise AssertionError(
            f"solution validation failed: max_abs_error={maximum_error}"
        )

    return {
        "max_abs_error": maximum_error,
        "native_status_codes": _rank_status_codes(status, status_size),
    }


def _validate_gesvd(
    *,
    singular_values: jax.Array,
    status: jax.Array,
    status_size: int,
    dtype_name: str,
    frobenius_squared: float,
) -> dict[str, object]:
    """Validate the singular values of a dense random matrix.

    We check the Frobenius identity ``sum(s_i**2) == ||A||_F**2``, which is
    exact and costs ``O(N**2)``, so it runs at every size.

    Args:
        singular_values: Replicated singular values returned by GESVD.
        status: Concatenated native status values.
        status_size: Number of status values emitted by one rank.
        dtype_name: Benchmark dtype, used to select numerical tolerance.
        frobenius_squared: ``||A||_F**2`` accumulated before the matrix was
            donated to the solver.

    Returns:
        Result fields recording the identity error and per-rank return codes.

    Raises:
        AssertionError: If either check, the status shape, or a status code fails.
    """
    single_precision = dtype_name in ("float32", "complex64")
    values = np.asarray(_global_numpy(singular_values), dtype=np.float64).reshape(-1)

    if np.any(values < 0.0):
        raise AssertionError("GESVD returned negative singular values")
    largest = float(values[0]) if values.size else 0.0
    if values.size > 1 and np.max(np.diff(values)) > 1e-5 * max(largest, 1.0):
        raise AssertionError("GESVD singular values are not in descending order")

    identity_error = abs(float(np.sum(values**2)) - frobenius_squared)
    identity_error /= frobenius_squared
    identity_tolerance = 5e-4 if single_precision else 1e-9
    if identity_error > identity_tolerance:
        raise AssertionError(
            "GESVD Frobenius identity failed: "
            f"relative_error={identity_error} tolerance={identity_tolerance}"
        )

    return {
        "frobenius_identity_error": identity_error,
        "native_status_codes": _rank_status_codes(status, status_size),
    }


def _package_versions() -> dict[str, str | None]:
    """Return installed package versions recorded with each result.

    Returns:
        Package-name to version mapping. Optional CUDA packages are recorded as
        ``None`` when their metadata is unavailable.
    """
    versions: dict[str, str | None] = {}
    for package in ("jax", "jaxlib", "jaxmg", "nvidia-cusolvermp-cu12"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_syevd(
    *,
    eigenvalues: jax.Array,
    status: jax.Array,
    status_size: int,
    dtype_name: str,
    trace: float,
    frobenius_squared: float,
) -> dict[str, object]:
    """Validate the eigenvalues of a dense symmetric matrix.

    Two exact identities hold for any symmetric or Hermitian matrix and cost
    ``O(N**2)`` or less, so both run at every size: ``sum(lambda_i) == trace(A)``
    and ``sum(lambda_i**2) == ||A||_F**2``. Because eigenvalues are signed, the
    trace identity constrains the result more tightly than the equivalent check
    can for singular values.

    Args:
        eigenvalues: Replicated ascending eigenvalues returned by SYEVD.
        status: Concatenated native status values.
        status_size: Number of status values emitted by one rank.
        dtype_name: Benchmark dtype, used to select numerical tolerance.
        trace: ``trace(A)`` accumulated before the matrix was donated.
        frobenius_squared: ``||A||_F**2`` accumulated before the matrix was
            donated.

    Returns:
        Result fields recording each error measure and per-rank return codes.

    Raises:
        AssertionError: If a check, the status shape, or a status code fails.
    """
    single_precision = dtype_name in ("float32", "complex64")
    tolerance = 5e-4 if single_precision else 1e-9
    values = np.asarray(_global_numpy(eigenvalues), dtype=np.float64).reshape(-1)

    # Both identities are normalized by ||A||_F
    scale = np.sqrt(frobenius_squared)
    if scale <= 0.0:
        raise AssertionError("input matrix has zero Frobenius norm")

    if values.size > 1 and np.min(np.diff(values)) < -1e-5 * scale:
        raise AssertionError("SYEVD eigenvalues are not in ascending order")

    trace_error = abs(float(np.sum(values)) - trace) / scale
    if trace_error > tolerance:
        raise AssertionError(
            "SYEVD trace identity failed: "
            f"relative_error={trace_error} tolerance={tolerance}"
        )

    identity_error = abs(float(np.sum(values**2)) - frobenius_squared)
    identity_error /= frobenius_squared
    if identity_error > tolerance:
        raise AssertionError(
            "SYEVD Frobenius identity failed: "
            f"relative_error={identity_error} tolerance={tolerance}"
        )

    return {
        "trace_identity_error": trace_error,
        "frobenius_identity_error": identity_error,
        "native_status_codes": _rank_status_codes(status, status_size),
    }


def _solve_iteration(
    *,
    solver,
    status_size: int,
    case: BenchmarkCase,
    mesh: Mesh,
    make_inputs,
    barrier: str,
) -> tuple[float, dict[str, object]]:
    """Time one POTRS or LU solve and validate its solution.

    Args:
        solver: ``jaxmg.potrs`` or ``jaxmg.lu_solve``.
        status_size: Number of native status values emitted by one rank.
        case: Case being measured.
        mesh: Process mesh matching the case grid.
        make_inputs: Compiled factory returning the sharded ``(A, b)`` pair.
        barrier: Unique prefix for this iteration's collective barriers.

    Returns:
        The slowest rank's solve duration and the validation result fields.
    """
    a, b = make_inputs()
    for array in (a, b):
        array.block_until_ready()
    multihost_utils.sync_global_devices(f"{barrier}_start")
    started = time.perf_counter()
    out, status = solver(
        a,
        b,
        T_A=case.tile_size,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
        return_status=True,
        pad=True,
    )
    out.block_until_ready()
    status.block_until_ready()
    multihost_utils.sync_global_devices(f"{barrier}_stop")
    elapsed = _global_max_seconds(time.perf_counter() - started)
    metrics = _validate(
        out=out, status=status, status_size=status_size, dtype_name=case.dtype
    )
    return elapsed, metrics


def _gesvd_iteration(
    *,
    solver,
    status_size: int,
    case: BenchmarkCase,
    mesh: Mesh,
    make_inputs,
    barrier: str,
) -> tuple[float, dict[str, object]]:
    """Time one reduced SVD and validate its singular values.

    Args:
        solver: ``jaxmg.gesvd``.
        status_size: Number of native status values emitted by one rank.
        case: Case being measured.
        mesh: Process mesh matching the case grid.
        make_inputs: Compiled factory returning the sharded dense matrix.
        barrier: Unique prefix for this iteration's collective barriers.

    Returns:
        The slowest rank's decomposition duration and the validation fields.
    """
    a = make_inputs()
    a.block_until_ready()
    # Accumulate in float64
    frobenius_squared = _global_scalar(
        jnp.sum(jnp.square(jnp.abs(a)), dtype=jnp.float64)
    )
    multihost_utils.sync_global_devices(f"{barrier}_start")
    started = time.perf_counter()
    u, singular_values, vh, status = solver(
        a,
        T_A=case.tile_size,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
        compute_u=True,
        compute_vh=True,
        full_matrices=False,
        return_status=True,
        pad=True,
    )
    for array in (u, singular_values, vh, status):
        array.block_until_ready()
    multihost_utils.sync_global_devices(f"{barrier}_stop")
    elapsed = _global_max_seconds(time.perf_counter() - started)
    metrics = _validate_gesvd(
        singular_values=singular_values,
        status=status,
        status_size=status_size,
        dtype_name=case.dtype,
        frobenius_squared=frobenius_squared,
    )
    return elapsed, metrics


def _syevd_iteration(
    *,
    solver,
    status_size: int,
    case: BenchmarkCase,
    mesh: Mesh,
    make_inputs,
    barrier: str,
) -> tuple[float, dict[str, object]]:
    """Time one symmetric eigendecomposition and validate its eigenvalues.

    ``trace(A)`` and ``||A||_F**2`` are both taken before the timed region,
    because SYEVD donates the input matrix.

    Args:
        solver: ``jaxmg.syevd``.
        status_size: Number of native status values emitted by one rank.
        case: Case being measured.
        mesh: Process mesh matching the case grid.
        make_inputs: Compiled factory returning the sharded symmetric matrix.
        barrier: Unique prefix for this iteration's collective barriers.

    Returns:
        The slowest rank's decomposition duration and the validation fields.
    """
    a = make_inputs()
    a.block_until_ready()
    # Accumulate both reductions in a wide dtype: a float32 accumulation over
    # the whole matrix loses the result to rounding at benchmark dimensions.
    trace_dtype = jnp.complex128 if case.dtype.startswith("complex") else jnp.float64
    trace = _global_scalar(jnp.real(jnp.trace(a, dtype=trace_dtype)))
    frobenius_squared = _global_scalar(
        jnp.sum(jnp.square(jnp.abs(a)), dtype=jnp.float64)
    )
    multihost_utils.sync_global_devices(f"{barrier}_start")
    started = time.perf_counter()
    eigenvalues, eigenvectors, status = solver(
        a,
        T_A=case.tile_size,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
        return_eigenvectors=True,
        return_status=True,
        pad=True,
    )
    for array in (eigenvalues, eigenvectors, status):
        array.block_until_ready()
    multihost_utils.sync_global_devices(f"{barrier}_stop")
    elapsed = _global_max_seconds(time.perf_counter() - started)
    metrics = _validate_syevd(
        eigenvalues=eigenvalues,
        status=status,
        status_size=status_size,
        dtype_name=case.dtype,
        trace=trace,
        frobenius_squared=frobenius_squared,
    )
    return elapsed, metrics


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Run the configured case and return its rank-zero JSON payload.

    Args:
        args: Parsed case description from ``_arguments``.

    Returns:
        Complete case metadata, timings, validation results, and package data.
    """
    if args.routine == "potrs":
        from jaxmg import potrs as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_POTRS_STATUS_SIZE as status_size,
        )

        run_iteration = _solve_iteration
    elif args.routine == "lu_solve":
        from jaxmg import lu_solve as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_LU_SOLVE_STATUS_SIZE as status_size,
        )

        run_iteration = _solve_iteration
    elif args.routine == "gesvd":
        from jaxmg import gesvd as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_GESVD_STATUS_SIZE as status_size,
        )

        run_iteration = _gesvd_iteration
    else:
        from jaxmg import syevd as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_SYEVD_STATUS_SIZE as status_size,
        )

        run_iteration = _syevd_iteration

    config = BenchmarkConfig.load(args.config)
    case = BenchmarkCase(
        args.routine, args.dtype, args.grid, args.matrix_size, args.tile_size
    )
    if case.needs_matrix_padding:
        raise ValueError(
            f"{case.case_id} needs matrix padding; N must be divisible by "
            f"{case.alignment_quantum}"
        )
    if jax.process_count() != case.grid.processes:
        raise RuntimeError(
            f"expected {case.grid.processes} processes, got {jax.process_count()}"
        )
    if len(jax.local_devices(backend="gpu")) != 1:
        raise RuntimeError("JAXMg requires one visible GPU per Python process")

    dtype = getattr(jnp, case.dtype)
    mesh = _make_mesh(case.grid)
    if case.routine == "gesvd":
        input_factory = _make_gesvd_input_factory
    else:
        input_factory = _make_input_factory
    make_inputs = input_factory(
        matrix_size=case.matrix_size, dtype=dtype, mesh=mesh
    )
    timings: list[float] = []
    metrics: dict[str, object] = {}
    for iteration in range(config.cold_runs + config.warm_runs):
        if iteration < config.cold_runs:
            phase, phase_iteration, phase_total = (
                "cold",
                iteration + 1,
                config.cold_runs,
            )
        else:
            phase, phase_iteration, phase_total = (
                "warm",
                iteration - config.cold_runs + 1,
                config.warm_runs,
            )
        if jax.process_index() == 0:
            print(
                f"{case.case_id}: {phase} {phase_iteration}/{phase_total} start",
                flush=True,
            )
        elapsed, metrics = run_iteration(
            solver=solver,
            status_size=status_size,
            case=case,
            mesh=mesh,
            make_inputs=make_inputs,
            barrier=f"{case.case_id}_{iteration}",
        )
        timings.append(elapsed)
        if jax.process_index() == 0:
            print(
                f"{case.case_id}: {phase} {phase_iteration}/{phase_total} "
                f"complete ({timings[-1]:.3f} s)",
                flush=True,
            )
        gc.collect()

    cold = timings[: config.cold_runs]
    warm = timings[config.cold_runs :]
    return {
        **case.to_dict(config),
        "status": "passed",
        "gpu_name": config.gpu_name,
        "visible_memory_mib": config.visible_memory_mib,
        "allocator": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "allocator_fraction": float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_median_seconds": float(np.median(warm)),
        "warm_min_seconds": float(np.min(warm)),
        "warm_max_seconds": float(np.max(warm)),
        **metrics,
        "jax_version": jax.__version__,
        "package_versions": _package_versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def main() -> None:
    """Initialize distributed JAX, run the case, and write one result on rank zero."""
    args = _arguments()
    # Slurm supplies rank and coordinator metadata. This makes the one GPU
    # assigned to each process explicit to JAX.
    local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(local_device_ids=[local_id])
    payload = _run(args)

    # Finish validation on every rank before rank zero writes the result.
    # Normal process exit releases the distributed runtime.
    multihost_utils.sync_global_devices("jaxmg_benchmark_case_complete")
    if jax.process_index() == 0:
        _atomic_json(Path(args.output), payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
