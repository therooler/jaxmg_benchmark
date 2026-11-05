"""CPU-only tests for benchmark planning and memory accounting."""

from pathlib import Path
import unittest

from benchmark.cusolvermp.model import (
    DEFAULT_BACKEND,
    NATIVE_TILE_SIZE,
    ROUTINES,
    BenchmarkCase,
    BenchmarkConfig,
    ProcessGrid,
    factor_grids,
    grids_for_backend,
    planned_sizes,
)
from benchmark.cusolvermp.run_suite import _selected_sizes, _srun_command
from benchmark.cusolvermp.submit import _submission_command


CONFIG = Path(__file__).parents[1] / "configs" / "isambard_gh200.toml"
H200_CONFIG = Path(__file__).parents[1] / "configs" / "example_h200_16gpu.toml"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = BenchmarkConfig.load(CONFIG)

    def test_isambard_allocator_budget(self):
        self.assertEqual(self.config.visible_memory_mib, 97871)
        self.assertEqual(self.config.cpus_per_gpu, 72)
        self.assertAlmostEqual(
            self.config.allocator_budget_per_gpu / 1024**3,
            94.6216,
            places=3,
        )
        self.assertEqual(self.config.suite_walltime, "24:00:00")
        self.assertEqual(self.config.suite_memory, "400G")

    def test_all_configurations_use_the_common_schema(self):
        for path in (CONFIG, H200_CONFIG):
            config = BenchmarkConfig.load(path)
            self.assertTrue(config.suite_walltime)
            self.assertTrue(config.suite_memory)

    def test_grids_are_generated_from_node_counts(self):
        self.assertEqual(
            self.config.grids,
            (
                ProcessGrid(4, 1), ProcessGrid(2, 2),
                ProcessGrid(8, 1), ProcessGrid(4, 2),
                ProcessGrid(12, 1), ProcessGrid(6, 2), ProcessGrid(4, 3),
                ProcessGrid(16, 1), ProcessGrid(8, 2), ProcessGrid(4, 4),
            ),
        )

    def test_eight_gpu_node_generates_h200_style_grids(self):
        h200 = BenchmarkConfig.load(H200_CONFIG)
        self.assertEqual(
            factor_grids(8),
            (ProcessGrid(8, 1), ProcessGrid(4, 2)),
        )
        self.assertIn(ProcessGrid(32, 1), h200.grids)
        self.assertIn(ProcessGrid(8, 4), h200.grids)

    def test_larger_hbm_generates_larger_frontier_cases(self):
        h200 = BenchmarkConfig.load(H200_CONFIG)
        gh200_sizes = planned_sizes(
            self.config,
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(8, 1),
            tile_size=256,
        )
        h200_sizes = planned_sizes(
            h200,
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(8, 1),
            tile_size=256,
        )
        self.assertGreater(max(h200_sizes), max(gh200_sizes))

    def test_alignment_quantum_uses_grid_lcm(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(6, 2), 24576, 1024)
        self.assertEqual(case.alignment_quantum, 6144)
        self.assertFalse(case.needs_matrix_padding)

    def test_misaligned_case_is_rejected_by_memory_model(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(4, 1), 5000, 1024)
        self.assertTrue(case.needs_matrix_padding)
        with self.assertRaises(ValueError):
            case.estimate_memory(self.config)

    def test_lu_adds_int64_pivots(self):
        potrs = BenchmarkCase("potrs", "float64", ProcessGrid(4, 1), 16384, 1024)
        lu = BenchmarkCase("lu_solve", "float64", ProcessGrid(4, 1), 16384, 1024)
        self.assertEqual(potrs.estimate_memory(self.config).pivot_bytes, 0)
        self.assertEqual(lu.estimate_memory(self.config).pivot_bytes, 16384 * 8)

    def test_gesvd_accounts_for_the_singular_vector_outputs(self):
        """GESVD holds U and Vh; the solves hold neither."""
        potrs = BenchmarkCase("potrs", "float64", ProcessGrid(4, 1), 16384, 1024)
        gesvd = BenchmarkCase("gesvd", "float64", ProcessGrid(4, 1), 16384, 1024)
        potrs_memory = potrs.estimate_memory(self.config)
        gesvd_memory = gesvd.estimate_memory(self.config)
        self.assertEqual(potrs_memory.vector_output_bytes, 0)
        self.assertEqual(
            gesvd_memory.vector_output_bytes, 2 * gesvd_memory.local_matrix_bytes
        )
        # GESVD takes no right-hand side and does no pivoting.
        self.assertEqual(gesvd_memory.rhs_capacity_bytes, 0)
        self.assertEqual(gesvd_memory.pivot_bytes, 0)
        self.assertGreater(
            gesvd_memory.known_total_bytes, potrs_memory.known_total_bytes
        )

    def test_syevd_accounts_for_the_eigenvector_output(self):
        """SYEVD keeps one matrix-sized result; GESVD keeps two."""
        syevd = BenchmarkCase("syevd", "float64", ProcessGrid(4, 1), 16384, 1024)
        gesvd = BenchmarkCase("gesvd", "float64", ProcessGrid(4, 1), 16384, 1024)
        syevd_memory = syevd.estimate_memory(self.config)
        gesvd_memory = gesvd.estimate_memory(self.config)
        self.assertEqual(
            syevd_memory.vector_output_bytes, syevd_memory.local_matrix_bytes
        )
        self.assertEqual(
            gesvd_memory.vector_output_bytes, 2 * gesvd_memory.local_matrix_bytes
        )
        # Neither decomposition takes a right-hand side or pivots.
        self.assertEqual(syevd_memory.rhs_capacity_bytes, 0)
        self.assertEqual(syevd_memory.pivot_bytes, 0)

    def test_decomposition_frontiers_are_smaller_than_the_solve_frontier(self):
        """Extra matrix-sized buffers must shrink the planned frontier sizes."""
        for grid in (ProcessGrid(8, 1), ProcessGrid(4, 4)):
            for dtype in ("float32", "complex128"):
                potrs_sizes = planned_sizes(
                    self.config,
                    routine="potrs",
                    dtype=dtype,
                    grid=grid,
                    tile_size=256,
                )
                for routine in ("gesvd", "syevd"):
                    sizes = planned_sizes(
                        self.config,
                        routine=routine,
                        dtype=dtype,
                        grid=grid,
                        tile_size=256,
                    )
                    self.assertLess(
                        max(sizes), max(potrs_sizes), f"{routine} {grid} {dtype}"
                    )

    def test_all_planned_sizes_need_no_matrix_padding(self):
        for routine in ROUTINES:
            for grid in self.config.grids:
                for dtype in self.config.dtypes:
                    for tile in self.config.tiles:
                        for size in planned_sizes(
                            self.config,
                            routine=routine,
                            dtype=dtype,
                            grid=grid,
                            tile_size=tile,
                        ):
                            case = BenchmarkCase(routine, dtype, grid, size, tile)
                            self.assertFalse(
                                case.needs_matrix_padding, case.case_id
                            )

    def test_large_grid_has_baseline_bridges_before_frontier(self):
        """Shared baselines fill the multi-node gap without entering 85% sweep."""
        sizes = planned_sizes(
            self.config,
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(4, 4),
            tile_size=256,
        )
        self.assertTrue({327680, 393216, 458752, 524288}.issubset(sizes))
        self.assertEqual(sizes[-8], 586752)
        self.assertNotIn(589824, sizes)

    def test_explicit_sizes_drop_padding_cases_per_tile(self):
        class Arguments:
            routine = "potrs"
            dtype = "float32"
            grid = ProcessGrid(4, 1)
            matrix_size = [4096, 5000]

        self.assertEqual(_selected_sizes(Arguments(), self.config, tile_size=1024), (4096,))

    def test_case_command_uses_the_reserved_cpu_share(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(4, 1), 4096, 1024)
        command = _srun_command(
            case=case,
            config=self.config,
            config_path=CONFIG,
            output=Path("results/example.json"),
        )
        self.assertEqual(command[command.index("--cpus-per-task") + 1], "72")
        self.assertEqual(command[command.index("--ntasks") + 1], "4")
        self.assertIn("--kill-on-bad-exit", command)


class NativeBackendTests(unittest.TestCase):
    """The single-GPU reference shares the planner with the distributed path."""

    @classmethod
    def setUpClass(cls):
        cls.config = BenchmarkConfig.load(CONFIG)

    @staticmethod
    def native_case(routine="potrs", dtype="float32", matrix_size=4096):
        """Return a single-GPU reference case for the given routine."""
        return BenchmarkCase(
            routine, dtype, ProcessGrid(1, 1), matrix_size, NATIVE_TILE_SIZE, "jax"
        )

    def test_backend_defaults_to_the_distributed_path(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(4, 1), 4096, 1024)
        self.assertEqual(case.backend, DEFAULT_BACKEND)
        self.assertFalse(case.is_native)

    def test_distributed_case_id_is_unchanged_by_the_backend_axis(self):
        # Completed cases are skipped by filename, so every recorded result
        # would be orphaned and rerun if this identifier ever moved.
        case = BenchmarkCase("potrs", "float32", ProcessGrid(4, 1), 4096, 1024)
        self.assertEqual(case.case_id, "potrs__float32__g4x1__n4096__t1024")

    def test_native_case_id_is_distinguishable(self):
        self.assertEqual(
            self.native_case().case_id, "potrs__float32__g1x1__n4096__t0__bjax"
        )

    def test_native_dimensions_need_no_alignment(self):
        case = self.native_case(matrix_size=5000)
        self.assertEqual(case.alignment_quantum, 1)
        self.assertFalse(case.needs_matrix_padding)

    def test_native_estimate_drops_the_redistribution_scratch(self):
        distributed = BenchmarkCase("potrs", "float32", ProcessGrid(1, 1), 4096, 1024)
        self.assertGreater(
            distributed.estimate_memory(self.config).redistribution_scratch_bytes, 0
        )
        native = self.native_case().estimate_memory(self.config)
        self.assertEqual(native.redistribution_scratch_bytes, 0)
        # The matrix itself is still the dominant known allocation.
        self.assertEqual(native.local_matrix_bytes, 4096**2 * 4)

    def test_native_lu_pivots_are_narrower_than_the_distributed_ones(self):
        native = self.native_case(routine="lu_solve").estimate_memory(self.config)
        distributed = BenchmarkCase(
            "lu_solve", "float32", ProcessGrid(1, 1), 4096, 1024
        ).estimate_memory(self.config)
        self.assertEqual(native.pivot_bytes, 4096 * 4)
        self.assertEqual(distributed.pivot_bytes, 4096 * 8)

    def test_native_frontier_is_far_below_the_distributed_one(self):
        grid = ProcessGrid(16, 1)
        distributed = planned_sizes(
            self.config, routine="potrs", dtype="float32", grid=grid, tile_size=256
        )
        native = planned_sizes(
            self.config,
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(1, 1),
            tile_size=NATIVE_TILE_SIZE,
            backend="jax",
        )
        # One GPU holds one sixteenth of the memory and a second matrix copy.
        self.assertLess(max(native), max(distributed) / 4)

    def test_native_backend_has_one_grid(self):
        self.assertEqual(grids_for_backend(self.config, "jax"), (ProcessGrid(1, 1),))
        self.assertEqual(
            grids_for_backend(self.config, DEFAULT_BACKEND), self.config.grids
        )

    def test_native_allocation_still_reserves_whole_nodes(self):
        # Sites enforce a minimum GRES per job; asking for one GPU is rejected
        # with QOSMinGRES. The allocation is whole-node, and the per-case srun
        # inside it is what narrows the run down to a single GPU.
        command = _submission_command(
            config=self.config,
            config_path=Path("/cfg.toml"),
            script_path=Path("/s.sbatch"),
            output_root=Path("/r"),
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(1, 1),
            backend="jax",
        )
        self.assertEqual(command[command.index("--nodes") + 1], "1")
        self.assertEqual(
            command[command.index("--gres") + 1], f"gpu:{self.config.gpus_per_node}"
        )
        self.assertEqual(
            command[command.index("--mem") + 1], self.config.suite_memory
        )

    def test_native_case_asks_for_a_single_task(self):
        command = _srun_command(
            case=self.native_case(),
            config=self.config,
            config_path=CONFIG,
            output=Path("results/example.json"),
        )
        self.assertEqual(command[command.index("--nodes") + 1], "1")
        self.assertEqual(command[command.index("--ntasks") + 1], "1")
        self.assertEqual(command[command.index("--ntasks-per-node") + 1], "1")
        self.assertEqual(command[command.index("--backend") + 1], "jax")

    def test_distributed_launch_is_untouched_by_the_backend_axis(self):
        # A regression lock: the whole argv, not a sample of it. The recorded
        # sweep is only comparable to new results if the distributed launch is
        # bit-for-bit what produced it, apart from the added backend variable.
        grid = ProcessGrid(4, 1)
        command = _srun_command(
            case=BenchmarkCase("potrs", "float32", grid, 4096, 1024),
            config=self.config,
            config_path=Path("/cfg.toml"),
            output=Path("/out.json"),
        )
        self.assertEqual(
            command[:10],
            ["srun", "--kill-on-bad-exit", "--nodes", "1", "--ntasks", "4",
             "--ntasks-per-node", "4", "--cpus-per-task", "72"],
        )
        tail = ["-m", "benchmark.cusolvermp.run_case",
                "--config", "/cfg.toml", "--routine", "potrs",
                "--dtype", "float32", "--grid", "4x1", "--backend", "cusolvermp",
                "--matrix-size", "4096", "--tile-size", "1024",
                "--output", "/out.json"]
        self.assertEqual(command[-len(tail):], tail)

    def test_native_frontier_brackets_the_measured_memory_limit(self):
        # NATIVE_MATRIX_COPIES is measured, not derived, so it can only be
        # defended by pinning the frontier it produces.
        sizes = planned_sizes(
            self.config,
            routine="potrs",
            dtype="float32",
            grid=ProcessGrid(1, 1),
            tile_size=NATIVE_TILE_SIZE,
            backend="jax",
        )
        budget = self.config.allocator_budget_per_gpu
        limit = int((budget / 4 / 4) ** 0.5)
        self.assertEqual(max(sizes), limit)
        # The fine sweep must approach that limit, not stop short of it.
        self.assertGreater(sorted(sizes)[-8], 0.9 * limit)

    def test_configurations_without_a_backend_key_stay_distributed(self):
        for path in (CONFIG, H200_CONFIG):
            with self.subTest(config=path.name):
                self.assertEqual(
                    BenchmarkConfig.load(path).backends, (DEFAULT_BACKEND,)
                )


if __name__ == "__main__":
    unittest.main()
