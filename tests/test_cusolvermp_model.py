"""CPU-only tests for benchmark planning and memory accounting."""

from pathlib import Path
import unittest

from benchmark.cusolvermp.model import (
    BenchmarkCase,
    BenchmarkConfig,
    ProcessGrid,
    factor_grids,
    planned_sizes,
)
from benchmark.cusolvermp.run_suite import _selected_sizes, _srun_command


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

    def test_gesvd_frontier_is_smaller_than_the_solve_frontier(self):
        """Three matrix-sized buffers must shrink the planned frontier sizes."""
        for grid in (ProcessGrid(8, 1), ProcessGrid(4, 4)):
            for dtype in ("float32", "complex128"):
                potrs_sizes = planned_sizes(
                    self.config,
                    routine="potrs",
                    dtype=dtype,
                    grid=grid,
                    tile_size=256,
                )
                gesvd_sizes = planned_sizes(
                    self.config,
                    routine="gesvd",
                    dtype=dtype,
                    grid=grid,
                    tile_size=256,
                )
                self.assertLess(
                    max(gesvd_sizes), max(potrs_sizes), f"{grid} {dtype}"
                )

    def test_all_planned_sizes_need_no_matrix_padding(self):
        for routine in ("potrs", "lu_solve", "gesvd"):
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


if __name__ == "__main__":
    unittest.main()
