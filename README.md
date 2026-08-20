# JAXMg Benchmarks

This repository benchmarks the distributed `jaxmg.potrs`, `jaxmg.lu_solve`,
`jaxmg.gesvd`, and `jaxmg.syevd` routines backed by cuSOLVERMp. Each GPU runs
one Python process.

The benchmark measures one cold call and one warm call for combinations of:

- matrix size;
- tile size;
- datatype;
- GPU count and two-dimensional process grid.

Every result is numerically validated. Matrix construction and validation are
outside the timed region.



## Requirements

You need:

- a Slurm cluster with NVIDIA GPUs;
- a working CUDA installation;
- a Python environment containing JAXMg and its CUDA dependencies;
- one or more complete GPU nodes available to a Slurm job.

Install the benchmark dependencies in the environment containing JAXMg:

```bash
git clone https://github.com/JacobTutt/jaxmg_benchmark.git
cd jaxmg_benchmark
python -m pip install -r requirements_cusolvermp_cuda12.txt
```

## 1. Describe Your Cluster

Copy the example profile:

```bash
cp configs/example_h200_8gpu.toml configs/my_cluster.toml
```

Edit the `[hardware]` section:

```toml
[hardware]
gpu_name = "NVIDIA H200"
visible_memory_mib = 143771
allocator_fraction = 0.99
gpus_per_node = 8
cpus_per_gpu = 16
```

The important values are:

- `visible_memory_mib`: memory reported for one GPU by `nvidia-smi`, in MiB;
- `gpus_per_node`: GPUs available on each requested node;
- `cpus_per_gpu`: CPU cores assigned to each GPU process.

Check the first value with:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Choose the node counts to benchmark:

```toml
[sweep]
node_counts = [1, 2, 3, 4]
```

The planner creates each non-transposed factor grid automatically. On an
eight-GPU node this gives:

```text
1 node:  8x1, 4x2
2 nodes: 16x1, 8x2, 4x4
3 nodes: 24x1, 12x2, 8x3, 6x4
4 nodes: 32x1, 16x2, 8x4
```

To select grids manually, replace `node_counts` with, for example:

```toml
grids = ["8x1", "4x2"]
```

The profile also contains the routines, datatypes, tile sizes, timing limits,
and Slurm resources. Optional site arguments can be added directly:

```toml
[execution]
cold_runs = 1
warm_runs = 3
case_timeout_seconds = 3600
```

`cold_runs` must remain `1`. Set `warm_runs` to the number of timed calls made
after compilation; the result records every warm duration and reports their
median, minimum, and maximum. `case_timeout_seconds` limits the complete fresh
process-group case, so increase it when adding warm calls or benchmarking very
large matrices.

Slurm resources and optional site arguments are configured separately:

```toml
[slurm]
# Maximum time for one complete routine/dtype/grid sweep.
suite_walltime = "12:00:00"
# Host CPU RAM requested per node. This is not GPU memory.
suite_memory = "512G"
submit_args = ["--partition=gpu", "--account=my-project"]
```

`suite_memory` becomes the Slurm `--mem` request and therefore means host CPU
RAM per allocated node. It is unrelated to `visible_memory_mib`, which controls
the GPU-memory-based matrix-size planner.

## 2. Set Up The Compute-Node Environment

Point the runner at the Python environment containing JAXMg:

```bash
export JAXMG_BENCHMARK_VENV="$HOME/.venvs/jaxmg"
```

If JAXMg is installed from an editable checkout rather than a wheel, also set:

```bash
export JAXMG_SOURCE_ROOT="$HOME/src/jaxmg"
```

Some clusters require modules or additional library paths on compute nodes.
Put those commands in a small shell script:

```bash
#!/usr/bin/env bash
module load cuda gcc
```

Then point the benchmark at it:

```bash
export JAXMG_BENCHMARK_SETUP="$PWD/cluster_setup/my_cluster.sh"
```

Skip this variable when your Python environment already provides everything.

There are two places to adapt Slurm execution:

- edit `[slurm]` in `configs/my_cluster.toml` for wall time, host memory,
  partition, account, or QOS arguments;
- edit `cluster_setup/my_cluster.sh` for `module load` commands and environment
  variables needed on the compute nodes.

The generic `slurm/suite.sbatch` runner normally does not need editing.

The runner defaults to:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=vmm
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
NCCL_CUMEM_ENABLE=0
```

Export different values before submission if your system requires them.

## 3. Review The Plan

Start with one small part of the sweep. This command only prints the planned
job and does not submit it:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/my_cluster.toml \
  --routine potrs \
  --dtype float32 \
  --grid 8x1
```

The output shows the number of cases, requested nodes and GPUs, wall time, and
the exact `sbatch` command.

Matrix sizes are derived from the configured GPU memory. More HBM or more GPUs
therefore produces larger near-limit cases automatically. Every selected `N`
also satisfies:

```text
N % (tile_size * lcm(process_rows, process_cols)) == 0
```

Consequently, the standard sweep does not require JAXMg padding.

## 4. Submit

After checking the printed command, add `--submit`:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/my_cluster.toml \
  --routine potrs \
  --dtype float32 \
  --grid 8x1 \
  --output-root results/my_cluster \
  --submit
```

Remove the routine, datatype, or grid filters to submit every corresponding
entry in the profile. Each solver configuration receives one outer Slurm job;
each matrix-size and tile-size case runs in a fresh `srun` process group.

Completed passing cases are skipped when a suite is submitted again. Recorded
failures are retried.

## Results

Results are stored as:

```text
results/my_cluster/
  cases/<routine>/<dtype>/<grid>/*.json
  logs/<routine>/<dtype>/<grid>/*.log
```

Each successful JSON record contains cold and warm timings, numerical error,
native status values, topology, package versions, and known memory estimates.
The estimate includes the local matrix, RHS capacity, redistribution scratch,
and LU pivots. cuSOLVERMp's internal workspace is not known before execution.

A successful record has this general form (some metadata fields are omitted
here for brevity):

```json
{
  "case_id": "potrs__float32__g8x1__n393216__t512",
  "status": "passed",
  "routine": "potrs",
  "dtype": "float32",
  "process_rows": 8,
  "process_cols": 1,
  "matrix_size": 393216,
  "tile_size": 512,
  "cold_seconds": [320.55],
  "warm_seconds": [317.19, 316.84, 317.02],
  "warm_median_seconds": 317.02,
  "warm_min_seconds": 316.84,
  "warm_max_seconds": 317.19,
  "max_abs_error": 2.98e-08,
  "native_status_codes": [0, 0, 0, 0, 0, 0, 0, 0],
  "local_matrix_bytes": 77309411328,
  "redistribution_scratch_bytes": 2415919104,
  "known_total_bytes": 79825993728,
  "known_budget_fraction": 0.7857,
  "solver_workspace_bytes": null
}
```

The principal fields are:

| Field | Meaning |
|---|---|
| `case_id` | Stable routine, datatype, grid, matrix-size, and tile-size identifier. |
| `cold_seconds` | Solve durations that include first-call compilation. |
| `warm_seconds` | Individual post-compilation solve durations. |
| `warm_median_seconds` | Primary warm timing used by the plotting command. |
| `max_abs_error` | Largest absolute difference from the expected solution. |
| `native_status_codes` | One native backend status value per participating rank; successful calls report zeros. |
| `local_matrix_bytes` | Input-matrix shard stored by one GPU process. |
| `redistribution_scratch_bytes` | Native redistribution scratch allocation per GPU process. |
| `known_total_bytes` | Matrix, RHS capacity, redistribution scratch, and LU pivots known before the solve. |
| `known_budget_fraction` | `known_total_bytes` divided by the configured per-GPU allocator budget. |
| `solver_workspace_bytes` | cuSOLVERMp workspace when known; currently `null` because the library chooses it internally. |

All durations are seconds and all memory sizes are bytes. The full record also
stores allocator settings, Slurm job information, GPU identity, JAX/JAXMg
versions, alignment information, and the global matrix size.

A failed case is deliberately smaller because timing and validation may never
have completed:

```json
{
  "case_id": "potrs__float32__g8x1__n450560__t256",
  "status": "failed",
  "routine": "potrs",
  "dtype": "float32",
  "process_rows": 8,
  "process_cols": 1,
  "matrix_size": 450560,
  "tile_size": 256,
  "returncode": 1,
  "elapsed_seconds": 18.42
}
```

`returncode` is the failed `srun` exit code, or `"timeout"` when the configured
case limit was reached. The corresponding file under `logs/` contains the
combined stdout and stderr needed to distinguish an OOM, NCCL error,
validation failure, or launcher problem.

Collect all records into CSV and JSONL tables:

```bash
python -m benchmark.cusolvermp.collect \
  --output-root results/my_cluster \
  --csv results/my_cluster/summary.csv \
  --jsonl results/my_cluster/summary.jsonl
```

The collected files contain one row or JSON object per case. Successful and
failed records are retained together, so filter on `status == "passed"` before
using timings in a performance plot.

Plot one completed configuration:

```bash
python -m benchmark.cusolvermp.plot \
  --output-root results/my_cluster \
  --routine potrs \
  --dtype float32 \
  --grid 8x1
```

`configs/isambard_gh200.toml` and `cluster_setup/isambard.sh` are examples of
a completed site profile. They are not required on another system.

## Troubleshooting

To debug hangs, use NCCL_DEBUG flags.

Both variables below are read by NCCL inside the compute-node processes, not by
the planner. `submit.py` builds the job with `--export ALL,...`, so exporting
them in the submitting shell is enough:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,P2P,GRAPH

python -m benchmark.cusolvermp.submit \
  --config configs/my_cluster.toml \
  --routine potrs --dtype float32 --grid 4x4 \
  --output-root results/my_cluster \
  --submit
```

### NCCL debug output

`NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET,P2P,GRAPH` reports
communicator setup, the selected network transport, peer-to-peer paths, and the
ring or tree topology. The output is written to the per-case file under
`logs/<routine>/<dtype>/<grid>/`, mixed with the benchmark's own output. Useful
patterns once a log exists:

```bash
grep -E 'NET/IB|NET/Socket' case.log     # transport chosen per channel
grep 'GPU Direct RDMA'      case.log     # whether GDR is active
grep 'Init COMPLETE'        case.log     # every communicator finished setup
```

This output is verbose: a single small case produces megabytes of log, and one
block is emitted per rank.

### Multi-node hangs

A multi-node case that produces no further output after its communicators
report `Init COMPLETE` is stalled in steady-state data transfer, which NCCL
does not log. Setting

```bash
export NCCL_IB_DISABLE=1
```

makes NCCL fall back to TCP sockets instead of InfiniBand verbs. If the case
then completes, the redistribution schedule and cuSOLVERMp are working and the
problem lies in the InfiniBand path — a cluster-level issue to report to your
site administrators rather than a benchmark bug. The TCP fallback is far slower
than IB, so use it only to isolate the fault, never for recorded timings.

A narrower variant keeps InfiniBand and disables only GPUDirect RDMA:

```bash
export NCCL_NET_GDR_LEVEL=0
```

`slurm/suite.sbatch` already applies this by default, because GPUDirect RDMA
deadlocks on some fabrics for particular cross-node transfer patterns. Export a
different value to override it.

