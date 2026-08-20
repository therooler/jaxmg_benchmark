#!/usr/bin/env bash
set -euo pipefail

# submit.py supplies the allocation, solver, dtype, grid, and output root.
: "${JAXMG_BENCHMARK_CONFIG:?Missing benchmark configuration}"
: "${JAXMG_BENCHMARK_ROUTINE:?Missing routine}"
: "${JAXMG_BENCHMARK_DTYPE:?Missing dtype}"
: "${JAXMG_BENCHMARK_GRID:?Missing process grid}"
: "${JAXMG_BENCHMARK_OUTPUT_ROOT:?Missing output directory}"

# A cluster profile may load modules and extend LD_LIBRARY_PATH here. Keeping
# that setup outside this generic runner makes the benchmark portable.
if [[ -n "${JAXMG_BENCHMARK_SETUP:-}" ]]; then
    source "${JAXMG_BENCHMARK_SETUP}"
fi
if [[ -n "${JAXMG_BENCHMARK_VENV:-}" ]]; then
    source "${JAXMG_BENCHMARK_VENV}/bin/activate"
fi

# CUDA compilation requires a writable node-local temporary directory.
export TMPDIR="${JAXMG_BENCHMARK_TMPDIR:-/tmp}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

if [[ -n "${JAXMG_SOURCE_ROOT:-}" ]]; then
    export PYTHONPATH="${JAXMG_SOURCE_ROOT}/src:${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
fi

# Discover CUDA libraries installed by NVIDIA's Python wheels. A cluster setup
# script can add system NCCL, UCC, or CUDA paths before this runner starts.
NVIDIA_LIBRARY_PATH="$(python - <<'PY'
from pathlib import Path
import site

paths = []
for root in site.getsitepackages():
    nvidia = Path(root) / "nvidia"
    if nvidia.is_dir():
        paths.extend(str(path) for path in sorted(nvidia.glob("*/lib")))
print(":".join(paths))
PY
)"
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-vmm}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.99}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# GPUDirect RDMA (GDRDMA) deadlocks on certain IB fabrics
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

extra_args=()
if [[ -n "${JAXMG_BENCHMARK_TILE_SIZE:-}" ]]; then
    extra_args+=(--tile-size "${JAXMG_BENCHMARK_TILE_SIZE}")
fi
if [[ -n "${JAXMG_BENCHMARK_MATRIX_SIZES:-}" ]]; then
    read -r -a matrix_sizes <<< "${JAXMG_BENCHMARK_MATRIX_SIZES}"
    for matrix_size in "${matrix_sizes[@]}"; do
        extra_args+=(--matrix-size "${matrix_size}")
    done
elif [[ -n "${JAXMG_BENCHMARK_MATRIX_SIZE:-}" ]]; then
    extra_args+=(--matrix-size "${JAXMG_BENCHMARK_MATRIX_SIZE}")
fi

python -m benchmark.cusolvermp.run_suite \
    --config "${JAXMG_BENCHMARK_CONFIG}" \
    --routine "${JAXMG_BENCHMARK_ROUTINE}" \
    --dtype "${JAXMG_BENCHMARK_DTYPE}" \
    --grid "${JAXMG_BENCHMARK_GRID}" \
    --output-root "${JAXMG_BENCHMARK_OUTPUT_ROOT}" \
    --retry-failures \
    "${extra_args[@]}"
