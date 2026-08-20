#!/usr/bin/env bash

# Isambard-AI Phase 2 runtime modules and communication libraries.
module load cray-python/3.11.7 cuda/12.6 gcc-native/14.2
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ucc/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/nccl/lib:${LD_LIBRARY_PATH:-}"
