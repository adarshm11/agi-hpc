#!/bin/bash
set -e

# Use explicit paths instead of module load
GCC_DIR=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gcc-12.2.0-7gle75fpui2uzq74izjwiloxtobg4v4v
CUDA_DIR=/opt/ohpc/pub/nvidia/cuda-12.2

export PATH=$GCC_DIR/bin:$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export CC=$GCC_DIR/bin/gcc
export CXX=$GCC_DIR/bin/g++
export CUDAHOSTCXX=$GCC_DIR/bin/g++

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "GCC: $(gcc --version | head -1)"
echo "NVCC: $(nvcc --version | tail -1)"
echo "Python: $(python --version)"

# Uninstall pre-built (glibc-incompatible) versions
pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true

# P100 = compute capability 6.0
export TORCH_CUDA_ARCH_LIST="6.0"
export MAX_JOBS=4

echo "=== Building causal-conv1d from source ==="
pip install --no-binary causal-conv1d causal-conv1d 2>&1 | tail -15
echo "=== Building mamba-ssm from source ==="
pip install --no-binary mamba-ssm mamba-ssm 2>&1 | tail -15

echo "=== Verify ==="
python -c "import causal_conv1d; print(causal_conv1d)" 2>&1 || echo "causal_conv1d FAILED"
python -c "import mamba_ssm; print(mamba_ssm)" 2>&1 || echo "mamba_ssm FAILED"
echo "Done at $(date)"
