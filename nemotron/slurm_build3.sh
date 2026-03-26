#!/bin/bash
#SBATCH --job-name=build_mamba
#SBATCH --output=build_mamba3-%j.log
#SBATCH --error=build_mamba3-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e

GCC_DIR=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gcc-12.2.0-7gle75fpui2uzq74izjwiloxtobg4v4v
CUDA_DIR=/opt/ohpc/pub/nvidia/cuda-12.2

export PATH=$GCC_DIR/bin:$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export CC=$GCC_DIR/bin/gcc
export CXX=$GCC_DIR/bin/g++
export CUDAHOSTCXX=$GCC_DIR/bin/g++
export TORCH_CUDA_ARCH_LIST="6.0"
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS=4

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "=== Build Environment ==="
nvidia-smi -L
echo "GCC: $(gcc --version | head -1)"
echo ""

echo "=== Building mamba-ssm v2.2.4 from local source ==="
cd ~/nemotron/src_packages/mamba
pip install --no-build-isolation --no-deps -e . 2>&1
echo "Exit code: $?"

echo ""
echo "=== Verify ==="
cd ~
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')"
python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; print('rmsnorm_fn OK')"
echo "=== Done at $(date) ==="
