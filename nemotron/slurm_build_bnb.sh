#!/bin/bash
#SBATCH --job-name=build_bnb
#SBATCH --output=build_bnb-%j.log
#SBATCH --error=build_bnb-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

GCC_DIR=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gcc-12.2.0-7gle75fpui2uzq74izjwiloxtobg4v4v
CUDA_DIR=/opt/ohpc/pub/nvidia/cuda-12.2

export PATH=$GCC_DIR/bin:$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export CC=$GCC_DIR/bin/gcc
export CXX=$GCC_DIR/bin/g++
export CUDA_HOME=$CUDA_DIR
export CUDA_VERSION=122

# Load cmake from spack
CMAKE_DIR=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-9.5.0/cmake-3.25.1-bfsr6bue75xjs2p4ouurifjw3dlabp6z
export PATH=$CMAKE_DIR/bin:$PATH

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "=== Building bitsandbytes 0.46.1 ==="
nvidia-smi -L | head -1

cd ~/nemotron/src_packages/bitsandbytes
git checkout 0.46.1
pip uninstall -y bitsandbytes 2>/dev/null || true

# Build
cmake -DCOMPUTE_BACKEND=cuda -S . -B build 2>&1 | tail -5
cmake --build build 2>&1 | tail -10
pip install --no-build-isolation --no-deps -e . 2>&1 | tail -5

echo "=== Verify ==="
python -c "import bitsandbytes; print('bnb version:', bitsandbytes.__version__)"
echo "=== Done at $(date) ==="
