#!/bin/bash
#SBATCH --job-name=rebuild_conv
#SBATCH --output=rebuild_conv-%j.log
#SBATCH --error=rebuild_conv-%j.err
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
export CUDAHOSTCXX=$GCC_DIR/bin/g++
export TORCH_CUDA_ARCH_LIST="6.0"
export MAX_JOBS=4

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "=== Rebuilding causal-conv1d v1.4.0 ==="
nvidia-smi -L

# Checkout compatible version
cd ~/nemotron/src_packages/causal-conv1d
pip uninstall -y causal-conv1d 2>/dev/null || true
git checkout v1.4.0

pip install --no-build-isolation -e . 2>&1 | tail -10

echo "=== Verify ==="
cd ~
python -c "import causal_conv1d; print('causal_conv1d version:', causal_conv1d.__version__)"
echo "=== Done at $(date) ==="
