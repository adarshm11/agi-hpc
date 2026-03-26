#!/bin/bash
#SBATCH --job-name=clean_build
#SBATCH --output=clean_build-%j.log
#SBATCH --error=clean_build-%j.err
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
export FORCE_CUDA=1
export MAX_JOBS=4

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "GPU: $(nvidia-smi -L | head -1)"
echo "GCC: $(gcc --version | head -1)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Clean rebuild causal-conv1d
echo "=== Clean rebuild causal-conv1d v1.4.0 ==="
cd ~/nemotron/src_packages/causal-conv1d
pip uninstall -y causal-conv1d 2>/dev/null || true
rm -rf build/ dist/ *.egg-info csrc/build/ 2>/dev/null
python setup.py clean 2>/dev/null || true
pip install --no-build-isolation --no-deps -e . 2>&1 | tail -15

echo ""
echo "=== Clean rebuild mamba-ssm v2.2.4 ==="
cd ~/nemotron/src_packages/mamba
pip uninstall -y mamba-ssm 2>/dev/null || true
rm -rf build/ dist/ *.egg-info 2>/dev/null
python setup.py clean 2>/dev/null || true
export MAMBA_FORCE_BUILD=TRUE
pip install --no-build-isolation --no-deps -e . 2>&1 | tail -15

echo ""
echo "=== Verify on GPU ==="
cd ~
python -c "
import torch
print('CUDA device:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))

# Test causal_conv1d on GPU
import causal_conv1d
print('causal_conv1d:', causal_conv1d.__version__)

from causal_conv1d import causal_conv1d_fn
x = torch.randn(1, 32, 64, device='cuda', dtype=torch.float16)
w = torch.randn(64, 4, device='cuda', dtype=torch.float16)
out = causal_conv1d_fn(x, w)
print('causal_conv1d test passed, output shape:', out.shape)

import mamba_ssm
print('mamba_ssm:', mamba_ssm.__version__)
print('All OK!')
"
echo "=== Done at $(date) ==="
