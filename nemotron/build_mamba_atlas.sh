#!/bin/bash
set -e
source /home/claude/env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.0"

echo "Python: $(which python3)"
echo "pip: $(which pip)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "torch CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""

echo "=== Installing causal-conv1d ==="
pip install causal-conv1d 2>&1 | tail -5

echo "=== Installing mamba-ssm ==="
pip install mamba-ssm 2>&1 | tail -5

echo "=== Verify ==="
python3 -c "import causal_conv1d; print('causal_conv1d:', causal_conv1d.__version__)"
python3 -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
echo "=== Done ==="
