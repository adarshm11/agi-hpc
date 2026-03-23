#!/bin/bash
#SBATCH --job-name=nemotron
#SBATCH --output=nemotron-%j.log
#SBATCH --error=nemotron-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

GCC_DIR=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gcc-12.2.0-7gle75fpui2uzq74izjwiloxtobg4v4v
CUDA_DIR=/opt/ohpc/pub/nvidia/cuda-12.2

export PATH=$GCC_DIR/bin:$CUDA_DIR/bin:~/nemotron/env/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export WORK_DIR=~/nemotron
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh
conda activate ~/nemotron/env

echo "=== Nemotron LoRA Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
nvidia-smi -L

cd ~/nemotron
python train_hpc.py 2>&1 | tee nemotron_output.log

echo "=== Done at $(date) ==="
