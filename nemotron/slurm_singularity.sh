#!/bin/bash
#SBATCH --job-name=nemotron
#SBATCH --output=nemotron-sif-%j.log
#SBATCH --error=nemotron-sif-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

echo "=== Nemotron LoRA Training (Singularity) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
nvidia-smi -L

# Pull container if not cached
SIF=~/nemotron/pytorch-24.12.sif
if [ ! -f "$SIF" ]; then
    echo "ERROR: Container not found at $SIF"
    echo "Pull it on login node first: singularity pull pytorch-24.12.sif docker://nvcr.io/nvidia/pytorch:24.12-py3"
    exit 1
fi

# Run training inside container
singularity exec --nv \
    --bind ~/nemotron:/workspace/nemotron \
    --bind ~/.cache/huggingface:/root/.cache/huggingface \
    "$SIF" \
    bash -c '
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    pip install -q peft accelerate bitsandbytes datasets trl sentencepiece 2>&1 | tail -3
    cd /workspace/nemotron
    python train_hpc.py 2>&1 | tee nemotron_output.log
    '

echo "=== Done at $(date) ==="
