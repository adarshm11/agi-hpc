#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without WARRANTIES or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Benchmark TurboQuant KV cache compression.

Measures compression ratio, reconstruction error, and throughput for
2-bit, 3-bit, and 4-bit quantisation on simulated KV cache tensors
matching Gemma 4 27B dimensions.

Adapted from Theory Radar's TurboBeam for the Gemma 4 Good Hackathon.

Usage (CPU only -- safe to run alongside GPU workloads):
    python scripts/benchmark_turboquant_kv.py

    # With CuPy on a specific GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_turboquant_kv.py --gpu

    # Quick mode (smaller tensors):
    python scripts/benchmark_turboquant_kv.py --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List

import numpy as np

# Ensure the source tree is importable when running from repo root
sys.path.insert(0, "src")

from agi.meta.llm.turboquant_kv import TurboQuantKV  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Gemma 4 27B model dimensions                                       #
# ------------------------------------------------------------------ #

GEMMA4_CONFIG = {
    "n_layers": 36,
    "n_kv_heads": 16,
    "head_dim": 256,
    "hidden_dim": 4096,
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding vectors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    denom = np.maximum(norm_a * norm_b, 1e-30)
    return float(np.mean(dot / denom))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean relative L2 error per vector."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    err = np.linalg.norm(a_flat - b_flat, axis=-1)
    orig = np.maximum(np.linalg.norm(a_flat, axis=-1), 1e-30)
    return float(np.mean(err / orig))


def benchmark_single(
    bits: int,
    seq_len: int,
    use_gpu: bool,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Run a single benchmark configuration.

    Args:
        bits: Quantisation width (2, 3, or 4).
        seq_len: Sequence length to simulate.
        use_gpu: Whether to use CuPy.
        n_warmup: Number of warm-up iterations (not timed).
        n_trials: Number of timed iterations.

    Returns:
        Dict with metrics.
    """
    head_dim = GEMMA4_CONFIG["head_dim"]
    n_kv_heads = GEMMA4_CONFIG["n_kv_heads"]
    batch = 1

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((batch, n_kv_heads, seq_len, head_dim)).astype(
        np.float16
    )

    tq = TurboQuantKV(
        head_dim=head_dim,
        n_heads=n_kv_heads,
        bits=bits,
        use_gpu=use_gpu,
        seed=0,
    )

    # Warm-up
    for _ in range(n_warmup):
        c = tq.compress(tensor)
        _ = tq.decompress(c)

    # Timed compression
    t0 = time.perf_counter()
    for _ in range(n_trials):
        compressed = tq.compress(tensor)
    compress_time = (time.perf_counter() - t0) / n_trials

    # Timed decompression
    t0 = time.perf_counter()
    for _ in range(n_trials):
        reconstructed = tq.decompress(compressed)
    decompress_time = (time.perf_counter() - t0) / n_trials

    # Quality metrics (on float32 for precision)
    tensor_f32 = tensor.astype(np.float32)
    cos_sim = _cosine_similarity(tensor_f32, reconstructed)
    mse = _mse(tensor_f32, reconstructed)
    rel_err = _relative_error(tensor_f32, reconstructed)

    # Throughput
    tensor_mb = tensor.nbytes / (1024**2)
    compress_throughput = tensor_mb / max(compress_time, 1e-9)
    decompress_throughput = tensor_mb / max(decompress_time, 1e-9)

    # Compression ratio
    ratio = compressed.compression_ratio(head_dim)

    return {
        "bits": bits,
        "seq_len": seq_len,
        "tensor_mb": tensor_mb,
        "compressed_mb": compressed.nbytes() / (1024**2),
        "ratio": ratio,
        "mse": mse,
        "cosine_sim": cos_sim,
        "relative_err": rel_err,
        "compress_ms": compress_time * 1000,
        "decompress_ms": decompress_time * 1000,
        "compress_mb_s": compress_throughput,
        "decompress_mb_s": decompress_throughput,
    }


def print_results(results: List[Dict[str, float]]) -> None:
    """Pretty-print benchmark results."""
    print()
    print("=" * 80)
    print("TurboQuant KV Cache Compression Benchmark")
    print(
        f"Model: Gemma 4 27B  (n_kv_heads={GEMMA4_CONFIG['n_kv_heads']}, "
        f"head_dim={GEMMA4_CONFIG['head_dim']})"
    )
    print("=" * 80)
    print()

    # Group by seq_len
    seq_lens = sorted({r["seq_len"] for r in results})

    for seq_len in seq_lens:
        group = [r for r in results if r["seq_len"] == seq_len]
        print(f"--- Context length: {seq_len} tokens ---")
        print(
            f"{'Bits':>6} {'Ratio':>8} {'MSE':>12} {'Cos Sim':>10} "
            f"{'Rel Err':>10} {'Comp ms':>10} {'Decomp ms':>10} "
            f"{'Comp MB/s':>10} {'Decomp MB/s':>11}"
        )

        for r in sorted(group, key=lambda x: x["bits"]):
            print(
                f"{r['bits']:>6d} {r['ratio']:>8.2f}x "
                f"{r['mse']:>12.6f} {r['cosine_sim']:>10.6f} "
                f"{r['relative_err']:>10.6f} "
                f"{r['compress_ms']:>10.1f} {r['decompress_ms']:>10.1f} "
                f"{r['compress_mb_s']:>10.1f} {r['decompress_mb_s']:>11.1f}"
            )
        print()

    # Memory savings summary for full model at various context lengths
    print("=" * 80)
    print("Full-model KV cache memory estimates (Gemma 4 27B, fp16 baseline)")
    print("Current implementation: uint8 storage (1 byte per index)")
    print("=" * 80)
    print(
        f"{'Context':>10} {'Bits':>6} {'Original':>12} {'Compressed':>12} "
        f"{'Ratio':>8} {'Saved':>12}"
    )

    for ctx in [2048, 4096, 8192, 16384, 32768]:
        for bits in [2, 3, 4]:
            est = TurboQuantKV.estimate_memory(
                n_layers=GEMMA4_CONFIG["n_layers"],
                n_kv_heads=GEMMA4_CONFIG["n_kv_heads"],
                head_dim=GEMMA4_CONFIG["head_dim"],
                seq_len=ctx,
                bits=bits,
                original_dtype="float16",
                bit_packed=False,
            )
            print(
                f"{ctx:>10d} {bits:>6d} "
                f"{est['original_gb']:>10.3f} GB "
                f"{est['compressed_gb']:>10.3f} GB "
                f"{est['ratio']:>7.2f}x "
                f"{est['saved_gb']:>10.3f} GB"
            )
        print()

    print("=" * 80)
    print("With ideal bit-packing (future optimisation)")
    print("=" * 80)
    print(
        f"{'Context':>10} {'Bits':>6} {'Original':>12} {'Compressed':>12} "
        f"{'Ratio':>8} {'Saved':>12}"
    )

    for ctx in [2048, 4096, 8192, 16384, 32768]:
        for bits in [2, 3, 4]:
            est = TurboQuantKV.estimate_memory(
                n_layers=GEMMA4_CONFIG["n_layers"],
                n_kv_heads=GEMMA4_CONFIG["n_kv_heads"],
                head_dim=GEMMA4_CONFIG["head_dim"],
                seq_len=ctx,
                bits=bits,
                original_dtype="float16",
                bit_packed=True,
            )
            print(
                f"{ctx:>10d} {bits:>6d} "
                f"{est['original_gb']:>10.3f} GB "
                f"{est['compressed_gb']:>10.3f} GB "
                f"{est['ratio']:>7.2f}x "
                f"{est['saved_gb']:>10.3f} GB"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant KV cache compression"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CuPy GPU backend (default: CPU/NumPy)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with smaller tensors",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed trials (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.quick:
        seq_lens = [256, 1024]
    else:
        seq_lens = [512, 2048, 8192]

    bits_list = [2, 3, 4]
    results: List[Dict[str, float]] = []

    for seq_len in seq_lens:
        for bits in bits_list:
            logger.info(
                "Benchmarking: bits=%d, seq_len=%d, gpu=%s",
                bits,
                seq_len,
                args.gpu,
            )
            r = benchmark_single(
                bits=bits,
                seq_len=seq_len,
                use_gpu=args.gpu,
                n_trials=args.trials,
            )
            results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
