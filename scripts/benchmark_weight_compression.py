#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Benchmark: TurboQuant SVD weight compression vs baselines.

Sweeps matrix sizes, energy thresholds, and bit widths to measure
compression ratio, reconstruction quality, and factored matmul speed.

Usage:
    python scripts/benchmark_weight_compression.py
    python scripts/benchmark_weight_compression.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agi.meta.llm.turboquant_weights import (  # noqa: E402
    TurboQuantWeights,
    WeightCompressionConfig,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(dot / norm) if norm > 1e-30 else 1.0


def _frobenius_relative(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / n) if n > 1e-30 else 0.0


def _random_weight(d_out: int, d_in: int, seed: int = 42) -> np.ndarray:
    """Weight matrix with Zipf-like singular value decay."""
    rng = np.random.default_rng(seed)
    k = min(d_out, d_in)
    U = np.linalg.qr(rng.standard_normal((d_out, k)))[0]
    Vt = np.linalg.qr(rng.standard_normal((d_in, k)))[0].T
    S = 1.0 / np.sqrt(np.arange(1, k + 1))
    return ((U * S[np.newaxis, :]) @ Vt).astype(np.float32)


def benchmark_single(
    d_out: int,
    d_in: int,
    energy: float,
    bits: int,
    seed: int = 42,
) -> dict:
    """Benchmark one configuration."""
    W = _random_weight(d_out, d_in, seed=seed)
    config = WeightCompressionConfig(
        energy_threshold=energy,
        bits=bits,
        seed=seed,
    )
    engine = TurboQuantWeights(config)

    # Compress
    t0 = time.perf_counter()
    cw = engine.compress_weight(W, name=f"{d_out}x{d_in}")
    compress_ms = (time.perf_counter() - t0) * 1000

    # Decompress (full reconstruction)
    t0 = time.perf_counter()
    W_hat = engine.decompress_weight(cw)
    decompress_ms = (time.perf_counter() - t0) * 1000

    # Quality
    cos = _cosine_similarity(W, W_hat)
    frob = _frobenius_relative(W, W_hat)

    # Factored matmul benchmark
    x = np.random.default_rng(seed + 1).standard_normal((32, d_in)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        engine.compressed_linear(x, cw)
    factored_ms = (time.perf_counter() - t0) / 10 * 1000

    # Dense matmul baseline
    t0 = time.perf_counter()
    for _ in range(10):
        x @ W.T
    dense_ms = (time.perf_counter() - t0) / 10 * 1000

    # Baselines
    original_bytes = W.nbytes
    q4_bytes = d_out * d_in * 4 / 8  # 4-bit uniform
    q2_bytes = d_out * d_in * 2.5 / 8  # ~Q2_K average

    return {
        "shape": f"{d_out}x{d_in}",
        "energy_threshold": energy,
        "bits": bits,
        "rank": cw.rank,
        "energy_retained": round(cw.energy_retained, 4),
        "original_mb": round(original_bytes / 1e6, 2),
        "compressed_mb": round(cw.nbytes() / 1e6, 2),
        "ratio": round(cw.compression_ratio(), 2),
        "vs_q4": round(q4_bytes / max(cw.nbytes(), 1), 2),
        "vs_q2k": round(q2_bytes / max(cw.nbytes(), 1), 2),
        "cosine_sim": round(cos, 6),
        "frobenius_rel_err": round(frob, 6),
        "compress_ms": round(compress_ms, 1),
        "decompress_ms": round(decompress_ms, 1),
        "factored_matmul_ms": round(factored_ms, 2),
        "dense_matmul_ms": round(dense_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant weight compression benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/weight_compression_results.json",
        help="Output JSON path",
    )
    parser.add_argument("--quick", action="store_true", help="Run smaller sweep")
    args = parser.parse_args()

    if args.quick:
        shapes = [(512, 512), (1024, 1024)]
        energies = [0.90, 0.95]
        bit_widths = [3]
    else:
        shapes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (4096, 11008)]
        energies = [0.85, 0.90, 0.95, 0.99]
        bit_widths = [2, 3, 4]

    results = []
    total = len(shapes) * len(energies) * len(bit_widths)
    print(f"Running {total} configurations...\n")

    for _i, (d_out, d_in) in enumerate(shapes):
        for energy in energies:
            for bits in bit_widths:
                label = f"[{len(results)+1}/{total}] {d_out}x{d_in} e={energy} b={bits}"
                print(f"  {label}", end=" ... ", flush=True)
                try:
                    r = benchmark_single(d_out, d_in, energy, bits)
                    results.append(r)
                    print(
                        f"rank={r['rank']} ratio={r['ratio']}x "
                        f"cos={r['cosine_sim']:.4f} "
                        f"vs_q4={r['vs_q4']}x"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for bits in bit_widths:
        subset = [r for r in results if r["bits"] == bits]
        if not subset:
            continue
        avg_cos = np.mean([r["cosine_sim"] for r in subset])
        avg_ratio = np.mean([r["ratio"] for r in subset])
        avg_vs_q4 = np.mean([r["vs_q4"] for r in subset])
        print(
            f"  {bits}-bit: avg cosine={avg_cos:.4f}, "
            f"avg ratio={avg_ratio:.1f}x, "
            f"avg vs Q4={avg_vs_q4:.2f}x smaller"
        )

    # Estimation table
    print("\nEstimated compression for typical LLM layers (fp16):")
    for d_out, d_in in [(4096, 4096), (4096, 11008), (11008, 4096)]:
        est = TurboQuantWeights.estimate_compression(d_out, d_in, rank=512, bits=3)
        print(
            f"  {d_out}x{d_in}: {est['ratio']:.1f}x compression "
            f"({est['savings_pct']:.0f}% savings)"
        )

    # Save
    output = {
        "benchmark_info": {
            "description": "TurboQuant SVD weight compression benchmark",
            "version": "0.6.0",
            "shapes": [f"{d}x{i}" for d, i in shapes],
            "energies": energies,
            "bit_widths": bit_widths,
        },
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
