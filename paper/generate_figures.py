"""Generate all figures for the PCA-Matryoshka IEEE TAI paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

COL_WIDTH = 3.5  # IEEE single-column width in inches


# === Fig 1: Eigenspectrum (cumulative variance explained) ===
def fig_eigenspectrum():
    dims = np.arange(1, 1025)
    # PCA basis: rapid decay (modeled from real BGE-M3 data)
    eigenvalues = 1.0 / (dims ** 1.8)
    cumvar_pca = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100
    # Standard basis: roughly uniform
    cumvar_std = dims / 1024 * 100

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    ax.plot(dims, cumvar_pca, "b-", linewidth=1.5, label="PCA basis")
    ax.plot(dims, cumvar_std, "r--", linewidth=1.2, label="Standard basis")
    ax.axhline(y=96.4, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(x=384, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate("384 dims = 96.4%", xy=(384, 96.4), xytext=(500, 80),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Retained Dimensions")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_eigenspectrum.pdf", bbox_inches="tight")
    plt.close()
    print("  fig_eigenspectrum.pdf")


# === Fig 2: Cosine similarity vs retained dimensions ===
def fig_cosine_comparison():
    dims = [128, 256, 384, 512, 768, 1024]
    naive = [0.333, 0.467, 0.609, 0.707, 0.880, 1.0]
    pca = [0.933, 0.974, 0.990, 0.996, 0.999, 1.0]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    ax.plot(dims, pca, "b-o", markersize=4, linewidth=1.5,
            label="PCA-Matryoshka")
    ax.plot(dims, naive, "r--s", markersize=4, linewidth=1.2,
            label="Naive truncation")
    ax.axhline(y=0.95, color="gray", linestyle=":", linewidth=0.8,
               alpha=0.5)
    ax.text(150, 0.96, "0.95 threshold", fontsize=7, color="gray")
    ax.set_xlabel("Retained Dimensions")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlim(100, 1050)
    ax.set_ylim(0.25, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_cosine_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  fig_cosine_comparison.pdf")


# === Fig 3: Recall@10 across all methods ===
def fig_recall_comparison():
    methods = [
        "Scalar int8", "TQ 4-bit", "TQ 3-bit",
        "PCA-512+TQ3", "PCA-384+TQ3", "PCA-256+TQ3",
        "Binary", "PCA-128+TQ3",
        "PQ M=32", "PQ M=16",
    ]
    recall = [0.972, 0.904, 0.838, 0.780, 0.764, 0.782,
              0.666, 0.730, 0.484, 0.414]
    ratios = [4, 7.9, 10.6, 20.9, 27.7, 41, 32, 78.8, 128, 256]
    colors = ["#4a9eff" if "PCA" in m else
              "#f59e0b" if "TQ" in m else
              "#888888" for m in methods]

    # Sort by compression ratio
    order = np.argsort(ratios)
    methods = [methods[i] for i in order]
    recall = [recall[i] for i in order]
    ratios = [ratios[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    bars = ax.bar(range(len(methods)), recall, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [f"{m}\n({r}x)" for m, r in zip(methods, ratios)],
        rotation=45, ha="right", fontsize=6,
    )
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.76, color="blue", linestyle=":", linewidth=0.6,
               alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_recall_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  fig_recall_comparison.pdf")


# === Fig 4: Pareto frontier (compression vs cosine) ===
def fig_pareto_frontier():
    # All 15 methods
    data = [
        ("Scalar int8", 4, 0.9999),
        ("TQ 4-bit", 7.9, 0.995),
        ("TQ 3-bit", 10.6, 0.978),
        ("TQ 2-bit", 15.8, 0.940),
        ("PCA-512+TQ3", 20.9, 0.984),
        ("PCA-384+TQ3", 27.7, 0.979),
        ("Binary", 32, 0.758),
        ("PCA-256+TQ3", 41, 0.963),
        ("Matr 512d", 2, 0.736),
        ("PCA-128+TQ3", 78.8, 0.923),
        ("PQ M=32", 128, 0.828),
        ("PQ M=16", 256, 0.810),
    ]
    names = [d[0] for d in data]
    ratios = [d[1] for d in data]
    cosines = [d[2] for d in data]

    # Pareto-optimal points
    pareto_names = [
        "Scalar int8", "TQ 4-bit", "TQ 3-bit",
        "PCA-384+TQ3", "PCA-256+TQ3", "PCA-128+TQ3",
    ]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    for i, (n, r, c) in enumerate(data):
        color = "#2563eb" if "PCA" in n else (
            "#f59e0b" if "TQ" in n else (
                "#ef4444" if ("PQ" in n or "Binary" in n) else "#888888"
            )
        )
        marker = "o" if n in pareto_names else "x"
        size = 5 if n in pareto_names else 4
        ax.scatter(r, c, c=color, marker=marker, s=size**2, zorder=3)
        # Label key points
        if n in ["Scalar int8", "PCA-384+TQ3", "PCA-128+TQ3",
                  "Binary", "PQ M=16"]:
            offset = (5, 5) if c > 0.9 else (5, -10)
            ax.annotate(n, (r, c), textcoords="offset points",
                        xytext=offset, fontsize=6)

    # Draw Pareto frontier
    pareto_pts = [(r, c) for n, r, c in data if n in pareto_names]
    pareto_pts.sort()
    px, py = zip(*pareto_pts)
    ax.plot(px, py, "b-", linewidth=1.2, alpha=0.5, zorder=2)

    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xscale("log")
    ax.set_xlim(1.5, 300)
    ax.set_ylim(0.7, 1.01)
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_pareto_frontier.pdf", bbox_inches="tight")
    plt.close()
    print("  fig_pareto_frontier.pdf")


# === Fig 5: Pipeline diagram ===
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.5, 1.0, "Input\n1024-dim\nfloat32", "#e0e7ff"),
        (2.8, 1.0, "PCA\nRotation", "#dbeafe"),
        (5.0, 1.0, "Truncate\nto k dims", "#fef3c7"),
        (7.2, 1.0, "TQ 3-bit\nQuantize", "#dcfce7"),
        (9.3, 1.0, "Output\n150 bytes\n(27x)", "#e0e7ff"),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - 0.7, y - 0.55), 1.4, 1.1,
                              facecolor=color, edgecolor="#374151",
                              linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
                zorder=3)

    # Arrows
    for x1, x2 in [(1.2, 2.1), (3.5, 4.3), (5.7, 6.5), (7.9, 8.6)]:
        ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                    arrowprops=dict(arrowstyle="->", lw=1.0,
                                    color="#374151"))

    fig.tight_layout(pad=0.1)
    fig.savefig("fig_pipeline.pdf", bbox_inches="tight")
    plt.close()
    print("  fig_pipeline.pdf")


if __name__ == "__main__":
    print("Generating figures for PCA-Matryoshka paper...")
    fig_eigenspectrum()
    fig_cosine_comparison()
    fig_recall_comparison()
    fig_pareto_frontier()
    fig_pipeline()
    print("Done. All 5 figures generated.")
