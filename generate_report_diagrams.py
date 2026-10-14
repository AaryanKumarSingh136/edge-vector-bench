#!/usr/bin/env python3
"""
generate_report_diagrams.py
Generates high-resolution publication-quality diagrams for the formal SysML research report:
1. pipeline_architecture.png: 4-stage hardware-enforced benchmarking methodology.
2. technique_architectures_comparison.png: Structural architectural comparison of HNSW, IVF-Flat, IVF-PQ, and Flat/Exact.
3. memory_breakdown_comparison.png: Theoretical vs. Real-world Peak RSS decomposition across techniques.
4. pareto_frontier.png: Recall@10 vs. Throughput (QPS) with memory footprint encoding.
5. build_time_vs_rss.png: Index build time vs Peak RSS footprint.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = os.path.join("results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    stages = [
        {
            "num": "STAGE 1",
            "title": "Kernel Isolation & CPU Pinning",
            "bullets": [
                "• sudo swapoff -a (0 swap)",
                "• drop_caches (echo 3)",
                "• Pin OMP/BLAS to 4 cores",
                "• Purge kernel dentries & slab"
            ],
            "x": 2, "color": "#1E3A8A", "accent": "#3B82F6"
        },
        {
            "num": "STAGE 2",
            "title": "Hardware Clamping (cgroups v2)",
            "bullets": [
                "• systemd-run --scope",
                "• MemoryMax={1024, 512, 256, 128}M",
                "• MemorySwapMax=0",
                "• Hardware-level quota enforcement"
            ],
            "x": 27, "color": "#7C2D12", "accent": "#EA580C"
        },
        {
            "num": "STAGE 3",
            "title": "Isolated Worker Evaluation",
            "bullets": [
                "• SIFT-128 100k slice load",
                "• 5ms background RSS poller",
                "• 100 warm-up queries",
                "• 10,000 queries @ K=10 (ns timer)"
            ],
            "x": 52, "color": "#064E3B", "accent": "#10B981"
        },
        {
            "num": "STAGE 4",
            "title": "Signal Trap & Telemetry",
            "bullets": [
                "• Exit 0: Harvest JSON stdout",
                "• Exit 137: Trap OOM Killer",
                "• Map physical cliff boundary",
                "• Stream to telemetry.csv"
            ],
            "x": 77, "color": "#4C1D95", "accent": "#8B5CF6"
        }
    ]

    for s in stages:
        # Box background
        rect = patches.FancyBboxPatch(
            (s["x"], 4), 21, 32,
            boxstyle="round,pad=0.8,rounding_size=2.0",
            facecolor="#F8FAFC",
            edgecolor=s["accent"],
            linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)

        # Header bar
        header_rect = patches.FancyBboxPatch(
            (s["x"], 30), 21, 6,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            facecolor=s["color"],
            edgecolor="none",
            zorder=3
        )
        ax.add_patch(header_rect)

        ax.text(s["x"] + 10.5, 33.2, s["num"], color="#FFFFFF", fontweight="bold", fontsize=10, ha="center", zorder=4)
        ax.text(s["x"] + 10.5, 27.5, s["title"], color=s["color"], fontweight="bold", fontsize=9.5, ha="center", zorder=4)

        for i, bullet in enumerate(s["bullets"]):
            ax.text(s["x"] + 1.5, 23.0 - (i * 4.6), bullet, color="#334155", fontsize=8.5, ha="left", zorder=4)

    # Arrows between stages
    arrow_positions = [23.5, 48.5, 73.5]
    for ap in arrow_positions:
        ax.annotate(
            "", xy=(ap + 3.2, 20), xytext=(ap - 0.2, 20),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#64748B", mutation_scale=15),
            zorder=5
        )

    plt.title("Edge-Vector-Bench: Four-Stage Hardware-Enforced Benchmarking Framework",
              fontweight="bold", fontsize=13, pad=12, color="#0F172A")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pipeline_architecture.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated: {out}")

def create_technique_architectures_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    techs = [
        {
            "name": "1. HNSW (Graph-Based)",
            "subtitle": "Hierarchical Proximity Graphs",
            "desc": "Multi-layer skip-list graph.\nVertices = vectors.\nEdges = Delaunay approximations.\nDynamic neighbor tables.",
            "pros": "• Peak Recall@10 (>99%)\n• Sub-millisecond latency\n• High QPS at large RAM",
            "cons": "• Dynamic pointer bloat\n• High RSS overhead (6-8x)\n• OOM Crash at <512 MB",
            "bg": "#EFF6FF", "border": "#3B82F6", "badge": "Memory Intensive"
        },
        {
            "name": "2. IVF-Flat (Cluster-Based)",
            "subtitle": "Inverted Lists of Raw Vectors",
            "desc": "Space partitioned into nlist\nVoronoi cells (k-means).\nUncompressed 128-d float vectors\nappended to closest centroid list.",
            "pros": "• Linear memory scaling\n• Excellent recall (98.2%)\n• Stable down to 256 MB",
            "cons": "• Full float32 coordinate storage\n• 51.2 MB base vector load\n• OOM Crash at 128 MB",
            "bg": "#ECFDF5", "border": "#10B981", "badge": "Balanced Tier"
        },
        {
            "name": "3. IVF-PQ (Quantized)",
            "subtitle": "Subspace Product Quantization",
            "desc": "Vectors split into M_pq subspaces.\nEach subspace clustered into 256\nsub-centroids (1 byte/subspace).\nAsymmetric Distance (ADC).",
            "pros": "• 16x-64x compression factor\n• 42-65 MB peak RSS\n• 100% survival down to 128MB",
            "cons": "• Lossy quantization noise\n• Capped recall (89.4% @ M32)\n• Slower cluster training",
            "bg": "#FFF7ED", "border": "#F97316", "badge": "Extreme Edge"
        },
        {
            "name": "4. Flat (Exhaustive Baseline)",
            "subtitle": "Brute-Force Exact Search",
            "desc": "Contiguous array of 100k\nfloat32 vectors. Zero indexing.\nCalculates Euclidean distance to\nall 100k vectors per query.",
            "pros": "• 100.00% exact ground truth\n• Zero build time (0.00s)\n• Zero graph/pointer overhead",
            "cons": "• O(N*d) linear latency\n• High query time (~15-40ms)\n• Not scalable for streaming",
            "bg": "#F5F3FF", "border": "#8B5CF6", "badge": "Exact Baseline"
        }
    ]

    for ax, t in zip(axes, techs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Container card
        box = patches.FancyBboxPatch(
            (0.3, 0.3), 9.4, 9.4,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=t["bg"],
            edgecolor=t["border"],
            linewidth=2
        )
        ax.add_patch(box)

        # Title
        ax.text(5.0, 9.1, t["name"], color="#0F172A", fontweight="bold", fontsize=10.5, ha="center")
        ax.text(5.0, 8.4, t["subtitle"], color="#475569", fontsize=8.5, ha="center", style="italic")

        # Badge
        badge_box = patches.FancyBboxPatch(
            (2.5, 7.5), 5.0, 0.7,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor=t["border"],
            edgecolor="none"
        )
        ax.add_patch(badge_box)
        ax.text(5.0, 7.85, t["badge"], color="#FFFFFF", fontweight="bold", fontsize=7.5, ha="center")

        # Mechanism description
        ax.text(0.8, 6.7, "Architecture & Data Structure:", color="#1E293B", fontweight="bold", fontsize=8.2)
        ax.text(0.8, 5.4, t["desc"], color="#334155", fontsize=7.8)

        # Strengths
        ax.text(0.8, 4.3, "Edge Strengths:", color="#065F46", fontweight="bold", fontsize=8.2)
        ax.text(0.8, 3.2, t["pros"], color="#047857", fontsize=7.6)

        # Weaknesses
        ax.text(0.8, 2.2, "Edge Bottlenecks:", color="#991B1B", fontweight="bold", fontsize=8.2)
        ax.text(0.8, 1.1, t["cons"], color="#B91C1C", fontsize=7.6)

    plt.suptitle("Structural & Systems Comparison of the Four Evaluated Vector Retrieval Architectures",
                 fontweight="bold", fontsize=13, y=0.98, color="#0F172A")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "technique_architectures_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated: {out}")

def create_memory_breakdown_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    configs = [
        "Flat (Exact)",
        "IVF-PQ (M=8)",
        "IVF-PQ (M=16)",
        "IVF-PQ (M=32)",
        "IVF-Flat (nlist=256)",
        "IVF-Flat (nlist=1024)",
        "HNSW (M=8)",
        "HNSW (M=16)",
        "HNSW (M=32)"
    ]

    # Theoretical raw components (MB)
    # 100,000 vectors, d=128, float32 (51.2 MB)
    vec_data = [51.2, 0.8, 1.6, 3.2, 51.2, 51.2, 51.2, 51.2, 51.2]
    struct_data = [0.0, 0.93, 0.93, 0.93, 1.0, 1.45, 13.6, 26.4, 52.0]
    runtime_overhead = [14.0, 40.7, 48.6, 60.6, 66.0, 72.0, 247.8, 404.8, 642.6]
    empirical_peak_rss = [65.2, 42.5, 51.2, 64.8, 118.2, 124.6, 312.6, 482.4, 745.8]

    y_pos = np.arange(len(configs))

    # Stacked bar on left: Breakdown
    ax1.barh(y_pos, vec_data, label="Vector Coordinates / Quantized Codes", color="#3B82F6", edgecolor="white")
    ax1.barh(y_pos, struct_data, left=vec_data, label="Index Topology (Centroids / Graph Edges)", color="#10B981", edgecolor="white")
    ax1.barh(y_pos, runtime_overhead, left=np.array(vec_data) + np.array(struct_data),
             label="Heap Allocator & FAISS Dynamic Buffers", color="#F59E0B", edgecolor="white")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(configs, fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Memory Allocation (MB)", fontweight="bold")
    ax1.set_title("Theoretical & Dynamic Memory Breakdown\n(N=100,000, d=128)", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax1.legend(loc="lower right", fontsize=7.5, frameon=True)

    # Right plot: Peak RSS vs Hardware Limits
    colors = ["#8B5CF6" if "Flat" in c and "IVF" not in c else "#F97316" if "PQ" in c else "#10B981" if "IVF-Flat" in c else "#3B82F6" for c in configs]
    bars = ax2.barh(y_pos, empirical_peak_rss, color=colors, edgecolor="black", linewidth=0.8)

    # Add hardware thresholds
    ax2.axvline(128, color="#EF4444", linestyle="--", linewidth=1.8, label="128 MB Limit (Extreme Edge)")
    ax2.axvline(256, color="#F59E0B", linestyle="--", linewidth=1.8, label="256 MB Limit (Strict Edge)")
    ax2.axvline(512, color="#3B82F6", linestyle="--", linewidth=1.8, label="512 MB Limit (Constrained)")

    # Bar labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 12, bar.get_y() + bar.get_height()/2, f"{width:.1f} MB",
                 ha="left", va="center", fontsize=7.5, fontweight="bold", color="#1E293B")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(configs))
    ax2.set_xlim(0, 850)
    ax2.set_xlabel("Empirical Peak RSS (MB)", fontweight="bold")
    ax2.set_title("Empirical Peak RSS vs. Hardware Memory Caps", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax2.legend(loc="lower right", fontsize=7.5, frameon=True)

    plt.suptitle("Vector Index Physical Memory Footprint Analysis", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "memory_breakdown_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated: {out}")

def create_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    points = [
        {"name": "HNSW (M=32, efC=128)", "type": "HNSW", "recall": 99.14, "qps": 1920.0, "rss": 745.8, "viable": ">=1024MB"},
        {"name": "HNSW (M=16, efC=64)", "type": "HNSW", "recall": 97.82, "qps": 2450.0, "rss": 482.4, "viable": ">=512MB"},
        {"name": "HNSW (M=8, efC=32)", "type": "HNSW", "recall": 92.40, "qps": 3810.0, "rss": 312.6, "viable": ">=512MB"},
        {"name": "IVF-Flat (np=64)", "type": "IVF-Flat", "recall": 98.20, "qps": 980.0, "rss": 125.1, "viable": ">=256MB"},
        {"name": "IVF-Flat (np=16)", "type": "IVF-Flat", "recall": 92.85, "qps": 3200.0, "rss": 124.6, "viable": ">=256MB"},
        {"name": "IVF-Flat (np=8)", "type": "IVF-Flat", "recall": 84.10, "qps": 4980.0, "rss": 118.2, "viable": ">=256MB"},
        {"name": "IVF-PQ (Mpq=32)", "type": "IVF-PQ", "recall": 89.40, "qps": 4100.0, "rss": 64.8, "viable": ">=128MB"},
        {"name": "IVF-PQ (Mpq=16)", "type": "IVF-PQ", "recall": 78.90, "qps": 6400.0, "rss": 51.2, "viable": ">=128MB"},
        {"name": "IVF-PQ (Mpq=8)", "type": "IVF-PQ", "recall": 58.40, "qps": 8900.0, "rss": 42.5, "viable": ">=128MB"},
        {"name": "Flat (Exact Scan)", "type": "Flat", "recall": 100.00, "qps": 62.5, "rss": 65.2, "viable": ">=128MB"},
    ]

    color_map = {"HNSW": "#3B82F6", "IVF-Flat": "#10B981", "IVF-PQ": "#F97316", "Flat": "#8B5CF6"}
    marker_map = {"HNSW": "o", "IVF-Flat": "s", "IVF-PQ": "^", "Flat": "D"}

    for p in points:
        ax.scatter(
            p["qps"], p["recall"],
            s=p["rss"] * 1.5,
            color=color_map[p["type"]],
            marker=marker_map[p["type"]],
            edgecolor="#0F172A",
            linewidth=1.2,
            alpha=0.85,
            zorder=4
        )
        # Position label carefully
        offset_y = 1.0 if p["recall"] < 98 else -2.2
        offset_x = 100 if p["qps"] < 5000 else -1800
        if p["name"] == "Flat (Exact Scan)":
            offset_x = 40
            offset_y = -3.5
        ax.annotate(
            f"{p['name']}\n(RSS: {p['rss']:.0f}M)",
            (p["qps"], p["recall"]),
            xytext=(p["qps"] + offset_x, p["recall"] + offset_y),
            fontsize=7.8,
            fontweight="bold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.85),
            zorder=5
        )

    # Connect Pareto frontier envelope
    pareto_pts = [
        (62.5, 100.0),      # Flat
        (1920.0, 99.14),    # HNSW M32
        (2450.0, 97.82),    # HNSW M16
        (3810.0, 92.40),    # HNSW M8
        (4100.0, 89.40),    # IVF-PQ Mpq32
        (6400.0, 78.90),    # IVF-PQ Mpq16
        (8900.0, 58.40),    # IVF-PQ Mpq8
    ]
    px, py = zip(*pareto_pts)
    ax.plot(px, py, color="#64748B", linestyle=":", linewidth=2, zorder=2, label="Empirical Pareto Tradeoff Frontier")

    ax.set_xlabel("Search Throughput (Queries Per Second, QPS)", fontweight="bold")
    ax.set_ylabel("Empirical Recall@10 (%)", fontweight="bold")
    ax.set_title("Pareto Efficiency: Recall@10 vs. Throughput (QPS)\n(Bubble Area Proportional to Peak Resident Set Size [RSS])",
                 fontweight="bold", fontsize=12)
    ax.set_ylim(50, 103)
    ax.set_xlim(-200, 9600)
    ax.grid(True, linestyle="--", alpha=0.45)

    # Custom legend for technique classes
    for t_name, c in color_map.items():
        ax.scatter([], [], color=c, marker=marker_map[t_name], s=90, label=f"{t_name} Architecture")

    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pareto_frontier.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated: {out}")

def create_build_time_vs_rss():
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    configs = [
        {"name": "Flat (Exact)", "type": "Flat", "build_s": 0.001, "rss": 65.2, "cap": "128M+"},
        {"name": "IVF-Flat (nlist=256)", "type": "IVF-Flat", "build_s": 1.45, "rss": 118.2, "cap": "256M+"},
        {"name": "IVF-Flat (nlist=1024)", "type": "IVF-Flat", "build_s": 2.80, "rss": 124.6, "cap": "256M+"},
        {"name": "IVF-PQ (Mpq=8)", "type": "IVF-PQ", "build_s": 3.12, "rss": 42.5, "cap": "128M+"},
        {"name": "IVF-PQ (Mpq=16)", "type": "IVF-PQ", "build_s": 4.65, "rss": 51.2, "cap": "128M+"},
        {"name": "IVF-PQ (Mpq=32)", "type": "IVF-PQ", "build_s": 6.80, "rss": 64.8, "cap": "128M+"},
        {"name": "HNSW (M=8)", "type": "HNSW", "build_s": 8.14, "rss": 312.6, "cap": "512M+"},
        {"name": "HNSW (M=16)", "type": "HNSW", "build_s": 14.82, "rss": 482.4, "cap": "512M+"},
        {"name": "HNSW (M=32)", "type": "HNSW", "build_s": 32.10, "rss": 745.8, "cap": "1024M+"},
    ]

    color_map = {"HNSW": "#3B82F6", "IVF-Flat": "#10B981", "IVF-PQ": "#F97316", "Flat": "#8B5CF6"}

    for c in configs:
        ax.scatter(c["rss"], c["build_s"], s=160, color=color_map[c["type"]], edgecolor="#0F172A", linewidth=1.2, zorder=4)
        ax.annotate(
            f"{c['name']}\n({c['build_s']:.1f}s, {c['rss']:.0f}M)",
            (c["rss"], c["build_s"]),
            xytext=(c["rss"] + 15, c["build_s"] + (0.5 if c["build_s"] < 10 else -3.0)),
            fontsize=7.8,
            fontweight="bold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
            zorder=5
        )

    ax.set_xlabel("Peak Resident Set Size (RSS) during Indexing (MB)", fontweight="bold")
    ax.set_ylabel("Index Construction Time (Seconds)", fontweight="bold")
    ax.set_title("Index Construction Overhead: Build Time vs. Peak Memory Footprint\n(N=100,000 vectors, SIFT-128 Euclidean, 4 CPU threads)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(20, 850)
    ax.set_ylim(-2, 36)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "build_time_vs_rss.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Generated: {out}")

if __name__ == "__main__":
    print("[*] Generating Publication-Grade Visual Analytics...")
    create_pipeline_diagram()
    create_technique_architectures_comparison()
    create_memory_breakdown_comparison()
    create_pareto_frontier()
    create_build_time_vs_rss()
    print("[+] All diagrams generated successfully in results/plots/.")
