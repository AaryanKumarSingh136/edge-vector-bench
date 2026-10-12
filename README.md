# Edge-Vector-Bench: Hardware-Enforced Vector Search Benchmark Suite

[![SysML: Systems for Machine Learning](https://img.shields.io/badge/SysML-Edge%20Vector%20Search-blue.svg)](#)
[![Hardware Enforcer: Linux cgroups v2](https://img.shields.io/badge/Kernel-cgroups%20v2%20MemoryMax-red.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical systems-level research and benchmarking framework for evaluating high-dimensional vector search indexing structures (**HNSW**, **IVF-Flat**, and **IVF-PQ**) under strict, hardware-enforced volatile memory limits.

---

## 1. Overview

Traditional approximate nearest neighbor (ANN) benchmarks evaluate vector search throughput (QPS) and Recall@K under unconstrained memory conditions on large server clusters. However, embedded edge AI systems (e.g., NVIDIA Jetson, Raspberry Pi, robotics, IoT gateways) operate under severe memory constraints where volatile RAM is strictly partitioned between OS kernels, vision pipelines, and deep neural network weights.

When vector indexing processes saturate physical memory on swap-disabled edge systems, the operating system kernel invokes the **Out-of-Memory (OOM) Killer**, terminating retrieval processes with **Signal 137 (`SIGKILL`)**.

`Edge-Vector-Bench` provides a production-grade test harness that utilizes **Linux Control Groups v2 (`cgroups v2`)** with zero swap (`MemorySwapMax=0`) and deterministic cold page-cache purging (`drop_caches`) to map the exact physical memory boundaries and performance cliffs of vector retrieval indices.

---

## 2. Architecture & Pipeline

```
+--------------------------------------------------------------------------------------------------+
|                    FOUR-STAGE HARDWARE-ENFORCED BENCHMARK PIPELINE                               |
+--------------------------------------------------------------------------------------------------+

  STAGE 1: ENVIRONMENT ISOLATION & CPU PINNING
  - Swap Elimination:           sudo swapoff -a
  - Thread Concurrency Anchor:  OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4
  - Kernel Page Cache Eviction: sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
                                |
                                v
  STAGE 2: HARDWARE-ENFORCED CGROUPS V2 CLAMPING
  - Tool: systemd-run --scope
  - Memory Quota:               -p MemoryMax={1024M, 512M, 256M, 128M}
  - Swap Prohibition:           -p MemorySwapMax=0
                                |
                                v
  STAGE 3: ISOLATED WORKER EVALUATION (single_run_eval.py)
  - Dataset Load:               SIFT-128 Euclidean benchmark
  - Memory Profiler:            High-frequency background RSS monitoring thread (5ms interval)
  - Index Construction:         Measure Build Time (s) and Peak RSS (MB)
  - Warmup Phase:               100 warm-up queries (discarded from metrics)
  - Query Profiling:            10,000 queries @ K=10 (per-query nanosecond timing)
  - Metric Derivation:          Recall@10, QPS, p50, p95, p99 search latency
                                |
                                v
  STAGE 4: KERNEL SIGNAL TRAPPER & TELEMETRY STREAMER (run_matrix.py)
  - Normal Exit (Code 0):       Harvest JSON telemetry -> Append to results/telemetry.csv
  - Kernel Kill (Code 137):     Trap SIGKILL -> Log status as OOM_CRASH -> Record failure boundary
  - Post-Run Flush:             Re-evict page caches before next grid configuration
```

---

## 3. Repository Structure

```
├── setup_environment.py     # Environment check, SIFT-128 download, slice preparation
├── single_run_eval.py       # Isolated worker, thread pinning, peak RSS, p50/p95/p99 profiling
├── run_matrix.py            # Matrix sweeper, cgroups v2 clamping, drop_caches, OOM trapper
├── generate_charts.py       # Publication-grade chart generation & survival analysis
├── requirements.txt         # Core dependencies
├── LICENSE                  # MIT License
└── README.md                # System documentation
```

---

## 4. Getting Started

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AaryanKumarSingh136/edge-vector-bench.git
cd edge-vector-bench
pip install -r requirements.txt
```

### Dataset Ingestion

Download the canonical SIFT-128 Euclidean benchmark dataset and extract the standardized evaluation slice:

```bash
python setup_environment.py
```

To also run an unconstrained brute-force exact baseline check:
```bash
python setup_environment.py --run_baseline
```

---

## 5. Running Evaluations

### Single Index Run
Execute an isolated benchmark run for a specific architecture with custom hyperparameters:

```bash
# Example: Evaluate HNSW (M=16, efConstruction=64, efSearch=64)
python single_run_eval.py --index HNSW --M 16 --efConstruction 64 --efSearch 64

# Example: Evaluate IVF-Flat (nlist=1024, nprobe=16)
python single_run_eval.py --index IVF-Flat --nlist 1024 --nprobe 16

# Example: Evaluate IVF-PQ (nlist=1024, nprobe=16, M_pq=16, nbits=8)
python single_run_eval.py --index IVF-PQ --nlist 1024 --nprobe 16 --M_pq 16 --nbits 8
```

### Full Hardware Constraint Sweep
To sweep memory limits (1024MB, 512MB, 256MB, 128MB) across the full hyperparameter grid:

```bash
# On a Linux host with systemd and root/sudo privileges:
sudo python run_matrix.py

# On local development or non-cgroup test environments:
python run_matrix.py --dry_run
```

Results are streamed live to `results/telemetry.csv`.

### Generating Visualizations

Generate publication-grade latency, recall, and OOM crash boundary plots:

```bash
python generate_charts.py
```

Outputs will be saved in `results/plots/`.

---

## 6. Supported Index Architectures

1. **HNSW (Hierarchical Navigable Small World):** Graph-based approximate nearest neighbor search with multi-layer skip-list hierarchy.
2. **IVF-Flat (Inverted File Flat):** Voronoi cell coarse quantization with uncompressed vector coordinates in inverted lists.
3. **IVF-PQ (Inverted File with Product Quantization):** Coarse Voronoi quantization combined with orthogonal subspace product quantization for high compression ratios.

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
