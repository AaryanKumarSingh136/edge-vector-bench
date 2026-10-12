#!/usr/bin/env python3
"""
single_run_eval.py
==================
Isolated Single-Index Evaluation Worker for Hardware-Enforced Vector Search.

Part of the empirical study:
"Analysis of Vector Database Indexing Techniques Under Hardware-Enforced Memory Constraints"
Authors: Aaryan Kumar Singh (24BCE1705) & Karan Gandhi (24BCE1709)
Track: Systems for Machine Learning (SysML) / Edge AI Data Infrastructure

Execution Model:
Invoked directly as an isolated process or inside a systemd-run cgroup v2 container.
Strictly controls thread pinning to eliminate cross-core CPU concurrency artifacts.
Measures index construction time, peak Resident Set Size (RSS), warm-up query execution,
per-query latency distributions (p50, p95, p99), and empirical Recall@10.

Outputs clean JSON telemetry as the final stdout line for automated harvesting.
"""

import os
import sys

# Hardcoded thread pinning BEFORE importing any BLAS, LAPACK, or FAISS libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import time
import json
import threading
import psutil
import h5py
import numpy as np
import faiss

# Redirect informational logging to stderr so stdout is reserved for telemetry
def log(msg):
    sys.stderr.write(f"[single_run_eval] {msg}\n")
    sys.stderr.flush()


class PeakMemoryTracker:
    """Active background thread monitoring RSS memory usage at high frequency."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_rss_bytes = 0
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self.peak_rss_bytes = self.process.memory_info().rss
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            final_rss = self.process.memory_info().rss
            if final_rss > self.peak_rss_bytes:
                self.peak_rss_bytes = final_rss
        except Exception:
            pass
        return self.peak_rss_bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-run vector index builder and performance benchmark worker under memory caps."
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        choices=["HNSW", "IVF-Flat", "IVF-PQ"],
        help="Target index architecture to instantiate.",
    )
    # HNSW Hyperparameters
    parser.add_argument("--M", type=int, default=16, help="HNSW: Number of bi-directional link connections per node.")
    parser.add_argument("--efConstruction", type=int, default=64, help="HNSW: Size of dynamic candidate list during construction.")
    parser.add_argument("--efSearch", type=int, default=64, help="HNSW: Size of dynamic candidate list during search.")

    # IVF-Flat / IVF-PQ Hyperparameters
    parser.add_argument("--nlist", type=int, default=256, help="IVF: Number of Voronoi partition centroids.")
    parser.add_argument("--nprobe", type=int, default=16, help="IVF: Number of closest centroids inspected per query.")

    # IVF-PQ Specific Hyperparameters
    parser.add_argument("--M_pq", type=int, default=16, help="IVF-PQ: Number of sub-vector quantization subspaces.")
    parser.add_argument("--nbits", type=int, default=8, help="IVF-PQ: Bits per sub-quantizer (default: 8, i.e., 256 centroids per subspace).")

    # Dataset & Evaluation Arguments
    parser.add_argument("--dataset_path", type=str, default="data/sift-128-100k.hdf5", help="Path to HDF5 benchmark archive.")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve per query.")
    parser.add_argument("--warmup_queries", type=int, default=100, help="Number of warmup queries to discard before timing.")
    parser.add_argument("--subset_size", type=int, default=100000, help="Number of base vectors to index.")

    return parser.parse_args()


def load_dataset(dataset_path, subset_size):
    """Load base vectors, evaluation query vectors, and ground-truth nearest neighbor indices."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Benchmark dataset archive not found at: {dataset_path}")

    log(f"Loading dataset from: {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        base_vectors = np.array(h5["train"][:subset_size], dtype=np.float32)
        query_vectors = np.array(h5["test"][:], dtype=np.float32)
        gt_neighbors = np.array(h5["neighbors"][:, :10], dtype=np.int32)

    # Ensure memory contiguity for optimized BLAS/SIMD memory loads
    base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    log(f"Dataset loaded: {base_vectors.shape[0]} base vectors, {query_vectors.shape[0]} queries, dim={base_vectors.shape[1]}")
    return base_vectors, query_vectors, gt_neighbors


def build_index(args, d, base_vectors, memory_tracker):
    """Construct and train the specified FAISS index, tracking build latency and memory footprint."""
    log(f"Instantiating index architecture: {args.index}")
    t_start = time.perf_counter()

    if args.index == "HNSW":
        # Hierarchical Navigable Small World Graph
        index = faiss.IndexHNSWFlat(d, args.M, faiss.METRIC_L2)
        index.hnsw.efConstruction = args.efConstruction
        index.hnsw.efSearch = args.efSearch
        log(f"Configured HNSW: M={args.M}, efConstruction={args.efConstruction}, efSearch={args.efSearch}")
        log(f"Populating HNSW graph with {len(base_vectors)} vectors...")
        index.add(base_vectors)

    elif args.index == "IVF-Flat":
        # Inverted File with Flat Voronoi Cells
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_L2)
        log(f"Training IVF-Flat quantizer with {args.nlist} centroids...")
        index.train(base_vectors)
        log(f"Adding vectors to inverted lists...")
        index.add(base_vectors)
        index.nprobe = args.nprobe
        log(f"Configured IVF-Flat: nlist={args.nlist}, nprobe={args.nprobe}")

    elif args.index == "IVF-PQ":
        # Inverted File with Product Quantization
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.M_pq, args.nbits, faiss.METRIC_L2)
        log(f"Training IVF-PQ quantizer and codebooks (nlist={args.nlist}, M_pq={args.M_pq}, nbits={args.nbits})...")
        index.train(base_vectors)
        log(f"Quantizing and assigning base vectors...")
        index.add(base_vectors)
        index.nprobe = args.nprobe
        log(f"Configured IVF-PQ: nlist={args.nlist}, nprobe={args.nprobe}, M_pq={args.M_pq}, nbits={args.nbits}")

    else:
        raise ValueError(f"Unsupported index architecture: {args.index}")

    build_time = time.perf_counter() - t_start
    log(f"Index build completed in {build_time:.4f} seconds.")
    return index, build_time


def evaluate_search(index, query_vectors, gt_neighbors, k, warmup_count):
    """
    Execute warm-up passes followed by per-query latency measurement and Recall@K calculation.
    """
    total_queries = query_vectors.shape[0]
    log(f"Executing {warmup_count} warmup queries...")
    if warmup_count > 0:
        warmup_slice = query_vectors[:warmup_count]
        _ = index.search(warmup_slice, k)

    log(f"Evaluating {total_queries} queries for K={k} with individual latency profiling...")
    latencies_ms = np.zeros(total_queries, dtype=np.float64)
    all_retrieved = np.zeros((total_queries, k), dtype=np.int64)

    # Perform individual query executions to profile exact tail latencies
    for i in range(total_queries):
        q = query_vectors[i:i+1]
        t0 = time.perf_counter_ns()
        _, retrieved_ids = index.search(q, k)
        t1 = time.perf_counter_ns()
        latencies_ms[i] = (t1 - t0) / 1e6
        all_retrieved[i] = retrieved_ids[0]

    # Calculate latency percentiles
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))
    mean_latency = float(np.mean(latencies_ms))
    total_search_time_s = float(np.sum(latencies_ms)) / 1000.0
    qps = float(total_queries / total_search_time_s) if total_search_time_s > 0 else 0.0

    # Calculate empirical Recall@K
    matches = 0
    total_possible = total_queries * k
    for i in range(total_queries):
        expected_set = set(gt_neighbors[i])
        actual_set = set(all_retrieved[i])
        matches += len(expected_set.intersection(actual_set))
    recall_at_k = float(matches / total_possible)

    log(f"Evaluation complete: Recall@{k}={recall_at_k*100:.2f}%, QPS={qps:.1f}, p50={p50_latency:.3f}ms, p99={p99_latency:.3f}ms")
    return {
        "recall_at_k": recall_at_k,
        "qps": qps,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "total_queries": total_queries,
    }


def main():
    args = parse_args()
    log("Initializing isolated worker telemetry tracker...")

    mem_tracker = PeakMemoryTracker(interval=0.005)
    mem_tracker.start()

    try:
        base_vectors, query_vectors, gt_neighbors = load_dataset(args.dataset_path, args.subset_size)
        d = base_vectors.shape[1]

        index, build_time_s = build_index(args, d, base_vectors, mem_tracker)
        metrics = evaluate_search(index, query_vectors, gt_neighbors, args.k, args.warmup_queries)

        peak_rss_bytes = mem_tracker.stop()
        peak_rss_mb = peak_rss_bytes / (1024 * 1024)
        log(f"Peak Resident Set Size (RSS): {peak_rss_mb:.2f} MB")

        telemetry = {
            "status": "SUCCESS",
            "index": args.index,
            "M": args.M if args.index == "HNSW" else None,
            "efConstruction": args.efConstruction if args.index == "HNSW" else None,
            "efSearch": args.efSearch if args.index == "HNSW" else None,
            "nlist": args.nlist if args.index in ["IVF-Flat", "IVF-PQ"] else None,
            "nprobe": args.nprobe if args.index in ["IVF-Flat", "IVF-PQ"] else None,
            "M_pq": args.M_pq if args.index == "IVF-PQ" else None,
            "nbits": args.nbits if args.index == "IVF-PQ" else None,
            "subset_size": args.subset_size,
            "dimension": d,
            "k": args.k,
            "build_time_s": round(build_time_s, 4),
            "peak_rss_mb": round(peak_rss_mb, 2),
            "recall_at_10": round(metrics["recall_at_k"], 4),
            "qps": round(metrics["qps"], 2),
            "mean_latency_ms": round(metrics["mean_latency_ms"], 4),
            "p50_latency_ms": round(metrics["p50_latency_ms"], 4),
            "p95_latency_ms": round(metrics["p95_latency_ms"], 4),
            "p99_latency_ms": round(metrics["p99_latency_ms"], 4),
            "exit_code": 0,
        }

    except Exception as exc:
        mem_tracker.stop()
        log(f"Runtime execution error: {exc}")
        telemetry = {
            "status": "ERROR",
            "index": args.index,
            "error_message": str(exc),
            "exit_code": 1,
        }
        # Print JSON error output on last line
        print(json.dumps(telemetry))
        sys.exit(1)

    # Output strictly valid JSON on the final stdout line
    print(json.dumps(telemetry))
    sys.exit(0)


if __name__ == "__main__":
    main()
