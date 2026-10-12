#!/usr/bin/env python3
"""
setup_environment.py
====================
Environment Verification and Benchmark Dataset Ingestion Pipeline.

Part of the empirical study:
"Analysis of Vector Database Indexing Techniques Under Hardware-Enforced Memory Constraints"
Authors: Aaryan Kumar Singh (24BCE1705) & Karan Gandhi (24BCE1709)
Track: Systems for Machine Learning (SysML) / Edge AI Data Infrastructure

Responsibilities:
1. Verify operating system prerequisites (Linux kernel, systemd-run cgroup v2 support, sudo access).
2. Validate storage capacity and directory topologies.
3. Automatically retrieve the canonical SIFT-128 Euclidean benchmark dataset (ann-benchmarks.com).
4. Perform structural and numerical validation of the HDF5 archive.
5. Extract an aligned 100k base vector evaluation slice with canonical ground-truth recalibration.
6. Execute an unconstrained FAISS IndexFlatL2 brute-force verification run to establish exact baseline metrics.
"""

import sys
import os
import platform
import shutil
import subprocess
import json
import time
import urllib.request
import h5py
import numpy as np

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

DATASET_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATA_DIR = os.path.abspath("data")
RAW_HDF5_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
PROCESSED_HDF5_PATH = os.path.join(DATA_DIR, "sift-128-100k.hdf5")
BASELINE_METRICS_PATH = os.path.join(DATA_DIR, "baseline_metrics.json")
RESULTS_DIR = os.path.abspath("results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

REQUIRED_PACKAGES = [
    ("faiss", "faiss-cpu"),
    ("h5py", "h5py"),
    ("numpy", "numpy"),
    ("psutil", "psutil"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


def print_banner():
    banner = """
================================================================================
  SysML Empirical Research Framework: Vector DB Memory Constraints
  Authors: Aaryan Kumar Singh (24BCE1705) & Karan Gandhi (24BCE1709)
  Environment Provisioning & Dataset Baseline Verification
================================================================================
"""
    print(banner)


def check_prerequisites():
    """Verify system architecture, Linux kernel, systemd-run, sudo privileges, and storage."""
    print("[*] Phase 1: Auditing Host System Prerequisites...")

    # OS Kernel Check
    current_os = platform.system()
    kernel_release = platform.release()
    print(f"    - Operating System Detected: {current_os} ({kernel_release})")
    if current_os != "Linux":
        print("    [!] WARNING: Target experiment execution requires Linux with systemd cgroups v2.")
        print("    [!] Environment verification will proceed, but run_matrix.py requires Linux host or VM.")
    else:
        print("    [+] Linux OS kernel verified.")

    # systemd-run binary check
    systemd_run_path = shutil.which("systemd-run")
    if systemd_run_path:
        print(f"    [+] systemd-run available at: {systemd_run_path}")
    else:
        print("    [!] WARNING: 'systemd-run' not found in PATH.")
        print("        Hardware-enforced cgroups v2 memory clamping will require systemd installed.")

    # Sudo / Root capability check
    is_root = (hasattr(os, "geteuid") and os.geteuid() == 0)
    if is_root:
        print("    [+] Execution environment has root privileges.")
    else:
        # Check if sudo can be invoked non-interactively
        try:
            res = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3)
            if res.returncode == 0:
                print("    [+] Sudo access available without password prompt.")
            else:
                print("    [*] Sudo password may be requested for kernel cache flushes (sync && drop_caches).")
        except Exception:
            print("    [*] Sudo validation deferred to execution time.")

    # Storage Capacity Check
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(DATA_DIR)
    free_gb = free_bytes / (1024 ** 3)
    print(f"    - Storage Capacity on target partition: {free_gb:.2f} GB free.")
    if free_gb < 2.0:
        raise RuntimeError(f"Insufficient storage space: {free_gb:.2f} GB available. Minimum 2.0 GB required.")
    print("    [+] Storage capacity satisfies empirical requirements (>2.0 GB).")

    # Python Dependencies Check
    print("[*] Auditing Python Runtime Dependencies...")
    for module_name, pkg_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
            print(f"    [+] Module '{module_name}' is installed and importable.")
        except ImportError:
            print(f"    [-] Missing dependency: '{module_name}' (Install via: pip install {pkg_name})")
            raise ImportError(f"Missing dependency: {pkg_name}. Execute: pip install -r requirements.txt")


class DownloadProgressBar:
    def __init__(self):
        self.last_update = 0
        self.start_time = time.time()

    def update(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        current_time = time.time()
        # Update progress at most every 0.25 seconds
        if current_time - self.last_update > 0.25 or downloaded >= total_size:
            self.last_update = current_time
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = current_time - self.start_time
            speed = mb_downloaded / elapsed if elapsed > 0 else 0
            bar_len = 35
            filled_len = int(bar_len * downloaded / total_size) if total_size > 0 else 0
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r    [{bar}] {percent:5.1f}% | {mb_downloaded:6.1f}/{mb_total:6.1f} MB | {speed:5.2f} MB/s", end="", flush=True)


def download_benchmark_dataset():
    """Download the canonical sift-128-euclidean dataset from ann-benchmarks repository with resume support."""
    import requests
    print(f"\n[*] Phase 2: Procuring SIFT-128 Euclidean Benchmark Archive...")
    print(f"    - Target URL: {DATASET_URL}")
    print(f"    - Destination: {RAW_HDF5_PATH}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Get remote total size
    try:
        head_resp = requests.head(DATASET_URL, headers=headers, timeout=15)
        head_resp.raise_for_status()
        total_size = int(head_resp.headers.get("content-length", 525128288))
    except Exception as e:
        print(f"    [!] Could not fetch head size ({e}), defaulting to 525,128,288 bytes.")
        total_size = 525128288

    if os.path.exists(RAW_HDF5_PATH):
        existing_size = os.path.getsize(RAW_HDF5_PATH)
        if existing_size >= total_size:
            print(f"    [+] Raw archive already complete ({existing_size / (1024*1024):.2f} MB). Skipping download.")
            return
        elif existing_size > 0:
            print(f"    [*] Found partial download ({existing_size / (1024*1024):.2f} MB). Resuming...")
    else:
        existing_size = 0

    max_retries = 25
    retry_count = 0
    start_time = time.time()

    while existing_size < total_size and retry_count < max_retries:
        retry_count += 1
        req_headers = dict(headers)
        if existing_size > 0:
            req_headers["Range"] = f"bytes={existing_size}-"

        try:
            with requests.get(DATASET_URL, headers=req_headers, stream=True, timeout=(15, 60)) as resp:
                if resp.status_code not in [200, 206]:
                    resp.raise_for_status()

                mode = "ab" if existing_size > 0 else "wb"
                chunk_size = 1024 * 1024  # 1 MB chunk
                last_update = 0

                with open(RAW_HDF5_PATH, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            existing_size += len(chunk)
                            curr = time.time()
                            if curr - last_update > 0.25 or existing_size >= total_size:
                                last_update = curr
                                pct = (existing_size / total_size) * 100 if total_size > 0 else 0
                                mb_down = existing_size / (1024 * 1024)
                                mb_tot = total_size / (1024 * 1024)
                                spd = (mb_down) / (curr - start_time) if curr > start_time else 0
                                bar_len = 35
                                filled = int(bar_len * existing_size / total_size) if total_size > 0 else 0
                                bar = "=" * filled + "-" * (bar_len - filled)
                                print(f"\r    [{bar}] {pct:5.1f}% | {mb_down:6.1f}/{mb_tot:6.1f} MB | {spd:5.2f} MB/s", end="", flush=True)

        except (requests.RequestException, Exception) as exc:
            if existing_size >= total_size:
                break
            print(f"\n    [!] Transient network interruption: {exc}. Retrying in 2 seconds (Attempt {retry_count}/{max_retries})...")
            time.sleep(2)
            if os.path.exists(RAW_HDF5_PATH):
                existing_size = os.path.getsize(RAW_HDF5_PATH)

    if existing_size < total_size:
        raise RuntimeError(f"Download incomplete after {max_retries} attempts: {existing_size}/{total_size} bytes downloaded.")

    print("\n    [+] Download verified and completed successfully.")


def validate_and_prepare_dataset(target_subset_size=100000):
    """
    Validate HDF5 structure and prepare the standardized 100k base vector benchmark slice.
    Calculates exact ground-truth nearest neighbors for the 100k slice to guarantee 100% calibration.
    """
    print("\n[*] Phase 3: Validating HDF5 Dataset Integrity and Slicing Benchmark Split...")
    import faiss

    with h5py.File(RAW_HDF5_PATH, "r") as raw_h5:
        expected_keys = ["train", "test", "neighbors", "distances"]
        for key in expected_keys:
            if key not in raw_h5:
                raise ValueError(f"Malformed HDF5: Required dataset '{key}' not found in {RAW_HDF5_PATH}")

        raw_train_shape = raw_h5["train"].shape
        raw_test_shape = raw_h5["test"].shape
        dim = raw_train_shape[1]

        print(f"    - Raw Training Base Vectors: {raw_train_shape[0]:,} vectors, {dim} dimensions (dtype: {raw_h5['train'].dtype})")
        print(f"    - Evaluation Query Vectors:  {raw_test_shape[0]:,} vectors, {dim} dimensions (dtype: {raw_h5['test'].dtype})")

        num_base = min(target_subset_size, raw_train_shape[0])
        print(f"    - Slicing standardized evaluation base: {num_base:,} vectors.")

        # Load raw base slice and query set into memory
        base_vectors = np.array(raw_h5["train"][:num_base], dtype=np.float32)
        query_vectors = np.array(raw_h5["test"][:], dtype=np.float32)

    # Compute deterministic ground-truth nearest neighbors (K=100) for the 100k slice
    print(f"    - Computing calibrated exact ground-truth nearest neighbors for {num_base:,} base vectors...")
    t0 = time.perf_counter()
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(base_vectors)
    k_gt = 100
    gt_distances, gt_neighbors = exact_index.search(query_vectors, k_gt)
    calibration_time = time.perf_counter() - t0
    print(f"    [+] Exact ground-truth computed in {calibration_time:.2f} seconds.")

    # Save to standardized evaluation file
    print(f"    - Writing structured benchmark archive to: {PROCESSED_HDF5_PATH}")
    with h5py.File(PROCESSED_HDF5_PATH, "w") as out_h5:
        out_h5.create_dataset("train", data=base_vectors, compression="gzip", compression_opts=4)
        out_h5.create_dataset("test", data=query_vectors, compression="gzip", compression_opts=4)
        out_h5.create_dataset("neighbors", data=gt_neighbors, compression="gzip", compression_opts=4)
        out_h5.create_dataset("distances", data=gt_distances, compression="gzip", compression_opts=4)

    file_size_mb = os.path.getsize(PROCESSED_HDF5_PATH) / (1024 * 1024)
    print(f"    [+] Standardized evaluation dataset generated ({file_size_mb:.2f} MB).")
    return PROCESSED_HDF5_PATH


def execute_exact_baseline(dataset_path):
    """
    Run unconstrained brute-force IndexFlatL2 search to verify 100% recall calibration
    and record foundational reference metrics.
    """
    print("\n[*] Phase 4: Executing Brute-Force IndexFlatL2 Verification Baseline...")
    import faiss

    with h5py.File(dataset_path, "r") as h5:
        base_vectors = np.array(h5["train"][:], dtype=np.float32)
        query_vectors = np.array(h5["test"][:], dtype=np.float32)
        gt_neighbors = np.array(h5["neighbors"][:, :10], dtype=np.int32)

    num_base, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]

    # Index creation & population
    t_build_start = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(base_vectors)
    build_time_s = time.perf_counter() - t_build_start

    # Query evaluation
    t_search_start = time.perf_counter()
    distances, retrieved_indices = index.search(query_vectors, 10)
    total_search_time_s = time.perf_counter() - t_search_start

    # Compute Recall@10 relative to ground truth
    matches = 0
    total_possible = num_queries * 10
    for i in range(num_queries):
        expected_set = set(gt_neighbors[i])
        actual_set = set(retrieved_indices[i])
        matches += len(expected_set.intersection(actual_set))
    recall_at_10 = matches / float(total_possible)

    qps = num_queries / total_search_time_s
    avg_latency_ms = (total_search_time_s / num_queries) * 1000.0

    print(f"    - Index Type:              FAISS IndexFlatL2 (Exact Brute-Force)")
    print(f"    - Base Vectors Indexed:    {num_base:,}")
    print(f"    - Index Build Time:        {build_time_s:.4f} s")
    print(f"    - Queries Evaluated:       {num_queries:,}")
    print(f"    - Total Search Time:       {total_search_time_s:.4f} s")
    print(f"    - Search Throughput (QPS): {qps:.2f} queries/sec")
    print(f"    - Mean Query Latency:      {avg_latency_ms:.4f} ms")
    print(f"    - Calibrated Recall@10:    {recall_at_10 * 100:.2f}%")

    if abs(recall_at_10 - 1.0) > 1e-6:
        raise ValueError(f"Ground-truth calibration failed! Expected Recall@10 = 1.0, obtained {recall_at_10}")

    print("    [+] Ground-truth calibration verified: 100.00% exact match.")

    baseline_metrics = {
        "index_type": "IndexFlatL2",
        "dataset": "SIFT-128-Euclidean-100k",
        "num_base_vectors": num_base,
        "num_query_vectors": num_queries,
        "dimension": dim,
        "build_time_seconds": round(build_time_s, 4),
        "total_search_time_seconds": round(total_search_time_s, 4),
        "queries_per_second": round(qps, 2),
        "mean_latency_ms": round(avg_latency_ms, 4),
        "recall_at_10": recall_at_10,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"    [+] Reference baseline telemetry written to: {BASELINE_METRICS_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Environment verification and dataset ingestion.")
    parser.add_argument("--run_baseline", action="store_true", help="Execute exact baseline test after download.")
    args, _ = parser.parse_known_args()

    print_banner()
    try:
        check_prerequisites()
        download_benchmark_dataset()
        dataset_path = validate_and_prepare_dataset(target_subset_size=100000)
        if args.run_baseline:
            execute_exact_baseline(dataset_path)
        else:
            print(f"\n[+] Real dataset verified and preserved at: {dataset_path}")
            print("[+] Tests deferred as requested. Run later with: python single_run_eval.py or python run_matrix.py\n")
    except Exception as exc:
        print(f"\n[FATAL ERROR] Environment setup failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
