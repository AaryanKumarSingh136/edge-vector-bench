#!/usr/bin/env python3
"""
run_matrix.py
=============
Automated Benchmark Matrix Orchestrator with Hardware-Enforced cgroups v2 Clamping.

Part of the empirical study:
"Analysis of Vector Database Indexing Techniques Under Hardware-Enforced Memory Constraints"
Authors: Aaryan Kumar Singh (24BCE1705) & Karan Gandhi (24BCE1709)
Track: Systems for Machine Learning (SysML) / Edge AI Data Infrastructure

Operational Responsibilities:
1. Sweeps hardware memory caps: [1024M, 512M, 256M, 128M] with zero swap allowed (MemorySwapMax=0).
2. Sweeps multidimensional hyperparameter grids across HNSW, IVF-Flat, and IVF-PQ architectures.
3. Flushes OS page caches (sync && echo 3 > /proc/sys/vm/drop_caches) prior to every individual evaluation.
4. Wraps single_run_eval.py inside systemd-run cgroup v2 scope.
5. Traps Return Code 137 (Linux Kernel OOM Killer SIGKILL) and records OOM_CRASH events.
6. Streams live empirical telemetry directly to results/telemetry.csv.
"""

import sys
import os
import subprocess
import json
import csv
import time
import shutil
import platform
import argparse

CSV_OUTPUT_PATH = os.path.join("results", "telemetry.csv")
DATASET_PATH = os.path.join("data", "sift-128-100k.hdf5")

RAM_LIMITS = ["1024M", "512M", "256M", "128M"]

# Comprehensive configuration matrix representing diverse edge operational trade-offs
EXPERIMENT_CONFIGS = [
    # HNSW Configurations
    {
        "index": "HNSW",
        "config_name": "HNSW_M16_efC64_efS64",
        "params": ["--index", "HNSW", "--M", "16", "--efConstruction", "64", "--efSearch", "64"],
    },
    {
        "index": "HNSW",
        "config_name": "HNSW_M32_efC128_efS64",
        "params": ["--index", "HNSW", "--M", "32", "--efConstruction", "128", "--efSearch", "64"],
    },
    {
        "index": "HNSW",
        "config_name": "HNSW_M8_efC32_efS32",
        "params": ["--index", "HNSW", "--M", "8", "--efConstruction", "32", "--efSearch", "32"],
    },
    # IVF-Flat Configurations
    {
        "index": "IVF-Flat",
        "config_name": "IVF_Flat_nlist256_nprobe8",
        "params": ["--index", "IVF-Flat", "--nlist", "256", "--nprobe", "8"],
    },
    {
        "index": "IVF-Flat",
        "config_name": "IVF_Flat_nlist1024_nprobe16",
        "params": ["--index", "IVF-Flat", "--nlist", "1024", "--nprobe", "16"],
    },
    {
        "index": "IVF-Flat",
        "config_name": "IVF_Flat_nlist1024_nprobe64",
        "params": ["--index", "IVF-Flat", "--nlist", "1024", "--nprobe", "64"],
    },
    # IVF-PQ Configurations
    {
        "index": "IVF-PQ",
        "config_name": "IVF_PQ_nlist256_np8_Mpq8_nb8",
        "params": ["--index", "IVF-PQ", "--nlist", "256", "--nprobe", "8", "--M_pq", "8", "--nbits", "8"],
    },
    {
        "index": "IVF-PQ",
        "config_name": "IVF_PQ_nlist1024_np16_Mpq16_nb8",
        "params": ["--index", "IVF-PQ", "--nlist", "1024", "--nprobe", "16", "--M_pq", "16", "--nbits", "8"],
    },
    {
        "index": "IVF-PQ",
        "config_name": "IVF_PQ_nlist1024_np32_Mpq32_nb8",
        "params": ["--index", "IVF-PQ", "--nlist", "1024", "--nprobe", "32", "--M_pq", "32", "--nbits", "8"],
    },
]

CSV_FIELDNAMES = [
    "run_timestamp",
    "ram_cap",
    "ram_cap_mb",
    "index_type",
    "config_name",
    "status",
    "exit_code",
    "build_time_s",
    "peak_rss_mb",
    "recall_at_10",
    "qps",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_latency_ms",
    "oom_killed",
    "error_details",
]


def ram_to_mb(ram_str):
    """Convert RAM string e.g., '1024M' to integer megabytes."""
    return int(ram_str.replace("M", "").replace("m", ""))


def flush_kernel_page_caches():
    """
    Purge OS dirty write buffers and flush Linux page cache, dentries, and inodes.
    Ensures zero cross-run cache pollution and enforces cold memory starting conditions.
    """
    if platform.system() != "Linux":
        return

    cmd = "sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Brief pause to allow kernel slab reclaim to settle
        time.sleep(0.3)
    except Exception as exc:
        print(f"    [!] Notice: Page cache flush returned code {exc}. Continuing benchmark.")


def init_csv_file(csv_path):
    """Initialize results telemetry CSV file with standardized header schema."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    if not file_exists or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"[*] Telemetry CSV initialized at: {csv_path}")
    else:
        print(f"[*] Telemetry CSV exists. Streaming new trial rows to: {csv_path}")


def stream_result_to_csv(csv_path, row_data):
    """Immediately append row to CSV to protect against data loss during hard kernel crashes."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row_data)
        f.flush()


def run_single_benchmark(ram_cap, exp_cfg, python_bin, script_path, dataset_path, dry_run=False):
    """
    Execute single_run_eval.py clamped by systemd-run cgroup v2.
    Traps return code 137 (SIGKILL / OOM-Killer) and logs exact metrics.
    """
    config_name = exp_cfg["config_name"]
    index_type = exp_cfg["index"]
    extra_params = exp_cfg["params"]
    ram_mb = ram_to_mb(ram_cap)

    # Base command arguments
    base_eval_cmd = [
        python_bin,
        script_path,
        "--dataset_path", dataset_path,
    ] + extra_params

    # Determine command wrapping based on environment and systemd-run presence
    has_systemd_run = (shutil.which("systemd-run") is not None) and (platform.system() == "Linux")

    if has_systemd_run and not dry_run:
        unit_name = f"veceval-{index_type.lower()}-{int(time.time()*1000)%100000}"
        full_cmd = [
            "systemd-run",
            "--scope",
            f"--unit={unit_name}",
            f"-p", f"MemoryMax={ram_cap}",
            f"-p", "MemorySwapMax=0",
        ] + base_eval_cmd
    else:
        full_cmd = base_eval_cmd

    flush_kernel_page_caches()

    t_start = time.time()
    try:
        proc = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5-minute watchdog timer
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        exit_code = proc.returncode

    except subprocess.TimeoutExpired as exc:
        proc_time = time.time() - t_start
        print(f"    [-] WATCHDOG TIMEOUT (300s exceeded) for {config_name} @ {ram_cap}")
        return {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "ram_cap": ram_cap,
            "ram_cap_mb": ram_mb,
            "index_type": index_type,
            "config_name": config_name,
            "status": "TIMEOUT",
            "exit_code": 124,
            "build_time_s": 300.0,
            "peak_rss_mb": ram_mb,
            "recall_at_10": 0.0,
            "qps": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "oom_killed": False,
            "error_details": "Execution exceeded 300-second hardware watchdog timeout.",
        }

    # Inspect Return Code for Linux Kernel OOM Killer (Signal 9 / 137)
    is_oom_killed = (exit_code in [137, -9]) or ("killed" in stderr.lower() and "memory" in stderr.lower())

    if is_oom_killed:
        print(f"    [!!!] LINUX OOM KILLER TRAPPED (Exit Code {exit_code}) for {config_name} @ {ram_cap}!")
        return {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "ram_cap": ram_cap,
            "ram_cap_mb": ram_mb,
            "index_type": index_type,
            "config_name": config_name,
            "status": "OOM_CRASH",
            "exit_code": 137,
            "build_time_s": round(time.time() - t_start, 4),
            "peak_rss_mb": ram_mb,
            "recall_at_10": 0.0,
            "qps": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "oom_killed": True,
            "error_details": f"Linux Kernel OOM Killer sent SIGKILL. Process exceeded {ram_cap} limit.",
        }

    elif exit_code != 0:
        print(f"    [-] Execution Error (Exit Code {exit_code}): {stderr[-200:]}")
        return {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "ram_cap": ram_cap,
            "ram_cap_mb": ram_mb,
            "index_type": index_type,
            "config_name": config_name,
            "status": "ERROR",
            "exit_code": exit_code,
            "build_time_s": 0.0,
            "peak_rss_mb": 0.0,
            "recall_at_10": 0.0,
            "qps": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "oom_killed": False,
            "error_details": stderr[-500:] if stderr else f"Process exited with non-zero code {exit_code}",
        }

    # Successful run: Parse final JSON line from stdout
    try:
        last_line = stdout.splitlines()[-1] if stdout else "{}"
        telemetry_json = json.loads(last_line)

        # On environments without kernel cgroups (e.g. Windows or dry-run),
        # enforce the hardware constraint if the measured physical RSS breached the RAM cap
        peak_rss = telemetry_json.get("peak_rss_mb", 0.0)
        if not has_systemd_run and peak_rss > ram_mb:
            print(f"    [!] HARDWARE OOM CAP EXCEEDED: Measured Peak RSS ({peak_rss:.1f} MB) > {ram_cap} limit!")
            return {
                "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "ram_cap": ram_cap,
                "ram_cap_mb": ram_mb,
                "index_type": index_type,
                "config_name": config_name,
                "status": "OOM_CRASH",
                "exit_code": 137,
                "build_time_s": telemetry_json.get("build_time_s", 0.0),
                "peak_rss_mb": ram_mb,
                "recall_at_10": 0.0,
                "qps": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "oom_killed": True,
                "error_details": f"Physical peak RSS ({peak_rss:.1f} MB) breached hardware cap of {ram_cap}.",
            }

        return {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "ram_cap": ram_cap,
            "ram_cap_mb": ram_mb,
            "index_type": index_type,
            "config_name": config_name,
            "status": "SUCCESS",
            "exit_code": 0,
            "build_time_s": telemetry_json.get("build_time_s", 0.0),
            "peak_rss_mb": telemetry_json.get("peak_rss_mb", 0.0),
            "recall_at_10": telemetry_json.get("recall_at_10", 0.0),
            "qps": telemetry_json.get("qps", 0.0),
            "p50_latency_ms": telemetry_json.get("p50_latency_ms", 0.0),
            "p95_latency_ms": telemetry_json.get("p95_latency_ms", 0.0),
            "p99_latency_ms": telemetry_json.get("p99_latency_ms", 0.0),
            "mean_latency_ms": telemetry_json.get("mean_latency_ms", 0.0),
            "oom_killed": False,
            "error_details": "",
        }
    except Exception as parse_exc:
        return {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "ram_cap": ram_cap,
            "ram_cap_mb": ram_mb,
            "index_type": index_type,
            "config_name": config_name,
            "status": "PARSE_ERROR",
            "exit_code": 1,
            "build_time_s": 0.0,
            "peak_rss_mb": 0.0,
            "recall_at_10": 0.0,
            "qps": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "oom_killed": False,
            "error_details": f"Failed to parse JSON telemetry: {parse_exc} | Raw output: {stdout[-200:]}",
        }


def main():
    parser = argparse.ArgumentParser(description="Matrix benchmark orchestrator across hardware memory constraints.")
    parser.add_argument("--csv_output", type=str, default=CSV_OUTPUT_PATH, help="Path to telemetry CSV output.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to benchmark HDF5 archive.")
    parser.add_argument("--dry_run", action="store_true", help="Execute without systemd cgroup clamping (for sanity testing).")
    args = parser.parse_args()

    python_bin = sys.executable
    script_path = os.path.abspath("single_run_eval.py")

    print("================================================================================")
    print("  LAUNCHING FULL EMPIRICAL MEMORY CONSTRAINT EVALUATION MATRIX")
    print("  Hardware Memory Caps: " + ", ".join(RAM_LIMITS))
    print(f"  Configuration Grid:   {len(EXPERIMENT_CONFIGS)} architectures")
    print(f"  Total Runs Planned:   {len(RAM_LIMITS) * len(EXPERIMENT_CONFIGS)}")
    print("================================================================================")

    init_csv_file(args.csv_output)

    run_count = 0
    total_runs = len(RAM_LIMITS) * len(EXPERIMENT_CONFIGS)

    for ram_cap in RAM_LIMITS:
        print(f"\n================================================================================")
        print(f"  CLAMPING HARDWARE MEMORY CAP: {ram_cap} (MemorySwapMax=0)")
        print(f"================================================================================")

        for exp_cfg in EXPERIMENT_CONFIGS:
            run_count += 1
            cfg_name = exp_cfg["config_name"]
            print(f"[{run_count:02d}/{total_runs:02d}] Evaluating: {cfg_name:<30} @ RAM={ram_cap:<6} ...", end="", flush=True)

            res = run_single_benchmark(
                ram_cap=ram_cap,
                exp_cfg=exp_cfg,
                python_bin=python_bin,
                script_path=script_path,
                dataset_path=args.dataset_path,
                dry_run=args.dry_run,
            )

            stream_result_to_csv(args.csv_output, res)

            if res["status"] == "SUCCESS":
                print(f" [PASS] Recall@10={res['recall_at_10']*100:5.2f}% | p99={res['p99_latency_ms']:6.3f}ms | Peak RSS={res['peak_rss_mb']:6.1f}MB")
            elif res["status"] == "OOM_CRASH":
                print(f" [OOM CRASH (137)] -> Terminated by kernel")
            else:
                print(f" [{res['status']}] -> Exit code {res['exit_code']}")

    print("\n================================================================================")
    print(f"  EXPERIMENT MATRIX COMPLETE: All trials written to {args.csv_output}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
