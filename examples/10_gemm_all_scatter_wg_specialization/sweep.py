#!/usr/bin/env python3
"""
Sweep script for GEMM all scatter workgroup specialization benchmarks with L2 cache analysis.

This script runs the benchmark across different matrix dimensions, profiles with rocprof,
analyzes L2 hit rates and performance, and generates visualizations and logs.
"""

import subprocess
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import the analyze function from the existing script
sys.path.insert(0, str(Path(__file__).parent))
from analyze_l2_hit_rate import parse_csv, calculate_hit_rate
import statistics


def run_benchmark(
    m: int,
    n: int,
    k: int,
    kernel_variant: str,
    script_dir: Path,
    num_ranks: int = 2,
    gemm_sms: int = None
) -> Tuple[float, float, float, float, str]:
    """
    Run a single benchmark configuration and extract L2 hit rate and performance.
    
    Args:
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Common dimension between matrices A and B
        kernel_variant: "baseline" or "spatial"
        script_dir: Directory containing the scripts
        num_ranks: Number of ranks/processes
        gemm_sms: Number of SMs for GEMM (None for auto-detect)
        
    Returns:
        Tuple of (avg_l2_hit_rate, std_l2_hit_rate, tflops, total_ms, output_json_path)
    """
    # Build command
    output_json = script_dir / f"log_m{m}_n{n}_k{k}_{kernel_variant}.json"
    cmd = [
        "./rocprof.sh", "python", "benchmark.py",
        "-m", str(m),
        "-n", str(n),
        "-k", str(k),
        "--datatype", "bf16",
        "--kernel_variant", kernel_variant,
        "--output_file", str(output_json),
        # "-v",  # validation
        "-b",  # benchmark
        "-r", str(num_ranks)
    ]
    
    # Add gemm_sms if specified
    if gemm_sms is not None:
        cmd.extend(["--gemm_sms", str(gemm_sms)])
    
    # Run benchmark with profiling
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=script_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return (0.0, 0.0, 0.0, 0.0, str(output_json))
    
    # Read benchmark results from JSON
    if not output_json.exists():
        print(f"Error: Benchmark results JSON not found at {output_json}")
        return (0.0, 0.0, 0.0, 0.0, str(output_json))
    
    with open(output_json, 'r') as f:
        bench_results = json.load(f)
    
    # Extract performance metrics
    tflops = bench_results.get("tflops", 0.0)
    total_ms = bench_results.get("total_ms", 0.0)
    
    # Parse the profiling output for L2 cache hit rates
    csv_path = script_dir / "profile_counter_collection.csv"
    if not csv_path.exists():
        print(f"Error: Profile output not found at {csv_path}")
        return (0.0, 0.0, tflops, total_ms, str(output_json))
    
    # Determine which kernel to analyze based on variant
    if kernel_variant == "spatial":
        kernel_name = "persistent_gemm_all_scatter_wg_specialization_spatial"
    else:
        kernel_name = "persistent_gemm_all_scatter_wg_specialization"
    
    kernel_names = [kernel_name]
    
    # Parse CSV
    kernel_hit_miss = parse_csv(csv_path, kernel_names)
    
    # Calculate hit rates
    all_hit_rates = []
    for kname in kernel_names:
        hit_miss_pairs = kernel_hit_miss.get(kname, [])
        if not hit_miss_pairs:
            continue
        
        for hits, misses in hit_miss_pairs:
            hit_rate = calculate_hit_rate(hits, misses)
            all_hit_rates.append(hit_rate)
    
    if not all_hit_rates:
        print(f"Warning: No hit rate data found for {kernel_names}")
        return (0.0, 0.0, tflops, total_ms, str(output_json))
    
    avg_hit_rate = statistics.mean(all_hit_rates)
    std_hit_rate = statistics.stdev(all_hit_rates) if len(all_hit_rates) > 1 else 0.0
    
    return (avg_hit_rate, std_hit_rate, tflops, total_ms, str(output_json))


def load_matrix_configs(csv_path: Path) -> List[Dict[str, int]]:
    """
    Load matrix configurations from a CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: m, n, k
        
    Returns:
        List of dictionaries with keys 'm', 'n', 'k'
    """
    configs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append({
                'm': int(row['m']),
                'n': int(row['n']),
                'k': int(row['k'])
            })
    return configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sweep GEMM performance and L2 cache for different matrix dimensions')
    parser.add_argument('-r', '--num-ranks', type=int, default=2,
                        help='Number of ranks/processes (default: 2)')
    parser.add_argument('--gemm-sms', type=int, default=None,
                        help='Number of SMs for GEMM workgroup specialization (default: auto-detect)')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to CSV file with matrix configurations (columns: m, n, k)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Load matrix configurations from CSV
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    matrix_configs = load_matrix_configs(config_path)
    datatype = "bf16"
    num_ranks = args.num_ranks
    gemm_sms = args.gemm_sms
    
    # Results storage
    results = []
    
    # Sweep configurations
    configs = [
        {"kernel_variant": "baseline", "label": "Baseline", "color": "#1f77b4"},
        {"kernel_variant": "spatial", "label": "Spatial Fusion", "color": "#ff7f0e"},
    ]
    
    print("="*70)
    print("Starting GEMM All Scatter Workgroup Specialization Sweep")
    print("="*70)
    print(f"Config file: {config_path}")
    print(f"Number of matrix configurations: {len(matrix_configs)}")
    print(f"Datatype: {datatype}")
    print(f"Num ranks: {num_ranks}")
    print(f"GEMM SMs: {gemm_sms if gemm_sms is not None else 'auto-detect'}")
    print("="*70 + "\n")
    
    # Run sweeps
    for config in configs:
        kernel_variant = config["kernel_variant"]
        label = config["label"]
        
        print(f"\n{'='*70}")
        print(f"Running sweep: {label}")
        print(f"{'='*70}\n")
        
        for mat_config in matrix_configs:
            m = mat_config['m']
            n = mat_config['n']
            k = mat_config['k']
            
            print(f"\nConfiguration: {label}, M={m}, N={n}, K={k}")
            print("-" * 50)
            
            avg_hit_rate, std_hit_rate, tflops, total_ms, output_json = run_benchmark(
                m=m,
                n=n,
                k=k,
                kernel_variant=kernel_variant,
                script_dir=script_dir,
                num_ranks=num_ranks,
                gemm_sms=gemm_sms
            )
            
            result = {
                "approach": label,
                "kernel_variant": kernel_variant,
                "m": m,
                "n": n,
                "k": k,
                "datatype": datatype,
                "num_ranks": num_ranks,
                "gemm_sms": gemm_sms,
                "avg_l2_hit_rate": avg_hit_rate,
                "std_l2_hit_rate": std_hit_rate,
                "tflops": tflops,
                "total_ms": total_ms,
                "output_json": output_json
            }
            
            results.append(result)
            
            print(f"L2 Hit Rate: {avg_hit_rate:.2f}% ± {std_hit_rate:.2f}%")
            print(f"Performance: {tflops:.3f} TFLOPS")
            print(f"Time: {total_ms:.3f} ms")
            print("-" * 50)
    
    # Save results to JSON
    json_path = script_dir / "sweep_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved JSON results to: {json_path}")
    
    # Save results to CSV
    csv_path = script_dir / "sweep_results.csv"
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Saved CSV results to: {csv_path}")
    
    # Generate L2 hit rate bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for plotting
    x_positions = list(range(len(matrix_configs)))
    width = 0.35
    
    for idx, config in enumerate(configs):
        label = config["label"]
        color = config["color"]
        kernel_variant = config["kernel_variant"]
        
        # Filter results for this config
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        hit_rates = [r["avg_l2_hit_rate"] for r in config_results]
        std_rates = [r["std_l2_hit_rate"] for r in config_results]
        
        # Plot bars
        offset = width * (idx - 0.5)
        ax.bar(
            [x + offset for x in x_positions],
            hit_rates,
            width,
            label=label,
            color=color,
            yerr=std_rates,
            capsize=5,
            alpha=0.8
        )
    
    # Customize plot
    ax.set_xlabel('Matrix Configuration (Per-GPU GEMM)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average L2 Hit Rate (%)', fontsize=12, fontweight='bold')
    gemm_sms_title = f"GEMM SMs: {gemm_sms}" if gemm_sms is not None else "GEMM SMs: auto"
    plot_title = f'L2 Cache Hit Rate: Spatial Fusion vs Baseline\n(GEMM All Scatter Workgroup Specialization, {gemm_sms_title})'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    # Show per-GPU dimensions (N and K are split across ranks)
    ax.set_xticklabels([f"M={c['m']}\nN={c['n']//num_ranks}\nK={c['k']//num_ranks}" for c in matrix_configs], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for idx, config in enumerate(configs):
        kernel_variant = config["kernel_variant"]
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        hit_rates = [r["avg_l2_hit_rate"] for r in config_results]
        
        offset = width * (idx - 0.5)
        for x_pos, hit_rate in zip(x_positions, hit_rates):
            ax.text(
                x_pos + offset,
                hit_rate + 2,
                f'{hit_rate:.1f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    
    # Save L2 hit rate figure
    gemm_sms_str = f"gemm_sms_{gemm_sms}" if gemm_sms is not None else "gemm_sms_auto"
    fig_path = script_dir / f"sweep_l2_hit_rate_{gemm_sms_str}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved L2 hit rate figure to: {fig_path}")
    
    # Generate TFLOPS comparison plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    for idx, config in enumerate(configs):
        label = config["label"]
        color = config["color"]
        kernel_variant = config["kernel_variant"]
        
        # Filter results for this config
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        tflops_vals = [r["tflops"] for r in config_results]
        
        # Plot bars
        offset = width * (idx - 0.5)
        ax2.bar(
            [x + offset for x in x_positions],
            tflops_vals,
            width,
            label=label,
            color=color,
            alpha=0.8
        )
    
    # Customize plot
    ax2.set_xlabel('Matrix Configuration (Per-GPU GEMM)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
    perf_plot_title = f'Performance: Spatial Fusion vs Baseline\n(GEMM All Scatter Workgroup Specialization, {gemm_sms_title})'
    ax2.set_title(perf_plot_title, fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    # Show per-GPU dimensions (N and K are split across ranks)
    ax2.set_xticklabels([f"M={c['m']}\nN={c['n']//num_ranks}\nK={c['k']//num_ranks}" for c in matrix_configs], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for idx, config in enumerate(configs):
        kernel_variant = config["kernel_variant"]
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        tflops_vals = [r["tflops"] for r in config_results]
        
        offset = width * (idx - 0.5)
        for x_pos, tflops in zip(x_positions, tflops_vals):
            ax2.text(
                x_pos + offset,
                tflops + max(tflops_vals) * 0.02,
                f'{tflops:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    
    # Save performance figure
    perf_fig_path = script_dir / f"sweep_performance_{gemm_sms_str}.png"
    plt.savefig(perf_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved performance figure to: {perf_fig_path}")
    
    # Generate execution time comparison plot
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    for idx, config in enumerate(configs):
        label = config["label"]
        color = config["color"]
        kernel_variant = config["kernel_variant"]
        
        # Filter results for this config
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        time_vals = [r["total_ms"] for r in config_results]
        
        # Plot bars
        offset = width * (idx - 0.5)
        ax3.bar(
            [x + offset for x in x_positions],
            time_vals,
            width,
            label=label,
            color=color,
            alpha=0.8
        )
    
    # Customize plot
    ax3.set_xlabel('Matrix Configuration (Per-GPU GEMM)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    time_plot_title = f'Execution Time: Spatial Fusion vs Baseline\n(GEMM All Scatter Workgroup Specialization, {gemm_sms_title})'
    ax3.set_title(time_plot_title, fontsize=14, fontweight='bold')
    ax3.set_xticks(x_positions)
    # Show per-GPU dimensions (N and K are split across ranks)
    ax3.set_xticklabels([f"M={c['m']}\nN={c['n']//num_ranks}\nK={c['k']//num_ranks}" for c in matrix_configs], rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for idx, config in enumerate(configs):
        kernel_variant = config["kernel_variant"]
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        time_vals = [r["total_ms"] for r in config_results]
        
        offset = width * (idx - 0.5)
        for x_pos, time_val in zip(x_positions, time_vals):
            ax3.text(
                x_pos + offset,
                time_val + max(time_vals) * 0.02,
                f'{time_val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    
    # Save execution time figure
    time_fig_path = script_dir / f"sweep_exec_time_{gemm_sms_str}.png"
    plt.savefig(time_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Saved execution time figure to: {time_fig_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Sweep Summary")
    print("="*70)
    for config in configs:
        label = config["label"]
        kernel_variant = config["kernel_variant"]
        config_results = [r for r in results if r["kernel_variant"] == kernel_variant]
        
        print(f"\n{label}:")
        for r in config_results:
            print(f"  M={r['m']:4d}, N={r['n']:4d}, K={r['k']:4d}: "
                  f"L2={r['avg_l2_hit_rate']:6.2f}%±{r['std_l2_hit_rate']:.2f}% | "
                  f"Perf={r['tflops']:6.3f} TFLOPS | "
                  f"Time={r['total_ms']:7.3f} ms")
    
    print("\n" + "="*70)
    print("Sweep completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
