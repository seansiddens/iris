#!/usr/bin/env python3
"""
Sweep script for spatial fusion benchmarks with L2 cache analysis.

This script runs the benchmark across different configurations, profiles with rocprof,
analyzes L2 hit rates, and generates visualizations and logs.
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
    num_tiles: int,
    spatial: bool,
    wg_specialized: bool,
    script_dir: Path
) -> Tuple[float, float, int, int, float, float, float]:
    """
    Run a single benchmark configuration and extract L2 hit rate and timing information.
    
    Args:
        num_tiles: Number of tiles for the benchmark
        spatial: Whether to enable spatial fusion
        wg_specialized: Whether to use workgroup specialized kernel
        script_dir: Directory containing the scripts
        
    Returns:
        Tuple of (average_hit_rate, std_hit_rate, producer_xcd, consumer_xcd, 
                  avg_exec_time, std_exec_time, avg_overlap_pct)
    """
    # Build command with JSON output
    benchmark_json = script_dir / "benchmark_results.json"
    cmd = ["./rocprof.sh", "python", "benchmark.py", "-n", str(num_tiles), "-o", str(benchmark_json)]
    if wg_specialized:
        cmd.append("-wg")
    if spatial:
        cmd.append("-s")
    
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
        return (0.0, 0.0, -1, -1, 0.0, 0.0, 0.0)
    
    # Read benchmark results from JSON
    if not benchmark_json.exists():
        print(f"Error: Benchmark results JSON not found at {benchmark_json}")
        return (0.0, 0.0, -1, -1, 0.0, 0.0, 0.0)
    
    with open(benchmark_json, 'r') as f:
        bench_results = json.load(f)
    
    # Extract metadata and results
    producer_xcd = bench_results["xcd_schedule"]["producer_xcd"]
    consumer_xcd = bench_results["xcd_schedule"]["consumer_xcd"]
    avg_exec_time = bench_results["timing_results"]["total_ms"]["mean"]
    std_exec_time = bench_results["timing_results"]["total_ms"]["std"]
    avg_overlap_pct = bench_results["timing_results"].get("overlap_percentage", 0.0)
    
    # VALIDATION: With spatial fusion enabled, producer and consumer MUST be on same XCD
    if not wg_specialized and spatial and producer_xcd != consumer_xcd:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: SPATIAL FUSION VALIDATION FAILED!\n"
            f"{'='*70}\n"
            f"Spatial fusion is ENABLED but kernels are on DIFFERENT XCDs!\n"
            f"  Producer XCD: {producer_xcd}\n"
            f"  Consumer XCD: {consumer_xcd}\n"
            f"  Expected: Both on same XCD\n"
            f"\n"
            f"This indicates spatial fusion is NOT working correctly.\n"
            f"The producer and consumer kernels must execute on the same XCD\n"
            f"for spatial fusion to provide L2 cache benefits.\n"
            f"{'='*70}\n"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    # Parse the profiling output
    csv_path = script_dir / "profile_counter_collection.csv"
    if not csv_path.exists():
        print(f"Error: Profile output not found at {csv_path}")
        return (0.0, 0.0, producer_xcd, consumer_xcd, avg_exec_time, std_exec_time, avg_overlap_pct)
    
    # Determine which kernel to analyze
    if wg_specialized:
        kernel_name = "workgroup_specialized_kernel"
    else:
        # For non-specialized, we'll average producer and consumer
        kernel_name = "producer_kernel"
    
    kernel_names = [kernel_name]
    if not wg_specialized:
        kernel_names.append("consumer_kernel")
    
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
        return (0.0, 0.0, producer_xcd, consumer_xcd, avg_exec_time, std_exec_time, avg_overlap_pct)
    
    avg_hit_rate = statistics.mean(all_hit_rates)
    std_hit_rate = statistics.stdev(all_hit_rates) if len(all_hit_rates) > 1 else 0.0
    
    return (avg_hit_rate, std_hit_rate, producer_xcd, consumer_xcd, avg_exec_time, std_exec_time, avg_overlap_pct)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sweep L2 cache performance for spatial fusion benchmarks')
    parser.add_argument('--concurrent', action='store_true',
                        help='Use concurrent kernels instead of workgroup specialized')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Configuration
    tile_counts = [1, 8, 64, 256, 1024, 2048, 4096, 8192]
    block_size = 256
    use_wg_specialized = not args.concurrent
    num_workgroups = 16 if use_wg_specialized else 8
    
    # Results storage
    results = []
    
    # Sweep configurations
    kernel_mode = "Workgroup Specialized" if use_wg_specialized else "Concurrent Kernels"
    configs = [
        {"spatial": False, "label": f"Baseline", "color": "#1f77b4"},
        {"spatial": True, "label": f"Spatial Fusion", "color": "#ff7f0e"},
    ]
    
    print("="*70)
    print("Starting Spatial Fusion L2 Cache Sweep")
    print("="*70)
    print(f"Tile counts: {tile_counts}")
    print(f"Block size: {block_size}")
    print(f"Kernel mode: {kernel_mode}")
    print(f"Workgroup specialized: {use_wg_specialized}")
    print(f"Num workgroups: {num_workgroups}")
    print("="*70 + "\n")
    
    # Run sweeps
    for config in configs:
        spatial = config["spatial"]
        label = config["label"]
        
        print(f"\n{'='*70}")
        print(f"Running sweep: {label}")
        print(f"{'='*70}\n")
        
        for num_tiles in tile_counts:
            print(f"\nConfiguration: {label}, num_tiles={num_tiles}")
            print("-" * 50)
            
            avg_hit_rate, std_hit_rate, producer_xcd, consumer_xcd, avg_exec_time, std_exec_time, avg_overlap_pct = run_benchmark(
                num_tiles=num_tiles,
                spatial=spatial,
                wg_specialized=use_wg_specialized,
                script_dir=script_dir
            )
            
            result = {
                "approach": label,
                "kernel_mode": kernel_mode,
                "spatial": spatial,
                "wg_specialized": use_wg_specialized,
                "num_tiles": num_tiles,
                "block_size": block_size,
                "num_workgroups": num_workgroups,
                "buffer_size": block_size * num_tiles,
                "producer_xcd": producer_xcd,
                "consumer_xcd": consumer_xcd,
                "avg_l2_hit_rate": avg_hit_rate,
                "std_l2_hit_rate": std_hit_rate,
                "avg_exec_time_ms": avg_exec_time,
                "std_exec_time_ms": std_exec_time,
                "avg_overlap_pct": avg_overlap_pct,
            }
            
            results.append(result)
            
            print(f"Producer XCD: {producer_xcd}, Consumer XCD: {consumer_xcd}")
            print(f"L2 Hit Rate: {avg_hit_rate:.2f}% ± {std_hit_rate:.2f}%")
            print(f"Exec Time: {avg_exec_time:.3f} ± {std_exec_time:.3f} ms")
            if not use_wg_specialized:
                print(f"Overlap: {avg_overlap_pct:.1f}%")
            print("-" * 50)
    
    # Save results to JSON with appropriate filename
    suffix = "concurrent" if not use_wg_specialized else "wg_specialized"
    json_path = script_dir / f"sweep_results_{suffix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved JSON results to: {json_path}")
    
    # Save results to CSV
    csv_path = script_dir / f"sweep_results_{suffix}.csv"
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Saved CSV results to: {csv_path}")
    
    # Generate bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for plotting
    x_positions = list(range(len(tile_counts)))
    width = 0.35
    
    for idx, config in enumerate(configs):
        label = config["label"]
        color = config["color"]
        spatial = config["spatial"]
        
        # Filter results for this config
        config_results = [r for r in results if r["spatial"] == spatial]
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
    ax.set_xlabel('Number of Tiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average L2 Hit Rate (%)', fontsize=12, fontweight='bold')
    plot_title = f'L2 Cache Hit Rate: Spatial Fusion vs Baseline\n({kernel_mode})'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(t) for t in tile_counts])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for idx, config in enumerate(configs):
        spatial = config["spatial"]
        config_results = [r for r in results if r["spatial"] == spatial]
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
    
    # Save figure with appropriate filename
    suffix = "concurrent" if not use_wg_specialized else "wg_specialized"
    fig_path = script_dir / f"sweep_l2_hit_rate_{suffix}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved L2 hit rate figure to: {fig_path}")
    
    # Generate execution time comparison plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    for idx, config in enumerate(configs):
        label = config["label"]
        color = config["color"]
        spatial = config["spatial"]
        
        # Filter results for this config
        config_results = [r for r in results if r["spatial"] == spatial]
        exec_times = [r["avg_exec_time_ms"] for r in config_results]
        std_times = [r["std_exec_time_ms"] for r in config_results]
        
        # Plot bars
        offset = width * (idx - 0.5)
        ax2.bar(
            [x + offset for x in x_positions],
            exec_times,
            width,
            label=label,
            color=color,
            yerr=std_times,
            capsize=5,
            alpha=0.8
        )
    
    # Customize plot
    ax2.set_xlabel('Number of Tiles', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    time_plot_title = f'Execution Time: Spatial Fusion vs Baseline\n({kernel_mode})'
    ax2.set_title(time_plot_title, fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(t) for t in tile_counts])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for idx, config in enumerate(configs):
        spatial = config["spatial"]
        config_results = [r for r in results if r["spatial"] == spatial]
        exec_times = [r["avg_exec_time_ms"] for r in config_results]
        
        offset = width * (idx - 0.5)
        for x_pos, exec_time in zip(x_positions, exec_times):
            ax2.text(
                x_pos + offset,
                exec_time + max(exec_times) * 0.02,
                f'{exec_time:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    
    # Save execution time figure
    time_fig_path = script_dir / f"sweep_exec_time_{suffix}.png"
    plt.savefig(time_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved execution time figure to: {time_fig_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Sweep Summary")
    print("="*70)
    for config in configs:
        label = config["label"]
        spatial = config["spatial"]
        config_results = [r for r in results if r["spatial"] == spatial]
        
        print(f"\n{label}:")
        for r in config_results:
            xcd_info = f" (P_XCD={r['producer_xcd']}, C_XCD={r['consumer_xcd']})"
            time_info = f" | Time: {r['avg_exec_time_ms']:.3f}±{r['std_exec_time_ms']:.3f}ms"
            overlap_info = f" | Overlap: {r['avg_overlap_pct']:.1f}%" if not use_wg_specialized and r['avg_overlap_pct'] > 0 else ""
            print(f"  {r['num_tiles']:5d} tiles: {r['avg_l2_hit_rate']:6.2f}% ± {r['std_l2_hit_rate']:.2f}%{xcd_info}{time_info}{overlap_info}")
    
    print("\n" + "="*70)
    print("Sweep completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
