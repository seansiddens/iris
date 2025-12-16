#!/usr/bin/env python3
"""
Wrapper script that runs the benchmark with rocprof counter collection
and analyzes L2 cache hit rates.

This script:
1. Runs the benchmark.py script with rocprofv3 to collect L2 cache counters
2. Parses the performance results from the benchmark JSON output
3. Analyzes the L2 hit rates from the rocprof CSV output
4. Combines and displays both performance and L2 cache statistics
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv


def parse_rocprof_csv(csv_path: Path, kernel_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parse rocprof CSV and calculate L2 hit rates for specified kernels.
    
    Args:
        csv_path: Path to the rocprof CSV file
        kernel_names: List of kernel names to analyze
        
    Returns:
        Dict mapping kernel name to stats dict with 'hits', 'misses', 'hit_rate'
    """
    kernel_set = set(kernel_names)
    kernel_stats: Dict[str, Dict[str, float]] = {
        k: {'hits': 0, 'misses': 0, 'invocations': 0, 'l2_cache_hit_values': []} 
        for k in kernel_names
    }
    
    if not csv_path.exists():
        return kernel_stats
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Group rows by Correlation_Id to pair hits and misses
        correlation_data: Dict[Tuple[str, str], Dict[str, float]] = {}
        
        for row in reader:
            kernel = row.get('Kernel_Name', '')
            if kernel not in kernel_set:
                continue
            
            corr_id = row.get('Correlation_Id', '')
            counter_name = row.get('Counter_Name', '')
            counter_value = float(row.get('Counter_Value', 0))
            key = (kernel, corr_id)
            
            if key not in correlation_data:
                correlation_data[key] = {}
            
            correlation_data[key][counter_name] = counter_value
        
        # Accumulate hits and misses
        for (kernel, corr_id), counters in correlation_data.items():
            if 'TCC_HIT_sum' in counters and 'TCC_MISS_sum' in counters:
                kernel_stats[kernel]['hits'] += counters['TCC_HIT_sum']
                kernel_stats[kernel]['misses'] += counters['TCC_MISS_sum']
                kernel_stats[kernel]['invocations'] += 1
            
            # Collect L2CacheHit values
            if 'L2CacheHit' in counters:
                kernel_stats[kernel]['l2_cache_hit_values'].append(counters['L2CacheHit'])
    
    # Calculate hit rates and average L2CacheHit
    for kernel, stats in kernel_stats.items():
        total = stats['hits'] + stats['misses']
        stats['hit_rate'] = (stats['hits'] / total * 100.0) if total > 0 else 0.0
        
        # Calculate average L2CacheHit
        if stats['l2_cache_hit_values']:
            stats['avg_l2_cache_hit'] = sum(stats['l2_cache_hit_values']) / len(stats['l2_cache_hit_values'])
        else:
            stats['avg_l2_cache_hit'] = None
    
    return kernel_stats


def run_benchmark_with_profiling(
    benchmark_args: List[str],
    output_json: str,
    profile_csv: Path,
    collect_l2: bool = True
) -> Tuple[bool, Optional[Dict]]:
    """
    Run the benchmark script with optional rocprof profiling.
    
    Args:
        benchmark_args: Command line arguments for benchmark.py
        output_json: Path to benchmark JSON output file
        profile_csv: Path where rocprof CSV should be written
        collect_l2: Whether to collect L2 cache counters with rocprof
        
    Returns:
        Tuple of (success, benchmark_results_dict)
    """
    benchmark_script = Path(__file__).parent / "benchmark.py"
    
    # Build the command
    if collect_l2:
        # Create rocprof input file with counters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("pmc: TCC_HIT_sum TCC_MISS_sum L2CacheHit\n")
            rocprof_input = f.name
        
        # rocprofv3 creates output in pmc_1/ directory with _counter_collection.csv suffix
        # We'll specify just the base name and find the actual file later
        profile_base = profile_csv.stem if hasattr(profile_csv, 'stem') else Path(profile_csv).stem
        
        cmd = [
            "rocprofv3",
            "--input", rocprof_input,
            "--output-file", profile_base,
            "--output-format", "csv",
            "--",  # Separator between rocprof options and application
            "python", str(benchmark_script)
        ] + benchmark_args
    else:
        cmd = ["python", str(benchmark_script)] + benchmark_args
    
    print(f"Running command: {' '.join(cmd)}\n")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        # Clean up temp file
        if collect_l2:
            Path(rocprof_input).unlink(missing_ok=True)
            
            # rocprofv3 creates output in pmc_1/ directory with modified filename
            # Find the actual output file
            actual_csv = Path("pmc_1") / f"{profile_base}_counter_collection.csv"
            if actual_csv.exists():
                # Move it to the desired location
                actual_csv.rename(profile_csv)
                # Clean up the pmc_1 directory if empty
                try:
                    Path("pmc_1").rmdir()
                except OSError:
                    pass  # Directory not empty or doesn't exist
        
        # Read benchmark results
        output_path = Path(output_json)
        if output_path.exists():
            with open(output_path, 'r') as f:
                benchmark_data = json.load(f)
            return True, benchmark_data
        else:
            print(f"Warning: Benchmark output file not found: {output_json}", file=sys.stderr)
            return True, None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        if collect_l2:
            Path(rocprof_input).unlink(missing_ok=True)
        return False, None


def display_combined_results(
    benchmark_data: Optional[Dict],
    l2_stats: Dict[str, Dict[str, float]],
    variant: str
):
    """
    Display combined performance and L2 cache statistics.
    
    Args:
        benchmark_data: Benchmark results from JSON
        l2_stats: L2 cache statistics by kernel
        variant: Kernel variant name (baseline/spatial)
    """
    print("\n" + "="*80)
    print(f"COMBINED RESULTS - Variant: {variant}")
    print("="*80)
    
    if benchmark_data:
        print("\nPerformance Metrics:")
        print("-" * 80)
        
        # Extract key metrics
        m = benchmark_data.get('M', 'N/A')
        n = benchmark_data.get('N', 'N/A')
        k = benchmark_data.get('K', 'N/A')
        world_size = benchmark_data.get('world_size', 'N/A')
        
        print(f"  Problem Size: M={m}, N={n}, K={k}")
        print(f"  World Size: {world_size}")
        print(f"  Block Size: M={benchmark_data.get('BLK_M', 'N/A')}, "
              f"N={benchmark_data.get('BLK_N', 'N/A')}, "
              f"K={benchmark_data.get('BLK_K', 'N/A')}")
        
        tflops = benchmark_data.get('tflops')
        total_ms = benchmark_data.get('total_ms')
        if tflops is not None:
            print(f"\n  Total Performance: {tflops:.2f} TFLOPS")
        if total_ms is not None:
            print(f"  Total Time: {total_ms:.3f} ms")
        
        gemm_ms = benchmark_data.get('gemm_ms')
        if gemm_ms is not None:
            print(f"  • GEMM Time: {gemm_ms:.3f} ms")
        
        gemm_sms = benchmark_data.get('gemm_sms')
        num_sms = benchmark_data.get('num_sms')
        if gemm_sms is not None and num_sms is not None:
            print(f"\n  SM Allocation:")
            print(f"    - GEMM: {gemm_sms} SMs")
            print(f"    - Communication: {num_sms - gemm_sms} SMs")
            print(f"    - Total: {num_sms} SMs")
    
    # Display L2 cache statistics
    has_l2_data = any(stats['invocations'] > 0 for stats in l2_stats.values())
    
    if has_l2_data:
        print("\nL2 Cache Statistics:")
        print("-" * 80)
        
        for kernel_name, stats in l2_stats.items():
            if stats['invocations'] > 0:
                print(f"\n  Kernel: {kernel_name}")
                print(f"    Invocations: {stats['invocations']:.0f}")
                print(f"    Total Hits: {stats['hits']:,.0f}")
                print(f"    Total Misses: {stats['misses']:,.0f}")
                print(f"    L2 Hit Rate: {stats['hit_rate']:.2f}%")
                
                # Display L2CacheHit if available
                avg_l2_cache_hit = stats.get('avg_l2_cache_hit')
                if avg_l2_cache_hit is not None:
                    print(f"    L2 Cache Hit (avg): {avg_l2_cache_hit:.4f}")
    else:
        print("\nL2 Cache Statistics:")
        print("-" * 80)
        print("  No L2 cache data collected (run with --profile-l2 to enable)")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark with L2 cache profiling and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings and L2 profiling
  ./run_with_l2_analysis.py --profile-l2
  
  # Run baseline variant with custom size
  ./run_with_l2_analysis.py --profile-l2 --kernel_variant baseline -m 8192 -n 4608 -k 36864
  
  # Run spatial variant with benchmarking
  ./run_with_l2_analysis.py --profile-l2 --kernel_variant spatial --benchmark
  
  # Run with validation
  ./run_with_l2_analysis.py --validate
        """
    )
    
    parser.add_argument(
        "--profile-l2",
        action="store_true",
        help="Enable L2 cache counter collection with rocprof"
    )
    parser.add_argument(
        "--profile-csv",
        type=str,
        default="profile_counter_collection.csv",
        help="Output path for rocprof CSV file"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="log.json",
        help="Output path for benchmark JSON file"
    )
    
    # Pass-through arguments for benchmark.py
    parser.add_argument("-m", type=int, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, help="Common dimension")
    parser.add_argument("--kernel_variant", type=str, choices=["baseline", "spatial"], 
                       help="Kernel variant to use")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation")
    parser.add_argument("--BLK_M", type=int, help="Block size M")
    parser.add_argument("--BLK_N", type=int, help="Block size N")
    parser.add_argument("--BLK_K", type=int, help="Block size K")
    parser.add_argument("--gsize_m", type=int, help="L2-cache locality swizzle parameter")
    parser.add_argument("--gemm_sms", type=int, help="Number of SMs for GEMM")
    parser.add_argument("--num_sms", type=int, help="Total number of SMs")
    parser.add_argument("--num_stages", type=int, help="Number of stages")
    parser.add_argument("-r", "--num_ranks", type=int, help="Number of ranks")
    
    args = parser.parse_args()
    
    # Build benchmark arguments
    benchmark_args = ["--output_file", args.output_json]
    
    if args.m is not None:
        benchmark_args.extend(["-m", str(args.m)])
    if args.n is not None:
        benchmark_args.extend(["-n", str(args.n)])
    if args.k is not None:
        benchmark_args.extend(["-k", str(args.k)])
    if args.kernel_variant:
        benchmark_args.extend(["--kernel_variant", args.kernel_variant])
    if args.benchmark:
        benchmark_args.append("--benchmark")
    if args.validate:
        benchmark_args.append("--validate")
    if args.BLK_M is not None:
        benchmark_args.extend(["--BLK_M", str(args.BLK_M)])
    if args.BLK_N is not None:
        benchmark_args.extend(["--BLK_N", str(args.BLK_N)])
    if args.BLK_K is not None:
        benchmark_args.extend(["--BLK_K", str(args.BLK_K)])
    if args.gsize_m is not None:
        benchmark_args.extend(["--gsize_m", str(args.gsize_m)])
    if args.gemm_sms is not None:
        benchmark_args.extend(["--gemm_sms", str(args.gemm_sms)])
    if args.num_sms is not None:
        benchmark_args.extend(["--num_sms", str(args.num_sms)])
    if args.num_stages is not None:
        benchmark_args.extend(["--num_stages", str(args.num_stages)])
    if args.num_ranks is not None:
        benchmark_args.extend(["-r", str(args.num_ranks)])
    
    # Run benchmark
    profile_csv = Path(args.profile_csv)
    success, benchmark_data = run_benchmark_with_profiling(
        benchmark_args,
        args.output_json,
        profile_csv,
        collect_l2=args.profile_l2
    )
    
    if not success:
        sys.exit(1)
    
    # Analyze L2 cache statistics
    kernel_names = [
        "persistent_gemm_all_scatter_wg_specialization",
        "persistent_gemm_all_scatter_wg_specialization_spatial"
    ]
    
    l2_stats = {}
    if args.profile_l2:
        l2_stats = parse_rocprof_csv(profile_csv, kernel_names)
    
    # Display combined results
    variant = args.kernel_variant if args.kernel_variant else benchmark_data.get('kernel_variant', 'unknown') if benchmark_data else 'unknown'
    display_combined_results(benchmark_data, l2_stats, variant)
    
    # Save combined results to a summary file
    if benchmark_data or l2_stats:
        summary_path = Path(args.output_json).with_suffix('.summary.json')
        summary = {
            'benchmark': benchmark_data,
            'l2_cache': l2_stats
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Combined results saved to: {summary_path}")


if __name__ == "__main__":
    main()
