#!/usr/bin/env python3
"""
Script to generate CSV files containing matrix configurations for GEMM sweeps.

This script provides several preset configurations and the ability to generate
custom configurations. All CSV files are saved to the 'datasets' directory.
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict


def generate_power_of_2_sweep(start: int = 1024, end: int = 16384, n_scale: float = 1.0, k_scale: float = 1.0) -> List[Dict[str, int]]:
    """
    Generate matrix configs with M values as powers of 2, with N and K as scaled multiples.
    
    Args:
        start: Starting value for M
        end: Ending value for M (inclusive)
        n_scale: Scaling factor for N (N = M * n_scale)
        k_scale: Scaling factor for K (K = M * k_scale)
        
    Returns:
        List of matrix configurations
    """
    configs = []
    m = start
    while m <= end:
        configs.append({
            'm': m,
            'n': int(m * n_scale),
            'k': int(m * k_scale)
        })
        m *= 2
    return configs


def generate_linear_sweep(start: int, end: int, step: int, n_scale: float = 1.0, k_scale: float = 1.0) -> List[Dict[str, int]]:
    """
    Generate matrix configs with linearly spaced M values.
    
    Args:
        start: Starting value for M
        end: Ending value for M (inclusive)
        step: Step size between M values
        n_scale: Scaling factor for N (N = M * n_scale)
        k_scale: Scaling factor for K (K = M * k_scale)
        
    Returns:
        List of matrix configurations
    """
    configs = []
    for m in range(start, end + 1, step):
        configs.append({
            'm': m,
            'n': int(m * n_scale),
            'k': int(m * k_scale)
        })
    return configs


def generate_fixed_k_sweep(m_values: List[int], n_values: List[int], k: int) -> List[Dict[str, int]]:
    """
    Generate matrix configs with fixed K and varying M and N.
    
    Args:
        m_values: List of M values
        n_values: List of N values (same length as m_values)
        k: Fixed K value
        
    Returns:
        List of matrix configurations
    """
    configs = []
    for m, n in zip(m_values, n_values):
        configs.append({
            'm': m,
            'n': n,
            'k': k
        })
    return configs


def generate_square_matrices(sizes: List[int]) -> List[Dict[str, int]]:
    """
    Generate square matrix configs (M = N = K).
    
    Args:
        sizes: List of matrix sizes
        
    Returns:
        List of matrix configurations
    """
    configs = []
    for size in sizes:
        configs.append({
            'm': size,
            'n': size,
            'k': size
        })
    return configs


def generate_skinny_matrices(m_values: List[int], n_ratio: float = 0.25) -> List[Dict[str, int]]:
    """
    Generate skinny matrix configs (N << M, K = M).
    
    Args:
        m_values: List of M values
        n_ratio: Ratio of N to M (N = M * n_ratio)
        
    Returns:
        List of matrix configurations
    """
    configs = []
    for m in m_values:
        configs.append({
            'm': m,
            'n': int(m * n_ratio),
            'k': m
        })
    return configs


def generate_wide_matrices(m_values: List[int], n_ratio: float = 4.0) -> List[Dict[str, int]]:
    """
    Generate wide matrix configs (N >> M, K = M).
    
    Args:
        m_values: List of M values
        n_ratio: Ratio of N to M (N = M * n_ratio)
        
    Returns:
        List of matrix configurations
    """
    configs = []
    for m in m_values:
        configs.append({
            'm': m,
            'n': int(m * n_ratio),
            'k': m
        })
    return configs


def generate_all_power_of_2_combinations(start: int = 256, end: int = 16384) -> List[Dict[str, int]]:
    """
    Generate all combinations of power-of-2 dimensions, sorted by total size (M*N*K).
    
    Args:
        start: Starting power of 2 (e.g., 256)
        end: Ending power of 2 (inclusive, e.g., 16384)
        
    Returns:
        List of matrix configurations sorted by size
    """
    # Generate all power of 2 values in range
    sizes = []
    size = start
    while size <= end:
        sizes.append(size)
        size *= 2
    
    # Generate all combinations
    configs = []
    for m in sizes:
        for n in sizes:
            for k in sizes:
                configs.append({
                    'm': m,
                    'n': n,
                    'k': k,
                    'size': m * n * k  # For sorting
                })
    
    # Sort by total size (M*N*K)
    configs.sort(key=lambda x: x['size'])
    
    # Remove the 'size' field before returning
    for config in configs:
        del config['size']
    
    return configs


def generate_custom_combinations(sizes: List[int]) -> List[Dict[str, int]]:
    """
    Generate all combinations of specific dimension sizes, sorted by total size (M*N*K).
    
    Args:
        sizes: List of dimension values to use (e.g., [512, 1024, 4096, 8192])
        
    Returns:
        List of matrix configurations sorted by size
    """
    # Generate all combinations
    configs = []
    for m in sizes:
        for n in sizes:
            for k in sizes:
                configs.append({
                    'm': m,
                    'n': n,
                    'k': k,
                    'size': m * n * k  # For sorting
                })
    
    # Sort by total size (M*N*K)
    configs.sort(key=lambda x: x['size'])
    
    # Remove the 'size' field before returning
    for config in configs:
        del config['size']
    
    return configs


def save_to_csv(configs: List[Dict[str, int]], filename: str, datasets_dir: Path):
    """
    Save matrix configurations to a CSV file.
    
    Args:
        configs: List of matrix configurations
        filename: Name of the output CSV file
        datasets_dir: Directory to save the file
    """
    filepath = datasets_dir / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['m', 'n', 'k'])
        writer.writeheader()
        writer.writerows(configs)
    
    print(f"✓ Generated {len(configs)} configurations -> {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate matrix configuration CSV files for GEMM sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all presets
  python generate_matrix_configs.py --all
  
  # Generate specific preset
  python generate_matrix_configs.py --preset default
  
  # Generate exhaustive power-of-2 sweep (all combinations from 256 to 16384)
  python generate_matrix_configs.py --preset exhaustive
  
  # Generate custom power-of-2 sweep
  python generate_matrix_configs.py --custom power2 --start 2048 --end 8192
  
  # Generate custom linear sweep
  python generate_matrix_configs.py --custom linear --start 1000 --end 5000 --step 1000
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Generate all preset configurations')
    parser.add_argument('--preset', type=str, choices=['default', 'small', 'large', 'square', 'skinny', 'wide', 'exhaustive', 'medium'],
                        help='Generate a specific preset configuration')
    parser.add_argument('--custom', type=str, choices=['power2', 'linear'],
                        help='Generate a custom configuration')
    parser.add_argument('--start', type=int, default=1024,
                        help='Starting M value for custom configs')
    parser.add_argument('--end', type=int, default=16384,
                        help='Ending M value for custom configs')
    parser.add_argument('--step', type=int, default=1024,
                        help='Step size for linear sweep')
    parser.add_argument('--n-scale', type=float, default=0.5,
                        help='Scaling factor for N (N = M * n_scale)')
    parser.add_argument('--k-scale', type=float, default=1.0,
                        help='Scaling factor for K (K = M * k_scale)')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output filename (saved to datasets/)')
    
    args = parser.parse_args()
    
    # Create datasets directory
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Matrix Configuration Generator")
    print("="*70)
    print(f"Output directory: {datasets_dir}\n")
    
    if args.all or args.preset == 'default' or (not args.all and not args.preset and not args.custom):
        # Default: Power of 2 sweep with N=M/2, K=M
        configs = generate_power_of_2_sweep(1024, 16384, n_scale=0.5, k_scale=1.0)
        save_to_csv(configs, "default.csv", datasets_dir)
    
    if args.all or args.preset == 'small':
        # Small matrices for quick testing
        configs = generate_power_of_2_sweep(512, 4096, n_scale=0.5, k_scale=1.0)
        save_to_csv(configs, "small.csv", datasets_dir)
    
    if args.all or args.preset == 'large':
        # Large matrices for performance testing
        configs = generate_power_of_2_sweep(4096, 32768, n_scale=0.5, k_scale=1.0)
        save_to_csv(configs, "large.csv", datasets_dir)
    
    if args.all or args.preset == 'square':
        # Square matrices (M = N = K)
        sizes = [1024, 2048, 4096, 8192, 16384]
        configs = generate_square_matrices(sizes)
        save_to_csv(configs, "square.csv", datasets_dir)
    
    if args.all or args.preset == 'skinny':
        # Skinny matrices (N << M)
        m_values = [1024, 2048, 4096, 8192, 16384]
        configs = generate_skinny_matrices(m_values, n_ratio=0.125)
        save_to_csv(configs, "skinny.csv", datasets_dir)
    
    if args.all or args.preset == 'wide':
        # Wide matrices (N >> M)
        m_values = [1024, 2048, 4096, 8192, 16384]
        configs = generate_wide_matrices(m_values, n_ratio=4.0)
        save_to_csv(configs, "wide.csv", datasets_dir)
    
    if args.all or args.preset == 'exhaustive':
        # All combinations of power-of-2 dimensions from 256 to 4096
        configs = generate_all_power_of_2_combinations(256, 4096)
        save_to_csv(configs, "exhaustive.csv", datasets_dir)
    
    if args.all or args.preset == 'medium':
        # All combinations of specific sizes: 512, 1024, 4096, 8192
        configs = generate_custom_combinations([512, 1024, 4096, 8192])
        save_to_csv(configs, "medium.csv", datasets_dir)
    
    if args.custom == 'power2':
        # Custom power-of-2 sweep
        configs = generate_power_of_2_sweep(args.start, args.end, args.n_scale, args.k_scale)
        filename = args.output if args.output else f"custom_power2_{args.start}_{args.end}.csv"
        save_to_csv(configs, filename, datasets_dir)
    
    if args.custom == 'linear':
        # Custom linear sweep
        configs = generate_linear_sweep(args.start, args.end, args.step, args.n_scale, args.k_scale)
        filename = args.output if args.output else f"custom_linear_{args.start}_{args.end}_step{args.step}.csv"
        save_to_csv(configs, filename, datasets_dir)
    
    print("\n" + "="*70)
    print("Configuration generation complete!")
    print("="*70)
    print(f"\nUse with sweep.py:")
    print(f"  python sweep.py -c datasets/<filename>.csv")


if __name__ == "__main__":
    main()
