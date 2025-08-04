#!/usr/bin/env python3
"""
Latent Dimension Sweep Runner

This script runs a hyperparameter sweep focusing on latent dimension variations
from 16 to 1024 in 4 steps, with both gradient ascent and evolutionary search
optimization methods for evaluation.
"""

import subprocess
import sys
import os

def run_latent_dimension_sweep():
    """
    Run the latent dimension sweep with the following configuration:
    - Latent dimensions: [16, 256, 512, 1024]
    - Optimization methods: ['gradient', 'evolutionary']
    - Total configurations: 8 (4 dimensions × 2 methods)
    """
    
    print("=" * 60)
    print("LATENT DIMENSION SWEEP")
    print("=" * 60)
    print("Configuration:")
    print("  - Latent dimensions: [16, 256, 512, 1024]")
    print("  - Optimization methods: [gradient, evolutionary]")
    print("  - Total runs: 8")
    print("  - Modes: train, eval, visualize")
    print("=" * 60)
    
    # Run the sweep
    cmd = [
        sys.executable, "main_sweep.py",
        "--mode", "all",  # train, eval, visualize
        "--start_run", "1",
        "--end_run", "8",  # 8 configurations total
        "--device", "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "auto"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the sweep
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("SWEEP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Sweep failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nSweep interrupted by user")
        return False

def list_configurations():
    """List all configurations that will be run."""
    print("Listing configurations...")
    cmd = [
        sys.executable, "main_sweep.py",
        "--list_configs"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error listing configurations: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run latent dimension sweep')
    parser.add_argument('--list', action='store_true', help='List configurations and exit')
    
    args = parser.parse_args()
    
    if args.list:
        list_configurations()
    else:
        success = run_latent_dimension_sweep()
        sys.exit(0 if success else 1) 