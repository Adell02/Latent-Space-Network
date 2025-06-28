#!/bin/bash

# Hyperparameter Sweep Runner Script
# Usage: ./run_sweep.sh [options]

# Default values
MODE="all"
START_RUN=1
END_RUN=""
DEVICE=""
PARALLEL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --start_run)
            START_RUN="$2"
            shift 2
            ;;
        --end_run)
            END_RUN="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --list_configs)
            python main_sweep.py --list_configs
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE        Modes to run (train, eval, visualize, all) [default: all]"
            echo "  --start_run N        Starting run number [default: 1]"
            echo "  --end_run N          Ending run number [default: all]"
            echo "  --device DEVICE      Device to use (cuda, cpu) [default: auto-detect]"
            echo "  --parallel           Run experiments in parallel"
            echo "  --list_configs       List all configurations and exit"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all configurations"
            echo "  $0 --start_run 1 --end_run 5         # Run first 5 configurations"
            echo "  $0 --mode train --device cuda       # Run only training on GPU"
            echo "  $0 --list_configs                    # List all configurations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python main_sweep.py --mode $MODE --start_run $START_RUN"

if [[ -n "$END_RUN" ]]; then
    CMD="$CMD --end_run $END_RUN"
fi

if [[ -n "$DEVICE" ]]; then
    CMD="$CMD --device $DEVICE"
fi

if [[ "$PARALLEL" == true ]]; then
    CMD="$CMD --parallel"
fi

# Run the sweep
echo "Starting hyperparameter sweep..."
echo "Command: $CMD"
echo ""

eval $CMD 