#!/bin/bash

# Multi-Model Evaluation Script
# This script evaluates different models under the same conditions using evaluate_all_artifacts.py
# The script generates unified datasets once and then processes models one by one for memory efficiency

echo "=== Multi-Model Evaluation Script ==="
echo "This script will:"
echo "1. Generate unified training and evaluation datasets ONCE"
echo "2. Process models one by one using evaluate_artifact.py"
echo "3. Free memory after each model for efficiency"
echo "4. Generate comparative metrics and visualizations"
echo ""

# Configuration
ARTIFACT_1="ga624-imperial-college-london/LPN_specialist_paper/LPN_specialist_paper_20250821_101933_main_20250821_101933_checkpoint_epoch_20:v0"
ARTIFACT_2="ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v7"

CONFIG_1="data_evaluation_1ENC_settings.json:model_1ENC_evaluation_settings.json"
CONFIG_2="data_evaluation_2ENC_settings.json:model_2ENC_evaluation_settings.json"

SPECIALIST_FLAGS="0 1"  # 0 = single encoder, 1 = specialist

# Dataset parameters
N_SAMPLES=5          # Support samples per key
N_QUERIES=10         # Query samples per key  
N_TRAINING=100       # Total training samples

echo "Configuration:"
echo "  Model 1: $ARTIFACT_1 (Single Encoder)"
echo "  Model 2: $ARTIFACT_2 (Specialist)"
echo "  Configs: $CONFIG_1, $CONFIG_2"
echo "  Specialist flags: $SPECIALIST_FLAGS"
echo "  Dataset: $N_SAMPLES support + $N_QUERIES query per key, $N_TRAINING training samples"
echo ""

# Check if required config files exist
echo "Checking configuration files..."
for config_pair in "$CONFIG_1" "$CONFIG_2"; do
    data_config=$(echo "$config_pair" | cut -d: -f1)
    model_config=$(echo "$config_pair" | cut -d: -f2)
    
    if [[ ! -f "$data_config" ]]; then
        echo "ERROR: Data config file not found: $data_config"
        exit 1
    fi
    
    if [[ ! -f "$model_config" ]]; then
        echo "ERROR: Model config file not found: $model_config"
        exit 1
    fi
    
    echo "  ✓ $data_config"
    echo "  ✓ $model_config"
done

echo ""
echo "All configuration files found. Starting evaluation..."
echo ""

# Run the evaluation
python3 evaluate_all_artifacts.py \
    --artifacts "$ARTIFACT_1" "$ARTIFACT_2" \
    --configs "$CONFIG_1" "$CONFIG_2" \
    --specialist_flags $SPECIALIST_FLAGS \
    --n_samples $N_SAMPLES \
    --n_queries $N_QUERIES \
    --n_training $N_TRAINING \
    --device cuda

echo ""
echo "=== Evaluation Complete ==="
echo "Check the output directory for results, metrics, and visualizations."
echo "The script generated unified datasets once and processed models sequentially for memory efficiency."
