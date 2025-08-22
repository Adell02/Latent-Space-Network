# Multi-Model Evaluation Guide

This guide explains how to use the modified `evaluate_all_artifacts.py` script to evaluate different models under the same conditions, ensuring fair comparison.

## Overview

The `evaluate_all_artifacts.py` script has been enhanced to support:
- **Separate configuration files** for each model (data + model configs)
- **Automatic specialist detection** or manual specification
- **Unified dataset generation** for fair comparison
- **Comparative metrics and visualizations**

## Key Features

### 1. **Same Dataset for All Models**
- Generates a single evaluation dataset using the first model's data configuration
- All models are evaluated on exactly the same support/query samples
- Ensures fair comparison by eliminating dataset variability

### 2. **Individual Model Configurations**
- Each model can have its own data and model architecture settings
- Supports both single-encoder and multi-encoder (specialist) models
- Automatic model architecture detection from checkpoints

### 3. **Comprehensive Comparison**
- Unified metrics across all models
- Comparative visualizations (accuracy charts, support vs query performance)
- Single WandB run for easy comparison

## Usage Examples

### Basic Multi-Model Evaluation

```bash
python3 evaluate_all_artifacts.py \
  --artifacts \
    ga624-imperial-college-london/LPN_specialist_paper/model1:latest \
    ga624-imperial-college-london/LPN_specialist_paper/model2:v1 \
  --configs \
    data_config1.json:model_config1.json \
    data_config2.json:model_config2.json \
  --specialist_flags 0 1
```

### Your Specific Use Case

```bash
python3 evaluate_all_artifacts.py \
  --artifacts \
    ga624-imperial-college-london/LPN_specialist_paper/LPN_specialist_paper_20250821_101933_main_20250821_101933_checkpoint_epoch_20:v0 \
    ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v7 \
  --configs \
    data_evaluation_1ENC_settings.json:model_1ENC_evaluation_settings.json \
    data_evaluation_2ENC_settings.json:model_2ENC_evaluation_settings.json \
  --specialist_flags 0 1
```

### Using the Shell Script

```bash
./run_multi_model_evaluation.sh
```

## Command Line Arguments

### Required Arguments

- `--artifacts`: List of WandB artifact IDs to evaluate
- `--configs`: List of config pairs in format "data_config:model_config" for each model

### Optional Arguments

- `--specialist_flags`: List of specialist flags (1) or single encoder (0) for each model
- `--n_samples`: Number of support samples per task (default: 5)
- `--n_queries`: Number of query samples per task (default: 10)
- `--run_dir`: Custom output directory
- `--no_wandb`: Disable WandB logging
- `--device`: Device to use (default: cuda if available)

### Legacy Support

- `--config`: Single config file for all models (legacy mode)
- `--specialist`: Global specialist flag for all models (legacy mode)

## Configuration File Structure

### Data Configuration Files
- `data_evaluation_1ENC_settings.json`
- `data_evaluation_2ENC_settings.json`
- Contains: evaluation settings, data settings, training settings

### Model Configuration Files
- `model_1ENC_evaluation_settings.json`
- `model_2ENC_evaluation_settings.json`
- Contains: model architecture, latent optimization settings

## How It Works

### 1. **Dataset Generation**
```
First model's data config → Generate unified dataset → All models use same dataset
```

### 2. **Model Loading**
```
For each model:
  - Download artifact
  - Load model-specific config
  - Create model with detected/configured architecture
  - Load checkpoint weights
```

### 3. **Evaluation**
```
For each model:
  - Evaluate on unified dataset
  - Collect metrics (support pre/post optimization, query performance)
  - Store results with model-specific naming
```

### 4. **Comparison**
```
- Aggregate metrics across all models
- Generate comparative visualizations
- Log unified results to WandB
- Save detailed results to JSON files
```

## Output Files

### Main Results
- `all_model_results.json`: Detailed results for each model
- `unified_metrics.json`: Aggregated metrics for comparison
- `model_metadata.json`: Model information and configuration details
- `unified_dataset_info.json`: Dataset information used for evaluation

### Visualizations
- `comparative_accuracies.png`: Bar chart comparing model accuracies
- `support_vs_query_comparison.png`: Support vs query performance comparison

## Example Output Structure

```json
{
  "model1/query": {
    "shape_accuracy": 0.85,
    "grid_accuracy": 0.78,
    "exact_accuracy": 0.72
  },
  "model2/query": {
    "shape_accuracy": 0.92,
    "grid_accuracy": 0.89,
    "exact_accuracy": 0.85
  }
}
```

## Troubleshooting

### Common Issues

1. **Config Mismatch**: Ensure number of configs matches number of artifacts
2. **Specialist Flag Mismatch**: Ensure number of specialist flags matches number of artifacts
3. **File Not Found**: Check that all configuration files exist and are accessible

### Validation

The script automatically validates:
- Configuration file existence
- Argument count consistency
- Model architecture compatibility

## Advanced Usage

### Custom Dataset Size

```bash
python3 evaluate_all_artifacts.py \
  --artifacts model1:latest model2:latest \
  --configs config1:config2 \
  --n_samples 10 \
  --n_queries 20
```

### Custom Output Directory

```bash
python3 evaluate_all_artifacts.py \
  --artifacts model1:latest model2:latest \
  --configs config1:config2 \
  --run_dir ./my_comparison_results
```

### Disable WandB

```bash
python3 evaluate_all_artifacts.py \
  --artifacts model1:latest model2:latest \
  --configs config1:config2 \
  --no_wandb
```

## Benefits of This Approach

1. **Fair Comparison**: Same dataset eliminates variability
2. **Efficient**: Single run evaluates multiple models
3. **Consistent**: Unified metrics and naming conventions
4. **Flexible**: Individual model configurations
5. **Comprehensive**: Comparative analysis and visualizations

## Migration from Individual Evaluation

### Before (Individual Runs)
```bash
# Model 1
python evaluate_artifact.py --artifact model1:latest --config_data config1.json --config_model model1.json

# Model 2  
python evaluate_artifact.py --artifact model2:latest --config_data config2.json --config_model model2.json --specialist
```

### After (Unified Run)
```bash
python evaluate_all_artifacts.py \
  --artifacts model1:latest model2:latest \
  --configs config1.json:model1.json config2.json:model2.json \
  --specialist_flags 0 1
```

This approach ensures both models are evaluated on the same dataset with consistent methodology.


