# Multi-Model Evaluation Guide

This guide explains how to use the modified `evaluate_all_artifacts.py` script to evaluate multiple models under the same conditions for fair comparison.

## **Key Features**

### **Memory Efficiency**
- **Generates unified datasets ONCE** (training and evaluation)
- **Processes models one by one** using `evaluate_artifact.py`
- **Frees memory after each model** for optimal resource usage
- **Avoids device mismatch issues** by using proven evaluation pipeline

### **Fair Comparison**
- **Same dataset for all models** ensures consistent evaluation
- **Individual model configs** allow proper architecture handling
- **Specialist flag specification** for correct model type detection
- **Unified metrics computation** across all evaluated models

## **Usage**

### **Basic Command Structure**
```bash
python3 evaluate_all_artifacts.py \
    --artifacts ARTIFACT1 ARTIFACT2 ... \
    --configs CONFIG1:CONFIG2 ... \
    --specialist_flags 0 1 ... \
    [--n_samples 5] \
    [--n_queries 10] \
    [--n_training 100] \
    [--device cuda] \
    [--run_dir output_directory]
```

### **Arguments**

- **`--artifacts`**: List of WandB artifact IDs to evaluate
- **`--configs`**: List of config pairs in format "data_config:model_config" for each model
- **`--specialist_flags`**: List of specialist flags (1) or single encoder (0) for each model
- **`--n_samples`**: Number of support samples per task (default: 5)
- **`--n_queries`**: Number of query samples per task (default: 10)
- **`--n_training`**: Total number of training samples (default: 100)
- **`--device`**: Device to use for evaluation (default: auto-detect)
- **`--run_dir`**: Output directory (default: auto-generated)

## **Example: 1ENC vs 2ENC Comparison**

### **Configuration Files**
You need separate configuration files for each model:

**1ENC Model:**
- `data_evaluation_1ENC_settings.json` - Data and evaluation settings
- `model_1ENC_evaluation_settings.json` - Model architecture settings

**2ENC Model:**
- `data_evaluation_2ENC_settings.json` - Data and evaluation settings  
- `model_2ENC_evaluation_settings.json` - Model architecture settings

### **Command**
```bash
python3 evaluate_all_artifacts.py \
    --artifacts \
        ga624-imperial-college-london/LPN_specialist_paper/LPN_specialist_paper_20250821_101933_main_20250821_101933_checkpoint_epoch_20:v0 \
        ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v7 \
    --configs \
        data_evaluation_1ENC_settings.json:model_1ENC_evaluation_settings.json \
        data_evaluation_2ENC_settings.json:model_2ENC_evaluation_settings.json \
    --specialist_flags 0 1 \
    --n_samples 5 \
    --n_queries 10 \
    --n_training 100
```

### **Shell Script**
Use the provided `run_multi_model_evaluation.sh` script for convenience:

```bash
./run_multi_model_evaluation.sh
```

## **How It Works**

### **1. Dataset Generation (Once)**
- Generates unified training dataset with specified number of samples
- Generates unified evaluation dataset with support/query samples per key
- Saves datasets to pickle files for reuse

### **2. Sequential Model Processing**
- Downloads and processes each model artifact individually
- Uses `evaluate_artifact.py` with unified dataset paths
- Clears GPU memory after each model for efficiency
- Collects results from each model's output directory

### **3. Unified Analysis**
- Computes comparative metrics across all models
- Generates comparative visualizations (accuracy charts, latent space plots)
- Logs unified metrics to WandB for easy comparison
- Saves comprehensive results and metadata

## **Output Structure**

```
output_directory/
├── unified_training_dataset.pkl          # Unified training data
├── unified_evaluation_dataset.pkl        # Unified evaluation data
├── unified_dataset_info.json            # Dataset metadata
├── model_1_MODELNAME/                   # Individual model results
│   ├── eval_aggregated_metrics.json
│   ├── training_latent_distance_metrics.json
│   └── latent_space_plots/
├── model_2_MODELNAME/                   # Individual model results
│   ├── eval_aggregated_metrics.json
│   ├── training_latent_distance_metrics.json
│   └── latent_space_plots/
├── all_model_results.json               # Combined results
├── unified_metrics.json                 # Comparative metrics
├── model_metadata.json                  # Model information
└── comparative visualizations           # Comparison plots
```

## **Benefits of This Approach**

### **Memory Efficiency**
- **No model loading conflicts** - models are processed sequentially
- **GPU memory cleared** after each model evaluation
- **Unified datasets** prevent redundant data generation
- **Scalable** to many models without memory issues

### **Consistency**
- **Same dataset** ensures fair comparison
- **Individual configs** handle model-specific requirements
- **Proven evaluation pipeline** from `evaluate_artifact.py`
- **Standardized metrics** across all models

### **Debugging**
- **Individual model outputs** for detailed inspection
- **Clear separation** of concerns between models
- **Easy to isolate** issues to specific models
- **Comprehensive logging** throughout the process

## **Troubleshooting**

### **Common Issues**

1. **Config file not found**: Ensure all config files exist and paths are correct
2. **Artifact download failure**: Check WandB credentials and artifact accessibility
3. **Memory issues**: The script automatically clears memory, but ensure sufficient GPU memory for individual models
4. **Evaluation failures**: Check individual model outputs for specific error details

### **Debug Mode**
Add debug prints by modifying the script or check the detailed output from each `evaluate_artifact.py` run.

## **Advanced Usage**

### **Custom Dataset Sizes**
Adjust `--n_samples`, `--n_queries`, and `--n_training` based on your needs and available memory.

### **Multiple Models**
The script can handle any number of models - just add more artifacts, configs, and specialist flags.

### **Custom Output Directory**
Use `--run_dir` to specify a custom output location for results.

### **WandB Integration**
The script automatically logs to WandB unless `--no_wandb` is specified, providing easy tracking and comparison of results.

## **Performance Tips**

1. **Use SSD storage** for faster dataset loading/saving
2. **Ensure sufficient GPU memory** for the largest model you'll evaluate
3. **Monitor memory usage** during evaluation
4. **Use appropriate batch sizes** in your config files for optimal performance

This approach provides a robust, memory-efficient way to compare multiple models while ensuring fair and consistent evaluation conditions.


