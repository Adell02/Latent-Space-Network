# Utility Functions

This directory contains utility functions for the Latent Program Network implementation. The utilities are organized into several modules, each handling specific aspects of the system.

## Module Overview

### 1. model_utils.py
Core model-related utilities:
- `count_model_parameters`: Analyzes and displays model parameter counts
- `create_run_directory`: Creates unique directories for model runs
- `set_seed`: Ensures reproducibility by setting random seeds
- `setup_logging`: Configures logging for training runs
- `prepare_dataloader`: Creates PyTorch DataLoader for training
- `save_checkpoint`: Saves model checkpoints during training
- `save_results`: Saves training results to disk
- `perform_latent_analysis`: Analyzes and visualizes latent space
- `load_model`: Loads saved model checkpoints
- `evaluate_model`: Evaluates model performance on test data

### 2. visualizers.py
Visualization utilities for data and model analysis:
- `visualize_full_transformation`: Visualizes input-output grid transformations
- `print_sequence_info`: Displays sequence information in tabular format
- `visualize_sequence_reconstruction`: Plots original vs reconstructed sequences
- `plot_training_and_latent`: Visualizes training loss and latent space
- `plot_latent_analysis`: Detailed latent space analysis plots
- `plot_epoch_accuracies`: Tracks accuracy metrics over training
- `plot_reconstructions`: Visualizes grid reconstructions
- `visualize_all_results`: Comprehensive results visualization
- `plot_evaluation_results`: Evaluation metrics visualization
- `visualize_stored_results`: Loads and visualizes stored results

### 3. data_preparation.py
Data processing utilities:
- `transform_grid_to_sequence`: Converts 2D grids to sequences
- `prepare_input_output_pair`: Prepares input-output pairs for the encoder

### 4. latent_functions.py
Latent space optimization utilities:
- `optimize_latent_z`: Optimizes latent variables via gradient descent

## Usage Examples

### Model Training and Evaluation
```python
from utils.model_utils import create_run_directory, setup_logging, save_checkpoint
from utils.visualizers import plot_training_and_latent

# Create run directory
run_dir = create_run_directory()

# Setup logging
logger = setup_logging(run_dir)

# Save checkpoint
save_checkpoint(model, optimizer, epoch, loss, run_dir)

# Visualize results
plot_training_and_latent(results)
```

### Data Preparation
```python
from utils.data_preparation import transform_grid_to_sequence, prepare_input_output_pair

# Transform grid to sequence
sequence = transform_grid_to_sequence(grid)

# Prepare input-output pair
full_sequence = prepare_input_output_pair(input_grid, output_grid)
```

### Latent Space Optimization
```python
from utils.latent_functions import optimize_latent_z

# Optimize latent variables
optimized_z = optimize_latent_z(model, input_seq, target_seq, num_steps=100, lr=0.01)
```

## Notes
- This directory will be updated as the repository grows
- New utility functions should be documented here
- Each module should maintain a clear separation of concerns
- All functions should include proper docstrings and type hints 