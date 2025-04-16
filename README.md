# Latent Space Network

This repository contains the implementation of a Latent Program Network for solving ARC (Abstraction and Reasoning Corpus) tasks. The model learns to represent and solve visual reasoning tasks in a latent space.

## Repository Structure

```
Latent-Space-Network/
├── models/                 # Model architecture definitions
│   └── base_model.py      # Main model implementation
├── utils/                 # Utility functions
│   ├── model_utils.py     # Model loading, saving, and evaluation
│   ├── visualizers.py     # Visualization tools
│   ├── data_preparation.py # Data processing utilities
│   └── latent_functions.py # Latent space operations
├── re_arc/               # ARC task processing
├── runs_re_arc/          # Directory for storing training runs
├── main.py              # Main script for training and evaluation
├── requirements.txt     # Python dependencies
└── requirements_win.txt # Windows-specific dependencies
```

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   - For Linux/Mac:
     ```bash
     pip install -r requirements.txt
     ```
   - For Windows:
     ```bash
     pip install -r requirements_win.txt
     ```

3. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

The `main.py` script provides several modes of operation:

### Training
```bash
python main.py --mode train --file_name <run_directory>
```

### Evaluation
```bash
python main.py --mode eval --file_name <run_directory> --keys <problem_keys> --n_eval_samples <number_of_samples> --n_eval_queries <number_of_queries> --epoch <epoch_number>
```

### Visualization
```bash
python main.py --mode visualize --file_name <run_directory> --visualize_n_values <number_of_values>
```

### Combined Operations
```bash
python main.py --mode all --file_name <run_directory> --keys <problem_keys> --n_eval_samples <number_of_samples> --n_eval_queries <number_of_queries> --epoch <epoch_number> --visualize_n_values <number_of_values>
```

### Command Line Arguments

- `--mode`: Operation mode (train/eval/visualize/all) or combination of modes
- `--file_name`: Directory for storing/containing model checkpoints and results (required)
- `--keys`: Problem keys for evaluation (space-separated, default: ["00d62c1b"])
- `--n_eval_samples`: Number of input-output pairs for Z optimization during evaluation (default: 8)
- `--n_eval_queries`: Number of queries for inference (default: 1)
- `--epoch`: Specific epoch to load for evaluation (default: 299)
- `--visualize_n_values`: Number of input-output pairs for visualization (default: 2)

## Example Usage

1. **Train a new model**:
   ```bash
   python main.py --mode train --file_name my_training_run
   ```

2. **Evaluate on specific problems**:
   ```bash
   python main.py --mode eval --file_name my_training_run --keys 00d62c1b 017c7c7b --n_eval_samples 8 --n_eval_queries 1 --epoch 299
   ```

3. **Evaluate and visualize results**:
   ```bash
   python main.py --mode eval visualize --file_name my_training_run --keys 00d62c1b --n_eval_samples 8 --n_eval_queries 1 --epoch 299 --visualize_n_values 2
   ```

## Output

- Training results are saved in the `runs_re_arc` directory
- Each run creates a new directory with:
  - Model checkpoints
  - Training metrics
  - Evaluation results
  - Visualizations

## Notes

- The model uses CUDA if available, falling back to CPU otherwise
- Training progress and metrics are logged to the console
- Visualization includes:
  - Training loss curves
  - Latent space analysis
  - Reconstruction examples
  - Evaluation metrics

## Requirements

- Python 3.8+
- PyTorch 2.6.0+
- CUDA-capable GPU (recommended)
- See requirements.txt for full dependency list
