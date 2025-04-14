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
python main.py --mode train
```

### Evaluation
```bash
python main.py --mode eval --run_dir <run_directory> --keys <problem_keys> --n_values <number_of_examples>
```

### Visualization
```bash
python main.py --mode visualize --run_dir <run_directory>
```

### Combined Operations
```bash
python main.py --mode all --run_dir <run_directory>
```

### Command Line Arguments

- `--mode`: Operation mode (train/eval/visualize/all)
- `--run_dir`: Directory containing model checkpoints (required for eval/visualize)
- `--keys`: Problem keys for evaluation (default: ['017c7c7b','00d62c1b','007bbfb7'])
- `--n_values`: Number of examples for evaluation (default: 100)
- `--epoch`: Specific epoch to load for evaluation (default: 49)

## Example Usage

1. **Train a new model**:
   ```bash
   python main.py --mode train
   ```

2. **Evaluate on specific problems**:
   ```bash
   python main.py --mode eval --run_dir original_w_gradient_ascent --keys 017c7c7b 00d62c1b --n_values 50
   ```

3. **Visualize results**:
   ```bash
   python main.py --mode visualize --run_dir original_w_gradient_ascent
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
