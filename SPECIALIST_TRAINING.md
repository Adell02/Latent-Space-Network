# Specialist Training System

This repository now includes a specialist training system for multi-encoder models that implements a 3-phase training approach for better domain specialization and knowledge sharing.

## Overview

The specialist training system (`train_specialist.py`) implements a structured 3-phase approach:

- **Phase A**: Pre-train each encoder individually on its assigned key subset
- **Phase B**: Freeze encoders, train shared decoder on mixed data from all domains  
- **Phase C**: Joint fine-tuning with reduced encoder learning rate

This approach allows encoders to develop domain-specific representations while learning to work together through a shared decoder.

## Requirements

- Multi-encoder model configuration (`num_encoders > 1` in model architecture settings)
- Multiple training keys for domain specialization
- Existing settings configuration via `utils.settings_manager`

## Usage

### Recommended: Use Main Specialist Script

```bash
# Complete workflow (train + evaluate + visualize)
python main_specialist.py --mode all --file_name my_experiment

# Train all three phases
python main_specialist.py --mode train --file_name my_experiment

# Train specific phases only
python main_specialist.py --mode train --file_name my_experiment --phases A,B

# Evaluate trained model
python main_specialist.py --mode eval --file_name my_experiment --epoch phase_c_final

# Visualize results
python main_specialist.py --mode visualize --file_name my_experiment --epoch phase_c_final
```

### Direct Training Script (Advanced)

```bash
# Run all three phases
python train_specialist.py --run_name specialist_experiment

# Run specific phases only
python train_specialist.py --phases A,B --run_name encoder_pretrain_only

# Run only joint fine-tuning (assumes Phase A & B completed)
python train_specialist.py --phases C --run_name joint_finetune
```

### Configuration

The specialist trainer uses `model_specialist_settings.json` for configuration. Key settings include:

```json
{
  "model_architecture": {
    "num_encoders": 4,
    "latent_dim": 128,
    // ... other model settings
  },
  "data_settings": {
    "training_keys": ["00d62c1b", "1cf80156", "25d8a9c8", "0a938d79"],
    "n": 10,
    "run_base_dir": "runs_specialist"
  },
  "specialist_training": {
    "phase_a": {"epochs": 20},
    "phase_b": {"epochs": 30}, 
    "phase_c": {"epochs": 10, "encoder_lr_multiplier": 0.05},
    "evaluation_between_phases": true,
    "phases_to_run": ["A", "B", "C"]
  }
}
```

All training parameters (epochs, learning rates, batch sizes) are now configurable through the settings file.

## Training Phases

### Phase A: Encoder Pre-training
- **Duration**: 20 epochs per encoder (configurable)
- **Objective**: Each encoder learns domain-specific representations
- **Training**: Individual encoders trained on their assigned key subsets
- **Output**: Individual encoder checkpoints (`encoder_0.ckpt`, `encoder_1.ckpt`, etc.)

```
Encoder 0: [key1, key2] → encoder_0.ckpt
Encoder 1: [key3, key4] → encoder_1.ckpt  
Encoder 2: [key5, key6] → encoder_2.ckpt
```

### Phase B: Decoder Training
- **Duration**: 30 epochs (configurable)
- **Objective**: Learn to decode from combined encoder representations
- **Training**: Frozen encoders, decoder trained on mixed data using PoE
- **Output**: Decoder checkpoint (`decoder.ckpt`)

### Phase C: Joint Fine-tuning
- **Duration**: 10 epochs (configurable)
- **Objective**: Fine-tune entire model while preserving specialization
- **Training**: All parameters trainable, but encoders use reduced learning rate (0.05x)
- **Output**: Complete model checkpoint (`full_joint.ckpt`)

## File Structure

```
utils/
├── training_helpers.py        # Helper functions for specialist training
├── settings_manager.py        # Updated with specialist settings support
├── model_utils.py            # Existing model utilities
└── ...

main_specialist.py            # Main interface (train + eval + visualize)
train_specialist.py           # Core specialist training script
model_specialist_settings.json # Specialist training configuration
training.py                   # Original training script (unchanged)
model_settings.json           # Original configuration (unchanged)
```

## Key Features

### Modular Phase Execution
Run individual phases or combinations as needed:
- Development: `--phases A` (test encoder pre-training)
- Research: `--phases A,B` (study encoder+decoder interaction)
- Production: `--phases A,B,C` (complete specialist training)

### Intelligent Data Mixing
- **Phase A**: Key-based dataset splitting ensures domain specialization
- **Phase B & C**: Round-robin mixed data loading for balanced training

### Parameter Management
- **Phase A**: Only target encoder trainable
- **Phase B**: Only decoder trainable  
- **Phase C**: All parameters trainable with differentiated learning rates

### Checkpointing
- Individual component checkpoints for flexibility
- Phase-specific checkpoints for resumption
- Complete model checkpoint for deployment

### Monitoring
- Comprehensive logging for each phase
- WandB integration with phase-specific metrics
- Evaluation between phases for progress tracking

## Advanced Usage

### Custom Phase Durations
Modify epoch counts in the training functions:
```python
# In train_specialist.py
phase_a_results = train_phase_a_pretraining(
    model, dataset_splits, device, logger, wandb_logger, run_dir, 
    phase_epochs=50  # Custom duration
)
```

### Custom Learning Rate Multiplier
Adjust encoder learning rate in Phase C:
```python
phase_c_results = train_phase_c_joint_finetuning(
    model, dataset_splits, device, logger, wandb_logger, run_dir,
    phase_epochs=10, encoder_lr_mult=0.01  # Even lower encoder LR
)
```

### Resume Training
```bash
# Resume from specific phase (feature to be implemented)
python train_specialist.py --resume_from_phase B --run_name resumed_training
```

## Monitoring and Evaluation

### WandB Metrics
The system logs comprehensive metrics:
- Phase A: `phase_a/encoder_{i}_loss`
- Phase B: `phase_b/decoder_loss` 
- Phase C: `phase_c/joint_loss`
- Evaluation: Standard evaluation metrics between phases

### Checkpoints
```
run_directory/
├── encoder_0.ckpt           # Phase A outputs
├── encoder_1.ckpt
├── encoder_2.ckpt
├── decoder.ckpt             # Phase B output
├── full_joint.ckpt          # Phase C output
├── checkpoint_pretrain_*.pt # Phase checkpoints
├── checkpoint_decoder_*.pt
├── checkpoint_joint_ft_*.pt
├── results.pkl             # Complete training results
└── evaluation_results.pkl  # Evaluation results
```

### Evaluation Support

The system provides comprehensive evaluation capabilities:

```bash
# Evaluate after each phase
python main_specialist.py --mode eval --file_name experiment --epoch phase_a_final
python main_specialist.py --mode eval --file_name experiment --epoch phase_b_final  
python main_specialist.py --mode eval --file_name experiment --epoch phase_c_final

# Custom evaluation settings
python main_specialist.py --mode eval --file_name experiment \
  --epoch phase_c_final --keys 00d62c1b 1cf80156 \
  --n_eval_samples 5 --n_eval_queries 200
```

**Evaluation Epochs:**
- `phase_a_final`: After encoder pre-training
- `phase_b_final`: After decoder training  
- `phase_c_final`: After joint fine-tuning (recommended for best performance)
- `<number>`: Specific training epoch number

## Benefits

1. **Domain Specialization**: Encoders develop expertise in specific problem types
2. **Knowledge Sharing**: Shared decoder learns to combine domain knowledge
3. **Stable Training**: Phased approach prevents interference between specialization and integration
4. **Flexibility**: Individual phases can be modified or skipped based on needs
5. **Debugging**: Clear separation allows easy identification of issues in specific components

## Comparison with Standard Training

| Aspect | Standard Training | Specialist Training |
|--------|------------------|-------------------|
| Encoder Training | Simultaneous | Sequential (Phase A) |
| Decoder Training | Joint | Separate (Phase B) |
| Final Integration | From scratch | Guided (Phase C) |
| Domain Specialization | Emergent | Explicit |
| Training Stability | Variable | High |
| Interpretability | Limited | High |

## Quick Start

1. **Configure**: Edit `model_specialist_settings.json` with your training keys and model architecture
2. **Train**: Run `python main_specialist.py --mode train --file_name my_experiment`
3. **Evaluate**: Run `python main_specialist.py --mode eval --file_name my_experiment --epoch phase_c_final`
4. **Visualize**: Run `python main_specialist.py --mode visualize --file_name my_experiment --epoch phase_c_final`

Or do all at once: `python main_specialist.py --mode all --file_name my_experiment`

## Advanced Usage

1. **Phase Selection**: `--phases A,B` to run only specific phases
2. **Resume Training**: `--resume_from_phase B` to continue from a specific phase
3. **Custom Evaluation**: Adjust evaluation keys, samples, and queries via command line
4. **Settings Tuning**: Modify `model_specialist_settings.json` for phase durations and learning rates
5. **WandB Integration**: Monitor training progress and compare with standard approaches

## Next Steps

1. **Experiment**: Try specialist training on your multi-encoder setup with the provided interface
2. **Monitor**: Use WandB to compare specialist vs standard training results
3. **Tune**: Adjust phase durations and learning rates in the settings file
4. **Extend**: Add custom evaluation or specialized loss functions per phase
5. **Research**: Study phase-specific latent representations and encoder specialization

The specialist training system provides a structured approach to multi-encoder training that promotes both specialization and collaboration, leading to more robust and interpretable models. 