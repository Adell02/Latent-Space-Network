# VQ-VAE Implementation for Latent Program Networks

This implementation adds Vector Quantization Variational Autoencoder (VQ-VAE) support to the Latent Program Network to prevent posterior collapse and force the decoder to learn meaningful representations from the encoder latents.

## What is VQ-VAE?

VQ-VAE replaces the continuous latent space of standard VAEs with a discrete latent space using vector quantization. This prevents posterior collapse (where the decoder ignores the latent codes) by forcing the encoder to map inputs to discrete codes from a learned codebook.

### Key Benefits:
1. **Prevents Posterior Collapse**: Discrete latents cannot collapse to a single point
2. **Forces Decoder Dependency**: Decoder must use the quantized latents to reconstruct outputs
3. **Improved Training Stability**: More stable than continuous VAEs with KL divergence
4. **Better Latent Utilization**: All codebook entries can be actively used

## How It Works

### 1. Architecture Changes
- **Encoder**: Outputs continuous vectors that are then quantized
- **Vector Quantization Layer**: Maps continuous vectors to nearest discrete codes
- **Decoder**: Receives discrete latent codes instead of continuous distributions

### 2. Loss Function
Instead of KL divergence, VQ-VAE uses:
- **Commitment Loss**: Encourages encoder to commit to codebook entries
- **Codebook Loss**: Updates codebook entries (handled by EMA)

```
L_VQ = L_reconstruction + β * L_commitment
```

### 3. Straight-Through Estimator
Gradients flow through the quantization layer using the straight-through estimator:
```
z_quantized = z_continuous + (quantized - z_continuous).detach()
```

## Configuration

### Basic Setup
Set `vq_vae.enabled = true` in your model settings:

```json
{
  "model_architecture": {
    "vq_vae": {
      "enabled": true,
      "num_embeddings": 512,
      "embedding_dim": 64,
      "commitment_cost": 0.25,
      "decay": 0.99,
      "epsilon": 1e-5,
      "use_ema": true,
      "restart_unused_codes": true,
      "restart_threshold": 1.0
    }
  }
}
```

### Key Parameters

#### `num_embeddings` (int, default: 512)
- Number of discrete codes in the codebook
- Higher values = more expressive latent space
- Should be large enough to capture data diversity
- Typical range: 256-2048

#### `embedding_dim` (int, default: 64)
- Dimension of each codebook entry
- Should match `latent_dim` in model architecture
- Higher dimensions = more expressive codes

#### `commitment_cost` (float, default: 0.25)
- Weight for commitment loss (β in the loss function)
- Encourages encoder to commit to codebook entries
- Higher values = stronger commitment
- Typical range: 0.1-0.5

#### `decay` (float, default: 0.99)
- EMA decay rate for codebook updates
- Higher values = slower codebook updates
- Typical range: 0.95-0.999

#### `use_ema` (bool, default: true)
- Whether to use Exponential Moving Average for codebook updates
- Recommended: true (more stable than gradient updates)

#### `restart_unused_codes` (bool, default: true)
- Whether to restart unused codebook entries
- Prevents "dead codes" that are never used
- Recommended: true

#### `restart_threshold` (float, default: 1.0)
- Threshold for restarting unused codes
- Codes with usage below this threshold are restarted
- Lower values = more aggressive restart

## Training Settings

### Beta Parameter
The `beta` parameter in training settings controls the VQ loss weight:
```json
{
  "training_settings": {
    "beta": 0.25
  }
}
```

- Higher beta = stronger commitment to discrete codes
- Lower beta = more focus on reconstruction
- Typical range: 0.1-0.5

### Disabled Features
Some features are automatically disabled with VQ-VAE:
- **Free-bits mechanism**: Not applicable to discrete latents
- **Contrastive KL margin**: Not meaningful for discrete latents
- **KL-based debug metrics**: Replaced with VQ-specific metrics

## Monitoring and Metrics

### WandB Logging
The implementation logs several VQ-VAE specific metrics:

#### Training Metrics
- `vq_loss`: Total VQ loss (commitment + codebook)
- `vq_codebook_perplexity`: Measure of codebook diversity
- `vq_codebook_usage_entropy`: Entropy of codebook usage
- `vq_codebook_usage_max/min`: Usage statistics

#### Codebook Health
- **Perplexity**: Higher is better (more diverse usage)
- **Usage Entropy**: Higher is better (more uniform usage)
- **Dead Codes**: Codes that are never used (should be minimal)

### Interpreting Metrics

#### Good Signs:
- High codebook perplexity (>50% of num_embeddings)
- High usage entropy (close to log(num_embeddings))
- Stable commitment loss
- No dead codes

#### Warning Signs:
- Low codebook perplexity (<10% of num_embeddings)
- Low usage entropy
- Increasing commitment loss
- Many dead codes

## Usage Examples

### Single Encoder Training
```bash
python main.py train --file_name vqvae_test --settings model_settings_vqvae_example.json
```

### Specialist Training
```bash
python main_specialist.py --mode train --file_name vqvae_specialist --phases A,B
```

### Evaluation
```bash
python main.py eval --file_name vqvae_test --epoch 30
```

## Implementation Details

### File Structure
- `models/vq_vae.py`: Core VQ-VAE implementation
- `models/base_model.py`: Integration with existing models
- `model_settings_vqvae_example.json`: Example configuration

### Key Classes
- `VectorQuantization`: Core quantization layer
- `VQVAEWrapper`: Integration wrapper
- `create_vq_vae_from_settings()`: Factory function

### Compatibility
- Works with both single and multi-encoder models
- Compatible with specialist training phases
- Supports all existing evaluation and visualization tools

## Troubleshooting

### Common Issues

#### Low Codebook Utilization
- Increase `num_embeddings`
- Decrease `commitment_cost`
- Enable `restart_unused_codes`

#### Training Instability
- Increase `decay` (slower codebook updates)
- Decrease `commitment_cost`
- Reduce learning rate

#### Poor Reconstruction
- Increase `commitment_cost`
- Increase `beta` in training settings
- Check codebook perplexity

### Debug Mode
Enable debug logging by setting `debug_kl_metrics: true` in enhanced training settings (though KL metrics won't be available for VQ-VAE).

## References

1. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. arXiv preprint arXiv:1711.00937.

2. Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. arXiv preprint arXiv:1906.00446.

3. Bonnet, A., & Macfarlane, J. (2024). Searching Latent Program Spaces. 