# Latent Program Network (LPN)

This directory contains the implementation of the Latent Program Network, a neural architecture designed for solving visual reasoning tasks through program synthesis in a latent space.

## Mathematical Foundations

### 1. Architecture Overview

The LPN consists of three main components:

1. **Encoder**: Maps input-output pairs to a latent program space
2. **Latent Optimization**: Refines the latent program to better explain the data
3. **Decoder**: Executes the latent program to generate outputs

### 2. Core Mathematical Formulation

#### 2.1 Encoder
The encoder learns a variational approximation of the posterior distribution over programs:

```math
q_\phi(z|x, y) = \mathcal{N}(\mu_\phi(x,y), \Sigma_\phi(x,y))
```

where:
- $z$ is the latent program representation
- $\phi$ are the encoder parameters
- $\mu_\phi$ and $\Sigma_\phi$ are neural networks that output the mean and covariance of the distribution

#### 2.2 Decoder
The decoder models the probability distribution of outputs given inputs and latent programs:

```math
p_\theta(y|x, z)
```

where:
- $\theta$ are the decoder parameters
- The decoder directly predicts output pixels without using a domain-specific language

#### 2.3 Latent Optimization
Given $n$ input-output pairs $\{(x_i, y_i)\}_{i=1 \dots n}$, the optimization process finds $z'$ that maximizes:

```math
z' \in \arg \max_z \sum_{i=1}^n \log p_\theta(y_i|x_i, z)
```

### 3. Training Objective

The model is trained using a combination of reconstruction loss and KL divergence:

```math
\mathcal{L_{\text{total}}}(\phi, \theta) = \mathcal{L_{\text{rec}}}(\phi, \theta) + \beta \mathcal{L_{\text{KL}}}(\phi)
```

where:

#### 3.1 Reconstruction Loss
```math
\mathcal{L_{\text{rec}}}(\phi, \theta) = \sum_{i=1}^n -\log p_\theta(y_i | x_i, z_i')
```

#### 3.2 KL Divergence Loss
```math
\mathcal{L_{\text{KL}}}(\phi) = \sum_{i=1}^n D_{\text{KL}} \left( q_\phi(z | x_i, y_i) \parallel \mathcal{N}(0, I) \right)
```

### 4. Latent Optimization Methods

#### 4.1 Gradient Ascent
The primary optimization method uses gradient ascent in the latent space:

```math
z_0' = \frac{1}{n} \sum_{i=1}^n z_i
```
```math
z_k' = z_{k-1}' + \alpha \cdot \nabla_z \sum_{i=1}^n \log p_\theta(y_i|x_i, z)|_{z=z_{k-1}'}
```

where:
- $\alpha$ is the learning rate
- $K$ is the number of optimization steps
- The gradient is computed through the decoder network

### 5. Implementation Details

The implementation in `base_model.py` includes:

1. **TransformerEncoder**: Processes input sequences to generate latent distributions
2. **TransformerDecoder**: Generates outputs from latent programs and inputs
3. **LatentProgramNetwork**: Combines encoder and decoder with optimization

Key hyperparameters:
- `LATENT_DIM`: Dimensionality of the latent space
- `HIDDEN_DIM`: Size of hidden layers
- `NUM_LAYERS`: Number of transformer layers
- `NUM_HEADS`: Number of attention heads
- `OPTIMIZE_Z_NUM_STEPS`: Number of gradient steps for latent optimization
- `OPTIMIZE_Z_LR`: Learning rate for latent optimization

### 6. Usage Example

```python
from models.base_model import LatentProgramNetwork

# Initialize model
model = LatentProgramNetwork(
    input_dim=1,
    latent_dim=256,
    hidden_dim=256,
    num_layers=6,
    num_heads=8
)

# Training
results, model = main_training()

# Evaluation
eval_results = evaluate_model_on_new_data(
    model,
    keys=['017c7c7b', '00d62c1b'],
    n_values=100
)
```

## References

1. Bonnet, A., & Macfarlane, J. (2024). Searching Latent Program Spaces.
2. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
