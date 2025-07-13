#!/usr/bin/env python3
"""
Vector Quantization Variational Autoencoder (VQ-VAE) Implementation

This module implements VQ-VAE to prevent posterior collapse by using discrete latent spaces.
The quantization forces the decoder to learn meaningful representations from the encoder.

Key features:
- Exponential Moving Average (EMA) for codebook updates
- Commitment loss to encourage encoder to commit to codebook entries
- Unused code restart mechanism to prevent dead codes
- Efficient straight-through estimator for gradients

Reference: Neural Discrete Representation Learning (van den Oord et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantization(nn.Module):
    """
    Vector Quantization layer with EMA updates and commitment loss.
    
    Args:
        num_embeddings: Number of discrete codes in the codebook
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss (encourages encoder to commit)
        decay: EMA decay rate for codebook updates
        epsilon: Small constant for numerical stability
        use_ema: Whether to use EMA updates for codebook
        restart_unused_codes: Whether to restart unused codes
        restart_threshold: Threshold for restarting unused codes
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        restart_unused_codes: bool = True,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.restart_unused_codes = restart_unused_codes
        self.restart_threshold = restart_threshold
        
        # Initialize codebook
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        
        if use_ema:
            # EMA buffers for codebook updates
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_weight', torch.randn(num_embeddings, embedding_dim))
            self.register_buffer('cluster_usage', torch.zeros(num_embeddings))
        
        # Initialize embeddings
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize codebook embeddings."""
        nn.init.uniform_(self.embeddings, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        if self.use_ema:
            self.ema_weight.data.copy_(self.embeddings.data)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through vector quantization.
        
        Args:
            inputs: Input tensor of shape [..., embedding_dim]
            
        Returns:
            quantized: Quantized tensor (same shape as inputs)
            vq_loss: Vector quantization loss
            encoding_indices: Indices of selected codes
        """
        # Flatten inputs for processing
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to all embeddings
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings**2, dim=1) - \
                   2 * torch.matmul(flat_inputs, self.embeddings.t())
        
        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(input_shape)
        
        # Update codebook if in training mode
        if self.training and self.use_ema:
            self._update_codebook(flat_inputs, encodings, encoding_indices)
        
        # Calculate losses
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        
        if self.use_ema:
            # Only commitment loss for EMA
            vq_loss = self.commitment_cost * commitment_loss
        else:
            # Both codebook and commitment loss for non-EMA
            codebook_loss = F.mse_loss(quantized, inputs.detach())
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.view(input_shape[:-1])
    
    def _update_codebook(self, flat_inputs: torch.Tensor, encodings: torch.Tensor, encoding_indices: torch.Tensor):
        """Update codebook using EMA."""
        # Update cluster sizes
        self.ema_cluster_size.mul_(self.decay).add_(
            torch.sum(encodings, dim=0), alpha=1 - self.decay
        )
        
        # Update embeddings
        dw = torch.matmul(encodings.t(), flat_inputs)
        self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # Normalize embeddings
        n = torch.sum(self.ema_cluster_size)
        cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        self.embeddings.data.copy_(self.ema_weight / cluster_size.unsqueeze(1))
        
        # Track usage for restart mechanism
        if self.restart_unused_codes:
            self.cluster_usage.mul_(self.decay).add_(
                torch.bincount(encoding_indices, minlength=self.num_embeddings).float(),
                alpha=1 - self.decay
            )
            
            # Restart unused codes
            unused_codes = self.cluster_usage < self.restart_threshold
            if unused_codes.any():
                # Replace unused codes with random vectors from input distribution
                n_replace = unused_codes.sum().item()
                if n_replace > 0 and len(flat_inputs) > 0:
                    # Ensure we don't try to replace more codes than we have samples
                    n_available = len(flat_inputs)
                    n_actual_replace = min(n_replace, n_available)
                    
                    # Sample random vectors from current batch
                    random_indices = torch.randperm(n_available)[:n_actual_replace]
                    replacement_vectors = flat_inputs[random_indices]
                    
                    # If we need more replacements than available samples, repeat samples
                    if n_replace > n_available:
                        # Repeat samples to fill all unused codes
                        repeat_factor = (n_replace + n_available - 1) // n_available  # Ceiling division
                        replacement_vectors = replacement_vectors.repeat(repeat_factor, 1)[:n_replace]
                    
                    # Ensure dtype compatibility (for mixed precision training)
                    replacement_vectors = replacement_vectors.to(self.embeddings.dtype)
                    self.embeddings.data[unused_codes] = replacement_vectors
                    if self.use_ema:
                        self.ema_weight.data[unused_codes] = replacement_vectors
                        self.cluster_usage[unused_codes] = self.restart_threshold
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Get current codebook usage statistics."""
        if self.use_ema:
            return self.cluster_usage / (self.cluster_usage.sum() + self.epsilon)
        else:
            return torch.ones(self.num_embeddings) / self.num_embeddings
    
    def get_codebook_perplexity(self) -> float:
        """Calculate codebook perplexity (measure of diversity)."""
        usage = self.get_codebook_usage()
        perplexity = torch.exp(-torch.sum(usage * torch.log(usage + self.epsilon)))
        return perplexity.item()


class VQVAEWrapper(nn.Module):
    """
    Wrapper for VQ-VAE functionality that can be integrated into existing models.
    
    This wrapper handles the transition between continuous and discrete latent spaces
    while maintaining compatibility with existing VAE-based models.
    """
    
    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = None,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        restart_unused_codes: bool = True,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim if embedding_dim is not None else latent_dim
        
        # Add projection layers if input and embedding dimensions differ
        if self.latent_dim != self.embedding_dim:
            self.pre_vq_proj = nn.Linear(self.latent_dim, self.embedding_dim)
            self.post_vq_proj = nn.Linear(self.embedding_dim, self.latent_dim)
        else:
            self.pre_vq_proj = None
            self.post_vq_proj = None
        
        self.vq_layer = VectorQuantization(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
            use_ema=use_ema,
            restart_unused_codes=restart_unused_codes,
            restart_threshold=restart_threshold,
        )
    
    def forward(self, z_continuous: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert continuous latents to discrete latents.
        
        Args:
            z_continuous: Continuous latent tensor
            
        Returns:
            z_discrete: Discrete latent tensor
            vq_loss: Vector quantization loss
            encoding_indices: Indices of selected codes
        """
        # Project to embedding dimension if needed
        if self.pre_vq_proj is not None:
            z_projected = self.pre_vq_proj(z_continuous)
        else:
            z_projected = z_continuous
            
        # Apply vector quantization
        z_quantized, vq_loss, encoding_indices = self.vq_layer(z_projected)
        
        # Project back to latent dimension if needed
        if self.post_vq_proj is not None:
            z_discrete = self.post_vq_proj(z_quantized)
        else:
            z_discrete = z_quantized
            
        return z_discrete, vq_loss, encoding_indices
    
    def get_metrics(self) -> dict:
        """Get VQ-VAE metrics for monitoring."""
        return {
            'codebook_usage': self.vq_layer.get_codebook_usage(),
            'codebook_perplexity': self.vq_layer.get_codebook_perplexity(),
            'num_embeddings': self.vq_layer.num_embeddings,
        }


def create_vq_vae_from_settings(model_architecture: dict) -> Optional[VQVAEWrapper]:
    """
    Create VQ-VAE wrapper from settings configuration.
    
    Args:
        model_architecture: Model architecture settings dict
        
    Returns:
        VQVAEWrapper instance if enabled, None otherwise
    """
    vq_config = model_architecture.get('vq_vae', {})
    
    if not vq_config.get('enabled', False):
        return None
    
    return VQVAEWrapper(
        latent_dim=model_architecture.get('latent_dim', 64),
        embedding_dim=vq_config.get('embedding_dim', model_architecture.get('latent_dim', 64)),
        num_embeddings=vq_config.get('num_embeddings', 512),
        commitment_cost=vq_config.get('commitment_cost', 0.25),
        decay=vq_config.get('decay', 0.99),
        epsilon=vq_config.get('epsilon', 1e-5),
        use_ema=vq_config.get('use_ema', True),
        restart_unused_codes=vq_config.get('restart_unused_codes', True),
        restart_threshold=vq_config.get('restart_threshold', 1.0),
    ) 