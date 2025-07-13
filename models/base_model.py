#!/usr/bin/env python3
"""
Base Model for Latent Program Network (LPN)

This module implements the core architecture of the LPN, including:
- TransformerEncoder: Processes input sequences to generate latent distributions
- TransformerDecoder: Generates outputs from latent programs and inputs
- LatentProgramNetwork: Combines encoder and decoder with optimization
- VQ-VAE support: Optional discrete latent space to prevent posterior collapse

The model supports both single and multi-encoder configurations.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple, List, Union, Optional, Dict, Any
import copy
import numpy as np

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (set_seed)
from utils.settings_manager import settings

# Import VQ-VAE components
from models.vq_vae import create_vq_vae_from_settings, VQVAEWrapper


#########################################
# TUNABLE SETTINGS
#########################################

def get_current_settings():
    """Get current settings from settings manager (for sweep compatibility)."""
    # Get settings from settings manager
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    
    return {
        'data_settings': data_settings,
        'model_architecture': model_architecture,
        'training_settings': training_settings,
        'latent_optimization': latent_optimization
    }

# For backward compatibility, expose commonly used constants as functions
def get_num_encoders():
    return get_current_settings()['model_architecture'].get('num_encoders', 1)

def get_latent_dim():
    return get_current_settings()['model_architecture']['latent_dim']

def get_encoder_hidden_dim():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('encoder_hidden_dim', model_arch.get('hidden_dim', 96))

def get_decoder_hidden_dim():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('decoder_hidden_dim', model_arch.get('hidden_dim', 96))

def get_encoder_layers():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('encoder_layers', model_arch.get('num_layers', 2))

def get_decoder_layers():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('decoder_layers', model_arch.get('num_layers', 2))

def get_encoder_heads():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('encoder_heads', model_arch.get('num_heads', 6))

def get_decoder_heads():
    model_arch = get_current_settings()['model_architecture']
    return model_arch.get('decoder_heads', model_arch.get('num_heads', 6))

def get_dropout():
    return get_current_settings()['model_architecture']['dropout']

def get_max_length():
    return get_current_settings()['model_architecture']['max_length']

def get_encoder_max_length():
    return get_current_settings()['model_architecture']['encoder_max_length']

def get_decoder_max_length():
    return get_current_settings()['model_architecture']['decoder_max_length']

def get_training_keys():
    data_settings = get_current_settings()['data_settings']
    training_keys = data_settings.get('training_keys', [data_settings.get('key', None)])
    if training_keys is None or not training_keys[0]:
        raise ValueError("No training keys specified in data_settings.")
    return training_keys

def get_training_seed():
    return get_current_settings()['data_settings']['training_seed']

def get_beta():
    return get_current_settings()['training_settings']['beta']

# Legacy constants for immediate backward compatibility (will be removed gradually)
# These should not be used in new code - use the getter functions instead
NUM_ENCODERS = 1  # Default fallback
LATENT_DIM = 128  # Default fallback
ENCODER_HIDDEN_DIM = 96  # Default fallback
DECODER_HIDDEN_DIM = 96  # Default fallback
ENCODER_LAYERS = 2  # Default fallback
DECODER_LAYERS = 2  # Default fallback
ENCODER_HEADS = 6  # Default fallback
DECODER_HEADS = 6  # Default fallback
DROPOUT = 1e-6  # Default fallback
MAX_LENGTH = 902  # Default fallback
ENCODER_MAX_LENGTH = 1805  # Default fallback
DECODER_MAX_LENGTH = 902  # Default fallback
BETA = 0.01  # Default fallback

# Initialize with current settings
try:
    current_settings = get_current_settings()
    data_settings = current_settings['data_settings']
    model_architecture = current_settings['model_architecture']
    training_settings = current_settings['training_settings']
    latent_optimization = current_settings['latent_optimization']

    # Data settings
    TRAINING_KEYS = get_training_keys()
    TRAINING_SEED = get_training_seed()
    N_EXAMPLES_PER_TASK = data_settings['n']

    # Model architecture settings
    NUM_ENCODERS = get_num_encoders()
    LATENT_DIM = get_latent_dim()
    ENCODER_HIDDEN_DIM = get_encoder_hidden_dim()
    DECODER_HIDDEN_DIM = get_decoder_hidden_dim()
    ENCODER_LAYERS = get_encoder_layers()
    DECODER_LAYERS = get_decoder_layers()
    ENCODER_HEADS = get_encoder_heads()
    DECODER_HEADS = get_decoder_heads()
    DROPOUT = get_dropout()
    MAX_LENGTH = get_max_length()
    ENCODER_MAX_LENGTH = get_encoder_max_length()
    DECODER_MAX_LENGTH = get_decoder_max_length()

    # Training settings
    BATCH_SIZE = training_settings['batch_size']
    NUM_EPOCHS = training_settings['num_epochs']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = get_beta()

    # Latent optimization settings
    OPTIMIZE_Z = latent_optimization['training']['enabled']
    OPTIMIZE_Z_NUM_STEPS = latent_optimization['training']['num_steps']
    OPTIMIZE_Z_LR = latent_optimization['training']['learning_rate']
    OPTIMIZE_Z_INFERENCE = latent_optimization['inference']['enabled']
    OPTIMIZE_Z_INFERENCE_NUM_STEPS = latent_optimization['inference']['num_steps']
    OPTIMIZE_Z_INFERENCE_LR = latent_optimization['inference']['learning_rate']
except Exception as e:
    print(f"Warning: Could not load settings at module level: {e}")
    # Use defaults already set above

set_seed(get_training_seed())

# -------------------------------------------------
#  Low‑level helper: diagonal‑Gaussian PoE
# -------------------------------------------------

def gaussian_poe(mu: torch.Tensor, logvar: torch.Tensor, debug=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiply K diagonal Gaussians.
    Args
    -----
    mu      : (K, B, D)
    logvar  : (K, B, D)
    debug   : bool, whether to print debugging information
    Returns
    -------
    fused_mu, fused_logvar : (B, D)
    """
    if debug:
        print(f"PoE Debug: Input shapes - mu: {mu.shape}, logvar: {logvar.shape}")
        print(f"PoE Debug: mu stats - min: {mu.min():.4f}, max: {mu.max():.4f}, mean: {mu.mean():.4f}")
        print(f"PoE Debug: logvar stats - min: {logvar.min():.4f}, max: {logvar.max():.4f}, mean: {logvar.mean():.4f}")
        
        # Check if encoders are producing identical outputs
        if mu.shape[0] > 1:
            mu_diff = torch.abs(mu[0] - mu[1]).mean()
            logvar_diff = torch.abs(logvar[0] - logvar[1]).mean()
            print(f"PoE Debug: Encoder differences - mu_diff: {mu_diff:.6f}, logvar_diff: {logvar_diff:.6f}")
            
            if mu_diff < 1e-6 and logvar_diff < 1e-6:
                print("WARNING: Encoders are producing nearly identical outputs!")
    
    precision   = torch.exp(-logvar)            # Σ⁻¹
    fused_var   = 1. / precision.sum(0)         # (B,D)
    fused_mu    = fused_var * (precision * mu).sum(0)
    fused_logvar = fused_var.log()
    
    if debug:
        print(f"PoE Debug: Output shapes - fused_mu: {fused_mu.shape}, fused_logvar: {fused_logvar.shape}")
        print(f"PoE Debug: fused_mu stats - min: {fused_mu.min():.4f}, max: {fused_mu.max():.4f}, mean: {fused_mu.mean():.4f}")
    
    return fused_mu, fused_logvar

def compute_encoder_covariance_traces(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute trace of covariance matrix (sum of variances) for each encoder.
    
    Args:
        mu (torch.Tensor): Encoder means, shape (K, B, D) [unused but kept for compatibility]
        logvar (torch.Tensor): Encoder log variances, shape (K, B, D)
    
    Returns:
        torch.Tensor: Covariance traces for each encoder, shape (K, B)
    """
    with torch.no_grad():
        # Trace of diagonal covariance = sum of variances = sum of exp(log_var)
        variances = torch.exp(logvar)  # Shape (K, B, D)
        traces = variances.sum(dim=2)  # Sum over latent dimensions, shape (K, B)
        return traces

# Backward compatibility alias
compute_encoder_influence_metrics = compute_encoder_covariance_traces

##############################
# Define Model Components
##############################

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = None, num_layers: int = None, 
                 num_heads: int = None, dropout: float = None, max_length: int = None):
        super().__init__()
        
        # Use current settings if parameters not provided
        if hidden_dim is None:
            hidden_dim = get_encoder_hidden_dim()
        if num_layers is None:
            num_layers = get_encoder_layers()
        if num_heads is None:
            num_heads = get_encoder_heads()
        if dropout is None:
            dropout = get_dropout()
        if max_length is None:
            max_length = get_encoder_max_length()
            
        # Embedding tables
        self.color_embedding = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)
        self.shape_embedding = nn.Embedding(num_embeddings=31, embedding_dim=hidden_dim)
        self.cls_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        # Positional embeddings (factorized into row, column, and channel components)
        self.row_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
        self.col_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
        self.channel_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim)

        # Transformer encoder (pre-layer normalization)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enable gradient checkpointing if specified in settings
        current_model_arch = get_current_settings()['model_architecture']
        if current_model_arch.get('use_gradient_checkpointing', False):
            for mod in self.transformer_encoder.layers:
                mod.use_checkpoint = True  # Enables gradient checkpointing

        # Output projections for latent distribution
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # VQ-VAE support
        self.vq_vae = create_vq_vae_from_settings(current_model_arch)
        self.use_vq_vae = self.vq_vae is not None
        
        if self.use_vq_vae:
            # For VQ-VAE, we only need one projection to continuous space before quantization
            self.fc_latent = nn.Linear(hidden_dim, get_latent_dim())
            print(f"✓ VQ-VAE enabled with {self.vq_vae.vq_layer.num_embeddings} embeddings")
        else:
            # Standard VAE projections
            self.fc_mu = nn.Linear(hidden_dim, get_latent_dim())
            self.fc_log_var = nn.Linear(hidden_dim, get_latent_dim())

    def create_padding_mask(self, shape_values: torch.Tensor) -> torch.Tensor:
        """Create padding mask based on shape values"""
        batch_size = shape_values.size(0)
        # Ensure rows and cols are integers
        rows = shape_values[:, 0].long().cpu().numpy()
        cols = shape_values[:, 1].long().cpu().numpy()

        masks = []
        for b in range(batch_size):
            r, c = int(rows[b]), int(cols[b])
            mask = torch.zeros(30, 30, dtype=torch.bool, device=shape_values.device)
            mask[:r, :c] = True
            masks.append(mask.flatten())

        return torch.stack(masks)

    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_seq.size(0)
        device = input_seq.device
        
        # Validate and clamp input values to prevent embedding index errors
        # Grid tokens (0-899) should be in range [0, 9]
        input_grid_tokens = torch.clamp(input_seq[:, :900], 0, 9).long()
        target_grid_tokens = torch.clamp(target_seq[:, :900], 0, 9).long()
        
        # Shape tokens (900-902) should be in range [0, 30]
        input_shape_tokens = torch.clamp(input_seq[:, 900:902], 0, 30).long()
        target_shape_tokens = torch.clamp(target_seq[:, 900:902], 0, 30).long()
        
        # Updated indexing: grid tokens first (0-899), then shape tokens (900:902)
        input_color_emb = self.color_embedding(input_grid_tokens)
        input_shape_emb = self.shape_embedding(input_shape_tokens)
        target_color_emb = self.color_embedding(target_grid_tokens)
        target_shape_emb = self.shape_embedding(target_shape_tokens)

        # Create padding masks using the shape tokens
        input_mask = self.create_padding_mask(input_shape_tokens)
        target_mask = self.create_padding_mask(target_shape_tokens)

        # Create position indices for a 30x30 grid
        pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
        pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
        row_emb = self.row_embedding(pos_i)
        col_emb = self.col_embedding(pos_j)

        # Create channel embeddings: 0 for input, 1 for target
        input_channel_emb = self.channel_embedding(torch.zeros(1, dtype=torch.long, device=device))
        target_channel_emb = self.channel_embedding(torch.ones(1, dtype=torch.long, device=device))

        # Combine positional embeddings and reshape to flattened grid
        input_pos_emb = (row_emb + col_emb + input_channel_emb).view(batch_size, 900, -1)
        target_pos_emb = (row_emb + col_emb + target_channel_emb).view(batch_size, 900, -1)

        # Combine embeddings with positional information
        input_emb = input_color_emb + input_pos_emb
        target_emb = target_color_emb + target_pos_emb

        # Append the shape embeddings after the grid tokens and add a CLS token at the end
        cls_emb = self.cls_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        combined_emb = torch.cat([input_emb, input_shape_emb, target_emb, target_shape_emb, cls_emb], dim=1)

        # Create attention mask (for grid tokens we use input_mask and target_mask, and ones for shape/CLS tokens)
        combined_mask = torch.cat([
            input_mask,
            torch.ones(batch_size, 2, dtype=torch.bool, device=device),
            target_mask,
            torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        ], dim=1)

        encoder_output = self.transformer_encoder(combined_emb, src_key_padding_mask=~combined_mask)
        cls_output = self.layer_norm(encoder_output[:, -1])
        
        if self.use_vq_vae:
            # VQ-VAE path: continuous -> discrete quantization
            z_continuous = self.fc_latent(cls_output)
            z_quantized, vq_loss, encoding_indices = self.vq_vae(z_continuous)
            # Return quantized latent and VQ loss (stored in log_var position for compatibility)
            return z_quantized, vq_loss.unsqueeze(0).expand(batch_size, -1) if vq_loss.dim() == 0 else vq_loss
        else:
            # Standard VAE path
            mu = self.fc_mu(cls_output)
            log_var = self.fc_log_var(cls_output)
            return mu, log_var

    def get_vq_metrics(self) -> Optional[Dict[str, Any]]:
        """Get VQ-VAE metrics if enabled."""
        if self.use_vq_vae:
            return self.vq_vae.get_metrics()
        return None

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = None, num_layers: int = None, 
                 num_heads: int = None, dropout: float = None):
        super().__init__()
        
        # Use current settings if parameters not provided
        if hidden_dim is None:
            hidden_dim = get_decoder_hidden_dim()
        if num_layers is None:
            num_layers = get_decoder_layers()
        if num_heads is None:
            num_heads = get_decoder_heads()
        if dropout is None:
            dropout = get_dropout()
            
        self.hidden_dim = hidden_dim

        # Embeddings for teacher forcing
        self.output_shape_embedding = nn.Embedding(num_embeddings=31, embedding_dim=hidden_dim)
        self.output_grid_embedding = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)

        # Positional embeddings for grid positions (we reuse row and col embeddings)
        self.row_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
        self.col_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)

        # Start token embedding (used for autoregressive mode)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Latent projection to initial decoder memory
        self.latent_projection = nn.Linear(LATENT_DIM, hidden_dim)

        # Memory projection (combines input and latent info)
        self.memory_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Enable gradient checkpointing if specified in settings
        current_model_arch = get_current_settings()['model_architecture']
        if current_model_arch.get('use_gradient_checkpointing', False):
            for mod in self.transformer_decoder.layers:
                mod.use_checkpoint = True

        # Output projections: one for shape tokens and one for grid tokens
        self.shape_output = nn.Linear(hidden_dim, 31)  # For shape values (indices 0-30)
        self.grid_output = nn.Linear(hidden_dim, 10)   # For grid values (indices 0-9)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def prepare_input_memory(self, z: torch.Tensor, input_seq: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Prepare memory from input sequence and latent vector."""
        batch_size = input_seq.size(0)
        device = input_seq.device

        # Updated indexing: grid tokens first, then shape tokens
        input_grid = input_seq[:, :900]
        input_shapes = input_seq[:, 900:902]

        # Add noise to input during training to force latent dependency
        if training and torch.rand(1).item() < 0.3:  # 30% chance during training
            # Add noise to grid tokens to prevent decoder from just copying
            noise_scale = 0.1
            grid_noise = torch.randint_like(input_grid.long(), 0, 2) * noise_scale  # Binary noise
            input_grid = input_grid + grid_noise
            input_grid = torch.clamp(input_grid, 0, 9)  # Keep in valid range [0,9]
            
            # Occasionally mask some input tokens entirely
            if torch.rand(1).item() < 0.2:  # 20% of the 30% noise cases
                mask_tokens = torch.rand(input_grid.shape, device=device) < 0.1  # Mask 10% of tokens
                input_grid = input_grid.masked_fill(mask_tokens, 0)  # Replace with 0 (background)

        # Clamp input values to ensure valid embedding indices
        input_grid = torch.clamp(input_grid, 0, 9)  # Grid tokens should be 0-9
        input_shapes = torch.clamp(input_shapes, 0, 30)  # Shape tokens should be 0-30

        # For memory we use the same embeddings as in the encoder
        grid_emb = self.output_grid_embedding(input_grid.long())
        shape_emb = self.output_shape_embedding(input_shapes.long())

        # Create positional embeddings for grid tokens
        pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
        pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
        pos_emb = (self.row_embedding(pos_i) + self.col_embedding(pos_j)).view(batch_size, 900, -1)
        grid_emb = grid_emb + pos_emb

        # Concatenate shape embeddings and grid embeddings for memory.
        memory_input = torch.cat([shape_emb, grid_emb], dim=1)

        latent_emb = self.latent_projection(z)
        
        # Apply latent dropout during training to force using both input and latent
        if training and torch.rand(1).item() < 0.2:  # 20% chance to dropout latent
            latent_emb = latent_emb * 0.1  # Severely attenuate latent signal
        
        # Ensure latent_emb has the correct batch size
        if latent_emb.size(0) != batch_size:
            latent_emb = latent_emb.expand(batch_size, -1)
        
        # Enhanced latent integration with gating mechanism
        # This forces the model to learn when to use latent vs input information
        latent_expanded = latent_emb.unsqueeze(1).expand(-1, memory_input.size(1), -1)
        
        # Learnable gating between input and latent information
        if not hasattr(self, 'input_latent_gate'):
            self.input_latent_gate = nn.Sequential(
                nn.Linear(memory_input.size(-1) + latent_emb.size(-1), memory_input.size(-1)),
                nn.Sigmoid()
            ).to(device)
        
        # Compute gating weights
        combined_for_gate = torch.cat([memory_input, latent_expanded], dim=-1)
        gate_weights = self.input_latent_gate(combined_for_gate)
        
        # Apply gating: balance between input context and latent information
        memory_input_gated = memory_input * gate_weights
        latent_contribution = latent_expanded * (1 - gate_weights)
        
        memory = torch.cat([memory_input_gated, latent_contribution], dim=-1)
        memory = self.memory_projection(memory)
        return memory

    def get_position_embedding(self, row_idx: int, col_idx: int, device: torch.device) -> torch.Tensor:
        row_emb = self.row_embedding(torch.tensor([row_idx], device=device))
        col_emb = self.col_embedding(torch.tensor([col_idx], device=device))
        return row_emb + col_emb

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

    def forward(self, z: torch.Tensor, input_seq: torch.Tensor, target_seq: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If target_seq is provided, use teacher forcing mode (batch parallel processing).
        Otherwise, fall back on autoregressive decoding.
        """
        batch_size = input_seq.size(0)
        device = input_seq.device
        memory = self.prepare_input_memory(z, input_seq, training=self.training)

        # ----- Teacher Forcing Mode -----
        # Clamp target sequence values to ensure valid embedding indices
        tgt_grid = torch.clamp(target_seq[:, :900], 0, 9).long()
        tgt_shape = torch.clamp(target_seq[:, 900:902], 0, 30).long()
        grid_emb = self.output_grid_embedding(tgt_grid)
        shape_emb = self.output_shape_embedding(tgt_shape)
        pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
        pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
        pos_emb_grid = (self.row_embedding(pos_i) + self.col_embedding(pos_j)).view(batch_size, 900, -1)
        grid_emb = grid_emb + pos_emb_grid
        teacher_tgt = torch.cat([grid_emb, shape_emb], dim=1)  # [B, 902, hidden_dim]

        decoder_output = self.transformer_decoder(tgt=teacher_tgt, memory=memory)
        decoder_output = self.layer_norm(decoder_output)
        grid_logits = self.grid_output(decoder_output[:, :900])
        shape_logits = self.shape_output(decoder_output[:, 900:902])
        return shape_logits, grid_logits

# -------------------------------------------------
#  Multi‑Encoder wrapper
# -------------------------------------------------
class MultiEncoderLPN(nn.Module):
    """K‑encoder → Independent decoders + shared decoder for PoE."""

    def __init__(
        self,
        num_encoders: int = None,
        *,
        latent_dim: int = None,
        encoder_hidden_dim: int = None,
        decoder_hidden_dim: int = None,
        encoder_layers: int = None,
        decoder_layers: int = None,
        encoder_heads: int = None,
        decoder_heads: int = None,
        dropout: float = None,
        encoder_max_length: int = None,
        decoder_max_length: int = None,
    ) -> None:
        super().__init__()
        
        # Use current settings if parameters not provided
        if num_encoders is None:
            num_encoders = get_num_encoders()
        if latent_dim is None:
            latent_dim = get_latent_dim()
        if encoder_hidden_dim is None:
            encoder_hidden_dim = get_encoder_hidden_dim()
        if decoder_hidden_dim is None:
            decoder_hidden_dim = get_decoder_hidden_dim()
        if encoder_layers is None:
            encoder_layers = get_encoder_layers()
        if decoder_layers is None:
            decoder_layers = get_decoder_layers()
        if encoder_heads is None:
            encoder_heads = get_encoder_heads()
        if decoder_heads is None:
            decoder_heads = get_decoder_heads()
        if dropout is None:
            dropout = get_dropout()
        if encoder_max_length is None:
            encoder_max_length = get_encoder_max_length()
        if decoder_max_length is None:
            decoder_max_length = get_decoder_max_length()
        
        self.latent_dim = latent_dim
        self.num_encoders = num_encoders
        
        # ---- Create separate encoder instances ----
        self.encoders = nn.ModuleList([
            TransformerEncoder(1, encoder_hidden_dim, encoder_layers, encoder_heads, dropout, encoder_max_length)
            for _ in range(num_encoders)
        ])
        
        # ---- Create independent decoders for each encoder ----
        self.independent_decoders = nn.ModuleList([
            TransformerDecoder(1, decoder_hidden_dim, decoder_layers, decoder_heads, dropout)
            for _ in range(num_encoders)
        ])
        
        # ---- Shared decoder for PoE ----
        self.shared_decoder = TransformerDecoder(1, decoder_hidden_dim, decoder_layers, decoder_heads, dropout)
        
        # ---- Legacy compatibility ----
        self.decoder = self.shared_decoder  # For backward compatibility

    # -------------------------------------------------
    #  Re‑parameterisation
    # -------------------------------------------------
    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Reparameterization for both VAE and VQ-VAE modes.
        
        For VQ-VAE: mu contains quantized latents, logvar contains VQ loss
        For VAE: standard reparameterization trick
        """
        # Check if any encoder is using VQ-VAE
        if hasattr(self.encoders[0], 'use_vq_vae') and self.encoders[0].use_vq_vae:
            # VQ-VAE mode: mu already contains quantized latents, no sampling needed
            return mu
        else:
            # Standard VAE reparameterization
            if sample:
                eps = torch.randn_like(mu)
                return mu + eps * torch.exp(0.5 * logvar)
            return mu

    def get_vq_metrics(self) -> Optional[Dict[str, Any]]:
        """Get VQ-VAE metrics from all encoders if enabled."""
        if hasattr(self.encoders[0], 'use_vq_vae') and self.encoders[0].use_vq_vae:
            metrics = {}
            for i, encoder in enumerate(self.encoders):
                encoder_metrics = encoder.get_vq_metrics()
                if encoder_metrics:
                    for key, value in encoder_metrics.items():
                        metrics[f'encoder_{i}_{key}'] = value
            return metrics
        return None

    # -------------------------------------------------
    #  Forward methods for training and inference
    # -------------------------------------------------
    def forward_single_encoder_with_independent_decoder(self, encoder_idx: int, input_seq: torch.Tensor, target_seq: torch.Tensor, 
                                                       sample_latent: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a single encoder with its independent decoder (Phase A)."""
        assert 0 <= encoder_idx < self.num_encoders, f"encoder_idx {encoder_idx} out of range [0, {self.num_encoders})"
        
        mu, logvar = self.encoders[encoder_idx](input_seq, target_seq)
        z = self._reparam(mu, logvar, sample_latent)
        shape_logits, grid_logits = self.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
        return (shape_logits, grid_logits), mu, logvar

    def forward_single_encoder(self, encoder_idx: int, input_seq: torch.Tensor, target_seq: torch.Tensor, 
                              training: bool = True, sample_latent: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a single encoder with shared decoder (legacy compatibility)."""
        assert 0 <= encoder_idx < self.num_encoders, f"encoder_idx {encoder_idx} out of range [0, {self.num_encoders})"
        
        mu, logvar = self.encoders[encoder_idx](input_seq, target_seq)
        z = self._reparam(mu, logvar, sample_latent)
        shape_logits, grid_logits = self.shared_decoder(z, input_seq, target_seq=target_seq)
        return (shape_logits, grid_logits), mu, logvar

    def forward_poe_with_shared_decoder(self, input_views: List[Tuple[torch.Tensor, torch.Tensor]], 
                                       sample_latent: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass using PoE of all encoders with shared decoder (Phase B)."""
        K = len(self.encoders)
        assert K == len(input_views), "#encoders ≠ #views"
        
        # Collect latent distributions from all encoders
        mu_list, logvar_list = [], []
        for (enc, (x, y)) in zip(self.encoders, input_views):
            μ, logσ2 = enc(x, y)
            mu_list.append(μ)
            logvar_list.append(logσ2)
        
        mu_stack = torch.stack(mu_list)        # (K,B,D)
        logvar_stack = torch.stack(logvar_list)
        
        # PoE fusion
        mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
        z = self._reparam(mu_star, logvar_star, sample_latent)
        
        # Use shared decoder
        x0, y0 = input_views[0]  # decoder conditions on one input grid
        shape_logits, grid_logits = self.shared_decoder(z, x0, target_seq=y0)
        return (shape_logits, grid_logits), mu_star, logvar_star

    def forward(
        self,
        input_views: List[Tuple[torch.Tensor, torch.Tensor]],  # [(x,y), ...] len = K
        *,
        training: bool = True,
        sample_latent: bool = True,
        use_poe: bool = False,  # New parameter to control PoE usage
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Legacy forward method - delegates to PoE with shared decoder."""
        return self.forward_poe_with_shared_decoder(input_views, sample_latent)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Public hook – always sample (variational) for now."""
        return self._reparam(mu, logvar, sample=True)

class LatentProgramNetwork(nn.Module):
    """Unified LPN that supports both single and multi-encoder configurations."""
    
    def __init__(self, input_dim: int = 1, latent_dim: int = None, 
                 encoder_hidden_dim: int = None, decoder_hidden_dim: int = None,
                 encoder_layers: int = None, decoder_layers: int = None,
                 encoder_heads: int = None, decoder_heads: int = None,
                 dropout: float = None, encoder_max_length: int = None, 
                 decoder_max_length: int = None, num_encoders: int = None):
        super().__init__()
        
        # Use current settings if parameters not provided
        if latent_dim is None:
            latent_dim = get_latent_dim()
        if encoder_hidden_dim is None:
            encoder_hidden_dim = get_encoder_hidden_dim()
        if decoder_hidden_dim is None:
            decoder_hidden_dim = get_decoder_hidden_dim()
        if encoder_layers is None:
            encoder_layers = get_encoder_layers()
        if decoder_layers is None:
            decoder_layers = get_decoder_layers()
        if encoder_heads is None:
            encoder_heads = get_encoder_heads()
        if decoder_heads is None:
            decoder_heads = get_decoder_heads()
        if dropout is None:
            dropout = get_dropout()
        if encoder_max_length is None:
            encoder_max_length = get_encoder_max_length()
        if decoder_max_length is None:
            decoder_max_length = get_decoder_max_length()
        if num_encoders is None:
            num_encoders = get_num_encoders()
        
        self.latent_dim = latent_dim
        self.num_encoders = num_encoders
        
        # Always initialize the multi-encoder wrapper
        self.multi_encoder = MultiEncoderLPN(
            num_encoders=num_encoders,
            latent_dim=latent_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_heads=encoder_heads,
            decoder_heads=decoder_heads,
            dropout=dropout,
            encoder_max_length=encoder_max_length,
            decoder_max_length=decoder_max_length
        )

        # Expose single‑encoder style attributes for backwards compatibility
        self.encoder = self.multi_encoder.encoders[0]
        self.decoder = self.multi_encoder.decoder

        # Treat all configurations as multi‑encoder; track if originally single
        self.is_multi_encoder = True
        self.is_actually_single_encoder = (num_encoders == 1)
    
    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Reparameterization for both VAE and VQ-VAE modes.
        
        For VQ-VAE: mu contains quantized latents, logvar contains VQ loss
        For VAE: standard reparameterization trick
        """
        # Check if using VQ-VAE
        if hasattr(self.multi_encoder.encoders[0], 'use_vq_vae') and self.multi_encoder.encoders[0].use_vq_vae:
            # VQ-VAE mode: mu already contains quantized latents, no sampling needed
            return mu
        else:
            # Standard VAE reparameterization
            if sample:
                eps = torch.randn_like(mu)
                return mu + eps * torch.exp(0.5 * logvar)
            return mu

    def get_vq_metrics(self) -> Optional[Dict[str, Any]]:
        """Get VQ-VAE metrics if enabled."""
        return self.multi_encoder.get_vq_metrics()

    def is_using_vq_vae(self) -> bool:
        """Check if the model is using VQ-VAE."""
        return hasattr(self.multi_encoder.encoders[0], 'use_vq_vae') and self.multi_encoder.encoders[0].use_vq_vae

    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor, encoder_idx: int = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Unified forward for both single and multi-encoder setups."""
        if encoder_idx is not None:
            # Individual encoder training/inference
            return self.multi_encoder.forward_single_encoder(
                encoder_idx, input_seq, target_seq, training=True, sample_latent=True
            )

        # PoE inference path – works even when num_encoders == 1
        input_views = [(input_seq, target_seq) for _ in range(self.num_encoders)]
        return self.multi_encoder(
            input_views, training=False, sample_latent=False, use_poe=True
        )

    # -----------------------------------------------------------------
    # Compatibility helper: deterministic reparameterization (mean only)
    # -----------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Public hook – always sample (variational) for now."""
        return self._reparam(mu, logvar, sample=True)

# -------------------------------------------------
#  Cross-pair reconstruction loss for generalization
# -------------------------------------------------

def compute_bonnet_cross_pair_loss(
    model, encoder_idx: int, input_seq: torch.Tensor, target_seq: torch.Tensor, 
    use_independent_decoder: bool = True, latent_opt_steps: int = 0, latent_opt_lr: float = 0.1
) -> torch.Tensor:
    """
    Compute Bonnet's cross-pair reconstruction loss exactly as in the LPN paper.
    
    For each target pair (x_i, y_i):
    1. z_i' = mean{z_j : j≠i} (mean of OTHER latents)
    2. Optionally: K gradient ascent steps on z_i'
    3. L_rec^i = -log p_θ(y_i | x_i, z_i')
    
    Total: L_rec = sum_i L_rec^i
    
    Args:
        model: The multi-encoder model
        encoder_idx: Which encoder to use
        input_seq: Input sequences [B, seq_len]
        target_seq: Target sequences [B, seq_len]
        use_independent_decoder: Whether to use independent or shared decoder
        latent_opt_steps: Number of gradient ascent steps (K in paper)
        latent_opt_lr: Learning rate for gradient ascent
    
    Returns:
        torch.Tensor: Cross-pair reconstruction loss
    """
    batch_size = input_seq.size(0)
    device = input_seq.device
    
    # Need at least 2 samples for cross-pairs
    if batch_size < 2:
        return torch.tensor(0.0, device=device)
    
    # Encode all samples to get latent codes z_i
    mu_all, logvar_all = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
    z_all = model.multi_encoder._reparam(mu_all, logvar_all, sample=True)  # [B, latent_dim]
    
    total_cross_pair_loss = 0.0
    
    for i in range(batch_size):
        # For target (x_i, y_i), compute z_i' = mean{z_j : j≠i}
        other_indices = [j for j in range(batch_size) if j != i]
        if not other_indices:
            continue
            
        z_other = z_all[other_indices]  # [B-1, latent_dim]
        z_i_prime = z_other.mean(dim=0, keepdim=True)  # [1, latent_dim]
        
        # Optional: Gradient ascent optimization of z_i'
        if latent_opt_steps > 0:
            z_i_prime = z_i_prime.clone().detach().requires_grad_(True)
            
            for step in range(latent_opt_steps):
                # Compute log likelihood for OTHER targets using current z_i'
                step_loss = 0.0
                for j in other_indices:
                    x_j = input_seq[j:j+1]
                    y_j = target_seq[j:j+1]
                    
                    if use_independent_decoder:
                        shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](z_i_prime, x_j, target_seq=y_j)
                    else:
                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_i_prime, x_j, target_seq=y_j)
                    
                    # Compute negative log likelihood (we want to maximize likelihood)
                    shape_targets = y_j[:, 900:902].long()
                    shape_nll = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='sum')
                    
                    r, c = int(y_j[0, 900].item()), int(y_j[0, 901].item())
                    n_pix = r * c
                    if n_pix > 0:
                        grid_nll = F.cross_entropy(grid_logits[0, :n_pix], y_j[0, :n_pix].long(), reduction='sum')
                    else:
                        grid_nll = torch.tensor(0.0, device=device)
                    
                    step_loss += shape_nll + grid_nll
                
                # Gradient ascent step
                if step_loss.requires_grad:
                    grad = torch.autograd.grad(step_loss, z_i_prime, retain_graph=(step < latent_opt_steps - 1))[0]
                    z_i_prime = z_i_prime - latent_opt_lr * grad  # Negative gradient for ascent
        
        # Now compute reconstruction loss for target i using optimized z_i'
        x_i = input_seq[i:i+1]
        y_i = target_seq[i:i+1]
        
        if use_independent_decoder:
            shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](z_i_prime, x_i, target_seq=y_i)
        else:
            shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_i_prime, x_i, target_seq=y_i)
        
        # Compute -log p_θ(y_i | x_i, z_i') as in Bonnet's formulation
        shape_targets = y_i[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='mean')
        
        r, c = int(y_i[0, 900].item()), int(y_i[0, 901].item())
        n_pix = r * c
        if n_pix > 0:
            grid_loss = F.cross_entropy(grid_logits[0, :n_pix], y_i[0, :n_pix].long(), reduction='mean')
        else:
            grid_loss = torch.tensor(0.0, device=device)
        
        sample_loss = shape_loss + grid_loss
        total_cross_pair_loss += sample_loss
    
    # Average over batch size to get proper scaling
    return total_cross_pair_loss / batch_size


def get_beta_warmup(current_epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """
    Compute beta warmup schedule: β(t) = β_max * min(t/T_warm, 1)
    
    Args:
        current_epoch: Current training epoch (1-indexed)
        warmup_epochs: Number of epochs for warmup
        beta_max: Maximum beta value
    
    Returns:
        float: Current beta value
    """
    if warmup_epochs <= 0:
        return beta_max
    
    warmup_factor = min(current_epoch / warmup_epochs, 1.0)
    return beta_max * warmup_factor


def get_cyclical_beta(current_epoch: int, cycle_length: int, beta_max: float) -> float:
    """
    Cyclical β-annealing: ramp β 0 → β_max over K epochs, then reset (repeat).
    
    Args:
        current_epoch: Current training epoch (1-indexed)
        cycle_length: Number of epochs per cycle (K)
        beta_max: Maximum beta value
    
    Returns:
        float: Current beta value
    """
    if cycle_length <= 0:
        return beta_max
    
    cycle_position = (current_epoch - 1) % cycle_length
    ramp_factor = cycle_position / cycle_length
    return beta_max * ramp_factor


def get_dynamic_anti_lambda(current_beta: float, beta_max: float, 
                           high_phase_multiplier: float = 3.0, 
                           low_phase_base: float = 0.01) -> float:
    """
    Dynamic λ scheduling: During high-β phases set λ = 2–5 × β_max; 
    during low-β phases drop it to 0.01.
    
    Args:
        current_beta: Current beta value
        beta_max: Maximum beta value
        high_phase_multiplier: Multiplier for high-beta phases (2-5)
        low_phase_base: Base value for low-beta phases
    
    Returns:
        float: Current anti-batch lambda value
    """
    # Consider "high-β phase" when β > 0.5 * β_max
    high_phase_threshold = 0.5 * beta_max
    
    if current_beta > high_phase_threshold:
        # High-β phase: λ = multiplier × β_max
        return high_phase_multiplier * beta_max
    else:
        # Low-β phase: use base value
        return low_phase_base


def apply_free_bits_per_dimension(mu: torch.Tensor, logvar: torch.Tensor, 
                                  delta_per_dim: float = 0.07) -> tuple:
    """
    FIXED Free-bits mechanism: Apply per-dimension clamping to force every latent dimension active.
    
    KL per dimension: kl_dim = 0.5*(μ² + σ² - 1 - log σ²)
    Clamp each dimension: kl_dim = torch.clamp(kl_dim, min=δ)
    
    Args:
        mu: Mean tensor [batch_size, latent_dim]
        logvar: Log variance tensor [batch_size, latent_dim] 
        delta_per_dim: Minimum rate per dimension (δ ≈ 0.07)
    
    Returns:
        tuple: (raw_kl_loss, clamped_kl_loss, kl_per_dim_mean)
    """
    # Calculate per-dimension KL: 0.5*(μ² + σ² - 1 - log σ²)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # [batch_size, latent_dim]
    
    # Raw KL loss (before clamping) for monitoring
    raw_kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))  # Sum over dims, mean over batch
    
    # Apply per-dimension free-bits clamping
    kl_per_dim_clamped = torch.clamp(kl_per_dim, min=delta_per_dim)  # [batch_size, latent_dim]
    
    # Clamped KL loss 
    clamped_kl_loss = torch.mean(torch.sum(kl_per_dim_clamped, dim=1))  # Sum over dims, mean over batch
    
    # Mean KL per dimension (for monitoring)
    kl_per_dim_mean = torch.mean(kl_per_dim)  # Average over batch and dimensions
    
    return raw_kl_loss, clamped_kl_loss, kl_per_dim_mean


def compute_contrastive_kl_margin(in_slice_kl: torch.Tensor, anti_kl: torch.Tensor, 
                                 tau: float = 1.5) -> torch.Tensor:
    """
    Contrastive KL "margin" mechanism:
    gap = kl_in.detach() - kl_anti        # >0 if encoder uses z for its own task
    anti_loss = F.relu(tau - gap)         # tau ≈ 1–2 nats
    
    Args:
        in_slice_kl: KL divergence for in-slice samples
        anti_kl: KL divergence for anti-batch samples
        tau: Target margin in nats
    
    Returns:
        torch.Tensor: Contrastive margin loss
    """
    # gap should be positive when encoder specializes on its own data
    gap = in_slice_kl.detach() - anti_kl
    # Penalize when gap < tau (encoder not specialized enough)
    margin_loss = F.relu(tau - gap)
    return margin_loss


# -------------------------------------------------
#  Convenience loss wrapper (matches existing API)
# -------------------------------------------------

def multinomial_loss(
    logits: Tuple[torch.Tensor, torch.Tensor],  # (shape_logits, grid_logits)
    target_seq: torch.Tensor,
    *,
    beta: float = BETA,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Bonnet-style multinomial loss computation."""
    shape_logits, grid_logits = logits
    shape_targets = target_seq[:, 900:902].long()
    shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

    batch_size = target_seq.size(0)
    grid_loss_sum = 0.0
    active_samples = 0
    for i in range(batch_size):
        r, c = map(int, target_seq[i, 900:902])
        n_pix = r * c
        if n_pix > 0:
            grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long())
            active_samples += 1
    grid_loss = grid_loss_sum / active_samples if active_samples > 0 else torch.tensor(0.0, device=target_seq.device)
    recon = shape_loss + grid_loss
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
    return recon + beta * kl


def compute_loss(model: nn.Module, input_seq: torch.Tensor, target_seq: torch.Tensor, 
                beta: float = BETA, return_components: bool = False, encoder_idx: int = None, 
                use_independent_decoder: bool = False, 
                # New parameters for specialist training
                current_epoch: int = None, anti_mask: torch.Tensor = None, 
                anti_batch_lambda: float = None, cross_pair_enabled: bool = False,
                cross_pair_num_pairs: int = None,
                # New parameters for enhanced training mechanisms
                use_cyclical_beta: bool = False, beta_cycle_length: int = 4,
                use_free_bits: bool = True, free_bits_delta: float = 0.07,
                use_dynamic_lambda: bool = True, lambda_high_multiplier: float = 3.0,
                use_contrastive_margin: bool = True, margin_tau: float = 1.5,
                debug_kl_metrics: bool = False) -> torch.Tensor:
    """
    Compute loss following Bonnet's LPN approach with enhanced mechanisms for specialist training.
    
    SUPPORTS VQ-VAE: When VQ-VAE is enabled, replaces KL divergence with VQ loss (commitment + codebook losses).
    
    ENHANCED MECHANISMS:
    1. Cyclical β-annealing: Ramp β 0 → β_max over K epochs, then reset
    2. Free-bits: Ensure minimum KL (δ ≈ 0.05-0.1 × latent_dim) to keep latents active
    3. Dynamic λ scheduling: Scale anti-batch penalty with β (λ = 2-5 × β_max during high-β)
    4. Contrastive KL margin: Enforce gap between in-slice and anti-batch KL losses
    5. VQ-VAE support: Use discrete latents to prevent posterior collapse
    
    L_total = L_rec + β * L_KL/VQ + λ * L_anti + margin_loss
    
    Args:
        model: The model (single or multi-encoder)
        input_seq: Input sequence
        target_seq: Target sequence  
        beta: Base KL divergence weight (also used for VQ loss weight)
        return_components: Whether to return loss components
        encoder_idx: Which encoder to use (None for PoE inference)
        use_independent_decoder: Whether to use independent decoder (Phase A) or shared decoder (Phase B)
        current_epoch: Current training epoch for beta warmup/cycling (1-indexed)
        anti_mask: Boolean mask indicating anti-samples [B] (True = anti-sample, False = in-slice)
        anti_batch_lambda: Base lambda weight for anti-batch KL regularization
        cross_pair_enabled: Whether to use cross-pair reconstruction loss
        cross_pair_num_pairs: Number of cross-pairs to sample (None = all pairs)
        use_cyclical_beta: Whether to use cyclical β-annealing
        beta_cycle_length: Number of epochs per β cycle
        use_free_bits: Whether to apply free-bits mechanism
        free_bits_delta: Minimum rate per dimension for free-bits
        use_dynamic_lambda: Whether to use dynamic λ scheduling
        lambda_high_multiplier: Multiplier for high-β phases
        use_contrastive_margin: Whether to use contrastive KL margin
        margin_tau: Target margin in nats for contrastive loss
        debug_kl_metrics: Whether to return debug KL metrics
    """
    device = input_seq.device
    
    # Get latent dimensionality for free-bits calculation
    latent_dim = get_latent_dim()
    
    # Check if model is using VQ-VAE
    is_vq_vae = hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae()
    
    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
        # ----------------------------------------------------------
        # Multi-encoder specialist training path
        # ----------------------------------------------------------
        if encoder_idx is None:
            # PoE training with shared decoder (Phase B)
            input_views = [(input_seq, target_seq) for _ in range(model.num_encoders)]
            reconstruction, mu_star, logvar_star = model.multi_encoder.forward_poe_with_shared_decoder(
                input_views, sample_latent=True
            )
            shape_logits, grid_logits = reconstruction
            
            # Bonnet's reconstruction loss computation
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
            
            batch_size = target_seq.size(0)
            grid_loss_sum = 0.0
            active_pixels_total = 0
            
            for i in range(batch_size):
                tgt_rows = int(target_seq[i, 900].item())
                tgt_cols = int(target_seq[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                
                if active_pixels > 0:
                    # Cross-entropy loss over active region only
                    loss_i = F.cross_entropy(grid_logits[i, :active_pixels], target_seq[i, :active_pixels].long())
                    grid_loss_sum += loss_i
                    active_pixels_total += 1  # Count samples, not pixels for averaging
                    
            grid_loss = grid_loss_sum / active_pixels_total if active_pixels_total > 0 else torch.tensor(0.0, device=device)
            reconstruction_loss = shape_loss + grid_loss
            
            # Latent regularization loss (KL or VQ)
            if is_vq_vae:
                # VQ-VAE: logvar_star contains VQ loss
                vq_loss = torch.mean(logvar_star)  # VQ loss is stored in logvar position
                latent_loss = vq_loss
                raw_kl_loss = torch.tensor(0.0, device=device)
                kl_per_dim_mean = torch.tensor(0.0, device=device)
            else:
                # Standard VAE: KL divergence with free-bits
                if use_free_bits:
                    raw_kl_loss, latent_loss, kl_per_dim_mean = apply_free_bits_per_dimension(mu_star, logvar_star, free_bits_delta)
                else:
                    latent_loss = 0.5 * torch.mean(torch.sum(mu_star.pow(2) + logvar_star.exp() - 1 - logvar_star, dim=1))
                    raw_kl_loss = latent_loss
                    kl_per_dim_mean = latent_loss / latent_dim
            
            # Compute effective beta (cyclical annealing if enabled)
            if use_cyclical_beta and current_epoch is not None:
                effective_beta = get_cyclical_beta(current_epoch, beta_cycle_length, beta)
            else:
                effective_beta = beta
            
            total_loss = reconstruction_loss + effective_beta * latent_loss
            
            if return_components:
                components = {
                    'total_loss': total_loss,
                    'reconstruction_loss': reconstruction_loss,
                    'shape_loss': shape_loss,
                    'grid_loss': grid_loss,
                    'effective_beta': effective_beta,
                }
                
                if is_vq_vae:
                    components.update({
                        'vq_loss': latent_loss,
                        'kl_loss': torch.tensor(0.0, device=device),  # No KL for VQ-VAE
                        'raw_kl_loss': torch.tensor(0.0, device=device),
                        'kl_per_dim': torch.tensor(0.0, device=device),
                    })
                else:
                    components.update({
                        'kl_loss': latent_loss,
                        'vq_loss': torch.tensor(0.0, device=device),  # No VQ for standard VAE
                        'raw_kl_loss': raw_kl_loss,
                        'kl_per_dim': kl_per_dim_mean,
                    })
                
                return components
            return total_loss
            
        else:
            # Individual encoder training (Phase A) - Enhanced with new mechanisms
            batch_size = input_seq.size(0)
            
            # Get latent distributions for KL computation
            mu, logvar = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
            
            # Separate in-slice and anti-samples if anti-batch is used
            if anti_mask is not None:
                in_slice_mask = ~anti_mask  # True for in-slice samples
                has_in_slice = in_slice_mask.any()
                has_anti = anti_mask.any()
            else:
                # No anti-batch, all samples are in-slice
                in_slice_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                has_in_slice = True
                has_anti = False
            
            # Initialize loss components
            reconstruction_loss = torch.tensor(0.0, device=device)
            latent_loss = torch.tensor(0.0, device=device)
            anti_latent_loss = torch.tensor(0.0, device=device)
            contrastive_margin_loss = torch.tensor(0.0, device=device)
            
            # 1. CROSS-PAIR RECONSTRUCTION LOSS (Bonnet's approach on in-slice samples)
            if cross_pair_enabled and has_in_slice:
                in_slice_indices = torch.where(in_slice_mask)[0]
                if len(in_slice_indices) >= 2:  # Need at least 2 in-slice samples
                    in_slice_input = input_seq[in_slice_indices]
                    in_slice_target = target_seq[in_slice_indices]
                    
                    # Get latent optimization settings
                    latent_settings = get_current_settings().get('latent_optimization', {})
                    training_settings = latent_settings.get('training', {})
                    opt_steps = training_settings.get('num_steps', 0) if training_settings.get('enabled', False) else 0
                    opt_lr = training_settings.get('learning_rate', 0.1)
                    
                    reconstruction_loss = compute_bonnet_cross_pair_loss(
                        model, encoder_idx, in_slice_input, in_slice_target,
                        use_independent_decoder, opt_steps, opt_lr
                    )
            
            # 2. LATENT REGULARIZATION LOSS (KL or VQ)
            raw_kl_loss = torch.tensor(0.0, device=device)
            kl_per_dim_mean = torch.tensor(0.0, device=device)
            vq_loss = torch.tensor(0.0, device=device)
            
            if has_in_slice:
                in_slice_mu = mu[in_slice_mask]
                in_slice_logvar = logvar[in_slice_mask]
                
                if in_slice_mu.size(0) > 0:
                    if is_vq_vae:
                        # VQ-VAE: logvar contains VQ loss
                        vq_loss = torch.mean(in_slice_logvar)
                        latent_loss = vq_loss
                    else:
                        # Standard VAE: KL divergence with free-bits
                        if use_free_bits:
                            raw_kl_loss, latent_loss, kl_per_dim_mean = apply_free_bits_per_dimension(in_slice_mu, in_slice_logvar, free_bits_delta)
                        else:
                            latent_loss = 0.5 * torch.mean(torch.sum(
                                in_slice_mu.pow(2) + in_slice_logvar.exp() - 1 - in_slice_logvar, dim=1
                            ))
                            raw_kl_loss = latent_loss
                            kl_per_dim_mean = latent_loss / latent_dim
            
            # 3. ANTI-BATCH REGULARIZATION (enhanced with dynamic scheduling)
            raw_anti_kl_loss = torch.tensor(0.0, device=device)
            anti_kl_per_dim_mean = torch.tensor(0.0, device=device)
            anti_vq_loss = torch.tensor(0.0, device=device)
            
            if has_anti:
                anti_mu = mu[anti_mask]
                anti_logvar = logvar[anti_mask]
                
                if anti_mu.size(0) > 0:
                    if is_vq_vae:
                        # VQ-VAE: anti-batch VQ loss
                        anti_vq_loss = torch.mean(anti_logvar)
                        anti_latent_loss = anti_vq_loss
                    else:
                        # Standard VAE: anti-batch KL (no free-bits clamping)
                        anti_latent_loss = 0.5 * torch.mean(torch.sum(
                            anti_mu.pow(2) + anti_logvar.exp() - 1 - anti_logvar, dim=1
                        ))
                        raw_anti_kl_loss = anti_latent_loss
                        anti_kl_per_dim_mean = anti_latent_loss / latent_dim
            
            # 4. BETA AND LAMBDA SCHEDULING
            # Compute effective beta (cyclical annealing if enabled)
            if use_cyclical_beta and current_epoch is not None:
                effective_beta = get_cyclical_beta(current_epoch, beta_cycle_length, beta)
            elif current_epoch is not None:
                # Apply beta warmup if specified
                specialist_settings = get_current_settings().get('specialist_training', {})
                phase_a_settings = specialist_settings.get('phase_a', {})
                warmup_epochs = phase_a_settings.get('beta_warmup_epochs', 5)
                effective_beta = get_beta_warmup(current_epoch, warmup_epochs, beta)
            else:
                effective_beta = beta
            
            # Compute effective lambda (dynamic scheduling if enabled)
            if use_dynamic_lambda and has_anti:
                effective_lambda = get_dynamic_anti_lambda(
                    effective_beta, beta, lambda_high_multiplier,
                    anti_batch_lambda if anti_batch_lambda is not None else 0.01
                )
            else:
                effective_lambda = anti_batch_lambda if anti_batch_lambda is not None else 0.01
            
            # 5. CONTRASTIVE MARGIN (enforce specialization gap)
            if use_contrastive_margin and has_in_slice and has_anti and not is_vq_vae:
                # Only apply contrastive margin for VAE (not meaningful for VQ-VAE)
                contrastive_margin_loss = compute_contrastive_kl_margin(
                    latent_loss, anti_latent_loss, margin_tau
                )
            
            # 6. TOTAL LOSS (enhanced with new mechanisms)
            total_loss = reconstruction_loss + effective_beta * latent_loss
            
            if has_anti:
                total_loss += effective_lambda * anti_latent_loss
            
            if use_contrastive_margin and not is_vq_vae:
                total_loss += contrastive_margin_loss
            
            # Debug metrics
            kl_gap_raw = (raw_kl_loss - raw_anti_kl_loss) if debug_kl_metrics and has_anti and not is_vq_vae else None
            kl_gap_clamped = (latent_loss - anti_latent_loss) if debug_kl_metrics and has_anti and not is_vq_vae else None
            
            if return_components:
                components = {
                    'total_loss': total_loss,
                    'reconstruction_loss': reconstruction_loss,
                    'cross_pair_loss': reconstruction_loss,
                    'in_slice_kl_loss': latent_loss,
                    'effective_beta': effective_beta,
                    'effective_lambda': effective_lambda,
                }
                
                if is_vq_vae:
                    components.update({
                        'vq_loss': vq_loss,
                        'anti_vq_loss': anti_vq_loss,
                        'kl_loss': torch.tensor(0.0, device=device),
                        'anti_kl_loss': torch.tensor(0.0, device=device),
                        'raw_kl_loss': torch.tensor(0.0, device=device),
                        'kl_per_dim': torch.tensor(0.0, device=device),
                    })
                else:
                    components.update({
                        'kl_loss': latent_loss,
                        'anti_kl_loss': anti_latent_loss,
                        'vq_loss': torch.tensor(0.0, device=device),
                        'anti_vq_loss': torch.tensor(0.0, device=device),
                        'raw_kl_loss': raw_kl_loss,
                        'kl_per_dim': kl_per_dim_mean,
                    })
                
                # Add enhanced mechanism losses
                if use_contrastive_margin and not is_vq_vae:
                    components['contrastive_margin_loss'] = contrastive_margin_loss
                
                # Add debug metrics
                if debug_kl_metrics and not is_vq_vae:
                    if kl_gap_raw is not None:
                        components['kl_gap'] = kl_gap_raw
                    if 'anti_kl_per_dim_mean' in locals():
                        components['anti_kl_per_dim'] = anti_kl_per_dim_mean
                
                return components
            
            return total_loss
            
    else:
        # Single encoder model - use Bonnet's standard approach with enhancements
        reconstruction, mu, log_var = model(input_seq, target_seq)
        shape_logits, grid_logits = reconstruction

        # Bonnet's reconstruction loss computation
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
        batch_size = target_seq.size(0)
        grid_loss_sum = 0.0
        active_samples_count = 0
        
        for i in range(batch_size):
            # Retrieve the target dimensions (active region) from the last two tokens.
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols

            if active_pixels > 0:
                # Compute cross-entropy only over the active region.
                loss_i = F.cross_entropy(grid_logits[i, :active_pixels], target_seq[i, :active_pixels].long())
                grid_loss_sum += loss_i
                active_samples_count += 1  # Count samples for averaging

        # Average grid loss over samples (not pixels)
        grid_loss = grid_loss_sum / active_samples_count if active_samples_count > 0 else torch.tensor(0.0, device=input_seq.device)
        reconstruction_loss = shape_loss + grid_loss

        # Latent regularization loss (KL or VQ)
        if is_vq_vae:
            # VQ-VAE: log_var contains VQ loss
            vq_loss = torch.mean(log_var)
            latent_loss = vq_loss
            raw_kl_loss = torch.tensor(0.0, device=input_seq.device)
            kl_per_dim_mean = torch.tensor(0.0, device=input_seq.device)
        else:
            # Standard VAE: KL divergence with free-bits
            if use_free_bits:
                raw_kl_loss, latent_loss, kl_per_dim_mean = apply_free_bits_per_dimension(mu, log_var, free_bits_delta)
            else:
                latent_loss = 0.5 * torch.mean(torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1))
                raw_kl_loss = latent_loss
                kl_per_dim_mean = latent_loss / latent_dim

        # Compute effective beta (cyclical annealing if enabled)
        if use_cyclical_beta and current_epoch is not None:
            effective_beta = get_cyclical_beta(current_epoch, beta_cycle_length, beta)
        else:
            effective_beta = beta

        total_loss = reconstruction_loss + effective_beta * latent_loss

        if return_components:
            components = {
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'shape_loss': shape_loss,
                'grid_loss': grid_loss,
                'effective_beta': effective_beta,
            }
            
            if is_vq_vae:
                components.update({
                    'vq_loss': vq_loss,
                    'kl_loss': torch.tensor(0.0, device=input_seq.device),
                    'raw_kl_loss': torch.tensor(0.0, device=input_seq.device),
                    'kl_per_dim': torch.tensor(0.0, device=input_seq.device),
                })
            else:
                components.update({
                    'kl_loss': latent_loss,
                    'vq_loss': torch.tensor(0.0, device=input_seq.device),
                    'raw_kl_loss': raw_kl_loss,
                    'kl_per_dim': kl_per_dim_mean,
                })
            
            return components
        
        return total_loss


