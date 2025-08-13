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
        current_settings = get_current_settings()
        if 'model_architecture' not in current_settings:
            raise KeyError("'model_architecture' key not found in settings. Please check your settings file and how it is loaded.")
        current_model_arch = current_settings['model_architecture']
        if 'vq_vae' not in current_model_arch:
            print("Warning: 'vq_vae' key not found in model_architecture. VQ-VAE will be disabled.")
        self.vq_vae = create_vq_vae_from_settings(current_model_arch)
        self.use_vq_vae = self.vq_vae is not None
        
        if self.use_vq_vae:
            # For VQ-VAE, we only need one projection to continuous space before quantization
            self.fc_latent = nn.Linear(hidden_dim, get_latent_dim())
            print(f"[ OK ] VQ-VAE enabled with {self.vq_vae.vq_layer.num_embeddings} embeddings")
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

    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # Return quantized latent, VQ loss, and encoding indices
            return z_quantized, vq_loss.unsqueeze(0).expand(batch_size, -1) if vq_loss.dim() == 0 else vq_loss, encoding_indices
        else:
            # Standard VAE path
            mu = self.fc_mu(cls_output)
            log_var = self.fc_log_var(cls_output)
            return mu, log_var, None

    def get_vq_metrics(self) -> Optional[Dict[str, Any]]:
        """Get VQ-VAE metrics if enabled."""
        if self.use_vq_vae:
            return self.vq_vae.get_metrics()
        return None

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = None,
                 num_layers: int = None, num_heads: int = None,
                 dropout: float = None):
        super().__init__()

        # Load hyperparameters from settings if not provided
        hidden_dim = hidden_dim or get_decoder_hidden_dim()
        num_layers = num_layers or get_decoder_layers()
        num_heads  = num_heads  or get_decoder_heads()
        dropout    = dropout    or get_dropout()

        model_arc_set = get_current_settings()['model_architecture']
        self.input_dropout_prob = model_arc_set.get('decoder_input_dropout', 0.0)
        self.hidden_dim = hidden_dim

        # Embeddings for outputs
        self.output_shape_embedding = nn.Embedding(31, hidden_dim)
        self.output_grid_embedding  = nn.Embedding(10, hidden_dim)

        # Input grid embeddings (reuse shape embedding)
        self.input_grid_embedding = nn.Embedding(10, hidden_dim)

        # FIXED: Embedding tables for distinct absolute positions
        # Row embedding: indices 0-31 (0-29 for grid, 30-31 for shape tokens)
        self.row_embedding = nn.Embedding(32, hidden_dim)  # Was 31, now 32
        # Col embedding: indices 0-30 (0-29 for grid, 30 for shape tokens)
        self.col_embedding = nn.Embedding(31, hidden_dim)  # Was 31, now 31

        # Latent projection
        self.latent_projection = nn.Linear(LATENT_DIM, hidden_dim)

        # CONVERTED: Transformer encoder layers (prefix token architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if get_current_settings()['model_architecture'].get('use_gradient_checkpointing', False):
            for layer in self.transformer.layers:
                layer.use_checkpoint = True

        # Output projections
        self.shape_output = nn.Linear(hidden_dim, 31)
        self.grid_output  = nn.Linear(hidden_dim, 10)
        self.layer_norm   = nn.LayerNorm(hidden_dim)

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create LPN-style causal mask: allow latent+shape+input (<903) to see everything,
        only mask the output grid portion (>=903).
        
        FIXED: Works on all PyTorch versions
        """
        # Build upper triangular mask (causal)
        tri = torch.triu(torch.ones(seq_len, seq_len, device=device), 1)
        tri[:903, :] = 0  # 0..902 (latent + rows/cols + input) unmasked
        mask = tri.bool()  # bool for ≥2.0, will cast to float later
        
        if torch.__version__ < '2':
            # float mask for 1.x
            mask = mask.float().masked_fill(mask, float('-inf'))
        
        return mask

    def get_position_embedding(self, row_idx: int, col_idx: int, device: torch.device) -> torch.Tensor:
        """Get positional embedding for grid positions (0-29, 0-29)"""
        return (self.row_embedding(torch.tensor([row_idx], device=device)) +
                self.col_embedding(torch.tensor([col_idx], device=device)))

    def get_shape_position_embeddings(self, device: torch.device) -> torch.Tensor:
        """
        Get distinct absolute positions for the two shape tokens (rows, cols).
        Mirror Bonnet's decoder exactly.
        """
        # Shape tokens get distinct absolute positions:
        # rows: (30, 30), cols: (31, 30)
        shape_pos = torch.stack([
            self.row_embedding.weight[30] + self.col_embedding.weight[30],  # rows
            self.row_embedding.weight[31] + self.col_embedding.weight[30],  # cols
        ])  # shape (2, H)
        return shape_pos

    def forward(self, z: torch.Tensor, input_seq: torch.Tensor,
                target_seq: torch.Tensor = None, 
                zero_latent: bool = False,
                perturb_latent: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIXED: Proper latent integration in decoder
        """
        batch_size, device = input_seq.size(0), input_seq.device
        
        # Handle sequence length mismatch
        if input_seq.size(1) == 900:
            if target_seq is not None and target_seq.size(1) >= 902:
                input_shape = target_seq[:, 900:902]
                input_grid = input_seq[:, :900]
                input_seq = torch.cat([input_grid, input_shape], dim=1)
            else:
                input_shape = torch.tensor([[30, 30]], device=device, dtype=input_seq.dtype).expand(batch_size, 2)
                input_grid = input_seq[:, :900]
                input_seq = torch.cat([input_grid, input_shape], dim=1)
        
        # Extract components
        input_grid = torch.clamp(input_seq[:, :900], 0, 9).long()
        input_shape = torch.clamp(input_seq[:, 900:902], 0, 30).long()
        
        # Create embeddings
        input_grid_emb = self.input_grid_embedding(input_grid)
        input_shape_emb = self.output_shape_embedding(input_shape)
        
        # FIXED: Proper latent integration
        latent_emb = self.latent_projection(z)  # [B, H]
        
        if zero_latent:
            latent_emb = torch.zeros_like(latent_emb)
        
        if perturb_latent > 0:
            noise = torch.randn_like(latent_emb) * perturb_latent
            latent_emb = latent_emb + noise
        
        # FIXED: Broadcast latent to all positions for proper integration
        # This ensures the latent affects ALL output positions
        latent_broadcast = latent_emb.unsqueeze(1).expand(-1, 902, -1)  # [B, 902, H]
        
        # FIXED: Add latent to input embeddings (not just as a separate token)
        input_grid_emb = input_grid_emb + latent_broadcast[:, :900, :]
        input_shape_emb = input_shape_emb + latent_broadcast[:, 900:902, :]
        
        # Add positional embeddings
        pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
        pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
        input_grid_emb = input_grid_emb + (self.row_embedding(pos_i) + self.col_embedding(pos_j)).view(batch_size, 900, -1)
        
        # FIXED: Simple autoregressive generation (no teacher forcing during inference)
        if target_seq is None:
            # Inference: generate autoregressively
            shape_logits = self.shape_output(input_shape_emb)
            grid_logits = self.grid_output(input_grid_emb)
            return shape_logits, grid_logits
        else:
            # Training: use teacher forcing but with proper latent integration
            target_grid = torch.clamp(target_seq[:, :900], 0, 9).long()
            target_shape = torch.clamp(target_seq[:, 900:902], 0, 30).long()
            
            target_grid_emb = self.output_grid_embedding(target_grid)
            target_shape_emb = self.output_shape_embedding(target_shape)
            
            # FIXED: Add latent to target embeddings too
            target_grid_emb = target_grid_emb + latent_broadcast[:, :900, :]
            target_shape_emb = target_shape_emb + latent_broadcast[:, 900:902, :]
            
            # Add positional embeddings
            target_grid_emb = target_grid_emb + (self.row_embedding(pos_i) + self.col_embedding(pos_j)).view(batch_size, 900, -1)
            
            # FIXED: Use transformer to process the integrated sequence
            seq_emb = torch.cat([input_grid_emb, input_shape_emb, target_grid_emb, target_shape_emb], dim=1)
            
            # Apply transformer
            output = self.layer_norm(self.transformer(seq_emb))
            
            # FIXED: Extract from correct positions
            shape_logits = self.shape_output(output[:, 900:902])  # Shape positions
            grid_logits = self.grid_output(output[:, :900])        # Grid positions
            
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
        
        mu, logvar, encoding_indices = self.encoders[encoder_idx](input_seq, target_seq)
        z = self._reparam(mu, logvar, sample_latent)
        shape_logits, grid_logits = self.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
        return (shape_logits, grid_logits), mu, logvar, encoding_indices

    def forward_single_encoder(self, encoder_idx: int, input_seq: torch.Tensor, target_seq: torch.Tensor, 
                              training: bool = True, sample_latent: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for a single encoder with shared decoder (legacy compatibility)."""
        assert 0 <= encoder_idx < self.num_encoders, f"encoder_idx {encoder_idx} out of range [0, {self.num_encoders})"
        
        mu, logvar, encoding_indices = self.encoders[encoder_idx](input_seq, target_seq)
        z = self._reparam(mu, logvar, sample_latent)
        shape_logits, grid_logits = self.shared_decoder(z, input_seq, target_seq=target_seq)
        return (shape_logits, grid_logits), mu, logvar, encoding_indices

    def forward_poe_with_shared_decoder(self, input_views: List[Tuple[torch.Tensor, torch.Tensor]], 
                                       sample_latent: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass using PoE of all encoders with shared decoder (Phase B)."""
        K = len(self.encoders)
        assert K == len(input_views), "#encoders ≠ #views"
        
        # Collect latent distributions from all encoders
        mu_list, logvar_list = [], []
        for (enc, (x, y)) in zip(self.encoders, input_views):
            mu, logvar, _ = enc(x, y)
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        mu_stack = torch.stack(mu_list)        # (K,B,D)
        logvar_stack = torch.stack(logvar_list)
        
        # PoE fusion
        mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
        z = self._reparam(mu_star, logvar_star, sample_latent)
        
        # Use shared decoder
        x0, y0 = input_views[0]  # decoder conditions on one input grid
        shape_logits, grid_logits = self.shared_decoder(z, x0, target_seq=y0)
        return (shape_logits, grid_logits), mu_star, logvar_star, None

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
    Compute Bonnet's cross-pair loss: -log p_θ(y_i | x_i, z̄_{-i})
    where z̄_{-i} is the average latent from other samples in the batch.
    
    This implements leave-one-out training as described in Bonnet's LPN paper.
    """
    device = input_seq.device
    batch_size = input_seq.size(0)
    total_cross_pair_loss = torch.tensor(0.0, device=device)
    
    # For each sample i, compute loss using average latent from other samples
    for i in range(batch_size):
        # Compute average latent from other samples (leave-one-out)
        other_latents = []
        for j in range(batch_size):
            if j != i:  # Exclude sample i
                x_j = input_seq[j:j+1]
                y_j = target_seq[j:j+1]
                
                # Get latent for sample j
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    mu_j, logvar_j, _ = model.multi_encoder.encoders[encoder_idx](x_j, y_j)
                else:
                    mu_j, logvar_j, _ = model.encoder(x_j, y_j)
                z_j = model.reparameterize(mu_j, logvar_j)
                other_latents.append(z_j)
        
        if not other_latents:
            continue  # Skip if no other samples
        
        # Average latent from other samples
        z_avg = torch.stack(other_latents).mean(dim=0, keepdim=True)
        
        # Initialize z_i' with the average latent
        z_i_prime = z_avg.clone().detach().requires_grad_(True)
        
        # Gradient optimization to maximize p(y_i | x_i, z_i')
        for step in range(latent_opt_steps):
            # CORRECT: Optimize for TARGET sample i
            x_i = input_seq[i:i+1]
            y_i = target_seq[i:i+1]
            
            if use_independent_decoder:
                shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](z_i_prime, x_i, target_seq=y_i)
            else:
                shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_i_prime, x_i, target_seq=y_i)
            
            # Compute loss for TARGET sample i
            shape_targets = y_i[:, 900:902].long()
            shape_nll = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='sum')
            
            r, c = int(y_i[0, 900].item()), int(y_i[0, 901].item())
            n_pix = r * c
            if n_pix > 0:
                grid_nll = F.cross_entropy(grid_logits[0, :n_pix], y_i[0, :n_pix].long(), reduction='sum')
            else:
                grid_nll = torch.tensor(0.0, device=device)
            
            step_loss = shape_nll + grid_nll
            
            # Gradient ascent to maximize p(y_i | x_i, z_i_prime)
            if step_loss.requires_grad:
                grad = torch.autograd.grad(step_loss, z_i_prime, retain_graph=(step < latent_opt_steps - 1))[0]
            z_i_prime = z_i_prime - latent_opt_lr * grad
        
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
    Compute beta warmup schedule: beta(t) = beta_max * min(t/T_warm, 1)
    
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
    Cyclical beta-annealing: ramp beta 0 -> beta_max over K epochs, then reset (repeat).
    
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
    Dynamic lambda scheduling: During high-beta phases set lambda = 2-5 x beta_max; 
    during low-beta phases drop it to 0.01.
    
    Args:
        current_beta: Current beta value
        beta_max: Maximum beta value
        high_phase_multiplier: Multiplier for high-beta phases (2-5)
        low_phase_base: Base value for low-beta phases
    
    Returns:
        float: Current anti-batch lambda value
    """
    # Consider "high-beta phase" when beta > 0.5 * beta_max
    high_phase_threshold = 0.5 * beta_max
    
    if current_beta > high_phase_threshold:
        # High-beta phase: lambda = multiplier x beta_max
        return high_phase_multiplier * beta_max
    else:
        # Low-beta phase: use base value
        return low_phase_base


def apply_free_bits_per_dimension(mu: torch.Tensor, logvar: torch.Tensor, 
                                  delta_per_dim: float = 0.07) -> tuple:
    """
    FIXED Free-bits mechanism: Apply per-dimension clamping to force every latent dimension active.
    
    KL per dimension: kl_dim = 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)
    Clamp each dimension: kl_dim = torch.clamp(kl_dim, min=delta)
    
    Args:
        mu: Mean tensor [batch_size, latent_dim]
        logvar: Log variance tensor [batch_size, latent_dim] 
        delta_per_dim: Minimum rate per dimension (delta ~= 0.07)
    
    Returns:
        tuple: (raw_kl_loss, clamped_kl_loss, kl_per_dim_mean)
    """
    # Calculate per-dimension KL: 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)
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
    anti_loss = F.relu(tau - gap)         # tau ~= 1-2 nats
    
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
                # New parameters for enhanced training mechanisms
                use_cyclical_beta: bool = False, beta_cycle_length: int = 4,
                use_free_bits: bool = True, free_bits_delta: float = 0.07,
                use_dynamic_lambda: bool = True, lambda_high_multiplier: float = 3.0,
                use_contrastive_margin: bool = True, margin_tau: float = 1.5,
                debug_kl_metrics: bool = False,
                # NEW: Repulsion loss parameters
                use_repulsion_loss: bool = False, repulsion_lambda: float = 0.1,
                repulsion_margin: float = 0.5, repulsion_logvar_min: float = -8.0) -> torch.Tensor:
    """
    Compute loss following Bonnet's LPN approach with enhanced mechanisms for specialist training.
    
    SUPPORTS VQ-VAE: When VQ-VAE is enabled, replaces KL divergence with VQ loss (commitment + codebook losses).
    
    ENHANCED MECHANISMS:
    1. Cyclical beta-annealing: Ramp beta 0 -> beta_max over K epochs, then reset
    2. Free-bits: Ensure minimum KL (δ ≈ 0.05-0.1 × latent_dim) to keep latents active
    3. Dynamic λ scheduling: Scale anti-batch penalty with β (λ = 2-5 × β_max during high-β)
    4. Contrastive KL margin: Enforce gap between in-slice and anti-batch KL losses
    5. VQ-VAE support: Use discrete latents to prevent posterior collapse
    6. REPULSION LOSS: Pairwise hinge repulsion between encoders (for joint training)
    
    L_total = L_rec + β * L_KL/VQ + λ * L_anti + margin_loss + λ_rep * L_repulsion
    
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
        use_cyclical_beta: Whether to use cyclical β-annealing
        beta_cycle_length: Number of epochs per β cycle
        use_free_bits: Whether to apply free-bits mechanism
        free_bits_delta: Minimum rate per dimension for free-bits
        use_dynamic_lambda: Whether to use dynamic λ scheduling
        lambda_high_multiplier: Multiplier for high-β phases
        use_contrastive_margin: Whether to use contrastive KL margin
        margin_tau: Target margin in nats for contrastive loss
        debug_kl_metrics: Whether to return debug KL metrics
        use_repulsion_loss: Whether to add repulsion loss (for joint training)
        repulsion_lambda: Weight for repulsion loss
        repulsion_margin: Hinge margin for repulsion loss
        repulsion_logvar_min: Minimum log variance for repulsion loss
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
            # PoE training with shared decoder (Phase B) - INCLUDES REPULSION LOSS
            K = model.num_encoders
            
            # Collect latent distributions from all encoders for repulsion loss
            mus, logvars = [], []
            for enc in model.multi_encoder.encoders:
                mu, logvar, _ = enc(input_seq, target_seq)
                logvar = logvar.clamp(min=repulsion_logvar_min)
                mus.append(mu)
                logvars.append(logvar)
            
            # PoE fusion
            mu_stack = torch.stack(mus)        # (K,B,D)
            logvar_stack = torch.stack(logvars)
            mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
            logvar_star = logvar_star.clamp(min=repulsion_logvar_min)
            
            # Reparameterization
            eps = torch.randn_like(mu_star)
            z = mu_star + eps * torch.exp(0.5*logvar_star)
            latent_magnitude = z.norm(dim=1).mean().item()  # Mean L2 norm per sample
            
            # Decode with shared decoder
            shape_logits, grid_logits = model.multi_encoder.shared_decoder(z, input_seq, target_seq=target_seq)
            
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
            
            # REPULSION LOSS (pairwise hinge repulsion between encoders)
            repulsion_loss = torch.tensor(0.0, device=device)
            if use_repulsion_loss and K > 1:
                pairs = []
                D = mus[0].size(1)
                for j in range(K):
                    for k in range(j+1,K):
                        muj, logvarj = mus[j], logvars[j]
                        muk, logvark = mus[k], logvars[k]
                        vj, vk = logvarj.exp(), logvark.exp()
                        kl_jk = 0.5*((vj/vk).sum(1) + ((muk-muj).pow(2)/vk).sum(1) - D + (logvark-logvarj).sum(1))
                        pairs.append(kl_jk)
                if pairs:
                    hinge = torch.stack([F.relu(repulsion_margin - p) for p in pairs])
                    repulsion_loss = hinge.mean()
            
            # Compute effective beta (cyclical annealing if enabled)
            if use_cyclical_beta and current_epoch is not None:
                effective_beta = get_cyclical_beta(current_epoch, beta_cycle_length, beta)
            else:
                effective_beta = beta
            
            # TOTAL LOSS with repulsion
            total_loss = reconstruction_loss + effective_beta * latent_loss + repulsion_lambda * repulsion_loss
            
            if return_components:
                components = {
                    'total_loss': total_loss,
                    'reconstruction_loss': reconstruction_loss,
                    'shape_loss': shape_loss,
                    'grid_loss': grid_loss,
                    'effective_beta': effective_beta,
                    'repulsion_loss': repulsion_loss,
                    'repulsion_lambda': repulsion_lambda,
                    'latent_magnitude': latent_magnitude
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
            mu, logvar, encoding_indices = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
            
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
                        'anti_vq_loss': anti_vq_loss,
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
        batch_size = input_seq.size(0)
        use_cross_pair = cross_pair_enabled and batch_size >= 2
        if use_cross_pair:
            # Use cross-pair loss for single-encoder
            # Use encoder_idx=0, and model.encoder/model.decoder
            # Get latent optimization settings
            latent_settings = get_current_settings().get('latent_optimization', {})
            training_settings = latent_settings.get('training', {})
            opt_steps = training_settings.get('num_steps', 0) if training_settings.get('enabled', False) else 0
            opt_lr = training_settings.get('learning_rate', 0.1)
            reconstruction_loss = compute_bonnet_cross_pair_loss(
                model, 0, input_seq, target_seq, True, opt_steps, opt_lr
            )
        else:
            reconstruction, mu, log_var, _ = model(input_seq, target_seq)
            shape_logits, grid_logits = reconstruction

            # Bonnet's reconstruction loss computation
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
            
            grid_loss_sum = 0.0
            active_samples_count = 0
            for i in range(batch_size):
                tgt_rows = int(target_seq[i, 900].item())
                tgt_cols = int(target_seq[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0:
                    loss_i = F.cross_entropy(grid_logits[i, :active_pixels], target_seq[i, :active_pixels].long())
                    grid_loss_sum += loss_i
                    active_samples_count += 1
            grid_loss = grid_loss_sum / active_samples_count if active_samples_count > 0 else torch.tensor(0.0, device=input_seq.device)
            reconstruction_loss = shape_loss + grid_loss

        # Latent regularization loss (KL or VQ)
        if is_vq_vae:
            # VQ-VAE: log_var contains VQ loss
            vq_loss = torch.mean(log_var) if not use_cross_pair else torch.tensor(0.0, device=input_seq.device)
            latent_loss = vq_loss
            raw_kl_loss = torch.tensor(0.0, device=input_seq.device)
            kl_per_dim_mean = torch.tensor(0.0, device=input_seq.device)
        else:
            # Standard VAE: KL divergence with free-bits
            if use_free_bits:
                raw_kl_loss, latent_loss, kl_per_dim_mean = apply_free_bits_per_dimension(mu, log_var, free_bits_delta) if not use_cross_pair else (torch.tensor(0.0, device=input_seq.device), torch.tensor(0.0, device=input_seq.device), torch.tensor(0.0, device=input_seq.device))
            else:
                latent_loss = 0.5 * torch.mean(torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1)) if not use_cross_pair else torch.tensor(0.0, device=input_seq.device)
                raw_kl_loss = latent_loss
                kl_per_dim_mean = latent_loss / latent_dim if not use_cross_pair else torch.tensor(0.0, device=input_seq.device)

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
                'shape_loss': shape_loss if not use_cross_pair else torch.tensor(0.0, device=input_seq.device),
                'grid_loss': grid_loss if not use_cross_pair else torch.tensor(0.0, device=input_seq.device),
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