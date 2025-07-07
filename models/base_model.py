import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple, List, Union
import copy

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (set_seed)
from utils.settings_manager import settings


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
        # Updated indexing: grid tokens first (0-899), then shape tokens (900:902)
        input_color_emb = self.color_embedding(input_seq[:, :900].long())
        input_shape_emb = self.shape_embedding(input_seq[:, 900:902].long())
        target_color_emb = self.color_embedding(target_seq[:, :900].long())
        target_shape_emb = self.shape_embedding(target_seq[:, 900:902].long())

        # Create padding masks using the shape tokens
        input_mask = self.create_padding_mask(input_seq[:, 900:902])
        target_mask = self.create_padding_mask(target_seq[:, 900:902])

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
        mu = self.fc_mu(cls_output)
        log_var = self.fc_log_var(cls_output)
        return mu, log_var

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
        if model_architecture.get('use_gradient_checkpointing', False):
            for mod in self.transformer_decoder.layers:
                mod.use_checkpoint = True

        # Output projections: one for shape tokens and one for grid tokens
        self.shape_output = nn.Linear(hidden_dim, 31)  # For shape values (indices 0-30)
        self.grid_output = nn.Linear(hidden_dim, 10)   # For grid values (indices 0-9)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def prepare_input_memory(self, z: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        """Prepare memory from input sequence and latent vector."""
        batch_size = input_seq.size(0)
        device = input_seq.device

        # Updated indexing: grid tokens first, then shape tokens
        input_grid = input_seq[:, :900]
        input_shapes = input_seq[:, 900:902]

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
        
        # Ensure latent_emb has the correct batch size
        if latent_emb.size(0) != batch_size:
            latent_emb = latent_emb.expand(batch_size, -1)
        
        memory = torch.cat([memory_input, latent_emb.unsqueeze(1).expand(-1, memory_input.size(1), -1)], dim=-1)
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
        memory = self.prepare_input_memory(z, input_seq)

        # ----- Teacher Forcing Mode -----
        tgt_grid = target_seq[:, :900].long()
        tgt_shape = target_seq[:, 900:902].long()
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
        """Deterministic when sample=False, otherwise draws one sample."""
        if sample:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

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
        """Wrapper to use the same sampling logic as the multi-encoder."""
        if sample:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

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
    shape_logits, grid_logits = logits
    shape_targets = target_seq[:, 900:902].long()
    shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

    batch_size = target_seq.size(0)
    grid_loss_sum, active = 0.0, 0
    for i in range(batch_size):
        r, c = map(int, target_seq[i, 900:902])
        n_pix = r * c
        if n_pix:
            grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
            active += n_pix
    grid_loss = grid_loss_sum / active if active else torch.tensor(0.0, device=target_seq.device)
    recon = shape_loss + grid_loss
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
    return recon + beta * kl

def compute_loss(model: nn.Module, input_seq: torch.Tensor, target_seq: torch.Tensor, beta: float = BETA, return_components: bool = False, encoder_idx: int = None, use_independent_decoder: bool = False) -> torch.Tensor:
    """
    Compute loss for both single and multi-encoder models.
    
    Args:
        model: The model (single or multi-encoder)
        input_seq: Input sequence
        target_seq: Target sequence  
        beta: KL divergence weight
        return_components: Whether to return loss components
        encoder_idx: Which encoder to use (None for PoE inference)
        use_independent_decoder: Whether to use independent decoder (Phase A) or shared decoder (Phase B)
    """
    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
        # ----------------------------------------------------------
        # Multi-encoder path
        # ----------------------------------------------------------
        if encoder_idx is None:
            # PoE training with shared decoder (Phase B)
            input_views = [(input_seq, target_seq) for _ in range(model.num_encoders)]
            reconstruction, mu_star, logvar_star = model.multi_encoder.forward_poe_with_shared_decoder(
                input_views, sample_latent=True
            )
            shape_logits, grid_logits = reconstruction
            
            # Use PoE latent for KL loss
            mu, logvar = mu_star, logvar_star
            
            # Compute reconstruction loss for PoE
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
            
            batch_size = target_seq.size(0)
            grid_loss_sum, active = 0.0, 0
            for i in range(batch_size):
                r, c = map(int, target_seq[i, 900:902])
                n_pix = r * c
                if n_pix > 0:
                    grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
                    active += n_pix
            grid_loss = grid_loss_sum / active if active > 0 else torch.tensor(0.0, device=target_seq.device)
            rec_loss = shape_loss + grid_loss
            
            # KL divergence loss
            kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
            
            total_loss = rec_loss + beta * kl_loss
            
            if return_components:
                return total_loss, shape_loss, grid_loss, kl_loss
            return total_loss
            
        else:
            # Individual encoder training (Phase A)
            if use_independent_decoder:
                # Phase A: Use encoder with its independent decoder
                reconstruction, mu, logvar = model.multi_encoder.forward_single_encoder_with_independent_decoder(
                    encoder_idx, input_seq, target_seq, sample_latent=True
                )
            else:
                # Legacy: Use encoder with shared decoder
                reconstruction, mu, logvar = model.multi_encoder.forward_single_encoder(
                    encoder_idx, input_seq, target_seq, sample_latent=True
                )
            
            shape_logits, grid_logits = reconstruction
            
            # Compute reconstruction loss
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
            
            batch_size = target_seq.size(0)
            grid_loss_sum, active = 0.0, 0
            for i in range(batch_size):
                r, c = map(int, target_seq[i, 900:902])
                n_pix = r * c
                if n_pix > 0:
                    grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
                    active += n_pix
            grid_loss = grid_loss_sum / active if active > 0 else torch.tensor(0.0, device=target_seq.device)
            rec_loss = shape_loss + grid_loss
            
            # KL divergence loss
            kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
            
            total_loss = rec_loss + beta * kl_loss
            
            if return_components:
                return total_loss, shape_loss, grid_loss, kl_loss
            return total_loss
            
    else:
        # Single encoder model (original implementation)
        reconstruction, mu, log_var = model(input_seq, target_seq)
        shape_logits, grid_logits = reconstruction

        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
        batch_size = target_seq.size(0)
        grid_loss_sum = 0.0
        active_elements_count = 0 # Keep track of total active pixels across batch for correct averaging
        for i in range(batch_size):
            # Retrieve the target dimensions (active region) from the last two tokens.
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols

            if active_pixels > 0:
                # Compute cross-entropy only over the active region. Sum losses for each sample.
                loss_i = F.cross_entropy(grid_logits[i, :active_pixels], target_seq[i, :active_pixels].long(), reduction='sum')
                grid_loss_sum += loss_i
                active_elements_count += active_pixels

        # Average grid loss over all active pixels in the batch
        grid_loss = grid_loss_sum / active_elements_count if active_elements_count > 0 else torch.tensor(0.0, device=input_seq.device)

        reconstruction_loss = shape_loss + grid_loss

        # KL divergence loss, normalized by batch size for consistency
        kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var) / mu.size(0) 

        total_loss = reconstruction_loss + beta * kl_loss
        
        if return_components:
            return total_loss, shape_loss, grid_loss, kl_loss
        return total_loss


