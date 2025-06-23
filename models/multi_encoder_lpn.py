# """Multi‑Encoder Latent‑Program Network
# ------------------------------------------------
# *   K weight‑shared Transformer encoders take **disjoint or overlapping views** of an ARC specimen.
# *   Each encoder outputs a diagonal‑Gaussian belief (μᵢ , log σ²ᵢ) **in one common latent basis**.
# *   A closed‑form **Product‑of‑Experts** fuses these K Gaussians to (μ★ , σ★²).
# *   A single Transformer decoder (inherited from the vanilla LPN) consumes z = μ★ (+ εσ★) at
#     training time and z = μ★ at inference; inner‑loop latent search operates on that same z.

# The module is drop‑in compatible with the existing training / inference scripts.
# """
# from __future__ import annotations
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys
# from typing import List, Tuple, Optional

# # add the parent directory to the path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.model_utils import (set_seed)
# from Multiencoder_LPN.settings_manager import settings


# #########################################
# # TUNABLE SETTINGS
# #########################################

# # Get settings from multiencoder settings manager
# data_settings = settings.get_data_settings()
# model_architecture = settings.get_model_architecture()
# training_settings = settings.get_training_settings()
# latent_optimization = settings.get_latent_optimization()

# # Data settings
# TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
# if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
#     raise ValueError("No training keys specified in data_settings.")

# TRAINING_SEED = data_settings['training_seed']
# N_EXAMPLES_PER_TASK = data_settings['n']

# # Model architecture settings
# NUM_ENCODERS = model_architecture['num_encoders']
# LATENT_DIM = model_architecture['latent_dim']
# ENCODER_HIDDEN_DIM = model_architecture.get('encoder_hidden_dim', model_architecture.get('hidden_dim', 96))
# DECODER_HIDDEN_DIM = model_architecture.get('decoder_hidden_dim', model_architecture.get('hidden_dim', 96))
# ENCODER_LAYERS = model_architecture.get('encoder_layers', model_architecture.get('num_layers', 2))
# DECODER_LAYERS = model_architecture.get('decoder_layers', model_architecture.get('num_layers', 2))
# ENCODER_HEADS = model_architecture.get('encoder_heads', model_architecture.get('num_heads', 6))
# DECODER_HEADS = model_architecture.get('decoder_heads', model_architecture.get('num_heads', 6))
# DROPOUT = model_architecture['dropout']
# MAX_LENGTH = model_architecture['max_length']
# ENCODER_MAX_LENGTH = model_architecture['encoder_max_length']
# DECODER_MAX_LENGTH = model_architecture['decoder_max_length']

# # Training settings
# BATCH_SIZE = training_settings['batch_size']
# NUM_EPOCHS = training_settings['num_epochs']
# LEARNING_RATE = training_settings['learning_rate']
# BETA = training_settings['beta']

# # Latent optimization settings
# OPTIMIZE_Z = latent_optimization['training']['enabled']
# OPTIMIZE_Z_NUM_STEPS = latent_optimization['training']['num_steps']
# OPTIMIZE_Z_LR = latent_optimization['training']['learning_rate']
# OPTIMIZE_Z_INFERENCE = latent_optimization['inference']['enabled']
# OPTIMIZE_Z_INFERENCE_NUM_STEPS = latent_optimization['inference']['num_steps']
# OPTIMIZE_Z_INFERENCE_LR = latent_optimization['inference']['learning_rate']

# set_seed(TRAINING_SEED)

# # -------------------------------------------------
# #  Low‑level helper: diagonal‑Gaussian PoE
# # -------------------------------------------------

# def gaussian_poe(mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Multiply K diagonal Gaussians.
#     Args
#     -----
#     mu      : (K, B, D)
#     logvar  : (K, B, D)
#     Returns
#     -------
#     fused_mu, fused_logvar : (B, D)
#     """
#     precision   = torch.exp(-logvar)            # Σ⁻¹
#     fused_var   = 1. / precision.sum(0)         # (B,D)
#     fused_mu    = fused_var * (precision * mu).sum(0)
#     fused_logvar = fused_var.log()
#     return fused_mu, fused_logvar

# ##############################
# # Define Model Components
# ##############################

# class TransformerEncoder(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = ENCODER_HIDDEN_DIM, num_layers: int = ENCODER_LAYERS, 
#                  num_heads: int = ENCODER_HEADS, dropout: float = DROPOUT, max_length: int = ENCODER_MAX_LENGTH):
#         super().__init__()
#         # Embedding tables
#         self.color_embedding = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)
#         self.shape_embedding = nn.Embedding(num_embeddings=31, embedding_dim=hidden_dim)
#         self.cls_embedding = nn.Parameter(torch.randn(1, hidden_dim))
#         # Positional embeddings (factorized into row, column, and channel components)
#         self.row_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
#         self.col_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
#         self.channel_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim)

#         # Transformer encoder (pre-layer normalization)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=4*hidden_dim,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Enable gradient checkpointing if specified in settings
#         if model_architecture.get('use_gradient_checkpointing', False):
#             for mod in self.transformer_encoder.layers:
#                 mod.use_checkpoint = True  # Enables gradient checkpointing

#         # Output projections for latent distribution
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.fc_mu = nn.Linear(hidden_dim, LATENT_DIM)
#         self.fc_log_var = nn.Linear(hidden_dim, LATENT_DIM)

#     def create_padding_mask(self, shape_values: torch.Tensor) -> torch.Tensor:
#         """Create padding mask based on shape values"""
#         batch_size = shape_values.size(0)
#         # Ensure rows and cols are integers
#         rows = shape_values[:, 0].long().cpu().numpy()
#         cols = shape_values[:, 1].long().cpu().numpy()

#         masks = []
#         for b in range(batch_size):
#             r, c = int(rows[b]), int(cols[b])
#             mask = torch.zeros(30, 30, dtype=torch.bool, device=shape_values.device)
#             mask[:r, :c] = True
#             masks.append(mask.flatten())

#         return torch.stack(masks)

#     def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size = input_seq.size(0)
#         device = input_seq.device
#         # Updated indexing: grid tokens first (0-899), then shape tokens (900:902)
#         input_color_emb = self.color_embedding(input_seq[:, :900].long())
#         input_shape_emb = self.shape_embedding(input_seq[:, 900:902].long())
#         target_color_emb = self.color_embedding(target_seq[:, :900].long())
#         target_shape_emb = self.shape_embedding(target_seq[:, 900:902].long())

#         # Create padding masks using the shape tokens
#         input_mask = self.create_padding_mask(input_seq[:, 900:902])
#         target_mask = self.create_padding_mask(target_seq[:, 900:902])

#         # Create position indices for a 30x30 grid
#         pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
#         pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
#         row_emb = self.row_embedding(pos_i)
#         col_emb = self.col_embedding(pos_j)

#         # Create channel embeddings: 0 for input, 1 for target
#         input_channel_emb = self.channel_embedding(torch.zeros(1, dtype=torch.long, device=device))
#         target_channel_emb = self.channel_embedding(torch.ones(1, dtype=torch.long, device=device))

#         # Combine positional embeddings and reshape to flattened grid
#         input_pos_emb = (row_emb + col_emb + input_channel_emb).view(batch_size, 900, -1)
#         target_pos_emb = (row_emb + col_emb + target_channel_emb).view(batch_size, 900, -1)

#         # Combine embeddings with positional information
#         input_emb = input_color_emb + input_pos_emb
#         target_emb = target_color_emb + target_pos_emb

#         # Append the shape embeddings after the grid tokens and add a CLS token at the end
#         cls_emb = self.cls_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
#         combined_emb = torch.cat([input_emb, input_shape_emb, target_emb, target_shape_emb, cls_emb], dim=1)

#         # Create attention mask (for grid tokens we use input_mask and target_mask, and ones for shape/CLS tokens)
#         combined_mask = torch.cat([
#             input_mask,
#             torch.ones(batch_size, 2, dtype=torch.bool, device=device),
#             target_mask,
#             torch.ones(batch_size, 3, dtype=torch.bool, device=device)
#         ], dim=1)

#         encoder_output = self.transformer_encoder(combined_emb, src_key_padding_mask=~combined_mask)
#         cls_output = self.layer_norm(encoder_output[:, -1])
#         mu = self.fc_mu(cls_output)
#         log_var = self.fc_log_var(cls_output)
#         return mu, log_var

# class TransformerDecoder(nn.Module):
#     def __init__(self, output_dim: int, hidden_dim: int = DECODER_HIDDEN_DIM, num_layers: int = DECODER_LAYERS, 
#                  num_heads: int = DECODER_HEADS, dropout: float = DROPOUT):
#         super().__init__()
#         self.hidden_dim = hidden_dim

#         # Embeddings for teacher forcing
#         self.output_shape_embedding = nn.Embedding(num_embeddings=31, embedding_dim=hidden_dim)
#         self.output_grid_embedding = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)

#         # Positional embeddings for grid positions (we reuse row and col embeddings)
#         self.row_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)
#         self.col_embedding = nn.Embedding(num_embeddings=30, embedding_dim=hidden_dim)

#         # Start token embedding (used for autoregressive mode)
#         self.start_token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

#         # Latent projection to initial decoder memory
#         self.latent_projection = nn.Linear(LATENT_DIM, hidden_dim)

#         # Memory projection (combines input and latent info)
#         self.memory_projection = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
#         # Enable gradient checkpointing if specified in settings
#         if model_architecture.get('use_gradient_checkpointing', False):
#             for mod in self.transformer_decoder.layers:
#                 mod.use_checkpoint = True  # Enables gradient checkpointing

#         # Output projections
#         self.shape_projection = nn.Linear(hidden_dim, 31)
#         self.grid_projection = nn.Linear(hidden_dim, 10)

#     def prepare_input_memory(self, z: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
#         """
#         Prepare the memory for the decoder by encoding the input grid with latent information.
        
#         Args:
#             z: Latent vector [batch_size, latent_dim]
#             input_seq: Input sequence [batch_size, sequence_length]
        
#         Returns:
#             memory: Prepared memory for decoder [batch_size, 900, hidden_dim]
#         """
#         batch_size = input_seq.size(0)
#         device = input_seq.device

#         # Extract input grid (900 tokens) and create embeddings
#         input_color_emb = self.output_grid_embedding(input_seq[:, :900].long())

#         # Create positional embeddings for the input grid
#         pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
#         pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
#         row_emb = self.row_embedding(pos_i)
#         col_emb = self.col_embedding(pos_j)
#         pos_emb = (row_emb + col_emb).view(batch_size, 900, -1)

#         # Combine color and positional embeddings
#         input_emb = input_color_emb + pos_emb

#         # Project latent to match hidden dimension and expand for each position
#         z_proj = self.latent_projection(z).unsqueeze(1).repeat(1, 900, 1)

#         # Combine input and latent information
#         combined = torch.cat([input_emb, z_proj], dim=-1)
#         memory = self.memory_projection(combined)

#         return memory

#     def get_position_embedding(self, row_idx: int, col_idx: int, device: torch.device) -> torch.Tensor:
#         """Get positional embedding for a specific grid position"""
#         row_emb = self.row_embedding(torch.tensor([row_idx], device=device))
#         col_emb = self.col_embedding(torch.tensor([col_idx], device=device))
#         return row_emb + col_emb

#     def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """Create causal mask for autoregressive generation"""
#         return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

#     def forward(self, z: torch.Tensor, input_seq: torch.Tensor, target_seq: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the decoder.
        
#         Args:
#             z: Latent vector [batch_size, latent_dim]
#             input_seq: Input sequence [batch_size, sequence_length]
#             target_seq: Target sequence for teacher forcing [batch_size, sequence_length]
        
#         Returns:
#             shape_logits: Shape predictions [batch_size, 31]
#             grid_logits: Grid predictions [batch_size, 900, 10]
#         """
#         batch_size = z.size(0)
#         device = z.device

#         # Prepare memory from input sequence and latent
#         memory = self.prepare_input_memory(z, input_seq)

#         if target_seq is not None:
#             # Teacher forcing mode: use target sequence
#             target_grid = target_seq[:, :900].long()
#             target_shape = target_seq[:, 900:902].long()

#             # Create target embeddings with positional encoding
#             target_emb = self.output_grid_embedding(target_grid)
#             pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
#             pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
#             row_emb = self.row_embedding(pos_i)
#             col_emb = self.col_embedding(pos_j)
#             pos_emb = (row_emb + col_emb).view(batch_size, 900, -1)
#             target_emb = target_emb + pos_emb

#             # Add shape embeddings
#             target_shape_emb = self.output_shape_embedding(target_shape)
#             full_target_emb = torch.cat([target_emb, target_shape_emb], dim=1)

#             # Create causal mask for the full sequence (900 grid + 2 shape tokens)
#             tgt_mask = self.create_causal_mask(902, device)

#             # Pass through transformer decoder
#             decoder_output = self.transformer_decoder(full_target_emb, memory, tgt_mask=tgt_mask)

#             # Split outputs
#             grid_output = decoder_output[:, :900]  # Grid predictions
#             shape_output = decoder_output[:, 900:902]  # Shape predictions

#             # Apply output projections
#             grid_logits = self.grid_projection(grid_output)
#             shape_logits = self.shape_projection(shape_output.mean(dim=1))  # Average the 2 shape tokens

#         else:
#             # Autoregressive mode: generate sequentially
#             # For now, implement a simplified version that uses the latent directly
#             grid_logits = self.grid_projection(memory)
#             # Generate shape prediction from the mean of the memory
#             shape_logits = self.shape_projection(memory.mean(dim=1))

#         return shape_logits, grid_logits

# # -------------------------------------------------
# #  Multi‑Encoder wrapper
# # -------------------------------------------------
# class MultiEncoderLPN(nn.Module):
#     """K‑encoder → PoE → single decoder."""

#     def __init__(
#         self,
#         num_encoders: int = NUM_ENCODERS,
#         *,
#         latent_dim: int = LATENT_DIM,
#         encoder_hidden_dim: int = ENCODER_HIDDEN_DIM,
#         decoder_hidden_dim: int = DECODER_HIDDEN_DIM,
#         encoder_layers: int = ENCODER_LAYERS,
#         decoder_layers: int = DECODER_LAYERS,
#         encoder_heads: int = ENCODER_HEADS,
#         decoder_heads: int = DECODER_HEADS,
#         dropout: float = DROPOUT,
#         encoder_max_length: int = ENCODER_MAX_LENGTH,
#         decoder_max_length: int = DECODER_MAX_LENGTH,
#     ) -> None:
#         super().__init__()
#         self.latent_dim = latent_dim
#         # ---- shared encoder weights ----
#         prototype = TransformerEncoder(1, encoder_hidden_dim, encoder_layers, encoder_heads, dropout, encoder_max_length)
#         self.encoders = nn.ModuleList([prototype for _ in range(num_encoders)])  # weight sharing via same ref
#         self.decoder  = TransformerDecoder(1, decoder_hidden_dim, decoder_layers, decoder_heads, dropout)

#     # -------------------------------------------------
#     #  Re‑parameterisation
#     # -------------------------------------------------
#     def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
#         if not sample:
#             return mu
#         std = torch.exp(0.5 * logvar)
#         return mu + torch.randn_like(std) * std

#     # -------------------------------------------------
#     #  Forward (training / inference)
#     # -------------------------------------------------
#     def forward(
#         self,
#         input_views: List[Tuple[torch.Tensor, torch.Tensor]],  # [(x_i, y_i), ...] len = K
#         *,
#         training: bool = True,
#         sample_latent: bool = True,
#     ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
#         """Args
#         -----
#         input_views : during **training** K *different* (x,y) pairs from the same ARC task;
#                       during **inference** K *identical* copies of the single (x,y₀) pair.
#         """
#         K = len(self.encoders)
#         assert K == len(input_views), "#encoders ≠ #views"
#         mu_list, logvar_list = [], []
#         for (enc, (x, y)) in zip(self.encoders, input_views):
#             μ, logσ2 = enc(x, y)
#             mu_list.append(μ)
#             logvar_list.append(logσ2)
#         mu_stack     = torch.stack(mu_list)        # (K,B,D)
#         logvar_stack = torch.stack(logvar_list)
#         mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
#         z = self._reparam(mu_star, logvar_star, sample_latent and training)
#         x0, y0 = input_views[0]                   # decoder always conditions on *one* input grid
#         # Always provide target_seq for proper loss computation, even during inference
#         shape_logits, grid_logits = self.decoder(z, x0, target_seq=y0)
#         return (shape_logits, grid_logits), mu_star, logvar_star

# # -------------------------------------------------
# #  Convenience loss wrapper (matches existing API)
# # -------------------------------------------------

# def multinomial_loss(
#     logits: Tuple[torch.Tensor, torch.Tensor],  # (shape_logits, grid_logits)
#     target_seq: torch.Tensor,
#     *,
#     beta: float = BETA,
#     mu: torch.Tensor,
#     logvar: torch.Tensor,
# ) -> torch.Tensor:
#     shape_logits, grid_logits = logits
#     shape_targets = target_seq[:, 900:902].long()
#     shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

#     batch_size = target_seq.size(0)
#     grid_loss_sum, active = 0.0, 0
#     for i in range(batch_size):
#         r, c = map(int, target_seq[i, 900:902])
#         n_pix = r * c
#         if n_pix:
#             grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
#             active += n_pix
#     grid_loss = grid_loss_sum / active if active else torch.tensor(0.0, device=target_seq.device)
#     recon = shape_loss + grid_loss
#     kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
#     return recon + beta * kl

# def compute_loss(model: nn.Module, input_views: List[Tuple[torch.Tensor, torch.Tensor]], beta: float = BETA, return_components: bool = False) -> torch.Tensor:
#     """
#     Compute loss for multi-encoder model.
    
#     Args:
#         model: MultiEncoderLPN model
#         input_views: List of (input_seq, target_seq) tuples for each encoder
#         beta: Beta parameter for KL loss weighting
#         return_components: Whether to return loss components separately
    
#     Returns:
#         loss: Computed loss value
#     """
#     (shape_logits, grid_logits), mu, log_var = model(input_views, training=True, sample_latent=True)
    
#     # Use the first view's target for loss computation (they should be equivalent)
#     target_seq = input_views[0][1]
    
#     loss = multinomial_loss(
#         (shape_logits, grid_logits),
#         target_seq,
#         beta=beta,
#         mu=mu,
#         logvar=log_var
#     )
    
#     if return_components:
#         # Compute individual components for logging
#         shape_targets = target_seq[:, 900:902].long()
#         shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
#         batch_size = target_seq.size(0)
#         grid_loss_sum, active = 0.0, 0
#         for i in range(batch_size):
#             r, c = map(int, target_seq[i, 900:902])
#             n_pix = r * c
#             if n_pix:
#                 grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
#                 active += n_pix
#         grid_loss = grid_loss_sum / active if active else torch.tensor(0.0, device=target_seq.device)
#         kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var) / mu.size(0)
        
#         return loss, shape_loss, grid_loss, kl_loss
    
#     return loss
