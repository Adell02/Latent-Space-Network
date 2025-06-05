import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (set_seed)
from utils.settings_manager import settings


#########################################
# TUNABLE SETTINGS
#########################################

# Get settings from settings manager
data_settings = settings.get_data_settings()
model_architecture = settings.get_model_architecture()
training_settings = settings.get_training_settings()
latent_optimization = settings.get_latent_optimization()

# Data settings
TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
    raise ValueError("No training keys specified in data_settings.")

TRAINING_SEED = data_settings['training_seed']
N_EXAMPLES_PER_TASK = data_settings['n']

# Model architecture settings
LATENT_DIM = model_architecture['latent_dim']
HIDDEN_DIM = model_architecture['hidden_dim']
NUM_LAYERS = model_architecture['num_layers']
NUM_HEADS = model_architecture['num_heads']
DROPOUT = model_architecture['dropout']
MAX_LENGTH = model_architecture['max_length']
ENCODER_MAX_LENGTH = model_architecture['encoder_max_length']
DECODER_MAX_LENGTH = model_architecture['decoder_max_length']

# Training settings
BATCH_SIZE = training_settings['batch_size']
NUM_EPOCHS = training_settings['num_epochs']
LEARNING_RATE = training_settings['learning_rate']
BETA = training_settings['beta']

# Latent optimization settings
OPTIMIZE_Z = latent_optimization['training']['enabled']
OPTIMIZE_Z_NUM_STEPS = latent_optimization['training']['num_steps']
OPTIMIZE_Z_LR = latent_optimization['training']['learning_rate']
OPTIMIZE_Z_INFERENCE = latent_optimization['inference']['enabled']
OPTIMIZE_Z_INFERENCE_NUM_STEPS = latent_optimization['inference']['num_steps']
OPTIMIZE_Z_INFERENCE_LR = latent_optimization['inference']['learning_rate']

set_seed(TRAINING_SEED)

##############################
# Define Model Components
##############################

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 dropout: float = DROPOUT, max_length: int = ENCODER_MAX_LENGTH):
        super().__init__()
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
        if model_architecture.get('use_gradient_checkpointing', False):
            for mod in self.transformer_encoder.layers:
                mod.use_checkpoint = True  # Enables gradient checkpointing

        # Output projections for latent distribution
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, LATENT_DIM)
        self.fc_log_var = nn.Linear(hidden_dim, LATENT_DIM)

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
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 dropout: float = DROPOUT):
        super().__init__()
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
            dim_feedforward=4*hidden_dim,
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

class LatentProgramNetwork(nn.Module):
    def __init__(self, input_dim: int = 1, latent_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS, dropout: float = DROPOUT,
                 encoder_max_length: int = ENCODER_MAX_LENGTH, decoder_max_length: int = DECODER_MAX_LENGTH):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Initialize components with device awareness
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads,
                                        dropout, max_length=encoder_max_length)
        self.decoder = TransformerDecoder(input_dim, hidden_dim, num_layers, num_heads, dropout)        
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(input_seq, target_seq)
        z = self.reparameterize(mu, log_var)
        shape_logits, grid_logits = self.decoder(z, input_seq, target_seq=target_seq)
        return (shape_logits, grid_logits), mu, log_var


def compute_loss(model: nn.Module, input_seq: torch.Tensor, target_seq: torch.Tensor, beta: float = BETA, return_components: bool = False) -> torch.Tensor:
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


