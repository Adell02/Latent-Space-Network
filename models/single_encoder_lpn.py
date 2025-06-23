import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.settings_manager import settings

# Get settings from settings manager
model_architecture = settings.get_model_architecture()

# Model architecture settings
LATENT_DIM = model_architecture['latent_dim']
HIDDEN_DIM = model_architecture['hidden_dim']
NUM_LAYERS = model_architecture['num_layers']
NUM_HEADS = model_architecture['num_heads']
DROPOUT = model_architecture['dropout']
ENCODER_MAX_LENGTH = model_architecture['encoder_max_length']
DECODER_MAX_LENGTH = model_architecture['decoder_max_length']

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
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Enable gradient checkpointing if specified in settings
        if model_architecture.get('use_gradient_checkpointing', False):
            for mod in self.transformer_decoder.layers:
                mod.use_checkpoint = True  # Enables gradient checkpointing

        # Output projections
        self.shape_projection = nn.Linear(hidden_dim, 31)
        self.grid_projection = nn.Linear(hidden_dim, 10)

    def prepare_input_memory(self, z: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        """Prepare the memory state for the decoder by combining latent and input information"""
        batch_size = input_seq.size(0)
        device = input_seq.device

        # Process the input sequence (same as encoder)
        input_color_emb = nn.Embedding(num_embeddings=10, embedding_dim=self.hidden_dim).to(device)(input_seq[:, :900].long())
        input_shape_emb = nn.Embedding(num_embeddings=31, embedding_dim=self.hidden_dim).to(device)(input_seq[:, 900:902].long())

        # Create position embeddings for the input
        pos_i = torch.arange(30, device=device).view(1, -1, 1).repeat(batch_size, 1, 30)
        pos_j = torch.arange(30, device=device).view(1, 1, -1).repeat(batch_size, 30, 1)
        row_emb = self.row_embedding(pos_i)
        col_emb = self.col_embedding(pos_j)
        input_pos_emb = (row_emb + col_emb).view(batch_size, 900, -1)

        # Combine input embeddings
        input_emb = input_color_emb + input_pos_emb
        input_combined = torch.cat([input_emb, input_shape_emb], dim=1)

        # Project latent to match hidden dimension and broadcast to sequence length
        latent_projected = self.latent_projection(z).unsqueeze(1).repeat(1, input_combined.size(1), 1)

        # Combine input and latent information
        combined = torch.cat([input_combined, latent_projected], dim=-1)
        memory = self.memory_projection(combined)

        return memory

    def get_position_embedding(self, row_idx: int, col_idx: int, device: torch.device) -> torch.Tensor:
        """Get position embedding for a specific grid position"""
        row_emb = self.row_embedding(torch.tensor([row_idx], device=device))
        col_emb = self.col_embedding(torch.tensor([col_idx], device=device))
        return row_emb + col_emb

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, z: torch.Tensor, input_seq: torch.Tensor, target_seq: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decoder
        z: latent vector (batch_size, latent_dim)
        input_seq: input sequence (batch_size, seq_len)
        target_seq: target sequence for teacher forcing (batch_size, seq_len) - optional
        """
        batch_size = z.size(0)
        device = z.device

        # Prepare memory from input and latent
        memory = self.prepare_input_memory(z, input_seq)

        if target_seq is not None:
            # Teacher forcing mode - use target sequence
            target_shape_emb = self.output_shape_embedding(target_seq[:, 900:902].long())
            target_grid_emb = self.output_grid_embedding(target_seq[:, :900].long())
            target_combined = torch.cat([target_grid_emb, target_shape_emb], dim=1)

            # Add start token at the beginning
            start_tokens = self.start_token_embedding.repeat(batch_size, 1, 1)
            decoder_input = torch.cat([start_tokens, target_combined[:, :-1]], dim=1)

            # Create causal mask
            seq_len = decoder_input.size(1)
            causal_mask = self.create_causal_mask(seq_len, device)

            # Decoder forward pass
            decoder_output = self.transformer_decoder(decoder_input, memory, tgt_mask=causal_mask)

            # Generate predictions  
            shape_logits = self.shape_projection(decoder_output[:, -2:])  # Last 2 tokens for shape
            grid_logits = self.grid_projection(decoder_output[:, 1:901])  # Tokens 1-900 for grid (900 tokens)

            return shape_logits, grid_logits
        else:
            # Autoregressive generation mode (for inference)
            # This would be implemented for actual inference
            raise NotImplementedError("Autoregressive generation not implemented yet") 