import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
from typing import Tuple, List
import pickle

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    set_seed, create_run_directory, setup_logging, prepare_dataloader,
    save_checkpoint, save_results, count_model_parameters, evaluate_model
)
from re_arc.main import generate_and_process_tasks

from utils.latent_functions import optimize_latent_z


#########################################
# TUNABLE SETTINGS
#########################################

# Data and Run Settings
KEY = "00d62c1b"                # Key to the problem #017c7c7b 00d62c1b 007bbfb7
n = 10                           # Number of generated examples to train per batch
RUN_BASE_DIR = "runs_re_arc"    # Base directory to save run outputs
TRAINING_SEED = 42

# DataLoader Settings
BATCH_SIZE = 128

# Model Architecture Settings
LATENT_DIM = 256                 # Dimensionality of the latent space
HIDDEN_DIM = 256                 # Dimensionality of embeddings / hidden states
NUM_LAYERS = 6                   # Number of transformer layers (encoder and decoder)
NUM_HEADS = 8                    # Number of attention heads
DROPOUT = 0.1                    # Dropout rate
MAX_LENGTH = 902                 # Length of each sequence (900 grid values + 2 shape values)
ENCODER_MAX_LENGTH = 1805        # For full sequence (input + output + CLS)
DECODER_MAX_LENGTH = 902         # For output sequence

# Training Settings
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4

# Add beta parameter to TUNABLE SETTINGS
BETA = 1.0  # Weighting factor for KL loss term

# Latent Optimization Settings (for inference and optionally during training)
OPTIMIZE_Z = True               # Set to True to run latent optimization
OPTIMIZE_Z_NUM_STEPS = 25       # Number of gradient steps to optimize z
OPTIMIZE_Z_LR = 0.5             # Learning rate for latent z optimization

# Latent Optimization Settings (for inference and optionally during training)
OPTIMIZE_Z_INFERENCE = True               # Set to True to run latent optimization
OPTIMIZE_Z_INFERENCE_NUM_STEPS = 25       # Number of gradient steps to optimize z
OPTIMIZE_Z_INFERENCE_LR = 0.5             # Learning rate for latent z optimization


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

        if target_seq is not None:
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
        else:
            # ----- Autoregressive Mode (Fallback) -----
            current_output = self.start_token_embedding.repeat(batch_size, 1, 1)
            shape_logits_list = []
            for i in range(2):
                tgt_mask = self.create_causal_mask(current_output.size(1), device)
                decoder_output = self.transformer_decoder(tgt=current_output, memory=memory, tgt_mask=tgt_mask)
                decoder_output = self.layer_norm(decoder_output[:, -1:])
                shape_logit = self.shape_output(decoder_output)
                shape_logits_list.append(shape_logit)
                next_token = shape_logit.argmax(dim=-1)
                token_embedding = self.output_shape_embedding(next_token)
                current_output = torch.cat([current_output, token_embedding], dim=1)
            shape_logits = torch.cat(shape_logits_list, dim=1)
            grid_logits_list = []
            chunk_size = 30
            for chunk_start in range(0, 900, chunk_size):
                chunk_predictions = []
                for i in range(chunk_start, chunk_start + chunk_size):
                    curr_row = i // 30
                    curr_col = i % 30
                    tgt_mask = self.create_causal_mask(current_output.size(1), device)
                    decoder_output = self.transformer_decoder(tgt=current_output, memory=memory, tgt_mask=tgt_mask)
                    decoder_output = self.layer_norm(decoder_output[:, -1:])
                    grid_logit = self.grid_output(decoder_output)
                    chunk_predictions.append(grid_logit)
                    next_token = grid_logit.argmax(dim=-1)
                    token_embedding = self.output_grid_embedding(next_token)
                    pos_embedding = self.get_position_embedding(curr_row, curr_col, device)
                    token_embedding = token_embedding + pos_embedding
                    current_output = torch.cat([current_output, token_embedding], dim=1)
                chunk_logits = torch.cat(chunk_predictions, dim=1)
                grid_logits_list.append(chunk_logits)
            grid_logits = torch.cat(grid_logits_list, dim=1)
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

def compute_loss(model: nn.Module, input_seq: torch.Tensor, target_seq: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    """
    Compute the total loss for the model using cross-pair reconstruction.
    
    The loss is composed of:
      - A shape loss computed over the two shape tokens
      - A grid loss computed *only* over the active region of the grid, as defined by the target's shape
      - A KL divergence loss on the latent parameters weighted by beta
    
    This formulation forces 100% accuracy (i.e. a zero loss) only if the entire active output
    (all pixels in the target region) are exactly reconstructed.
    """
    if OPTIMIZE_Z:
        reconstruction, mu, log_var = model(input_seq, target_seq)
        shape_logits, grid_logits = reconstruction
        grid_targets = target_seq[:, :900].long()
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
        grid_loss = F.cross_entropy(grid_logits.reshape(-1, 10), grid_targets.reshape(-1), ignore_index=-1)
        reconstruction_loss = shape_loss + grid_loss
        kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var)
        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss

    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Ensure inputs are on the correct device
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)
    
    batch_size = input_seq.size(0)
    total_loss = 0.0
    
    # For each sample in the batch, use all other samples to reconstruct it
    for i in range(batch_size):
        # Get the target sample to reconstruct
        target_input = input_seq[i:i+1]  # Keep batch dimension
        target_output = target_seq[i:i+1]
        
        # Get all other samples in the batch
        other_inputs = torch.cat([input_seq[:i], input_seq[i+1:]])
        other_outputs = torch.cat([target_seq[:i], target_seq[i+1:]])
        
        # Encode all other samples
        mu, log_var = model.encoder(other_inputs, other_outputs)
        
        # Sample from the aggregated distribution (mean of all other samples)
        z = model.reparameterize(mu.mean(dim=0, keepdim=True), log_var.mean(dim=0, keepdim=True))
        
        # Decode using the target input and sampled latent
        shape_logits, grid_logits = model.decoder(z, target_input)
        
        # Compute shape loss (for the two shape tokens)
        shape_targets = target_output[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
        # Compute grid loss only over the active region
        tgt_rows = int(target_output[0, 900].item())
        tgt_cols = int(target_output[0, 901].item())
        active_pixels = tgt_rows * tgt_cols
        
        # Compute cross-entropy only over the active region
        grid_loss = F.cross_entropy(grid_logits[0, :active_pixels], target_output[0, :active_pixels].long())
        
        # Total reconstruction loss for this sample
        reconstruction_loss = shape_loss + grid_loss
        
        # KL divergence loss (weighted by beta)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var)
        
        # Add to total loss
        total_loss += reconstruction_loss + beta * kl_loss
    
    # Average over batch
    return total_loss / batch_size


##############################
# Main Training Function
##############################

def train_model(model, dataloader, optimizer, run_dir, logger):
    model.train()
    total_loss = 0
    shape_loss_sum = 0
    grid_loss_sum = 0
    kl_loss_sum = 0

    logger.info("-" * 60)
    logger.info("Starting training batch loop...")
    total_batches = len(dataloader)

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        progress = (batch_idx + 1) / total_batches * 100
        device = next(model.parameters()).device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        loss = compute_loss(model, input_seq, target_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute individual losses for logging purposes
        shape_logits, grid_logits = model.decoder(model.reparameterize(*model.encoder(input_seq, target_seq)), input_seq, target_seq=target_seq)
        grid_targets = target_seq[:, :900].long()
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        grid_loss = F.cross_entropy(grid_logits.reshape(-1, 10), grid_targets.reshape(-1), ignore_index=-1)
        reconstruction_loss = shape_loss + grid_loss
        mu, log_var = model.encoder(input_seq, target_seq)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        shape_loss_sum += shape_loss.item()
        grid_loss_sum += grid_loss.item()
        kl_loss_sum += kl_loss.item()

        logger.info(f"Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
        logger.info(f"  Shape Loss: {shape_loss.item():.4f}")
        logger.info(f"  Grid Loss: {grid_loss.item():.4f}")
        logger.info(f"  KL Loss: {kl_loss.item():.4f}")
        logger.info(f"  Total Loss: {loss.item():.4f}")

        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_shape_loss = shape_loss_sum / (batch_idx + 1)
            avg_grid_loss = grid_loss_sum / (batch_idx + 1)
            avg_kl_loss = kl_loss_sum / (batch_idx + 1)
            logger.info("\nRunning Averages:")
            logger.info(f"  Avg Shape Loss: {avg_shape_loss:.4f}")
            logger.info(f"  Avg Grid Loss: {avg_grid_loss:.4f}")
            logger.info(f"  Avg KL Loss: {avg_kl_loss:.4f}")
            logger.info(f"  Avg Total Loss: {avg_loss:.4f}\n")

    avg_loss = total_loss / total_batches
    avg_shape_loss = shape_loss_sum / total_batches
    avg_grid_loss = grid_loss_sum / total_batches
    avg_kl_loss = kl_loss_sum / total_batches

    logger.info("=" * 60)
    logger.info("Epoch Summary:")
    logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
    logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
    logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
    logger.info(f"  Final Avg Total Loss: {avg_loss:.4f}")
    logger.info("=" * 60)

    return avg_loss


##############################
# Run Inference
##############################

def evaluate_model_on_new_data(model, keys, n_values, seed, device='cuda'):
    """
    Generate new data and evaluate the model on it.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_values: List of numbers of examples to generate
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation results for each key and n
    """

    set_seed(seed)
    results = {}
    
    for key in keys:
        results[key] = {}
        print(f"\nEvaluating key {key} with {n_values} examples...")
        
        # Generate new data
        _, _, _, input_sequences, output_sequences = generate_and_process_tasks(key, n_values)
        test_dataloader = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)

        # Evaluate overall performance
        metrics = evaluate_model(model, test_dataloader, device=device)

        results[key] = metrics
        results[key]['reconstruction_results'] = {
            'input_sequences': input_sequences,
            'output_sequences': output_sequences,
            'reconstructions': results[key]['reconstruction_results']['reconstructions']
        }

    return results



def main_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_dir = create_run_directory()
    logger = setup_logging(run_dir)
    logger.info(f"Starting training for ARC problem {KEY}")
    print("Run directory created:", run_dir)

    logger.info("Generating and preparing data...")
    print("Generating and preparing data...")
    # Use the generated examples function with the defined n examples.
    _,_,_,input_sequences,output_sequences = generate_and_process_tasks(KEY, n)
    logger.info(f"Generated {len(input_sequences)} pairs of sequences")
    print(f"Generated {len(input_sequences)} pairs of sequences.")

    dataloader = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)

    logger.info("Initializing model...")
    print("Initializing model...")
    model = LatentProgramNetwork().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model and optimizer initialized.")

    count_model_parameters(model)
    print("Model parameter count completed.")

    # Initialize results dictionary with an epoch accuracy list.
    results = {
        'epoch_losses': [],
        'epoch_accuracies': [],  # Will hold accuracy per epoch
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_sequences': input_sequences,
        'output_sequences': output_sequences
    }

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        logger.info("\n" + "=" * 80)
        logger.info(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} started.")
        logger.info("=" * 80)

        avg_loss = train_model(model, dataloader, optimizer, run_dir, logger)
        results['epoch_losses'].append(avg_loss)
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed.")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate accuracy at the end of each epoch.
        model.eval()
        epoch_shape_correct = 0
        epoch_shape_tokens = 0
        epoch_grid_correct = 0
        epoch_grid_tokens = 0
        sample_exact_correct = 0  # Count of samples that exactly match the output.
        total_samples = 0

        # We now leave the no_grad block for the latent optimization step.
        for batch_input, batch_target in dataloader:
            total_samples += batch_input.size(0)
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Optimize z with gradients enabled:
            if OPTIMIZE_Z:
                z = None
                with torch.enable_grad():
                    z = optimize_latent_z(model, batch_input, batch_target,
                                          num_steps=OPTIMIZE_Z_NUM_STEPS, lr=OPTIMIZE_Z_LR)
            else:
                mu, log_var = model.encoder(batch_input, batch_target)
                z = model.reparameterize(mu, log_var)

            # Now, perform decoding with no_grad.
            with torch.no_grad():
                shape_logits, grid_logits = model.decoder(z, batch_target, target_seq=batch_target)
                shape_pred = shape_logits.argmax(dim=-1)  # (batch_size, 2)
                grid_pred = grid_logits.argmax(dim=-1)      # (batch_size, 900)
                shape_tgt = batch_target[:, 900:902].long()
                grid_tgt = batch_target[:, :900].long()

            epoch_shape_correct += (shape_pred == shape_tgt).sum().item()
            epoch_shape_tokens += shape_tgt.numel()
            for i in range(batch_input.size(0)):
                tgt_rows = int(batch_target[i, 900].item())
                tgt_cols = int(batch_target[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                epoch_grid_correct += (grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                epoch_grid_tokens += active_pixels
                if torch.all(shape_pred[i] == shape_tgt[i]) and torch.all(grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]):
                    sample_exact_correct += 1

        epoch_shape_accuracy = epoch_shape_correct / epoch_shape_tokens if epoch_shape_tokens > 0 else 0.0
        epoch_grid_accuracy = epoch_grid_correct / epoch_grid_tokens if epoch_grid_tokens > 0 else 0.0
        epoch_overall_accuracy = (epoch_shape_correct + epoch_grid_correct) / (epoch_shape_tokens + epoch_grid_tokens) if (epoch_shape_tokens + epoch_grid_tokens) > 0 else 0.0
        sample_level_accuracy = sample_exact_correct / total_samples if total_samples > 0 else 0.0

        results['epoch_accuracies'].append({
            'epoch': epoch + 1,
            'shape_accuracy': epoch_shape_accuracy,
            'grid_accuracy': epoch_grid_accuracy,
            'overall_accuracy': epoch_overall_accuracy,
            'sample_exact_accuracy': sample_level_accuracy
        })

        logger.info(f"Epoch {epoch+1} Accuracy -- Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")
        print(f"Epoch {epoch+1} Accuracy: Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")

        model.train()  # Return to training mode for next epoch

        if (epoch + 1) % 5 == 0:
            logger.info(f"Saving checkpoint at epoch {epoch+1}...")
            save_checkpoint(model, optimizer, epoch, avg_loss, run_dir)
            print(f"Checkpoint saved at epoch {epoch+1}.")

    print("Training complete. Starting final evaluation...")
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        for batch_input, batch_target in dataloader:
            batch_idx += 1
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            mu, log_var = model.encoder(batch_input, batch_target)
            if OPTIMIZE_Z:
                with torch.enable_grad():
                    z = optimize_latent_z(model, batch_input, batch_target, num_steps=OPTIMIZE_Z_NUM_STEPS, lr=OPTIMIZE_Z_LR)
            else:
                z = model.reparameterize(mu, log_var)
            shape_logits, grid_logits = model.decoder(z, batch_target, target_seq=batch_target)
            results['latent_mus'].append(mu.cpu())
            results['latent_log_vars'].append(log_var.cpu())
            results['latent_zs'].append(z.cpu())
            results['reconstructions'].append((shape_logits.cpu(), grid_logits.cpu()))
            print(f"Final evaluation: Processed batch {batch_idx}")

    save_results(results, run_dir)
    print("Results saved in:", run_dir)
    return results, model
