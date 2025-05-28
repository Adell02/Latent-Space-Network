import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
from typing import Tuple, List
import pickle
import numpy as np
import json

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    set_seed, create_run_directory, setup_logging, prepare_dataloader,
    save_checkpoint, save_results, count_model_parameters, evaluate_model
)
from re_arc.main import generate_and_process_tasks
from utils.latent_functions import optimize_latent_z, get_optimized_z
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


##############################
# Main Training Function
##############################

def train_model(model, dataloader, optimizer, run_dir, logger, scaler, use_mixed_precision, gradient_accumulation_steps, current_epoch_num, total_epochs):
    model.train()
    epoch_total_loss = 0
    epoch_shape_loss_sum = 0
    epoch_grid_loss_sum = 0
    epoch_kl_loss_sum = 0
    
    optimizer.zero_grad() # Ensure gradients are zeroed at the start of accumulation cycle / epoch

    logger.info("-" * 60)
    logger.info(f"Starting training batch loop for Epoch {current_epoch_num}/{total_epochs}...")
    total_batches = len(dataloader)

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        device = next(model.parameters()).device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            # Pass beta from global settings (or training_settings directly)
            loss, shape_loss_comp, grid_loss_comp, kl_loss_comp = compute_loss(model, input_seq, target_seq, beta=BETA, return_components=True)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_total_loss += loss.item() * gradient_accumulation_steps # Unscale for logging
        epoch_shape_loss_sum += shape_loss_comp.item()
        epoch_grid_loss_sum += grid_loss_comp.item()
        epoch_kl_loss_sum += kl_loss_comp.item()
        
        progress = (batch_idx + 1) / total_batches * 100
        # Log less frequently if accumulating gradients
        log_frequency = gradient_accumulation_steps * 5 
        if (batch_idx + 1) % log_frequency == 0 or (batch_idx + 1) == total_batches:
            # Log individual unscaled losses for the current batch/step
            logger.info(f"Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
            logger.info(f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f} (Shape: {shape_loss_comp.item():.4f}, Grid: {grid_loss_comp.item():.4f}, KL: {kl_loss_comp.item():.4f})")


    avg_loss_for_epoch = epoch_total_loss / total_batches
    avg_shape_loss = epoch_shape_loss_sum / total_batches
    avg_grid_loss = epoch_grid_loss_sum / total_batches
    avg_kl_loss = epoch_kl_loss_sum / total_batches
    
    logger.info("=" * 60)
    logger.info(f"Epoch {current_epoch_num} Summary:")
    logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
    logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
    logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
    logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
    logger.info("=" * 60)

    return avg_loss_for_epoch, avg_shape_loss, avg_grid_loss, avg_kl_loss


def main_training(file_store_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reload settings variables to ensure they are current, especially if an optimization function was called
    global data_settings, model_architecture, training_settings, latent_optimization
    global TRAINING_KEYS, N_EXAMPLES_PER_TASK
    global LATENT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, BETA
    global OPTIMIZE_Z, OPTIMIZE_Z_NUM_STEPS, OPTIMIZE_Z_LR
    global OPTIMIZE_Z_INFERENCE, OPTIMIZE_Z_INFERENCE_NUM_STEPS, OPTIMIZE_Z_INFERENCE_LR

    settings.load_settings() # Force reload from file to ensure all parts of the code use updated settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()

    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings after reload.")
    N_EXAMPLES_PER_TASK = data_settings['n']
    
    LATENT_DIM = model_architecture['latent_dim']
    HIDDEN_DIM = model_architecture['hidden_dim']
    NUM_LAYERS = model_architecture['num_layers']
    NUM_HEADS = model_architecture['num_heads']
    DROPOUT = model_architecture['dropout']
    
    BATCH_SIZE = training_settings['batch_size']
    NUM_EPOCHS = training_settings['num_epochs']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']

    OPTIMIZE_Z = latent_optimization['training']['enabled']
    OPTIMIZE_Z_NUM_STEPS = latent_optimization['training']['num_steps']
    OPTIMIZE_Z_LR = latent_optimization['training']['learning_rate']
    OPTIMIZE_Z_INFERENCE = latent_optimization['inference']['enabled']
    OPTIMIZE_Z_INFERENCE_NUM_STEPS = latent_optimization['inference']['num_steps']
    OPTIMIZE_Z_INFERENCE_LR = latent_optimization['inference']['learning_rate']
    
    set_seed(data_settings['training_seed'])


    run_dir = create_run_directory(file_store_name)
    logger = setup_logging(run_dir)
    logger.info(f"Starting training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)

    logger.info("Generating and preparing data...")
    print("Generating and preparing data...")

    all_input_sequences = []
    all_output_sequences = []
    logger.info(f"Generating data for tasks: {TRAINING_KEYS}")
    print(f"Generating data for tasks: {TRAINING_KEYS}")

    for task_key in TRAINING_KEYS:
        logger.info(f"Processing task: {task_key} with {N_EXAMPLES_PER_TASK} examples")
        print(f"Processing task: {task_key} with {N_EXAMPLES_PER_TASK} examples")
        try:
            _, _, _, task_input_sequences, task_output_sequences = generate_and_process_tasks(task_key, N_EXAMPLES_PER_TASK)
            all_input_sequences.extend(task_input_sequences)
            all_output_sequences.extend(task_output_sequences)
            logger.info(f"Generated {len(task_input_sequences)} pairs for task {task_key}")
            print(f"Generated {len(task_input_sequences)} pairs for task {task_key}")
        except Exception as e:
            logger.error(f"Error generating data for task {task_key}: {e}")
            print(f"Error generating data for task {task_key}: {e}")
            continue 
    
    if not all_input_sequences:
        logger.error("No data generated from any task. Exiting training.")
        print("No data generated from any task. Exiting training.")
        return None, None

    input_sequences = all_input_sequences
    output_sequences = all_output_sequences
    logger.info(f"Total generated {len(input_sequences)} pairs of sequences from {len(TRAINING_KEYS)} tasks.")
    print(f"Total generated {len(input_sequences)} pairs of sequences from {len(TRAINING_KEYS)} tasks.")

    dataloader = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)

    logger.info("Initializing model...")
    print("Initializing model...")
    # Model instantiation will pick up global LATENT_DIM, HIDDEN_DIM etc. which are now reloaded
    model = LatentProgramNetwork().to(device) 

    optimizer_weight_decay = training_settings.get('optimizer_weight_decay', 0.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=optimizer_weight_decay)
    print(f"Model and optimizer initialized. Optimizer weight decay: {optimizer_weight_decay}")

    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    logger.info(f"Mixed precision training enabled: {use_mixed_precision}")

    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Learning rate scheduler
    lr_scheduler_config = training_settings.get('learning_rate_scheduler', {'type': 'none'})
    scheduler = None
    if lr_scheduler_config['type'] == 'cosine':
        # CosineAnnealingLR needs T_max which is total number of steps.
        # Total steps = (num_epochs - warmup_epochs) * len(dataloader) / gradient_accumulation_steps
        # This is a bit tricky if warmup is per epoch. Let's simplify:
        # If using warmup, scheduler starts after warmup.
        # For now, let's assume T_max is for the entire training duration after warmup.
        # A common setup for cosine annealing with warmup is to use warmup for N epochs,
        # then cosine anneal for M-N epochs.
        # Or, simpler: a warmup phase, then a fixed scheduler.
        # For this integration, let's stick to a simple CosineAnnealingLR without complex warmup logic
        # directly tied to step count, or use a simpler StepLR if cosine is too complex here.
        # A more robust implementation would wrap this in a custom scheduler class.
        # For now, this is a basic setup.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS * len(dataloader) // gradient_accumulation_steps, # Approximation of total steps
            eta_min=lr_scheduler_config.get('lr_min', 1e-6)
        )
        logger.info(f"Using CosineAnnealingLR scheduler. T_max={scheduler.T_max}, eta_min={scheduler.eta_min}")
    elif lr_scheduler_config['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_config.get('step_size', 30), gamma=lr_scheduler_config.get('gamma', 0.1))
        logger.info(f"Using StepLR scheduler. Step_size={scheduler.step_size}, gamma={scheduler.gamma}")


    count_model_parameters(model)
    print("Model parameter count completed.")

    results = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_metrics': [], # To store shape, grid, kl losses per epoch
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_sequences': [seq.tolist() for seq in input_sequences], # Convert to list for JSON
        'output_sequences': [seq.tolist() for seq in output_sequences], # Convert to list for JSON
        'losses_gradient_ascent': []
    }

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        logger.info("\\n" + "=" * 80)
        logger.info(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS} started. LR: {current_lr}")
        logger.info("=" * 80)

        avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss = train_model(
            model, dataloader, optimizer, run_dir, logger, 
            scaler, use_mixed_precision, gradient_accumulation_steps,
            current_epoch_num=epoch+1, total_epochs=NUM_EPOCHS
        )
        results['epoch_losses'].append(avg_loss)
        results['epoch_metrics'].append({
            'epoch': epoch + 1,
            'avg_shape_loss': avg_shape_loss,
            'avg_grid_loss': avg_grid_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_total_loss': avg_loss,
            'learning_rate': current_lr
        })
        
        if scheduler:
            scheduler.step() # Step the scheduler each epoch

        logger.info(f"\\nEpoch {epoch+1}/{NUM_EPOCHS} completed.")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate accuracy at the end of each epoch.
        model.eval()
        epoch_shape_correct = 0
        epoch_shape_tokens = 0
        epoch_grid_correct = 0
        epoch_grid_tokens = 0
        sample_exact_correct = 0
        total_samples_eval = 0 # Use a different variable for clarity during evaluation

        # We now leave the no_grad block for the latent optimization step if it's enabled for inference.
        # The get_optimized_z function handles its own grad context.
        for batch_input_eval, batch_target_eval in dataloader: # Can use a validation dataloader here if available
            total_samples_eval += batch_input_eval.size(0)
            batch_input_eval = batch_input_eval.to(device)
            batch_target_eval = batch_target_eval.to(device)

            # Use the appropriate latent optimization method for inference
            if OPTIMIZE_Z_INFERENCE: # Check inference specific flag
                # Pass inference specific num_steps and lr
                z_eval, losses_eval = get_optimized_z(model, batch_input_eval, batch_target_eval, 
                                                      num_steps=OPTIMIZE_Z_INFERENCE_NUM_STEPS, 
                                                      lr=OPTIMIZE_Z_INFERENCE_LR,
                                                      # Add a context flag if get_optimized_z needs to know it's inference
                                                      # context='inference' # if get_optimized_z signature is updated
                                                      )
                if losses_eval is not None and isinstance(results.get('losses_gradient_ascent'), list) : # Check if list before append
                    results['losses_gradient_ascent'].append(losses_eval) # This might grow very large
            else:
                with torch.no_grad(): # Ensure no grads if not optimizing z
                    mu_eval, log_var_eval = model.encoder(batch_input_eval, batch_target_eval)
                    z_eval = model.reparameterize(mu_eval, log_var_eval)

            # Now, perform decoding with no_grad.
            with torch.no_grad():
                # The decoder's target_seq argument is for teacher forcing. 
                # For true autoregressive generation during eval, it should be None
                # or handle it differently if evaluating reconstruction of target.
                # Current model.decoder uses target_seq for teacher-forced decoding.
                shape_logits_eval, grid_logits_eval = model.decoder(z_eval, batch_input_eval, target_seq=batch_target_eval)
                
                shape_pred_eval = shape_logits_eval.argmax(dim=-1)
                grid_pred_eval = grid_logits_eval.argmax(dim=-1)
                shape_tgt_eval = batch_target_eval[:, 900:902].long()
                grid_tgt_eval = batch_target_eval[:, :900].long()

            epoch_shape_correct += (shape_pred_eval == shape_tgt_eval).sum().item()
            epoch_shape_tokens += shape_tgt_eval.numel()
            for i in range(batch_input_eval.size(0)):
                tgt_rows_eval = int(batch_target_eval[i, 900].item())
                tgt_cols_eval = int(batch_target_eval[i, 901].item())
                active_pixels_eval = tgt_rows_eval * tgt_cols_eval
                if active_pixels_eval > 0:
                    epoch_grid_correct += (grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]).sum().item()
                    epoch_grid_tokens += active_pixels_eval
                    if torch.all(shape_pred_eval[i] == shape_tgt_eval[i]) and \
                       torch.all(grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]):
                        sample_exact_correct += 1
                elif torch.all(shape_pred_eval[i] == shape_tgt_eval[i]): # If grid is empty, only shape matters for exact
                    sample_exact_correct += 1


        epoch_shape_accuracy = epoch_shape_correct / epoch_shape_tokens if epoch_shape_tokens > 0 else 0.0
        epoch_grid_accuracy = epoch_grid_correct / epoch_grid_tokens if epoch_grid_tokens > 0 else 0.0
        epoch_overall_accuracy = (epoch_shape_correct + epoch_grid_correct) / (epoch_shape_tokens + epoch_grid_tokens) if (epoch_shape_tokens + epoch_grid_tokens) > 0 else 0.0
        sample_level_accuracy = sample_exact_correct / total_samples_eval if total_samples_eval > 0 else 0.0

        results['epoch_accuracies'].append({
            'epoch': epoch + 1,
            'shape_accuracy': epoch_shape_accuracy,
            'grid_accuracy': epoch_grid_accuracy,
            'overall_accuracy': epoch_overall_accuracy,
            'sample_exact_accuracy': sample_level_accuracy
        })

        logger.info(f"Epoch {epoch+1} Accuracy -- Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")
        print(f"Epoch {epoch+1} Accuracy: Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")

        # model.train() # Already called at the start of train_model function for the next epoch

        if (epoch + 1) % training_settings.get('save_checkpoint_interval', 50) == 0 or (epoch + 1) == NUM_EPOCHS : # Save more frequently or based on setting
            logger.info(f"Saving checkpoint at epoch {epoch+1}...")
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, run_dir) # Save epoch as 1-indexed
            print(f"Checkpoint saved at epoch {epoch+1}.")

    print("Training complete. Starting final evaluation on the dataloader (can be train or val)...")
    model.eval()
    # Final evaluation loop (similar to per-epoch evaluation but maybe on a dedicated test set if available)
    # For now, it re-uses the main dataloader for collecting final latent representations.
    # This part is mostly for collecting latent variables and reconstructions.
    final_eval_batch_count = 0
    for batch_input, batch_target in dataloader: # Consider using a different dataloader for final eval if needed
        final_eval_batch_count +=1
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        # Consistent z retrieval for final eval
        if OPTIMIZE_Z_INFERENCE:
            z, losses = get_optimized_z(model, batch_input, batch_target, 
                                        num_steps=OPTIMIZE_Z_INFERENCE_NUM_STEPS, 
                                        lr=OPTIMIZE_Z_INFERENCE_LR)
            # if losses is not None and isinstance(results.get('losses_gradient_ascent'), list):
            #    results['losses_gradient_ascent'].append(losses) # Decide if needed for final eval too
        else:
            with torch.no_grad():
                mu, log_var = model.encoder(batch_input, batch_target)
                z = model.reparameterize(mu, log_var)
        
        with torch.no_grad(): # Ensure no grads for decoder pass
            shape_logits, grid_logits = model.decoder(z, batch_input, target_seq=batch_target) # Using target_seq for reconstruction eval
            # Store only a subset of these potentially large tensors if memory is an issue
            if isinstance(results.get('latent_mus'), list):
                 results['latent_mus'].append(mu.cpu().numpy().tolist() if 'mu' in locals() else [])
            if isinstance(results.get('latent_log_vars'), list):
                 results['latent_log_vars'].append(log_var.cpu().numpy().tolist() if 'log_var' in locals() else [])
            if isinstance(results.get('latent_zs'), list):
                 results['latent_zs'].append(z.cpu().numpy().tolist())
            if isinstance(results.get('reconstructions'), list):
                 results['reconstructions'].append(
                     (shape_logits.cpu().numpy().tolist(), grid_logits.cpu().numpy().tolist())
                 )
        print(f"Final evaluation: Processed batch {final_eval_batch_count}/{len(dataloader)}")
        if final_eval_batch_count > 20 and len(dataloader)>20 : # Limit stored reconstructions for large datasets
            print(f"Limiting stored latent vars/reconstructions to first {final_eval_batch_count} batches to save memory.")
            break


    save_results(results, run_dir) # save_results now handles JSON directly if modified, or save a separate JSON here
    # Example of saving the main results dictionary to JSON (ensure tensors are converted to lists/numbers)
    try:
        # save_training_json(results, run_dir) # If you created this function
        pass # The save_results in model_utils.py should be updated or a new json save function added
    except Exception as e:
        logger.error(f"Could not save results as JSON: {e}")

    print("Results saved in:", run_dir)
    return results, model


##############################
# Run Inference
##############################

def main_test(model, keys, n_samples, n_queries, seed, device='cuda'):
    """
    Generate new data and evaluate the model on it.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: List of numbers of input-output pairs to generate
        n_queries: List of numbers of queries to do inference
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation results for each key and n_samples   
    """

    set_seed(seed)
    results = {}
    
    for key in keys:
        results[key] = {}
        print(f"\nEvaluating key {key} with {n_samples} samples and {n_queries} queries...")
        
        # Generate new data
        _, _, _, input_samples_sequences, output_samples_sequences = generate_and_process_tasks(key, n_samples)
        samples_dataloader = prepare_dataloader(input_samples_sequences, output_samples_sequences, BATCH_SIZE)

        # Generate queries
        _, _, _, input_queries_sequences, output_queries_sequences = generate_and_process_tasks(key, n_queries)
        queries_dataloader = prepare_dataloader(input_queries_sequences, output_queries_sequences, BATCH_SIZE)

        # Evaluate overall performance
        metrics = evaluate_model(model, samples_dataloader, queries_dataloader, device=device)

        results[key] = metrics
        results[key]['reconstruction_results'] = {
            'input_samples_sequences': input_samples_sequences,
            'output_samples_sequences': output_samples_sequences,
            'input_queries_sequences': input_queries_sequences,
            'output_queries_sequences': output_queries_sequences,
            'support_reconstructions': results[key]['reconstruction_results']['support_reconstructions'],
            'query_reconstructions': results[key]['reconstruction_results']['query_reconstructions']
        }

    return results