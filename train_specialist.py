#!/usr/bin/env python3
"""
Specialist Training Script for Multi-Encoder Models

Implements 2-phase training approach:
- Phase A: Train each encoder with its independent decoder on domain-specific data
- Phase B: Train shared decoder using PoE of all pre-trained encoders

Usage:
    python train_specialist.py [--phases A,B] [--resume_from_phase PHASE]
"""

import torch
from torch.optim import Adam
import json
import numpy as np
import os
import argparse
from tqdm import tqdm

from models.base_model import LatentProgramNetwork, compute_loss
from utils.settings_manager import settings
from re_arc.main import generate_and_process_tasks
from utils.data_preparation import split_dataset_by_keys_for_multi_encoder
from utils.training_helpers import (
    create_mixed_domains_dataloader,
    setup_phase_training,
    save_encoder_checkpoint,
    load_all_encoder_checkpoints,
    save_decoder_checkpoint,
    load_decoder_checkpoint,
    save_full_model_checkpoint,
    save_phase_checkpoint,
    print_parameter_status,
    count_trainable_parameters,
    save_independent_decoder_checkpoint,
    load_all_independent_decoder_checkpoints,
    initialize_shared_decoder_from_independent_decoders,
)

from utils.model_utils import (
    set_seed,
    create_run_directory,
    setup_logging,
    prepare_dataloader,
    save_checkpoint,
    save_results,
    count_model_parameters,
    save_model_params,
    collect_latent_data,
)

from utils.wandb_logger import init_wandb_for_mode, get_wandb_logger
from utils.evaluation_utils import run_quick_evaluation, should_run_evaluation, log_evaluation_to_wandb


def generate_specialist_reconstruction_plot(model, dataloader, device, epoch, phase, encoder_idx=None, 
                                          use_independent_decoder=False, key="sample", return_mu_sigma=False):
    """
    Generate a simple Input-Target-Reconstruction plot for specialist training.
    
    Args:
        model: Current model state
        dataloader: Data loader with samples
        device: Device to run on
        epoch: Current epoch/step
        phase: Training phase ('A' or 'B')
        encoder_idx: Encoder index for Phase A (None for Phase B PoE)
        use_independent_decoder: Whether to use independent decoder
        key: Sample key for naming
        return_mu_sigma: Whether to return latent statistics
        
    Returns:
        tuple: (plot_path, wandb_key) for WandB logging
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from utils.data_preparation import extract_grid_from_sequence
    from models.base_model import compute_loss
    
    model.eval()
    with torch.no_grad():
        # Get the first sample from the dataloader
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Take only the first sample
            input_sample = input_seq[0:1]
            target_sample = target_seq[0:1]
            
            # Generate reconstruction
            if encoder_idx is not None:
                # Phase A: specific encoder + independent decoder
                mu, log_var = model.multi_encoder.encoders[encoder_idx](input_sample, target_sample)
                z = model.reparameterize(mu, log_var)
                if use_independent_decoder:
                    shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](z, input_sample, target_seq=target_sample)
                else:
                    shape_logits, grid_logits = model.multi_encoder.shared_decoder(z, input_sample, target_seq=target_sample)
            else:
                # Phase B: PoE + shared decoder
                mu, log_var = model(input_sample, target_sample)[1:3]  # PoE inference
                z = model.reparameterize(mu, log_var)
                shape_logits, grid_logits = model.multi_encoder.shared_decoder(z, input_sample, target_seq=target_sample)
            
            # Extract grids for visualization
            input_grid, input_shape = extract_grid_from_sequence(input_sample[0].cpu().numpy())
            target_grid, target_shape = extract_grid_from_sequence(target_sample[0].cpu().numpy())
            
            # Create reconstruction grid from model outputs
            shape_pred = shape_logits[0].argmax(dim=-1).cpu().numpy()
            grid_pred = grid_logits[0].argmax(dim=-1).cpu().numpy()
            
            # Reconstruct the output sequence and extract grid
            recon_seq = target_sample[0].cpu().numpy().copy()
            recon_seq[900:902] = shape_pred  # Shape dimensions
            if len(shape_pred) >= 2 and shape_pred[0] > 0 and shape_pred[1] > 0:
                recon_seq[:min(len(grid_pred), 900)] = grid_pred[:min(len(grid_pred), 900)]
            
            recon_grid, recon_shape = extract_grid_from_sequence(recon_seq)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Input
            axes[0].imshow(input_grid, cmap='viridis', interpolation='nearest')
            axes[0].set_title(f'Input\n{input_shape[0]}×{input_shape[1]}')
            axes[0].axis('off')
            
            # Target
            axes[1].imshow(target_grid, cmap='viridis', interpolation='nearest')
            axes[1].set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
            axes[1].axis('off')
            
            # Reconstruction
            axes[2].imshow(recon_grid, cmap='viridis', interpolation='nearest')
            axes[2].set_title(f'Reconstruction\n{recon_shape[0]}×{recon_shape[1]}')
            axes[2].axis('off')
            
            # Set overall title
            if phase == 'A':
                fig.suptitle(f'Phase A Epoch {epoch}: Encoder {encoder_idx} + Independent Decoder', fontsize=14)
                wandb_key = f'encoder_{encoder_idx}_{key}'
            else:
                fig.suptitle(f'Phase B Epoch {epoch}: PoE + Shared Decoder', fontsize=14)
                wandb_key = f'poe_{key}'
            
            plt.tight_layout()
            
            # Save plot (we don't need to keep the file, just for temp WandB upload)
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                plot_path = tmp_file.name
            plt.close()
            
            # Compute latent statistics if requested
            if return_mu_sigma:
                mu_mean = mu.detach().cpu().mean().item()
                sigma_mean = torch.exp(0.5 * log_var).detach().cpu().mean().item()
            
            if return_mu_sigma:
                return plot_path, wandb_key, mu_mean, sigma_mean
            else:
                return plot_path, wandb_key
            
    if return_mu_sigma:
        return None, None, None, None
    else:
        return None, None


def generate_phase_b_final_comparison_plot(model, encoder_datasets, device, run_dir, wandb_logger, global_step):
    """
    Generate a comprehensive comparison plot showing each encoder's reconstruction 
    (using their independent decoders) vs PoE reconstruction (using shared decoder)
    with bar graph of sigma values showing uncertainty distributions.
    
    VISUALIZATION BEHAVIOR:
    - Each encoder uses its own INDEPENDENT DECODER for reconstruction comparison
    - PoE uses the SHARED DECODER for reconstruction 
    - Shows uncertainty distributions (sigma bar graph) for all latent representations
    - Demonstrates encoder specialization vs PoE combination
    - Uses representative sample selection to ensure fair comparison
    
    Args:
        model: Current model state
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to run on
        run_dir: Run directory for saving plots
        wandb_logger: WandB logger instance
        global_step: Current training step
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from utils.data_preparation import extract_grid_from_sequence
    from utils.settings_manager import settings
    import tempfile
    import os
    import random
    
    model.eval()
    with torch.no_grad():
        num_encoders = len(encoder_datasets)
        
        # Get settings for reproducible sample selection
        data_settings = settings.get_data_settings()
        training_keys = data_settings.get('training_keys', [])
        eval_seed = data_settings.get('eval_seed', 42)
        
        # Set seed for deterministic sample selection
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        
        # Find a representative sample that exists across multiple encoders for fair comparison
        representative_sample = None
        sample_info = None
        
        # Strategy: Use first available sample from the encoder with most data
        encoder_with_most_data = 0
        max_samples = 0
        for enc_idx, (inputs, outputs) in enumerate(encoder_datasets):
            if inputs and outputs and len(inputs) > max_samples:
                max_samples = len(inputs)
                encoder_with_most_data = enc_idx
        
        inputs, outputs = encoder_datasets[encoder_with_most_data]
        if not inputs or not outputs:
            print(f"⚠ Warning: No data available for Phase B final comparison plot")
            return
            
        # Log which encoder's data we're using for transparency
        encoder_keys = []
        try:
            # Try to get key information from splitting statistics if available
            splitting_stats_file = os.path.join(run_dir, 'splitting_statistics.json')
            if os.path.exists(splitting_stats_file):
                import json
                with open(splitting_stats_file, 'r') as f:
                    splitting_stats = json.load(f)
                    if 'keys_per_encoder' in splitting_stats:
                        encoder_keys = splitting_stats['keys_per_encoder'].get(str(encoder_with_most_data), [])
        except Exception:
            pass
        
        if not encoder_keys and training_keys:
            # Fallback: assign keys based on encoder index
            encoder_keys = [training_keys[encoder_with_most_data % len(training_keys)]]
        
        sample_info = {
            'source_encoder': encoder_with_most_data,
            'encoder_keys': encoder_keys,
            'total_samples': len(inputs),
            'sample_index': 0  # Always use first sample for consistency
        }
        
        print(f"Phase B Final Plot - Using sample from Encoder {encoder_with_most_data}")
        print(f"  Source encoder keys: {encoder_keys}")
        print(f"  Total samples available: {len(inputs)}")
        print(f"  Sample index: {sample_info['sample_index']}")
        
        from utils.model_utils import prepare_dataloader
        dataloader = prepare_dataloader(inputs, outputs, batch_size=1)
        
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Take only the first sample
            input_sample = input_seq[0:1]
            target_sample = target_seq[0:1]
            
            # Extract input and target grids for reference
            input_grid, input_shape = extract_grid_from_sequence(input_sample[0].cpu().numpy())
            target_grid, target_shape = extract_grid_from_sequence(target_sample[0].cpu().numpy())
            
            # Create figure with two rows: reconstructions on top, sigma bar graph below
            fig_width = 4 * (2 + num_encoders + 1)  # Input, Target, N encoders, PoE
            fig, axes = plt.subplots(2, 2 + num_encoders + 1, figsize=(fig_width, 12))
            
            # Top row for reconstructions
            recon_axes = axes[0]
            
            # Input (with source information)
            recon_axes[0].imshow(input_grid, cmap='viridis', interpolation='nearest')
            input_title = f'Input\n{input_shape[0]}×{input_shape[1]}'
            if encoder_keys:
                input_title += f'\nFrom: {encoder_keys[0]}'
            recon_axes[0].set_title(input_title)
            recon_axes[0].axis('off')
            
            # Target
            recon_axes[1].imshow(target_grid, cmap='viridis', interpolation='nearest')
            recon_axes[1].set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
            recon_axes[1].axis('off')
            
            encoder_stats = []
            all_sigma_values = []  # Store sigma values for bar graph
            encoder_colors = plt.cm.Set1(np.linspace(0, 1, num_encoders + 1))  # +1 for PoE
            
            # Each encoder with its independent decoder
            for enc_idx in range(num_encoders):
                # Generate reconstruction using encoder + independent decoder
                mu, log_var = model.multi_encoder.encoders[enc_idx](input_sample, target_sample)
                z = model.reparameterize(mu, log_var)
                shape_logits, grid_logits = model.multi_encoder.independent_decoders[enc_idx](z, input_sample, target_seq=target_sample)
                
                # Compute latent statistics - keep full sigma vector for bar graph
                mu_mean = mu.detach().cpu().mean().item()
                sigma_values = torch.exp(0.5 * log_var).detach().cpu().flatten().numpy()  # All sigma values
                sigma_mean = sigma_values.mean()
                encoder_stats.append((mu_mean, sigma_mean))
                all_sigma_values.append((f'Encoder {enc_idx}', sigma_values, encoder_colors[enc_idx]))
                
                # Create reconstruction grid
                shape_pred = shape_logits[0].argmax(dim=-1).cpu().numpy()
                grid_pred = grid_logits[0].argmax(dim=-1).cpu().numpy()
                
                recon_seq = target_sample[0].cpu().numpy().copy()
                recon_seq[900:902] = shape_pred
                if len(shape_pred) >= 2 and shape_pred[0] > 0 and shape_pred[1] > 0:
                    recon_seq[:min(len(grid_pred), 900)] = grid_pred[:min(len(grid_pred), 900)]
                
                recon_grid, recon_shape = extract_grid_from_sequence(recon_seq)
                
                # Plot reconstruction with familiarity indicator
                ax_idx = 2 + enc_idx
                recon_axes[ax_idx].imshow(recon_grid, cmap='viridis', interpolation='nearest')
                
                # Determine if this encoder is familiar with this sample
                is_familiar = (enc_idx == encoder_with_most_data)
                familiarity_indicator = "👁️" if is_familiar else "❓"
                
                title = f'{familiarity_indicator} Encoder {enc_idx}\n+ Independent Decoder\nμ̄={mu_mean:.3f}, σ̄={sigma_mean:.3f}'
                recon_axes[ax_idx].set_title(title, fontsize=10)
                recon_axes[ax_idx].axis('off')
            
            # PoE + Shared Decoder
            mu_poe, log_var_poe = model(input_sample, target_sample)[1:3]  # PoE inference
            z_poe = model.reparameterize(mu_poe, log_var_poe)
            shape_logits_poe, grid_logits_poe = model.multi_encoder.shared_decoder(z_poe, input_sample, target_seq=target_sample)
            
            # Compute PoE latent statistics
            mu_poe_mean = mu_poe.detach().cpu().mean().item()
            sigma_poe_values = torch.exp(0.5 * log_var_poe).detach().cpu().flatten().numpy()  # All sigma values
            sigma_poe_mean = sigma_poe_values.mean()
            all_sigma_values.append(('PoE', sigma_poe_values, encoder_colors[num_encoders]))
            
            # Create PoE reconstruction grid
            shape_pred_poe = shape_logits_poe[0].argmax(dim=-1).cpu().numpy()
            grid_pred_poe = grid_logits_poe[0].argmax(dim=-1).cpu().numpy()
            
            recon_seq_poe = target_sample[0].cpu().numpy().copy()
            recon_seq_poe[900:902] = shape_pred_poe
            if len(shape_pred_poe) >= 2 and shape_pred_poe[0] > 0 and shape_pred_poe[1] > 0:
                recon_seq_poe[:min(len(grid_pred_poe), 900)] = grid_pred_poe[:min(len(grid_pred_poe), 900)]
            
            recon_grid_poe, recon_shape_poe = extract_grid_from_sequence(recon_seq_poe)
            
            # Plot PoE reconstruction
            poe_ax_idx = 2 + num_encoders
            recon_axes[poe_ax_idx].imshow(recon_grid_poe, cmap='viridis', interpolation='nearest')
            recon_axes[poe_ax_idx].set_title(f'🔄 PoE + Shared Decoder\nμ̄={mu_poe_mean:.3f}, σ̄={sigma_poe_mean:.3f}')
            recon_axes[poe_ax_idx].axis('off')
            
            # Create bar graph of sigma values in the bottom row
            # Use the middle columns for the bar graph spanning across most of the width
            hist_start_col = 1
            hist_end_col = 2 + num_encoders
            
            # Remove individual subplot axes in the bar graph area and create a single spanning axis
            for i in range(hist_start_col, hist_end_col + 1):
                axes[1, i].remove()
            
            # Create a new subplot that spans the desired columns
            bar_ax = plt.subplot2grid((2, 2 + num_encoders + 1), (1, hist_start_col), 
                                     colspan=hist_end_col - hist_start_col + 1, fig=fig)
            
            # Create bar graph with latent position on x-axis and sigma values on y-axis
            latent_dim = all_sigma_values[0][1].shape[0]  # Get latent dimension
            x_positions = np.arange(latent_dim)
            
            # Set width for bars (make them narrower so multiple encoders can fit)
            bar_width = 0.8 / len(all_sigma_values)
            
            for idx, (name, sigma_vals, color) in enumerate(all_sigma_values):
                # Calculate offset for this encoder's bars
                x_offset = (idx - len(all_sigma_values)/2 + 0.5) * bar_width
                
                # Plot bars for this encoder/PoE
                bar_ax.bar(x_positions + x_offset, sigma_vals, width=bar_width, 
                          alpha=0.7, color=color, label=name)
            
            bar_ax.set_xlabel('Latent Dimension Index')
            bar_ax.set_ylabel('Sigma (Uncertainty) Values')
            bar_ax.set_title('Latent Uncertainty per Dimension\n(Lower σ = more certain; PoE should combine most certain elements)')
            bar_ax.legend()
            bar_ax.grid(True, alpha=0.3, axis='y')
            
            # Set x-axis ticks to show latent dimension indices
            bar_ax.set_xticks(x_positions[::max(1, latent_dim//20)])  # Show every nth tick to avoid crowding
            bar_ax.set_xticklabels(x_positions[::max(1, latent_dim//20)])
            
            # Add mean lines as horizontal reference
            for name, sigma_vals, color in all_sigma_values:
                mean_val = sigma_vals.mean()
                bar_ax.axhline(mean_val, color=color, linestyle='--', alpha=0.8, 
                              label=f'{name} mean: {mean_val:.3f}')
            
            # Hide unused bar graph subplot areas
            axes[1, 0].axis('off')  # Hide first column in bottom row
            if poe_ax_idx + 1 < len(axes[1]):  # Hide any remaining columns
                axes[1, -1].axis('off')
            
            # Set overall title with statistics summary and sample information
            stats_text = " | ".join([f"Enc{i}: μ̄={stats[0]:.3f}, σ̄={stats[1]:.3f}" for i, stats in enumerate(encoder_stats)])
            sample_source = f"Sample from Encoder {encoder_with_most_data}" + (f" ({encoder_keys[0]})" if encoder_keys else "")
            fig.suptitle(f'Phase B Final: Encoder Reconstructions vs PoE + Uncertainty Analysis\n{sample_source} | {stats_text} | PoE: μ̄={mu_poe_mean:.3f}, σ̄={sigma_poe_mean:.3f}', fontsize=11)
            
            plt.tight_layout()
            
            # Save plot
            comparison_plot_path = os.path.join(run_dir, 'phase_b_final_encoder_poe_comparison.png')
            plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
            
            # Also save as temp file for WandB upload
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                temp_plot_path = tmp_file.name
            
            plt.close()
            
            # Log to WandB with detailed sample information
            if wandb_logger:
                try:
                    import wandb
                    wandb_logger._safe_log({
                        "phase_b_final/encoder_poe_comparison": wandb.Image(temp_plot_path),
                        "phase_b_final/sample_source_encoder": encoder_with_most_data,
                        "phase_b_final/sample_keys": encoder_keys,
                        "phase_b_final/sample_index": sample_info['sample_index']
                    }, step_hint=global_step)
                    
                    print(f"✓ Logged Phase B final encoder vs PoE comparison plot")
                    print(f"  Sample source: Encoder {encoder_with_most_data} ({encoder_keys})")
                    print(f"  Familiarity indicators: 👁️=familiar, ❓=unfamiliar")
                except Exception as wandb_error:
                    print(f"⚠ Failed to log Phase B final comparison plot: {wandb_error}")
                finally:
                    # Clean up temp file
                    os.unlink(temp_plot_path)
            
            break  # Only process first sample


def build_model(device):
    """Build and return LatentProgramNetwork."""
    return LatentProgramNetwork().to(device)


def train_phase_a_pretraining(model, encoder_datasets, device, logger, wandb_logger, run_dir, phase_epochs=None):
    """
    Phase A: Pre-train each encoder with its independent decoder.
    Each encoder learns to encode and decode its domain data independently.
    
    Args:
        model: Multi-encoder model
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to train on
        logger: Logger instance
        wandb_logger: WandB logger
        run_dir: Run directory for saving checkpoints
        phase_epochs: Number of epochs to train each encoder
    
    Returns:
        dict: Training results for Phase A
    """
    logger.info("=" * 80)
    logger.info("PHASE A: ENCODER + INDEPENDENT DECODER PRE-TRAINING")
    logger.info("=" * 80)
    
    # Get training settings
    training_settings = settings.get_training_settings()
    specialist_settings = settings.get_specialist_training_settings()
    data_settings = settings.get_data_settings()
    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    
    BATCH_SIZE = training_settings['batch_size']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    
    # Use settings for phase epochs if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_a']['epochs']
    
    # Phase A setup - all parameters unfrozen but we'll train one encoder+decoder at a time
    setup_phase_training(model, 'pretrain')
    print_parameter_status(model, 'pretrain')
    
    # Use gradient accumulation and mixed precision
    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    
    num_encoders = len(encoder_datasets)
    phase_a_results = {
        'encoder_losses': {i: [] for i in range(num_encoders)},
        'encoder_epochs': phase_epochs,
        'total_encoders': num_encoders
    }
    
    for encoder_idx in range(num_encoders):
        inputs, outputs = encoder_datasets[encoder_idx]
        
        if not inputs or not outputs:
            logger.info(f"Encoder {encoder_idx}: No data available, skipping...")
            continue
            
        logger.info(f"\n--- Training Encoder {encoder_idx} with Independent Decoder ---")
        logger.info(f"Data: {len(inputs)} training samples")
        print(f"Training Encoder {encoder_idx} with independent decoder ({len(inputs)} samples)...")
        
        # Create dataloader for this encoder
        dataloader = prepare_dataloader(inputs, outputs, BATCH_SIZE)
        
        # Freeze all other encoders and decoders
        for other_idx in range(num_encoders):
            if other_idx != encoder_idx:
                # Freeze other encoders
                for param in model.multi_encoder.encoders[other_idx].parameters():
                    param.requires_grad = False
                # Freeze other independent decoders
                for param in model.multi_encoder.independent_decoders[other_idx].parameters():
                    param.requires_grad = False
            else:
                # Unfreeze current encoder and its independent decoder
                for param in model.multi_encoder.encoders[encoder_idx].parameters():
                    param.requires_grad = True
                for param in model.multi_encoder.independent_decoders[encoder_idx].parameters():
                    param.requires_grad = True
        
        # Freeze shared decoder during Phase A
        for param in model.multi_encoder.shared_decoder.parameters():
            param.requires_grad = False
        
        # Create optimizer for current encoder + independent decoder
        trainable_params = []
        trainable_params.extend(model.multi_encoder.encoders[encoder_idx].parameters())
        trainable_params.extend(model.multi_encoder.independent_decoders[encoder_idx].parameters())
        optimizer = Adam(trainable_params, lr=LEARNING_RATE)
        
        # Training loop for this encoder + independent decoder
        encoder_losses = []
        
        for epoch in range(phase_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = len(dataloader)
            
            # Progress bar for this encoder's training
            pbar = tqdm(dataloader, desc=f"Encoder {encoder_idx} Epoch {epoch+1}/{phase_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                    # Train encoder with its independent decoder
                    loss = compute_loss(
                        model, input_seq, target_seq, 
                        beta=BETA, encoder_idx=encoder_idx, use_independent_decoder=True
                    )
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
            
            avg_epoch_loss = epoch_loss / num_batches
            encoder_losses.append(avg_epoch_loss)
            
            logger.info(f"Encoder {encoder_idx} Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
            
            # Compute a unique global step for wandb (ensures continuous timeline across encoders)
            global_step = encoder_idx * phase_epochs + epoch + 1  # 1-indexed
            
            # Log to wandb - every epoch, not just evaluation epochs
            if wandb_logger:
                wandb_logger.log_training_metrics(global_step, {
                    f'phase_a/encoder_{encoder_idx}_loss': avg_epoch_loss,
                    'phase': 'A',
                    'current_encoder': encoder_idx
                })
                
                # Basic visualizations are handled by the reconstruction plots above
            
            # Generate simple reconstruction plot for this encoder every epoch
            if wandb_logger and (epoch + 1) % 1 == 0:  # Every epoch
                try:
                    # Generate reconstruction plot
                    plot_path, wandb_key = generate_specialist_reconstruction_plot(
                        model, dataloader, device, epoch + 1, phase='A',
                        encoder_idx=encoder_idx, use_independent_decoder=True,
                        key=TRAINING_KEYS[encoder_idx % len(TRAINING_KEYS)]
                    )
                    
                    if plot_path and wandb_key:
                        # Log to wandb with proper step using WandbLogger's safe method
                        import wandb
                        import os
                        try:
                            wandb_logger._safe_log({
                                f"phase_a_reconstruction/{wandb_key}": wandb.Image(plot_path)
                            }, step_hint=global_step)
                            logger.info(f"✓ Logged Phase A reconstruction for Encoder {encoder_idx} at step {global_step}")
                        except Exception as wandb_error:
                            logger.warning(f"Failed to log Phase A reconstruction to WandB: {wandb_error}")
                        
                        # Clean up temp file
                        os.unlink(plot_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate reconstruction plot for Encoder {encoder_idx} at epoch {epoch+1}: {e}")
                
                model.train()  # Return to training mode
        
        # Save encoder checkpoint
        encoder_checkpoint_path = save_encoder_checkpoint(model, encoder_idx, run_dir)
        logger.info(f"✓ Encoder {encoder_idx} saved to {encoder_checkpoint_path}")
        
        # Save independent decoder checkpoint
        decoder_checkpoint_path = save_independent_decoder_checkpoint(model, encoder_idx, run_dir)
        logger.info(f"✓ Independent decoder {encoder_idx} saved to {decoder_checkpoint_path}")
        
        phase_a_results['encoder_losses'][encoder_idx] = encoder_losses
        
        # Log final encoder performance
        final_loss = encoder_losses[-1] if encoder_losses else float('inf')
        logger.info(f"Encoder {encoder_idx} final loss: {final_loss:.4f}")
        print(f"✓ Encoder {encoder_idx} + independent decoder training complete (final loss: {final_loss:.4f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE A COMPLETE - All encoders + independent decoders pre-trained")
    logger.info("=" * 60)
    
    # Create Phase A summary visualization: plot all encoder losses together
    if wandb_logger:
        logger.info("Creating Phase A summary visualization...")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Plot all encoder losses on one graph
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
        
        for encoder_idx in range(num_encoders):
            if encoder_idx in phase_a_results['encoder_losses'] and phase_a_results['encoder_losses'][encoder_idx]:
                losses = phase_a_results['encoder_losses'][encoder_idx]
                epochs = range(1, len(losses) + 1)
                encoder_keys = splitting_statistics['keys_per_encoder'][encoder_idx] if 'splitting_statistics' in locals() else [f'key_{encoder_idx}']
                label = f'Encoder {encoder_idx} ({", ".join(encoder_keys)})'
                plt.plot(epochs, losses, 'o-', color=colors[encoder_idx], 
                        linewidth=2, markersize=4, label=label)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Phase A: Individual Encoder Training Losses')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and log to wandb
        phase_a_plot_path = os.path.join(run_dir, 'phase_a_encoder_losses.png')
        plt.savefig(phase_a_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        try:
            import wandb
            try:
                wandb_logger._safe_log({
                    "phase_a/encoder_losses_summary": wandb.Image(phase_a_plot_path),
                    "phase_a/completed": True
                })
                logger.info("✓ Phase A summary plot logged to WandB")
            except Exception as wandb_error:
                logger.warning(f"Failed to upload Phase A summary plot to WandB: {wandb_error}")
        except Exception as e:
            logger.warning(f"Failed to log Phase A summary to WandB: {e}")
        
        logger.info("✓ Phase A summary visualization completed")
    
    return phase_a_results


def train_phase_b_decoder(model, encoder_datasets, device, logger, wandb_logger, run_dir, phase_epochs=None):
    """
    Phase B: Train shared decoder using PoE of all pre-trained encoders.
    
    CRITICAL BEHAVIOR:
    - All encoders are FROZEN (not trainable)
    - All independent decoders are FROZEN (not trainable) 
    - Independent decoders are ONLY used for visualization comparisons
    - Only the shared decoder is trainable and used for actual training
    - Training uses PoE (Product of Experts) from frozen encoders + shared decoder
    
    Args:
        model: Multi-encoder model  
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to train on
        logger: Logger instance
        wandb_logger: WandB logger
        run_dir: Run directory
        phase_epochs: Number of epochs for decoder training
    
    Returns:
        dict: Training results for Phase B
    """
    logger.info("=" * 80)
    logger.info("PHASE B: SHARED DECODER TRAINING WITH POE")
    logger.info("=" * 80)
    
    # Load pre-trained encoders and independent decoders
    logger.info("Loading pre-trained encoders and independent decoders...")
    load_all_encoder_checkpoints(model, run_dir, device)
    
    # Initialize shared decoder with averaged weights from independent decoders
    logger.info("Initializing shared decoder with averaged weights from independent decoders...")
    initialize_shared_decoder_from_independent_decoders(model, run_dir, device)
    
    # Get training settings
    training_settings = settings.get_training_settings()
    specialist_settings = settings.get_specialist_training_settings()
    data_settings = settings.get_data_settings()
    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    
    BATCH_SIZE = training_settings['batch_size']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    
    # Use settings for phase epochs if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_b']['epochs']
    
    # Calculate a starting global step offset based on Phase A length (num_encoders * phase_a_epochs)
    phase_a_epochs_total = specialist_settings['phase_a']['epochs'] * len(model.multi_encoder.encoders)
    base_global_step = phase_a_epochs_total  # Phase B steps will start from here
    
    # Phase B setup - freeze encoders and independent decoders, unfreeze shared decoder
    # Freeze all encoders
    for encoder in model.multi_encoder.encoders:
        for param in encoder.parameters():
            param.requires_grad = False
    
    # Freeze all independent decoders (they won't be used in Phase B)
    for decoder in model.multi_encoder.independent_decoders:
        for param in decoder.parameters():
            param.requires_grad = False
    
    # Unfreeze shared decoder
    for param in model.multi_encoder.shared_decoder.parameters():
        param.requires_grad = True
    
    print_parameter_status(model, 'decoder')
    
    # Create mixed domains dataloader
    num_encoders = len(encoder_datasets)
    mixed_dataloader = create_mixed_domains_dataloader(
        encoder_datasets, num_encoders, BATCH_SIZE, shuffle=True
    )
    
    logger.info(f"Mixed dataloader created with {len(mixed_dataloader)} batches")
    print(f"Training shared decoder with PoE on mixed data ({len(mixed_dataloader)} batches)...")
    
    # Log training data distribution for Phase B
    total_samples = sum(len(inputs) for inputs, outputs in encoder_datasets)
    logger.info(f"Phase B training data summary:")
    logger.info(f"  Total samples across all encoders: {total_samples}")
    for enc_idx, (inputs, outputs) in enumerate(encoder_datasets):
        logger.info(f"  Encoder {enc_idx}: {len(inputs)} samples")
    
    # Create optimizer for shared decoder only
    optimizer = Adam(model.multi_encoder.shared_decoder.parameters(), lr=LEARNING_RATE)
    
    # Use gradient accumulation and mixed precision  
    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    
    phase_b_results = {
        'decoder_losses': [],
        'decoder_epochs': phase_epochs
    }
    
    # Training loop for shared decoder with PoE
    for epoch in range(phase_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(mixed_dataloader)
        
        pbar = tqdm(mixed_dataloader, desc=f"Phase B Epoch {epoch+1}/{phase_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (input_seq, target_seq, encoder_indices) in enumerate(pbar):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            encoder_indices = encoder_indices.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                # Use PoE inference with shared decoder (encoder_idx=None triggers PoE)
                loss = compute_loss(
                    model, input_seq, target_seq,
                    beta=BETA, encoder_idx=None, use_independent_decoder=False
                )
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        phase_b_results['decoder_losses'].append(avg_epoch_loss)
        
        # Compute unique global step for Phase B
        global_step = base_global_step + epoch + 1
        
        logger.info(f"Phase B Epoch {epoch+1} (global step {global_step}): Loss = {avg_epoch_loss:.4f}")
        
        # Log to wandb - every epoch, not just evaluation epochs
        if wandb_logger:
            wandb_logger.log_training_metrics(global_step, {
                'phase_b/shared_decoder_loss': avg_epoch_loss,
                'phase': 'B'
            })
            
            # Basic visualizations are handled by the reconstruction plots below
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == phase_epochs:
            checkpoint_path = save_phase_checkpoint(
                model, optimizer, 'decoder', epoch + 1, avg_epoch_loss, run_dir
            )
            logger.info(f"Phase B checkpoint saved: {checkpoint_path}")
    
        # Generate PoE reconstruction plots for each training key every epoch  
        if wandb_logger and (epoch + 1) % 1 == 0:  # Every epoch
            try:
                # Create dataloaders for each training key to generate reconstruction plots
                for key_idx, training_key in enumerate(TRAINING_KEYS):
                    # Find the encoder dataset that corresponds to this key
                    # Use encoder_datasets[key_idx % len(encoder_datasets)] as a fallback
                    if key_idx < len(encoder_datasets):
                        key_inputs, key_outputs = encoder_datasets[key_idx]
                    else:
                        # Fallback: use the first dataset
                        key_inputs, key_outputs = encoder_datasets[0]
                    
                    if key_inputs and key_outputs:
                        # Create a small dataloader for this key
                        key_dataloader = prepare_dataloader(key_inputs, key_outputs, batch_size=1)
                        
                        # Generate PoE reconstruction plot for this key
                        plot_path, wandb_key = generate_specialist_reconstruction_plot(
                            model, key_dataloader, device, epoch + 1, phase='B',
                            encoder_idx=None, use_independent_decoder=False,  # PoE + shared decoder
                            key=training_key
                        )
                        
                        if plot_path and wandb_key:
                            # Log to wandb with proper step using WandbLogger's safe method
                            import wandb
                            import os
                            try:
                                wandb_logger._safe_log({
                                    f"phase_b_reconstruction/{wandb_key}": wandb.Image(plot_path)
                                }, step_hint=global_step)
                                logger.info(f"✓ Logged Phase B PoE reconstruction for key {training_key} at step {global_step}")
                            except Exception as wandb_error:
                                logger.warning(f"Failed to log Phase B PoE reconstruction to WandB: {wandb_error}")
                            
                            # Clean up temp file
                            os.unlink(plot_path)
                            
            except Exception as e:
                logger.warning(f"Failed to generate Phase B reconstruction plots at epoch {epoch+1}: {e}")
            
            # -----------------------------------------------------------------
            # Extra visualisations for Phase B: show what each encoder's latent
            # produces when decoded by (a) its own independent decoder and (b)
            # the shared decoder. Also log PoE latent stats. This gives deeper
            # insight into encoder/decoder alignment during Phase B training.
            # -----------------------------------------------------------------
            if wandb_logger:
                try:
                    import wandb, os

                    for enc_idx, (enc_inputs, enc_outputs) in enumerate(encoder_datasets):
                        if not enc_inputs or not enc_outputs:
                            continue

                        # Build a lightweight dataloader (single sample)
                        enc_loader = prepare_dataloader(enc_inputs, enc_outputs, batch_size=1)

                        # === Encoder latent + Independent Decoder ===
                        plot_path, wandb_key, mu_mean, sigma_mean = generate_specialist_reconstruction_plot(
                            model, enc_loader, device, epoch + 1, phase='B',
                            encoder_idx=enc_idx, use_independent_decoder=True,
                            key=TRAINING_KEYS[enc_idx % len(TRAINING_KEYS)],
                            return_mu_sigma=True
    )
                        if plot_path and wandb_key:
                            try:
                                wandb_logger._safe_log({
                                    f"phase_b_reconstruction/enc{enc_idx}_independent/{wandb_key}": wandb.Image(plot_path)
                                }, step_hint=global_step)
                                logger.debug(f"✓ Logged Phase B encoder {enc_idx} independent reconstruction at step {global_step}")
                            except Exception as wandb_error:
                                logger.warning(f"Failed to log Phase B encoder {enc_idx} independent reconstruction: {wandb_error}")
                            os.unlink(plot_path)
    
                        # === Encoder latent + Shared Decoder ===
                        plot_path, wandb_key, mu_mean, sigma_mean = generate_specialist_reconstruction_plot(
                            model, enc_loader, device, epoch + 1, phase='B',
                            encoder_idx=enc_idx, use_independent_decoder=False,
                            key=TRAINING_KEYS[enc_idx % len(TRAINING_KEYS)],
                            return_mu_sigma=True
                        )
                        if plot_path and wandb_key:
                            try:
                                wandb_logger._safe_log({
                                    f"phase_b_reconstruction/enc{enc_idx}_shared/{wandb_key}": wandb.Image(plot_path)
                                }, step_hint=global_step)
                                logger.debug(f"✓ Logged Phase B encoder {enc_idx} shared reconstruction at step {global_step}")
                            except Exception as wandb_error:
                                logger.warning(f"Failed to log Phase B encoder {enc_idx} shared reconstruction: {wandb_error}")
                            os.unlink(plot_path)

                    # === PoE latent + Shared Decoder (overall) ===
                    if encoder_datasets:
                        poe_inputs, poe_outputs = encoder_datasets[0]
                        poe_loader = prepare_dataloader(poe_inputs, poe_outputs, batch_size=1)
                        plot_path, wandb_key, mu_mean, sigma_mean = generate_specialist_reconstruction_plot(
                            model, poe_loader, device, epoch + 1, phase='B',
                            encoder_idx=None, use_independent_decoder=False,
                            key='poe', return_mu_sigma=True
                        )
                        if plot_path and wandb_key:
                            try:
                                wandb_logger._safe_log({
                                    f"phase_b_reconstruction/{wandb_key}": wandb.Image(plot_path)
                                }, step_hint=global_step)
                                logger.debug(f"✓ Logged Phase B PoE overall reconstruction at step {global_step}")
                            except Exception as wandb_error:
                                logger.warning(f"Failed to log Phase B PoE overall reconstruction: {wandb_error}")
                            os.unlink(plot_path)

                except Exception as e:
                    logger.warning(f"Failed to generate extended Phase B reconstructions at epoch {epoch+1}: {e}")
            
            model.train()  # Return to training mode
    
    # Save final shared decoder checkpoint
    decoder_checkpoint_path = save_decoder_checkpoint(model, run_dir)
    logger.info(f"✓ Shared decoder saved to {decoder_checkpoint_path}")
    
    final_loss = phase_b_results['decoder_losses'][-1] if phase_b_results['decoder_losses'] else float('inf')
    logger.info(f"\n" + "=" * 60)
    logger.info(f"PHASE B COMPLETE - Shared decoder training with PoE finished (final loss: {final_loss:.4f})")
    logger.info("=" * 60)
    
    # Generate final comparison plot showing all encoder reconstructions vs PoE
    if wandb_logger:
        logger.info("Generating final Phase B encoder vs PoE comparison plot...")
        final_global_step = base_global_step + phase_epochs
        generate_phase_b_final_comparison_plot(
            model, encoder_datasets, device, run_dir, wandb_logger, final_global_step
        )
        logger.info("✓ Phase B final comparison plot generated")
        
    # Create Phase B summary visualization and final comprehensive evaluation
    if wandb_logger:
        logger.info("Creating Phase B final evaluation and visualizations...")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Plot Phase B training losses
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(phase_b_results['decoder_losses']) + 1)
        plt.plot(epochs, phase_b_results['decoder_losses'], 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Phase B: Shared Decoder Training Loss (PoE)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and log to wandb
        phase_b_plot_path = os.path.join(run_dir, 'phase_b_decoder_loss.png')
        plt.savefig(phase_b_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        try:
            import wandb
            try:
                wandb_logger._safe_log({
                    "phase_b/decoder_loss_summary": wandb.Image(phase_b_plot_path),
                    "phase_b/completed": True
                })
                logger.info("✓ Phase B summary plot logged to WandB")
            except Exception as wandb_error:
                logger.warning(f"Failed to upload Phase B summary plot to WandB: {wandb_error}")
        except Exception as e:
            logger.warning(f"Failed to log Phase B summary to WandB: {e}")
        
        logger.info("✓ Phase B visualization completed")
    
    return phase_b_results


def run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, phase_name):
    """Run evaluation between training phases."""
    logger.info(f"\n--- Evaluation after {phase_name} ---")
    print(f"Running evaluation after {phase_name}...")
    
    try:
        eval_results = run_quick_evaluation(model, run_dir, epoch=f"{phase_name}_final")
        if eval_results and wandb_logger:
            log_evaluation_to_wandb(eval_results, run_dir, f"{phase_name}_final", wandb_logger)
            logger.info(f"✓ Evaluation results logged to wandb for {phase_name}")
        return eval_results
    except Exception as e:
        logger.warning(f"Evaluation failed after {phase_name}: {e}")
        return None


def main_specialist_training(file_store_name, phases_to_run=None, resume_from_phase=None):
    """
    Main specialist training function implementing 2-phase training.
    
    Args:
        file_store_name: Name for run directory
        phases_to_run: List of phases to run ('A', 'B') - uses settings default if None
        resume_from_phase: Phase to resume from (if any)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get current settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    repulsion_loss_settings = settings.get_repulsion_loss_settings()
    wandb_settings = settings.get_wandb_settings()
    specialist_settings = settings.get_specialist_training_settings()
    
    # Use specialist settings for defaults
    if phases_to_run is None:
        phases_to_run = specialist_settings['phases_to_run']
    
    evaluation_between_phases = specialist_settings.get('evaluation_between_phases', True)
    
    # Validate multi-encoder configuration
    NUM_ENCODERS = model_architecture.get('num_encoders', 1)
    if NUM_ENCODERS <= 1:
        raise ValueError("Specialist training requires num_encoders > 1. Please update model_architecture settings.")
    
    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    if not TRAINING_KEYS or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings.")
    
    N_EXAMPLES_PER_TASK = data_settings['n']
    
    print(f"Specialist training configuration:")
    print(f"- Number of encoders: {NUM_ENCODERS}")
    print(f"- Training keys: {TRAINING_KEYS}")
    print(f"- Examples per task: {N_EXAMPLES_PER_TASK}")
    print(f"- Phases to run: {phases_to_run}")
    print(f"- Evaluation between phases: {evaluation_between_phases}")
    print(f"- Phase A epochs: {specialist_settings['phase_a']['epochs']}")
    print(f"- Phase B epochs: {specialist_settings['phase_b']['epochs']}")
    
    # Validate phases for new approach
    valid_phases = ['A', 'B']
    if any(phase not in valid_phases for phase in phases_to_run):
        invalid_phases = [p for p in phases_to_run if p not in valid_phases]
        print(f"Warning: Invalid phases {invalid_phases} removed. Valid phases are: {valid_phases}")
        phases_to_run = [p for p in phases_to_run if p in valid_phases]
        print(f"Updated phases to run: {phases_to_run}")
    
    set_seed(data_settings['training_seed'])
    
    # Create run directory and setup logging
    run_dir = create_run_directory(file_store_name)
    logger = setup_logging(run_dir)
    logger.info(f"Starting specialist training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)
    
    # Initialize wandb
    wandb_logger = None
    if wandb_settings.get('enabled', False):
        wandb_logger = init_wandb_for_mode('specialist_train', run_dir)
        if wandb_logger:
            logger.info(f"✓ Wandb logging enabled: {wandb_logger.run.name}")
            # Ensure trajectory plots are enabled for specialist training
            if not wandb_settings.get('log_trajectory_plots', False):
                wandb_settings['log_trajectory_plots'] = True  # Enable by default for specialist mode
        else:
            logger.info("⚠ Wandb initialization failed, continuing without wandb")
    
    # Generate and split data for multi-encoder training
    logger.info("Generating and splitting data for specialist training...")
    print("Generating and splitting data...")
    
    dataset_splits, key_to_encoder_mapping, splitting_statistics = split_dataset_by_keys_for_multi_encoder(
        TRAINING_KEYS, NUM_ENCODERS, N_EXAMPLES_PER_TASK, generate_and_process_tasks
    )
    
    # Log data splitting info
    logger.info(f"Data splitting complete:")
    for encoder_idx, (inputs, outputs) in enumerate(dataset_splits):
        encoder_keys = splitting_statistics['keys_per_encoder'][encoder_idx]
        logger.info(f"  Encoder {encoder_idx}: {len(inputs)} samples from keys {encoder_keys}")
    
    # Initialize model first so param_info is available
    logger.info("Initializing multi-encoder model...")
    model = build_model(device)
    param_info = count_model_parameters(model)
    logger.info(f"Model initialized with {param_info['total_params']:,} parameters")
    
    # Collect **all** training sequences (needed for latent visualisation later)
    all_inputs, all_outputs = [], []
    for enc_inputs, enc_outputs in dataset_splits:
        all_inputs.extend(enc_inputs)
        all_outputs.extend(enc_outputs)

    results = {
        'specialist_training': True,
        'phases_completed': [],
        'training_metadata': {
            'key_to_encoder_mapping': key_to_encoder_mapping,
            'splitting_statistics': splitting_statistics,
            'training_keys': TRAINING_KEYS,
            'num_encoders': NUM_ENCODERS,
            'phases_planned': phases_to_run
        },
        'model_parameter_info': param_info,
        # Flatten sequences to plain python lists for pickle safety
        'input_sequences': [seq.tolist() if hasattr(seq, 'tolist') else seq for seq in all_inputs],
        'output_sequences': [seq.tolist() if hasattr(seq, 'tolist') else seq for seq in all_outputs]
    }
    
    # Run phases
    phase_a_epochs = specialist_settings['phase_a']['epochs']
    phase_b_epochs = specialist_settings['phase_b']['epochs']
    try:
        if 'A' in phases_to_run:
            logger.info("\n" + "=" * 100)
            logger.info("STARTING PHASE A: ENCODER + INDEPENDENT DECODER PRE-TRAINING")
            logger.info("=" * 100)
            
            phase_a_results = train_phase_a_pretraining(
                model, dataset_splits, device, logger, wandb_logger, run_dir,phase_a_epochs
            )
            results['phase_a'] = phase_a_results
            results['phases_completed'].append('A')
            
            # Evaluation after Phase A (if enabled)
            if evaluation_between_phases:
                eval_results_a = run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, "Phase A")
                if eval_results_a:
                    results['phase_a']['evaluation'] = eval_results_a
            
            # Add PoE accuracy snapshot to results
            poe_accuracy = run_quick_evaluation(model, run_dir, epoch=f"Phase A Epoch {phase_a_epochs}")
            results['phase_a']['poe_accuracies'] = [poe_accuracy]
            # Save results after each epoch
            save_results(results, run_dir)
        
        if 'B' in phases_to_run:
            logger.info("\n" + "=" * 100)
            logger.info("STARTING PHASE B: SHARED DECODER TRAINING WITH POE")
            logger.info("=" * 100)
            
            phase_b_results = train_phase_b_decoder(
                model, dataset_splits, device, logger, wandb_logger, run_dir,phase_b_epochs
            )
            results['phase_b'] = phase_b_results
            results['phases_completed'].append('B')
            
            # Evaluation after Phase B (if enabled)
            if evaluation_between_phases:
                eval_results_b = run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, "Phase B")
                if eval_results_b:
                    results['phase_b']['evaluation'] = eval_results_b
            
            # Add PoE accuracy snapshot to results
            poe_accuracy = run_quick_evaluation(model, run_dir, epoch=f"Phase B Epoch {phase_b_epochs}")
            results['phase_b']['poe_accuracies'] = [poe_accuracy]
            
            # Save final complete model after Phase B
            final_loss = phase_b_results['decoder_losses'][-1] if phase_b_results['decoder_losses'] else float('inf')
            final_model_path = save_full_model_checkpoint(model, None, phase_b_epochs, final_loss, run_dir)
            logger.info(f"✓ Final specialist model saved to {final_model_path}")
            
            # Save final model as a WandB artifact
            if wandb_logger:
                import wandb
                artifact = wandb.Artifact('final_specialist_model', type='model')
                artifact.add_file(final_model_path)
                wandb.log_artifact(artifact)
                print("✓ Final specialist model uploaded to WandB as an artifact")
            
            # Save results after each epoch
            save_results(results, run_dir)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Save final results
        logger.info("Saving final specialist training results...")
        save_results(results, run_dir)
        
        # Save model parameters
        save_model_params(run_dir, param_info)
        
        if wandb_logger:
            wandb_logger.finish()
        
        logger.info("=" * 80)
        logger.info("SPECIALIST TRAINING COMPLETE")
        logger.info(f"Phases completed: {results['phases_completed']}")
        logger.info("Final model uses PoE of specialized encoders with shared decoder")
        logger.info(f"Results saved in: {run_dir}")
        logger.info("=" * 80)
        
        print("\nSpecialist training complete!")
        print(f"Phases completed: {results['phases_completed']}")
        print("Final model ready: PoE of specialized encoders with shared decoder")
        print(f"Results saved in: {run_dir}")
    
    return results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist Multi-Encoder Training")
    parser.add_argument("--phases", type=str, default="A,B",
                       help="Comma-separated phases to run (A,B)")
    parser.add_argument("--resume_from_phase", type=str, default=None,
                       help="Phase to resume from (A,B)")
    parser.add_argument("--run_name", type=str, default="specialist_training",
                       help="Name for the run directory")
    
    args = parser.parse_args()
    
    # Parse phases
    phases_to_run = [p.strip().upper() for p in args.phases.split(',')]
    valid_phases = ['A', 'B']
    
    if not all(p in valid_phases for p in phases_to_run):
        print(f"Error: Invalid phases. Valid phases are: {valid_phases}")
        exit(1)
    
    print(f"Starting specialist training with phases: {phases_to_run}")
    
    try:
        results, model = main_specialist_training(
            args.run_name, 
            phases_to_run=phases_to_run,
            resume_from_phase=args.resume_from_phase
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 