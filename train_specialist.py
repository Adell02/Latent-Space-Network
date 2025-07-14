#!/usr/bin/env python3
"""
Specialist Training Script for Multi-Encoder Models

Implements 2-phase training approach:
- Phase A: Train each encoder with its independent decoder on domain-specific data
- Phase B: Train shared decoder using PoE of all pre-trained encoders

ENHANCED MULTI-KEY EVALUATION SYSTEM:
- Evaluates ALL keys specified in evaluation_settings['eval_keys'] independently
- Each key generates its own reconstructions and trajectory visualizations  
- Metrics (grid accuracy, pixel accuracy, etc.) are aggregated across ALL keys
- Provides comprehensive evaluation while maintaining efficiency
- Supports different eval_keys from training_keys for validation

ENHANCED TRAINING MECHANISMS:
This implementation includes advanced mechanisms to guarantee latent activity 
and specialization based on latest research:

1. FREE-BITS MECHANISM:
   - Guarantees minimum KL divergence per dimension: kl_loss = clamp(kl_loss, min=δ)
   - Prevents posterior collapse by enforcing δ ≈ 0.05-0.1 × latent_dim
   - Keeps latents active even during early training phases
   - Debug signal: KL/dim should stabilize ≥ 0.2 after warm-up

2. CYCLICAL β-ANNEALING:
   - Ramps β from 0 → β_max over K epochs, then resets (repeats)
   - Typical params: β_max ≈ 1e-3, K = 3-5 epochs
   - Prevents KL vanishing while allowing periodic high-capacity phases
   - Balances reconstruction quality with latent utilization

3. DYNAMIC λ SCHEDULING:
   - Scales anti-batch penalty with current β value
   - High-β phases: λ = 2-5 × β_max (strong specialization pressure)
   - Low-β phases: λ = 0.01 (relaxed specialization)
   - Maintains KL gap between in-slice and anti-batch samples

4. CONTRASTIVE KL MARGIN:
   - Enforces gap between in-slice and anti-batch KL losses
   - gap = kl_in.detach() - kl_anti (should be >0 for specialization)
   - anti_loss = ReLU(τ - gap) where τ ≈ 1-2 nats
   - Only penalizes when encoder fails to maintain specialization gap

5. DEBUG METRICS:
   - KL per dimension tracking for each encoder
   - Anti-batch KL monitoring
   - Specialization gap (kl_in - kl_anti) tracking
   - Effective β and λ values logged to WandB

These mechanisms work together to:
- Guarantee minimum latent activity (free-bits)
- Prevent posterior collapse (cyclical β)
- Enforce specialization (dynamic λ + contrastive margin)
- Enable reliable uncertainty estimation for routing/OOD detection

Configuration:
Set enhanced_training parameters in model_specialist_settings.json:
```json
"enhanced_training": {
    "use_cyclical_beta": true,
    "beta_cycle_length": 4,
    "use_free_bits": true,
    "free_bits_delta": 0.07,
    "use_dynamic_lambda": true,
    "lambda_high_multiplier": 3.0,
    "use_contrastive_margin": true,
    "margin_tau": 1.5,
    "debug_kl_metrics": true
}
```

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
from utils.visualizers import generate_per_dimension_kl_plot


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
                mu, log_var,_ = model.multi_encoder.encoders[encoder_idx](input_sample, target_sample)
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
            axes[0].set_title(f'Input (Train)\n{input_shape[0]}×{input_shape[1]}')
            axes[0].axis('off')
            
            # Target
            axes[1].imshow(target_grid, cmap='viridis', interpolation='nearest')
            axes[1].set_title(f'Target (Train)\n{target_shape[0]}×{target_shape[1]}')
            axes[1].axis('off')
            
            # Reconstruction
            axes[2].imshow(recon_grid, cmap='viridis', interpolation='nearest')
            axes[2].set_title(f'Reconstruction (Train)\n{recon_shape[0]}×{recon_shape[1]}')
            axes[2].axis('off')
            
            # Set overall title with clear data source indication
            if phase == 'A':
                fig.suptitle(f'Phase A Epoch {epoch}: Encoder {encoder_idx} + Independent Decoder\nTRAINING DATA (Key: {key})', fontsize=14)
                wandb_key = f'encoder_{encoder_idx}_{key}'
            else:
                fig.suptitle(f'Phase B Epoch {epoch}: PoE + Shared Decoder\nTRAINING DATA (Key: {key})', fontsize=14)
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


def generate_latent_histograms(model, dataloader, device, encoder_idx, epoch, wandb_logger, global_step):
    """
    Generate histograms of latent means and log(sigmas) for a specific encoder.
    
    Args:
        model: Current model state
        dataloader: Data loader with samples
        device: Device to run on
        encoder_idx: Encoder index
        epoch: Current epoch
        wandb_logger: WandB logger instance
        global_step: Global training step
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile
    import os
    
    model.eval()
    all_mus = []
    all_log_vars = []
    
    with torch.no_grad():
        # Collect latent statistics from a subset of data (efficient)
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            if batch_idx >= 20:  # Limit to 20 batches for efficiency
                break
                
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Get latent distributions from specific encoder
            mu, log_var,_ = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
            all_mus.append(mu.cpu().numpy())
            all_log_vars.append(log_var.cpu().numpy())
    
    if not all_mus:
        return
    
    # Concatenate all samples
    all_mus = np.concatenate(all_mus, axis=0).flatten()
    all_log_vars = np.concatenate(all_log_vars, axis=0).flatten()
    
    # Create histogram plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of means
    ax1.hist(all_mus, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='Prior μ=0')
    ax1.set_xlabel('Latent Mean (μ)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Encoder {encoder_idx} - Latent Means\nEpoch {epoch}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mu_mean, mu_std = np.mean(all_mus), np.std(all_mus)
    ax1.text(0.05, 0.95, f'μ={mu_mean:.3f}, σ={mu_std:.3f}', 
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Histogram of log variances
    ax2.hist(all_log_vars, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='Prior log σ²=0')
    ax2.set_xlabel('Log Variance (log σ²)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Encoder {encoder_idx} - Log Variances\nEpoch {epoch}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    logvar_mean, logvar_std = np.mean(all_log_vars), np.std(all_log_vars)
    ax2.text(0.05, 0.95, f'μ={logvar_mean:.3f}, σ={logvar_std:.3f}', 
             transform=ax2.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.suptitle(f'Phase A: Encoder {encoder_idx} Latent Distributions\nTraining Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    
    # Save and log to wandb
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
        temp_plot_path = tmp_file.name
    plt.close()
    
    if wandb_logger:
        try:
            import wandb
            wandb_logger._safe_log({
                f"phase_a_latents/encoder_{encoder_idx}_histograms": wandb.Image(temp_plot_path),
                f"phase_a_latents/encoder_{encoder_idx}_mu_mean": mu_mean,
                f"phase_a_latents/encoder_{encoder_idx}_mu_std": mu_std,
                f"phase_a_latents/encoder_{encoder_idx}_logvar_mean": logvar_mean,
                f"phase_a_latents/encoder_{encoder_idx}_logvar_std": logvar_std,
            }, step_hint=global_step)
            print(f"✓ Logged latent histograms for Encoder {encoder_idx} at epoch {epoch}")
        except Exception as e:
            print(f"⚠ Failed to log latent histograms: {e}")
        finally:
            os.unlink(temp_plot_path)
    
    model.train()


def generate_phase_b_final_comparison_plot(model, encoder_datasets, device, run_dir, wandb_logger, global_step):
    """
    Generate comprehensive encoder vs PoE comparison plots and accuracy matrix.
    Creates separate figures for training and evaluation data, plus accuracy matrix.
    
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
    from utils.model_utils import prepare_dataloader
    from re_arc.main import generate_and_process_tasks
    from utils.evaluation_utils import run_quick_evaluation
    import tempfile
    import os
    
    model.eval()
    with torch.no_grad():
        num_encoders = len(encoder_datasets)
        
        # Get evaluation settings to use ALL eval_keys
        eval_settings = settings.get_evaluation_settings()
        eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
        data_settings = settings.get_data_settings()
        eval_seed = data_settings.get('eval_seed', 42)
        n_samples_per_key = eval_settings.get('eval_n_samples', 2)
        
        print(f"Creating Phase B final plots for keys: {eval_keys}")
        print(f"Using {n_samples_per_key} samples per key")
        
        # 1. COMPREHENSIVE ANALYSIS WITH TRAINING DATA
        print("\n=== Creating comprehensive analysis with TRAINING DATA ===")
        _create_comprehensive_analysis_plot(model, encoder_datasets, device, "training", 
                                           n_samples_per_key, wandb_logger, global_step, "training")
        
        # 2. COMPREHENSIVE ANALYSIS WITH EVALUATION DATA  
        print("\n=== Creating comprehensive analysis with EVALUATION DATA ===")
        from utils.model_utils import set_seed
        set_seed(eval_seed)
        
        eval_samples = []
        for eval_key in eval_keys:
            try:
                print(f"Generating evaluation data for key '{eval_key}'...")
                _, _, _, input_sequences, output_sequences = generate_and_process_tasks(eval_key, n_samples_per_key)
                
                if not input_sequences or not output_sequences:
                    print(f"⚠ Warning: No evaluation data for key '{eval_key}', skipping...")
                    continue
                
                eval_dataloader = prepare_dataloader(input_sequences, output_sequences, batch_size=1)
                key_samples = []
                for i, (input_seq, target_seq) in enumerate(eval_dataloader):
                    if i >= n_samples_per_key:
                        break
                    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                    key_samples.append((input_seq, target_seq, eval_key, i))
                
                eval_samples.extend(key_samples)
                print(f"✓ Loaded {len(key_samples)} evaluation samples from key '{eval_key}'")
                
            except Exception as e:
                print(f"⚠ Warning: Failed to generate evaluation data for key '{eval_key}': {e}")
                continue
        
        if eval_samples:
            _create_comprehensive_analysis_plot(model, eval_samples, device, "evaluation", 
                                               n_samples_per_key, wandb_logger, global_step, "evaluation")
        else:
            print("⚠ Error: No evaluation data could be generated - skipping evaluation analysis")
        
        # 3. ACCURACY MATRIX (KEY vs ENCODER/POE)
        print("\n=== Creating accuracy matrix ===")
        _create_accuracy_matrix(model, eval_keys, num_encoders, device, wandb_logger, global_step)


def _create_comprehensive_analysis_plot(model, data_source, device, data_type, n_samples_per_key, wandb_logger, global_step, source_name):
    """Create comprehensive encoder vs PoE analysis plot (without bar graph)."""
    import matplotlib.pyplot as plt
    import tempfile
    import os
    from utils.data_preparation import extract_grid_from_sequence
    
    if data_type == "training":
        # Use training data from encoder datasets
        num_encoders = len(data_source)
        all_samples = []
        for enc_idx, (inputs, outputs) in enumerate(data_source):
            if inputs and outputs:
                from utils.model_utils import prepare_dataloader
                dataloader = prepare_dataloader(inputs, outputs, batch_size=1)
                for i, (input_seq, target_seq) in enumerate(dataloader):
                    if i >= n_samples_per_key:
                        break
                    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                    all_samples.append((input_seq, target_seq, f"training_enc_{enc_idx}", i))
    else:
        # Use evaluation data 
        all_samples = data_source
        num_encoders = len(model.multi_encoder.encoders)
    
    if not all_samples:
        print(f"⚠ No {data_type} data available for analysis")
        return
    
    print(f"Analyzing {len(all_samples)} {data_type} samples...")
    
    # Process samples for visualization
    sample_data = []
    for sample_idx, (input_seq, target_seq, source_key, sample_num) in enumerate(all_samples):
        if sample_idx >= min(len(all_samples), n_samples_per_key * 4):  # Limit for visualization
            break
            
        # Extract grids
        input_grid, input_shape = extract_grid_from_sequence(input_seq[0].cpu().numpy())
        target_grid, target_shape = extract_grid_from_sequence(target_seq[0].cpu().numpy())
        
        # Get encoder reconstructions
        encoder_reconstructions = []
        for enc_idx in range(num_encoders):
            mu, log_var,_ = model.multi_encoder.encoders[enc_idx](input_seq, target_seq)
            z = model.reparameterize(mu, log_var)
            shape_logits, grid_logits = model.multi_encoder.independent_decoders[enc_idx](
                z, input_seq, target_seq=target_seq
            )
            
            shape_pred = shape_logits[0].argmax(dim=-1).cpu().numpy()
            grid_pred = grid_logits[0].argmax(dim=-1).cpu().numpy()
            recon_seq = target_seq[0].cpu().numpy().copy()
            recon_seq[900:902] = shape_pred
            if len(shape_pred) >= 2 and shape_pred[0] > 0 and shape_pred[1] > 0:
                recon_seq[:min(len(grid_pred), 900)] = grid_pred[:min(len(grid_pred), 900)]
            recon_grid, recon_shape = extract_grid_from_sequence(recon_seq)
            encoder_reconstructions.append((recon_grid, recon_shape))
        
        # Get PoE reconstruction
        mu_poe, log_var_poe = model(input_seq, target_seq)[1:3]
        z_poe = model.reparameterize(mu_poe, log_var_poe)
        shape_logits_poe, grid_logits_poe = model.multi_encoder.shared_decoder(z_poe, input_seq, target_seq=target_seq)
        
        shape_pred_poe = shape_logits_poe[0].argmax(dim=-1).cpu().numpy()
        grid_pred_poe = grid_logits_poe[0].argmax(dim=-1).cpu().numpy()
        recon_seq_poe = target_seq[0].cpu().numpy().copy()
        recon_seq_poe[900:902] = shape_pred_poe
        if len(shape_pred_poe) >= 2 and shape_pred_poe[0] > 0 and shape_pred_poe[1] > 0:
            recon_seq_poe[:min(len(grid_pred_poe), 900)] = grid_pred_poe[:min(len(grid_pred_poe), 900)]
        poe_recon_grid, poe_recon_shape = extract_grid_from_sequence(recon_seq_poe)
        
        sample_data.append({
            'input_grid': input_grid,
            'input_shape': input_shape,
            'target_grid': target_grid,
            'target_shape': target_shape,
            'encoder_reconstructions': encoder_reconstructions,
            'poe_reconstruction': (poe_recon_grid, poe_recon_shape),
            'source_key': source_key,
            'sample_num': sample_num
        })
    
    # Create visualization (without bar graph)
    fig = plt.figure(figsize=(4 * (num_encoders + 3), 8))
    gs = fig.add_gridspec(2, num_encoders + 3, height_ratios=[1.5, 1.5], hspace=0.3, wspace=0.3)
    
    # Show first few samples
    for row in range(min(2, len(sample_data))):
        sample = sample_data[row]
        
        # Input and Target
        ax_input = fig.add_subplot(gs[row, 0])
        ax_input.imshow(sample['input_grid'], cmap='viridis', interpolation='nearest')
        ax_input.set_title(f'Input {row+1}\n{sample["input_shape"][0]}×{sample["input_shape"][1]}\nKey: {sample["source_key"].split("_")[-1] if "_" in sample["source_key"] else sample["source_key"]}', fontsize=10)
        ax_input.axis('off')
        
        ax_target = fig.add_subplot(gs[row, 1])
        ax_target.imshow(sample['target_grid'], cmap='viridis', interpolation='nearest')
        ax_target.set_title(f'Target {row+1}\n{sample["target_shape"][0]}×{sample["target_shape"][1]}', fontsize=10)
        ax_target.axis('off')
        
        # Encoder reconstructions
        for enc_idx in range(num_encoders):
            ax_enc = fig.add_subplot(gs[row, 2 + enc_idx])
            recon_grid, recon_shape = sample['encoder_reconstructions'][enc_idx]
            ax_enc.imshow(recon_grid, cmap='viridis', interpolation='nearest')
            ax_enc.set_title(f'Encoder {enc_idx}\n{recon_shape[0]}×{recon_shape[1]}', fontsize=10)
            ax_enc.axis('off')
        
        # PoE reconstruction
        ax_poe = fig.add_subplot(gs[row, 2 + num_encoders])
        poe_recon_grid, poe_recon_shape = sample['poe_reconstruction']
        ax_poe.imshow(poe_recon_grid, cmap='viridis', interpolation='nearest')
        ax_poe.set_title(f'PoE\n{poe_recon_shape[0]}×{poe_recon_shape[1]}', fontsize=10, fontweight='bold')
        ax_poe.axis('off')
    
    plt.suptitle(f'Comprehensive Encoder vs PoE Analysis\n{data_type.upper()} DATA', fontsize=16)
    plt.tight_layout()
    
    # Save and log
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=200, bbox_inches='tight')
        temp_plot_path = tmp_file.name
    plt.close()
    
    if wandb_logger:
        try:
            import wandb
            wandb_logger._safe_log({
                f"phase_b_final/comprehensive_analysis_{source_name}": wandb.Image(temp_plot_path),
                f"phase_b_final/{source_name}_samples_analyzed": len(sample_data)
            }, step_hint=global_step)
            print(f"✓ Logged comprehensive analysis for {data_type} data")
        except Exception as e:
            print(f"⚠ Failed to log {data_type} analysis: {e}")
        finally:
            os.unlink(temp_plot_path)


def _create_accuracy_matrix(model, eval_keys, num_encoders, device, wandb_logger, global_step):
    """Create colored accuracy matrix: Keys (x-axis) vs Encoders+PoE (y-axis)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.evaluation_utils import run_quick_evaluation
    import tempfile
    import os
    
    print("Computing accuracy matrix...")
    
    # Initialize accuracy matrix
    # Rows: Encoders (0 to num_encoders-1) + PoE (last row)
    # Cols: eval_keys
    accuracy_matrix = np.zeros((num_encoders + 1, len(eval_keys)))
    
    for key_idx, eval_key in enumerate(eval_keys):
        print(f"  Evaluating key '{eval_key}'...")
        
        # Evaluate each individual encoder
        for enc_idx in range(num_encoders):
            try:
                eval_results = run_quick_evaluation(
                    model, "", epoch=f"matrix_eval", eval_keys=[eval_key],
                    encoder_idx=enc_idx, use_independent_decoder=True
                )
                if eval_results and eval_key in eval_results:
                    accuracy = eval_results[eval_key]['metrics'].get('grid_accuracy', 0.0)
                    accuracy_matrix[enc_idx, key_idx] = accuracy
                    print(f"    Encoder {enc_idx}: {accuracy:.3f}")
            except Exception as e:
                print(f"    Encoder {enc_idx}: Failed ({e})")
                accuracy_matrix[enc_idx, key_idx] = 0.0
        
        # Evaluate PoE
        try:
            eval_results = run_quick_evaluation(
                model, "", epoch=f"matrix_eval", eval_keys=[eval_key],
                encoder_idx=None, use_independent_decoder=False  # PoE + shared decoder
            )
            if eval_results and eval_key in eval_results:
                accuracy = eval_results[eval_key]['metrics'].get('grid_accuracy', 0.0)
                accuracy_matrix[num_encoders, key_idx] = accuracy
                print(f"    PoE: {accuracy:.3f}")
        except Exception as e:
            print(f"    PoE: Failed ({e})")
            accuracy_matrix[num_encoders, key_idx] = 0.0
    
    # Create matrix visualization
    fig, ax = plt.subplots(figsize=(max(8, len(eval_keys) * 1.2), max(6, (num_encoders + 1) * 0.8)))
    
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(eval_keys)))
    ax.set_xticklabels([key[:8] + "..." if len(key) > 8 else key for key in eval_keys], rotation=45, ha='right')
    ax.set_yticks(range(num_encoders + 1))
    ax.set_yticklabels([f'Encoder {i}' for i in range(num_encoders)] + ['PoE'])
    
    # Add text annotations
    for i in range(num_encoders + 1):
        for j in range(len(eval_keys)):
            text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Evaluation Keys')
    ax.set_ylabel('Encoders + PoE')
    ax.set_title('Accuracy Matrix: Keys vs Encoders/PoE')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Grid Accuracy', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save and log
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=200, bbox_inches='tight')
        temp_plot_path = tmp_file.name
    plt.close()
    
    if wandb_logger:
        try:
            import wandb
            wandb_logger._safe_log({
                "phase_b_final/accuracy_matrix": wandb.Image(temp_plot_path),
                "phase_b_final/matrix_shape": f"{num_encoders + 1}x{len(eval_keys)}",
                "phase_b_final/eval_keys": eval_keys
            }, step_hint=global_step)
            print("✓ Logged accuracy matrix")
        except Exception as e:
            print(f"⚠ Failed to log accuracy matrix: {e}")
        finally:
            os.unlink(temp_plot_path)


def build_model(device, wandb_logger=None, global_step=None):
    """Build and return LatentProgramNetwork with architecture visualization."""
    from utils.model_architecture_viz import generate_architecture_visualizations, log_model_summary
    
    model = LatentProgramNetwork().to(device)
    
    # Generate architecture visualizations and upload to wandb
    if wandb_logger:
        print("🏗️ Generating model architecture visualizations...")
        generate_architecture_visualizations(model, wandb_logger, device, global_step)
        log_model_summary(model, wandb_logger, global_step)
    
    return model


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
    
    # Get specialist Phase A settings for enhanced loss
    phase_a_settings = specialist_settings['phase_a']
    anti_batch_size = phase_a_settings.get('anti_batch_size', 0.0)
    anti_batch_lambda = phase_a_settings.get('anti_batch_lambda', BETA)
    cross_pair_settings = phase_a_settings.get('cross_pair_loss', {})
    cross_pair_enabled = cross_pair_settings.get('enabled', False)
    cross_pair_num_pairs = cross_pair_settings.get('num_pairs', 4)
    
    # Use settings for phase epochs if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_a']['epochs']
    
    # Log specialist training configuration
    logger.info(f"Specialist Phase A configuration:")
    logger.info(f"  Cross-pair reconstruction: {'ENABLED' if cross_pair_enabled else 'DISABLED'}")
    if cross_pair_enabled:
        logger.info(f"    - Cross-pair sampling: {cross_pair_num_pairs if cross_pair_num_pairs else 'ALL'} pairs")
    logger.info(f"  Anti-batch training: {'ENABLED' if anti_batch_size > 0 else 'DISABLED'}")
    if anti_batch_size > 0:
        logger.info(f"    - Anti-batch proportion: {anti_batch_size:.1%}")
        logger.info(f"    - Anti-batch λ: {anti_batch_lambda:.3f}")
    logger.info(f"  Beta warmup epochs: {phase_a_settings.get('beta_warmup_epochs', 5)}")
    
    # Log enhanced training mechanisms
    phase_a_enhanced = phase_a_settings.get('enhanced_training', {})
    logger.info(f"Enhanced training mechanisms:")
    logger.info(f"  - Free-bits (minimum KL): {'ENABLED' if phase_a_enhanced.get('use_free_bits', True) else 'DISABLED'}")
    if phase_a_enhanced.get('use_free_bits', True):
        logger.info(f"    δ = {phase_a_enhanced.get('free_bits_delta', 0.07):.3f} per dimension")
    logger.info(f"  - Cyclical β-annealing: {'ENABLED' if phase_a_enhanced.get('use_cyclical_beta', False) else 'DISABLED'}")
    if phase_a_enhanced.get('use_cyclical_beta', False):
        logger.info(f"    Cycle length: {phase_a_enhanced.get('beta_cycle_length', 4)} epochs")
    logger.info(f"  - Dynamic λ scheduling: {'ENABLED' if phase_a_enhanced.get('use_dynamic_lambda', True) else 'DISABLED'}")
    if phase_a_enhanced.get('use_dynamic_lambda', True):
        logger.info(f"    High-β multiplier: {phase_a_enhanced.get('lambda_high_multiplier', 3.0)}")
    logger.info(f"  - Contrastive KL margin: {'ENABLED' if phase_a_enhanced.get('use_contrastive_margin', True) else 'DISABLED'}")
    if phase_a_enhanced.get('use_contrastive_margin', True):
        logger.info(f"    Margin τ: {phase_a_enhanced.get('margin_tau', 1.5):.1f} nats")
    logger.info(f"  - Debug KL metrics: {'ENABLED' if phase_a_enhanced.get('debug_kl_metrics', False) else 'DISABLED'}")
    
    print(f"Phase A Enhanced Training: cross-pair={'ON' if cross_pair_enabled else 'OFF'}, anti-batch={'ON' if anti_batch_size > 0 else 'OFF'}")
    
    # Enhanced mechanisms summary
    enhanced_summary = []
    if phase_a_enhanced.get('use_free_bits', True):
        enhanced_summary.append("free-bits")
    if phase_a_enhanced.get('use_cyclical_beta', False):
        enhanced_summary.append("cyclical-β")
    if phase_a_enhanced.get('use_dynamic_lambda', True):
        enhanced_summary.append("dynamic-λ")
    if phase_a_enhanced.get('use_contrastive_margin', True):
        enhanced_summary.append("contrastive-margin")
    
    print(f"Enhanced mechanisms: {', '.join(enhanced_summary) if enhanced_summary else 'NONE'}")
    print(f"Target KL/dim after warm-up: ≥ 0.2 (debug_kl_metrics={'ON' if phase_a_enhanced.get('debug_kl_metrics', False) else 'OFF'})")
    
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
            
            # Calculate consistent global step for this epoch (used for ALL logging)
            global_step = encoder_idx * phase_epochs + epoch + 1
            
            # Progress bar for this encoder's training
            pbar = tqdm(dataloader, desc=f"Encoder {encoder_idx} Epoch {epoch+1}/{phase_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                    # Generate anti-batch samples for domain specialization
                    batch_size = input_seq.size(0)
                    anti_samples_count = int(batch_size * anti_batch_size) if anti_batch_size > 0 else 0
                    
                    if anti_samples_count > 0:
                        # Create anti-batch: samples from OTHER encoders' domains
                        other_encoder_indices = [i for i in range(num_encoders) if i != encoder_idx]
                        if other_encoder_indices:
                            # Sample from other encoders' datasets
                            other_datasets = [encoder_datasets[i] for i in other_encoder_indices if encoder_datasets[i][0]]
                            if other_datasets:
                                # Combine other encoders' data
                                all_other_inputs = []
                                all_other_outputs = []
                                for other_inputs, other_outputs in other_datasets:
                                    all_other_inputs.extend(other_inputs[:anti_samples_count//len(other_datasets)+1])
                                    all_other_outputs.extend(other_outputs[:anti_samples_count//len(other_datasets)+1])
                                
                                if len(all_other_inputs) >= anti_samples_count:
                                    # Sample anti-batch
                                    anti_indices = torch.randperm(len(all_other_inputs))[:anti_samples_count]
                                    anti_inputs = [all_other_inputs[i] for i in anti_indices]
                                    anti_outputs = [all_other_outputs[i] for i in anti_indices]
                                    
                                    # Convert to tensors
                                    anti_input_seq = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in anti_inputs]).to(device)
                                    anti_target_seq = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in anti_outputs]).to(device)
                                    
                                    # Combine in-slice and anti-batch samples
                                    combined_input = torch.cat([input_seq, anti_input_seq], dim=0)
                                    combined_target = torch.cat([target_seq, anti_target_seq], dim=0)
                                    
                                    # Create anti-mask: False for in-slice, True for anti-batch
                                    anti_mask = torch.cat([
                                        torch.zeros(batch_size, dtype=torch.bool, device=device),  # in-slice
                                        torch.ones(anti_samples_count, dtype=torch.bool, device=device)  # anti-batch
                                    ])
                                    
                                    # Use combined batch for training
                                    input_seq, target_seq = combined_input, combined_target
                                else:
                                    anti_mask = None
                            else:
                                anti_mask = None
                        else:
                            anti_mask = None
                    else:
                        anti_mask = None
                    
                    # SPECIALIST LOSS: Cross-pair reconstruction + beta warmup + anti-batch KL + ENHANCED MECHANISMS
                    # Get enhanced training settings
                    phase_a_enhanced = phase_a_settings.get('enhanced_training', {})
                    use_cyclical_beta = phase_a_enhanced.get('use_cyclical_beta', False)
                    beta_cycle_length = phase_a_enhanced.get('beta_cycle_length', 4)
                    use_free_bits = phase_a_enhanced.get('use_free_bits', True)
                    free_bits_delta = phase_a_enhanced.get('free_bits_delta', 0.07)
                    use_dynamic_lambda = phase_a_enhanced.get('use_dynamic_lambda', True)
                    lambda_high_multiplier = phase_a_enhanced.get('lambda_high_multiplier', 3.0)
                    use_contrastive_margin = phase_a_enhanced.get('use_contrastive_margin', True)
                    margin_tau = phase_a_enhanced.get('margin_tau', 1.5)
                    debug_kl_metrics = phase_a_enhanced.get('debug_kl_metrics', False)
                    
                    loss_result = compute_loss(
                        model, input_seq, target_seq, 
                        beta=BETA, encoder_idx=encoder_idx, use_independent_decoder=True,
                        # Specialist training parameters
                        current_epoch=epoch + 1,  # 1-indexed for beta warmup
                        anti_mask=anti_mask,  # Enable anti-batch training
                        anti_batch_lambda=anti_batch_lambda,
                        cross_pair_enabled=cross_pair_enabled,
                        cross_pair_num_pairs=cross_pair_num_pairs,
                        # Enhanced mechanisms
                        use_cyclical_beta=use_cyclical_beta,
                        beta_cycle_length=beta_cycle_length,
                        use_free_bits=use_free_bits,
                        free_bits_delta=free_bits_delta,
                        use_dynamic_lambda=use_dynamic_lambda,
                        lambda_high_multiplier=lambda_high_multiplier,
                        use_contrastive_margin=use_contrastive_margin,
                        margin_tau=margin_tau,
                        debug_kl_metrics=debug_kl_metrics,
                        return_components=True  # Get detailed loss breakdown
                    )
                    
                    # Extract total loss and components
                    if isinstance(loss_result, dict):
                        loss = loss_result['total_loss']
                        # Log detailed components for monitoring (use consistent global_step)
                        if batch_idx == 0 and wandb_logger:  # Log once per epoch for efficiency
                            log_dict = {
                                f'phase_a/encoder_{encoder_idx}_cross_pair_loss': loss_result['cross_pair_loss'].item(),
                                f'phase_a/encoder_{encoder_idx}_in_slice_kl_loss': loss_result['in_slice_kl_loss'].item(),
                                f'phase_a/encoder_{encoder_idx}_effective_beta': loss_result['effective_beta'],
                                f'phase_a/encoder_{encoder_idx}_effective_lambda': loss_result['effective_lambda'],
                                f'phase_a/encoder_{encoder_idx}_epoch': epoch + 1
                            }
                            
                            # Add VQ-VAE or KL metrics
                            if hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae():
                                log_dict[f'phase_a/encoder_{encoder_idx}_vq_loss'] = loss_result['vq_loss'].item()
                                if 'anti_vq_loss' in loss_result:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_anti_vq_loss'] = loss_result['anti_vq_loss'].item()
                                
                                # Add VQ-VAE metrics
                                vq_metrics = model.multi_encoder.encoders[encoder_idx].get_vq_metrics()
                                if vq_metrics:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_vq_codebook_perplexity'] = vq_metrics.get('codebook_perplexity', 0.0)
                                    log_dict[f'phase_a/encoder_{encoder_idx}_vq_num_embeddings'] = vq_metrics.get('num_embeddings', 0)
                                    
                                    # Log codebook usage statistics
                                    codebook_usage = vq_metrics.get('codebook_usage', None)
                                    if codebook_usage is not None:
                                        log_dict[f'phase_a/encoder_{encoder_idx}_vq_codebook_usage_entropy'] = -torch.sum(codebook_usage * torch.log(codebook_usage + 1e-10)).item()
                                        log_dict[f'phase_a/encoder_{encoder_idx}_vq_codebook_usage_max'] = torch.max(codebook_usage).item()
                                        log_dict[f'phase_a/encoder_{encoder_idx}_vq_codebook_usage_min'] = torch.min(codebook_usage).item()
                            else:
                                # Standard KL metrics
                                if 'anti_kl_loss' in loss_result:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_anti_kl_loss'] = loss_result['anti_kl_loss'].item()
                            
                            # Add enhanced mechanism metrics
                            if 'contrastive_margin_loss' in loss_result:
                                log_dict[f'phase_a/encoder_{encoder_idx}_contrastive_margin_loss'] = loss_result['contrastive_margin_loss'].item()
                            
                            # Add debug KL metrics if enabled
                            if debug_kl_metrics and not (hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae()):
                                if 'kl_per_dim' in loss_result and loss_result['kl_per_dim'] is not None:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_kl_per_dim'] = loss_result['kl_per_dim'].item()
                                if 'anti_kl_per_dim' in loss_result and loss_result['anti_kl_per_dim'] is not None:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_anti_kl_per_dim'] = loss_result['anti_kl_per_dim'].item()
                                if 'kl_gap' in loss_result and loss_result['kl_gap'] is not None:
                                    log_dict[f'phase_a/encoder_{encoder_idx}_kl_gap'] = loss_result['kl_gap'].item()
                            
                            wandb_logger.log_training_metrics(global_step, log_dict)
                    else:
                        loss = loss_result  # Fallback for backward compatibility
                    
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
            
            # Log to wandb - every epoch, not just evaluation epochs (using global_step calculated above)
            if wandb_logger:
                wandb_logger.log_training_metrics(global_step, {
                    f'phase_a/encoder_{encoder_idx}_loss': avg_epoch_loss,
                    'phase': 'A',
                    'current_encoder': encoder_idx
                })
                
                # Run periodic evaluation to track accuracy progression
                from utils.evaluation_utils import should_run_evaluation, run_quick_evaluation, log_evaluation_to_wandb
                wandb_settings = settings.get_wandb_settings()
                eval_interval = wandb_settings.get('eval_log_interval', 10)  # Evaluate every 10 epochs by default
                
                if should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                    print(f"Running evaluation for Encoder {encoder_idx} at epoch {epoch + 1}...")
                    try:
                        # Use ALL eval keys from settings for consistent evaluation
                        eval_settings = settings.get_evaluation_settings()
                        eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
                        print(f"  Evaluating on ALL keys: {eval_keys}")
                        
                        # Evaluate this specific encoder with its independent decoder on ALL eval keys
                        eval_results = run_quick_evaluation(
                            model, run_dir, global_step, 
                            eval_keys=eval_keys,  # Use ALL eval keys
                            encoder_idx=encoder_idx, use_independent_decoder=True
                        )
                        if eval_results:
                            # Log with encoder-specific prefix
                            log_evaluation_to_wandb(
                                eval_results, run_dir, global_step, wandb_logger, 
                                current_model=model, phase=f"phase_a_enc{encoder_idx}"
                            )
                            print(f"✓ Evaluation logged for Encoder {encoder_idx} at epoch {epoch + 1} on ALL keys")
                    except Exception as eval_error:
                        logger.warning(f"Evaluation failed for Encoder {encoder_idx} at epoch {epoch + 1}: {eval_error}")
                
                # Generate latent distribution histograms at evaluation intervals for ALL encoders
                if should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                    try:
                        generate_latent_histograms(model, dataloader, device, encoder_idx, epoch + 1, wandb_logger, global_step)
                        print(f"✓ Latent histograms logged for Encoder {encoder_idx} at epoch {epoch + 1}")
                    except Exception as hist_error:
                        logger.warning(f"Latent histogram generation failed for Encoder {encoder_idx} at epoch {epoch + 1}: {hist_error}")
                
                # Generate per-dimension KL divergence plots at evaluation intervals
                if should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                    try:
                        generate_per_dimension_kl_plot(model, dataloader, device, epoch + 1, encoder_idx=encoder_idx, wandb_logger=wandb_logger, global_step=global_step)
                        print(f"✓ Per-dimension KL plot logged for Encoder {encoder_idx} at epoch {epoch + 1}")
                    except Exception as kl_error:
                        logger.warning(f"Per-dimension KL plot generation failed for Encoder {encoder_idx} at epoch {epoch + 1}: {kl_error}")
                
                # Basic visualizations are handled by the reconstruction plots above
            
            # Generate reconstruction plot at evaluation intervals            
            if wandb_logger and should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                try:
                    plot_path, wandb_key = generate_specialist_reconstruction_plot(
                        model, dataloader, device, epoch + 1, phase='A',
                        encoder_idx=encoder_idx, use_independent_decoder=True,
                        key=TRAINING_KEYS[encoder_idx % len(TRAINING_KEYS)]
                    )
                    
                    if plot_path and wandb_key:
                        import wandb
                        import os
                        try:
                            wandb_logger._safe_log({
                                f"phase_a_reconstruction/{wandb_key}": wandb.Image(plot_path)
                            }, step_hint=global_step)
                            logger.info(f"✓ Logged Phase A reconstruction for Encoder {encoder_idx} at step {global_step}")
                        except Exception as wandb_error:
                            logger.warning(f"Failed to log Phase A reconstruction to WandB: {wandb_error}")
                        
                        os.unlink(plot_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate reconstruction plot for Encoder {encoder_idx} at epoch {epoch+1}: {e}")
                
                model.train()
        
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
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            # Note: os is already imported at module level
            
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
                    # Use the final global step from the last encoder/epoch for Phase A summary
                    final_phase_a_step = (num_encoders - 1) * phase_epochs + phase_epochs
                    wandb_logger._safe_log({
                        "phase_a/encoder_losses_summary": wandb.Image(phase_a_plot_path),
                        "phase_a/completed": True
                    }, step_hint=final_phase_a_step)
                    logger.info("✓ Phase A summary plot logged to WandB")
                except Exception as wandb_error:
                    logger.warning(f"Failed to upload Phase A summary plot to WandB: {wandb_error}")
            except Exception as e:
                logger.warning(f"Failed to log Phase A summary to WandB: {e}")
            
            logger.info("✓ Phase A summary visualization completed")
            
        except Exception as plot_error:
            logger.warning(f"Failed to create Phase A summary plot: {plot_error}")
            logger.info("Phase A training completed successfully despite visualization error")
    
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
                # Get enhanced training settings for Phase B
                phase_b_settings = specialist_settings.get('phase_b', {})
                phase_b_enhanced = phase_b_settings.get('enhanced_training', {})
                use_cyclical_beta = phase_b_enhanced.get('use_cyclical_beta', False)
                beta_cycle_length = phase_b_enhanced.get('beta_cycle_length', 4)
                use_free_bits = phase_b_enhanced.get('use_free_bits', True)
                free_bits_delta = phase_b_enhanced.get('free_bits_delta', 0.07)
                debug_kl_metrics = phase_b_enhanced.get('debug_kl_metrics', False)
                
                # Use PoE inference with shared decoder (encoder_idx=None triggers PoE) + enhanced mechanisms
                loss_result = compute_loss(
                    model, input_seq, target_seq,
                    beta=BETA, encoder_idx=None, use_independent_decoder=False,
                    # Enhanced mechanisms for Phase B
                    current_epoch=epoch + 1,  # 1-indexed for cyclical beta
                    use_cyclical_beta=use_cyclical_beta,
                    beta_cycle_length=beta_cycle_length,
                    use_free_bits=use_free_bits,
                    free_bits_delta=free_bits_delta,
                    debug_kl_metrics=debug_kl_metrics,
                    return_components=True  # Get detailed loss breakdown
                )
                
                # Extract loss and components
                if isinstance(loss_result, dict):
                    loss = loss_result['total_loss']
                    # Log Phase B enhanced metrics periodically
                    if batch_idx == 0 and wandb_logger:  # Log once per epoch
                        log_dict = {
                            'phase_b/effective_beta': loss_result['effective_beta'],
                            'phase_b/reconstruction_loss': loss_result['reconstruction_loss'].item(),
                        }
                        
                        # Add VQ-VAE or KL metrics
                        if hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae():
                            log_dict['phase_b/vq_loss'] = loss_result['vq_loss'].item()
                            
                            # Add VQ-VAE metrics from all encoders
                            vq_metrics = model.multi_encoder.get_vq_metrics()
                            if vq_metrics:
                                # Aggregate metrics across encoders
                                total_perplexity = 0
                                total_usage_entropy = 0
                                num_encoders = 0
                                
                                for key, value in vq_metrics.items():
                                    if 'codebook_perplexity' in key:
                                        total_perplexity += value
                                        num_encoders += 1
                                    elif 'codebook_usage' in key and isinstance(value, torch.Tensor):
                                        usage_entropy = -torch.sum(value * torch.log(value + 1e-10)).item()
                                        total_usage_entropy += usage_entropy
                                
                                if num_encoders > 0:
                                    log_dict['phase_b/vq_avg_codebook_perplexity'] = total_perplexity / num_encoders
                                    log_dict['phase_b/vq_avg_codebook_usage_entropy'] = total_usage_entropy / num_encoders
                        else:
                            log_dict['phase_b/kl_loss'] = loss_result['kl_loss'].item()
                        
                        # Add debug KL metrics if enabled
                        if debug_kl_metrics and 'kl_per_dim' in loss_result and loss_result['kl_per_dim'] is not None:
                            log_dict['phase_b/kl_per_dim'] = loss_result['kl_per_dim'].item()
                        
                        wandb_logger.log_training_metrics(base_global_step + epoch + 1, log_dict)
                else:
                    loss = loss_result  # Fallback for backward compatibility
                    
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
            
            # Run periodic evaluation to track PoE accuracy progression
            from utils.evaluation_utils import should_run_evaluation, run_quick_evaluation, log_evaluation_to_wandb
            wandb_settings = settings.get_wandb_settings()
            eval_interval = wandb_settings.get('eval_log_interval', 10)  # Evaluate every 10 epochs by default
            
            if should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                print(f"Running PoE evaluation at Phase B epoch {epoch + 1}...")
                try:
                    # Use ALL eval keys from settings for consistent evaluation
                    eval_settings = settings.get_evaluation_settings()
                    eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
                    print(f"  Evaluating PoE on ALL keys: {eval_keys}")
                    
                    # Evaluate PoE with shared decoder on ALL eval keys
                    eval_results = run_quick_evaluation(
                        model, run_dir, global_step, 
                        eval_keys=eval_keys,  # Use ALL eval keys
                        encoder_idx=None, use_independent_decoder=False  # PoE + shared decoder
                    )
                    if eval_results:
                        # Log with phase B prefix
                        log_evaluation_to_wandb(
                            eval_results, run_dir, global_step, wandb_logger, 
                            current_model=model, phase="phase_b_poe"
                        )
                        print(f"✓ PoE evaluation logged at Phase B epoch {epoch + 1} on ALL keys")
                except Exception as eval_error:
                    logger.warning(f"PoE evaluation failed at Phase B epoch {epoch + 1}: {eval_error}")
            
            # Generate per-dimension KL divergence plots at evaluation intervals for PoE
            if should_run_evaluation(epoch + 1, eval_interval, phase_epochs):
                try:
                    generate_per_dimension_kl_plot(model, mixed_dataloader, device, epoch + 1, encoder_idx=None, wandb_logger=wandb_logger, global_step=global_step)
                    print(f"✓ Per-dimension KL plot logged for PoE at Phase B epoch {epoch + 1}")
                except Exception as kl_error:
                    logger.warning(f"Per-dimension KL plot generation failed for PoE at Phase B epoch {epoch + 1}: {kl_error}")
            
            # Basic visualizations are handled by the reconstruction plots below
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == phase_epochs:
            checkpoint_path = save_phase_checkpoint(
                model, optimizer, 'decoder', epoch + 1, avg_epoch_loss, run_dir
            )
            logger.info(f"Phase B checkpoint saved: {checkpoint_path}")
    
        # Skip per-epoch reconstruction plots - rely on final comparison plot instead for efficiency
    
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
                # Use the final global step from Phase B for summary
                final_phase_b_step = base_global_step + phase_epochs
                wandb_logger._safe_log({
                    "phase_b/decoder_loss_summary": wandb.Image(phase_b_plot_path),
                    "phase_b/completed": True
                }, step_hint=final_phase_b_step)
                logger.info("✓ Phase B summary plot logged to WandB")
            except Exception as wandb_error:
                logger.warning(f"Failed to upload Phase B summary plot to WandB: {wandb_error}")
        except Exception as e:
            logger.warning(f"Failed to log Phase B summary to WandB: {e}")
        
        logger.info("✓ Phase B visualization completed")
    
    return phase_b_results


def run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, phase_name):
    """Run evaluation between training phases using ALL evaluation keys."""
    logger.info(f"\n--- Evaluation after {phase_name} ---")
    print(f"Running evaluation after {phase_name}...")
    
    try:
        # Use ALL eval keys from settings for comprehensive evaluation
        eval_settings = settings.get_evaluation_settings()
        eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
        print(f"  Evaluating after {phase_name} on ALL keys: {eval_keys}")
        
        eval_results = run_quick_evaluation(
            model, run_dir, epoch=f"{phase_name}_final", eval_keys=eval_keys  # Use ALL eval keys
        )
        if eval_results and wandb_logger:
            # Pass the current model to avoid loading from disk
            log_evaluation_to_wandb(eval_results, run_dir, f"{phase_name}_final", wandb_logger, 
                                  current_model=model, phase=phase_name.lower().replace(' ', '_'))
            logger.info(f"✓ Evaluation results logged to wandb for {phase_name} on ALL keys")
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
    
    # Load specialist settings explicitly
    import os
    from utils.settings_manager import init_settings
    global settings
    specialist_settings_file = "model_specialist_settings.json"
    if os.path.exists(specialist_settings_file):
        settings = init_settings(specialist_settings_file)
        print(f"✓ Loaded specialist settings from {specialist_settings_file}")
    else:
        print(f"⚠ Warning: {specialist_settings_file} not found, using default settings")
    
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
    
    # ========================================
    # FREEZE CONFIGURATION FOR CONSISTENCY
    # ========================================
    frozen_config_path = os.path.join(run_dir, 'frozen_config.json')
    
    if resume_from_phase and os.path.exists(frozen_config_path):
        # Resuming: Load frozen config from run directory
        print(f"🔄 Resuming from phase {resume_from_phase}: Loading frozen config from run directory")
        logger.info(f"Resuming training - loading frozen config from {frozen_config_path}")
        
        with open(frozen_config_path, 'r') as f:
            frozen_config = json.load(f)
        
        # Update settings with frozen config
        settings.set_settings(frozen_config)
        
        # Re-extract settings after loading frozen config
        data_settings = settings.get_data_settings()
        model_architecture = settings.get_model_architecture()
        training_settings = settings.get_training_settings()
        latent_optimization = settings.get_latent_optimization()
        repulsion_loss_settings = settings.get_repulsion_loss_settings()
        wandb_settings = settings.get_wandb_settings()
        specialist_settings = settings.get_specialist_training_settings()
        
        # Update derived variables
        TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
        NUM_ENCODERS = model_architecture.get('num_encoders', 1)
        N_EXAMPLES_PER_TASK = data_settings['n']
        
        print(f"✓ Loaded frozen config: {NUM_ENCODERS} encoders, decoder_hidden_dim={model_architecture.get('decoder_hidden_dim')}")
        
    else:
        # Starting fresh: Save current config as frozen config
        print("💾 Starting fresh training: Freezing current config in run directory")
        logger.info(f"Saving frozen config to {frozen_config_path}")
        
        current_config = settings.get_settings()
        with open(frozen_config_path, 'w') as f:
            json.dump(current_config, f, indent=2)
        
        print(f"✓ Frozen config saved: {NUM_ENCODERS} encoders, decoder_hidden_dim={model_architecture.get('decoder_hidden_dim')}")
    
    logger.info(f"Starting specialist training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Using frozen config - Model architecture: encoders={NUM_ENCODERS}, decoder_hidden_dim={model_architecture.get('decoder_hidden_dim')}, decoder_layers={model_architecture.get('decoder_layers')}")
    logger.info(f"Full frozen settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)
    
    # Initialize wandb
    wandb_logger = None
    if wandb_settings.get('enabled', False):
        wandb_logger = init_wandb_for_mode('specialist_train', run_dir)
        if wandb_logger:
            logger.info(f"✓ Wandb logging enabled: {wandb_logger.run.name}")
            # Verify trajectory plot settings
            trajectory_plots_enabled = wandb_settings.get('log_trajectory_plots', False)
            trajectory_max_samples = wandb_settings.get('trajectory_max_samples', 3)
            eval_interval = wandb_settings.get('eval_log_interval', 10)
            logger.info(f"✓ Trajectory plots: {'ENABLED' if trajectory_plots_enabled else 'DISABLED'}")
            logger.info(f"✓ Trajectory max samples: {trajectory_max_samples}")
            logger.info(f"✓ Evaluation/plot interval: every {eval_interval} epochs")
            if trajectory_plots_enabled:
                print(f"Trajectory plots enabled: {trajectory_max_samples} samples every {eval_interval} epochs")
            else:
                print("⚠ Warning: Trajectory plots are disabled in WandB settings")
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
    model = build_model(device, wandb_logger, global_step=0)
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
            
            # Add PoE accuracy snapshot using ALL eval keys
            eval_settings = settings.get_evaluation_settings()
            eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
            print(f"Recording Phase A PoE accuracy snapshot on ALL keys: {eval_keys}")
            poe_accuracy = run_quick_evaluation(
                model, run_dir, epoch=f"Phase A Epoch {phase_a_epochs}", eval_keys=eval_keys  # Use ALL eval keys
            )
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
            
            # Add PoE accuracy snapshot using ALL eval keys
            eval_settings = settings.get_evaluation_settings()
            eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
            print(f"Recording Phase B PoE accuracy snapshot on ALL keys: {eval_keys}")
            poe_accuracy = run_quick_evaluation(
                model, run_dir, epoch=f"Phase B Epoch {phase_b_epochs}", eval_keys=eval_keys  # Use ALL eval keys
            )
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