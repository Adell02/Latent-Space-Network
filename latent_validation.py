#!/usr/bin/env python3
"""
Simple Latent Validation for Latent Program Networks

Tests whether the encoder is learning meaningful latents or if the decoder
is absorbing all the load. Uploads comprehensive results to WandB.

Key Tests:
1. LATENT SWAP: Use latent from sample A to decode input B  
2. ZERO LATENT: Use zero latent vector (should fail if latents matter)
3. RANDOM LATENT: Use random latent vector (should fail if latents matter)
4. LATENT SPACE VISUALIZATION: 2D projection showing clustering

Usage:
    python latent_validation.py --file_name MODEL_DIR --epoch EPOCH
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tempfile
import os

from utils.settings_manager import init_settings
from utils.model_utils import load_model
from utils.wandb_logger import init_wandb_for_mode
from utils.data_preparation import extract_grid_from_sequence
from re_arc.main import generate_and_process_tasks
from models.base_model import gaussian_poe


def latent_swap_test(model, input_seqs, target_seqs, device, n_samples=5):
    """
    Test 1: LATENT SWAP
    For multi-encoder: use PoE of all encoders' latents for each sample.
    """
    model.eval()
    results = []
    correct_recons = []
    with torch.no_grad():
        n = min(n_samples, len(input_seqs))
        is_multi = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
        num_encoders = len(model.multi_encoder.encoders) if is_multi else 1
        # Collect all encoder mus/logvars for all samples
        all_mus = []  # shape: [n, num_encoders, latent_dim]
        all_logvars = []
        for i in range(n):
            input_seq = torch.tensor(input_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            target_seq = torch.tensor(target_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            if is_multi:
                mus = []
                logvars = []
                for enc in model.multi_encoder.encoders:
                    mu, logvar, _ = enc(input_seq, target_seq)
                    mus.append(mu[0])
                    logvars.append(logvar[0])
                mus = torch.stack(mus)  # [num_encoders, latent_dim]
                logvars = torch.stack(logvars)
            else:
                _, mu, logvar = model(input_seq, target_seq)
                mus = mu[0].unsqueeze(0)
                logvars = logvar[0].unsqueeze(0)
            all_mus.append(mus)
            all_logvars.append(logvars)
        # Compute correct reconstructions (PoE for each sample)
        for i in range(n):
            input_seq = torch.tensor(input_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            target_seq = torch.tensor(target_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            if is_multi:
                mu_star, logvar_star = gaussian_poe(all_mus[i], all_logvars[i])
                z_star = mu_star.unsqueeze(0)
                shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_star, input_seq, target_seq=target_seq)
            else:
                shape_logits, grid_logits = model.decoder(all_mus[i][0].unsqueeze(0), input_seq, target_seq=target_seq)
            correct_recons.append({
                'input_source': i,
                'input_seq': input_seqs[i],
                'target_seq': target_seqs[i],
                'shape_logits': shape_logits[0].argmax(dim=-1).cpu().numpy(),
                'grid_logits': grid_logits[0].argmax(dim=-1).cpu().numpy()
            })
        # Swaps: use PoE of sample i's latents for sample j's input
        for i in range(n):
            for j in range(n):
                if i != j:
                    input_seq = torch.tensor(input_seqs[j], dtype=torch.float32).unsqueeze(0).to(device)
                    target_seq = torch.tensor(target_seqs[j], dtype=torch.float32).unsqueeze(0).to(device)
                    if is_multi:
                        mu_star, logvar_star = gaussian_poe(all_mus[i], all_logvars[i])
                        z_star = mu_star.unsqueeze(0)
                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_star, input_seq, target_seq=target_seq)
                    else:
                        shape_logits, grid_logits = model.decoder(all_mus[i][0].unsqueeze(0), input_seq, target_seq=target_seq)
                    results.append({
                        'latent_source': i,
                        'input_source': j,
                        'input_seq': input_seqs[j],
                        'target_seq': target_seqs[j],
                        'shape_logits': shape_logits[0].argmax(dim=-1).cpu().numpy(),
                        'grid_logits': grid_logits[0].argmax(dim=-1).cpu().numpy()
                    })
    return correct_recons, results


def zero_random_latent_test(model, input_seqs, target_seqs, device, n_samples=3):
    """
    Test 2 & 3: ZERO LATENT and RANDOM LATENT
    For multi-encoder: use PoE of all encoders' zero/random latents.
    """
    model.eval()
    results = {'zero': [], 'random': [], 'correct': []}
    is_multi = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = len(model.multi_encoder.encoders) if is_multi else 1
    latent_dim = model.multi_encoder.latent_dim if is_multi else model.latent_dim
    with torch.no_grad():
        n = min(n_samples, len(input_seqs))
        # Precompute all mus/logvars for correct recon
        all_mus = []
        all_logvars = []
        for i in range(n):
            input_seq = torch.tensor(input_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            target_seq = torch.tensor(target_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            if is_multi:
                mus = []
                logvars = []
                for enc in model.multi_encoder.encoders:
                    mu, logvar, _ = enc(input_seq, target_seq)
                    mus.append(mu[0])
                    logvars.append(logvar[0])
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
            else:
                _, mu, logvar = model(input_seq, target_seq)
                mus = mu[0].unsqueeze(0)
                logvars = logvar[0].unsqueeze(0)
            all_mus.append(mus)
            all_logvars.append(logvars)
        for i in range(n):
            input_seq = torch.tensor(input_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            target_seq = torch.tensor(target_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            # Correct recon (PoE)
            if is_multi:
                mu_star, logvar_star = gaussian_poe(all_mus[i], all_logvars[i])
                z_star = mu_star.unsqueeze(0)
                shape_logits_corr, grid_logits_corr = model.multi_encoder.shared_decoder(z_star, input_seq, target_seq=target_seq)
            else:
                shape_logits_corr, grid_logits_corr = model.decoder(all_mus[i][0].unsqueeze(0), input_seq, target_seq=target_seq)
            results['correct'].append({
                'input_seq': input_seqs[i],
                'target_seq': target_seqs[i],
                'shape_logits': shape_logits_corr[0].argmax(dim=-1).cpu().numpy(),
                'grid_logits': grid_logits_corr[0].argmax(dim=-1).cpu().numpy()
            })
            # Zero latent (PoE of all-zero)
            if is_multi:
                zero_latents = torch.zeros(num_encoders, latent_dim, device=device)
                zero_logvars = torch.zeros(num_encoders, latent_dim, device=device)
                mu_star, logvar_star = gaussian_poe(zero_latents, zero_logvars)
                z_star = mu_star.unsqueeze(0)
                shape_logits_zero, grid_logits_zero = model.multi_encoder.shared_decoder(z_star, input_seq, target_seq=target_seq)
            else:
                zero_latent = torch.zeros(1, latent_dim, device=device)
                shape_logits_zero, grid_logits_zero = model.decoder(zero_latent, input_seq, target_seq=target_seq)
            # Random latent (PoE of all-random)
            if is_multi:
                rand_latents = torch.randn(num_encoders, latent_dim, device=device)
                rand_logvars = torch.zeros(num_encoders, latent_dim, device=device)
                mu_star, logvar_star = gaussian_poe(rand_latents, rand_logvars)
                z_star = mu_star.unsqueeze(0)
                shape_logits_rand, grid_logits_rand = model.multi_encoder.shared_decoder(z_star, input_seq, target_seq=target_seq)
            else:
                rand_latent = torch.randn(1, latent_dim, device=device)
                shape_logits_rand, grid_logits_rand = model.decoder(rand_latent, input_seq, target_seq=target_seq)
            results['zero'].append({
                'input_seq': input_seqs[i],
                'target_seq': target_seqs[i],
                'shape_logits': shape_logits_zero[0].argmax(dim=-1).cpu().numpy(),
                'grid_logits': grid_logits_zero[0].argmax(dim=-1).cpu().numpy()
            })
            results['random'].append({
                'input_seq': input_seqs[i],
                'target_seq': target_seqs[i],
                'shape_logits': shape_logits_rand[0].argmax(dim=-1).cpu().numpy(),
                'grid_logits': grid_logits_rand[0].argmax(dim=-1).cpu().numpy()
            })
    return results


def latent_space_visualization(model, input_seqs, target_seqs, device, max_samples=50, sample_keys=None):
    """
    Test 4: LATENT SPACE VISUALIZATION
    Project latents to 2D and color by properties (keys, grid size, dominant color, etc.)
    If latents are meaningful, should see clustering by semantic properties.
    
    Args:
        sample_keys: List of keys corresponding to each sample (for key-based coloring)
    """
    model.eval()
    latents = []
    properties = []
    
    with torch.no_grad():
        for i in range(min(max_samples, len(input_seqs))):
            input_seq = torch.tensor(input_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            target_seq = torch.tensor(target_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
            
            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                # Use first encoder for simplicity
                mu, logvar,_ = model.multi_encoder.encoders[0](input_seq, target_seq)
            else:
                _, mu, logvar = model(input_seq, target_seq)
            
            latents.append(mu[0].cpu().numpy())
            
            # Extract semantic properties
            try:
                target_grid, target_shape = extract_grid_from_sequence(target_seqs[i])
                grid_size = target_shape[0] * target_shape[1]
                dominant_color = np.bincount(target_grid.flatten().astype(int)).argmax()
                
                # Add key information if available
                sample_key = sample_keys[i] if sample_keys and i < len(sample_keys) else 'unknown'
                
                properties.append({
                    'key': sample_key,
                    'grid_size': grid_size,
                    'dominant_color': dominant_color,
                    'rows': target_shape[0],
                    'cols': target_shape[1],
                    'complexity': grid_size + len(np.unique(target_grid.flatten()))  # Size + color diversity
                })
            except:
                sample_key = sample_keys[i] if sample_keys and i < len(sample_keys) else 'unknown'
                properties.append({
                    'key': sample_key,
                    'grid_size': 0,
                    'dominant_color': 0,
                    'rows': 0,
                    'cols': 0,
                    'complexity': 0
                })
    
    latents = np.array(latents)
    
    # PCA projection to 2D
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    explained_variance = pca.explained_variance_ratio_
    
    # t-SNE projection to 2D (if we have enough samples)
    latents_tsne = None
    if len(latents) >= 10:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
            latents_tsne = tsne.fit_transform(latents)
        except:
            pass
    
    return latents_pca, latents_tsne, properties, explained_variance


def create_visualization_plots(swap_results_tuple, zero_random_results, latents_pca, latents_tsne, properties, explained_variance=None, run_dir=None):
    """Create comprehensive visualization plots for WandB."""
    import matplotlib.pyplot as plt
    import os
    plots = {}
    correct_recons, swap_results = swap_results_tuple
    # 1. LATENT SWAP VISUALIZATION
    if correct_recons and swap_results:
        print("Creating latent swap plot...")
        n_show = min(3, len(correct_recons))
        fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12))
        if n_show == 1:
            axes = axes.reshape(3, 1)
        for idx in range(n_show):
            try:
                # Original target
                target_grid, _ = extract_grid_from_sequence(correct_recons[idx]['target_seq'])
                axes[0, idx].imshow(target_grid, cmap='viridis', interpolation='nearest')
                axes[0, idx].set_title(f'Target {idx}')
                axes[0, idx].axis('off')
                # Correct reconstruction
                corr_seq = correct_recons[idx]['target_seq'].copy()
                corr_seq[900:902] = correct_recons[idx]['shape_logits']
                if len(correct_recons[idx]['shape_logits']) >= 2 and correct_recons[idx]['shape_logits'][0] > 0:
                    corr_seq[:min(len(correct_recons[idx]['grid_logits']), 900)] = correct_recons[idx]['grid_logits'][:900]
                corr_grid, _ = extract_grid_from_sequence(corr_seq)
                axes[1, idx].imshow(corr_grid, cmap='viridis', interpolation='nearest')
                axes[1, idx].set_title('Correct Recon')
                axes[1, idx].axis('off')
                # Swapped reconstruction (pick the first swap for this input)
                swap = next((s for s in swap_results if s['input_source'] == idx), None)
                if swap:
                    swap_seq = swap['target_seq'].copy()
                    swap_seq[900:902] = swap['shape_logits']
                    if len(swap['shape_logits']) >= 2 and swap['shape_logits'][0] > 0:
                        swap_seq[:min(len(swap['grid_logits']), 900)] = swap['grid_logits'][:900]
                    swap_grid, _ = extract_grid_from_sequence(swap_seq)
                    axes[2, idx].imshow(swap_grid, cmap='viridis', interpolation='nearest')
                    axes[2, idx].set_title(f'Swapped Recon\n(Latent {swap["latent_source"]} → Input {swap["input_source"]})')
                    axes[2, idx].axis('off')
                else:
                    axes[2, idx].text(0.5, 0.5, 'No swap', ha='center', va='center')
            except Exception as e:
                for row in range(3):
                    axes[row, idx].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
        plt.suptitle('LATENT SWAP TEST\nRow 1: Target, Row 2: Correct Recon, Row 3: Swapped Recon', fontsize=14)
        plt.tight_layout()
        fname = os.path.join(run_dir, 'latent_swap.png') if run_dir else 'latent_swap.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plots['latent_swap'] = fname
        print(f"  ✓ Latent swap plot saved: {fname}")
        plt.close()
    else:
        print("  ⚠ No swap results, skipping latent swap plot.")
    # 2. ZERO/RANDOM LATENT VISUALIZATION
    if zero_random_results['zero'] and zero_random_results['correct']:
        print("Creating zero/random latent plot...")
        n_show = min(3, len(zero_random_results['zero']))
        fig, axes = plt.subplots(4, n_show, figsize=(5 * n_show, 16))
        if n_show == 1:
            axes = axes.reshape(4, 1)
        for idx in range(n_show):
            try:
                # Original target
                target_grid, _ = extract_grid_from_sequence(zero_random_results['correct'][idx]['target_seq'])
                axes[0, idx].imshow(target_grid, cmap='viridis', interpolation='nearest')
                axes[0, idx].set_title(f'Target {idx}')
                axes[0, idx].axis('off')
                # Correct reconstruction
                corr_seq = zero_random_results['correct'][idx]['target_seq'].copy()
                corr_seq[900:902] = zero_random_results['correct'][idx]['shape_logits']
                if len(zero_random_results['correct'][idx]['shape_logits']) >= 2 and zero_random_results['correct'][idx]['shape_logits'][0] > 0:
                    corr_seq[:min(len(zero_random_results['correct'][idx]['grid_logits']), 900)] = zero_random_results['correct'][idx]['grid_logits'][:900]
                corr_grid, _ = extract_grid_from_sequence(corr_seq)
                axes[1, idx].imshow(corr_grid, cmap='viridis', interpolation='nearest')
                axes[1, idx].set_title('Correct Recon')
                axes[1, idx].axis('off')
                # Zero latent reconstruction
                zero_seq = zero_random_results['zero'][idx]['target_seq'].copy()
                zero_seq[900:902] = zero_random_results['zero'][idx]['shape_logits']
                if len(zero_random_results['zero'][idx]['shape_logits']) >= 2 and zero_random_results['zero'][idx]['shape_logits'][0] > 0:
                    zero_seq[:min(len(zero_random_results['zero'][idx]['grid_logits']), 900)] = zero_random_results['zero'][idx]['grid_logits'][:900]
                zero_grid, _ = extract_grid_from_sequence(zero_seq)
                axes[2, idx].imshow(zero_grid, cmap='viridis', interpolation='nearest')
                axes[2, idx].set_title('Zero Latent')
                axes[2, idx].axis('off')
                # Random latent reconstruction
                rand_seq = zero_random_results['random'][idx]['target_seq'].copy()
                rand_seq[900:902] = zero_random_results['random'][idx]['shape_logits']
                if len(zero_random_results['random'][idx]['shape_logits']) >= 2 and zero_random_results['random'][idx]['shape_logits'][0] > 0:
                    rand_seq[:min(len(zero_random_results['random'][idx]['grid_logits']), 900)] = zero_random_results['random'][idx]['grid_logits'][:900]
                rand_grid, _ = extract_grid_from_sequence(rand_seq)
                axes[3, idx].imshow(rand_grid, cmap='viridis', interpolation='nearest')
                axes[3, idx].set_title('Random Latent')
                axes[3, idx].axis('off')
            except Exception as e:
                for row in range(4):
                    axes[row, idx].text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center')
        plt.suptitle('ZERO/RANDOM LATENT TEST\nRow 1: Target, Row 2: Correct Recon, Row 3: Zero Latent, Row 4: Random Latent', fontsize=14)
        plt.tight_layout()
        fname = os.path.join(run_dir, 'zero_random_latent.png') if run_dir else 'zero_random_latent.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plots['zero_random_latent'] = fname
        print(f"  ✓ Zero/random latent plot saved: {fname}")
        plt.close()
    else:
        print("  ⚠ No zero/random results, skipping zero/random latent plot.")
    
    # 3. ENHANCED LATENT SPACE VISUALIZATION
    if latents_pca is not None and len(latents_pca) > 0:
        print("Creating latent space plot...")
        # Get unique keys and create color mapping
        keys = [p['key'] for p in properties]
        unique_keys = list(set(keys))
        n_keys = len(unique_keys)
        
        # Create distinct colors for keys
        if n_keys > 1:
            key_colors = plt.cm.tab10(np.linspace(0, 1, min(n_keys, 10)))
            if n_keys > 10:
                key_colors = plt.cm.tab20(np.linspace(0, 1, min(n_keys, 20)))
        else:
            key_colors = ['blue']
        
        key_to_color = {key: key_colors[i % len(key_colors)] for i, key in enumerate(unique_keys)}
        point_colors = [key_to_color[key] for key in keys]
        
        # Create comprehensive latent space plots
        n_plots = 3 if latents_tsne is not None else 2
        fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))
        if n_plots == 1:
            axes = axes.reshape(2, 1)
        
        # Row 1: PCA plots
        # PCA by Key
        for i, key in enumerate(unique_keys):
            key_mask = np.array(keys) == key
            if np.any(key_mask):
                axes[0, 0].scatter(latents_pca[key_mask, 0], latents_pca[key_mask, 1], 
                                 c=[key_to_color[key]], label=f'Key: {key}', alpha=0.7, s=60)
        axes[0, 0].set_title(f'PCA by Problem Key\nPC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%}' if explained_variance is not None else 'PCA by Problem Key')
        axes[0, 0].set_xlabel('PC 1')
        axes[0, 0].set_ylabel('PC 2')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PCA by Grid Size
        grid_sizes = [p['grid_size'] for p in properties]
        scatter1 = axes[0, 1].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                    c=grid_sizes, cmap='viridis', alpha=0.7, s=60)
        axes[0, 1].set_title('PCA by Grid Size')
        axes[0, 1].set_xlabel('PC 1')
        axes[0, 1].set_ylabel('PC 2')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 1], label='Grid Size')
        
        # PCA by Complexity (if t-SNE available)
        if latents_tsne is not None:
            complexities = [p['complexity'] for p in properties]
            scatter2 = axes[0, 2].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                        c=complexities, cmap='plasma', alpha=0.7, s=60)
            axes[0, 2].set_title('PCA by Complexity')
            axes[0, 2].set_xlabel('PC 1')
            axes[0, 2].set_ylabel('PC 2')
            axes[0, 2].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[0, 2], label='Complexity')
        
        # Row 2: t-SNE plots (if available)
        if latents_tsne is not None:
            # t-SNE by Key
            for i, key in enumerate(unique_keys):
                key_mask = np.array(keys) == key
                if np.any(key_mask):
                    axes[1, 0].scatter(latents_tsne[key_mask, 0], latents_tsne[key_mask, 1], 
                                     c=[key_to_color[key]], label=f'Key: {key}', alpha=0.7, s=60)
            axes[1, 0].set_title('t-SNE by Problem Key')
            axes[1, 0].set_xlabel('t-SNE 1')
            axes[1, 0].set_ylabel('t-SNE 2')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
            
            # t-SNE by Grid Size
            scatter3 = axes[1, 1].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                                        c=grid_sizes, cmap='viridis', alpha=0.7, s=60)
            axes[1, 1].set_title('t-SNE by Grid Size')
            axes[1, 1].set_xlabel('t-SNE 1')
            axes[1, 1].set_ylabel('t-SNE 2')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=axes[1, 1], label='Grid Size')
            
            # t-SNE by Complexity
            scatter4 = axes[1, 2].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                                        c=complexities, cmap='plasma', alpha=0.7, s=60)
            axes[1, 2].set_title('t-SNE by Complexity')
            axes[1, 2].set_xlabel('t-SNE 1')
            axes[1, 2].set_ylabel('t-SNE 2')
            axes[1, 2].grid(True, alpha=0.3)
            plt.colorbar(scatter4, ax=axes[1, 2], label='Complexity')
        else:
            # Fill second row with additional PCA analyses
            complexities = [p['complexity'] for p in properties]
            scatter_alt = axes[1, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                           c=complexities, cmap='plasma', alpha=0.7, s=60)
            axes[1, 0].set_title('PCA by Complexity')
            axes[1, 0].set_xlabel('PC 1')
            axes[1, 0].set_ylabel('PC 2')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter_alt, ax=axes[1, 0], label='Complexity')
            
            # Hide unused subplot
            axes[1, 1].axis('off')
            if n_plots > 2:
                axes[1, 2].axis('off')
        
        plt.suptitle(f'LATENT SPACE CLUSTERING ANALYSIS\n{len(unique_keys)} Problem Keys • {len(latents_pca)} Samples', fontsize=16)
        plt.tight_layout()
        
        fname = os.path.join(run_dir, 'latent_space.png') if run_dir else 'latent_space.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plots['latent_space'] = fname
        print(f"  ✓ Latent space plot saved: {fname}")
        plt.close()
    else:
        print("  ⚠ No latent PCA data, skipping latent space plot.")
    
    return plots


def run_latent_validation_for_specialist(run_dir, model, device, eval_keys, n_samples_per_key=5, wandb_logger=None, step_hint=None):
    print("\n=== SPECIALIST LATENT VALIDATION ===")
    if step_hint is None:
        step_hint = 99999  # Always log at a high step to avoid WandB step order issues
    all_inputs, all_outputs, sample_keys = [], [], []
    for key in eval_keys[:2]:
        try:
            from re_arc.main import generate_and_process_tasks
            _, _, _, inputs, outputs = generate_and_process_tasks(key, n_samples_per_key)
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            sample_keys.extend([key] * len(inputs))
            print(f"  ✓ Generated {len(inputs)} samples for key '{key}'")
        except Exception as e:
            print(f"  ⚠ Failed to generate data for key {key}: {e}")
    if not all_inputs:
        print("⚠ No data available for latent validation")
        return {'success': False, 'reason': 'no_data'}
    try:
        print(f"  Running latent validation on {len(all_inputs)} samples from {len(set(sample_keys))} keys...")
        correct_recons, swap_results = latent_swap_test(model, all_inputs, all_outputs, device, n_samples=min(5, len(all_inputs)))
        zero_random_results = zero_random_latent_test(model, all_inputs, all_outputs, device, n_samples=min(3, len(all_inputs)))
        latents_pca, latents_tsne, properties, explained_variance = latent_space_visualization(
            model, all_inputs, all_outputs, device, max_samples=min(30, len(all_inputs)), sample_keys=sample_keys
        )
        latent_plots = create_visualization_plots((correct_recons, swap_results), zero_random_results, latents_pca, latents_tsne, properties, explained_variance, run_dir=run_dir)
        if latent_plots:
            print(f"  ✓ Generated plots: {list(latent_plots.keys())}")
            for k, v in latent_plots.items():
                print(f"    - {k}: {v}")
        else:
            print("  ⚠ No plots were generated by latent validation.")
        # Log to wandb if provided
        if wandb_logger:
            import wandb
            log_dict = {
                'latent_validation/n_samples_tested': len(all_inputs),
                'latent_validation/n_swap_tests': len(swap_results),
                'latent_validation/n_zero_random_tests': len(zero_random_results['zero']),
                'latent_validation/latent_space_samples': len(latents_pca) if latents_pca is not None else 0,
                'latent_validation/data_source': f'specialist_latent_validation_keys_{eval_keys[:2]}'
            }
            for plot_name, plot_path in latent_plots.items():
                if os.path.exists(plot_path):
                    log_dict[f'latent_validation/{plot_name}'] = wandb.Image(plot_path)
                    print(f"  ✓ Uploaded {plot_name} to WandB: {plot_path}")
                else:
                    print(f"  ⚠ Plot file missing for WandB upload: {plot_path}")
            wandb_logger._safe_log(log_dict, step_hint=step_hint)
        print("  ✓ Latent validation complete.")
        return {
            'success': True,
            'n_samples_tested': len(all_inputs),
            'n_swap_tests': len(swap_results),
            'n_zero_random_tests': len(zero_random_results['zero']),
            'latent_space_samples': len(latents_pca) if latents_pca is not None else 0,
            'test_keys': eval_keys[:2]
        }
    except Exception as e:
        print(f"⚠ Latent validation failed: {e}")
        import traceback; traceback.print_exc()
        return {'success': False, 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Validate Latent Program Network latent learning')
    parser.add_argument('--file_name', type=str, required=True, help='Model directory name')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch to load')
    parser.add_argument('--settings', type=str, default='model_settings.json', help='Settings file')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples to test')
    args = parser.parse_args()
    
    # Initialize settings and load model
    print(f"Loading settings from: {args.settings}")
    settings = init_settings(args.settings)
    
    data_settings = settings.get_data_settings()
    wandb_settings = settings.get_wandb_settings()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    run_dir = os.path.join(data_settings['run_base_dir'], args.file_name)
    print(f"Loading model from {run_dir}, epoch {args.epoch}")
    
    model, _, _, _ = load_model(run_dir, epoch=args.epoch, device=device)
    model.eval()
    
    # Generate test data with key tracking
    print("Generating test data...")
    eval_keys = data_settings.get('training_keys', ['00d62c1b'])[:2]  # Use first 2 keys
    n_samples_per_key = args.n_samples // len(eval_keys)
    
    all_inputs, all_outputs, sample_keys = [], [], []
    for key in eval_keys:
        try:
            _, _, _, inputs, outputs = generate_and_process_tasks(key, n_samples_per_key)
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            sample_keys.extend([key] * len(inputs))  # Track which key each sample came from
            print(f"Generated {len(inputs)} samples for key '{key}'")
        except Exception as e:
            print(f"Warning: Failed to generate data for key {key}: {e}")
    
    print(f"Generated {len(all_inputs)} test samples from {len(eval_keys)} keys")
    
    # Initialize WandB
    wandb_logger = None
    if wandb_settings.get('enabled', False):
        wandb_logger = init_wandb_for_mode('latent_validation', run_dir)
        if wandb_logger:
            print(f"✓ WandB logging enabled: {wandb_logger.run.name}")
    
    # Run tests
    print("\n=== RUNNING LATENT VALIDATION TESTS ===")
    
    print("1. Latent Swap Test...")
    correct_recons, swap_results = latent_swap_test(model, all_inputs, all_outputs, device, n_samples=5)
    
    print("2. Zero/Random Latent Test...")
    zero_random_results = zero_random_latent_test(model, all_inputs, all_outputs, device, n_samples=3)
    
    print("3. Latent Space Visualization...")
    latents_pca, latents_tsne, properties, explained_variance = latent_space_visualization(
        model, all_inputs, all_outputs, device, max_samples=min(50, len(all_inputs)), sample_keys=sample_keys
    )
    
    # Create visualizations
    print("4. Creating visualization plots...")
    plots = create_visualization_plots((correct_recons, swap_results), zero_random_results, latents_pca, latents_tsne, properties, explained_variance, run_dir=run_dir)
    
    # Log to WandB
    if wandb_logger:
        print("5. Logging results to WandB...")
        
        try:
            import wandb
            log_dict = {
                'latent_validation/epoch_tested': args.epoch,
                'latent_validation/n_samples_tested': len(all_inputs),
                'latent_validation/n_swap_tests': len(swap_results),
                'latent_validation/n_zero_random_tests': len(zero_random_results['zero']),
                'latent_validation/latent_space_samples': len(latents_pca) if latents_pca is not None else 0,
                'latent_validation/data_source': f'validation_epoch_{args.epoch}_keys_{eval_keys}'
            }
            
            # Add plots
            for plot_name, plot_path in plots.items():
                if os.path.exists(plot_path):
                    log_dict[f'latent_validation/{plot_name}'] = wandb.Image(plot_path)
                    print(f"  ✓ Uploaded {plot_name} to WandB: {plot_path}")
                else:
                    print(f"  ⚠ Plot file missing for WandB upload: {plot_path}")
            
            wandb_logger._safe_log(log_dict, step_hint=args.epoch)
            print(f"✓ Results logged to WandB at epoch {args.epoch}")
            
            # Interpretation guide
            interpretation = """
            LATENT VALIDATION INTERPRETATION GUIDE:
            
            🔴 BAD SIGNS (Decoder absorbing all load):
            - Latent swaps produce normal-looking reconstructions
            - Zero/random latents produce reasonable outputs  
            - Latent space shows no semantic clustering
            
            🟢 GOOD SIGNS (Encoder learning meaningful latents):
            - Latent swaps produce novel/different combinations
            - Zero/random latents produce garbage/noise
            - Latent space clusters by semantic properties (size, color, etc.)
            
            📊 WHAT TO LOOK FOR:
            - Swap test: Different from targets = latents matter
            - Zero/random test: Garbage output = latents matter
            - Space visualization: Clustering = meaningful structure
            """
            
            wandb_logger._safe_log({
                'latent_validation/interpretation_guide': interpretation
            }, step_hint=args.epoch)
            
        except Exception as e:
            print(f"⚠ Failed to log to WandB: {e}")
        
        # Clean up temporary files
        # for plot_path in plots.values():
        #     try:
        #         os.unlink(plot_path)
        #     except:
        #         pass
        
        wandb_logger.finish()
    
    print("\n✅ LATENT VALIDATION COMPLETE")
    print(f"Results logged for epoch {args.epoch}")
    print("\nInterpretation:")
    print("- Check WandB for detailed visualizations")
    print("- Look for latent swaps producing novel combinations")
    print("- Zero/random latents should produce garbage if latents matter")
    print("- Latent space should show semantic clustering")


if __name__ == "__main__":
    main() 