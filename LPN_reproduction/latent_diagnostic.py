# File: LPN_reproduction/latent_diagnostics.py

import argparse
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model, set_seed
from utils.settings_manager import settings
from utils.visualizers import load_evaluation_latent_data

def analyze_training_latent_variance(run_dir, save_dir=None):
    """
    Measure variance per dimension over the training μ-cloud to identify active dimensions.
    """
    print("=== TRAINING LATENT VARIANCE ANALYSIS ===")
    
    # Load training latent data (mu vectors from encoder)
    encoded_data = load_evaluation_latent_data(run_dir, return_all_components=True)
    
    if encoded_data is None:
        print("❌ No encoded training latents found. Run evaluation first.")
        return None
    
    train_mu = encoded_data['latent_mus']  # Shape: (n_samples, latent_dim)
    latent_dim = train_mu.shape[1]
    
    print(f"Training data shape: {train_mu.shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # 1. Compute variance per dimension
    var_per_dim = train_mu.var(axis=0)
    
    print(f"\nVariance per dimension:")
    print(f"Min variance: {var_per_dim.min():.6f}")
    print(f"Max variance: {var_per_dim.max():.6f}")
    print(f"Mean variance: {var_per_dim.mean():.6f}")
    print(f"Std variance: {var_per_dim.std():.6f}")
    
    # 2. Identify the highest-variance axes (most active)
    n_active = min(8, latent_dim)  # Look at top 8 dimensions as suggested
    active_dims = np.argsort(var_per_dim)[-n_active:]  # Indices of highest variance dims
    
    print(f"\nTop {n_active} most active dimensions (highest variance):")
    for i, dim_idx in enumerate(active_dims):
        print(f"  Rank {i+1}: Dimension {dim_idx}, Variance: {var_per_dim[dim_idx]:.6f}")
    
    # 3. Create visualization
    if save_dir is None:
        save_dir = run_dir
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot variance per dimension
    ax1.bar(range(latent_dim), var_per_dim)
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Variance')
    ax1.set_title('Variance per Latent Dimension (Training μ-cloud)')
    ax1.grid(True, alpha=0.3)
    
    # Highlight active dimensions
    ax1.bar(active_dims, var_per_dim[active_dims], color='red', alpha=0.7, 
            label=f'Top {n_active} active dims')
    ax1.legend()
    
    # Distribution of variances
    ax2.hist(var_per_dim, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(var_per_dim.mean(), color='red', linestyle='--', label=f'Mean: {var_per_dim.mean():.6f}')
    ax2.set_xlabel('Variance')
    ax2.set_ylabel('Number of Dimensions')
    ax2.set_title('Distribution of Variances Across Dimensions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    variance_plot_path = os.path.join(save_dir, 'latent_variance_analysis.png')
    plt.savefig(variance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Variance analysis plot saved to: {variance_plot_path}")
    
    return {
        'train_mu': train_mu,
        'var_per_dim': var_per_dim,
        'active_dims': active_dims,
        'latent_dim': latent_dim
    }

def analyze_trajectory_on_active_dims(trajectory_info_list, variance_analysis, save_dir):
    """
    Project z-trajectory on the most active dimensions only to see if the jump shrinks.
    """
    print("\n=== TRAJECTORY ANALYSIS ON ACTIVE DIMENSIONS ===")
    
    if variance_analysis is None:
        print("❌ Need variance analysis first")
        return
    
    active_dims = variance_analysis['active_dims']
    print(f"Analyzing trajectories on active dimensions: {active_dims}")
    
    # Analyze each trajectory
    trajectory_jumps_full = []
    trajectory_jumps_active = []
    
    for i, trajectory_info in enumerate(trajectory_info_list):
        z_vectors = trajectory_info.get('z_vectors', [])
        
        if len(z_vectors) < 2:
            continue
            
        z_array = np.array(z_vectors)  # Shape: (n_steps, latent_dim)
        
        # Calculate total movement in full space
        total_movement_full = np.linalg.norm(z_array[-1] - z_array[0])
        
        # Calculate total movement in active dimensions only
        z_active = z_array[:, active_dims]  # Project onto active dims
        total_movement_active = np.linalg.norm(z_active[-1] - z_active[0])
        
        trajectory_jumps_full.append(total_movement_full)
        trajectory_jumps_active.append(total_movement_active)
        
        print(f"Sample {i}: Full space movement: {total_movement_full:.6f}, "
              f"Active dims movement: {total_movement_active:.6f}, "
              f"Ratio: {total_movement_active/total_movement_full:.3f}")
    
    if trajectory_jumps_full:
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sample_indices = range(len(trajectory_jumps_full))
        
        ax1.bar(sample_indices, trajectory_jumps_full, alpha=0.7, label='Full latent space')
        ax1.bar(sample_indices, trajectory_jumps_active, alpha=0.7, label='Active dims only')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Total Movement (L2 norm)')
        ax1.set_title('Trajectory Movement: Full vs Active Dimensions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ratio plot
        ratios = [active/full if full > 0 else 0 for active, full in zip(trajectory_jumps_active, trajectory_jumps_full)]
        ax2.bar(sample_indices, ratios, alpha=0.7, color='green')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Active / Full Movement Ratio')
        ax2.set_title('Movement Ratio (Active Dims / Full Space)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal movement')
        ax2.legend()
        
        plt.tight_layout()
        movement_plot_path = os.path.join(save_dir, 'trajectory_movement_analysis.png')
        plt.savefig(movement_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Movement analysis plot saved to: {movement_plot_path}")
        
        avg_ratio = np.mean(ratios)
        print(f"\nAverage movement ratio (active/full): {avg_ratio:.3f}")
        if avg_ratio < 0.5:
            print("✓ Confirms hypothesis: Most movement is in high-variance dimensions")
        else:
            print("⚠ Movement is more distributed across dimensions")

def analyze_gradient_norms_during_optimization(model, trajectory_info_list, device='cuda', save_dir=None):
    """
    Analyze gradient norms per dimension during optimization to find near-zero gradients.
    """
    print("\n=== GRADIENT NORM ANALYSIS ===")
    
    if not trajectory_info_list:
        print("❌ No trajectory data available")
        return
    
    # Get model settings for latent dimension
    settings_obj = settings
    latent_dim = settings_obj.get_model_architecture()['latent_dim']
    
    print(f"Analyzing gradients for latent dimension: {latent_dim}")
    
    # Track gradients for each dimension across all samples
    gradient_activity = np.zeros(latent_dim)  # Count of non-zero gradients per dimension
    gradient_magnitudes = [[] for _ in range(latent_dim)]  # Store all gradient values per dim
    
    model.eval()
    
    print("Recomputing gradients for trajectory analysis...")
    
    for sample_idx, trajectory_info in enumerate(tqdm(trajectory_info_list, desc="Analyzing gradients")):
        input_sample = torch.tensor(trajectory_info['input_sample']).unsqueeze(0).to(device).float()
        target_sample = torch.tensor(trajectory_info['target_sample']).unsqueeze(0).to(device).float()
        
        # Get initial z from encoder
        with torch.no_grad():
            mu, log_var,_ = model.encoder(input_sample, target_sample)
            z = model.reparameterize(mu, log_var)
        
        # Simulate one optimization step to get gradients
        z = z.detach().requires_grad_(True)
        
        # Forward pass
        shape_logits, grid_logits = model.decoder(z, input_sample, target_seq=target_sample)
        
        # Compute loss
        shape_targets = target_sample[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
        
        # Grid loss
        grid_loss_list = []
        batch_size = input_sample.size(0)
        for i in range(batch_size):
            tgt_rows = int(target_sample[i, 900].item())
            tgt_cols = int(target_sample[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits[i, :active_pixels],
                                       target_sample[i, :active_pixels].long())
                grid_loss_list.append(loss_i)

        grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else \
                   torch.tensor(0.0, device=input_sample.device, requires_grad=True)
        
        reconstruction_loss = shape_loss + grid_loss
        
        # Backward pass to get gradients
        reconstruction_loss.backward()
        
        # Analyze gradients
        if z.grad is not None:
            grad_z = z.grad[0]  # Remove batch dimension
            
            # Check which dimensions have non-zero gradients
            for d in range(latent_dim):
                if abs(grad_z[d].item()) > 1e-8:  # Non-zero threshold
                    gradient_activity[d] += 1
                gradient_magnitudes[d].append(abs(grad_z[d].item()))
    
    # Calculate statistics
    total_samples = len(trajectory_info_list)
    gradient_activity_ratio = gradient_activity / total_samples
    
    print(f"\nGradient activity per dimension (ratio of samples with non-zero gradients):")
    for d in range(latent_dim):
        avg_magnitude = np.mean(gradient_magnitudes[d]) if gradient_magnitudes[d] else 0
        print(f"  Dim {d:3d}: {gradient_activity_ratio[d]:.3f} "
              f"(avg magnitude: {avg_magnitude:.6f})")
    
    # Find dimensions with very low gradient activity
    low_activity_dims = np.where(gradient_activity_ratio < 0.1)[0]  # Less than 10% activity
    high_activity_dims = np.where(gradient_activity_ratio > 0.9)[0]  # More than 90% activity
    
    print(f"\nDimensions with low gradient activity (<10%): {low_activity_dims}")
    print(f"Dimensions with high gradient activity (>90%): {high_activity_dims}")
    print(f"Proportion of low-activity dimensions: {len(low_activity_dims)/latent_dim:.3f}")
    
    # Create visualization
    if save_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gradient activity ratio
        ax1.bar(range(latent_dim), gradient_activity_ratio)
        ax1.set_xlabel('Latent Dimension')
        ax1.set_ylabel('Gradient Activity Ratio')
        ax1.set_title('Gradient Activity per Dimension\n(Fraction of samples with non-zero gradients)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Low activity threshold')
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='High activity threshold')
        ax1.legend()
        
        # Average gradient magnitudes
        avg_magnitudes = [np.mean(mags) if mags else 0 for mags in gradient_magnitudes]
        ax2.bar(range(latent_dim), avg_magnitudes)
        ax2.set_xlabel('Latent Dimension')
        ax2.set_ylabel('Average Gradient Magnitude')
        ax2.set_title('Average Gradient Magnitude per Dimension')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        gradient_plot_path = os.path.join(save_dir, 'gradient_analysis.png')
        plt.savefig(gradient_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gradient analysis plot saved to: {gradient_plot_path}")
    
    return {
        'gradient_activity_ratio': gradient_activity_ratio,
        'low_activity_dims': low_activity_dims,
        'high_activity_dims': high_activity_dims,
        'avg_magnitudes': avg_magnitudes
    }

def run_comprehensive_diagnostics(run_dir, key, save_dir=None):
    """
    Run all diagnostic analyses to confirm the variance and gradient issues.
    """
    print("=" * 60)
    print("COMPREHENSIVE LATENT SPACE DIAGNOSTICS")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = run_dir
    
    # Load model and evaluation results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, _, _, _ = load_model(run_dir, device=device)
    
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if not os.path.exists(eval_file):
        print(f"❌ No evaluation results found at {eval_file}")
        return
    
    with open(eval_file, 'rb') as f:
        eval_results = pickle.load(f)
    
    if key not in eval_results:
        print(f"❌ Key '{key}' not found in evaluation results")
        print(f"Available keys: {list(eval_results.keys())}")
        return
    
    # Get trajectory info
    key_results = eval_results[key]
    trajectory_info_list = []
    
    if 'metrics' in key_results and 'trajectory_info' in key_results['metrics']:
        trajectory_info_list = key_results['metrics']['trajectory_info']
    elif 'trajectory_info' in key_results:
        trajectory_info_list = key_results['trajectory_info']
    
    if not trajectory_info_list:
        print(f"❌ No trajectory information found for key '{key}'")
        return
    
    print(f"Found {len(trajectory_info_list)} trajectory samples to analyze")
    
    # 1. Analyze training latent variance
    variance_analysis = analyze_training_latent_variance(run_dir, save_dir)
    
    # 2. Analyze trajectory movement on active dimensions
    if variance_analysis:
        analyze_trajectory_on_active_dims(trajectory_info_list, variance_analysis, save_dir)
    
    # 3. Analyze gradient norms
    gradient_analysis = analyze_gradient_norms_during_optimization(
        model, trajectory_info_list, device, save_dir
    )
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if variance_analysis:
        var_per_dim = variance_analysis['var_per_dim']
        active_dims = variance_analysis['active_dims']
        
        print(f"✓ Latent dimension: {variance_analysis['latent_dim']}")
        print(f"✓ Variance range: {var_per_dim.min():.6f} - {var_per_dim.max():.6f}")
        print(f"✓ Most active dimensions: {active_dims}")
        print(f"✓ Variance concentration: Top 8 dims have {(var_per_dim[active_dims].sum() / var_per_dim.sum()):.3f} of total variance")
    
    if gradient_analysis:
        low_activity_ratio = len(gradient_analysis['low_activity_dims']) / len(gradient_analysis['gradient_activity_ratio'])
        print(f"✓ Low gradient activity dimensions: {low_activity_ratio:.3f} of total")
        print(f"✓ High gradient activity dimensions: {len(gradient_analysis['high_activity_dims'])}")
    
    print(f"\n✓ All diagnostic plots saved to: {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run latent space diagnostics')
    parser.add_argument('--run_dir', type=str, required=True,
                      help='Directory containing the model and evaluation results')
    parser.add_argument('--key', type=str, required=True,
                      help='Problem key to analyze')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save diagnostic plots (defaults to run_dir)')
    return parser.parse_args()

def main():
    """
    Example command: python LPN_reproduction/latent_diagnostics.py --run_dir runs_re_arc/test_2s --key pattern_task
    """
    args = parse_args()
    run_comprehensive_diagnostics(args.run_dir, args.key, args.save_dir)

if __name__ == "__main__":
    main()