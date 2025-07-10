#!/usr/bin/env python3

"""
Trajectory Evaluation and Visualization

This module provides comprehensive trajectory visualization for latent space optimization.
It supports both single-encoder and multi-encoder models with the following features:

DECODER TYPE SELECTION:
The 'trajectory_decoder_type' setting in evaluation_settings controls which decoder
is used for trajectory reconstructions:
  - "shared": Use the shared decoder (default for PoE models)
  - "independent": Use individual encoder's independent decoders
  
This setting affects:
  - Which reconstructions are stored during trajectory evaluation
  - How reconstructions are displayed in trajectory plots
  - The title information shown in visualizations

VISUALIZATION FEATURES:
  - Input/Target grids
  - Latent space trajectories with consistent t-SNE coordinates
  - Reconstruction quality at different optimization steps
  - Error maps showing prediction vs target differences
  - Loss progression during optimization
  - Multi-encoder: Individual encoder vs PoE comparisons
  - Single-encoder: Standard trajectory analysis
"""

import sys
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import argparse
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import LatentProgramNetwork, compute_encoder_influence_metrics
from utils.settings_manager import settings
from utils.model_utils import load_model, set_seed
from utils.visualizers import load_evaluation_latent_data, get_comprehensive_latent_data_for_trajectory
from utils.data_preparation import extract_grid_from_sequence, safe_extract_reconstruction_grid

def print_trajectory_info(trajectory_info):
    """
    Print the structure of trajectory information in a readable format.
    """
    if not trajectory_info:
        print("No trajectory information available.")
        return
    
    print("\nTrajectory Information Structure:")
    print("=" * 50)
    
    # Print first sample's information
    sample = trajectory_info[0]
    print("\nSample 0:")
    print("-" * 30)
    
    # Print shapes and types of each component
    print("input_sample:")
    print(f"  Shape: {sample['input_sample'].shape}")
    print(f"  Type: {sample['input_sample'].dtype}")
    
    print("\ntarget_sample:")
    print(f"  Shape: {sample['target_sample'].shape}")
    print(f"  Type: {sample['target_sample'].dtype}")
    
    print("\nz_vectors:")
    if sample['z_vectors']:
        print(f"  Number of z vectors: {len(sample['z_vectors'])}")
        print(f"  Shape of each z vector: {sample['z_vectors'][0].shape}")
        print(f"  Type: {sample['z_vectors'][0].dtype}")
        print(f"  Note: Includes initial z (step 0) and optimization steps")
    else:
        print("  No z vectors available")
    
    print("\nlosses:")
    print(f"  Type: {type(sample['losses'])}")
    if isinstance(sample['losses'], list):
        print(f"  Number of loss values: {len(sample['losses'])}")
        print(f"  Loss values: {sample['losses'][:5]}..." if len(sample['losses']) > 5 else f"  Loss values: {sample['losses']}")
        if len(sample['losses']) > 0:
            print(f"  Initial loss (step 0): {sample['losses'][0]:.4f}")
            if len(sample['losses']) > 1:
                print(f"  Final loss: {sample['losses'][-1]:.4f}")
                print(f"  Loss improvement: {sample['losses'][0] - sample['losses'][-1]:+.4f}")
    else:
        print(f"  Value: {sample['losses']}")
    
    # Print encoder information if available (equivalent to training data)
    print("\nEncoder Information (equivalent to training data):")
    if sample.get('encoder_mu') is not None:
        print(f"  encoder_mu shape: {sample['encoder_mu'].shape}")
        print(f"  encoder_log_var shape: {sample.get('encoder_log_var', 'None')}")
        print(f"  initial_z shape: {sample.get('initial_z', 'None')}")
    else:
        print("  No encoder information stored")
    
    print(f"\nTotal number of samples: {len(trajectory_info)}")
    print("Note: Reconstructions are computed on-demand during visualization using the model decoder.")
    print("Note: Trajectory now includes initial step (step 0) with encoder output and initial loss.")


def load_training_latent_data(run_dir):
    """
    Load comprehensive latent data (encoders + PoE) for trajectory visualization background.
    This reuses the comprehensive latent space that combines all encoder and PoE latents.
    Returns both high-dimensional data and precomputed t-SNE coordinates for consistency.
    """
    print("Loading comprehensive latent data (encoders + PoE) for background visualization...")
    
    # Import the comprehensive latent data function from visualizers
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from utils.visualizers import get_comprehensive_latent_data_for_trajectory
        
        combined_latents, tsne_2d, labels, colors = get_comprehensive_latent_data_for_trajectory(run_dir)
        
        if combined_latents is not None:
            print(f"✓ Successfully loaded comprehensive latent data:")
            print(f"  - Total samples: {len(combined_latents)}")
            print(f"  - Latent dimensionality: {combined_latents.shape[1]}")
            print(f"  - t-SNE 2D coordinates: {tsne_2d.shape}")
            print(f"  - Data types: {len(set(labels))}")
            print(f"  - Includes: Individual encoders + PoE latents")
            
            # Return both high-dimensional data and precomputed t-SNE coordinates
            return combined_latents, tsne_2d, labels, colors
        else:
            print("⚠ Warning: No comprehensive latent data available")
            return None, None, None, None
            
    except ImportError as e:
        print(f"⚠ Warning: Could not import comprehensive latent data function: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"⚠ Warning: Could not load comprehensive latent data: {e}")
        return None, None, None, None

def visualize_input_target(trajectory_info, save_path):
    """
    Create a visualization showing input and target grids.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Input sample (using visualizers.py approach)
    input_seq = trajectory_info['input_sample']
    input_grid, input_shape = extract_grid_from_sequence(input_seq)
    ax1.imshow(input_grid, cmap='viridis')
    ax1.set_title(f'Input\n{input_shape[0]}×{input_shape[1]}')
    ax1.axis('off')
    
    # Right: Target sample
    target_seq = trajectory_info['target_sample']
    target_grid, target_shape = extract_grid_from_sequence(target_seq)
    ax2.imshow(target_grid, cmap='viridis')
    ax2.set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def find_trajectory_points_in_tsne_space(z_vectors, training_latent_data, training_tsne_2d, training_labels):
    """
    Map trajectory z vectors to existing t-SNE space by finding nearest neighbors
    in the high-dimensional space and using their 2D coordinates.
    """
    if training_latent_data is None or training_tsne_2d is None or not z_vectors:
        return None
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    
    # Normalize both training data and trajectory vectors using the same scaler
    scaler = StandardScaler()
    training_normalized = scaler.fit_transform(training_latent_data)
    
    z_array = np.array(z_vectors)
    z_normalized = scaler.transform(z_array)
    
    # Find nearest neighbors in normalized high-dimensional space
    n_neighbors = min(3, len(training_latent_data))  # Use 3 neighbors for interpolation
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(training_normalized)
    distances, indices = nbrs.kneighbors(z_normalized)
    
    # Map to 2D using weighted average of nearest neighbors
    trajectory_2d = []
    for i in range(len(z_vectors)):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        
        # Use inverse distance weighting (avoid division by zero)
        weights = 1.0 / (neighbor_distances + 1e-8)
        weights = weights / weights.sum()
        
        # Weighted average of neighbor 2D coordinates
        weighted_2d = np.average(training_tsne_2d[neighbor_indices], weights=weights, axis=0)
        trajectory_2d.append(weighted_2d)
    
    return np.array(trajectory_2d)

def visualize_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device='cuda'):
    """
    Create a comprehensive trajectory visualization.
    Automatically detects single-encoder vs multi-encoder and calls appropriate function.
    """
    # Check if this is a multi-encoder model
    is_multi_encoder = trajectory_info.get('is_multi_encoder', False)
    
    # Get trajectory decoder type setting (using global settings import)
    eval_settings = settings.get_evaluation_settings()
    decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
    print(f"Using trajectory decoder type: {decoder_type}")
    
    if is_multi_encoder:
        print("Multi-encoder model detected - using enhanced visualization")
        return visualize_multi_encoder_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device)
    
    print("Single-encoder model detected - using standard visualization")
    
    # Load training latent data for background with precomputed t-SNE
    print("Loading training latent data for background visualization...")
    training_latent_data, training_tsne_2d, training_labels, training_colors = load_training_latent_data(run_dir)
    
    # Create figure
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # Extract data from trajectory_info
    input_seq = trajectory_info['input_sample']
    target_seq = trajectory_info['target_sample']
    z_vectors = trajectory_info['z_vectors']
    losses = trajectory_info['losses']
    
    # Input and Target grids
    input_grid, input_shape = extract_grid_from_sequence(input_seq)
    target_grid, target_shape = extract_grid_from_sequence(target_seq)
    
    axs[0, 0].imshow(input_grid, cmap='viridis')
    axs[0, 0].set_title(f'Input\n{input_shape[0]}×{input_shape[1]}')
    axs[0, 0].axis('off')
    
    axs[1, 0].imshow(target_grid, cmap='viridis')
    axs[1, 0].set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
    axs[1, 0].axis('off')
    
    # Plot trajectory in latent space using precomputed t-SNE coordinates
    if z_vectors and len(z_vectors) >= 2:
        # Use precomputed t-SNE coordinates for consistency
        if training_latent_data is not None and training_tsne_2d is not None:
            # Plot training background using precomputed coordinates
            if training_colors is not None and training_labels is not None:
                # Use the existing colors from comprehensive latent data
                unique_labels = list(set(training_labels))
                for label in unique_labels:
                    indices = [i for i, l in enumerate(training_labels) if l == label]
                    if indices:
                        x_coords = training_tsne_2d[indices, 0]
                        y_coords = training_tsne_2d[indices, 1]
                        color = training_colors[indices[0]]  # Get color for this class
                        
                        # Use alpha for background effect
                        axs[0, 1].scatter(x_coords, y_coords, c=color, alpha=0.3, s=12, 
                                       edgecolors='none', label=f'Training {label.replace("training_enc_", "Enc ")}')
            else:
                # Fallback to gray
                axs[0, 1].scatter(training_tsne_2d[:, 0], training_tsne_2d[:, 1],
                                   c='lightgray', alpha=0.4, s=15, edgecolors='none', 
                                   label='Training Data')
            
            # Map trajectory points to the same t-SNE space
            z_2d = find_trajectory_points_in_tsne_space(z_vectors, training_latent_data, training_tsne_2d, training_labels)
            
            if z_2d is not None:
                # Plot trajectory
                trajectory_scatter = axs[0, 1].scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                                     s=80, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Draw arrows between consecutive trajectory points
                for i in range(len(z_2d) - 1):
                    axs[0, 1].annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=2))
                
                # Mark start and end points
                axs[0, 1].scatter(z_2d[0, 0], z_2d[0, 1], c='green', s=150, marker='o', 
                                label='Start', edgecolors='black', linewidth=2, zorder=10)
                axs[0, 1].scatter(z_2d[-1, 0], z_2d[-1, 1], c='red', s=150, marker='s', 
                                label='End', edgecolors='black', linewidth=2, zorder=10)
                
                # Add colorbar
                cbar = plt.colorbar(trajectory_scatter, ax=axs[0, 1], shrink=0.8)
                cbar.set_label('Loss', rotation=270, labelpad=20)
                
                axs[0, 1].set_title('Latent Trajectory (Consistent t-SNE)')
                axs[0, 1].legend()
                axs[0, 1].grid(True, alpha=0.3)
            else:
                axs[0, 1].text(0.5, 0.5, 'Could not map\ntrajectory to t-SNE', 
                              ha='center', va='center', transform=axs[0, 1].transAxes)
                axs[0, 1].set_title('Latent Trajectory (Error)')
        else:
            # Fallback: use independent t-SNE (will be inconsistent)
            z_array = np.array(z_vectors)
            scaler = StandardScaler()
            z_array_normalized = scaler.fit_transform(z_array)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_array)-1), n_iter=1000)
            z_2d = tsne.fit_transform(z_array_normalized)
            
            trajectory_scatter = axs[0, 1].scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                                 s=80, alpha=0.9, edgecolors='black', linewidth=1)
            axs[0, 1].set_title('Latent Trajectory (Independent t-SNE)')
    else:
        axs[0, 1].text(0.5, 0.5, 'No trajectory data', 
                      ha='center', va='center', transform=axs[0, 1].transAxes)
        axs[0, 1].set_title('Latent Trajectory')
    
    # Use safe extraction to avoid scalar conversion errors
    extract_reconstruction_grid = safe_extract_reconstruction_grid

    # Use stored reconstructions for different trajectory steps
    trajectory_reconstructions = trajectory_info.get('poe_trajectory_reconstructions', {})
    if trajectory_reconstructions:
        reconstruction_labels = ['initial', 'middle', 'final']
        display_labels = ['Start', 'Mid', 'End']
        
        for i, (recon_label, display_label) in enumerate(zip(reconstruction_labels, display_labels)):
            if i < 3:  # Only 3 slots available
                row = 0 if i < 2 else 1
                col = 2 + (i % 2)
                
                if recon_label in trajectory_reconstructions and trajectory_reconstructions[recon_label] is not None:
                    recon_data = trajectory_reconstructions[recon_label]
                    shape_logits = recon_data['shape_logits']
                    grid_logits = recon_data['grid_logits']
                    
                    # Extract reconstruction grid from stored data
                    recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                    
                    if recon_grid is not None:
                        axs[row, col].imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        axs[row, col].set_title(f'Reconstruction {display_label}\n{recon_rows}×{recon_cols}')
                    else:
                        axs[row, col].text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                         transform=axs[row, col].transAxes, fontsize=8)
                        axs[row, col].set_title(f'Reconstruction {display_label}')
                else:
                    axs[row, col].text(0.5, 0.5, f'No Data\n{display_label}', ha='center', va='center', 
                                     transform=axs[row, col].transAxes, fontsize=8)
                    axs[row, col].set_title(f'Reconstruction {display_label}')
                
                axs[row, col].axis('off')
    else:
        # Fallback message for missing reconstruction data
        for i in range(3):
            row = 0 if i < 2 else 1
            col = 2 + (i % 2)
            axs[row, col].text(0.5, 0.5, 'No Stored\nReconstruction\nData', 
                             ha='center', va='center', transform=axs[row, col].transAxes)
            axs[row, col].set_title(f'Reconstruction {i}')
            axs[row, col].axis('off')
    
    # Loss progression plot
    if losses and len(losses) > 1:
        axs[1, 1].plot(losses, 'b-o', linewidth=2, markersize=4)
        axs[1, 1].set_title('Loss Progression')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].grid(True, alpha=0.3)
    else:
        axs[1, 1].text(0.5, 0.5, 'No loss data', 
                      ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Loss Progression')
    
    # Get decoder type for title (using global settings import)
    eval_settings = settings.get_evaluation_settings()
    decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
    decoder_display = "Independent Decoders" if decoder_type == "independent" else "Shared Decoder"
    
    plt.suptitle(f'Single-Encoder Trajectory Analysis\nEVALUATION DATA - Trajectory Reconstructions: {decoder_display}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_all_samples_comprehensive(trajectory_info_list, model, save_path, run_dir, device='cuda'):
    """
    Visualize all samples in a comprehensive way, handling both single and multi-encoder models.
    """
    if not trajectory_info_list:
        print("No trajectory information to visualize")
        return
    
    # Check if this is a multi-encoder model
    is_multi_encoder = trajectory_info_list[0].get('is_multi_encoder', False)
    num_encoders = trajectory_info_list[0].get('num_encoders', 1) if is_multi_encoder else 1
    
    print(f"Creating comprehensive visualization for {len(trajectory_info_list)} samples")
    if is_multi_encoder:
        print(f"Multi-encoder model detected with {num_encoders} encoders")
    else:
        print("Single-encoder model detected")
    
    # Load training latent data for background with precomputed t-SNE
    print("Loading training latent data for background...")
    training_latent_data, training_tsne_2d, training_labels, training_colors = load_training_latent_data(run_dir)
    
    # Create figure with expanded layout for multi-encoder
    if is_multi_encoder:
        # For multi-encoder, use individual visualization for each sample
        for sample_idx, trajectory_info in enumerate(trajectory_info_list):
            individual_save_path = save_path.replace('.png', f'_sample_{sample_idx}.png')
            print(f"Creating individual visualization for sample {sample_idx + 1}")
            visualize_multi_encoder_comprehensive_trajectory(trajectory_info, model, individual_save_path, run_dir, device)
        print(f"Created {len(trajectory_info_list)} individual multi-encoder visualizations")
        return
        
    # Single encoder layout - simpler
    fig = plt.figure(figsize=(20, 6 * len(trajectory_info_list)))
    rows_per_sample = 1
    cols = 6  # Input | Target | Latent | 3 Reconstructions
    
    total_rows = rows_per_sample * len(trajectory_info_list)
    
    for sample_idx, trajectory_info in enumerate(trajectory_info_list):
        print(f"Processing sample {sample_idx + 1}/{len(trajectory_info_list)}")
        
        # Calculate row indices for this sample
        start_row = sample_idx * rows_per_sample
        
        # Single encoder layout - simpler
        gs = fig.add_gridspec(total_rows, cols, height_ratios=[1] * total_rows)
        
        # Input, Target, Latent trajectory, 3 reconstructions
        ax_input = fig.add_subplot(gs[start_row, 0])
        ax_target = fig.add_subplot(gs[start_row, 1])
        ax_latent = fig.add_subplot(gs[start_row, 2])
        
        input_grid, input_shape = extract_grid_from_sequence(trajectory_info['input_sample'])
        target_grid, target_shape = extract_grid_from_sequence(trajectory_info['target_sample'])
        
        ax_input.imshow(input_grid, cmap='viridis')
        ax_input.set_title(f'Sample {sample_idx + 1} Input\n{input_shape[0]}×{input_shape[1]}')
        ax_input.axis('off')
        
        ax_target.imshow(target_grid, cmap='viridis')
        ax_target.set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
        ax_target.axis('off')
        
        # Plot latent trajectory using consistent t-SNE
        z_vectors = trajectory_info['z_vectors']
        losses = trajectory_info['losses']
        
        if z_vectors and len(z_vectors) >= 2:
            if training_latent_data is not None and training_tsne_2d is not None:
                # Plot training background using precomputed coordinates
                ax_latent.scatter(training_tsne_2d[:, 0], training_tsne_2d[:, 1],
                                c='lightgray', alpha=0.3, s=10, edgecolors='none')
                
                # Map trajectory to consistent t-SNE space
                z_2d = find_trajectory_points_in_tsne_space(z_vectors, training_latent_data, training_tsne_2d, training_labels)
                
                if z_2d is not None:
                    ax_latent.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                    s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
                    
                    for i in range(len(z_2d) - 1):
                        ax_latent.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1))
                    
                    ax_latent.set_title('Latent Trajectory (Consistent)')
                else:
                    ax_latent.text(0.5, 0.5, 'Mapping Error', 
                                 ha='center', va='center', transform=ax_latent.transAxes)
                    ax_latent.set_title('Latent Trajectory (Error)')
            else:
                ax_latent.text(0.5, 0.5, 'No background data', 
                             ha='center', va='center', transform=ax_latent.transAxes)
                ax_latent.set_title('Latent Trajectory')
            
            ax_latent.grid(True, alpha=0.3)
        else:
            ax_latent.text(0.5, 0.5, 'No trajectory', 
                         ha='center', va='center', transform=ax_latent.transAxes)
            ax_latent.set_title('Latent Trajectory')
        
        # Use safe extraction to avoid scalar conversion errors
        extract_reconstruction_grid = safe_extract_reconstruction_grid

        # Use stored reconstructions for different trajectory steps
        trajectory_reconstructions = trajectory_info.get('poe_trajectory_reconstructions', {})
        if trajectory_reconstructions:
            reconstruction_labels = ['initial', 'middle', 'final']
            display_labels = ['Start', 'Mid', 'End']
            
            for i, (recon_label, display_label) in enumerate(zip(reconstruction_labels, display_labels)):
                if i < 3:  # Columns 3-5
                    ax_recon = fig.add_subplot(gs[start_row, 3 + i])
                    
                    if recon_label in trajectory_reconstructions and trajectory_reconstructions[recon_label] is not None:
                        recon_data = trajectory_reconstructions[recon_label]
                        shape_logits = recon_data['shape_logits']
                        grid_logits = recon_data['grid_logits']
                        
                        # Extract reconstruction grid from stored data
                        recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                        
                        if recon_grid is not None:
                            ax_recon.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                            ax_recon.set_title(f'{display_label}\n{recon_rows}×{recon_cols}')
                        else:
                            ax_recon.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                        transform=ax_recon.transAxes, fontsize=8)
                            ax_recon.set_title(f'{display_label}')
                    else:
                        ax_recon.text(0.5, 0.5, f'No Data\n{display_label}', ha='center', va='center', 
                                    transform=ax_recon.transAxes, fontsize=8)
                        ax_recon.set_title(f'{display_label}')
                    
                    ax_recon.axis('off')
        else:
            # Fallback message for missing reconstruction data
            for i in range(3):
                ax_recon = fig.add_subplot(gs[start_row, 3 + i])
                ax_recon.text(0.5, 0.5, 'No Stored\nReconstruction\nData', 
                            ha='center', va='center', transform=ax_recon.transAxes)
                ax_recon.set_title(f'Reconstruction {i}')
                ax_recon.axis('off')
    
    # Get decoder type for title (using global settings import)
    eval_settings = settings.get_evaluation_settings()
    decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
    decoder_display = "Independent Decoders" if decoder_type == "independent" else "Shared Decoder"
    
    title = f'Single-Encoder Trajectory Analysis - All Samples\nEVALUATION DATA - Trajectory Reconstructions: {decoder_display}'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to {save_path}")



def visualize_multi_encoder_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device='cuda'):
    """
    Create a comprehensive visualization for multi-encoder models showing:
    - Input/Target grids
    - Individual encoder latent trajectories
    - PoE latent trajectory 
    - Individual encoder reconstructions (from stored data)
    - PoE reconstructions (from stored data)
    All with training background in latent space using consistent t-SNE coordinates.
    """
    if not trajectory_info.get('is_multi_encoder', False):
        print("Warning: This function is for multi-encoder models only.")
        return visualize_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device)
    
    num_encoders = trajectory_info.get('num_encoders', 1)
    print(f"Creating multi-encoder visualization for {num_encoders} encoders...")
    
    # Get trajectory decoder type setting (using global settings import)
    eval_settings = settings.get_evaluation_settings()
    decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
    decoder_display = "Independent Decoders" if decoder_type == "independent" else "Shared Decoder"
    print(f"Trajectory used: {decoder_display}")

    # Load training latent data for background with precomputed t-SNE
    print("Loading training latent data for background visualization...")
    training_latent_data, training_tsne_2d, training_labels, training_colors = load_training_latent_data(run_dir)

    # Create figure with optimized layout
    fig = plt.figure(figsize=(20, 15))
    
    # Define grid layout: 3 rows, 8 columns
    # Row 0: Input | Merged Latent Space (5 cols) | Loss Plot (2 cols)
    # Row 1: Target | Individual Encoder Reconstructions (4 cols) | PoE Reconstructions (3 cols)
    # Row 2: Error | Individual Encoder Error Maps (4 cols) | PoE Error Maps (3 cols)
    gs = fig.add_gridspec(4, 8, width_ratios=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1], height_ratios=[1, 1.5, 1.5, 1.5])
    
    # Input and Target (left column)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[1, 0])
    ax_error_label = fig.add_subplot(gs[2, 0])
    
    input_grid, input_shape = extract_grid_from_sequence(trajectory_info['input_sample'])
    ax_input.imshow(input_grid, cmap='viridis', interpolation='nearest', aspect='equal')
    ax_input.set_title(f'Input\n{input_shape[0]}×{input_shape[1]}', fontsize=12, fontweight='bold')
    ax_input.axis('off')
    
    target_grid, target_shape = extract_grid_from_sequence(trajectory_info['target_sample'])
    ax_target.imshow(target_grid, cmap='viridis', interpolation='nearest', aspect='equal')
    ax_target.set_title(f'Target\n{target_shape[0]}×{target_shape[1]}', fontsize=12, fontweight='bold')
    ax_target.axis('off')
    
    ax_error_label.text(0.5, 0.5, 'Error Maps\n(Pred - Target)', ha='center', va='center', 
                       transform=ax_error_label.transAxes, fontsize=10, fontweight='bold')
    ax_error_label.axis('off')
    
    # Merged latent space visualization using consistent t-SNE
    ax_merged_latent = fig.add_subplot(gs[0, 1:6])
    ax_loss = fig.add_subplot(gs[0, 6:8])
    
    individual_trajectories = trajectory_info.get('individual_encoder_trajectories', {})
    z_vectors = trajectory_info['z_vectors']
    losses = trajectory_info['losses']
    
    if training_latent_data is not None and training_tsne_2d is not None:
        # Plot training background using precomputed coordinates
        if training_colors is not None and training_labels is not None:
            # Use the existing colors from comprehensive latent data
            unique_labels = list(set(training_labels))
            for label in unique_labels:
                indices = [i for i, l in enumerate(training_labels) if l == label]
                if indices:
                    x_coords = training_tsne_2d[indices, 0]
                    y_coords = training_tsne_2d[indices, 1]
                    color = training_colors[indices[0]]  # Get color for this class
                    
                    # Use alpha for background effect
                    ax_merged_latent.scatter(x_coords, y_coords, c=color, alpha=0.3, s=12, 
                                           edgecolors='none', label=f'Training {label.replace("training_enc_", "Enc ")}')
        else:
            # Fallback to gray
            ax_merged_latent.scatter(training_tsne_2d[:, 0], training_tsne_2d[:, 1],
                                   c='lightgray', alpha=0.4, s=15, edgecolors='none', 
                                   label='Training Data')
        
        # Plot PoE trajectory using consistent mapping
        if z_vectors and len(z_vectors) >= 2:
            z_2d = find_trajectory_points_in_tsne_space(z_vectors, training_latent_data, training_tsne_2d, training_labels)
            
            if z_2d is not None:
                # Draw trajectory path
                ax_merged_latent.plot(z_2d[:, 0], z_2d[:, 1], 'k-', alpha=0.7, linewidth=2, 
                                    label='PoE Trajectory Path')
                
                # Plot trajectory points colored by loss
                trajectory_scatter = ax_merged_latent.scatter(z_2d[:, 0], z_2d[:, 1], 
                                                            c=losses, cmap='plasma', 
                                                            s=80, alpha=0.9, 
                                                            edgecolors='black', linewidth=1,
                                                            zorder=8)
                
                # Draw arrows between consecutive trajectory points
                for i in range(len(z_2d) - 1):
                    ax_merged_latent.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                            arrowprops=dict(arrowstyle='->', color='darkred', 
                                                          alpha=0.8, lw=2, zorder=9))
                
                # Mark start and end points
                ax_merged_latent.scatter(z_2d[0, 0], z_2d[0, 1], c='green', s=200, marker='*', 
                                       label='PoE Start', edgecolors='black', linewidth=2, zorder=11)
                ax_merged_latent.scatter(z_2d[-1, 0], z_2d[-1, 1], c='red', s=200, marker='X', 
                                       label='PoE End', edgecolors='black', linewidth=2, zorder=11)
        
        ax_merged_latent.set_title('Consistent Latent Space:\nTraining Data + PoE Trajectory Only', fontsize=12)
        ax_merged_latent.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, 
                               shadow=True, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        ax_merged_latent.grid(True, alpha=0.3)
    else:
        ax_merged_latent.text(0.5, 0.5, 'No precomputed\nt-SNE data available', 
                            ha='center', va='center', transform=ax_merged_latent.transAxes)
        ax_merged_latent.set_title('Consistent Latent Space (No Data)')
    
    # Plot optimization losses
    if losses and len(losses) > 1:
        steps = list(range(len(losses)))
        ax_loss.plot(steps, losses, 'b-', marker='o', linewidth=2, markersize=4)
        ax_loss.set_title('Optimization Losses', fontsize=12)
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        
        # Add start/end annotations
        ax_loss.annotate(f'Start: {losses[0]:.3f}', xy=(0, losses[0]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax_loss.annotate(f'End: {losses[-1]:.3f}', xy=(len(losses)-1, losses[-1]), 
                       xytext=(5, -15), textcoords='offset points', fontsize=8)
    else:
        ax_loss.text(0.5, 0.5, 'No Loss\nData', 
                    ha='center', va='center', transform=ax_loss.transAxes)
        ax_loss.set_title('Optimization Losses')
    
    # Use safe extraction to avoid scalar conversion errors
    extract_reconstruction_grid = safe_extract_reconstruction_grid
    
    # Individual encoder reconstructions (row 1, columns 1-4) - USE STORED DATA
    individual_reconstructions = trajectory_info.get('individual_encoder_reconstructions', {})
    for enc_idx in range(min(num_encoders, 4)):  # Limit to 4 encoders for layout
        ax_enc_recon = fig.add_subplot(gs[1, enc_idx + 1])
        ax_enc_error = fig.add_subplot(gs[2, enc_idx + 1])
        
        encoder_key = f'encoder_{enc_idx}'
        if encoder_key in individual_reconstructions and individual_reconstructions[encoder_key] is not None:
            enc_recon_data = individual_reconstructions[encoder_key]
            shape_logits = enc_recon_data['shape_logits']
            grid_logits = enc_recon_data['grid_logits']
            
            # Extract reconstruction grid from stored data
            recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
            
            if recon_grid is not None:
                # Plot reconstruction
                ax_enc_recon.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                ax_enc_recon.set_title(f'Encoder {enc_idx}\n{recon_rows}×{recon_cols}', fontsize=10)
                
                # === NEW: print latent μ / σ averages under the image ===
                enc_traj = individual_trajectories.get(encoder_key, {}) if 'individual_trajectories' in locals() else {}
                enc_mu = enc_traj.get('mu')
                enc_log_var = enc_traj.get('log_var')
                if enc_mu is not None and enc_log_var is not None:
                    enc_mu_mean = float(np.mean(enc_mu))
                    enc_sigma_mean = float(np.mean(np.exp(0.5 * enc_log_var)))
                    ax_enc_recon.text(0.5, -0.12, f"μ={enc_mu_mean:.2f} σ={enc_sigma_mean:.2f}",
                                      transform=ax_enc_recon.transAxes, ha='center', va='top', fontsize=8)
                
                # Calculate and plot error map
                if recon_rows == target_shape[0] and recon_cols == target_shape[1]:
                    error_map = recon_grid.astype(float) - target_grid.astype(float)
                    im_error = ax_enc_error.imshow(error_map, cmap='RdBu', vmin=-5, vmax=5, 
                                                 interpolation='nearest', aspect='equal')
                    ax_enc_error.set_title(f'Error {enc_idx}', fontsize=10)
                else:
                    ax_enc_error.text(0.5, 0.5, 'Size\nMismatch', ha='center', va='center', 
                                    transform=ax_enc_error.transAxes, fontsize=8)
                    ax_enc_error.set_title(f'Error {enc_idx}', fontsize=10)
            else:
                ax_enc_recon.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                transform=ax_enc_recon.transAxes, fontsize=8)
                ax_enc_recon.set_title(f'Encoder {enc_idx}', fontsize=10)
                ax_enc_error.text(0.5, 0.5, f'No Error', ha='center', va='center', 
                                transform=ax_enc_error.transAxes, fontsize=8)
                ax_enc_error.set_title(f'Error {enc_idx}', fontsize=10)
            
            ax_enc_recon.axis('off')
            ax_enc_error.axis('off')
        else:
            ax_enc_recon.text(0.5, 0.5, f'Enc {enc_idx}\nNo Data', 
                            ha='center', va='center', transform=ax_enc_recon.transAxes)
            ax_enc_recon.set_title(f'Encoder {enc_idx}', fontsize=10)
            ax_enc_recon.axis('off')
            
            ax_enc_error.text(0.5, 0.5, f'Enc {enc_idx}\nNo Error', 
                            ha='center', va='center', transform=ax_enc_error.transAxes)
            ax_enc_error.set_title(f'Error {enc_idx}', fontsize=10)
            ax_enc_error.axis('off')
    
    # PoE reconstructions at different trajectory steps (row 1, columns 5-7) - USE STORED DATA
    poe_reconstructions = trajectory_info.get('poe_trajectory_reconstructions', {})
    if poe_reconstructions:
        reconstruction_labels = ['initial', 'middle', 'final']
        display_labels = ['Start', 'Mid', 'End']
        
        for i, (recon_label, display_label) in enumerate(zip(reconstruction_labels, display_labels)):
            if i < 3:  # Columns 5-7 (indices 5, 6, 7)
                ax_poe_recon = fig.add_subplot(gs[1, 5 + i])
                ax_poe_error = fig.add_subplot(gs[2, 5 + i])
                
                if recon_label in poe_reconstructions and poe_reconstructions[recon_label] is not None:
                    poe_recon_data = poe_reconstructions[recon_label]
                    shape_logits = poe_recon_data['shape_logits']
                    grid_logits = poe_recon_data['grid_logits']
                    
                    # Extract reconstruction grid from stored data
                    recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                    
                    if recon_grid is not None:
                        # Plot reconstruction
                        ax_poe_recon.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        ax_poe_recon.set_title(f'PoE {display_label}\n{recon_rows}×{recon_cols}', fontsize=10)
                        
                        # === NEW: μ/σ text for PoE ===
                        if trajectory_info.get('z_vectors'):
                            poe_z_vec = trajectory_info['z_vectors'][0]  # initial PoE z
                            poe_mu_mean = float(np.mean(poe_z_vec))
                            poe_sigma_mean = float(np.std(poe_z_vec))
                            ax_poe_recon.text(0.5, -0.12, f"μ={poe_mu_mean:.2f} σ={poe_sigma_mean:.2f}",
                                              transform=ax_poe_recon.transAxes, ha='center', va='top', fontsize=8)
                        
                        # Calculate and plot error map
                        if recon_rows == target_shape[0] and recon_cols == target_shape[1]:
                            error_map = recon_grid.astype(float) - target_grid.astype(float)
                            im_error = ax_poe_error.imshow(error_map, cmap='RdBu', vmin=-5, vmax=5, 
                                                         interpolation='nearest', aspect='equal')
                            ax_poe_error.set_title(f'PoE Error {display_label}', fontsize=10)
                        else:
                            ax_poe_error.text(0.5, 0.5, 'Size\nMismatch', ha='center', va='center', 
                                            transform=ax_poe_error.transAxes, fontsize=8)
                            ax_poe_error.set_title(f'PoE Error {display_label}', fontsize=10)
                    else:
                        ax_poe_recon.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                        transform=ax_poe_recon.transAxes, fontsize=8)
                        ax_poe_recon.set_title(f'PoE {display_label}', fontsize=10)
                        ax_poe_error.text(0.5, 0.5, f'No Error', ha='center', va='center', 
                                        transform=ax_poe_error.transAxes, fontsize=8)
                        ax_poe_error.set_title(f'PoE Error {display_label}', fontsize=10)
                else:
                    ax_poe_recon.text(0.5, 0.5, f'PoE {display_label}\nNo Data', 
                                    ha='center', va='center', transform=ax_poe_recon.transAxes)
                    ax_poe_recon.set_title(f'PoE {display_label}', fontsize=10)
                    ax_poe_error.text(0.5, 0.5, f'PoE {display_label}\nNo Error', 
                                    ha='center', va='center', transform=ax_poe_error.transAxes)
                    ax_poe_error.set_title(f'PoE Error {display_label}', fontsize=10)
                
                ax_poe_recon.axis('off')
                ax_poe_error.axis('off')
    else:
        # Fallback message for missing PoE reconstruction data
        for i in range(3):
            ax_poe_recon = fig.add_subplot(gs[1, 5 + i])
            ax_poe_error = fig.add_subplot(gs[2, 5 + i])
            ax_poe_recon.text(0.5, 0.5, 'No PoE\nReconstruction\nData', 
                            ha='center', va='center', transform=ax_poe_recon.transAxes)
            ax_poe_recon.set_title(f'PoE Step {i}', fontsize=10)
            ax_poe_recon.axis('off')
            ax_poe_error.text(0.5, 0.5, 'No PoE\nError\nData', 
                            ha='center', va='center', transform=ax_poe_error.transAxes)
            ax_poe_error.set_title(f'PoE Error {i}', fontsize=10)
            ax_poe_error.axis('off')
    
    # Encoder Influence Analysis - showing how each encoder affects the PoE
    # ------------------------------------------------------------------
    ax_influence = fig.add_subplot(gs[3, 1:7])  # span the central columns
    ax_influence.set_title('Encoder Influence on PoE Latent', fontsize=11)
    ax_influence.grid(alpha=0.3)

    # Collect encoder mu and log_var data
    encoder_mus = []
    encoder_log_vars = []
    enc_colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    for enc_idx in range(num_encoders):
        enc_traj = individual_trajectories.get(f'encoder_{enc_idx}', {})
        enc_mu = enc_traj.get('mu')
        enc_log_var = enc_traj.get('log_var')
        
        if enc_mu is not None and enc_log_var is not None:
            # Convert to tensors and add batch dimension if needed
            if not isinstance(enc_mu, torch.Tensor):
                enc_mu = torch.tensor(enc_mu, dtype=torch.float32)
            if not isinstance(enc_log_var, torch.Tensor):
                enc_log_var = torch.tensor(enc_log_var, dtype=torch.float32)
            
            # Ensure proper shape (batch_size, latent_dim)
            if enc_mu.dim() == 1:
                enc_mu = enc_mu.unsqueeze(0)
            if enc_log_var.dim() == 1:
                enc_log_var = enc_log_var.unsqueeze(0)
            
            encoder_mus.append(enc_mu)
            encoder_log_vars.append(enc_log_var)
    
    if len(encoder_mus) >= 2:  # Need at least 2 encoders for influence analysis
        # Stack tensors: (num_encoders, batch_size, latent_dim)
        mu_stack = torch.stack(encoder_mus)
        log_var_stack = torch.stack(encoder_log_vars)
        
        # Compute influence indices
        influence_indices = compute_encoder_influence_metrics(mu_stack, log_var_stack)
        mean_influences = influence_indices.mean(dim=1).cpu().numpy()  # Average over batch
        
        # Create bar plot
        encoder_labels = [f'Enc {i}' for i in range(len(mean_influences))]
        bars = ax_influence.bar(encoder_labels, mean_influences, 
                               color=[enc_colors[i] for i in range(len(mean_influences))],
                               alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mean_influences)):
            ax_influence.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add equal influence line
        equal_influence = 1.0 / len(mean_influences)
        ax_influence.axhline(equal_influence, color='red', linestyle='--', alpha=0.8,
                            label=f'Equal Influence ({equal_influence:.3f})')
        
        ax_influence.set_ylabel('Influence Index')
        ax_influence.set_xlabel('Encoder')
        ax_influence.legend(fontsize=8)
        ax_influence.set_ylim(0, max(mean_influences) * 1.2)
    else:
        ax_influence.text(0.5, 0.5, 'Insufficient encoder data\nfor influence analysis', 
                         ha='center', va='center', transform=ax_influence.transAxes)

    # ------------------------------------------------------------------
    # tighten layout / save as before
    plt.suptitle(f'Multi-Encoder Analysis: Individual vs PoE ({num_encoders} Encoders)\nEVALUATION DATA - Trajectory Reconstructions: {decoder_display}', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved optimized multi-encoder trajectory visualization to: {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize model trajectory in latent space')
    parser.add_argument('--run_dir', type=str, required=True,
                      help='Directory containing the evaluation results')
    parser.add_argument('--key', type=str, required=True,
                      help='Problem key to visualize')
    parser.add_argument('--sample_idx', type=int, default=None,
                      help='Index of specific sample to visualize (default: all samples)')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save visualizations (defaults to run_dir)')
    parser.add_argument('--epoch', type=int, default=None,
                      help='Epoch to load model from (defaults to latest)')
    return parser.parse_args()

def main():
    """
    Example command: python LPN_reproduction\evaluate_trajectory.py --key pattern_task --run_dir runs_re_arc\test_2s
    """
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set save directory
    save_dir = args.save_dir if args.save_dir else args.run_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model (we need it for reconstruction computation)
    print("Loading model...")
    model, _, _, _ = load_model(args.run_dir, epoch=args.epoch, device=device)
    
    # Load evaluation results
    eval_file = os.path.join(args.run_dir, 'evaluation_results.pkl')
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"No evaluation results found in {args.run_dir}")
    
    with open(eval_file, 'rb') as f:
        eval_results = pickle.load(f)
    
    print(f"\n=== EVALUATION RESULTS DIAGNOSTIC ===")
    print(f"Available keys in evaluation results: {list(eval_results.keys())}")
    
    # Get trajectory information for the specified key
    if args.key not in eval_results:
        print(f"Available keys: {list(eval_results.keys())}")
        raise KeyError(f"No results found for key {args.key}")
    
    key_results = eval_results[args.key]
    print(f"\nAnalyzing results for key '{args.key}':")
    print(f"Results structure: {list(key_results.keys())}")
    
    # FIX: Look for trajectory_info in the correct location
    trajectory_info_list = []
    
    # Try different possible locations for trajectory_info
    if 'metrics' in key_results and 'trajectory_info' in key_results['metrics']:
        trajectory_info_list = key_results['metrics']['trajectory_info']
        print(f"✓ Found trajectory info in metrics: {len(trajectory_info_list)} samples")
    elif 'trajectory_info' in key_results:
        trajectory_info_list = key_results['trajectory_info'] 
        print(f"✓ Found trajectory info at root level: {len(trajectory_info_list)} samples")
    else:
        print(f"⚠ Trajectory info not found. Available in metrics: {list(key_results.get('metrics', {}).keys())}")
        print(f"Available at root: {list(key_results.keys())}")
    
    if not trajectory_info_list:
        print(f"\n⚠ WARNING: No trajectory information found for key {args.key}")
        print("\nDebug information:")
        if 'metrics' in key_results:
            metrics = key_results['metrics']
            print(f"Used latent optimization: {metrics.get('used_latent_optimization', 'Not specified')}")
            print(f"Available metrics keys: {list(metrics.keys())}")
        
        # Check optimization settings (using global settings import)
        latent_opt = settings.get_latent_optimization()
        print(f"\nLatent optimization settings:")
        print(f"Inference enabled: {latent_opt['inference']['enabled']}")
        print(f"Inference steps: {latent_opt['inference']['num_steps']}")
        return
    
    # Print trajectory information structure
    print_trajectory_info(trajectory_info_list)
    
    if args.sample_idx is not None:
        # Visualize specific sample
        if args.sample_idx >= len(trajectory_info_list):
            raise ValueError(f"Sample index {args.sample_idx} out of range (max: {len(trajectory_info_list)-1})")
        
        print(f"\nCreating visualization for sample {args.sample_idx}...")
        
        # Create input/target visualization
        input_target_path = os.path.join(save_dir, f'input_target_{args.key}_sample{args.sample_idx}.png')
        print("Creating input/target visualization...")
        visualize_input_target(trajectory_info_list[args.sample_idx], input_target_path)
        print(f"Input/target visualization saved to {input_target_path}")
        
        # Create comprehensive trajectory visualization
        trajectory_path = os.path.join(save_dir, f'comprehensive_trajectory_{args.key}_sample{args.sample_idx}.png')
        print("Creating comprehensive trajectory visualization...")
        visualize_comprehensive_trajectory(trajectory_info_list[args.sample_idx], model, trajectory_path, args.run_dir, device)
        print(f"Comprehensive trajectory visualization saved to {trajectory_path}")
    else:
        # Visualize all samples in one figure
        print(f"\nCreating comprehensive visualization for all {len(trajectory_info_list)} samples...")
        
        all_samples_path = os.path.join(save_dir, f'all_samples_comprehensive_trajectory_{args.key}.png')
        print("Creating comprehensive trajectory visualization for all samples...")
        visualize_all_samples_comprehensive(trajectory_info_list, model, all_samples_path, args.run_dir, device)
        print(f"All samples comprehensive trajectory visualization saved to {all_samples_path}")

if __name__ == "__main__":
    main()
