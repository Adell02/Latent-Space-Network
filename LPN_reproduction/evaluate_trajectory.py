#!/usr/bin/env python3

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

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import LatentProgramNetwork
from utils.settings_manager import settings
from utils.model_utils import load_model, set_seed
from utils.visualizers import load_evaluation_latent_data, get_comprehensive_latent_data_for_trajectory

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

def extract_grid_from_sequence(sequence, max_rows=30, max_cols=30):
    """
    Extract grid from ARC sequence format and get actual dimensions.
    This matches the approach in visualizers.py plot_reconstructions function.
    """
    sequence = np.array(sequence)
    
    if len(sequence) >= 902:
        # ARC format: shape info is at the end, not at positions 900-901
        rows = int(sequence[-2])
        cols = int(sequence[-1])
        
        # Grid data is the first 900 elements
        grid_flat = sequence[:900]
        grid_full = grid_flat.reshape(30, 30)
        actual_grid = grid_full[:rows, :cols]
        return actual_grid, (rows, cols)
    else:
        # Try to infer from shape
        grid_size = int(np.sqrt(len(sequence)))
        if grid_size * grid_size == len(sequence):
            grid = sequence.reshape(grid_size, grid_size)
            return grid, (grid_size, grid_size)
        else:
            # Fallback: assume it's already the right shape
            return sequence, sequence.shape

def load_training_latent_data(run_dir):
    """
    Load comprehensive latent data (encoders + PoE) for trajectory visualization background.
    This reuses the comprehensive latent space that combines all encoder and PoE latents.
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
            print(f"  - Data types: {len(set(labels))}")
            print(f"  - Includes: Individual encoders + PoE latents")
            
            # Return the original high-dimensional data for consistent t-SNE with trajectory
            return combined_latents
        else:
            print("⚠ Warning: No comprehensive latent data available")
            return None
            
    except ImportError as e:
        print(f"⚠ Warning: Could not import comprehensive latent data function: {e}")
        return None
    except Exception as e:
        print(f"⚠ Warning: Could not load comprehensive latent data: {e}")
        return None

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

def visualize_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device='cuda'):
    """
    Create a comprehensive trajectory visualization.
    Automatically detects single-encoder vs multi-encoder and calls appropriate function.
    """
    # Check if this is a multi-encoder model
    is_multi_encoder = trajectory_info.get('is_multi_encoder', False)
    
    if is_multi_encoder:
        print("Multi-encoder model detected - using enhanced visualization")
        return visualize_multi_encoder_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device)
    
    print("Single-encoder model detected - using standard visualization")
    
    # Load training latent data for background
    print("Loading training latent data for background visualization...")
    training_latent_data = load_training_latent_data(run_dir)
    
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
    
    # Plot trajectory in latent space with training background
    if z_vectors and len(z_vectors) >= 2:
        z_array = np.array(z_vectors)
        
        # Combine trajectory and training data for consistent t-SNE
        if training_latent_data is not None:
            combined_data = np.vstack([training_latent_data, z_array])
            scaler = StandardScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            combined_2d = tsne.fit_transform(combined_data_scaled)
            
            n_training = training_latent_data.shape[0]
            training_2d = combined_2d[:n_training]
            z_2d = combined_2d[n_training:]
            
            # Plot training background
            axs[0, 1].scatter(training_2d[:, 0], training_2d[:, 1],
                            c=np.arange(len(training_2d)), cmap='viridis', 
                            alpha=0.3, s=20, edgecolors='none')
        else:
            scaler = StandardScaler()
            z_array_normalized = scaler.fit_transform(z_array)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_array)-1), n_iter=1000)
            z_2d = tsne.fit_transform(z_array_normalized)
        
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
        
        axs[0, 1].set_title('Latent Trajectory (t-SNE)')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
    else:
        axs[0, 1].text(0.5, 0.5, 'No trajectory data', 
                      ha='center', va='center', transform=axs[0, 1].transAxes)
        axs[0, 1].set_title('Latent Trajectory')
    
    # Generate reconstructions for different trajectory steps
    if z_vectors and len(z_vectors) > 0:
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device).float()
        target_tensor = torch.tensor(target_seq).unsqueeze(0).to(device).float()
        
        model.eval()
        with torch.no_grad():
            # Reconstruct at different steps
            if len(z_vectors) >= 3:
                indices = [0, len(z_vectors)//2, len(z_vectors)-1]  # Start, middle, end
                labels = ['Start', 'Mid', 'End']
            else:
                indices = list(range(len(z_vectors)))
                labels = [f'Step {i}' for i in indices]
            
            for i, (idx, label) in enumerate(zip(indices, labels)):
                if i < 3:  # Only 3 slots available
                    z_step = torch.tensor(z_vectors[idx]).unsqueeze(0).to(device).float()
                    shape_logits, grid_logits = model.decoder(z_step, input_tensor, target_seq=target_tensor)
                    
                    shape_array = shape_logits.cpu().numpy()
                    grid_array = grid_logits.cpu().numpy()
                    
                    pred_shapes = np.argmax(shape_array, axis=-1)[0]
                    pred_grid = np.argmax(grid_array, axis=-1)[0]
                    
                    recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                    recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                    
                    row = 0 if i < 2 else 1
                    col = 2 + (i % 2)
                    axs[row, col].imshow(recon_grid, cmap='viridis')
                    axs[row, col].set_title(f'Reconstruction {label}\n{recon_rows}×{recon_cols}')
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
    
    plt.suptitle('Single-Encoder Trajectory Analysis', fontsize=16)
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
    
    # Load training latent data for background
    print("Loading training latent data for background...")
    training_latent_data = load_training_latent_data(run_dir)
    
    # Create figure with expanded layout for multi-encoder
    if is_multi_encoder:
        fig = plt.figure(figsize=(24, 8 * len(trajectory_info_list)))
        rows_per_sample = 2
        cols = 8  # Input/Target | Encoder spaces | PoE space | Reconstructions
    else:
        fig = plt.figure(figsize=(20, 6 * len(trajectory_info_list)))
        rows_per_sample = 1
        cols = 6  # Input | Target | Latent | 3 Reconstructions
    
    total_rows = rows_per_sample * len(trajectory_info_list)
    
    for sample_idx, trajectory_info in enumerate(trajectory_info_list):
        print(f"Processing sample {sample_idx + 1}/{len(trajectory_info_list)}")
        
        # Calculate row indices for this sample
        start_row = sample_idx * rows_per_sample
        
        if is_multi_encoder:
            # Multi-encoder layout
            # Row 1: Input | Encoder latent spaces | PoE latent
            # Row 2: Target | Encoder reconstructions | PoE reconstructions
            
            gs = fig.add_gridspec(total_rows, cols, 
                                height_ratios=[1] * total_rows,
                                width_ratios=[1, 1.5, 1.5, 1.5, 1, 1, 1, 1])
            
            # Input and Target
            ax_input = fig.add_subplot(gs[start_row, 0])
            ax_target = fig.add_subplot(gs[start_row + 1, 0])
            
            input_grid, input_shape = extract_grid_from_sequence(trajectory_info['input_sample'])
            target_grid, target_shape = extract_grid_from_sequence(trajectory_info['target_sample'])
            
            ax_input.imshow(input_grid, cmap='viridis')
            ax_input.set_title(f'Sample {sample_idx + 1} Input\n{input_shape[0]}×{input_shape[1]}')
            ax_input.axis('off')
            
            ax_target.imshow(target_grid, cmap='viridis')
            ax_target.set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
            ax_target.axis('off')
            
            # Individual encoder latent spaces (columns 1-4, top row)
            individual_trajectories = trajectory_info.get('individual_encoder_trajectories', {})
            
            for enc_idx in range(min(num_encoders, 3)):  # Limit to 3 encoders for layout
                ax_enc_latent = fig.add_subplot(gs[start_row, enc_idx + 1])
                
                encoder_key = f'encoder_{enc_idx}'
                if encoder_key in individual_trajectories:
                    enc_data = individual_trajectories[encoder_key]
                    enc_z = enc_data['z']
                    
                    if training_latent_data is not None:
                        # Show encoder point in context of training data
                        enc_z_flat = enc_z.flatten().reshape(1, -1)
                        combined_data = np.vstack([training_latent_data, enc_z_flat])
                        
                        scaler = StandardScaler()
                        combined_data_scaled = scaler.fit_transform(combined_data)
                        
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data)-1), n_iter=1000)
                        combined_2d = tsne.fit_transform(combined_data_scaled)
                        
                        training_2d = combined_2d[:-1]
                        enc_2d = combined_2d[-1:]
                        
                        # Plot training background
                        ax_enc_latent.scatter(training_2d[:, 0], training_2d[:, 1],
                                            c=np.arange(len(training_2d)), cmap='viridis', 
                                            alpha=0.3, s=10, edgecolors='none')
                        
                        # Plot encoder point
                        ax_enc_latent.scatter(enc_2d[0, 0], enc_2d[0, 1], c='red', s=50, marker='o', 
                                            edgecolors='black', linewidth=1, zorder=10)
                    
                    ax_enc_latent.set_title(f'Encoder {enc_idx}')
                    ax_enc_latent.grid(True, alpha=0.3)
                else:
                    ax_enc_latent.text(0.5, 0.5, f'Enc {enc_idx}\nNo Data', 
                                     ha='center', va='center', transform=ax_enc_latent.transAxes)
                    ax_enc_latent.set_title(f'Encoder {enc_idx}')
            
            # PoE latent trajectory (column 4, top row)
            ax_poe_latent = fig.add_subplot(gs[start_row, 4])
            
            z_vectors = trajectory_info['z_vectors']
            losses = trajectory_info['losses']
            
            if z_vectors and len(z_vectors) >= 2:
                z_array = np.array(z_vectors)
                
                if training_latent_data is not None:
                    combined_data = np.vstack([training_latent_data, z_array])
                    scaler = StandardScaler()
                    combined_data_scaled = scaler.fit_transform(combined_data)
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                    combined_2d = tsne.fit_transform(combined_data_scaled)
                    
                    n_training = training_latent_data.shape[0]
                    training_2d = combined_2d[:n_training]
                    z_2d = combined_2d[n_training:]
                    
                    # Plot training background
                    ax_poe_latent.scatter(training_2d[:, 0], training_2d[:, 1],
                                        c=np.arange(len(training_2d)), cmap='viridis', 
                                        alpha=0.3, s=10, edgecolors='none')
                else:
                    scaler = StandardScaler()
                    z_array_normalized = scaler.fit_transform(z_array)
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_array)-1), n_iter=1000)
                    z_2d = tsne.fit_transform(z_array_normalized)
                
                # Plot PoE trajectory
                trajectory_scatter = ax_poe_latent.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                                         s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
                
                # Draw arrows
                for i in range(len(z_2d) - 1):
                    ax_poe_latent.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1))
                
                ax_poe_latent.set_title('PoE Trajectory')
                ax_poe_latent.grid(True, alpha=0.3)
            else:
                ax_poe_latent.text(0.5, 0.5, 'No PoE\nTrajectory', 
                                 ha='center', va='center', transform=ax_poe_latent.transAxes)
                ax_poe_latent.set_title('PoE Trajectory')
            
            # Bottom row: Reconstructions
            input_tensor = torch.tensor(trajectory_info['input_sample']).unsqueeze(0).to(device).float()
            target_tensor = torch.tensor(trajectory_info['target_sample']).unsqueeze(0).to(device).float()
            
            model.eval()
            with torch.no_grad():
                # Individual encoder reconstructions (columns 1-3, bottom row)
                for enc_idx in range(min(num_encoders, 3)):
                    ax_enc_recon = fig.add_subplot(gs[start_row + 1, enc_idx + 1])
                    
                    encoder_key = f'encoder_{enc_idx}'
                    if encoder_key in individual_trajectories:
                        enc_data = individual_trajectories[encoder_key]
                        enc_z = torch.tensor(enc_data['z']).to(device).float()
                        
                        enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                            enc_z, input_tensor, target_seq=target_tensor
                        )
                        
                        shape_array = enc_shape_logits.cpu().numpy()
                        grid_array = enc_grid_logits.cpu().numpy()
                        
                        pred_shapes = np.argmax(shape_array, axis=-1)[0]
                        pred_grid = np.argmax(grid_array, axis=-1)[0]
                        
                        recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                        recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                        
                        ax_enc_recon.imshow(recon_grid, cmap='viridis')
                        ax_enc_recon.set_title(f'Enc {enc_idx} Recon\n{recon_rows}×{recon_cols}')
                        ax_enc_recon.axis('off')
                    else:
                        ax_enc_recon.text(0.5, 0.5, f'Enc {enc_idx}\nNo Recon', 
                                        ha='center', va='center', transform=ax_enc_recon.transAxes)
                        ax_enc_recon.set_title(f'Enc {enc_idx} Recon')
                        ax_enc_recon.axis('off')
                
                # PoE reconstructions (columns 4-7, bottom row)
                if z_vectors and len(z_vectors) > 0:
                    if len(z_vectors) >= 3:
                        indices = [0, len(z_vectors)//2, len(z_vectors)-1]
                        labels = ['Start', 'Mid', 'End']
                    else:
                        indices = list(range(len(z_vectors)))
                        labels = [f'Step {i}' for i in indices]
                    
                    for i, (idx, label) in enumerate(zip(indices, labels)):
                        if i < 4:  # Columns 4-7
                            ax_poe_recon = fig.add_subplot(gs[start_row + 1, 4 + i])
                            
                            z_step = torch.tensor(z_vectors[idx]).unsqueeze(0).to(device).float()
                            shape_logits, grid_logits = model.multi_encoder.decoder(
                                z_step, input_tensor, target_seq=target_tensor
                            )
                            
                            shape_array = shape_logits.cpu().numpy()
                            grid_array = grid_logits.cpu().numpy()
                            
                            pred_shapes = np.argmax(shape_array, axis=-1)[0]
                            pred_grid = np.argmax(grid_array, axis=-1)[0]
                            
                            recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                            recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                            
                            ax_poe_recon.imshow(recon_grid, cmap='viridis')
                            ax_poe_recon.set_title(f'PoE {label}\n{recon_rows}×{recon_cols}')
                            ax_poe_recon.axis('off')
        
        else:
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
            
            # Plot latent trajectory
            z_vectors = trajectory_info['z_vectors']
            losses = trajectory_info['losses']
            
            if z_vectors and len(z_vectors) >= 2:
                z_array = np.array(z_vectors)
                
                if training_latent_data is not None:
                    combined_data = np.vstack([training_latent_data, z_array])
                    scaler = StandardScaler()
                    combined_data_scaled = scaler.fit_transform(combined_data)
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                    combined_2d = tsne.fit_transform(combined_data_scaled)
                    
                    n_training = training_latent_data.shape[0]
                    training_2d = combined_2d[:n_training]
                    z_2d = combined_2d[n_training:]
                    
                    # Plot training background
                    ax_latent.scatter(training_2d[:, 0], training_2d[:, 1],
                                    c=np.arange(len(training_2d)), cmap='viridis', 
                                    alpha=0.3, s=10, edgecolors='none')
                else:
                    scaler = StandardScaler()
                    z_array_normalized = scaler.fit_transform(z_array)
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_array)-1), n_iter=1000)
                    z_2d = tsne.fit_transform(z_array_normalized)
                
                # Plot trajectory
                ax_latent.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
                
                for i in range(len(z_2d) - 1):
                    ax_latent.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1))
                
                ax_latent.set_title('Latent Trajectory')
                ax_latent.grid(True, alpha=0.3)
            else:
                ax_latent.text(0.5, 0.5, 'No trajectory', 
                             ha='center', va='center', transform=ax_latent.transAxes)
                ax_latent.set_title('Latent Trajectory')
            
            # Reconstructions
            if z_vectors and len(z_vectors) > 0:
                input_tensor = torch.tensor(trajectory_info['input_sample']).unsqueeze(0).to(device).float()
                target_tensor = torch.tensor(trajectory_info['target_sample']).unsqueeze(0).to(device).float()
                
                model.eval()
                with torch.no_grad():
                    if len(z_vectors) >= 3:
                        indices = [0, len(z_vectors)//2, len(z_vectors)-1]
                        labels = ['Start', 'Mid', 'End']
                    else:
                        indices = list(range(len(z_vectors)))
                        labels = [f'Step {i}' for i in indices]
                    
                    for i, (idx, label) in enumerate(zip(indices, labels)):
                        if i < 3:  # Columns 3-5
                            ax_recon = fig.add_subplot(gs[start_row, 3 + i])
                            
                            z_step = torch.tensor(z_vectors[idx]).unsqueeze(0).to(device).float()
                            shape_logits, grid_logits = model.decoder(z_step, input_tensor, target_seq=target_tensor)
                            
                            shape_array = shape_logits.cpu().numpy()
                            grid_array = grid_logits.cpu().numpy()
                            
                            pred_shapes = np.argmax(shape_array, axis=-1)[0]
                            pred_grid = np.argmax(grid_array, axis=-1)[0]
                            
                            recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                            recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                            
                            ax_recon.imshow(recon_grid, cmap='viridis')
                            ax_recon.set_title(f'{label}\n{recon_rows}×{recon_cols}')
                            ax_recon.axis('off')
    
    title = f'{"Multi-Encoder" if is_multi_encoder else "Single-Encoder"} Trajectory Analysis - All Samples'
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
    - Individual encoder reconstructions (using pre-computed data from evaluation if available)
    - PoE reconstructions (using pre-computed data from evaluation if available)
    All with training background in latent space.
    """
    if not trajectory_info.get('is_multi_encoder', False):
        print("Warning: This function is for multi-encoder models only.")
        return visualize_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device)
    
    num_encoders = trajectory_info.get('num_encoders', 1)
    print(f"Creating multi-encoder visualization for {num_encoders} encoders...")
    
    # Load training latent data for background
    print("Loading training latent data for background visualization...")
    training_latent_data = load_training_latent_data(run_dir)
    
    # Create figure with expanded layout for multi-encoder
    fig = plt.figure(figsize=(24, 12))
    
    # Define grid layout: 
    # Row 1: Input/Target | Merged Latent Space (Training + Encoders + PoE Trajectory)
    # Row 2: Individual Encoder Reconstructions | PoE Reconstructions  
    gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1], height_ratios=[1, 1])
    
    # Input and Target (left column)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[1, 0])
    
    input_grid, input_shape = extract_grid_from_sequence(trajectory_info['input_sample'])
    im_input = ax_input.imshow(input_grid, cmap='viridis')
    ax_input.set_title(f'Input\n{input_shape[0]}×{input_shape[1]}')
    ax_input.axis('off')
    
    target_grid, target_shape = extract_grid_from_sequence(trajectory_info['target_sample'])
    im_target = ax_target.imshow(target_grid, cmap='viridis')
    ax_target.set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
    ax_target.axis('off')
    
    # Merged latent space visualization (columns 1-5, top row)
    ax_merged_latent = fig.add_subplot(gs[0, 1:6])
    
    individual_trajectories = trajectory_info.get('individual_encoder_trajectories', {})
    z_vectors = trajectory_info['z_vectors']
    losses = trajectory_info['losses']
    
    if training_latent_data is not None and z_vectors and len(z_vectors) >= 2:
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
            
            # Collect all individual encoder z vectors
            encoder_zs = []
            encoder_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            encoder_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'h']
            
            for enc_idx in range(num_encoders):
                encoder_key = f'encoder_{enc_idx}'
                if encoder_key in individual_trajectories:
                    enc_data = individual_trajectories[encoder_key]
                    enc_z = enc_data['z'].flatten().reshape(1, -1)
                    encoder_zs.append(enc_z)
            
            # Prepare PoE trajectory data
            z_array = np.array(z_vectors)
            
            # Combine ALL data for consistent t-SNE: training + individual encoders + PoE trajectory
            all_data_for_tsne = [training_latent_data]
            
            # Add individual encoder points
            for enc_z in encoder_zs:
                all_data_for_tsne.append(enc_z)
            
            # Add PoE trajectory
            all_data_for_tsne.append(z_array)
            
            combined_data = np.vstack(all_data_for_tsne)
            
            # Apply StandardScaler and t-SNE
            scaler = StandardScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data)//4), n_iter=1000)
            combined_2d = tsne.fit_transform(combined_data_scaled)
            
            # Split the results back
            n_training = training_latent_data.shape[0]
            training_2d = combined_2d[:n_training]
            
            current_idx = n_training
            encoder_2d_points = []
            for enc_z in encoder_zs:
                enc_2d = combined_2d[current_idx:current_idx + len(enc_z)]
                encoder_2d_points.append(enc_2d)
                current_idx += len(enc_z)
            
            z_2d = combined_2d[current_idx:current_idx + len(z_array)]
            
            # Plot training background
            ax_merged_latent.scatter(training_2d[:, 0], training_2d[:, 1],
                                   c='lightgray', alpha=0.4, s=15, edgecolors='none', 
                                   label='Training Data')
            
            # Plot individual encoder initial estimates
            for enc_idx, enc_2d in enumerate(encoder_2d_points):
                if len(enc_2d) > 0:
                    color = encoder_colors[enc_idx % len(encoder_colors)]
                    marker = encoder_markers[enc_idx % len(encoder_markers)]
                    ax_merged_latent.scatter(enc_2d[0, 0], enc_2d[0, 1], 
                                           c=color, s=150, marker=marker,
                                           label=f'Encoder {enc_idx}', 
                                           edgecolors='black', linewidth=2, zorder=10)
            
            # Plot PoE trajectory
            if len(z_2d) >= 2:
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
                
                # Add colorbar for trajectory losses
                cbar = plt.colorbar(trajectory_scatter, ax=ax_merged_latent, shrink=0.6)
                cbar.set_label('PoE Loss', rotation=270, labelpad=20)
            
            ax_merged_latent.set_title('Merged Latent Space:\nTraining Data + Individual Encoders + PoE Trajectory', fontsize=12)
            ax_merged_latent.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_merged_latent.grid(True, alpha=0.3)
            
        except ImportError:
            print("Warning: sklearn not available for t-SNE visualization")
            ax_merged_latent.text(0.5, 0.5, 'sklearn required\nfor t-SNE visualization', 
                                ha='center', va='center', transform=ax_merged_latent.transAxes)
            ax_merged_latent.set_title('Merged Latent Space (Error)')
        except Exception as e:
            print(f"Warning: Error creating merged latent space visualization: {e}")
            ax_merged_latent.text(0.5, 0.5, f'Error creating\nmerged visualization:\n{str(e)[:50]}...', 
                                ha='center', va='center', transform=ax_merged_latent.transAxes)
            ax_merged_latent.set_title('Merged Latent Space (Error)')
    else:
        ax_merged_latent.text(0.5, 0.5, 'Insufficient data\nfor merged visualization', 
                            ha='center', va='center', transform=ax_merged_latent.transAxes)
        ax_merged_latent.set_title('Merged Latent Space')
    
    # Bottom row: Reconstructions
    # Individual encoder reconstructions (columns 1-4, bottom row)
    input_tensor = torch.tensor(trajectory_info['input_sample']).unsqueeze(0).to(device).float()
    target_tensor = torch.tensor(trajectory_info['target_sample']).unsqueeze(0).to(device).float()
    
    model.eval()
    with torch.no_grad():
        # Individual encoder reconstructions - use pre-computed if available
        pre_computed_reconstructions = trajectory_info.get('individual_encoder_reconstructions', {})
        
        for enc_idx in range(min(num_encoders, 4)):
            ax_enc_recon = fig.add_subplot(gs[1, enc_idx + 1])
            
            encoder_key = f'encoder_{enc_idx}'
            
            # Try to use pre-computed reconstruction first
            if encoder_key in pre_computed_reconstructions and pre_computed_reconstructions[encoder_key] is not None:
                try:
                    pre_computed = pre_computed_reconstructions[encoder_key]
                    shape_array = pre_computed['shape_logits']
                    grid_array = pre_computed['grid_logits']
                    
                    pred_shapes = np.argmax(shape_array, axis=-1)[0]
                    pred_grid = np.argmax(grid_array, axis=-1)[0]
                    
                    recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                    recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                    
                    ax_enc_recon.imshow(recon_grid, cmap='viridis')
                    ax_enc_recon.set_title(f'Enc {enc_idx} Recon\n{recon_rows}×{recon_cols} (cached)')
                    ax_enc_recon.axis('off')
                    continue
                except Exception as e:
                    print(f"Warning: Could not use pre-computed reconstruction for encoder {enc_idx}: {e}")
            
            # Fallback to computing reconstruction on the fly
            if encoder_key in individual_trajectories:
                try:
                    enc_data = individual_trajectories[encoder_key]
                    enc_z = torch.tensor(enc_data['z']).to(device).float()
                    
                    # Get individual encoder reconstruction
                    enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                        enc_z, input_tensor, target_seq=target_tensor
                    )
                    
                    # Process reconstruction
                    shape_array = enc_shape_logits.cpu().numpy()
                    grid_array = enc_grid_logits.cpu().numpy()
                    
                    pred_shapes = np.argmax(shape_array, axis=-1)[0]
                    pred_grid = np.argmax(grid_array, axis=-1)[0]
                    
                    recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                    recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                    
                    ax_enc_recon.imshow(recon_grid, cmap='viridis')
                    ax_enc_recon.set_title(f'Enc {enc_idx} Recon\n{recon_rows}×{recon_cols}')
                    ax_enc_recon.axis('off')
                except Exception as e:
                    print(f"Warning: Could not compute reconstruction for encoder {enc_idx}: {e}")
                    ax_enc_recon.text(0.5, 0.5, f'Enc {enc_idx}\nRecon Error', 
                                    ha='center', va='center', transform=ax_enc_recon.transAxes)
                    ax_enc_recon.set_title(f'Enc {enc_idx} Recon')
                    ax_enc_recon.axis('off')
            else:
                ax_enc_recon.text(0.5, 0.5, f'Enc {enc_idx}\nNo Recon', 
                                ha='center', va='center', transform=ax_enc_recon.transAxes)
                ax_enc_recon.set_title(f'Enc {enc_idx} Recon')
                ax_enc_recon.axis('off')
        
        # PoE reconstructions at different trajectory steps (columns 5-7, bottom row)
        # Try to use pre-computed PoE reconstructions first
        pre_computed_poe = trajectory_info.get('poe_trajectory_reconstructions', {})
        
        if pre_computed_poe and len(pre_computed_poe) >= 3:
            # Use pre-computed reconstructions
            for i, (label, recon_data) in enumerate(list(pre_computed_poe.items())[:3]):
                ax_poe_recon = fig.add_subplot(gs[1, 5 + i])
                
                if recon_data is not None:
                    try:
                        shape_array = recon_data['shape_logits']
                        grid_array = recon_data['grid_logits']
                        
                        pred_shapes = np.argmax(shape_array, axis=-1)[0]
                        pred_grid = np.argmax(grid_array, axis=-1)[0]
                        
                        recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                        recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                        
                        ax_poe_recon.imshow(recon_grid, cmap='viridis')
                        ax_poe_recon.set_title(f'PoE {label.title()}\n{recon_rows}×{recon_cols} (cached)')
                        ax_poe_recon.axis('off')
                    except Exception as e:
                        print(f"Warning: Could not use pre-computed PoE reconstruction {label}: {e}")
                        ax_poe_recon.text(0.5, 0.5, f'PoE {label}\nError', 
                                        ha='center', va='center', transform=ax_poe_recon.transAxes)
                        ax_poe_recon.set_title(f'PoE {label}')
                        ax_poe_recon.axis('off')
                else:
                    ax_poe_recon.text(0.5, 0.5, f'PoE {label}\nNo Data', 
                                    ha='center', va='center', transform=ax_poe_recon.transAxes)
                    ax_poe_recon.set_title(f'PoE {label}')
                    ax_poe_recon.axis('off')
        
        elif z_vectors and len(z_vectors) > 0:
            # Fallback to computing reconstructions on the fly
            if len(z_vectors) >= 3:
                indices = [0, len(z_vectors)//2, len(z_vectors)-1]  # Start, middle, end
                labels = ['Start', 'Mid', 'End']
            else:
                indices = list(range(len(z_vectors)))
                labels = [f'Step {i}' for i in indices]
            
            for i, (idx, label) in enumerate(zip(indices, labels)):
                if i < 3:  # Only 3 slots available
                    ax_poe_recon = fig.add_subplot(gs[1, 5 + i])
                    
                    try:
                        z_step = torch.tensor(z_vectors[idx]).unsqueeze(0).to(device).float()
                        shape_logits, grid_logits = model.multi_encoder.decoder(
                            z_step, input_tensor, target_seq=target_tensor
                        )
                        
                        shape_array = shape_logits.cpu().numpy()
                        grid_array = grid_logits.cpu().numpy()
                        
                        pred_shapes = np.argmax(shape_array, axis=-1)[0]
                        pred_grid = np.argmax(grid_array, axis=-1)[0]
                        
                        recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
                        recon_grid = pred_grid.reshape(30, 30)[:recon_rows, :recon_cols]
                        
                        ax_poe_recon.imshow(recon_grid, cmap='viridis')
                        ax_poe_recon.set_title(f'PoE {label}\n{recon_rows}×{recon_cols}')
                        ax_poe_recon.axis('off')
                    except Exception as e:
                        print(f"Warning: Could not compute PoE reconstruction for step {idx}: {e}")
                        ax_poe_recon.text(0.5, 0.5, f'PoE {label}\nError', 
                                        ha='center', va='center', transform=ax_poe_recon.transAxes)
                        ax_poe_recon.set_title(f'PoE {label}')
                        ax_poe_recon.axis('off')
        else:
            for i in range(3):
                ax_poe_recon = fig.add_subplot(gs[1, 5 + i])
                ax_poe_recon.text(0.5, 0.5, 'No PoE\nRecon', 
                                ha='center', va='center', transform=ax_poe_recon.transAxes)
                ax_poe_recon.set_title(f'PoE Step {i}')
                ax_poe_recon.axis('off')
    
    plt.suptitle(f'Multi-Encoder Comprehensive Analysis ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved multi-encoder trajectory visualization to: {save_path}")

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
        
        # Check optimization settings
        from utils.settings_manager import settings
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
