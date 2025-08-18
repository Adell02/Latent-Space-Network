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
from matplotlib.gridspec import GridSpec

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import LatentProgramNetwork
from utils.settings_manager import settings
from utils.model_utils import load_model, set_seed
from utils.visualizers import load_evaluation_latent_data, get_comprehensive_latent_data_for_trajectory
from utils.data_preparation import extract_grid_from_sequence, safe_extract_reconstruction_grid

# Use safe extraction to avoid scalar conversion errors
extract_reconstruction_grid = safe_extract_reconstruction_grid

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

def load_task_level_latent_data(task_latent_data, task_trajectories=None):
    """
    Load task-level latent data for visualization.
    This creates one point per task instead of per-sample or per-trajectory-step.
    
    Args:
        task_latent_data: Dict containing task latents from evaluate_model_with_task_optimization
        task_trajectories: Optional dict containing trajectory information
        
    Returns:
        tuple: (all_latents, all_labels, all_colors, task_metadata)
    """
    print("Loading task-level latent data...")
    
    all_latents = []
    all_labels = []
    all_colors = []
    task_metadata = {}
    
    if 'task_latents' not in task_latent_data:
        print("WARNING: No task_latents found in task_latent_data!")
        return all_latents, all_labels, all_colors, task_metadata
    
    # Define colors for different tasks
    import matplotlib.pyplot as plt
    import numpy as np
    
    task_keys = list(task_latent_data['task_latents'].keys())
    colors = plt.cm.Set1(np.linspace(0, 1, min(len(task_keys), 9)))  # Use Set1 colormap
    
    print(f"Processing {len(task_keys)} tasks...")
    
    for i, (task_key, task_data) in enumerate(task_latent_data['task_latents'].items()):
        # Get the final optimized latent for this task
        task_latent = task_data['latent_z']
        final_loss = task_data['final_loss']
        num_support_samples = task_data['num_support_samples']
        
        # Ensure latent is flattened
        if len(task_latent.shape) > 1:
            task_latent = task_latent.flatten()
        
        all_latents.append(task_latent)
        all_labels.append(f"task_{task_key}")
        
        # Use different colors for different tasks
        color = colors[i % len(colors)] if i < len(colors) else plt.cm.Set1(i % 9)
        all_colors.append(color)
        
        # Store metadata for this task
        task_metadata[task_key] = {
            'final_loss': final_loss,
            'num_support_samples': num_support_samples,
            'latent_vector': task_latent,
            'color': color
        }
        
        print(f"  Task '{task_key}': {num_support_samples} support samples, final loss: {final_loss:.4f}")
    
    print(f"Loaded {len(all_latents)} task-level latent vectors")
    
    return all_latents, all_labels, all_colors, task_metadata

#     # Helper function to generate reconstruction from latent vector with proper input sequences
def generate_reconstruction_from_latent(z_vector, model, device, input_seq=None, target_seq=None):
    """Generate reconstruction from latent vector using model decoder with proper input sequences"""
    try:
        with torch.no_grad():
            if not isinstance(z_vector, torch.Tensor):
                z_vector = torch.tensor(z_vector, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                z_vector = z_vector.unsqueeze(0) if z_vector.dim() == 1 else z_vector
            
            # Use the actual input and target sequences if available
            if input_seq is not None and target_seq is not None:
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
                shape_logits, grid_logits = model.decoder(z_vector, input_tensor, target_seq=target_tensor)
            else:
                # Fallback to just latent vector (may not work for all decoders)
                shape_logits, grid_logits = model.decoder(z_vector)
            
            recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
            return recon_grid, recon_rows, recon_cols
    except Exception as e:
        print(f"DEBUG: Error generating reconstruction: {e}")
        return None, None, None


def visualize_comprehensive_trajectory(trajectory_info, model, save_path, run_dir, device='cuda', ood_enabled=False, ood_task_keys=None):
    """
    Create a comprehensive multi-encoder trajectory visualization with training samples in background.
    Shows input/target, trajectory in latent space with training background, and reconstructions.
    Args:
        ood_enabled: Whether OOD sampling is enabled
        ood_task_keys: List of OOD task keys used for this evaluation
    """
    # Configure matplotlib to prevent issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    plt.rcParams['figure.max_open_warning'] = 0  # Suppress warning about too many figures
    
    print("Creating multi-encoder visualization for", getattr(model, 'num_encoders', 1), "encoders...")
    
    # Handle trajectory_info type - could be dict or list
    if isinstance(trajectory_info, list):
        if len(trajectory_info) > 0:
            # Use the first trajectory if it's a list
            trajectory_info = trajectory_info[0]
            print("Using first trajectory from list")
        else:
            print("[ ERROR ] Empty trajectory_info list")
            return
    
    print("Trajectory used:", trajectory_info.get('trajectory_type', 'Unknown'))
    
    # Use safe extraction to avoid scalar conversion errors
    extract_reconstruction_grid = safe_extract_reconstruction_grid
    
    print("Loading unified latent data (training + support + query + trajectory)...")
    unified = load_unified_latent_data_with_trajectory(
        run_dir, model, device, trajectory_info, ood_enabled=ood_enabled, ood_task_keys=ood_task_keys
    )
    # Be robust to different return arities (7 vs 8 values)
    if isinstance(unified, (list, tuple)):
        if len(unified) == 8:
            training_latent_data, training_tsne_2d, training_labels, training_colors, trajectory_tsne_2d, all_labels, all_tsne_2d, actual_evaluated_keys = unified
        elif len(unified) == 7:
            training_latent_data, training_tsne_2d, training_labels, training_colors, trajectory_tsne_2d, all_labels, all_tsne_2d = unified
            actual_evaluated_keys = []
        else:
            training_latent_data = training_tsne_2d = training_labels = training_colors = None
            trajectory_tsne_2d = all_labels = all_tsne_2d = None
            actual_evaluated_keys = []
    else:
        training_latent_data = training_tsne_2d = training_labels = training_colors = None
        trajectory_tsne_2d = all_labels = all_tsne_2d = None
        actual_evaluated_keys = []

    # Create figure with GridSpec for complex multi-encoder layout
    num_encoders = getattr(model, 'num_encoders', 1)
    # Calculate dynamic layout based on number of encoders
    total_cols = max(8, num_encoders + 5)  # Ensure enough columns for all encoders + POE + input/target
    fig = plt.figure(figsize=(3 * total_cols, 16))  # Scale figure width based on columns
    gs = GridSpec(5, total_cols, figure=fig)  # 5 rows, dynamic columns
    
    # ROW 0: INPUT SAMPLE | LATENT SPACE | LOSS PROGRESSION
    input_seq = trajectory_info['input_sample']
    input_grid, input_shape = extract_grid_from_sequence(input_seq)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(input_grid, cmap='viridis')
    ax_input.set_title(f'Input Sample\n{input_shape[0]}×{input_shape[1]}')
    ax_input.axis('off')
    
    # Latent space (spans columns 1 to middle of total columns)
    latent_end_col = max(5, (total_cols + 1) // 2)
    ax_latent = fig.add_subplot(gs[0, 1:latent_end_col])
    
    # Loss progression (spans remaining columns)
    ax_loss = fig.add_subplot(gs[0, latent_end_col:])
    
    # ROW 1: TARGET SAMPLE | RECONSTRUCTIONS FOR EACH ENCODER | POE INITIAL/MID/FINAL OPTIMISATION
    target_seq = trajectory_info['target_sample']
    target_grid, target_shape = extract_grid_from_sequence(target_seq)
    ax_target = fig.add_subplot(gs[1, 0])
    ax_target.imshow(target_grid, cmap='viridis')
    ax_target.set_title(f'Target Sample\n{target_shape[0]}×{target_shape[1]}')
    ax_target.axis('off')
    
    # Individual encoder reconstructions (dynamic columns based on num_encoders)
    ax_enc_recons = []
    for enc_idx in range(num_encoders):
        ax = fig.add_subplot(gs[1, 1 + enc_idx])
        ax_enc_recons.append(ax)
    
    # POE reconstructions (columns after encoders)
    poe_start_col = 1 + num_encoders
    ax_poe_initial = fig.add_subplot(gs[1, poe_start_col])
    ax_poe_mid = fig.add_subplot(gs[1, poe_start_col + 1])
    ax_poe_final = fig.add_subplot(gs[1, poe_start_col + 2])
    
    # ROW 2: ERROR MAPS
    ax_error_maps = fig.add_subplot(gs[2, :])
    
    # ROW 3: INPUT QUERY | TARGET QUERY | RECONSTRUCTIONS QUERY FOR EACH ENCODER | INITIAL/FINAL POE RECONSTRUCTION
    # Input query
    ax_query_input = fig.add_subplot(gs[3, 0])
    query_input = trajectory_info.get('query_input')
    print(f"DEBUG: query_input available: {query_input is not None}")
    if query_input is not None:
        query_input_grid, query_input_shape = extract_grid_from_sequence(query_input)
        ax_query_input.imshow(query_input_grid, cmap='viridis')
        ax_query_input.set_title(f'Input Query\n{query_input_shape[0]}×{query_input_shape[1]}')
    else:
        ax_query_input.text(0.5, 0.5, 'Input Query\n(No Data)', ha='center', va='center', transform=ax_query_input.transAxes)
        ax_query_input.set_title('Input Query')
    ax_query_input.axis('off')
    
    # Target query
    ax_query_target = fig.add_subplot(gs[3, 1])
    query_target = trajectory_info.get('query_target')
    print(f"DEBUG: query_target available: {query_target is not None}")
    if query_target is not None:
        query_target_grid, query_target_shape = extract_grid_from_sequence(query_target)
        ax_query_target.imshow(query_target_grid, cmap='viridis')
        ax_query_target.set_title(f'Target Query\n{query_target_shape[0]}×{query_target_shape[1]}')
    else:
        ax_query_target.text(0.5, 0.5, 'Target Query\n(No Data)', ha='center', va='center', transform=ax_query_target.transAxes)
        ax_query_target.set_title('Target Query')
    ax_query_target.axis('off')
    
    # Query reconstructions for each encoder (dynamic columns)
    ax_query_recons = []
    for enc_idx in range(num_encoders):
        ax = fig.add_subplot(gs[3, 2 + enc_idx])
        ax_query_recons.append(ax)
    
    # Initial and final POE reconstructions (columns after encoders)
    query_poe_start_col = 2 + num_encoders
    ax_query_poe_initial = fig.add_subplot(gs[3, query_poe_start_col])
    ax_query_poe_final = fig.add_subplot(gs[3, query_poe_start_col + 1])
    
    # ROW 4: ERROR MAPS
    ax_query_error_maps = fig.add_subplot(gs[4, :])
    
    # Get trajectory data early
    z_vectors = trajectory_info['z_vectors']
    losses = trajectory_info.get('losses', [])
    
    # Plot trajectory in latent space using precomputed t-SNE coordinates
    if z_vectors is not None and len(z_vectors) >= 2:
        # Use precomputed t-SNE coordinates for consistency
        if training_latent_data is not None and training_tsne_2d is not None:
            # Plot training background using precomputed coordinates - COLOR BY ENCODER + HIGHLIGHT SAME KEY
            if training_colors is not None and training_labels is not None:
                # Extract keys and encoders from training labels (assuming format like "training_enc_0_key_1234")
                training_keys = []
                training_encoders = []
                for label in training_labels:
                    if '_key_' in label and 'training_enc_' in label:
                        # Extract encoder and key
                        encoder_part = label.split('_key_')[0]
                        encoder = encoder_part.split('training_enc_')[-1]
                        key = label.split('_key_')[-1]
                        training_keys.append(key)
                        # Fix: Handle None encoder case
                        try:
                            training_encoders.append(int(encoder))
                        except (ValueError, TypeError):
                            training_encoders.append(0)  # Default to 0 if not an int
                    else:
                        training_keys.append(label)
                        training_encoders.append(0)  # Default encoder
                
                # Get the key of the sample being evaluated
                evaluated_key = trajectory_info.get('evaluated_key', trajectory_info.get('task_key', 'unknown'))
                print(f"DEBUG: Using evaluated_key: '{evaluated_key}'")
                
                # Create light colors for encoders (different light colors for each encoder)
                unique_encoders = sorted(list(set(training_encoders)))
                encoder_colors = {}
                for enc in unique_encoders:
                    try:
                        # Ensure encoder is a valid integer and get color safely
                        enc_int = int(enc) if enc is not None else 0
                        encoder_colors[enc] = plt.cm.tab10(enc_int % 10)
                    except (ValueError, TypeError):
                        # Fallback to default color if encoder is invalid
                        encoder_colors[enc] = plt.cm.tab10(0)
                
                # Create bright colors for the evaluated key (replaced yellow with more visible colors)
                bright_colors = ['red', 'orange', 'darkorange', 'lime', 'cyan', 'magenta', 'pink', 'brown']
                # Fix: Ensure evaluated_key is a string and handle hash properly
                if evaluated_key is None:
                    evaluated_key = 'unknown'
                evaluated_color = bright_colors[abs(hash(str(evaluated_key))) % len(bright_colors)]
                
                # Plot background points colored by encoder (light colors)
                for encoder in unique_encoders:
                    indices = [i for i, enc in enumerate(training_encoders) if enc == encoder]
                    if indices:
                        x_coords = training_tsne_2d[indices, 0]
                        y_coords = training_tsne_2d[indices, 1]
                        color = encoder_colors[encoder]
                        
                        # Use alpha for background effect with color validation
                        try:
                            ax_latent.scatter(x_coords, y_coords, color=color, alpha=0.6, s=12, 
                                           edgecolors='none', label=f'Encoder {encoder} (Light)')
                        except Exception as color_error:
                            print(f"DEBUG: Color error for encoder {encoder}, using default color: {color_error}")
                            # Use default color as fallback
                            ax_latent.scatter(x_coords, y_coords, color='blue', alpha=0.6, s=12, 
                                           edgecolors='none', label=f'Encoder {encoder} (Light)')
                
                # Highlight samples with the same key as evaluated sample (bright colors)
                same_key_indices = [i for i, key in enumerate(training_keys) if key == evaluated_key]
                if same_key_indices:
                    x_coords = training_tsne_2d[same_key_indices, 0]
                    y_coords = training_tsne_2d[same_key_indices, 1]
                    
                    # Use bright color for same key samples with color validation
                    try:
                        ax_latent.scatter(x_coords, y_coords, color=evaluated_color, alpha=0.9, s=25, 
                                       edgecolors='black', linewidth=1.5, 
                                       label=f'Same Key: {evaluated_key[:8]} (Bright)')
                    except Exception as color_error:
                        print(f"DEBUG: Color error for evaluated key, using default color: {color_error}")
                        # Use default color as fallback
                        ax_latent.scatter(x_coords, y_coords, color='red', alpha=0.9, s=25, 
                                       edgecolors='black', linewidth=1.5, 
                                       label=f'Same Key: {evaluated_key[:8]} (Bright)')
                
                # Plot support and query samples for the evaluated key only
                if all_labels is not None:
                    print(f"DEBUG: Checking for support/query samples in {len(all_labels)} total labels")
                    print(f"DEBUG: First few all_labels: {all_labels[:5]}")
                    print(f"DEBUG: Looking for keys: {actual_evaluated_keys}")
                    print(f"DEBUG: all_tsne_2d available: {all_tsne_2d is not None}")
                    if all_tsne_2d is not None:
                        print(f"DEBUG: all_tsne_2d shape: {all_tsne_2d.shape}")
                    
                    # Filter support samples for the evaluated key
                    support_indices = []
                    for i, label in enumerate(all_labels):
                        if 'support' in label:
                            for key in actual_evaluated_keys:
                                if key in label:
                                    support_indices.append(i)
                                    break
                    
                    print(f"DEBUG: Found {len(support_indices)} support samples for keys {actual_evaluated_keys}")
                    if support_indices and all_tsne_2d is not None:
                        x_coords = all_tsne_2d[support_indices, 0]
                        y_coords = all_tsne_2d[support_indices, 1]
                        ax_latent.scatter(x_coords, y_coords, color='blue', alpha=0.8, s=15, 
                                    marker='s', edgecolors='black', linewidth=1.0)
                    
                    # Filter query samples for the evaluated key
                    query_indices = []
                    for i, label in enumerate(all_labels):
                        if 'query' in label:
                            for key in actual_evaluated_keys:
                                if key in label:
                                    query_indices.append(i)
                                    break                    
                    print(f"DEBUG: Found {len(query_indices)} query samples for keys {actual_evaluated_keys}")
                    if query_indices and all_tsne_2d is not None:
                        x_coords = all_tsne_2d[query_indices, 0]
                        y_coords = all_tsne_2d[query_indices, 1]
                        ax_latent.scatter(x_coords, y_coords, color='red', alpha=0.8, s=15, 
                                    marker='^', edgecolors='black', linewidth=1.0)
                    
                    # Also check all labels for support/query patterns
                    all_support_labels = [l for l in all_labels if 'support' in l]
                    all_query_labels = [l for l in all_labels if 'query' in l]
                    print(f"DEBUG: All support labels: {all_support_labels}")
                    print(f"DEBUG: All query labels: {all_query_labels}")
                else:
                    print(f"DEBUG: No all_labels available for support/query plotting")
            else:
                print(f"DEBUG: Missing training_colors or training_labels for background plotting")
                print(f"DEBUG: training_colors available: {training_colors is not None}")
                print(f"DEBUG: training_labels available: {training_labels is not None}")
        else:
            print(f"DEBUG: Missing training_latent_data or training_tsne_2d for background plotting")
            print(f"DEBUG: training_latent_data available: {training_latent_data is not None}")
            print(f"DEBUG: training_tsne_2d available: {training_tsne_2d is not None}")
        
        # Use unified trajectory coordinates (already computed in unified t-SNE)
        z_2d = trajectory_tsne_2d
        
        if z_2d is not None:
            # Fix: Ensure losses array matches trajectory length
            if len(losses) != len(z_2d):
                print(f"Warning: losses length ({len(losses)}) != trajectory length ({len(z_2d)})")
                # Use the shorter length to avoid mismatch
                min_length = min(len(losses), len(z_2d))
                z_2d = z_2d[:min_length]
                losses = losses[:min_length]
                print(f"Adjusted to length: {min_length}")
            
            # Plot trajectory
            trajectory_scatter = ax_latent.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                                 s=100, alpha=1.0, edgecolors='black', linewidth=2)
            
            # Draw arrows between consecutive trajectory points
            for i in range(len(z_2d) - 1):
                ax_latent.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=2))
            
            # Mark start and end points
            ax_latent.scatter(z_2d[0, 0], z_2d[0, 1], color='green', s=200, marker='o', 
                            label='Start', edgecolors='black', linewidth=3, zorder=10, alpha=1.0)
            ax_latent.scatter(z_2d[-1, 0], z_2d[-1, 1], color='red', s=200, marker='s', 
                            label='End', edgecolors='black', linewidth=3, zorder=10, alpha=1.0)
            
            # Add colorbar
            cbar = plt.colorbar(trajectory_scatter, ax=ax_latent, shrink=0.8)
            cbar.set_label('Loss', rotation=270, labelpad=20)
            
            ax_latent.set_title('Latent Trajectory (Light=Encoder, Bright=Same Key)')
            
            # Create comprehensive legend with all data types
            legend_elements = []
            
            # Define unique_encoders based on training data
            if training_labels:
                # Extract encoder indices from training labels
                training_encoders = []
                for label in training_labels:
                    if 'training_enc_' in label:
                        # Extract encoder index from label like "training_enc_0_key_xxx"
                        parts = label.split('_')
                        if len(parts) >= 3:
                            try:
                                enc_idx = int(parts[2])
                                training_encoders.append(enc_idx)
                            except ValueError:
                                continue
                unique_encoders = sorted(list(set(training_encoders)))
            else:
                # Fallback: use model's encoder count
                unique_encoders = list(range(getattr(model, 'num_encoders', 1)))
            
            # Ensure encoder_colors exists even if background plotting was skipped
            if 'encoder_colors' not in locals():
                try:
                    encoder_colors = {enc: plt.cm.tab10(int(enc) % 10) for enc in unique_encoders}
                except Exception:
                    encoder_colors = {enc: 'gray' for enc in unique_encoders}

            # Add training samples to legend
            for encoder in unique_encoders:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=encoder_colors.get(encoder, 'gray'), 
                                               markersize=8, label=f'Training Encoder {encoder}'))
            
            # Add support and query samples to legend for the evaluated key only
            if all_labels is not None:
                support_labels = []
                query_labels = []
                for key in actual_evaluated_keys:
                    support_labels.extend([l for l in all_labels if 'support' in l and key in l])
                    query_labels.extend([l for l in all_labels if 'query' in l and key in l])
                
                if support_labels:
                    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                   markerfacecolor='blue', 
                                                   markersize=8, label=f'Support Samples ({len(support_labels)})'))
                
                if query_labels:
                    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                                   markerfacecolor='red', 
                                                   markersize=8, label=f'Query Samples ({len(query_labels)})'))
            
            # Add trajectory elements
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor='green', 
                                           markersize=10, label='Trajectory Start'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                           markerfacecolor='red', 
                                           markersize=10, label='Trajectory End'))
            
            ax_latent.legend(handles=legend_elements, loc='upper right')
            ax_latent.grid(True, alpha=0.3)
        else:
            print(f"DEBUG: No trajectory_tsne_2d available for trajectory plotting")
            ax_latent.text(0.5, 0.5, 'No trajectory data', 
                          ha='center', va='center', transform=ax_latent.transAxes)
            ax_latent.set_title('Latent Trajectory')
    else:
        print(f"DEBUG: No z_vectors available for trajectory plotting")
        print(f"DEBUG: z_vectors available: {z_vectors is not None}")
        print(f"DEBUG: z_vectors length: {len(z_vectors) if z_vectors else 0}")
        ax_latent.text(0.5, 0.5, 'No trajectory data', 
                      ha='center', va='center', transform=ax_latent.transAxes)
        ax_latent.set_title('Latent Trajectory')
    
    # Plot loss progression
    if losses and len(losses) > 1:
        ax_loss.plot(losses, 'b-o', linewidth=2, markersize=4)
        ax_loss.set_title('Loss Progression')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.text(0.5, 0.5, 'No loss data', 
                    ha='center', va='center', transform=ax_loss.transAxes)
        ax_loss.set_title('Loss Progression')
    
    # Use safe extraction to avoid scalar conversion errors
    extract_reconstruction_grid = safe_extract_reconstruction_grid
    
    # Use stored reconstructions for different trajectory steps
    trajectory_reconstructions = trajectory_info.get('poe_trajectory_reconstructions', {})
    
    # Plot individual encoder reconstructions (ROW 1, dynamic columns)
    num_encoders = getattr(model, 'num_encoders', 1)
    encoder_axes = ax_enc_recons  # Use the dynamic list we created earlier
    
    for enc_idx in range(num_encoders):
        if enc_idx < len(encoder_axes) and encoder_axes[enc_idx] is not None:
            ax_enc = encoder_axes[enc_idx]
            
            # Try to get individual encoder reconstruction data
            encoder_reconstructions = trajectory_info.get('individual_encoder_reconstructions', {})
            print(f"DEBUG: encoder_reconstructions keys: {list(encoder_reconstructions.keys())}")
            
            if encoder_reconstructions and f'encoder_{enc_idx}' in encoder_reconstructions:
                # Use stored encoder reconstruction
                recon_data = encoder_reconstructions[f'encoder_{enc_idx}']
                print(f"DEBUG: encoder_{enc_idx} recon_data keys: {list(recon_data.keys()) if recon_data else 'None'}")
                
                if recon_data is not None and 'shape_logits' in recon_data and 'grid_logits' in recon_data:
                    shape_logits = recon_data['shape_logits']
                    grid_logits = recon_data['grid_logits']
                    
                    # Extract reconstruction grid from stored data
                    recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                    
                    if recon_grid is not None:
                        ax_enc.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        ax_enc.set_title(f'Encoder {enc_idx}\n{recon_rows}×{recon_cols}')
                    else:
                        print(f"DEBUG: encoder_{enc_idx} reconstruction grid extraction failed")
                        ax_enc.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                   transform=ax_enc.transAxes, fontsize=8)
                        ax_enc.set_title(f'Encoder {enc_idx}')
                else:
                    print(f"DEBUG: encoder_{enc_idx} missing shape_logits or grid_logits")
                    ax_enc.text(0.5, 0.5, f'Encoder {enc_idx}\nReconstruction\n(No Data)', 
                               ha='center', va='center', transform=ax_enc.transAxes, fontsize=8)
                    ax_enc.set_title(f'Encoder {enc_idx}')
            else:
                print(f"DEBUG: encoder_{enc_idx} not found in individual_encoder_reconstructions")
                # Try to generate reconstruction from latent vectors
                if z_vectors and len(z_vectors) > 0:
                    # Use the final z vector for encoder reconstruction
                    final_z = z_vectors[-1]
                    recon_grid, recon_rows, recon_cols = generate_reconstruction_from_latent(
                        final_z, model, device, input_seq, target_seq
                    )
                    if recon_grid is not None:
                        ax_enc.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        ax_enc.set_title(f'Encoder {enc_idx}\n{recon_rows}×{recon_cols}')
                    else:
                        ax_enc.text(0.5, 0.5, f'Encoder {enc_idx}\nReconstruction\n(Generated)', 
                                   ha='center', va='center', transform=ax_enc.transAxes, fontsize=8)
                        ax_enc.set_title(f'Encoder {enc_idx}')
                else:
                    ax_enc.text(0.5, 0.5, f'Encoder {enc_idx}\nReconstruction\n(Not Available)', 
                               ha='center', va='center', transform=ax_enc.transAxes, fontsize=8)
                    ax_enc.set_title(f'Encoder {enc_idx}')
            
            ax_enc.axis('off')
    
    # Plot POE reconstructions (ROW 1, columns 4-6)
    poe_axes = [ax_poe_initial, ax_poe_mid, ax_poe_final]
    poe_labels = ['Initial', 'Mid', 'Final']
    
    for i, (ax_poe, label) in enumerate(zip(poe_axes, poe_labels)):
        if trajectory_reconstructions and f'{label.lower()}' in trajectory_reconstructions and trajectory_reconstructions[f'{label.lower()}'] is not None:
            recon_data = trajectory_reconstructions[f'{label.lower()}']
            print(f"DEBUG: POE {label.lower()} recon_data keys: {list(recon_data.keys()) if recon_data else 'None'}")
            
            shape_logits = recon_data['shape_logits']
            grid_logits = recon_data['grid_logits']
            
            # Extract reconstruction grid from stored data
            recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
            
            if recon_grid is not None:
                ax_poe.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                ax_poe.set_title(f'POE {label}\n{recon_rows}×{recon_cols}')
            else:
                print(f"DEBUG: POE {label.lower()} reconstruction grid extraction failed")
                ax_poe.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                           transform=ax_poe.transAxes, fontsize=8)
                ax_poe.set_title(f'POE {label}')
        else:
            print(f"DEBUG: POE {label.lower()} not found in trajectory_reconstructions")
            # Try to generate reconstruction from latent vectors
            if z_vectors and len(z_vectors) > 0:
                # Select appropriate z vector based on label
                if label == 'Initial':
                    z_vector = z_vectors[0]
                elif label == 'Mid':
                    z_vector = z_vectors[len(z_vectors)//2]
                else:  # Final
                    z_vector = z_vectors[-1]
                
                recon_grid, recon_rows, recon_cols = generate_reconstruction_from_latent(
                    z_vector, model, device, input_seq, target_seq
                )
                if recon_grid is not None:
                    ax_poe.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                    ax_poe.set_title(f'POE {label}\n{recon_rows}×{recon_cols}')
                else:
                    ax_poe.text(0.5, 0.5, f'POE {label}\nReconstruction\n(Generated)', 
                               ha='center', va='center', transform=ax_poe.transAxes, fontsize=8)
                    ax_poe.set_title(f'POE {label}')
            else:
                ax_poe.text(0.5, 0.5, f'No Data\n{label}', ha='center', va='center', 
                           transform=ax_poe.transAxes, fontsize=8)
                ax_poe.set_title(f'POE {label}')
        
        ax_poe.axis('off')
    
    # Plot error maps (ROW 2)
    plot_error_maps_row(ax_error_maps, trajectory_info, model, num_encoders)
    
    # Plot query reconstructions (ROW 3, dynamic columns)
    query_encoder_axes = ax_query_recons  # Use the dynamic list we created earlier
    query_encoder_reconstructions = trajectory_info.get('query_encoder_reconstructions', {})
    print(f"DEBUG: query_encoder_reconstructions keys: {list(query_encoder_reconstructions.keys())}")
    
    for enc_idx in range(num_encoders):
        if enc_idx < len(query_encoder_axes) and query_encoder_axes[enc_idx] is not None:
            ax_query_enc = query_encoder_axes[enc_idx]
            
            # Try to get query encoder reconstruction data
            if query_encoder_reconstructions and f'encoder_{enc_idx}' in query_encoder_reconstructions:
                recon_data = query_encoder_reconstructions[f'encoder_{enc_idx}']
                print(f"DEBUG: query encoder_{enc_idx} recon_data keys: {list(recon_data.keys()) if recon_data else 'None'}")
                
                if recon_data is not None and 'shape_logits' in recon_data and 'grid_logits' in recon_data:
                    shape_logits = recon_data['shape_logits']
                    grid_logits = recon_data['grid_logits']
                    
                    # Extract reconstruction grid from stored data
                    recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                    
                    if recon_grid is not None:
                        ax_query_enc.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        ax_query_enc.set_title(f'Query Encoder {enc_idx}\n{recon_rows}×{recon_cols}')
                    else:
                        print(f"DEBUG: query encoder_{enc_idx} reconstruction grid extraction failed")
                        ax_query_enc.text(0.5, 0.5, f'Invalid\nDims', ha='center', va='center', 
                                       transform=ax_query_enc.transAxes, fontsize=8)
                        ax_query_enc.set_title(f'Query Encoder {enc_idx}')
                else:
                    print(f"DEBUG: query encoder_{enc_idx} missing shape_logits or grid_logits")
                    ax_query_enc.text(0.5, 0.5, f'Query Encoder {enc_idx}\nReconstruction\n(No Data)', 
                                   ha='center', va='center', transform=ax_query_enc.transAxes, fontsize=8)
                    ax_query_enc.set_title(f'Query Encoder {enc_idx}')
            else:
                print(f"DEBUG: query encoder_{enc_idx} not found in query_encoder_reconstructions")
                # Try to generate reconstruction from latent vectors
                if z_vectors and len(z_vectors) > 0:
                    # Use the final z vector for query encoder reconstruction
                    final_z = z_vectors[-1]
                    recon_grid, recon_rows, recon_cols = generate_reconstruction_from_latent(
                        final_z, model, device, input_seq, target_seq
                    )
                    if recon_grid is not None:
                        ax_query_enc.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                        ax_query_enc.set_title(f'Query Encoder {enc_idx}\n{recon_rows}×{recon_cols}')
                    else:
                        ax_query_enc.text(0.5, 0.5, f'Query Encoder {enc_idx}\nReconstruction\n(Generated)', 
                                       ha='center', va='center', transform=ax_query_enc.transAxes, fontsize=8)
                        ax_query_enc.set_title(f'Query Encoder {enc_idx}')
                else:
                    ax_query_enc.text(0.5, 0.5, f'Query Encoder {enc_idx}\nReconstruction\n(Not Available)', 
                                   ha='center', va='center', transform=ax_query_enc.transAxes, fontsize=8)
                    ax_query_enc.set_title(f'Query Encoder {enc_idx}')
            
            ax_query_enc.axis('off')
    
    # Plot query POE reconstructions (ROW 3, columns 5-6)
    query_poe_reconstructions = trajectory_info.get('query_poe_reconstructions', {})
    print(f"DEBUG: query_poe_reconstructions keys: {list(query_poe_reconstructions.keys())}")
    
    # Initial POE reconstruction
    if query_poe_reconstructions and 'initial' in query_poe_reconstructions:
        recon_data = query_poe_reconstructions['initial']
        print(f"DEBUG: query POE initial recon_data keys: {list(recon_data.keys()) if recon_data else 'None'}")
        
        shape_logits = recon_data['shape_logits']
        grid_logits = recon_data['grid_logits']
        
        recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
        
        if recon_grid is not None:
            ax_query_poe_initial.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
            ax_query_poe_initial.set_title(f'Query POE Initial\n{recon_rows}×{recon_cols}')
        else:
            print(f"DEBUG: query POE initial reconstruction grid extraction failed")
            ax_query_poe_initial.text(0.5, 0.5, 'Invalid\nDims', ha='center', va='center', 
                                     transform=ax_query_poe_initial.transAxes, fontsize=8)
            ax_query_poe_initial.set_title('Query POE Initial')
    else:
        print(f"DEBUG: query POE initial not found in query_poe_reconstructions")
        # Try to generate reconstruction from latent vectors
        if z_vectors and len(z_vectors) > 0:
            # Use the initial z vector for query POE initial reconstruction
            initial_z = z_vectors[0]
            recon_grid, recon_rows, recon_cols = generate_reconstruction_from_latent(
                initial_z, model, device, input_seq, target_seq
            )
            if recon_grid is not None:
                ax_query_poe_initial.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                ax_query_poe_initial.set_title(f'Query POE Initial\n{recon_rows}×{recon_cols}')
            else:
                ax_query_poe_initial.text(0.5, 0.5, 'Query POE\nInitial\n(Generated)', 
                                         ha='center', va='center', transform=ax_query_poe_initial.transAxes, fontsize=8)
                ax_query_poe_initial.set_title('Query POE Initial')
        else:
            ax_query_poe_initial.text(0.5, 0.5, 'Query POE\nInitial\n(Not Available)', 
                                     ha='center', va='center', transform=ax_query_poe_initial.transAxes, fontsize=8)
            ax_query_poe_initial.set_title('Query POE Initial')
    ax_query_poe_initial.axis('off')
    
    # Final POE reconstruction
    if query_poe_reconstructions and 'final' in query_poe_reconstructions:
        recon_data = query_poe_reconstructions['final']
        print(f"DEBUG: query POE final recon_data keys: {list(recon_data.keys()) if recon_data else 'None'}")
        
        shape_logits = recon_data['shape_logits']
        grid_logits = recon_data['grid_logits']
        
        recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
        
        if recon_grid is not None:
            ax_query_poe_final.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
            ax_query_poe_final.set_title(f'Query POE Final\n{recon_rows}×{recon_cols}')
        else:
            print(f"DEBUG: query POE final reconstruction grid extraction failed")
            ax_query_poe_final.text(0.5, 0.5, 'Invalid\nDims', ha='center', va='center', 
                                   transform=ax_query_poe_final.transAxes, fontsize=8)
            ax_query_poe_final.set_title('Query POE Final')
    else:
        print(f"DEBUG: query POE final not found in query_poe_reconstructions")
        # Try to generate reconstruction from latent vectors
        if z_vectors and len(z_vectors) > 0:
            # Use the final z vector for query POE final reconstruction
            final_z = z_vectors[-1]
            recon_grid, recon_rows, recon_cols = generate_reconstruction_from_latent(
                final_z, model, device, input_seq, target_seq
            )
            if recon_grid is not None:
                ax_query_poe_final.imshow(recon_grid, cmap='viridis', interpolation='nearest', aspect='equal')
                ax_query_poe_final.set_title(f'Query POE Final\n{recon_rows}×{recon_cols}')
            else:
                ax_query_poe_final.text(0.5, 0.5, 'Query POE\nFinal\n(Generated)', 
                                       ha='center', va='center', transform=ax_query_poe_final.transAxes, fontsize=8)
                ax_query_poe_final.set_title('Query POE Final')
        else:
            ax_query_poe_final.text(0.5, 0.5, 'Query POE\nFinal\n(Not Available)', 
                                   ha='center', va='center', transform=ax_query_poe_final.transAxes, fontsize=8)
            ax_query_poe_final.set_title('Query POE Final')
    ax_query_poe_final.axis('off')
    
    # Plot query error maps (ROW 4)
    plot_query_error_maps_row(ax_query_error_maps, trajectory_info, model, num_encoders)
    
    # Get decoder type for title (using global settings import)
    eval_settings = settings.get_evaluation_settings()
    decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
    decoder_display = "Independent Decoders" if decoder_type == "independent" else "Shared Decoder"
    
    # Add data summary to title (filtered by evaluated key)
    training_count = len([l for l in training_labels if 'training' in l]) if training_labels else 0
    support_count = 0
    query_count = 0
    for key in actual_evaluated_keys:
        support_count += len([l for l in all_labels if 'support' in l and key in l]) if all_labels else 0
        query_count += len([l for l in all_labels if 'query' in l and key in l]) if all_labels else 0
    data_summary = f"Training: {training_count}, Support: {support_count}, Query: {query_count}"
    
    ood_indicator = ""
    if ood_enabled or any('ood_' in label for label in all_labels if label):
        ood_indicator = " (OOD SAMPLES)"
        print(f"DEBUG: OOD sampling enabled, adding OOD indicator to title: {ood_indicator}")
    
    plt.suptitle(f'Multi-Encoder Trajectory Analysis{ood_indicator}\nEVALUATION DATA - Trajectory Reconstructions: {decoder_display}\nData: {data_summary}', fontsize=16)
    
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"DEBUG: tight_layout failed, using default layout: {e}")
        # Fallback to default layout if tight_layout fails
        pass
    
    # Debug: Print the save path
    print(f"DEBUG: visualize_multi_encoder_comprehensive_trajectory saving to: {save_path}")
    print(f"DEBUG: Directory exists: {os.path.exists(os.path.dirname(save_path))}")
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"DEBUG: savefig failed: {e}")
        # Try without bbox_inches if it fails
        try:
            plt.savefig(save_path, dpi=150)
        except Exception as e2:
            print(f"DEBUG: savefig without bbox_inches also failed: {e2}")
            return None
    
    plt.close()
    print(f"[ OK ] Saved optimized multi-encoder trajectory visualization to: {save_path}")

def create_error_map(target_grid, reconstruction_grid, title="Error Map"):
    """
    Create an error map showing the difference between target and reconstruction.
    Returns the error map and a boolean indicating if it was successfully created.
    """
    try:
        # Ensure grids have the same dimensions
        if target_grid.shape == reconstruction_grid.shape:
            # Calculate error as difference between target and reconstruction
            error_map = target_grid - reconstruction_grid
            
            # Create the error map visualization
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
            
            # Use a diverging colormap for error visualization (red for positive, blue for negative)
            im = ax.imshow(error_map, cmap='RdBu_r', interpolation='nearest', aspect='equal', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Error (Target - Recon)')
            
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
            # Convert to image array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img_array, True
        else:
            # Dimensions don't match, return None
            return None, False
    except Exception as e:
        print(f"Error creating error map: {e}")
        return None, False

def plot_error_maps_row(ax_error_maps, trajectory_info, model, num_encoders):
    """
    Plot error maps for all encoders and POE reconstructions in a single row.
    """
    # Get input and target grids
    input_seq = trajectory_info['input_sample']
    target_seq = trajectory_info['target_sample']
    input_grid, input_shape = extract_grid_from_sequence(input_seq)
    target_grid, target_shape = extract_grid_from_sequence(target_seq)
    
    # Note: Input and target grids may have different shapes in ARC.
    # Only require reconstruction grids to match target grid when computing error maps.
    
    # Get trajectory reconstructions
    trajectory_reconstructions = trajectory_info.get('poe_trajectory_reconstructions', {})
    individual_encoder_reconstructions = trajectory_info.get('individual_encoder_reconstructions', {})
    
    # Calculate number of error maps to show
    num_error_maps = num_encoders + 3  # encoders + initial/mid/final POE
    
    # Create subplots for error maps
    if num_error_maps <= 8:  # Can fit in one row
        cols = num_error_maps
        rows = 1
    else:  # Need multiple rows
        cols = 8
        rows = (num_error_maps + 7) // 8
    
    # Create GridSpec for error maps
    gs_error = ax_error_maps.get_gridspec()
    fig = ax_error_maps.figure
    
    # Remove the original ax_error_maps
    ax_error_maps.remove()
    
    # Create subplots for error maps directly below their corresponding grids
    error_axes = []
    
    # Create individual error map subplots positioned below each reconstruction
    # Encoder error maps (positioned below encoder reconstructions - dynamic columns)
    for enc_idx in range(num_encoders):
        ax = fig.add_subplot(gs_error[2, 1 + enc_idx])  # Row 2, dynamic columns
        error_axes.append(ax)
    
    # POE error maps (positioned below POE reconstructions - dynamic columns)
    poe_start_col = 1 + num_encoders
    for poe_idx in range(3):  # initial, mid, final
        if num_encoders + poe_idx < num_error_maps:
            ax = fig.add_subplot(gs_error[2, poe_start_col + poe_idx])  # Row 2, dynamic columns
            error_axes.append(ax)
    
    # Use consistent color scheme for all error maps
    consistent_cmap = 'RdBu_r'  # Red-Blue diverging colormap
    
    # Plot encoder error maps
    for enc_idx in range(num_encoders):
        if enc_idx < len(error_axes):
            ax = error_axes[enc_idx]
            
            # Try to get encoder reconstruction
            if individual_encoder_reconstructions and f'encoder_{enc_idx}' in individual_encoder_reconstructions:
                recon_data = individual_encoder_reconstructions[f'encoder_{enc_idx}']
                shape_logits = recon_data['shape_logits']
                grid_logits = recon_data['grid_logits']
                
                recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                
                if recon_grid is not None and recon_grid.shape == target_grid.shape:
                    # Create error map with consistent color scheme
                    error_map = target_grid - recon_grid
                    
                    # Calculate error statistics for consistent visualization
                    error_range = max(abs(error_map.min()), abs(error_map.max()))
                    vmin, vmax = -error_range, error_range
                    
                    im = ax.imshow(error_map, cmap=consistent_cmap, interpolation='nearest', aspect='equal', vmin=vmin, vmax=vmax)
                    
                    # Calculate error metrics
                    mae = np.mean(np.abs(error_map))
                    mse = np.mean(error_map**2)
                    
                    ax.set_title(f'Encoder {enc_idx}\nMAE:{mae:.3f} MSE:{mse:.3f}', fontsize=8)
                    ax.axis('off')
                else:
                    # Distinguish between missing reconstruction vs shape mismatch
                    msg = 'No Recon' if recon_grid is None else 'Wrong Shape'
                    ax.text(0.5, 0.5, f'Encoder {enc_idx}\n{msg}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.set_title(f'Encoder {enc_idx}')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Encoder {enc_idx}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'Encoder {enc_idx}')
                ax.axis('off')
    
    # Plot POE error maps (initial, mid, final)
    poe_labels = ['initial', 'mid', 'final']
    for i, label in enumerate(poe_labels):
        if num_encoders + i < len(error_axes):
            ax = error_axes[num_encoders + i]
            
            if trajectory_reconstructions and label in trajectory_reconstructions:
                recon_data = trajectory_reconstructions[label]
                shape_logits = recon_data['shape_logits']
                grid_logits = recon_data['grid_logits']
                
                recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                
                if recon_grid is not None and recon_grid.shape == target_grid.shape:
                    # Create error map with consistent color scheme
                    error_map = target_grid - recon_grid
                    
                    # Calculate error statistics for consistent visualization
                    error_range = max(abs(error_map.min()), abs(error_map.max()))
                    vmin, vmax = -error_range, error_range
                    
                    im = ax.imshow(error_map, cmap=consistent_cmap, interpolation='nearest', aspect='equal', vmin=vmin, vmax=vmax)
                    
                    # Calculate error metrics
                    mae = np.mean(np.abs(error_map))
                    mse = np.mean(error_map**2)
                    
                    ax.set_title(f'POE {label.title()}\nMAE:{mae:.3f} MSE:{mse:.3f}', fontsize=8)
                    ax.axis('off')
                else:
                    msg = 'No Recon' if recon_grid is None else 'Wrong Shape'
                    ax.text(0.5, 0.5, f'POE {label.title()}\n{msg}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.set_title(f'POE {label.title()}')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'POE {label.title()}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'POE {label.title()}')
                ax.axis('off')
    
    # Add overall title
    fig.text(0.5, 0.02, 'Error Maps (Target - Reconstruction)', ha='center', va='bottom', fontsize=12)

def plot_query_error_maps_row(ax_query_error_maps, trajectory_info, model, num_encoders):
    """
    Plot error maps for query reconstructions (encoders and POE).
    """
    # Get query input and target grids
    query_input = trajectory_info.get('query_input')
    query_target = trajectory_info.get('query_target')
    
    if query_input is None or query_target is None:
        ax_query_error_maps.text(0.5, 0.5, 'No query data available', 
                                ha='center', va='center', transform=ax_query_error_maps.transAxes, fontsize=10)
        ax_query_error_maps.set_title('Query Error Maps (No Data)')
        ax_query_error_maps.axis('off')
        return
    
    input_grid, input_shape = extract_grid_from_sequence(query_input)
    target_grid, target_shape = extract_grid_from_sequence(query_target)
    
    # Note: Query input and target grids may have different shapes in ARC.
    # Only require reconstruction grids to match query target grid when computing error maps.
    
    # Get query reconstructions
    query_encoder_reconstructions = trajectory_info.get('query_encoder_reconstructions', {})
    query_poe_reconstructions = trajectory_info.get('query_poe_reconstructions', {})
    
    # Calculate number of error maps to show
    num_error_maps = num_encoders + 2  # encoders + initial/final POE
    
    # Create subplots for error maps
    if num_error_maps <= 8:  # Can fit in one row
        cols = num_error_maps
        rows = 1
    else:  # Need multiple rows
        cols = 8
        rows = (num_error_maps + 7) // 8
    
    # Create GridSpec for error maps
    gs_error = ax_query_error_maps.get_gridspec()
    fig = ax_query_error_maps.figure
    
    # Remove the original ax_query_error_maps
    ax_query_error_maps.remove()
    
    # Create subplots for query error maps directly below their corresponding grids
    error_axes = []
    
    # Create individual error map subplots positioned below each query reconstruction
    # Query encoder error maps (positioned below query encoder reconstructions - dynamic columns)
    for enc_idx in range(num_encoders):
        ax = fig.add_subplot(gs_error[4, 2 + enc_idx])  # Row 4, dynamic columns
        error_axes.append(ax)
    
    # Query POE error maps (positioned below query POE reconstructions - dynamic columns)
    query_poe_start_col = 2 + num_encoders
    for poe_idx in range(2):  # initial/final only for query
        if num_encoders + poe_idx < num_error_maps:
            ax = fig.add_subplot(gs_error[4, query_poe_start_col + poe_idx])  # Row 4, dynamic columns
            error_axes.append(ax)
    
    # Use consistent color scheme for all query error maps
    consistent_cmap = 'RdBu_r'  # Red-Blue diverging colormap
    
    # Plot query encoder error maps
    for enc_idx in range(num_encoders):
        if enc_idx < len(error_axes):
            ax = error_axes[enc_idx]
            
            # Try to get query encoder reconstruction
            if query_encoder_reconstructions and f'encoder_{enc_idx}' in query_encoder_reconstructions:
                recon_data = query_encoder_reconstructions[f'encoder_{enc_idx}']
                shape_logits = recon_data['shape_logits']
                grid_logits = recon_data['grid_logits']
                
                recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                
                if recon_grid is not None and recon_grid.shape == target_grid.shape:
                    # Create error map with consistent color scheme
                    error_map = target_grid - recon_grid
                    
                    # Calculate error statistics for consistent visualization
                    error_range = max(abs(error_map.min()), abs(error_map.max()))
                    vmin, vmax = -error_range, error_range
                    
                    im = ax.imshow(error_map, cmap=consistent_cmap, interpolation='nearest', aspect='equal', vmin=vmin, vmax=vmax)
                    
                    # Calculate error metrics
                    mae = np.mean(np.abs(error_map))
                    mse = np.mean(error_map**2)
                    
                    ax.set_title(f'Query Enc {enc_idx}\nMAE:{mae:.3f} MSE:{mse:.3f}', fontsize=8)
                    ax.axis('off')
                else:
                    msg = 'No Recon' if recon_grid is None else 'Wrong Shape'
                    ax.text(0.5, 0.5, f'Query Encoder {enc_idx}\n{msg}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.set_title(f'Query Encoder {enc_idx}')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Query Encoder {enc_idx}\nNo Data', 
                       ha='center', va='center', transform=ax_query_error_maps.transAxes, fontsize=8)
                ax.set_title(f'Query Encoder {enc_idx}')
                ax.axis('off')
    
    # Plot query POE error maps (initial, final)
    poe_labels = ['initial', 'final']
    for i, label in enumerate(poe_labels):
        if num_encoders + i < len(error_axes):
            ax = error_axes[num_encoders + i]
            
            if query_poe_reconstructions and label in query_poe_reconstructions:
                recon_data = query_poe_reconstructions[label]
                shape_logits = recon_data['shape_logits']
                grid_logits = recon_data['grid_logits']
                
                recon_grid, recon_rows, recon_cols = extract_reconstruction_grid(shape_logits, grid_logits)
                
                if recon_grid is not None and recon_grid.shape == target_grid.shape:
                    # Create error map with consistent color scheme
                    error_map = target_grid - recon_grid
                    
                    # Calculate error statistics for consistent visualization
                    error_range = max(abs(error_map.min()), abs(error_map.max()))
                    vmin, vmax = -error_range, error_range
                    
                    im = ax.imshow(error_map, cmap=consistent_cmap, interpolation='nearest', aspect='equal', vmin=vmin, vmax=vmax)
                    
                    # Calculate error metrics
                    mae = np.mean(np.abs(error_map))
                    mse = np.mean(error_map**2)
                    
                    ax.set_title(f'Query POE {label.title()}\nMAE:{mae:.3f} MSE:{mse:.3f}', fontsize=8)
                    ax.axis('off')
                else:
                    msg = 'No Recon' if recon_grid is None else 'Wrong Shape'
                    ax.text(0.5, 0.5, f'Query POE {label.title()}\n{msg}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.set_title(f'Query POE {label.title()}')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Query POE {label.title()}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'Query POE {label.title()}')
                ax.axis('off')
    
    # Add overall title
    fig.text(0.5, 0.02, 'Query Error Maps (Target - Reconstruction)', ha='center', va='bottom', fontsize=12)

def load_unified_latent_data_with_trajectory(run_dir, model, device, trajectory_info, eval_results=None, evaluated_key=None, ood_enabled=False, ood_task_keys=None):
    """
    Load ALL latent vectors (training + support + query + trajectory) and apply unified t-SNE.
    This ensures all points are plotted using the same t-SNE transformation for consistency.
    
    ✅ FIXED: Now uses REAL optimized latents from training instead of pre-saved sequences.
    """
    print("Loading unified latent data (training + support + query + trajectory)...")
    
    # Get number of encoders from model
    num_encoders = getattr(model, 'num_encoders', 1)
    
    # ✅ FIX: Use REAL optimized latents from training instead of pre-saved sequences
    all_latents = []
    all_labels = []
    all_colors = []
    
    # ✅ FIX: Use passed evaluated_key parameter instead of extracting from trajectory_info
    if evaluated_key is None:
        # Fallback: try to extract from trajectory_info
        if isinstance(trajectory_info, dict):
            evaluated_key = trajectory_info.get('evaluated_key', '')
        elif isinstance(trajectory_info, list) and len(trajectory_info) > 0:
            if isinstance(trajectory_info[0], dict):
                evaluated_key = trajectory_info[0].get('evaluated_key', '')
    
    actual_evaluated_keys = []
    if ood_enabled and ood_task_keys:
        actual_evaluated_keys = ood_task_keys
        print(f"DEBUG: Using OOD task keys for evaluation: {actual_evaluated_keys}")
    else:
        actual_evaluated_keys = [evaluated_key] if evaluated_key else []
        print(f"DEBUG: Using evaluation key: {evaluated_key}")
    
    print(f"DEBUG: Evaluating keys: {actual_evaluated_keys}")
    
    # ✅ FIX: Use stored optimized latents from training (REAL SAMPLES USED IN TRAINING!)
    if hasattr(model, 'epoch_optimized_latents') and model.epoch_optimized_latents:
        training_latents = model.epoch_optimized_latents['latents']
        training_keys = model.epoch_optimized_latents['keys']
        training_encoder_indices = model.epoch_optimized_latents['encoder_indices']
        
        print(f"[ OK ] Using {len(training_latents)} REAL OPTIMIZED latents from training (actual samples used in epoch)")
        
        # Process REAL training latents
        for i, (latent, key, encoder_idx) in enumerate(zip(training_latents, training_keys, training_encoder_indices)):
            all_latents.append(latent)
            all_labels.append(f"training_enc_{encoder_idx}_key_{key}")
            
            # Color based on whether this key matches any of the actual evaluated keys
            if key in actual_evaluated_keys:
                all_colors.append(plt.cm.Set1(encoder_idx if encoder_idx is not None else 0))  # Bright color for matching key
            else:
                all_colors.append(plt.cm.tab10((encoder_idx if encoder_idx is not None else 0) % 10))  # More visible color for other keys
    else:
        print("[ WARNING ] No stored optimized latents found, falling back to pre-saved sequences")
        
        # Fallback to the old method (but this should rarely happen)
        results_file = os.path.join(run_dir, "results.pkl")
        if not os.path.exists(results_file):
            print("[ WARNING ] Warning: No results.pkl found — proceeding without training latents")
            # Do NOT return; continue to build unified dataset from eval_results and trajectory only
            training_latents = None
        else:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            # Get actual training data from results
            input_sequences = results.get('input_sequences', None)
            output_sequences = results.get('output_sequences', None)
            key_list = results.get('key_list', None)
            if input_sequences is None:
                print("[ WARNING ] Warning: No training sequences found in results")
            else:
                print(f"[ OK ] Found training data: {len(input_sequences)} sequences with keys: {key_list[:5]}...")
                # Process training data
                print("Computing training latent vectors...")
                model.eval()
                with torch.no_grad():
                    for i, (input_seq, output_seq, key) in enumerate(zip(input_sequences, output_sequences, key_list)):
                        input_tensor = torch.tensor(input_seq, dtype=torch.float32, device=device).unsqueeze(0)
                        output_tensor = torch.tensor(output_seq, dtype=torch.float32, device=device).unsqueeze(0)
                        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                            assigned_encoder = 0
                            if hasattr(model, 'training_metadata') and model.training_metadata:
                                key_to_encoder_mapping = model.training_metadata.get('key_to_encoder_mapping', {})
                                for enc_idx, keys in key_to_encoder_mapping.items():
                                    if key in keys:
                                        assigned_encoder = enc_idx
                                        break
                            mu, log_var, _ = model.multi_encoder.encoders[assigned_encoder](input_tensor, output_tensor)
                            z = model.reparameterize(mu, log_var)
                            z_numpy = z.cpu().numpy().flatten()
                            all_latents.append(z_numpy)
                            all_labels.append(f"training_enc_{assigned_encoder}_key_{key}")
                            all_colors.append(plt.cm.tab10(assigned_encoder % 10))
                        else:
                            mu, log_var, _ = model.encoder(input_tensor, output_tensor)
                            z = model.reparameterize(mu, log_var)
                            z_numpy = z.cpu().numpy().flatten()
                            all_latents.append(z_numpy)
                            all_labels.append(f"training_enc_0_key_{key}")
                            all_colors.append(plt.cm.tab10(0))
    
    print(f"DEBUG: Training data summary - Total: {len(all_latents)}, Matching key '{evaluated_key}': {sum(1 for key in training_keys if key == evaluated_key) if 'training_keys' in locals() else 0}")
    
    # Load support and query data from evaluation results
    print("Loading support and query latent vectors from evaluation results...")
    try:
        # Use provided eval_results if available, otherwise load from file
        if eval_results is not None:
            print("DEBUG: Using provided evaluation results")
            key_results = eval_results.get('key_results', {})
            print(f"DEBUG: Using provided evaluation results for {len(key_results)} keys")
        else:
            # Load evaluation results from file
            eval_results_file = os.path.join(run_dir, 'evaluation_results.pkl')
            if os.path.exists(eval_results_file):
                with open(eval_results_file, 'rb') as f:
                    eval_results = pickle.load(f)
                key_results = eval_results.get('key_results', {})
                print(f"DEBUG: Found evaluation results for {len(key_results)} keys")
            else:
                print("DEBUG: No evaluation results file found")
                key_results = {}
        
        # Load support and query latents for the evaluated key
        if evaluated_key in key_results:
            key_data = key_results[evaluated_key]
            print(f"DEBUG: Found data for evaluated key '{evaluated_key}'")
            
            ood_task_keys = []
            if 'raw_data' in key_data and 'ood_task_keys' in key_data['raw_data']:
                ood_task_keys = key_data['raw_data']['ood_task_keys']
                print(f"DEBUG: Using OOD task keys: {ood_task_keys}")
            
            # ✅ FIX: Use OOD task keys for labeling instead of evaluation key
            if ood_task_keys and len(ood_task_keys) > 0 and ood_enabled:
                support_label = f"support_ood_{ood_task_keys[0]}"  # Use first OOD task key
                query_label = f"query_ood_{ood_task_keys[0]}"      # Use first OOD task key
                print(f"DEBUG: Using OOD labels - support: {support_label}, query: {query_label}")
            else:
                support_label = f"support_{evaluated_key}"
                query_label = f"query_{evaluated_key}"
                print(f"DEBUG: Using evaluation key labels - support: {support_label}, query: {query_label}")
            

            # Extract support and query latent data
            support_query_latents = []
            support_query_labels = []
            support_query_colors = []
            
            # Process support samples
            if 'evaluation_latent_data' in key_data:
                eval_latent_data = key_data['evaluation_latent_data']
                print(f"DEBUG: evaluation_latent_data keys: {list(eval_latent_data.keys())}")
                
                # Check if this is task optimization format (has support_latents/query_latents)
                if 'support_latents' in eval_latent_data and 'query_latents' in eval_latent_data:
                    print("DEBUG: Using task optimization format")
                    
                    # Extract support latents (task optimization format)
                    support_latents = eval_latent_data['support_latents']
                    print(f"DEBUG: Found {len(support_latents)} support latents (task optimization)")
                    for i, latent in enumerate(support_latents):
                        # Handle different data types (tensor, numpy array, or dict)
                        if isinstance(latent, dict):
                            # If it's a dict, look for common keys
                            if 'latent_z' in latent:
                                latent_data = latent['latent_z']
                            elif 'z' in latent:
                                latent_data = latent['z']
                            elif 'latent' in latent:
                                latent_data = latent['latent']
                            else:
                                print(f"DEBUG: Unexpected dict structure in support latent {i}: {list(latent.keys())}")
                                continue
                        else:
                            latent_data = latent
                        
                        # Convert to numpy and flatten
                        if hasattr(latent_data, 'cpu'):
                            latent_numpy = latent_data.cpu().numpy().flatten()
                        elif hasattr(latent_data, 'flatten'):
                            latent_numpy = latent_data.flatten()
                        else:
                            latent_numpy = np.array(latent_data).flatten()
                        
                        support_query_latents.append(latent_numpy)
                        support_query_labels.append(support_label)
                        support_query_colors.append('blue')
                    
                    # Extract query latents (task optimization format)
                    query_latents = eval_latent_data['query_latents']
                    print(f"DEBUG: Found {len(query_latents)} query latents (task optimization)")
                    for i, latent in enumerate(query_latents):
                        # Handle different data types (tensor, numpy array, or dict)
                        if isinstance(latent, dict):
                            # If it's a dict, look for common keys
                            if 'latent_z' in latent:
                                latent_data = latent['latent_z']
                            elif 'z' in latent:
                                latent_data = latent['z']
                            elif 'latent' in latent:
                                latent_data = latent['latent']
                            else:
                                print(f"DEBUG: Unexpected dict structure in query latent {i}: {list(latent.keys())}")
                                continue
                        else:
                            latent_data = latent
                        
                        # Convert to numpy and flatten
                        if hasattr(latent_data, 'cpu'):
                            latent_numpy = latent_data.cpu().numpy().flatten()
                        elif hasattr(latent_data, 'flatten'):
                            latent_numpy = latent_data.flatten()
                        else:
                            latent_numpy = np.array(latent_data).flatten()
                        
                        support_query_latents.append(latent_numpy)
                        support_query_labels.append(query_label)
                        support_query_colors.append('red')
                
                # Process support samples (regular evaluation format)
                elif 'support' in eval_latent_data:
                    support_data = eval_latent_data['support']
                    print(f"DEBUG: Processing support data with keys: {list(support_data.keys())}")
                    
                    # Extract PoE support latents
                    if 'poe' in support_data and 'latent_zs' in support_data['poe']:
                        poe_support_latents = support_data['poe']['latent_zs']
                        print(f"DEBUG: Found {len(poe_support_latents)} PoE support latents")
                        
                        for i, latent in enumerate(poe_support_latents):
                            if hasattr(latent, 'cpu'):
                                latent_numpy = latent.cpu().numpy().flatten()
                            else:
                                latent_numpy = latent.flatten()
                            support_query_latents.append(latent_numpy)
                            support_query_labels.append(support_label)
                            support_query_colors.append('blue')
                    
                    # Extract individual encoder support latents
                    for enc_idx in range(num_encoders):
                        enc_key = f'encoder_{enc_idx}'
                        if enc_key in support_data and 'latent_zs' in support_data[enc_key]:
                            enc_support_latents = support_data[enc_key]['latent_zs']
                            print(f"DEBUG: Found {len(enc_support_latents)} {enc_key} support latents")
                            
                            for i, latent in enumerate(enc_support_latents):
                                if hasattr(latent, 'cpu'):
                                    latent_numpy = latent.cpu().numpy().flatten()
                                else:
                                    latent_numpy = latent.flatten()
                                support_query_latents.append(latent_numpy)
                                support_query_labels.append(support_label)
                                support_query_colors.append('blue')
                
                # Process query samples
                if 'query' in eval_latent_data:
                    query_data = eval_latent_data['query']
                    print(f"DEBUG: Processing query data with keys: {list(query_data.keys())}")
                    
                    # Extract PoE query latents
                    if 'poe' in query_data and 'latent_zs' in query_data['poe']:
                        poe_query_latents = query_data['poe']['latent_zs']
                        print(f"DEBUG: Found {len(poe_query_latents)} PoE query latents")
                        
                        for i, latent in enumerate(poe_query_latents):
                            if hasattr(latent, 'cpu'):
                                latent_numpy = latent.cpu().numpy().flatten()
                            else:
                                latent_numpy = latent.flatten()
                            support_query_latents.append(latent_numpy)
                            support_query_labels.append(query_label)
                            support_query_colors.append('red')
                    
                    # Extract individual encoder query latents
                    for enc_idx in range(num_encoders):
                        enc_key = f'encoder_{enc_idx}'
                        if enc_key in query_data and 'latent_zs' in query_data[enc_key]:
                            enc_query_latents = query_data[enc_key]['latent_zs']
                            print(f"DEBUG: Found {len(enc_query_latents)} {enc_key} query latents")
                            
                            for i, latent in enumerate(enc_query_latents):
                                if hasattr(latent, 'cpu'):
                                    latent_numpy = latent.cpu().numpy().flatten()
                                else:
                                    latent_numpy = latent.flatten()
                                support_query_latents.append(latent_numpy)
                                support_query_labels.append(query_label)
                                support_query_colors.append('red')
            
            # Add support/query latents to the unified dataset
            if support_query_latents:
                print(f"DEBUG: Adding {len(support_query_latents)} support/query samples from evaluation results")
                all_latents.extend(support_query_latents)
                all_labels.extend(support_query_labels)
                all_colors.extend(support_query_colors)
                print(f"[ OK ] Added {len(support_query_latents)} support/query samples from evaluation results")
            else:
                print(f"[ WARNING ] No support/query latents found for key '{evaluated_key}'")
        else:
            print(f"[ WARNING ] No evaluation data found for key '{evaluated_key}'")
    except Exception as e:
        print(f"[ WARNING ] Warning: Could not load support/query data from evaluation results: {e}")
        import traceback
        traceback.print_exc()
    
    
    # Add trajectory points to unified dataset (show all points, no limiting)
    print("Adding trajectory latent vectors...")
    z_vectors = trajectory_info.get('z_vectors', [])
    losses = trajectory_info.get('losses', [])  # ✅ FIX: Get losses from trajectory_info
    
    if z_vectors:
        print(f"DEBUG: Processing {len(z_vectors)} trajectory vectors, showing all points")
        
        # ✅ FIX: Show all trajectory points without limiting
        labels = [f'step{i}' for i in range(len(z_vectors))]
        print(f"DEBUG: Using all {len(z_vectors)} trajectory points")
        
        for i, (label, z_vec) in enumerate(zip(labels, z_vectors)):
            # Handle both tensor and numpy array cases
            if hasattr(z_vec, 'cpu'):
                # Tensor case
                z_numpy = z_vec.cpu().numpy().flatten()
            else:
                # Numpy array case
                z_numpy = z_vec.flatten()
            all_latents.append(z_numpy)
            all_labels.append(f"trajectory_{label}")
            all_colors.append('red')  # Red for trajectory points
        print(f"[ OK ] Added {len(labels)} trajectory points: {labels}")
    
    # Convert to numpy arrays - ensure all vectors have the same shape and are numeric
    print(f"DEBUG: Processing {len(all_latents)} latent vectors...")
    
    # First, ensure all latents are proper numpy arrays with numeric data
    cleaned_latents = []
    
    for i, latent in enumerate(all_latents):
        
        try:
            # Convert to numpy array if it isn't already
            if not isinstance(latent, np.ndarray):
                latent = np.array(latent)
            
            # Ensure it's numeric
            if not np.issubdtype(latent.dtype, np.number):
                # Try to convert to float
                latent = latent.astype(np.float32)
            
            # Ensure it's 1D
            if latent.ndim > 1:
                latent = latent.flatten()
            
            cleaned_latents.append(latent)
            
        except Exception as e:
            print(f"WARNING: Error processing latent {i}: {e}")
            # Skip problematic latents
            continue
    
    if not cleaned_latents:
        print("ERROR: No valid latent vectors found!")
        return None, None, None, None, None, None, None
    
    # Check shapes and find the most common shape
    shapes = [latent.shape for latent in cleaned_latents]
    shape_counts = {}
    for shape in shapes:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    # Use the most common shape as target
    target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
    
    # Ensure all vectors have the target shape
    normalized_latents = []
    for i, latent in enumerate(cleaned_latents):
        if latent.shape != target_shape:
            # Try to reshape or pad/truncate to match target shape
            if len(latent.shape) == 1 and len(target_shape) == 1:
                if latent.shape[0] > target_shape[0]:
                    # Truncate
                    latent = latent[:target_shape[0]]
                elif latent.shape[0] < target_shape[0]:
                    # Pad with zeros
                    padded = np.zeros(target_shape[0], dtype=latent.dtype)
                    padded[:latent.shape[0]] = latent
                    latent = padded
            else:
                # For multi-dimensional, flatten and reshape
                flattened = latent.flatten()
                if len(flattened) > np.prod(target_shape):
                    flattened = flattened[:int(np.prod(target_shape))]
                elif len(flattened) < np.prod(target_shape):
                    padded = np.zeros(int(np.prod(target_shape)), dtype=latent.dtype)
                    padded[:len(flattened)] = flattened
                    flattened = padded
                latent = flattened.reshape(target_shape)
        normalized_latents.append(latent)
    
    # Convert to numpy array with explicit dtype
    try:
        all_latents = np.array(normalized_latents, dtype=np.float32)
    except Exception as e:
        print(f"WARNING: Error creating numpy array: {e}")
        # Fallback: try without dtype specification
        try:
            all_latents = np.array(normalized_latents)
        except Exception as e3:
            print(f"ERROR: Even fallback failed: {e3}")
            return None, None, None, None, None, None, None
    
    # Ensure colors and labels lists match the cleaned latents
    if len(all_colors) != len(normalized_latents):
        # Truncate or extend colors list to match
        if len(all_colors) > len(normalized_latents):
            all_colors = all_colors[:len(normalized_latents)]
        else:
            # Extend with default colors
            default_color = plt.cm.tab10(0)
            while len(all_colors) < len(normalized_latents):
                all_colors.append(default_color)
    
    if len(all_labels) != len(normalized_latents):
        # Truncate or extend labels list to match
        if len(all_labels) > len(normalized_latents):
            all_labels = all_labels[:len(normalized_latents)]
        else:
            # Extend with default labels
            while len(all_labels) < len(normalized_latents):
                all_labels.append("unknown")
    
    # Convert color tuples to simple color names for numpy compatibility
    simple_colors = []
    for color in all_colors:
        if isinstance(color, tuple):
            # Convert matplotlib color tuple to simple color name
            if len(color) == 4:  # RGBA
                r, g, b, a = color
                if r > 0.8 and g > 0.8 and b > 0.8:
                    simple_colors.append('gray')
                elif r > 0.5 and g > 0.5 and b < 0.5:
                    simple_colors.append('blue')
                elif r > 0.5 and g < 0.5 and b > 0.5:
                    simple_colors.append('red')
                elif r < 0.5 and g > 0.5 and b > 0.5:
                    simple_colors.append('green')
                else:
                    simple_colors.append('black')
            else:
                simple_colors.append('gray')
        else:
            simple_colors.append(str(color))
    
    try:
        all_colors = np.array(simple_colors)
    except Exception as e:
        print(f"WARNING: Error creating colors array: {e}")
        # Fallback: create simple colors array
        all_colors = np.array(['gray'] * len(normalized_latents))
        print(f"DEBUG: Created fallback colors array: {all_colors.shape}")
    
    print(f"[ OK ] Unified dataset: {len(all_latents)} total samples")
    print(f"  - Training: {len([l for l in all_labels if 'training' in l])}")
    print(f"  - Support: {len([l for l in all_labels if 'support' in l])}")
    print(f"  - Query: {len([l for l in all_labels if 'query' in l])}")
    print(f"  - Trajectory: {len([l for l in all_labels if 'trajectory' in l])}")
    
    # Apply unified t-SNE to ALL latent vectors together
    print("Applying unified t-SNE to all latent vectors...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    
    
    try:
        scaler = StandardScaler()
        latents_normalized = scaler.fit_transform(all_latents)
        
        print(f"DEBUG: Normalized latents shape: {latents_normalized.shape}")
        print(f"DEBUG: Normalized latents min/max: {latents_normalized.min():.4f}/{latents_normalized.max():.4f}")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents)-1), max_iter=1000)
        tsne_2d = tsne.fit_transform(latents_normalized)
        
        print(f"DEBUG: t-SNE result shape: {tsne_2d.shape}")
    except Exception as e:
        print(f"DEBUG: Error in t-SNE processing: {e}")
        print(f"DEBUG: all_latents contains NaN: {np.isnan(all_latents).any()}")
        print(f"DEBUG: all_latents contains Inf: {np.isinf(all_latents).any()}")
        return None, None, None, None, None, None, None
    
    # Split back into original data and trajectory data
    training_indices = [i for i, label in enumerate(all_labels) if 'training' in label]
    support_indices = [i for i, label in enumerate(all_labels) if 'support' in label]
    query_indices = [i for i, label in enumerate(all_labels) if 'query' in label]
    trajectory_indices = [i for i, label in enumerate(all_labels) if 'trajectory' in label]
    
    # Return training data and trajectory data separately
    training_latents = all_latents[training_indices] if training_indices else None
    training_tsne_2d = tsne_2d[training_indices] if training_indices else None
    training_labels = [all_labels[i] for i in training_indices] if training_indices else []
    training_colors = [all_colors[i] for i in training_indices] if training_indices else []
    
    trajectory_tsne_2d = tsne_2d[trajectory_indices] if trajectory_indices else None
    
    print(f"[ OK ] Unified t-SNE applied successfully")
    print(f"  - Training samples: {len(training_indices)}")
    print(f"  - Support samples: {len(support_indices)}")
    print(f"  - Query samples: {len(query_indices)}")
    print(f"  - Trajectory points: {len(trajectory_indices)}")
    
    return training_latents, training_tsne_2d, training_labels, training_colors, trajectory_tsne_2d, all_labels, tsne_2d, actual_evaluated_keys
