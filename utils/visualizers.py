"""
Cleaned and optimized visualization utilities for the Latent Space Network.
This file contains essential visualization functions with deprecated and unused code removed.
"""

from utils.data_preparation import (
    transform_grid_to_sequence,
    extract_grid_from_sequence
)

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn.manifold import TSNE
import pickle
from utils.settings_manager import settings
import matplotlib.patches as mpatches

# Unified color palette for latent space visualizations
COLOR_PALETTE = {
    'training_encoder_0': '#FF6B6B',    # Red
    'training_encoder_1': '#4ECDC4',    # Teal
    'training_encoder_2': '#45B7D1',    # Blue
    'training_encoder_3': '#96CEB4',    # Green
    'support_encoder_0': '#FFB6C1',     # Light Red
    'support_encoder_1': '#B4E7E1',     # Light Teal
    'support_encoder_2': '#B3D9FF',     # Light Blue
    'support_encoder_3': '#C8E6C9',     # Light Green
    'query_encoder_0': '#8B0000',       # Dark Red
    'query_encoder_1': '#006666',       # Dark Teal
    'query_encoder_2': '#0066CC',       # Dark Blue
    'query_encoder_3': '#2E8B57',       # Dark Green
    'support_poe': '#FFD700',           # Gold
    'query_poe': '#FF8C00',             # Dark Orange
    'training_encoded': '#DDA0DD',      # Plum
    'training': '#DDA0DD',              # Plum (alias for single encoder)
    'support': '#FFB6C1',               # Light Red
    'query': '#8B0000',                 # Dark Red
}

# Get settings
evaluation_settings = settings.get_evaluation_settings()
DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']

##############################
# CORE UTILITY FUNCTIONS
##############################

def get_epoch_accuracies_for_plotting(results):
    """Extract epoch accuracy data for plotting, handling both single and multi-encoder formats."""
    if 'epoch_accuracies' not in results or not results['epoch_accuracies']:
        print("No epoch accuracy data found in results")
        return []
    
    epoch_accuracies = results['epoch_accuracies']
    processed_accuracies = []
    
    for epoch_data in epoch_accuracies:
        if not isinstance(epoch_data, dict):
            continue
            
        # Check if this is multi-encoder format with detailed structure
        if 'individual_encoders' in epoch_data:
            # Multi-encoder format: aggregate individual encoder accuracies for plotting
            individual_encoders = epoch_data['individual_encoders']
            
            if individual_encoders:
                # Calculate average accuracies across all encoders for the main plot
                shape_accs = [enc_data.get('shape_accuracy', 0.0) for enc_data in individual_encoders.values()]
                grid_accs = [enc_data.get('grid_accuracy', 0.0) for enc_data in individual_encoders.values()]
                overall_accs = [enc_data.get('overall_accuracy', 0.0) for enc_data in individual_encoders.values()]
                exact_accs = [enc_data.get('sample_exact_accuracy', 0.0) for enc_data in individual_encoders.values()]
                
                processed_epoch = {
                    'epoch': epoch_data['epoch'],
                    'shape_accuracy': sum(shape_accs) / len(shape_accs),
                    'grid_accuracy': sum(grid_accs) / len(grid_accs),
                    'overall_accuracy': sum(overall_accs) / len(overall_accs),
                    'sample_exact_accuracy': sum(exact_accs) / len(exact_accs),
                    'evaluation_name': f'Multi-Encoder Avg ({len(individual_encoders)} encoders)',
                    'individual_encoder_data': individual_encoders
                }
                processed_accuracies.append(processed_epoch)
            
        elif 'shape_accuracy' in epoch_data:
            # Single encoder format or already processed format
            processed_epoch = {
                'epoch': epoch_data.get('epoch', len(processed_accuracies) + 1),
                'shape_accuracy': epoch_data.get('shape_accuracy', 0.0),
                'grid_accuracy': epoch_data.get('grid_accuracy', 0.0),
                'overall_accuracy': epoch_data.get('overall_accuracy', 0.0),
                'sample_exact_accuracy': epoch_data.get('sample_exact_accuracy', 0.0),
                'evaluation_name': epoch_data.get('evaluation_name', 'Model')
            }
            processed_accuracies.append(processed_epoch)
    
    print(f"Processed {len(processed_accuracies)} epoch accuracy records for plotting")
    return processed_accuracies

def load_evaluation_latent_data(run_dir, return_all_components=False):
    """Load latent data from evaluation results for visualization."""
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if not os.path.exists(eval_file):
        print(f"No evaluation results found at {eval_file}")
        return None
    
    try:
        with open(eval_file, 'rb') as f:
            eval_results = pickle.load(f)
        
        # Look for latent data in evaluation results
        for key, key_results in eval_results.items():
            if 'evaluation_latent_data' in key_results:
                latent_data = key_results['evaluation_latent_data']
                print(f"✓ Found evaluation latent data for key: {key}")
                
                if return_all_components:
                    return latent_data
                
                # Extract just the latent vectors for simple visualization
                all_latents = []
                for data_type, type_data in latent_data.items():
                    if isinstance(type_data, dict):
                        for encoder_key, encoder_data in type_data.items():
                            if isinstance(encoder_data, dict) and 'latent_zs' in encoder_data:
                                latents = encoder_data['latent_zs']
                                if isinstance(latents, (list, np.ndarray)) and len(latents) > 0:
                                    all_latents.extend(latents)
                
                if all_latents:
                    return np.array(all_latents)
        
        print("No evaluation latent data found in results")
        return None
    except Exception as e:
        print(f"Error loading evaluation latent data: {e}")
        return None

def get_comprehensive_latent_data(run_dir):
    """Get comprehensive latent data from evaluation results avoiding duplicates."""
    try:
        eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
        if not os.path.exists(eval_file):
            return None, None, None, None
            
        with open(eval_file, 'rb') as f:
            eval_results = pickle.load(f)
        
        all_latent_data = []
        all_labels = []
        all_colors = []
        
        def add_data(latents, label, color):
            if latents is not None and len(latents) > 0:
                all_latent_data.append(np.array(latents))
                all_colors.extend([color] * len(latents))
                all_labels.extend([label] * len(latents))
        
        # Handle different evaluation result structures
        key_data = None
        
        # Check if we have the new key_results structure
        if 'key_results' in eval_results and isinstance(eval_results['key_results'], dict):
            key_results = eval_results['key_results']
            print(f"Using key_results structure with keys: {list(key_results.keys())}")
            # Process the first available key
            sample_key = next(iter(key_results.keys()))
            key_data = key_results[sample_key]
        else:
            # Handle legacy structure - find first non-metadata key
            metadata_keys = {'evaluation_metadata', 'key_results', 'aggregated_metrics', 'training_latent_data'}
            problem_keys = {k: v for k, v in eval_results.items() if k not in metadata_keys}
            
            if problem_keys:
                sample_key = next(iter(problem_keys.keys()))
                key_data = problem_keys[sample_key]
            else:
                print("No valid problem keys found in evaluation results")
                return None, None, None, None
        
        print(f"Processing latent data from key: {sample_key}")
        print(f"Available data in key_data: {list(key_data.keys())}")
        
        # Check for training latent data in multiple possible locations
        training_data = None
        if 'training_latent_data' in key_data:
            training_data = key_data['training_latent_data']
            print(f"Found training_latent_data in key_data with keys: {list(training_data.keys()) if training_data else 'None'}")
        elif 'training_latent_data' in eval_results:
            training_data = eval_results['training_latent_data'] 
            print(f"Found training_latent_data in eval_results with keys: {list(training_data.keys()) if training_data else 'None'}")
        else:
            print("No training_latent_data found in key_data or eval_results")
            print("This likely means the evaluation was run without training data collection")
        
        # Add training latent data - split by encoders for multi-encoder models
        if training_data:
            
            # Check if multi-encoder by presence of multiple encoders
            is_multi = len([k for k in training_data.keys() if k.startswith('encoder_')]) > 1
            
            if is_multi:
                # Multi-encoder: show individual encoder training data only
                for key in sorted(training_data.keys()):
                    if key.startswith('encoder_') and 'latent_zs' in training_data[key]:
                        encoder_idx = key.split('_')[1]
                        color_key = f'training_encoder_{encoder_idx}'
                        color = COLOR_PALETTE.get(color_key, COLOR_PALETTE['training_encoded'])
                        add_data(training_data[key]['latent_zs'], f'training_enc_{encoder_idx}', color)
                
                # Note: No training_poe data - PoE is only for inference/evaluation
            else:
                # Single encoder: use encoder_0
                if 'encoder_0' in training_data and 'latent_zs' in training_data['encoder_0']:
                    add_data(training_data['encoder_0']['latent_zs'], 'training', COLOR_PALETTE['training_encoded'])
        
        # Add evaluation latent data (support/query) - only PoE results, not individual encoder samples
        if 'evaluation_latent_data' in key_data:
            eval_data = key_data['evaluation_latent_data']
            
            # Process support and query data - only add PoE latent vectors, not the original samples
            for data_type in ['support', 'query']:
                if data_type in eval_data:
                    type_data = eval_data[data_type]
                    
                    # Check if multi-encoder
                    is_multi = 'poe' in type_data or len([k for k in type_data.keys() if k.startswith('encoder_')]) > 1
                    
                    if is_multi:
                        # Multi-encoder: ONLY use PoE latent vectors (not the input samples)
                        if 'poe' in type_data and 'latent_zs' in type_data['poe']:
                            color_key = f'{data_type}_poe'
                            color = COLOR_PALETTE.get(color_key, COLOR_PALETTE.get(data_type, '#888888'))
                            add_data(type_data['poe']['latent_zs'], f'{data_type}', color)
                    else:
                        # Single encoder: use encoder_0 latent vectors only
                        if 'encoder_0' in type_data and 'latent_zs' in type_data['encoder_0']:
                            color = COLOR_PALETTE.get(data_type, '#888888')
                            add_data(type_data['encoder_0']['latent_zs'], data_type, color)
                        elif 'latent_zs' in type_data:
                            # Fallback: direct access to latent vectors
                            color = COLOR_PALETTE.get(data_type, '#888888')
                            add_data(type_data['latent_zs'], data_type, color)
        
        if not all_latent_data:
            print("No latent data found for visualization")
            return None, None, None, None
        
        # Combine all data
        combined_latents = np.vstack(all_latent_data)
        unique_data_types = len(set(all_labels))
        print(f"✓ Combined {combined_latents.shape[0]} latent vectors from {len(all_latent_data)} sources")
        print(f"✓ Data types: {unique_data_types}")
        print(f"✓ Labels distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        print(f"✓ Note: Showing individual encoder training data + PoE support/query latents")
        
        # Create t-SNE projection
        perplexity = min(30, len(combined_latents) // 4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        latents_2d = tsne.fit_transform(combined_latents)
        
        return combined_latents, latents_2d, all_labels, all_colors
        
    except Exception as e:
        print(f"Error getting comprehensive latent data: {e}")
        return None, None, None, None

def get_comprehensive_latent_data_for_trajectory(run_dir):
    """
    Get comprehensive latent data for trajectory visualization background.
    Simplified version for trajectory visualization.
    
    Args:
        run_dir: Directory containing training and evaluation results
        
    Returns:
        tuple: (combined_latents, tsne_2d, labels, colors) or (None, None, None, None) if no data
    """
    try:
        # Use existing function but return in the expected format
        combined_latents, latents_2d, labels, colors = get_comprehensive_latent_data(run_dir)
        
        if combined_latents is not None:
            return combined_latents, latents_2d, labels, colors
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"⚠ Warning: Could not load comprehensive latent data for trajectory: {e}")
        return None, None, None, None

##############################
# CORE PLOTTING FUNCTIONS
##############################

def plot_epoch_accuracies(results, save_dir=None):
    """Plot epoch accuracies over time."""
    accuracy_data = get_epoch_accuracies_for_plotting(results)
    
    if not accuracy_data:
        print("No accuracy data available for plotting")
        return
    
    epochs = [data['epoch'] for data in accuracy_data]
    shape_accuracies = [data['shape_accuracy'] for data in accuracy_data]
    grid_accuracies = [data['grid_accuracy'] for data in accuracy_data]
    overall_accuracies = [data['overall_accuracy'] for data in accuracy_data]
    exact_accuracies = [data['sample_exact_accuracy'] for data in accuracy_data]
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, shape_accuracies, label='Shape Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, grid_accuracies, label='Grid Accuracy', marker='s', linewidth=2)
    plt.plot(epochs, overall_accuracies, label='Overall Accuracy', marker='^', linewidth=2)
    plt.plot(epochs, exact_accuracies, label='Sample Exact Accuracy', marker='d', linewidth=2)
    
    plt.title('Training Accuracy Over Time', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'epoch_accuracies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Epoch accuracies plot saved")
    else:
        plt.show()

def plot_z_optimization_losses(results, save_dir=None):
    """Plot z optimization losses during training."""
    if 'losses_gradient_ascent' not in results:
        print("No z optimization losses found")
        return
    
    losses_gradient_ascent = results['losses_gradient_ascent']
    
    if not losses_gradient_ascent:
        print("No z optimization data to plot")
        return

    plt.figure(figsize=(12, 6))
    
    for i, losses in enumerate(losses_gradient_ascent):
        if losses:  # Only plot if there are losses to plot
            plt.plot(losses, alpha=0.7, label=f'Sample {i+1}')
    
    plt.title('Z Optimization Losses (Gradient Ascent)', fontsize=16)
    plt.xlabel('Optimization Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if len(losses_gradient_ascent) <= 10:  # Only show legend if not too many lines
        plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'z_optimization_losses.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Z optimization losses plot saved")
    else:
        plt.show()

# Function removed - replaced by enhanced version

def plot_comprehensive_latent_space(results, eval_results=None, save_dir=None):
    """Plot comprehensive latent space visualization using encoded training data."""
    print("Creating comprehensive latent space visualization...")
    
    if not save_dir:
        print("No save directory provided for loading data")
        return
    
    # Get comprehensive latent data using the cleaned-up function
    combined_latents, latents_2d, labels, colors = get_comprehensive_latent_data(save_dir)
    
    if combined_latents is None:
        print("No latent data available for visualization")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define desired legend order: encoders first (0, 1, 2...), then support/query
    unique_labels = list(set(labels))
    
    # Sort labels to show encoders first, then support/query
    def label_sort_key(label):
        if label.startswith('training_enc_'):
            # Extract encoder number for sorting
            try:
                enc_num = int(label.split('_')[-1])
                return (0, enc_num)  # Group 0, sort by encoder number
            except:
                return (0, 999)  # Fallback for malformed encoder labels
        elif label == 'training' or label == 'training_encoded':
            return (1, 0)  # Group 1, general training
        elif label == 'support':
            return (2, 0)  # Group 2, support
        elif label == 'query':
            return (2, 1)  # Group 2, query
        else:
            return (3, 0)  # Group 3, other labels
    
    sorted_labels = sorted(unique_labels, key=label_sort_key)
    legend_elements = []
    
    # Plot in the desired order
    for label in sorted_labels:
        # Get indices for this label
        indices = [i for i, l in enumerate(labels) if l == label]
        x_coords = [latents_2d[i][0] for i in indices]
        y_coords = [latents_2d[i][1] for i in indices]
        color = colors[indices[0]]  # All points with same label have same color
        
        # Clean up label names for display
        display_label = label
        if label.startswith('training_enc_'):
            enc_num = label.split('_')[-1]
            display_label = f'Encoder {enc_num}'
        elif label == 'training' or label == 'training_encoded':
            display_label = 'Training'
        elif label == 'support':
            display_label = 'Support'
        elif label == 'query':
            display_label = 'Query'
        
        plt.scatter(x_coords, y_coords, c=color, s=30, alpha=0.6, 
                   edgecolors='black', linewidth=0.3, label=f'{display_label} ({len(indices)})')
        legend_elements.append(mpatches.Patch(color=color, label=f'{display_label} (n={len(indices)})'))
    
    plt.title('Latent Space Visualization (t-SNE)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'latent_space_visualization.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Clean latent space visualization saved (individual encoders + PoE latents)")
    else:
        plt.show()

##############################
# MAIN VISUALIZATION FUNCTIONS
##############################

def visualize_all_results(results, save_dir=None, eval_results=None, epoch=None):
    """Plot all visualizations for the results using unified multi-encoder processing."""
    
    print("\nPlotting comprehensive latent space...")
    plot_comprehensive_latent_space(results, eval_results=eval_results, save_dir=save_dir)

    print("\nPlotting epoch accuracies over time...")
    plot_epoch_accuracies(results, save_dir)

    # Try to plot multi-encoder training accuracies if available
    if results and save_dir:
        try:
            print("\nPlotting multi-encoder training accuracies...")
            plot_multi_encoder_accuracies(results, save_dir)
        except Exception as e:
            print(f"⚠ Could not create multi-encoder accuracy plots: {e}")

    # Plot trajectory reconstructions if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting trajectory reconstructions...")
            plot_multi_encoder_trajectory_reconstructions(eval_results, save_dir, epoch=epoch)
        except Exception as e:
            print(f"⚠ Could not create trajectory reconstructions: {e}")

    if 'losses_gradient_ascent' in results:
        if len(results['losses_gradient_ascent']) > 0:
            print("\nPlotting z optimization losses...")
            plot_z_optimization_losses(results, save_dir)

    # Plot training reconstruction analysis if training results available
    if results and save_dir:
        try:
            print("\nPlotting training reconstruction analysis...")
            plot_training_reconstruction_analysis(results, save_dir, max_examples=2)
        except Exception as e:
            print(f"⚠ Could not create training reconstruction analysis: {e}")

    # Plot evaluation reconstruction analysis if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting evaluation reconstruction analysis...")
            plot_poe_reconstruction_analysis(eval_results, save_dir, max_examples=2)
        except Exception as e:
            print(f"⚠ Could not create evaluation reconstruction analysis: {e}")

    # Plot encoder influence analysis if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting encoder influence analysis...")
            plot_encoder_influence_analysis(eval_results, save_dir)
        except Exception as e:
            print(f"⚠ Could not create encoder influence analysis: {e}")

def visualize_stored_results(run_dir, epoch=None):
    """Load and visualize results from a previous run with optional epoch specification."""
    print(f"Looking for results in: {run_dir}")
    
    # Initialize wandb for visualize mode
    wandb_logger = None
    try:
        from utils.wandb_logger import init_wandb_for_mode
        wandb_logger = init_wandb_for_mode('visualize', run_dir)
    except Exception as e:
        print(f"⚠ Could not initialize wandb for visualize: {e}")
    
    # Try to load training results
    results_file = os.path.join(run_dir, 'results.pkl')
    results = None
    
    if os.path.exists(results_file):
        print("Found training results file, loading...")
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            print("✓ Training results loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load training results: {e}")
            results = None
    else:
        print("No training results file found (results.pkl)")
    
    # Load model parameters
    params_file = os.path.join(run_dir, 'model_params.pkl')
    model_params = None
    
    if os.path.exists(params_file):
        print("Found model parameters file, loading...")
        try:
            with open(params_file, 'rb') as f:
                model_params = pickle.load(f)
            print("✓ Model parameters loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load model parameters: {e}")
            model_params = None
    else:
        print("No model parameters file found (model_params.pkl)")
    
    # Try to load and visualize evaluation results
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    eval_results = None
    if os.path.exists(eval_file):
        print("\nFound evaluation results file, loading...")
        try:
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
            print("✓ Evaluation results loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load evaluation results: {e}")
    else:
        print("No evaluation results found (evaluation_results.pkl)")
    
    # Visualize training results if available
    if results is not None:
        print("\nVisualizing training results...")
        try:
            visualize_all_results(results, run_dir, eval_results, epoch=epoch)
        except Exception as e:
            print(f"⚠ Warning: Could not visualize training results: {e}")
    else:
        print("Skipping training results visualization (no training results available)")
    
    # Generate model summary as JSON
    if model_params is not None:
        print("\nGenerating comprehensive experiment summary JSON...")
        try:
            summary_data = generate_experiment_summary_json(results, model_params, run_dir, eval_results, epoch=epoch)
            print("✓ Experiment summary JSON generated successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not generate experiment summary JSON: {e}")
    
    # Upload all plots to wandb if initialized
    if wandb_logger and wandb_logger.is_initialized:
        try:
            print(f"\n🔼 UPLOADING PLOTS TO WANDB...")
            uploaded_count = wandb_logger.upload_all_plots(run_dir, epoch)
            if uploaded_count > 0:
                print(f"✓ Successfully uploaded {uploaded_count} plots to wandb")
            else:
                print("⚠ No plots were uploaded to wandb")
        except Exception as e:
            print(f"\n⚠ Could not upload plots to wandb: {e}")
    else:
        print(f"\n⚠ Wandb not available - plots saved locally only")

    # Finish wandb session
    if wandb_logger:
        try:
            wandb_logger.finish()
            print("✓ Wandb session closed")
        except Exception as e:
            print(f"⚠ Error closing wandb session: {e}")

    # Summary of what was found and processed
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Run directory: {run_dir}")
    print(f"Requested epoch: {epoch if epoch else 'latest'}")
    print(f"Training results: {'✓ Found and processed' if results is not None else '✗ Not found'}")
    print(f"Model parameters: {'✓ Found and processed' if model_params is not None else '✗ Not found'}")
    eval_found = os.path.exists(os.path.join(run_dir, 'evaluation_results.pkl'))
    print(f"Evaluation results: {'✓ Found and processed' if eval_found else '✗ Not found'}")
    print(f"Wandb upload: {'✓ Completed' if wandb_logger and wandb_logger.is_initialized else '✗ Not available'}")
    
    if results is None and not eval_found:
        print("\n⚠ Warning: No results files found in the specified directory.")
        print("Make sure the directory contains either 'results.pkl' or 'evaluation_results.pkl'")
    elif results is None:
        print("\n✓ Evaluation-only visualization completed successfully")
    elif not eval_found:
        print("\n✓ Training-only visualization completed successfully")
    else:
        print("\n✓ Complete visualization (training + evaluation) completed successfully")

##############################
# RECOVERED FUNCTIONS FOR SPECIFIC VISUALIZATIONS
##############################

# Function removed - no longer needed

def extract_multi_encoder_accuracies(results):
    """Extract per-encoder accuracy data from multi-encoder training results."""
    if 'epoch_accuracies' not in results or not results['epoch_accuracies']:
        return None
    
    detailed_epochs = []
    for epoch_data in results['epoch_accuracies']:
        if isinstance(epoch_data, dict) and 'individual_encoders' in epoch_data:
            detailed_epochs.append(epoch_data)
    
    if not detailed_epochs:
        return None
    
    encoder_indices = set()
    for epoch_data in detailed_epochs:
        encoder_indices.update(epoch_data['individual_encoders'].keys())
    
    num_encoders = len(encoder_indices)
    encoder_accuracies = {
        'epochs': [ep['epoch'] for ep in detailed_epochs],
        'individual_encoders': {
            enc_idx: {
                'shape_accuracy': [],
                'grid_accuracy': [],
                'overall_accuracy': [],
                'sample_exact_accuracy': []
            } for enc_idx in encoder_indices
        },
        'has_poe_data': False
    }
    
    for epoch_data in detailed_epochs:
        for enc_idx in encoder_indices:
            if enc_idx in epoch_data['individual_encoders']:
                enc_data = epoch_data['individual_encoders'][enc_idx]
                encoder_accuracies['individual_encoders'][enc_idx]['shape_accuracy'].append(enc_data['shape_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['grid_accuracy'].append(enc_data['grid_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['overall_accuracy'].append(enc_data['overall_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['sample_exact_accuracy'].append(enc_data['sample_exact_accuracy'])
            else:
                encoder_accuracies['individual_encoders'][enc_idx]['shape_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['grid_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['overall_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['sample_exact_accuracy'].append(0.0)
    
    return encoder_accuracies

def plot_multi_encoder_accuracies(results, save_dir=None):
    """Plot detailed multi-encoder accuracy curves showing per-encoder performance during training."""
    accuracy_data = extract_multi_encoder_accuracies(results)
    
    if accuracy_data is None:
        print("No detailed multi-encoder accuracy data found, skipping accuracy plots")
        return
    
    epochs = accuracy_data['epochs']
    num_encoders = len(accuracy_data['individual_encoders'])
    has_poe_data = accuracy_data.get('has_poe_data', False)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    metrics = ['shape_accuracy', 'grid_accuracy', 'overall_accuracy', 'sample_exact_accuracy']
    metric_titles = ['Shape Accuracy', 'Grid Accuracy', 'Overall Accuracy', 'Sample Exact Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axs[idx // 2, idx % 2]
        
        for i, (encoder_idx, enc_data) in enumerate(accuracy_data['individual_encoders'].items()):
            ax.plot(epochs, enc_data[metric], marker='o', label=f'Encoder {encoder_idx}', 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        if has_poe_data and 'poe_accuracy' in accuracy_data:
            ax.plot(epochs, accuracy_data['poe_accuracy'][metric], 'k-', linewidth=4, 
                   label='Product of Experts (PoE)', alpha=0.9)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    title_suffix = " (Training - Individual Encoders)" if not has_poe_data else " (Evaluation - Including PoE)"
    plt.suptitle(f'Multi-Encoder Accuracy{title_suffix} ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        filename = 'multi_encoder_training_accuracies.png' if not has_poe_data else 'multi_encoder_accuracies.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder accuracy plot saved to {save_dir}/{filename}")
    else:
        plt.show()

def plot_multi_encoder_trajectory_reconstructions(eval_results, save_dir=None, epoch=None):
    """
    Create trajectory reconstruction plots for multi-encoder models by reusing evaluate_trajectory.py functionality.
    Shows individual encoder outputs (once) and PoE trajectory evolution during optimization (5 steps).
    
    Args:
        eval_results: Dictionary containing evaluation results for each key
        save_dir: Directory to save plots (optional)
    """
    # Import required functions from LPN_reproduction/evaluate_trajectory.py
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from LPN_reproduction.evaluate_trajectory import visualize_multi_encoder_comprehensive_trajectory
        from utils.model_utils import load_model
        # Note: settings is already imported globally at the top of the file
    except ImportError as e:
        print(f"Warning: Could not import trajectory visualization functions: {e}")
        return
    
    if not eval_results:
        print("Warning: No evaluation results provided for trajectory reconstruction")
        return
    
    # Get the visualization limit from settings
    evaluation_settings = settings.get_evaluation_settings()
    visualize_n_values = evaluation_settings.get('visualize_n_values', 3)  # Default to 3 if not found
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Filter for actual problem keys (skip metadata keys)
    metadata_keys = {'evaluation_metadata', 'key_results', 'aggregated_metrics', 'training_latent_data'}
    problem_keys = {k: v for k, v in eval_results.items() if k not in metadata_keys}
    
    # If we have key_results structure, use that instead
    if 'key_results' in eval_results and isinstance(eval_results['key_results'], dict):
        problem_keys = eval_results['key_results']
        print(f"Using key_results structure with keys: {list(problem_keys.keys())}")
    else:
        print(f"Using direct keys: {list(problem_keys.keys())}")
    
    for key, key_results in problem_keys.items():
        print(f"\n=== Processing trajectory reconstructions for key: {key} ===")
    
        # Ensure key_results is a dict with proper structure
        if not isinstance(key_results, dict):
            print(f"⚠ Key '{key}' does not contain valid results structure")
            continue
    
        # Look for trajectory information
        trajectory_info_list = []
        if 'metrics' in key_results and 'trajectory_info' in key_results['metrics']:
            trajectory_info_list = key_results['metrics']['trajectory_info']
        elif 'trajectory_info' in key_results:
            trajectory_info_list = key_results['trajectory_info']
        
        if not trajectory_info_list:
            print(f"⚠ No trajectory information found for key {key}")
            print(f"  Available keys in results: {list(key_results.keys())}")
            if 'metrics' in key_results:
                print(f"  Available keys in metrics: {list(key_results['metrics'].keys())}")
            continue
        
        # Filter for trajectories (unified processing for both single and multi-encoder)
        valid_trajectories = []
        for t in trajectory_info_list:
            is_multi = t.get('is_multi_encoder', False)
            num_encoders = t.get('num_encoders', 1)
            # Include both actual multi-encoder and single-encoder trajectories
            if is_multi or num_encoders >= 1:
                valid_trajectories.append(t)
        
        if not valid_trajectories:
            print(f"⚠ No trajectory information found for key {key}")
            continue
        
        # Determine trajectory type
        sample_trajectory = valid_trajectories[0]
        num_encoders = sample_trajectory.get('num_encoders', 1)
        trajectory_type = f"single encoder (unified)" if num_encoders == 1 else f"multi-encoder ({num_encoders} encoders)"
        print(f"  Found {len(valid_trajectories)} trajectory samples ({trajectory_type})")
        
        # Load model for reconstruction computation - use the same logic as load_model()
        model = None
        if save_dir:
            try:
                # Use provided epoch parameter or extract from evaluation metadata
                epoch_to_load = epoch
                if not epoch_to_load and 'evaluation_metadata' in eval_results:
                    epoch_to_load = eval_results['evaluation_metadata'].get('epoch')
                
                model, _, _, _ = load_model(save_dir, epoch=epoch_to_load, device=device)
                print(f"  ✓ Loaded model from {save_dir}" + (f" (epoch {epoch_to_load})" if epoch_to_load else ""))
            except Exception as e:
                print(f"  ⚠ Could not load model: {e}")
                continue
        else:
            print(f"  ⚠ No save directory provided for model loading")
            continue
                
        # Create visualizations for each sample (limit to visualize_n_values)
        max_samples = min(visualize_n_values, len(valid_trajectories))
        print(f"  Creating visualizations for {max_samples} samples (limited by visualize_n_values={visualize_n_values})...")
        
        for sample_idx, trajectory_info in enumerate(valid_trajectories[:max_samples]):
            try:
                # Create filename
                filename = f'multi_encoder_trajectory_reconstruction_sample_{sample_idx}.png'
                save_path = os.path.join(save_dir, filename) if save_dir else filename
                
                print(f"    Sample {sample_idx + 1}: Creating comprehensive visualization...")
                
                # Use the enhanced evaluate_trajectory function
                visualize_multi_encoder_comprehensive_trajectory(
                    trajectory_info, model, save_path, save_dir, device=device
                )
                
                print(f"    ✓ Saved: {save_path}")
                                
            except Exception as e:
                print(f"    ⚠ Error creating visualization for sample {sample_idx}: {e}")
                continue
        
        print(f"  ✓ Completed trajectory reconstructions for key {key}")
    
    print(f"\n✓ Multi-encoder trajectory reconstruction visualization complete!")

def generate_experiment_summary_json(results, model_params, save_dir=None, eval_results=None, epoch=None):
    """Generate comprehensive experiment summary as JSON with accurate statistics and parameters."""
    import json
    from datetime import datetime
    
    try:
        # Load evaluation results for performance metrics
        eval_results = None
        if save_dir:
            try:
                eval_file = os.path.join(save_dir, 'evaluation_results.pkl')
                if os.path.exists(eval_file):
                    with open(eval_file, 'rb') as f:
                        eval_results = pickle.load(f)
            except Exception as e:
                print(f"⚠ Could not load evaluation results: {e}")
        
        # Load actual model to get precise parameter count
        total_params = None
        trainable_params = None
        try:
            if save_dir:
                from utils.model_utils import load_model
                # Use provided epoch parameter or extract from evaluation metadata
                epoch_to_load = epoch
                if not epoch_to_load and eval_results and 'evaluation_metadata' in eval_results:
                    epoch_to_load = eval_results['evaluation_metadata'].get('epoch')
                
                model, _, _, _ = load_model(save_dir, epoch=epoch_to_load, device='cpu')
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception as e:
            print(f"⚠ Could not load model for parameter counting: {e}")
        
        # Build comprehensive summary
        summary = {
            "experiment_info": {
                "experiment_name": "Latent Program Network",
                "generated_at": datetime.now().isoformat(),
                "run_directory": save_dir if save_dir else "unknown"
            },
            
            "model_architecture": {
                "type": "multi_encoder" if model_params.get('NUM_ENCODERS', 1) > 1 else "single_encoder",
                "num_encoders": model_params.get('NUM_ENCODERS', 1),
                "latent_dimension": model_params.get('LATENT_DIM'),
                "encoder_layers": model_params.get('ENCODER_LAYERS'),
                "decoder_layers": model_params.get('DECODER_LAYERS'),
                "encoder_heads": model_params.get('ENCODER_HEADS'),
                "decoder_heads": model_params.get('DECODER_HEADS'),
                "encoder_hidden_dim": model_params.get('ENCODER_HIDDEN_DIM'),
                "decoder_hidden_dim": model_params.get('DECODER_HIDDEN_DIM'),
                "dropout": model_params.get('DROPOUT'),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "parameter_size_mb": round(total_params * 4 / (1024 * 1024), 2) if total_params else None
            },
            
            "training_configuration": {
                "num_epochs": model_params.get('NUM_EPOCHS'),
                "batch_size": model_params.get('BATCH_SIZE'),
                "learning_rate": model_params.get('LEARNING_RATE'),
                "beta_kl_regularization": model_params.get('BETA'),
                "optimizer": model_params.get('OPTIMIZER', 'AdamW'),
                "scheduler": model_params.get('SCHEDULER'),
                "gradient_accumulation_steps": model_params.get('gradient_accumulation_steps', 1),
                "use_mixed_precision": model_params.get('use_mixed_precision', False),
                "device": model_params.get('device', 'cuda'),
                "data_augmentation": model_params.get('data_augmentation', False)
            },
            
            "performance_results": {},
            "training_history": {},
            "evaluation_settings": {},
            "technical_details": {}
        }
        
        # Performance Results from Evaluation
        if eval_results:
            if 'aggregated_metrics' in eval_results:
                agg = eval_results['aggregated_metrics']
                summary["performance_results"]["evaluation_metrics"] = {
                    "shape_accuracy": {
                        "mean": agg.get('avg_shape_accuracy', 0.0),
                        "std": agg.get('std_shape_accuracy', 0.0)
                    },
                    "grid_accuracy": {
                        "mean": agg.get('avg_grid_accuracy', 0.0),
                        "std": agg.get('std_grid_accuracy', 0.0)
                    },
                    "sample_exact_accuracy": {
                        "mean": agg.get('avg_sample_exact_accuracy', 0.0),
                        "std": agg.get('std_sample_exact_accuracy', 0.0)
                    },
                    "support_loss": {
                        "mean": agg.get('avg_support_loss', 0.0),
                        "std": agg.get('std_support_loss', 0.0)
                    },
                    "query_loss": {
                        "mean": agg.get('avg_query_loss', 0.0),
                        "std": agg.get('std_query_loss', 0.0)
                    }
                }
            
            # Multi-encoder specific analysis
            if 'key_results' in eval_results and model_params.get('NUM_ENCODERS', 1) > 1:
                sample_key = next(iter(eval_results['key_results'].keys()))
                key_data = eval_results['key_results'][sample_key]
                metrics = key_data.get('metrics', {})
                
                # Add multi-encoder statistics if available
                if 'individual_encoder_accuracies' in metrics and 'poe_metrics' in metrics:
                    poe_acc = metrics['poe_metrics'].get('sample_exact_accuracy', 0.0)
                    individual_accs = [acc.get('sample_exact_accuracy', 0.0) for acc in metrics['individual_encoder_accuracies'].values()]
                    
                    if individual_accs:
                        summary["performance_results"]["multi_encoder_analysis"] = {
                            "poe_accuracy": poe_acc,
                            "individual_encoder_accuracies": individual_accs,
                            "best_individual_accuracy": max(individual_accs) if individual_accs else 0.0,
                            "worst_individual_accuracy": min(individual_accs) if individual_accs else 0.0,
                            "avg_individual_accuracy": sum(individual_accs) / len(individual_accs) if individual_accs else 0.0
                        }
            
            # Evaluation settings
            if 'evaluation_metadata' in eval_results:
                meta = eval_results['evaluation_metadata']
                summary["evaluation_settings"] = {
                    "problem_keys": meta.get('keys', []),
                    "num_problem_keys": len(meta.get('keys', [])),
                    "support_samples_per_key": meta.get('n_samples'),
                    "query_samples_per_key": meta.get('n_queries'),
                    "random_seed": meta.get('seed'),
                    "latent_optimization_enabled": meta.get('latent_optimization_enabled', False),
                    "optimization_steps": meta.get('optimization_steps'),
                    "optimization_learning_rate": meta.get('optimization_lr')
                }
        
        # Training History from Results
        if results:
            if 'epoch_accuracies' in results and results['epoch_accuracies']:
                # Extract training progression
                epochs = []
                shape_accs = []
                grid_accs = []
                exact_accs = []
                
                for epoch_data in results['epoch_accuracies']:
                    if isinstance(epoch_data, dict):
                        epochs.append(epoch_data.get('epoch', len(epochs) + 1))
                        
                        # Handle multi-encoder format
                        if 'individual_encoders' in epoch_data:
                            # Average across encoders for main metrics
                            individual = epoch_data['individual_encoders']
                            if individual:
                                shape_avg = sum(enc['shape_accuracy'] for enc in individual.values()) / len(individual)
                                grid_avg = sum(enc['grid_accuracy'] for enc in individual.values()) / len(individual)
                                exact_avg = sum(enc['sample_exact_accuracy'] for enc in individual.values()) / len(individual)
                                
                                shape_accs.append(shape_avg)
                                grid_accs.append(grid_avg)
                                exact_accs.append(exact_avg)
                        else:
                            # Single encoder format
                            shape_accs.append(epoch_data.get('shape_accuracy', 0.0))
                            grid_accs.append(epoch_data.get('grid_accuracy', 0.0))
                            exact_accs.append(epoch_data.get('sample_exact_accuracy', 0.0))
                
                summary["training_history"] = {
                    "epochs": epochs,
                    "shape_accuracy_progression": shape_accs,
                    "grid_accuracy_progression": grid_accs,
                    "sample_exact_accuracy_progression": exact_accs,
                    "final_training_accuracies": {
                        "shape": shape_accs[-1] if shape_accs else None,
                        "grid": grid_accs[-1] if grid_accs else None,
                        "sample_exact": exact_accs[-1] if exact_accs else None
                    }
                }
            
            if 'epoch_losses' in results and results['epoch_losses']:
                summary["training_history"]["loss_progression"] = results['epoch_losses']
                summary["training_history"]["final_loss"] = results['epoch_losses'][-1]
        
        # Technical Details
        summary["technical_details"] = {
            "pytorch_device": model_params.get('device', 'cuda'),
            "mixed_precision_training": model_params.get('use_mixed_precision', False),
            "gradient_accumulation_steps": model_params.get('gradient_accumulation_steps', 1),
            "data_augmentation": model_params.get('data_augmentation', False),
            "model_checkpoint_interval": model_params.get('save_checkpoint_interval', 50),
            "evaluation_interval": model_params.get('evaluation_interval', 25)
        }
        
        # Save to JSON file
        if save_dir:
            json_path = os.path.join(save_dir, 'comprehensive_experiment_summary.json')
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"✓ Comprehensive experiment summary saved to {json_path}")
        
        return summary
        
    except Exception as e:
        print(f"Error generating experiment summary JSON: {e}")
        # Return minimal fallback summary
        fallback_summary = {
            "experiment_info": {
                "experiment_name": "Latent Program Network",
                "generated_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            },
            "model_architecture": {
                "type": "multi_encoder" if model_params.get('NUM_ENCODERS', 1) > 1 else "single_encoder",
                "num_encoders": model_params.get('NUM_ENCODERS', 1),
                "latent_dimension": model_params.get('LATENT_DIM'),
            },
            "error_details": str(e)
        }
        
        if save_dir:
            json_path = os.path.join(save_dir, 'comprehensive_experiment_summary.json')
            with open(json_path, 'w') as f:
                json.dump(fallback_summary, f, indent=2, default=str)
            print(f"⚠ Fallback experiment summary saved to {json_path}")
        
        return fallback_summary 

##############################
# POE RECONSTRUCTION ANALYSIS
##############################

def plot_reconstruction_analysis(data_results, save_dir=None, max_examples=2, data_type="evaluation", dataset_name="Test Dataset"):
    """
    Create comprehensive reconstruction analysis showing:
    1. Bar graph of pixel accuracy distribution (% correct pixels)
    2. Bar graph of grid size accuracy distribution (% correct grid sizes)  
    3. Representative sample reconstructions with clear visual comparison
    
    Args:
        data_results: Results dictionary containing reconstruction data
        save_dir: Directory to save plots
        max_examples: Number of example reconstructions to show
        data_type: Type of data ("training" or "evaluation")
        dataset_name: Display name for the dataset
    """
    if not data_results:
        print(f"No {data_type} results provided")
        return
    
    # Handle different result structures based on data type
    key_results_dict = {}
    
    if data_type == "evaluation":
        if 'key_results' in data_results and isinstance(data_results['key_results'], dict):
            key_results_dict = data_results['key_results']
            print(f"Using key_results structure with keys: {list(key_results_dict.keys())}")
        else:
            metadata_keys = {'evaluation_metadata', 'aggregated_metrics', 'training_latent_data'}
            key_results_dict = {k: v for k, v in data_results.items() if k not in metadata_keys}
            print(f"Using direct structure with keys: {list(key_results_dict.keys())}")
    
    elif data_type == "training":
        # For training data, look for reconstruction results in training results
        if 'reconstruction_results' in data_results:
            # Training results have reconstruction_results directly
            key_results_dict = {'training_key': data_results}
            print(f"Using training reconstruction results")
        else:
            print(f"No reconstruction results found in training data")
            return
    
    if not key_results_dict:
        print(f"No problem keys found in {data_type} results")
        return
    
    # Aggregate data from all available keys for comprehensive analysis
    all_pixel_accuracies = []
    all_grid_size_correct = []
    all_reconstructions = []
    total_samples = 0
    
    for key, key_data in key_results_dict.items():
        # Try multiple possible locations for reconstruction results
        reconstruction_results = None
        if 'reconstruction_results' in key_data:
            reconstruction_results = key_data['reconstruction_results']
        elif 'metrics' in key_data and 'reconstruction_results' in key_data['metrics']:
            reconstruction_results = key_data['metrics']['reconstruction_results']
        elif isinstance(key_data, dict):
            # Look for direct reconstruction data in specialist training results
            potential_keys = ['poe_query_reconstructions', 'query_reconstructions', 'reconstructions']
            for pot_key in potential_keys:
                if pot_key in key_data:
                    reconstruction_results = {pot_key: key_data[pot_key]}
                    break
        
        if not reconstruction_results:
            print(f"No reconstruction results found for key {key}")
            continue
        
        # Get reconstructions based on data type
        reconstructions = None
        reconstruction_type = "Model"
        
        if data_type == "evaluation":
            # For evaluation data: prefer PoE, fallback to general
            if 'poe_query_reconstructions' in reconstruction_results:
                reconstructions = reconstruction_results['poe_query_reconstructions']
                reconstruction_type = "PoE"
            elif 'query_reconstructions' in reconstruction_results:
                reconstructions = reconstruction_results['query_reconstructions']
                reconstruction_type = "Model"
        
        elif data_type == "training":
            # For training data: look for training reconstructions
            if 'training_reconstructions' in reconstruction_results:
                reconstructions = reconstruction_results['training_reconstructions']
                reconstruction_type = "Training Model"
            elif 'reconstructions' in reconstruction_results:
                reconstructions = reconstruction_results['reconstructions']
                reconstruction_type = "Training Model"
        
        if not reconstructions:
            print(f"No {data_type} reconstructions found for key {key}")
            continue
        
        print(f"Processing {len(reconstructions)} {reconstruction_type} reconstructions from key {key}")
        
        # Process reconstructions for this key
        for recon_data in reconstructions:
            try:
                target_seq = np.array(recon_data['target'])
                shape_logits, grid_logits = recon_data['reconstruction']
                
                # Convert to numpy arrays if they aren't already
                shape_logits = np.array(shape_logits)
                grid_logits = np.array(grid_logits)
                
                # Extract target information
                target_grid, target_shape = extract_grid_from_sequence(target_seq)
                target_rows, target_cols = target_shape[0], target_shape[1]
                
                # Get predictions
                shape_pred = np.argmax(shape_logits, axis=-1)
                grid_pred = np.argmax(grid_logits, axis=-1)
                
                # Handle shape prediction format (could be scalar or array)
                if np.isscalar(shape_pred):
                    pred_rows = pred_cols = int(shape_pred)
                elif len(shape_pred) >= 2:
                    pred_rows, pred_cols = int(shape_pred[0]), int(shape_pred[1])
                else:
                    pred_rows = pred_cols = int(shape_pred[0]) if len(shape_pred) > 0 else 0
                
                # Grid size accuracy
                grid_size_match = 1.0 if (target_rows == pred_rows and target_cols == pred_cols) else 0.0
                all_grid_size_correct.append(grid_size_match)
                
                # Pixel accuracy calculation
                pixel_accuracy = 0.0
                if target_rows > 0 and target_cols > 0 and pred_rows > 0 and pred_cols > 0:
                    if target_rows == pred_rows and target_cols == pred_cols:
                        # Same dimensions - direct comparison
                        active_pixels = target_rows * target_cols
                        if active_pixels <= len(grid_pred):
                            pred_grid = grid_pred[:active_pixels].reshape(target_rows, target_cols)
                            if pred_grid.shape == target_grid.shape:
                                correct_pixels = np.sum(target_grid == pred_grid)
                                pixel_accuracy = (correct_pixels / active_pixels) * 100 if active_pixels > 0 else 0
                    # If dimensions don't match, accuracy is 0 (already set above)
                
                all_pixel_accuracies.append(pixel_accuracy)
                
                # Store reconstruction data for visualization
                all_reconstructions.append({
                    'target_seq': target_seq,
                    'target_grid': target_grid,
                    'target_shape': (target_rows, target_cols),
                    'shape_logits': shape_logits,
                    'grid_logits': grid_logits,
                    'pred_shape': (pred_rows, pred_cols),
                    'pixel_accuracy': pixel_accuracy,
                    'grid_size_match': grid_size_match,
                    'key': key
                })
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing reconstruction from key {key}: {e}")
                all_pixel_accuracies.append(0)
                all_grid_size_correct.append(0)
    
    if not all_reconstructions:
        print("No valid reconstructions found for analysis")
        
        # Create a fallback informational plot
        try:
            import matplotlib.pyplot as plt
            import os
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'{dataset_name} Reconstruction Analysis\n\nNo valid reconstructions found.\n\nThis can happen if:\n• Evaluation reconstruction data is missing\n• Reconstruction format is incompatible\n• No evaluation keys have reconstruction results\n\nCheck evaluation logs for details.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
            ax.set_title(f'{dataset_name} Reconstruction Analysis - No Data', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            if save_dir:
                filename = f'{data_type}_reconstruction_analysis.png'
                plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
                print(f"Saved fallback {data_type} reconstruction analysis plot")
            plt.close()
        except Exception as fallback_error:
            print(f"Could not create fallback reconstruction analysis plot: {fallback_error}")
        return
    
    print(f"Analyzed {total_samples} total test samples from {len(key_results_dict)} problem keys")
    
    # Create figure with improved layout: histograms as bar charts at top, sample reconstructions below
    num_examples = min(max_examples, len(all_reconstructions))
    fig = plt.figure(figsize=(16, 8 + 4 * num_examples))
    
    # Use gridspec for better control over layout
    gs = fig.add_gridspec(2 + num_examples, 4, 
                         height_ratios=[2, 2] + [3] * num_examples,
                         width_ratios=[1, 1, 1, 1])
    
    # === TOP ROW: PERFORMANCE ANALYSIS CHARTS ===
    
    # Left: Pixel Accuracy Distribution (as bar chart)
    ax_pixel = fig.add_subplot(gs[0, :2])  # Span first two columns
    
    # Create bins for pixel accuracy
    pixel_bins = [0, 20, 40, 60, 80, 100]
    pixel_bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    pixel_counts = []
    
    for i in range(len(pixel_bins)-1):
        count = sum(1 for acc in all_pixel_accuracies if pixel_bins[i] <= acc < pixel_bins[i+1])
        pixel_counts.append(count)
    
    # Handle edge case for 100% accuracy
    pixel_counts[-1] += sum(1 for acc in all_pixel_accuracies if acc == 100.0)
    
    bars1 = ax_pixel.bar(pixel_bin_labels, pixel_counts, alpha=0.8, color='skyblue', 
                        edgecolor='navy', linewidth=1.5)
    ax_pixel.set_title(f'{dataset_name}: Pixel Accuracy Distribution', fontsize=14, fontweight='bold')
    ax_pixel.set_xlabel('Pixel Accuracy Range')
    ax_pixel.set_ylabel('Number of Samples')
    ax_pixel.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars1, pixel_counts):
        if count > 0:
            ax_pixel.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                         str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add statistics
    mean_pixel_acc = np.mean(all_pixel_accuracies)
    ax_pixel.axhline(y=len(all_pixel_accuracies) * 0.1, color='red', linestyle='--', alpha=0)  # Hidden line for legend
    ax_pixel.text(0.02, 0.98, f'Mean: {mean_pixel_acc:.1f}%\nSamples: {len(all_pixel_accuracies)}', 
                 transform=ax_pixel.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Right: Grid Size Accuracy (as bar chart)
    ax_grid = fig.add_subplot(gs[0, 2:])  # Span last two columns
    
    correct_grid_sizes = sum(all_grid_size_correct)
    incorrect_grid_sizes = len(all_grid_size_correct) - correct_grid_sizes
    
    grid_labels = ['Correct Size', 'Incorrect Size']
    grid_counts = [correct_grid_sizes, incorrect_grid_sizes]
    grid_colors = ['lightgreen', 'lightcoral']
    
    bars2 = ax_grid.bar(grid_labels, grid_counts, alpha=0.8, color=grid_colors, 
                       edgecolor='darkgreen', linewidth=1.5)
    ax_grid.set_title(f'{dataset_name}: Grid Size Accuracy', fontsize=14, fontweight='bold')
    ax_grid.set_ylabel('Number of Samples')
    ax_grid.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and percentages on bars
    for bar, count, label in zip(bars2, grid_counts, grid_labels):
        percentage = (count / len(all_grid_size_correct)) * 100
        ax_grid.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                    fontweight='bold')
    
    # Add overall statistics
    grid_accuracy_pct = (correct_grid_sizes / len(all_grid_size_correct)) * 100
    ax_grid.text(0.02, 0.98, f'Accuracy: {grid_accuracy_pct:.1f}%\nSamples: {len(all_grid_size_correct)}', 
                transform=ax_grid.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # === SECOND ROW: SUMMARY STATISTICS ===
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')
    
    # Calculate comprehensive statistics
    perfect_reconstructions = sum(1 for acc in all_pixel_accuracies if acc == 100.0)
    good_reconstructions = sum(1 for acc in all_pixel_accuracies if acc >= 80.0)
    poor_reconstructions = sum(1 for acc in all_pixel_accuracies if acc < 20.0)
    
    summary_text = f"""
    📊 {dataset_name.upper()} PERFORMANCE SUMMARY ({reconstruction_type} Model)
    
    🎯 Overall Performance:  •  Total Samples: {total_samples}  •  Problem Keys: {len(key_results_dict)}
    
    🔍 Pixel Accuracy:  •  Mean: {mean_pixel_acc:.1f}%  •  Perfect (100%): {perfect_reconstructions} samples  •  Good (≥80%): {good_reconstructions} samples  •  Poor (<20%): {poor_reconstructions} samples
    
    📐 Grid Size Accuracy:  •  Correct Dimensions: {correct_grid_sizes}/{len(all_grid_size_correct)} ({grid_accuracy_pct:.1f}%)  •  Dimension Errors: {incorrect_grid_sizes} samples
    """
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes, 
                   fontsize=11, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # === BOTTOM ROWS: SAMPLE RECONSTRUCTIONS ===
    
    # Select diverse examples for visualization
    examples_to_show = []
    
    # Try to get one perfect example and one imperfect example
    perfect_examples = [r for r in all_reconstructions if r['pixel_accuracy'] == 100.0]
    imperfect_examples = [r for r in all_reconstructions if r['pixel_accuracy'] < 100.0]
    
    if perfect_examples:
        examples_to_show.append(perfect_examples[0])
    if imperfect_examples and len(examples_to_show) < num_examples:
        examples_to_show.append(imperfect_examples[0])
    
    # Fill remaining slots with random samples
    remaining_examples = [r for r in all_reconstructions if r not in examples_to_show]
    while len(examples_to_show) < num_examples and remaining_examples:
        examples_to_show.append(remaining_examples.pop(0))
    
    for i, recon_data in enumerate(examples_to_show):
        row_idx = 2 + i
        
        try:
            target_grid = recon_data['target_grid']
            target_rows, target_cols = recon_data['target_shape']
            shape_logits = recon_data['shape_logits']
            grid_logits = recon_data['grid_logits']
            pred_rows, pred_cols = recon_data['pred_shape']
            pixel_accuracy = recon_data['pixel_accuracy']
            grid_size_match = recon_data['grid_size_match']
            key = recon_data['key']
            
            # Create reconstruction grid
            if pred_rows > 0 and pred_cols > 0 and pred_rows <= 30 and pred_cols <= 30:
                grid_pred = np.argmax(grid_logits, axis=-1)
                active_pixels = pred_rows * pred_cols
                if active_pixels <= len(grid_pred):
                    recon_grid = grid_pred[:active_pixels].reshape(pred_rows, pred_cols)
                else:
                    recon_grid = np.zeros((pred_rows, pred_cols))
            else:
                recon_grid = np.zeros((1, 1))
            
            # Plot ground truth
            ax_gt = fig.add_subplot(gs[row_idx, 0])
            ax_gt.imshow(target_grid, cmap='viridis', interpolation='nearest')
            ax_gt.set_title(f'Ground Truth\n{target_rows}×{target_cols}', fontsize=11, fontweight='bold')
            ax_gt.axis('off')
            
            # Plot reconstruction  
            ax_recon = fig.add_subplot(gs[row_idx, 1])
            ax_recon.imshow(recon_grid, cmap='viridis', interpolation='nearest')
            size_status = "✓" if grid_size_match else "✗"
            ax_recon.set_title(f'{reconstruction_type} Reconstruction\n{pred_rows}×{pred_cols} {size_status}', 
                              fontsize=11, fontweight='bold')
            ax_recon.axis('off')
            
            # Plot difference/error map
            ax_diff = fig.add_subplot(gs[row_idx, 2])
            if target_grid.shape == recon_grid.shape and target_rows > 0 and target_cols > 0:
                diff_map = (target_grid != recon_grid).astype(float)
                ax_diff.imshow(diff_map, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
                correct_pixels = np.sum(target_grid == recon_grid)
                total_pixels = target_rows * target_cols
                error_pixels = total_pixels - correct_pixels
                ax_diff.set_title(f'Error Map\n{error_pixels}/{total_pixels} errors', 
                                 fontsize=11, fontweight='bold')
            else:
                # Different shapes - show size mismatch
                ax_diff.text(0.5, 0.5, 'SIZE\nMISMATCH', ha='center', va='center', 
                            fontsize=12, fontweight='bold', color='red',
                            transform=ax_diff.transAxes)
                ax_diff.set_title('Dimension Error', fontsize=11, fontweight='bold', color='red')
            ax_diff.axis('off')
            
            # Performance summary for this sample
            ax_perf = fig.add_subplot(gs[row_idx, 3])
            ax_perf.axis('off')
            
            perf_color = 'green' if pixel_accuracy >= 80 else 'orange' if pixel_accuracy >= 50 else 'red'
            size_color = 'green' if grid_size_match else 'red'
            
            perf_text = f"""Sample {i+1}
            
Key: {key}

🎯 Pixel Accuracy:
{pixel_accuracy:.1f}%

📐 Grid Size:
{"Correct" if grid_size_match else "Wrong"}

Overall Quality:
{"Excellent" if pixel_accuracy >= 90 else "Good" if pixel_accuracy >= 70 else "Fair" if pixel_accuracy >= 50 else "Poor"}"""
            
            ax_perf.text(0.1, 0.5, perf_text, transform=ax_perf.transAxes, 
                        fontsize=10, va='center', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=perf_color, alpha=0.2))
            
        except Exception as e:
            print(f"Error visualizing reconstruction {i}: {e}")
            # Show error placeholder
            ax_error = fig.add_subplot(gs[row_idx, :])
            ax_error.text(0.5, 0.5, f'Visualization Error\n{str(e)[:50]}...', 
                         ha='center', va='center', transform=ax_error.transAxes,
                         fontsize=12, color='red')
            ax_error.set_title(f'Sample {i+1} - Error', fontsize=11, fontweight='bold', color='red')
            ax_error.axis('off')
    
    plt.suptitle(f'{reconstruction_type} Model: {dataset_name} Reconstruction Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_dir:
        filename = f'{data_type}_reconstruction_analysis.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {dataset_name} reconstruction analysis saved to: {save_path}")
        print(f"  - Analyzed {total_samples} {data_type} samples from {len(key_results_dict)} problem keys")
        print(f"  - Mean pixel accuracy: {mean_pixel_acc:.1f}%")
        print(f"  - Grid size accuracy: {grid_accuracy_pct:.1f}%")
    else:
        plt.show()

def plot_poe_reconstruction_analysis(eval_results, save_dir=None, max_examples=2):
    """
    Backward compatibility wrapper for evaluation reconstruction analysis.
    """
    return plot_reconstruction_analysis(eval_results, save_dir, max_examples, 
                                       data_type="evaluation", dataset_name="Test Dataset")

def plot_training_reconstruction_analysis(training_results, save_dir=None, max_examples=2):
    """
    Create comprehensive training reconstruction analysis by generating reconstructions from training data.
    """
    if not training_results or not save_dir:
        print("No training results or save directory provided")
        return
        
    # Generate reconstruction results from training data and saved model
    try:
        import matplotlib.pyplot as plt
        import os
        from utils.model_utils import load_model
        from utils.data_preparation import extract_grid_from_sequence
        import torch
        import numpy as np
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        model, _, _, _ = load_model(save_dir, epoch=None, device=device)
        model.eval()
        
        # Get training sequences from results
        input_sequences = training_results.get('input_sequences', [])
        output_sequences = training_results.get('output_sequences', [])
        
        if not input_sequences or not output_sequences:
            print("No training sequences found in results")
            return
        
        # Sample a subset for reconstruction analysis (limit to avoid memory issues)
        num_samples = min(50, len(output_sequences))  # Use up to 50 samples
        sample_indices = np.random.choice(len(output_sequences), num_samples, replace=False)
        
        # Generate reconstructions
        training_reconstructions = []
        with torch.no_grad():
            for idx in sample_indices:
                try:
                    target_seq = np.array(output_sequences[idx])
                    input_seq = np.array(input_sequences[idx])
                    
                    # Convert to tensors
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Get model reconstruction
                    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                        # Multi-encoder: use PoE
                        (shape_logits, grid_logits), mu, logvar = model(input_tensor, target_tensor)
                    else:
                        # Single encoder
                        (shape_logits, grid_logits), mu, logvar = model(input_tensor, target_tensor)
                    
                    # Store reconstruction data
                    training_reconstructions.append({
                        'target': target_seq.tolist(),
                        'reconstruction': (shape_logits[0].cpu().numpy(), grid_logits[0].cpu().numpy())
                    })
                except Exception as e:
                    print(f"Error generating reconstruction for sample {idx}: {e}")
                    continue
        
        if not training_reconstructions:
            print("No training reconstructions could be generated")
            # Create a fallback simple analysis message
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.text(0.5, 0.5, 'Training Reconstruction Analysis\n\nNo reconstructions could be generated.\nThis can happen if:\n• Model loading failed\n• No training sequences found\n• Reconstruction generation errors\n\nCheck logs for specific error details.', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
                ax.set_title('Training Reconstruction Analysis - No Data', fontsize=14)
                ax.axis('off')
                plt.tight_layout()
                if save_dir:
                    plt.savefig(os.path.join(save_dir, 'training_reconstruction_analysis.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("Saved fallback training reconstruction analysis plot")
            except Exception as fallback_error:
                print(f"Could not create fallback plot: {fallback_error}")
            return
        
        # Create a mock training results structure with reconstruction results
        mock_training_results = {
            'reconstruction_results': {
                'training_reconstructions': training_reconstructions
            }
        }
        
        print(f"Generated {len(training_reconstructions)} training reconstructions for analysis")
        
        # Call the generic reconstruction analysis function
        return plot_reconstruction_analysis(mock_training_results, save_dir, max_examples, 
                                           data_type="training", dataset_name="Training Dataset")
        
    except Exception as e:
        print(f"Error creating training reconstruction analysis: {e}")
        return 

##############################
# ENCODER INFLUENCE ANALYSIS
##############################

def plot_encoder_influence_analysis(eval_results, save_dir=None):
    """
    Plot encoder influence analysis showing distribution of mean-influence indices 
    for each encoder across evaluation samples, grouped by evaluation keys.
    
    Args:
        eval_results: Dictionary containing evaluation results for each key
        save_dir: Directory to save plots (optional)
    """
    if not eval_results:
        print("No evaluation results provided for influence analysis")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Handle different result structures
    key_results_dict = {}
    if 'key_results' in eval_results and isinstance(eval_results['key_results'], dict):
        key_results_dict = eval_results['key_results']
        print(f"Using key_results structure with keys: {list(key_results_dict.keys())}")
    else:
        metadata_keys = {'evaluation_metadata', 'aggregated_metrics', 'training_latent_data'}
        key_results_dict = {k: v for k, v in eval_results.items() if k not in metadata_keys}
        print(f"Using direct structure with keys: {list(key_results_dict.keys())}")
    
    if not key_results_dict:
        print("No problem keys found in evaluation results")
        return
    
    # Check if we have influence metrics in any key - try multiple locations
    influence_data_found = False
    num_encoders = 0
    
    for key, key_data in key_results_dict.items():
        influence_metrics = None
        
        # Try multiple possible locations for influence metrics
        if 'metrics' in key_data and 'encoder_influence_metrics' in key_data['metrics']:
            influence_metrics = key_data['metrics']['encoder_influence_metrics']
        elif 'encoder_influence_metrics' in key_data:
            influence_metrics = key_data['encoder_influence_metrics']
        elif 'metrics' in key_data and 'influence_metrics' in key_data['metrics']:
            influence_metrics = key_data['metrics']['influence_metrics']
        
        if influence_metrics and len(influence_metrics) > 0:
            influence_data_found = True
            # Determine number of encoders from first sample
            sample_influences = influence_metrics[0]
            if isinstance(sample_influences, dict):
                num_encoders = len([k for k in sample_influences.keys() if k.startswith('encoder_')])
                if num_encoders > 0:
                    break
    
    if not influence_data_found:
        print("No encoder influence metrics found in evaluation results")
        print("Influence metrics are only available for multi-encoder models with PoE evaluation")
        print("This is expected for single-encoder models or when PoE evaluation is not enabled")
        
        # Create a fallback informational plot
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'Encoder Influence Analysis\n\nNo influence metrics found.\n\nThis is expected for:\n• Single-encoder models\n• When PoE evaluation is not enabled\n• When encoder influence calculation fails\n\nInfluence metrics are only available for\nmulti-encoder models with PoE evaluation.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax.set_title('Encoder Influence Analysis - No Data', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'encoder_influence_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved fallback encoder influence analysis plot")
        except Exception as fallback_error:
            print(f"Could not create fallback influence plot: {fallback_error}")
        return
    
    print(f"Found encoder influence metrics for {num_encoders} encoders")
    
    # Collect influence data for each key
    keys_with_data = []
    for key, key_data in key_results_dict.items():
        influence_metrics = None
        
        # Try multiple possible locations for influence metrics (same as above)
        if 'metrics' in key_data and 'encoder_influence_metrics' in key_data['metrics']:
            influence_metrics = key_data['metrics']['encoder_influence_metrics']
        elif 'encoder_influence_metrics' in key_data:
            influence_metrics = key_data['encoder_influence_metrics']
        elif 'metrics' in key_data and 'influence_metrics' in key_data['metrics']:
            influence_metrics = key_data['metrics']['influence_metrics']
        
        if influence_metrics and len(influence_metrics) > 0:
            keys_with_data.append((key, influence_metrics))
    
    if not keys_with_data:
        print("No keys with valid influence metrics found")
        return
    
    print(f"Creating influence analysis plots for {len(keys_with_data)} evaluation keys")
    
    # Create figure with subplots for each key
    num_keys = len(keys_with_data)
    fig, axes = plt.subplots(num_keys, 1, figsize=(12, 4 * num_keys))
    
    # Handle single key case
    if num_keys == 1:
        axes = [axes]
    
    # Color palette for encoders
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    for plot_idx, (key, influence_metrics) in enumerate(keys_with_data):
        ax = axes[plot_idx]
        
        # Organize influence data by encoder
        encoder_influences = {f'encoder_{i}': [] for i in range(num_encoders)}
        
        for sample_influences in influence_metrics:
            for enc_name, influence_value in sample_influences.items():
                if enc_name in encoder_influences:
                    encoder_influences[enc_name].append(influence_value)
        
        # Create histograms for each encoder
        bin_edges = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        alpha = 0.7
        
        for enc_idx in range(num_encoders):
            enc_name = f'encoder_{enc_idx}'
            influences = encoder_influences[enc_name]
            
            if influences:
                ax.hist(influences, bins=bin_edges, alpha=alpha, 
                       color=colors[enc_idx], label=f'Encoder {enc_idx}', 
                       edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_title(f'Encoder Influence Distribution - Key: {key}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mean Influence Index', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Add statistics text
        stats_text = []
        total_samples = len(influence_metrics)
        stats_text.append(f'Total Samples: {total_samples}')
        
        # Calculate mean influence for each encoder
        for enc_idx in range(num_encoders):
            enc_name = f'encoder_{enc_idx}'
            influences = encoder_influences[enc_name]
            if influences:
                mean_influence = np.mean(influences)
                std_influence = np.std(influences)
                stats_text.append(f'Enc {enc_idx}: μ={mean_influence:.3f}, σ={std_influence:.3f}')
        
        # Add stats box
        stats_str = '\n'.join(stats_text)
        ax.text(0.98, 0.98, stats_str, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
               fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        filename = 'encoder_influence_analysis.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Encoder influence analysis saved to: {save_path}")
        print(f"  - Analyzed {num_encoders} encoders across {len(keys_with_data)} evaluation keys")
        
        # Print summary statistics
        print(f"\n📊 ENCODER INFLUENCE SUMMARY:")
        for key, influence_metrics in keys_with_data:
            print(f"\nKey: {key}")
            encoder_influences = {f'encoder_{i}': [] for i in range(num_encoders)}
            
            for sample_influences in influence_metrics:
                for enc_name, influence_value in sample_influences.items():
                    if enc_name in encoder_influences:
                        encoder_influences[enc_name].append(influence_value)
            
            for enc_idx in range(num_encoders):
                enc_name = f'encoder_{enc_idx}'
                influences = encoder_influences[enc_name]
                if influences:
                    mean_inf = np.mean(influences)
                    std_inf = np.std(influences)
                    min_inf = np.min(influences)
                    max_inf = np.max(influences)
                    print(f"  Encoder {enc_idx}: μ={mean_inf:.4f}, σ={std_inf:.4f}, range=[{min_inf:.4f}, {max_inf:.4f}]")
            
            # Calculate encoder dominance statistics
            dominant_encoder_counts = {f'encoder_{i}': 0 for i in range(num_encoders)}
            for sample_influences in influence_metrics:
                # Find most influential encoder for this sample
                max_influence = max(sample_influences.values())
                for enc_name, influence_value in sample_influences.items():
                    if influence_value == max_influence:
                        dominant_encoder_counts[enc_name] += 1
                        break  # In case of ties, count the first one
            
            print(f"  Dominance (most influential per sample):")
            for enc_idx in range(num_encoders):
                enc_name = f'encoder_{enc_idx}'
                count = dominant_encoder_counts[enc_name]
                percentage = (count / len(influence_metrics)) * 100
                print(f"    Encoder {enc_idx}: {count}/{len(influence_metrics)} samples ({percentage:.1f}%)")
    else:
        plt.show()