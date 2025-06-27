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

    print("\nPlotting PoE reconstruction analysis...")
    plot_poe_reconstruction_analysis(eval_results, save_dir, max_examples=2)

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
        from utils.settings_manager import settings
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

def plot_poe_reconstruction_analysis(eval_results, save_dir=None, max_examples=2):
    """
    Create PoE reconstruction analysis showing:
    1. Histogram of active pixels accuracy (% correct pixels)
    2. Histogram of grid size accuracy (% correct grid sizes)
    3. Reconstructions: query vs ground truth (for 2 queries)
    """
    if not eval_results:
        print("No evaluation results provided")
        return
    
    # Handle different evaluation result structures
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
    
    # Get first key's reconstruction data
    first_key = next(iter(key_results_dict.keys()))
    key_data = key_results_dict[first_key]
    
    if 'reconstruction_results' not in key_data:
        print(f"No reconstruction results found for key {first_key}")
        return
    
    reconstruction_results = key_data['reconstruction_results']
    
    # Get PoE query reconstructions
    query_reconstructions = None
    if 'poe_query_reconstructions' in reconstruction_results:
        query_reconstructions = reconstruction_results['poe_query_reconstructions']
        reconstruction_type = "PoE"
    elif 'query_reconstructions' in reconstruction_results:
        query_reconstructions = reconstruction_results['query_reconstructions']
        reconstruction_type = "Model"
    else:
        print("No query reconstructions found")
        return
    
    print(f"Creating PoE reconstruction analysis for {len(query_reconstructions)} {reconstruction_type} reconstructions...")
    
    # Calculate accuracy metrics for histograms
    pixel_accuracies = []
    grid_size_correct = []
    for recon_data in query_reconstructions:
        try:
            target_seq = recon_data['target']
            shape_logits, grid_logits = recon_data['reconstruction']
            target_grid, target_shape = extract_grid_from_sequence(target_seq)
            target_rows, target_cols = target_shape
            shape_pred = np.argmax(shape_logits, axis=-1)
            grid_pred = np.argmax(grid_logits, axis=-1)
            pred_rows, pred_cols = shape_pred[0], shape_pred[1]
            grid_size_correct.append(1.0 if (target_rows, target_cols) == (pred_rows, pred_cols) else 0.0)
            if target_shape == (pred_rows, pred_cols) and target_rows > 0 and target_cols > 0:
                if pred_rows > 0 and pred_cols > 0 and pred_rows <= 30 and pred_cols <= 30:
                    recon_grid = grid_pred[:pred_rows * pred_cols].reshape(pred_rows, pred_cols)
                    correct_pixels = np.sum(target_grid == recon_grid) if target_grid.shape == recon_grid.shape else 0
                    total_pixels = target_rows * target_cols
                    accuracy = (correct_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                else:
                    accuracy = 0
            else:
                accuracy = 0
            pixel_accuracies.append(accuracy)
        except Exception as e:
            print(f"Error processing reconstruction for histogram: {e}")
            pixel_accuracies.append(0)
            grid_size_correct.append(0)
    
    # Create figure with subplots: 2 histograms side by side on top, reconstructions below
    num_examples = min(max_examples, len(query_reconstructions))
    fig = plt.figure(figsize=(16, 6 + 3 * num_examples))
    gs = fig.add_gridspec(2 + num_examples, 2, height_ratios=[1] + [1] + [2]*num_examples, width_ratios=[1, 1])
    
    # Top row: Pixel accuracy histogram (left)
    ax_hist1 = fig.add_subplot(gs[0, 0])
    ax_hist1.hist(pixel_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax_hist1.set_title('Active Pixels Accuracy Distribution', fontsize=12, fontweight='bold')
    ax_hist1.set_xlabel('Accuracy (%)')
    ax_hist1.set_ylabel('Count')
    ax_hist1.grid(True, alpha=0.3)
    ax_hist1.axvline(np.mean(pixel_accuracies), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(pixel_accuracies):.1f}%')
    ax_hist1.legend()
    
    # Top row: Grid size accuracy histogram (right)
    ax_hist2 = fig.add_subplot(gs[0, 1])
    grid_size_percentages = [acc * 100 for acc in grid_size_correct]
    ax_hist2.hist(grid_size_percentages, bins=[0, 50, 100], alpha=0.7, color='lightgreen', edgecolor='black')
    ax_hist2.set_title('Grid Size Accuracy Distribution', fontsize=12, fontweight='bold')
    ax_hist2.set_xlabel('Accuracy (%)')
    ax_hist2.set_ylabel('Count')
    ax_hist2.set_xticks([0, 50, 100])
    ax_hist2.grid(True, alpha=0.3)
    correct_count = sum(grid_size_correct)
    total_count = len(grid_size_correct)
    ax_hist2.axvline((correct_count/total_count)*100, color='red', linestyle='--',
                     label=f'Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)')
    ax_hist2.legend()
    
    # Below: Reconstructions for num_examples queries
    for i in range(num_examples):
        recon_data = query_reconstructions[i]
        try:
            target_seq = recon_data['target']
            shape_logits, grid_logits = recon_data['reconstruction']
            target_grid, target_shape = extract_grid_from_sequence(target_seq)
            target_rows, target_cols = target_shape
            shape_pred = np.argmax(shape_logits, axis=-1)
            grid_pred = np.argmax(grid_logits, axis=-1)
            pred_rows, pred_cols = shape_pred[0], shape_pred[1]
            if target_shape == (pred_rows, pred_cols) and target_rows > 0 and target_cols > 0:
                if pred_rows > 0 and pred_cols > 0 and pred_rows <= 30 and pred_cols <= 30:
                    recon_grid = grid_pred[:pred_rows * pred_cols].reshape(pred_rows, pred_cols)
                    correct_pixels = np.sum(target_grid == recon_grid) if target_grid.shape == recon_grid.shape else 0
                    total_pixels = target_rows * target_cols
                    accuracy = (correct_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                else:
                    recon_grid = np.zeros((1, 1))
                    accuracy = 0
            else:
                recon_grid = np.zeros((1, 1))
                accuracy = 0
            # Plot ground truth
            ax_gt = fig.add_subplot(gs[1 + i, 0])
            ax_gt.imshow(target_grid, cmap='viridis', interpolation='nearest')
            ax_gt.set_title(f'Ground Truth\n{target_rows}×{target_cols}', fontsize=10)
            ax_gt.axis('off')
            # Plot reconstruction
            ax_recon = fig.add_subplot(gs[1 + i, 1])
            ax_recon.imshow(recon_grid, cmap='viridis', interpolation='nearest')
            ax_recon.set_title(f'{reconstruction_type} Recon\n{pred_rows}×{pred_cols}\nAcc: {accuracy:.1f}%', fontsize=10)
            ax_recon.axis('off')
        except Exception as e:
            print(f"Error processing reconstruction {i}: {e}")
            ax_error = fig.add_subplot(gs[1 + i, :])
            ax_error.text(0.5, 0.5, f'Error\n{str(e)[:20]}...', ha='center', va='center', transform=ax_error.transAxes)
            ax_error.set_title(f'Example {i+1} - Error', fontsize=10)
            ax_error.axis('off')
    
    plt.suptitle(f'PoE Reconstruction Analysis - Key: {first_key}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, 'poe_reconstruction_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ PoE reconstruction analysis saved to: {save_path}")
    else:
        plt.show() 