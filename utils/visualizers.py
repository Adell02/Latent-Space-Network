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
import tempfile
import matplotlib.cm as cm
import hashlib

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
                print(f"[ OK ] Found evaluation latent data for key: {key}")
                
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
        key_results_dict = {}
        
        # Check if we have the new key_results structure
        if 'key_results' in eval_results and isinstance(eval_results['key_results'], dict):
            key_results_dict = eval_results['key_results']
            print(f"Using key_results structure with keys: {list(key_results_dict.keys())}")
        else:
            # Handle legacy structure - find all non-metadata keys
            metadata_keys = {'evaluation_metadata', 'key_results', 'aggregated_metrics', 'training_latent_data'}
            key_results_dict = {k: v for k, v in eval_results.items() if k not in metadata_keys}
            print(f"Using direct structure with keys: {list(key_results_dict.keys())}")
        
        if not key_results_dict:
            print("No valid problem keys found in evaluation results")
            return None, None, None, None
        
        print(f"Processing latent data from ALL {len(key_results_dict)} keys: {list(key_results_dict.keys())}")
        
        # Check for training latent data (should be consistent across keys)
        training_data = None
        sample_key = next(iter(key_results_dict.keys()))
        sample_key_data = key_results_dict[sample_key]
        
        if 'training_latent_data' in sample_key_data:
            training_data = sample_key_data['training_latent_data']
            print(f"Found training_latent_data in key_data with keys: {list(training_data.keys()) if training_data else 'None'}")
        elif 'training_latent_data' in eval_results:
            training_data = eval_results['training_latent_data'] 
            print(f"Found training_latent_data in eval_results with keys: {list(training_data.keys()) if training_data else 'None'}")
        else:
            print("No training_latent_data found in key_data or eval_results")
            print("This likely means the evaluation was run without training data collection")
        
        # Add training latent data - split by encoders for multi-encoder models
        # (Training data is the same across all keys, so we only add it once)
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
        
        # Add evaluation latent data from ALL keys (support/query) - only PoE results
        total_support_samples = 0
        total_query_samples = 0
        
        for eval_key, key_data in key_results_dict.items():
            print(f"Processing evaluation data from key: {eval_key}")
            
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
                                # Add key suffix to label for clarity
                                label = f'{data_type}_{eval_key[:8]}'  # Truncate key for readability
                                add_data(type_data['poe']['latent_zs'], label, color)
                                
                                if data_type == 'support':
                                    total_support_samples += len(type_data['poe']['latent_zs'])
                                else:
                                    total_query_samples += len(type_data['poe']['latent_zs'])
                        else:
                            # Single encoder: use encoder_0 latent vectors only
                            if 'encoder_0' in type_data and 'latent_zs' in type_data['encoder_0']:
                                color = COLOR_PALETTE.get(data_type, '#888888')
                                label = f'{data_type}_{eval_key[:8]}'  # Truncate key for readability
                                add_data(type_data['encoder_0']['latent_zs'], label, color)
                                
                                if data_type == 'support':
                                    total_support_samples += len(type_data['encoder_0']['latent_zs'])
                                else:
                                    total_query_samples += len(type_data['encoder_0']['latent_zs'])
                            elif 'latent_zs' in type_data:
                                # Fallback: direct access to latent vectors
                                color = COLOR_PALETTE.get(data_type, '#888888')
                                label = f'{data_type}_{eval_key[:8]}'  # Truncate key for readability
                                add_data(type_data['latent_zs'], label, color)
                                
                                if data_type == 'support':
                                    total_support_samples += len(type_data['latent_zs'])
                                else:
                                    total_query_samples += len(type_data['latent_zs'])
        
        if not all_latent_data:
            print("No latent data found for visualization")
            return None, None, None, None
        
        # Combine all data
        combined_latents = np.vstack(all_latent_data)
        unique_data_types = len(set(all_labels))
        print(f"[ OK ] Combined {combined_latents.shape[0]} latent vectors from {len(all_latent_data)} sources")
        print(f"[ OK ] Processed ALL {len(key_results_dict)} evaluation keys")
        print(f"[ OK ] Support samples: {total_support_samples}, Query samples: {total_query_samples}")
        print(f"[ OK ] Data types: {unique_data_types}")
        print(f"[ OK ] Labels distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        print(f"[ OK ] Note: Showing individual encoder training data + PoE support/query latents from ALL keys")
        
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
        print(f"[ WARNING ] Warning: Could not load comprehensive latent data for trajectory: {e}")
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
    # --- NEW: Use the new evaluation latent space plot style ---
    if eval_results is not None:
        plot_evaluation_latent_space_by_key_and_encoder(eval_results, save_dir)
        return
    # --- fallback to old code for training data ---
    combined_latents, latents_2d, labels, colors = get_comprehensive_latent_data(save_dir)
    if combined_latents is None:
        print("No latent data available for visualization")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Define desired legend order: encoders first (0, 1, 2...), then support/query grouped by type
    unique_labels = list(set(labels))
    
    # Sort labels to show encoders first, then support/query grouped
    def label_sort_key(label):
        if label.startswith('training_enc_'):
            # Extract encoder number for sorting
            try:
                enc_num = int(label.split('_')[2])  # training_enc_X
                return (0, enc_num)  # Group 0, sort by encoder number
            except:
                return (0, 999)  # Fallback for malformed encoder labels
        elif label == 'training' or label == 'training_encoded':
            return (1, 0)  # Group 1, general training
        elif label.startswith('support_'):
            return (2, 0, label)  # Group 2, support (sort by key name)
        elif label.startswith('query_'):
            return (2, 1, label)  # Group 2, query (sort by key name)
        else:
            return (3, 0, label)  # Group 3, other labels
    
    sorted_labels = sorted(unique_labels, key=label_sort_key)
    legend_elements = []
    
    # Group support/query data by type for cleaner legend
    support_labels = [l for l in sorted_labels if l.startswith('support_')]
    query_labels = [l for l in sorted_labels if l.startswith('query_')]
    other_labels = [l for l in sorted_labels if not l.startswith(('support_', 'query_'))]
    
    # Plot in the desired order
    for label in other_labels + support_labels + query_labels:
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
        elif label.startswith('support_'):
            key_suffix = label.split('_', 1)[1] if '_' in label else 'unknown'
            display_label = f'Support ({key_suffix})'
        elif label.startswith('query_'):
            key_suffix = label.split('_', 1)[1] if '_' in label else 'unknown'
            display_label = f'Query ({key_suffix})'
        
        plt.scatter(x_coords, y_coords, c=color, s=30, alpha=0.6, 
                   edgecolors='black', linewidth=0.3, label=f'{display_label} ({len(indices)})')
        legend_elements.append(mpatches.Patch(color=color, label=f'{display_label} (n={len(indices)})'))
    
    # Add summary counts for support/query if we have multiple keys
    total_support = sum(len([i for i, l in enumerate(labels) if l == label]) for label in support_labels)
    total_query = sum(len([i for i, l in enumerate(labels) if l == label]) for label in query_labels)
    
    title = 'Latent Space Visualization (t-SNE)'
    if total_support > 0 and total_query > 0:
        num_support_keys = len(support_labels)
        num_query_keys = len(query_labels)
        title += f'\nSupport: {total_support} samples from {num_support_keys} keys, Query: {total_query} samples from {num_query_keys} keys'
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'latent_space_visualization.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("[ OK ] Comprehensive latent space visualization saved (ALL keys aggregated)")
        print(f"  - Training encoders: {len([l for l in unique_labels if l.startswith('training_enc_')])} encoders")
        print(f"  - Support samples: {total_support} from {len(support_labels)} keys")
        print(f"  - Query samples: {total_query} from {len(query_labels)} keys")
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
            print(f"[ WARNING ] Could not create multi-encoder accuracy plots: {e}")

    # Plot trajectory reconstructions if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting trajectory reconstructions...")
            plot_multi_encoder_trajectory_reconstructions(eval_results, save_dir, epoch=epoch)
        except Exception as e:
            print(f"[ WARNING ] Could not create trajectory reconstructions: {e}")

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
            print(f"[ WARNING ] Could not create training reconstruction analysis: {e}")

    # Plot evaluation reconstruction analysis if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting evaluation reconstruction analysis...")
            plot_poe_reconstruction_analysis(eval_results, save_dir, max_examples=2)
        except Exception as e:
            print(f"[ WARNING ] Could not create evaluation reconstruction analysis: {e}")

    # Plot encoder influence analysis if evaluation results available
    if eval_results and save_dir:
        try:
            print("\nPlotting encoder influence analysis...")
            plot_encoder_influence_analysis(eval_results, save_dir)
        except Exception as e:
            print(f"[ WARNING ] Could not create encoder influence analysis: {e}")

def visualize_stored_results(run_dir, epoch=None):
    """Load and visualize results from a previous run with optional epoch specification."""
    print(f"Looking for results in: {run_dir}")
    
    # Initialize wandb for visualize mode
    wandb_logger = None
    try:
        from utils.wandb_logger import init_wandb_for_mode
        wandb_logger = init_wandb_for_mode('visualize', run_dir)
    except Exception as e:
        print(f"[ WARNING ] Could not initialize wandb for visualize: {e}")
    
    # Try to load training results
    results_file = os.path.join(run_dir, 'results.pkl')
    results = None
    
    if os.path.exists(results_file):
        print("Found training results file, loading...")
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            print("[ OK ] Training results loaded successfully")
        except Exception as e:
            print(f"[ WARNING ] Warning: Could not load training results: {e}")
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
            print("[ OK ] Model parameters loaded successfully")
        except Exception as e:
            print(f"[ WARNING ] Warning: Could not load model parameters: {e}")
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
            print("[ OK ] Evaluation results loaded successfully")
        except Exception as e:
            print(f"[ WARNING ] Warning: Could not load evaluation results: {e}")
    else:
        print("No evaluation results found (evaluation_results.pkl)")
    
    # Visualize training results if available
    if results is not None:
        print("\nVisualizing training results...")
        try:
            visualize_all_results(results, run_dir, eval_results, epoch=epoch)
        except Exception as e:
            print(f"[ WARNING ] Warning: Could not visualize training results: {e}")
    else:
        print("Skipping training results visualization (no training results available)")
    
    # Generate model summary as JSON
    if model_params is not None:
        print("\nGenerating comprehensive experiment summary JSON...")
        try:
            summary_data = generate_experiment_summary_json(results, model_params, run_dir, eval_results, epoch=epoch)
            print("[ OK ] Experiment summary JSON generated successfully")
        except Exception as e:
            print(f"[ WARNING ] Warning: Could not generate experiment summary JSON: {e}")
    
    # Upload all plots to wandb if initialized
    if wandb_logger and wandb_logger.is_initialized:
        try:
            print(f"\n🔼 UPLOADING PLOTS TO WANDB...")
            uploaded_count = wandb_logger.upload_all_plots(run_dir, epoch)
            if uploaded_count > 0:
                print(f"[ OK ] Successfully uploaded {uploaded_count} plots to wandb")
            else:
                print("[ WARNING ] No plots were uploaded to wandb")
        except Exception as e:
            print(f"\n[ WARNING ] Could not upload plots to wandb: {e}")
    else:
        print(f"\n[ WARNING ] Wandb not available - plots saved locally only")

    # Finish wandb session
    if wandb_logger:
        try:
            wandb_logger.finish()
            print("[ OK ] Wandb session closed")
        except Exception as e:
            print(f"[ WARNING ] Error closing wandb session: {e}")

    # Summary of what was found and processed
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Run directory: {run_dir}")
    print(f"Requested epoch: {epoch if epoch else 'latest'}")
    print(f"Training results: {'[ OK ] Found and processed' if results is not None else '✗ Not found'}")
    print(f"Model parameters: {'[ OK ] Found and processed' if model_params is not None else '✗ Not found'}")
    eval_found = os.path.exists(os.path.join(run_dir, 'evaluation_results.pkl'))
    print(f"Evaluation results: {'[ OK ] Found and processed' if eval_found else '✗ Not found'}")
    print(f"Wandb upload: {'[ OK ] Completed' if wandb_logger and wandb_logger.is_initialized else '✗ Not available'}")
    
    if results is None and not eval_found:
        print("\n[ WARNING ] Warning: No results files found in the specified directory.")
        print("Make sure the directory contains either 'results.pkl' or 'evaluation_results.pkl'")
    elif results is None:
        print("\n[ OK ] Evaluation-only visualization completed successfully")
    elif not eval_found:
        print("\n[ OK ] Training-only visualization completed successfully")
    else:
        print("\n[ OK ] Complete visualization (training + evaluation) completed successfully")

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
            print(f"[ WARNING ] Key '{key}' does not contain valid results structure")
            continue
    
        # Look for trajectory information
        trajectory_info_list = []
        if 'metrics' in key_results and 'trajectory_info' in key_results['metrics']:
            trajectory_info_list = key_results['metrics']['trajectory_info']
        elif 'trajectory_info' in key_results:
            trajectory_info_list = key_results['trajectory_info']
        
        if not trajectory_info_list:
            print(f"[ WARNING ] No trajectory information found for key {key}")
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
            print(f"[ WARNING ] No trajectory information found for key {key}")
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
                print(f"  [ OK ] Loaded model from {save_dir}" + (f" (epoch {epoch_to_load})" if epoch_to_load else ""))
            except Exception as e:
                print(f"  [ WARNING ] Could not load model: {e}")
                continue
        else:
            print(f"  [ WARNING ] No save directory provided for model loading")
            continue
                
        # Create visualizations for each sample (limit to visualize_n_values)
        max_samples = min(visualize_n_values, len(valid_trajectories))
        print(f"  Creating visualizations for {max_samples} samples (limited by visualize_n_values={visualize_n_values})...")
        
        for sample_idx, trajectory_info in enumerate(valid_trajectories[:max_samples]):
            try:
                # Create trajectory plots folder
                trajectory_plots_dir = os.path.join(save_dir, "trajectory_plots") if save_dir else "trajectory_plots"
                os.makedirs(trajectory_plots_dir, exist_ok=True)
                
                # Create filename
                filename = f'multi_encoder_trajectory_reconstruction_sample_{sample_idx}.png'
                save_path = os.path.join(trajectory_plots_dir, filename)
                
                print(f"    Sample {sample_idx + 1}: Creating comprehensive visualization...")
                
                # Use the enhanced evaluate_trajectory function
                visualize_multi_encoder_comprehensive_trajectory(
                    trajectory_info, model, save_path, save_dir, device=device
                )
                
                print(f"    [ OK ] Saved: {save_path}")
                                
            except Exception as e:
                print(f"    [ WARNING ] Error creating visualization for sample {sample_idx}: {e}")
                continue
        
        print(f"  [ OK ] Completed trajectory reconstructions for key {key}")
    
    print(f"\n[ OK ] Multi-encoder trajectory reconstruction visualization complete!")

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
                print(f"[ WARNING ] Could not load evaluation results: {e}")
        
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
            print(f"[ WARNING ] Could not load model for parameter counting: {e}")
        
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
            print(f"[ OK ] Comprehensive experiment summary saved to {json_path}")
        
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
            print(f"[ WARNING ] Fallback experiment summary saved to {json_path}")
        
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
        
        if reconstructions is None or len(reconstructions) == 0:
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
            size_status = "[ OK ]" if grid_size_match else "✗"
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
        print(f"[ OK ] {dataset_name} reconstruction analysis saved to: {save_path}")
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
                        (shape_logits, grid_logits), mu, logvar, _ = model(input_tensor, target_tensor)
                    else:
                        # Single encoder
                        (shape_logits, grid_logits), mu, logvar, _ = model(input_tensor, target_tensor)
                    
                    # Store reconstruction data
                    training_reconstructions.append({
                        'target': target_seq.tolist(),
                        'reconstruction': (shape_logits[0].detach().cpu().numpy(), grid_logits[0].detach().cpu().numpy())
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
        
        # Try multiple possible locations for covariance traces
        if 'metrics' in key_data and 'encoder_covariance_traces' in key_data['metrics']:
            influence_metrics = key_data['metrics']['encoder_covariance_traces']
        elif 'encoder_covariance_traces' in key_data:
            influence_metrics = key_data['encoder_covariance_traces']
        elif 'metrics' in key_data and 'encoder_influence_metrics' in key_data['metrics']:
            # Backward compatibility
            influence_metrics = key_data['metrics']['encoder_influence_metrics']
        elif 'encoder_influence_metrics' in key_data:
            # Backward compatibility
            influence_metrics = key_data['encoder_influence_metrics']
        
        if influence_metrics and len(influence_metrics) > 0:
            influence_data_found = True
            # Determine number of encoders from first sample
            sample_influences = influence_metrics[0]
            if isinstance(sample_influences, dict):
                num_encoders = len([k for k in sample_influences.keys() if k.startswith('encoder_')])
                if num_encoders > 0:
                    break
    
    if not influence_data_found:
        print("No encoder covariance traces found in evaluation results")
        print("Covariance traces are only available for multi-encoder models with PoE evaluation")
        print("This is expected for single-encoder models or when PoE evaluation is not enabled")
        
        # Create a fallback informational plot
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'Encoder Covariance Analysis\n\nNo covariance traces found.\n\nThis is expected for:\n• Single-encoder models\n• When PoE evaluation is not enabled\n• When encoder covariance calculation fails\n\nCovariance traces are only available for\nmulti-encoder models with PoE evaluation.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax.set_title('Encoder Covariance Analysis - No Data', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'encoder_covariance_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved fallback encoder covariance analysis plot")
        except Exception as fallback_error:
            print(f"Could not create fallback influence plot: {fallback_error}")
        return
    
    print(f"Found encoder covariance traces for {num_encoders} encoders")
    
    # Collect influence data for each key
    keys_with_data = []
    for key, key_data in key_results_dict.items():
        influence_metrics = None
        
        # Try multiple possible locations for covariance traces (same as above)
        if 'metrics' in key_data and 'encoder_covariance_traces' in key_data['metrics']:
            influence_metrics = key_data['metrics']['encoder_covariance_traces']
        elif 'encoder_covariance_traces' in key_data:
            influence_metrics = key_data['encoder_covariance_traces']
        elif 'metrics' in key_data and 'encoder_influence_metrics' in key_data['metrics']:
            # Backward compatibility
            influence_metrics = key_data['metrics']['encoder_influence_metrics']
        elif 'encoder_influence_metrics' in key_data:
            # Backward compatibility
            influence_metrics = key_data['encoder_influence_metrics']
        
        if influence_metrics and len(influence_metrics) > 0:
            keys_with_data.append((key, influence_metrics))
    
    if not keys_with_data:
        print("No keys with valid covariance traces found")
        return
    
    print(f"Creating covariance analysis plots for {len(keys_with_data)} evaluation keys")
    
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
        ax.set_title(f'Encoder Covariance Trace Distribution - Key: {key}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Covariance Trace (Σσᵢ²)', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Add statistics text
        stats_text = []
        total_samples = len(influence_metrics)
        stats_text.append(f'Total Samples: {total_samples}')
        
        # Calculate mean covariance trace for each encoder
        for enc_idx in range(num_encoders):
            enc_name = f'encoder_{enc_idx}'
            influences = encoder_influences[enc_name]
            if influences:
                mean_trace = np.mean(influences)
                std_trace = np.std(influences)
                stats_text.append(f'Enc {enc_idx}: mu={mean_trace:.2f}, sigma={std_trace:.2f}')
        
        # Add stats box
        stats_str = '\n'.join(stats_text)
        ax.text(0.98, 0.98, stats_str, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
               fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        filename = 'encoder_covariance_analysis.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[ OK ] Encoder covariance analysis saved to: {save_path}")
        print(f"  - Analyzed {num_encoders} encoders across {len(keys_with_data)} evaluation keys")
        
        # Print summary statistics
        print(f"\n📊 ENCODER COVARIANCE SUMMARY:")
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
                    print(f"  Encoder {enc_idx}: mu={mean_inf:.4f}, sigma={std_inf:.4f}, range=[{min_inf:.4f}, {max_inf:.4f}]")
            
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

##############################
# PER-DIMENSION KL DIVERGENCE ANALYSIS
##############################

def generate_per_dimension_kl_plot(model, dataloader, device, epoch, encoder_idx=None, wandb_logger=None, global_step=None):
    """
    Generate per-dimension KL divergence bar plot showing KL divergence for each latent dimension.
    
    KL divergence per dimension: 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)
    where sigma^2 = exp(log_var), so log sigma^2 = log_var
    
    Args:
        model: Current model state
        dataloader: Data loader with samples
        device: Device to run on
        epoch: Current epoch
        encoder_idx: Encoder index (None for PoE)
        wandb_logger: WandB logger instance
        global_step: Global training step
    """    
    model.eval()
    
    # Collect latent statistics from multiple batches
    all_mus = []
    all_log_vars = []
    
    with torch.no_grad():
        # Collect latent statistics from batches (limit to avoid memory issues)
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= 10:  # Limit to 10 batches for efficiency
                break
                
            # Handle different dataloader formats
            if len(batch_data) == 2:
                input_seq, target_seq = batch_data
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
            elif len(batch_data) == 3:
                # Mixed domains dataloader format
                input_seq, target_seq, _ = batch_data
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
            else:
                continue
            
            # Get latent distributions
            if encoder_idx is not None:
                # Specific encoder
                if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
                    mu, log_var,_ = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                else:
                    # Single encoder model
                    mu, log_var,_ = model.encoder(input_seq, target_seq)
            else:
                # PoE or single encoder
                if hasattr(model, 'multi_encoder') and model.multi_encoder and hasattr(model.multi_encoder, 'encoders'):
                    # Multi-encoder: use PoE
                    _, mu, log_var = model(input_seq, target_seq)
                else:
                    # Single encoder
                    mu, log_var,_ = model.encoder(input_seq, target_seq)
            
            all_mus.append(mu.detach().cpu().numpy())
            all_log_vars.append(log_var.detach().cpu().numpy())
    
    if not all_mus:
        print("No latent data collected for KL divergence plot")
        return
    
    # Concatenate all samples: shape (total_samples, latent_dim)
    all_mus = np.concatenate(all_mus, axis=0)
    all_log_vars = np.concatenate(all_log_vars, axis=0)
    
    # Calculate per-dimension KL divergence (SAME as apply_free_bits_per_dimension)
    # KL = 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)
    # where sigma^2 = exp(log_var), so log sigma^2 = log_var
    mu_squared = all_mus ** 2
    sigma_squared = np.exp(all_log_vars)
    log_sigma_squared = all_log_vars
    
    # Per-dimension KL for each sample [samples, latent_dim]
    kl_per_sample_per_dim = 0.5 * (mu_squared + sigma_squared - 1 - log_sigma_squared)
    
    # Average over samples to get per-dimension KL [latent_dim]
    kl_per_dim_raw = np.mean(kl_per_sample_per_dim, axis=0)
    
    # Also show what free-bits clamping would do (δ = 0.07 typically)
    delta_per_dim = 0.07
    kl_per_dim_clamped = np.maximum(kl_per_dim_raw, delta_per_dim)
    
    # Use raw values for the main plot, but show both in statistics
    kl_per_dim = kl_per_dim_raw
    
    latent_dim = len(kl_per_dim)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(max(8, latent_dim * 0.3), 6))
    
    # Create bars
    dimensions = np.arange(latent_dim)
    bars = ax.bar(dimensions, kl_per_dim, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('KL Divergence per Dimension', fontsize=12)
    
    # Set title based on encoder
    if encoder_idx is not None:
        title = f'Per-Dimension KL Divergence - Encoder {encoder_idx}\nEpoch {epoch}'
    else:
        title = f'Per-Dimension KL Divergence - PoE\nEpoch {epoch}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars (for reasonable number of dimensions)
    if latent_dim <= 20:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add statistics (RAW vs CLAMPED to detect collapse)
    mean_kl_raw = np.mean(kl_per_dim_raw)
    mean_kl_clamped = np.mean(kl_per_dim_clamped)
    std_kl = np.std(kl_per_dim_raw)
    total_kl_raw = np.sum(kl_per_dim_raw)
    total_kl_clamped = np.sum(kl_per_dim_clamped)
    max_kl = np.max(kl_per_dim_raw)
    min_kl = np.min(kl_per_dim_raw)
    dims_collapsed = np.sum(kl_per_dim_raw < delta_per_dim)
    
    stats_text = f'RAW Statistics:\nMean: {mean_kl_raw:.3f}\nStd: {std_kl:.3f}\nTotal: {total_kl_raw:.3f}\nMax: {max_kl:.3f}\nMin: {min_kl:.3f}\n\nFree-bits (δ={delta_per_dim}):\nClamped Total: {total_kl_clamped:.3f}\nCollapsed Dims: {dims_collapsed}/{latent_dim}\nSamples: {all_mus.shape[0]}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
           fontsize=10)
    
    # Add horizontal lines for monitoring
    ax.axhline(y=mean_kl_raw, color='blue', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean Raw: {mean_kl_raw:.3f}')
    ax.axhline(y=delta_per_dim, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Free-bits δ: {delta_per_dim:.3f}')
    ax.legend(loc='upper left')
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, latent_dim - 0.5)
    ax.set_ylim(0, max(kl_per_dim) * 1.1)
    
    # Set x-axis ticks
    if latent_dim <= 30:
        ax.set_xticks(dimensions)
        ax.set_xticklabels([f'D{i}' for i in dimensions], rotation=45 if latent_dim > 15 else 0)
    else:
        # For high-dimensional spaces, use fewer ticks
        step = max(1, latent_dim // 10)
        tick_positions = dimensions[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'D{i}' for i in tick_positions])
    
    plt.tight_layout()
    
    # Save and log to wandb
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
        temp_plot_path = tmp_file.name
    plt.close()
    
    if wandb_logger and global_step is not None:
        try:
            import wandb
            
            # Create wandb key based on context
            if encoder_idx is not None:
                wandb_key = f'kl_analysis/per_dimension_kl_encoder_{encoder_idx}'
                metrics_key = f'kl_analysis/encoder_{encoder_idx}'
            else:
                wandb_key = f'kl_analysis/per_dimension_kl_poe'
                metrics_key = f'kl_analysis/poe'
            
            # Log plot and metrics (RAW + collapse detection)
            log_dict = {
                wandb_key: wandb.Image(temp_plot_path),
                f'{metrics_key}_mean_kl_per_dim_raw': mean_kl_raw,
                f'{metrics_key}_mean_kl_per_dim_clamped': mean_kl_clamped,
                f'{metrics_key}_std_kl_per_dim': std_kl,
                f'{metrics_key}_total_kl_raw': total_kl_raw,
                f'{metrics_key}_total_kl_clamped': total_kl_clamped,
                f'{metrics_key}_max_kl_per_dim': max_kl,
                f'{metrics_key}_min_kl_per_dim': min_kl,
                f'{metrics_key}_collapsed_dimensions': dims_collapsed,
                f'{metrics_key}_collapse_fraction': dims_collapsed / latent_dim,
                f'{metrics_key}_free_bits_delta': delta_per_dim,
                f'{metrics_key}_latent_dimension': latent_dim,
                f'{metrics_key}_samples_analyzed': all_mus.shape[0]
            }
            
            wandb_logger._safe_log(log_dict, step_hint=global_step)
            
            print(f"[ OK ] Per-dimension KL plot logged to WandB at step {global_step}")
            print(f"  - Encoder: {'PoE' if encoder_idx is None else encoder_idx}")
            print(f"  - Latent dimension: {latent_dim}")
            print(f"  - Mean KL per dimension (RAW): {mean_kl_raw:.3f}")
            print(f"  - Collapsed dimensions: {dims_collapsed}/{latent_dim} ({dims_collapsed/latent_dim:.1%})")
            print(f"  - Samples analyzed: {all_mus.shape[0]}")
            
        except Exception as e:
            print(f"[ WARNING ] Failed to log per-dimension KL plot to WandB: {e}")
        finally:
            os.unlink(temp_plot_path)
    else:
        print(f"[ OK ] Per-dimension KL plot generated for epoch {epoch}")
        print(f"  - Encoder: {'PoE' if encoder_idx is None else encoder_idx}")
        print(f"  - Latent dimension: {latent_dim}")
        print(f"  - Mean KL per dimension (RAW): {mean_kl_raw:.3f}")
        print(f"  - Collapsed dimensions: {dims_collapsed}/{latent_dim} ({dims_collapsed/latent_dim:.1%})")
        print(f"  - Samples analyzed: {all_mus.shape[0]}")
        os.unlink(temp_plot_path)
    
    model.train()

# --- Global persistent table for latent space slider ---
wandb_latent_table = None

def plot_latent_space_by_key_and_encoder(latent_tuples, title, save_path=None, key_colors=None, epoch=None, phase=None, infinite_dataloader=False, logvars=None, wandb_logger=None, slider_table=None, upload_slider=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    if not latent_tuples or len(latent_tuples) == 0:
        print("No latent data to plot.")
        return

    latents = np.array([x[0] for x in latent_tuples])
    keys = [x[1] for x in latent_tuples]
    encoders = [x[2] for x in latent_tuples]
    unique_keys = sorted(list(set(keys)))
    if key_colors is None:
        key_colors = {k: cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
    colors = [key_colors[k] for k in keys]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    tsne_coords = tsne.fit_transform(latents)

    # Certainty as -logvar.mean(1)
    if logvars is not None:
        certainties = -np.mean(logvars, axis=1)
        min_c, max_c = np.min(certainties), np.max(certainties)
        sizes = 40 + 160 * (certainties - min_c) / (max_c - min_c + 1e-8)
    else:
        certainties = [1]*len(tsne_coords)
        sizes = 60

    plt.figure(figsize=(12, 10))
    for i, (coord, key, enc) in enumerate(zip(tsne_coords, keys, encoders)):
        plt.scatter(coord[0], coord[1], color=key_colors[key], s=sizes[i] if isinstance(sizes, np.ndarray) else sizes, alpha=0.7, edgecolors='k', linewidths=0.3)
        label = f"{str(key)[:4]}"
        if enc is not None:
            label += f"/E{enc}"
        plt.text(coord[0], coord[1], label, fontsize=7, color=key_colors[key], alpha=0.85)
    for k in unique_keys:
        plt.scatter([], [], color=key_colors[k], label=f"{str(k)[:4]}")
    plt.legend(title="Key (first 4 chars)", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    full_title = title
    if phase:
        full_title = f"{phase.title()} Latent Space: " + full_title
    if epoch is not None:
        full_title += f" (Epoch {epoch+1})"
    if infinite_dataloader:
        full_title += " [Infinite Dataloader]"
    plt.title(full_title, fontsize=15)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved latent space plot: {save_path}")
    else:
        plt.show()

    # Note: WandB logging is now handled in the calling function (plot_training_latent_space_per_epoch)
    # using the correct approach of logging the same image key across epochs

def extract_latent_key_encoder_tuples_from_eval(eval_results, use_training_data=False, return_logvar=False):
    tuples = []
    logvars_list = [] if return_logvar else None
    if not eval_results:
        return (tuples, None) if return_logvar else tuples
    # Handle both new and legacy structures
    key_results_dict = {}
    if 'key_results' in eval_results and isinstance(eval_results['key_results'], dict):
        key_results_dict = eval_results['key_results']
    else:
        metadata_keys = {'evaluation_metadata', 'key_results', 'aggregated_metrics', 'training_latent_data'}
        key_results_dict = {k: v for k, v in eval_results.items() if k not in metadata_keys}
    if use_training_data:
        training_data = None
        sample_key = next(iter(key_results_dict.keys()), None)
        if sample_key:
            sample_key_data = key_results_dict[sample_key]
            if 'training_latent_data' in sample_key_data:
                training_data = sample_key_data['training_latent_data']
            elif 'training_latent_data' in eval_results:
                training_data = eval_results['training_latent_data']
        if training_data:
            for enc_key, enc_data in training_data.items():
                encoder_idx = None
                if enc_key.startswith('encoder_'):
                    encoder_idx = int(enc_key.split('_')[1])
                latents = enc_data.get('latent_zs', [])
                logvars = enc_data.get('latent_log_vars', []) if return_logvar else None
                keys = enc_data.get('keys', [None]*len(latents))
                for i, (latent, key) in enumerate(zip(latents, keys)):
                    tuples.append((latent, key, encoder_idx))
                    if return_logvar and logvars is not None:
                        logvars_list.append(logvars[i])
        return (tuples, np.array(logvars_list)) if return_logvar else tuples
    # Otherwise, use evaluation_latent_data
    for key, key_data in key_results_dict.items():
        if 'evaluation_latent_data' in key_data:
            eval_data = key_data['evaluation_latent_data']
            for data_type in ['support', 'query']:
                if data_type in eval_data:
                    type_data = eval_data[data_type]
                    # Multi-encoder: use PoE and/or individual encoders
                    if 'poe' in type_data and 'latent_zs' in type_data['poe']:
                        latents = type_data['poe']['latent_zs']
                        logvars = type_data['poe'].get('latent_log_vars', None) if return_logvar else None
                        for i, latent in enumerate(latents):
                            tuples.append((latent, key, 'poe'))
                            if return_logvar and logvars is not None:
                                logvars_list.append(logvars[i])
                    for enc_key, enc_data in type_data.items():
                        if enc_key.startswith('encoder_') and 'latent_zs' in enc_data:
                            encoder_idx = int(enc_key.split('_')[1])
                            latents = enc_data['latent_zs']
                            logvars = enc_data.get('latent_log_vars', None) if return_logvar else None
                            for i, latent in enumerate(latents):
                                tuples.append((latent, key, encoder_idx))
                                if return_logvar and logvars is not None:
                                    logvars_list.append(logvars[i])
    return (tuples, np.array(logvars_list)) if return_logvar else tuples

def extract_latent_key_encoder_tuples_from_dataloader(model, dataloader, device, key_list=None, encoder_idx=None, max_batches=10, return_logvar=False):
    model.eval()
    tuples = []
    logvars_list = [] if return_logvar else None
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= max_batches:
                break
            
            # ← MODIFIED: Handle keys properly
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                input_seq, target_seq, batch_keys = batch[:3]
                batch_keys = list(batch_keys)  # Ensure it's a list
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_seq, target_seq = batch[:2]
                batch_keys = None
            else:
                continue
                
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            # Get latent
            if hasattr(model, 'multi_encoder') and model.multi_encoder and encoder_idx is not None:
                mu, logvar, _ = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                z = model.reparameterize(mu, logvar)
                enc_idx = encoder_idx
            elif hasattr(model, 'encoder'):
                mu, logvar, _ = model.encoder(input_seq, target_seq)
                z = model.reparameterize(mu, logvar)
                enc_idx = encoder_idx if encoder_idx is not None else 0
            else:
                continue
            z_np = z.detach().cpu().numpy()
            if return_logvar:
                logvars_list.append(logvar.detach().cpu().numpy())
            if batch_keys is None:
                if key_list is not None:
                    batch_keys = key_list[batch_count*input_seq.size(0):(batch_count+1)*input_seq.size(0)]
                else:
                    # For infinite dataloader without keys, use generic labels
                    batch_keys = [f"batch_{batch_count}_sample_{i}" for i in range(z_np.shape[0])]
            else:
                batch_keys = list(batch_keys)
            for latent, key in zip(z_np, batch_keys):
                tuples.append((latent, key, enc_idx))
            batch_count += 1
    model.train()
    if return_logvar:
        logvars_all = np.concatenate(logvars_list, axis=0) if logvars_list else None
        return tuples, logvars_all
    return tuples

def plot_training_latent_space_per_epoch(model, dataloader, device, epoch, save_dir, key_list=None, encoder_idx=None, infinite_dataloader=False, max_batches=10, wandb_logger=None, input_sequences=None, output_sequences=None, training_keys=None, slider_table=None, upload_slider=False, use_task_optimization=True):
    """
    Plot training latent space using task-level optimization for proper task clustering.
    Creates ONE point per task instead of multiple points per realization.
    """
    if use_task_optimization and dataloader is not None:
        # Use task-level optimization for proper task clustering
        latent_data, task_keys = extract_latent_data_from_dataloader(
            dataloader, model, max_batches=max_batches, use_optimization=True
        )
        
        # Convert to expected format with real task keys
        all_tuples = []
        for encoder_key, encoder_data in latent_data.items():
            encoder_idx_val = int(encoder_key.split('_')[1]) if '_' in encoder_key else 0
            for i, latent_z in enumerate(encoder_data['latent_zs']):
                # Now task_keys has one key per latent (not per sample)
                task_key = task_keys[i] if task_keys and i < len(task_keys) else f"unknown_task_{i}"
                all_tuples.append((latent_z, task_key, encoder_idx_val))
        
        title = f"Training Latent Space (Task-Level Optimized) - Epoch {epoch+1}"
        save_path = os.path.join(save_dir, "latent_space_plots", f"training_task_level_epoch_{epoch+1}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plot_latent_space_by_key_and_encoder(all_tuples, title=title, save_path=save_path, epoch=epoch, phase="training_task_level", infinite_dataloader=False, wandb_logger=wandb_logger, slider_table=slider_table, upload_slider=upload_slider)
        return
    
    # Original per-sample implementation"""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    
    # Create latent space plots folder
    latent_plots_dir = os.path.join(save_dir, "latent_space_plots")
    os.makedirs(latent_plots_dir, exist_ok=True)
    
    # Collect all latent data efficiently
    all_tuples = []
    all_logvars = []
    
    if input_sequences is not None and output_sequences is not None and training_keys is not None:
        # Use pre-computed sequences
        model.eval()
        batch_size = 128
        input_arr = np.array(input_sequences)
        output_arr = np.array(output_sequences)
        
        if key_list is not None and len(key_list) == len(input_arr):
            sample_keys = key_list
        else:
            n_per_key = len(input_arr) // len(training_keys)
            sample_keys = []
            for k in training_keys:
                sample_keys.extend([k] * n_per_key)
            sample_keys.extend([training_keys[-1]] * (len(input_arr) - len(sample_keys)))
        
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                for i in range(0, len(input_arr), batch_size):
                    batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    mu, logvar, _ = model.multi_encoder.encoders[enc_idx](batch_inputs, batch_outputs)
                    z = model.reparameterize(mu, logvar)
                    z_np = z.detach().cpu().numpy()
                    logvar_np = logvar.detach().cpu().numpy()
                    batch_keys = sample_keys[i:i+batch_size]
                    for latent, key, logvar_sample in zip(z_np, batch_keys, logvar_np):
                        all_tuples.append((latent, key, enc_idx))
                        all_logvars.append(logvar_sample)
        else:
            for i in range(0, len(input_arr), batch_size):
                batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                mu, logvar, _ = model.encoder(batch_inputs, batch_outputs)
                z = model.reparameterize(mu, logvar)
                z_np = z.detach().cpu().numpy()
                logvar_np = logvar.detach().cpu().numpy()
                batch_keys = sample_keys[i:i+batch_size]
                for latent, key, logvar_sample in zip(z_np, batch_keys, logvar_np):
                    all_tuples.append((latent, key, 0))
                    all_logvars.append(logvar_sample)
        model.train()
    else:
        # Fallback to dataloader-based approach
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                t, l = extract_latent_key_encoder_tuples_from_dataloader(
                    model, dataloader, device, key_list=key_list, encoder_idx=enc_idx, max_batches=max_batches, return_logvar=True
                )
                all_tuples.extend(t)
                if l is not None:
                    all_logvars.extend(l)
        else:
            tuples, logvars = extract_latent_key_encoder_tuples_from_dataloader(
                model, dataloader, device, key_list=key_list, encoder_idx=encoder_idx, max_batches=max_batches, return_logvar=True
            )
            all_tuples.extend(tuples)
            if logvars is not None:
                all_logvars.extend(logvars)
    
    if not all_tuples:
        print("No training latent data to plot for epoch.")
        return
    
    # Convert to numpy arrays for efficiency
    latents = np.array([x[0] for x in all_tuples])
    keys = [x[1] for x in all_tuples]
    encoders = [x[2] for x in all_tuples]
    logvars_array = np.array(all_logvars) if all_logvars else None
    
    # Compute t-SNE once for all data
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    tsne_coords = tsne.fit_transform(latents)
    
    # Compute certainties
    if logvars_array is not None:
        certainties = -np.mean(logvars_array, axis=1)
        min_c, max_c = np.min(certainties), np.max(certainties)
        sizes = 40 + 160 * (certainties - min_c) / (max_c - min_c + 1e-8)
    else:
        sizes = np.full(len(tsne_coords), 60)
    
    # Get unique keys and encoders for color mapping
    unique_keys = sorted(list(set(keys)))
    unique_encoders = sorted(list(set(encoders)))
    
    # Create color maps
    key_colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
    encoder_colors = {e: plt.cm.tab10(i % 10) for i, e in enumerate(unique_encoders)}
    
    # 1. Create separate plots for each encoder
    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
        for enc_idx in unique_encoders:
            # Filter data for this encoder
            enc_mask = [e == enc_idx for e in encoders]
            enc_coords = tsne_coords[enc_mask]
            enc_keys = [k for i, k in enumerate(keys) if enc_mask[i]]
            enc_sizes = sizes[enc_mask] if isinstance(sizes, np.ndarray) else [sizes[i] for i in range(len(sizes)) if enc_mask[i]]
            
            plt.figure(figsize=(10, 8))
            for i, (coord, key) in enumerate(zip(enc_coords, enc_keys)):
                plt.scatter(coord[0], coord[1], color=key_colors[key], s=enc_sizes[i], alpha=0.7, edgecolors='k', linewidths=0.3)
            
            # Add legend
            for k in unique_keys:
                plt.scatter([], [], color=key_colors[k], label=f"{str(k)[:8]}")
            
            plt.legend(title="Task Keys", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
            plt.title(f"Training Latent Space - Encoder {enc_idx} (Epoch {epoch+1})", fontsize=14)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            
            save_path = os.path.join(latent_plots_dir, f"latent_space_training_encoder_{enc_idx}_epoch_{epoch+1}.png")
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved encoder {enc_idx} latent space plot: {save_path}")
    
    # 2. Create overview plots: color by encoder and color by task key
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by encoder
    for i, (coord, enc) in enumerate(zip(tsne_coords, encoders)):
        ax1.scatter(coord[0], coord[1], color=encoder_colors[enc], s=sizes[i], alpha=0.7, edgecolors='k', linewidths=0.3)
    
    for enc in unique_encoders:
        ax1.scatter([], [], color=encoder_colors[enc], label=f"Encoder {enc}")
    
    ax1.legend(title="Encoders", fontsize=10)
    ax1.set_title(f"Training Latent Space - Color by Encoder (Epoch {epoch+1})", fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Plot 2: Color by task key
    for i, (coord, key) in enumerate(zip(tsne_coords, keys)):
        ax2.scatter(coord[0], coord[1], color=key_colors[key], s=sizes[i], alpha=0.7, edgecolors='k', linewidths=0.3)
    
    for key in unique_keys:
        ax2.scatter([], [], color=key_colors[key], label=f"{str(key)[:8]}")
    
    ax2.legend(title="Task Keys", fontsize=10)
    ax2.set_title(f"Training Latent Space - Color by Task Key (Epoch {epoch+1})", fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    overview_save_path = os.path.join(latent_plots_dir, f"latent_space_training_overview_epoch_{epoch+1}.png")
    plt.savefig(overview_save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved overview latent space plot: {overview_save_path}")
    
    # Upload to WandB if available
    if wandb_logger is not None and hasattr(wandb_logger, 'is_initialized') and wandb_logger.is_initialized:
        try:
            import wandb
            
            # Upload individual encoder plots
            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                for enc_idx in unique_encoders:
                    enc_save_path = os.path.join(latent_plots_dir, f"latent_space_training_encoder_{enc_idx}_epoch_{epoch+1}.png")
                    wandb_logger._safe_log({
                        f"training_latent_space_encoder_{enc_idx}": wandb.Image(enc_save_path, caption=f"Encoder {enc_idx} - Epoch {epoch+1}")
                    }, step_hint=epoch+1)
            
            # Upload overview plot
            wandb_logger._safe_log({
                "training_latent_space_overview": wandb.Image(overview_save_path, caption=f"Overview - Epoch {epoch+1}")
            }, step_hint=epoch+1)
            
            print(f"[ OK ] Uploaded training latent space plots to wandb (epoch {epoch+1})")
            
        except Exception as e:
            print(f"[ WARNING ] Could not upload latent space plots to wandb: {e}")

def plot_evaluation_latent_space_by_key_and_encoder(eval_results, save_dir, epoch=None, wandb_logger=None, slider_table=None, upload_slider=False, use_task_optimization=True):
    """
    Plot evaluation latent space showing query and support samples with different colors and shapes.
    Shows task keys with different colors, and query/support samples with different markers.
    """
    if use_task_optimization and 'task_latent_data' in eval_results:
        print("Creating comprehensive evaluation latent space plot...")
        latent_plots_dir = os.path.join(save_dir, "latent_space_plots")
        os.makedirs(latent_plots_dir, exist_ok=True)
        save_path = os.path.join(latent_plots_dir, f"eval_comprehensive_epoch_{epoch+1}.png" if epoch else "eval_comprehensive.png")
        
        task_latent_data = eval_results['task_latent_data']
        all_tuples = []
        sample_types = []  # Track if sample is 'query' or 'support'
        
        # Extract task latents (one per task)
        task_latents_dict = task_latent_data.get('task_latents', {})
        print(f"DEBUG: Found {len(task_latents_dict)} task latents in task_latent_data")
        print(f"DEBUG: Task keys: {list(task_latents_dict.keys())}")
            
        for task_key, task_data in task_latents_dict.items():
            if 'latent_z' in task_data:
                latent_z = task_data['latent_z']  # Already numpy array
                all_tuples.append((latent_z.flatten(), task_key, 0))
                sample_types.append('task')  # Task-level point
                print(f"DEBUG: Added task latent for key '{task_key}' with shape {latent_z.shape}")
            else:
                print(f"DEBUG: Task key '{task_key}' missing 'latent_z' field")
        
        # For task optimization, we have task-level latents which are the main visualization focus
        # Support and query samples are raw tensors, not individual latents
        # So we'll just use the task-level latents for now
        print(f"Created {len(all_tuples)} task-level latent points for visualization")
        
        # Note: In task optimization, individual support/query latents aren't stored
        # because the whole point is to optimize one latent per task, not per sample
        
        if all_tuples:
            title = "Evaluation Latent Space (Query + Support + Task Keys)"
            plot_evaluation_latent_space_comprehensive(all_tuples, sample_types, title=title, save_path=save_path, epoch=epoch, phase="evaluation_comprehensive", infinite_dataloader=False, wandb_logger=wandb_logger, slider_table=slider_table, upload_slider=upload_slider)
            print(f"[ OK ] Saved comprehensive evaluation latent space plot: {save_path}")
            return
        else:
            print("No evaluation latent data found for visualization")
    
    # Fallback to original per-sample visualization
    print("No evaluation latent data to plot.")
    return

def upload_latent_space_slider_to_wandb(tuples, tsne_coords, certainties, keys, encoders, wandb_logger, phase, epoch=None):
    """
    Upload latent space data to wandb for panel creation with epoch slider.
    This accumulates data for the panel slider, not creating an interactive table.
    """
    try:
        import wandb
        
        # Create a table with epoch information for panel slider
        data = []
        for i, (coord, certainty, key, enc) in enumerate(zip(tsne_coords, certainties, keys, encoders)):
            data.append({
                "tsne_x": coord[0],
                "tsne_y": coord[1],
                "certainty": certainty,
                "key": str(key),
                "encoder": str(enc),
                "epoch": epoch + 1 if epoch is not None else "final",
                "phase": phase
            })
        
        # Create a wandb table for panel data
        table = wandb.Table(data=data, columns=["tsne_x", "tsne_y", "certainty", "key", "encoder", "epoch", "phase"])
        
        # Log the table data for panel creation (this will be used by the panel slider)
        wandb_logger._safe_log({
            f"{phase}_latent_space_panel_data_epoch_{epoch+1 if epoch is not None else 'final'}": table
        }, step_hint=epoch+1 if epoch is not None else None)
        
        print(f"[ OK ] Uploaded {phase} latent space data for panel slider (epoch {epoch+1 if epoch is not None else 'final'})")
        
    except Exception as e:
        print(f"[ WARNING ] Could not upload latent space data for panel: {e}")

def create_training_evolution_slider(model, dataloader, device, current_epoch, save_dir, wandb_logger, 
                                   key_list=None, encoder_idx=None, infinite_dataloader=False, max_batches=3,
                                   input_sequences=None, output_sequences=None, training_keys=None,
                                   log_interval=1, max_epochs_to_show=20):
    """
    Create a training evolution slider that shows latent space evolution across epochs.
    Accumulates data from multiple epochs and creates a single interactive chart.
    
    Args:
        model: The trained model
        dataloader: DataLoader for training data
        device: Device to run on
        current_epoch: Current epoch number (0-indexed)
        save_dir: Directory to save plots
        wandb_logger: WandB logger instance
        key_list: List of keys for each sample
        encoder_idx: Specific encoder to use
        infinite_dataloader: Whether using infinite dataloader
        max_batches: Maximum batches to process
        input_sequences: Pre-computed input sequences
        output_sequences: Pre-computed output sequences
        training_keys: List of training keys
        log_interval: How often to log (every N epochs)
        max_epochs_to_show: Maximum number of epochs to show in slider
    """
    # Only create slider every log_interval epochs
    if (current_epoch + 1) % log_interval != 0:
        return None
    
    # Create evolution folder
    evolution_dir = os.path.join(save_dir, "training_evolution")
    os.makedirs(evolution_dir, exist_ok=True)
    
    # Initialize or load existing evolution data
    evolution_data_file = os.path.join(evolution_dir, "evolution_data.pkl")
    evolution_data = {
        'epochs': [],
        'latent_tuples': [],
        'logvars': [],
        'tsne_coords': [],
        'certainties': [],
        'keys': [],
        'encoders': []
    }
    
    if os.path.exists(evolution_data_file):
        try:
            with open(evolution_data_file, 'rb') as f:
                evolution_data = pickle.load(f)
        except Exception as e:
            print(f"[ WARNING ] Could not load existing evolution data: {e}")
    
    # Extract current epoch's latent data
    current_tuples = []
    current_logvars = None
    
    if input_sequences is not None and output_sequences is not None and training_keys is not None:
        model.eval()
        batch_size = 128
        input_arr = np.array(input_sequences)
        output_arr = np.array(output_sequences)
        
        if key_list is not None and len(key_list) == len(input_arr):
            sample_keys = key_list
        else:
            n_per_key = len(input_arr) // len(training_keys)
            sample_keys = []
            for k in training_keys:
                sample_keys.extend([k] * n_per_key)
            sample_keys.extend([training_keys[-1]] * (len(input_arr) - len(sample_keys)))
        
        logvars_list = []
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                for i in range(0, len(input_arr), batch_size):
                    batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    mu, logvar, _ = model.multi_encoder.encoders[enc_idx](batch_inputs, batch_outputs)
                    z = model.reparameterize(mu, logvar)
                    z_np = z.detach().cpu().numpy()
                    logvars_list.append(logvar.detach().cpu().numpy())
                    batch_keys = sample_keys[i:i+batch_size]
                    for latent, key in zip(z_np, batch_keys):
                        current_tuples.append((latent, key, enc_idx))
            current_logvars = np.concatenate(logvars_list, axis=0)
        else:
            for i in range(0, len(input_arr), batch_size):
                batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                mu, logvar, _ = model.encoder(batch_inputs, batch_outputs)
                z = model.reparameterize(mu, logvar)
                z_np = z.detach().cpu().numpy()
                logvars_list.append(logvar.detach().cpu().numpy())
                batch_keys = sample_keys[i:i+batch_size]
                for latent, key in zip(z_np, batch_keys):
                    current_tuples.append((latent, key, 0))
            current_logvars = np.concatenate(logvars_list, axis=0)
        model.train()
    else:
        # Fallback to dataloader-based approach
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                t, l = extract_latent_key_encoder_tuples_from_dataloader(
                    model, dataloader, device, key_list=key_list, encoder_idx=enc_idx, max_batches=max_batches, return_logvar=True
                )
                current_tuples += t
                if l is not None:
                    if current_logvars is None:
                        current_logvars = l
                    else:
                        current_logvars = np.concatenate([current_logvars, l], axis=0)
        else:
            current_tuples, current_logvars = extract_latent_key_encoder_tuples_from_dataloader(
                model, dataloader, device, key_list=key_list, encoder_idx=encoder_idx, max_batches=max_batches, return_logvar=True
            )
    
    if not current_tuples:
        print("No training latent data to add to evolution slider.")
        return None
    
    # Add current epoch data to evolution data
    evolution_data['epochs'].append(current_epoch + 1)
    evolution_data['latent_tuples'].append(current_tuples)
    evolution_data['logvars'].append(current_logvars)
    
    # Calculate t-SNE coordinates and other data for interactive plots
    if current_tuples:
        latents = np.array([x[0] for x in current_tuples])
        keys = [x[1] for x in current_tuples]
        encoders = [x[2] for x in current_tuples]
        
        # Calculate certainties from logvars
        if current_logvars is not None:
            certainties = -np.mean(current_logvars, axis=1)
        else:
            certainties = np.ones(len(latents)) * 0.5
        
        # t-SNE for interactive plot
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
        tsne_coords = tsne.fit_transform(latents)
        
        # Store the data for interactive plots
        evolution_data['tsne_coords'].append(tsne_coords)
        evolution_data['certainties'].append(certainties)
        evolution_data['keys'].append(keys)
        evolution_data['encoders'].append(encoders)
    
    # Limit the number of epochs to show
    if len(evolution_data['epochs']) > max_epochs_to_show:
        # Remove oldest epochs
        remove_count = len(evolution_data['epochs']) - max_epochs_to_show
        evolution_data['epochs'] = evolution_data['epochs'][remove_count:]
        evolution_data['latent_tuples'] = evolution_data['latent_tuples'][remove_count:]
        evolution_data['logvars'] = evolution_data['logvars'][remove_count:]
        evolution_data['tsne_coords'] = evolution_data['tsne_coords'][remove_count:]
        evolution_data['certainties'] = evolution_data['certainties'][remove_count:]
        evolution_data['keys'] = evolution_data['keys'][remove_count:]
        evolution_data['encoders'] = evolution_data['encoders'][remove_count:]
    
    # Save updated evolution data
    try:
        with open(evolution_data_file, 'wb') as f:
            pickle.dump(evolution_data, f)
    except Exception as e:
        print(f"[ WARNING ] Could not save evolution data: {e}")
    
    # Create evolution slider plot
    if len(evolution_data['epochs']) >= 2:  # Need at least 2 epochs for evolution
        create_evolution_slider_plot(evolution_data, evolution_dir, current_epoch, wandb_logger)
    
    return evolution_data

def create_evolution_slider_plot(evolution_data, save_dir, current_epoch, wandb_logger):
    """
    Create an interactive slider plot showing latent space evolution across epochs.
    
    Args:
        evolution_data: Dictionary containing evolution data across epochs
        save_dir: Directory to save the plot
        current_epoch: Current epoch number
        wandb_logger: WandB logger instance
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    
    if len(evolution_data['epochs']) < 2:
        return
    
    # Create evolution folder
    evolution_dir = os.path.join(save_dir, "training_evolution")
    os.makedirs(evolution_dir, exist_ok=True)
    
    # Create figure with subplots for each epoch
    n_epochs = len(evolution_data['epochs'])
    fig, axes = plt.subplots(2, min(5, n_epochs), figsize=(20, 8))
    if n_epochs == 1:
        axes = axes.reshape(1, -1)
    elif n_epochs <= 5:
        axes = axes.reshape(1, -1)
    
    # Get all unique keys for consistent coloring
    all_keys = set()
    for tuples in evolution_data['latent_tuples']:
        for _, key, _ in tuples:
            all_keys.add(key)
    unique_keys = sorted(list(all_keys))
    key_colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
    
    # Plot each epoch
    for epoch_idx, (epoch, tuples, logvars) in enumerate(zip(
        evolution_data['epochs'], 
        evolution_data['latent_tuples'], 
        evolution_data['logvars']
    )):
        if epoch_idx >= 10:  # Only show first 10 epochs in grid
            break
            
        ax = axes[epoch_idx // 5, epoch_idx % 5] if n_epochs > 5 else axes[epoch_idx]
        
        if not tuples:
            continue
        
        # Extract data
        latents = np.array([x[0] for x in tuples])
        keys = [x[1] for x in tuples]
        encoders = [x[2] for x in tuples]
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
        tsne_coords = tsne.fit_transform(latents)
        
        # Certainty
        if logvars is not None:
            certainties = -np.mean(logvars, axis=1)
            min_c, max_c = np.min(certainties), np.max(certainties)
            sizes = 40 + 160 * (certainties - min_c) / (max_c - min_c + 1e-8)
        else:
            sizes = 60
        
        # Plot points
        for i, (coord, key, enc) in enumerate(zip(tsne_coords, keys, encoders)):
            ax.scatter(coord[0], coord[1], color=key_colors[key], s=sizes[i] if isinstance(sizes, np.ndarray) else sizes, 
                      alpha=0.7, edgecolors='k', linewidths=0.3)
        
        ax.set_title(f'Epoch {epoch}', fontsize=10)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    # Add legend
    if n_epochs <= 5:
        legend_ax = axes[-1] if n_epochs > 1 else axes
    else:
        legend_ax = axes[1, -1]
    
    for k in unique_keys:
        legend_ax.scatter([], [], color=key_colors[k], label=f"{str(k)[:4]}")
    legend_ax.legend(title="Key (first 4 chars)", fontsize=8, loc='upper right')
    
    plt.suptitle(f'Training Latent Space Evolution (Epochs {evolution_data["epochs"][0]}-{evolution_data["epochs"][-1]})', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    evolution_plot_path = os.path.join(evolution_dir, f"training_evolution_epoch_{current_epoch+1}.png")
    plt.savefig(evolution_plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    # Upload to wandb with proper slider format
    if wandb_logger is not None and hasattr(wandb_logger, 'is_initialized') and wandb_logger.is_initialized:
        try:
            import wandb
            
            # Create a wandb table for the evolution slider
            evolution_table = wandb.Table(columns=["epoch", "image", "description"])
            
            # Add each epoch's plot to the table
            for epoch_num in evolution_data['epochs']:
                epoch_plot_path = os.path.join(evolution_dir, f"training_evolution_epoch_{epoch_num}.png")
                if os.path.exists(epoch_plot_path):
                    evolution_table.add_data(epoch_num, wandb.Image(epoch_plot_path), f"Training Evolution Epoch {epoch_num}")
            
            # Log the evolution slider as interactive panel
            wandb_logger._safe_log({
                f"training_evolution_slider": evolution_table
            }, step_hint=current_epoch+1)
            
            # Also log the current evolution plot as individual image
            wandb_logger._safe_log({
                f"training_evolution_epoch_{current_epoch+1}": wandb.Image(evolution_plot_path)
            }, step_hint=current_epoch+1)
            
            # Create interactive scatter plot for training evolution (like trajectory plots)
            if len(evolution_data['epochs']) > 1:
                # Combine all epochs' data for interactive plot
                all_tsne_coords = []
                all_certainties = []
                all_keys = []
                all_encoders = []
                all_epochs = []
                
                for epoch_idx, epoch_num in enumerate(evolution_data['epochs']):
                    if epoch_idx < len(evolution_data['tsne_coords']):
                        all_tsne_coords.extend(evolution_data['tsne_coords'][epoch_idx])
                        all_certainties.extend(evolution_data['certainties'][epoch_idx])
                        all_keys.extend(evolution_data['keys'][epoch_idx])
                        all_encoders.extend(evolution_data['encoders'][epoch_idx])
                        all_epochs.extend([epoch_num] * len(evolution_data['tsne_coords'][epoch_idx]))
                
                if all_tsne_coords:
                    # Create interactive scatter plot
                    data = []
                    for i, (coord, certainty, key, enc, epoch) in enumerate(zip(all_tsne_coords, all_certainties, all_keys, all_encoders, all_epochs)):
                        data.append({
                            "tsne_x": coord[0],
                            "tsne_y": coord[1],
                            "certainty": certainty,
                            "key": str(key),
                            "encoder": str(enc),
                            "epoch": epoch
                        })
                    
                    table = wandb.Table(data=data, columns=["tsne_x", "tsne_y", "certainty", "key", "encoder", "epoch"])
                    plot = wandb.plot_table(
                        "wandb/scatter",
                        table,
                        {"x": "tsne_x", "y": "tsne_y", "size": "certainty", "color": "key", "label": "encoder", "group": "epoch"},
                        title=f"Training Latent Space Evolution (Epochs {evolution_data['epochs'][0]}-{evolution_data['epochs'][-1]})"
                    )
                    wandb_logger._safe_log({
                        f"training_evolution_interactive": plot
                    }, step_hint=current_epoch+1)
                    
                    print(f"[ OK ] Uploaded training evolution plot and slider to wandb: {evolution_plot_path}")
        except Exception as e:
            print(f"[ WARNING ] Could not upload training evolution plot to wandb: {e}")
    
    print(f"[ OK ] Created training evolution plot: {evolution_plot_path}")

def create_training_latent_space_panel(model, dataloader, device, current_epoch, save_dir, wandb_logger, 
                                     key_list=None, encoder_idx=None, infinite_dataloader=False, max_batches=3,
                                     input_sequences=None, output_sequences=None, training_keys=None,
                                     log_interval=1, max_epochs_to_show=20):
    """
    Create a training latent space panel with epoch slider for wandb.
    This creates a panel that allows sliding between different epochs to visualize latent space evolution.
    
    Args:
        model: The trained model
        dataloader: DataLoader for training data
        device: Device to run on
        current_epoch: Current epoch number (0-indexed)
        save_dir: Directory to save plots
        wandb_logger: WandB logger instance
        key_list: List of keys for each sample
        encoder_idx: Specific encoder to use
        infinite_dataloader: Whether using infinite dataloader
        max_batches: Maximum batches to process
        input_sequences: Pre-computed input sequences
        output_sequences: Pre-computed output sequences
        training_keys: List of training keys
        log_interval: How often to log (every N epochs)
        max_epochs_to_show: Maximum number of epochs to show in panel
    """
    # Only create panel every log_interval epochs
    if (current_epoch + 1) % log_interval != 0:
        return None
    
    # Create panel folder
    panel_dir = os.path.join(save_dir, "training_panel")
    os.makedirs(panel_dir, exist_ok=True)
    
    # Initialize or load existing panel data
    panel_data_file = os.path.join(panel_dir, "panel_data.pkl")
    panel_data = {
        'epochs': [],
        'latent_tuples': [],
        'logvars': [],
        'tsne_coords': [],
        'certainties': [],
        'keys': [],
        'encoders': []
    }
    
    if os.path.exists(panel_data_file):
        try:
            with open(panel_data_file, 'rb') as f:
                panel_data = pickle.load(f)
        except Exception as e:
            print(f"[ WARNING ] Could not load existing panel data: {e}")
    
    # Extract current epoch's latent data
    current_tuples = []
    current_logvars = None
    
    if input_sequences is not None and output_sequences is not None and training_keys is not None:
        model.eval()
        batch_size = 128
        input_arr = np.array(input_sequences)
        output_arr = np.array(output_sequences)
        
        if key_list is not None and len(key_list) == len(input_arr):
            sample_keys = key_list
        else:
            n_per_key = len(input_arr) // len(training_keys)
            sample_keys = []
            for k in training_keys:
                sample_keys.extend([k] * n_per_key)
            sample_keys.extend([training_keys[-1]] * (len(input_arr) - len(sample_keys)))
        
        logvars_list = []
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                for i in range(0, len(input_arr), batch_size):
                    batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                    mu, logvar, _ = model.multi_encoder.encoders[enc_idx](batch_inputs, batch_outputs)
                    z = model.reparameterize(mu, logvar)
                    z_np = z.detach().cpu().numpy()
                    logvars_list.append(logvar.detach().cpu().numpy())
                    batch_keys = sample_keys[i:i+batch_size]
                    for latent, key in zip(z_np, batch_keys):
                        current_tuples.append((latent, key, enc_idx))
            current_logvars = np.concatenate(logvars_list, axis=0)
        else:
            for i in range(0, len(input_arr), batch_size):
                batch_inputs = torch.tensor(input_arr[i:i+batch_size], dtype=torch.float32, device=device)
                batch_outputs = torch.tensor(output_arr[i:i+batch_size], dtype=torch.float32, device=device)
                mu, logvar, _ = model.encoder(batch_inputs, batch_outputs)
                z = model.reparameterize(mu, logvar)
                z_np = z.detach().cpu().numpy()
                logvars_list.append(logvar.detach().cpu().numpy())
                batch_keys = sample_keys[i:i+batch_size]
                for latent, key in zip(z_np, batch_keys):
                    current_tuples.append((latent, key, 0))
            current_logvars = np.concatenate(logvars_list, axis=0)
        model.train()
    else:
        # Fallback to dataloader-based approach
        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
            num_encoders = getattr(model, 'num_encoders', 1)
            for enc_idx in range(num_encoders):
                t, l = extract_latent_key_encoder_tuples_from_dataloader(
                    model, dataloader, device, key_list=key_list, encoder_idx=enc_idx, max_batches=max_batches, return_logvar=True
                )
                current_tuples += t
                if l is not None:
                    if current_logvars is None:
                        current_logvars = l
                    else:
                        current_logvars = np.concatenate([current_logvars, l], axis=0)
        else:
            current_tuples, current_logvars = extract_latent_key_encoder_tuples_from_dataloader(
                model, dataloader, device, key_list=key_list, encoder_idx=encoder_idx, max_batches=max_batches, return_logvar=True
            )
    
    if not current_tuples:
        print("No training latent data for panel.")
        return None
    
    # Calculate t-SNE coordinates for current epoch
    latents = np.array([x[0] for x in current_tuples])
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    current_tsne_coords = tsne.fit_transform(latents)
    
    # Calculate certainties from logvars if available
    if current_logvars is not None:
        current_certainties = -np.mean(current_logvars, axis=1)
    else:
        current_certainties = np.ones(len(latents)) * 0.5
    
    # Extract keys and encoders
    current_keys = [x[1] for x in current_tuples]
    current_encoders = [x[2] for x in current_tuples]
    
    # Add current epoch data to panel data
    panel_data['epochs'].append(current_epoch + 1)
    panel_data['latent_tuples'].append(current_tuples)
    panel_data['logvars'].append(current_logvars)
    panel_data['tsne_coords'].append(current_tsne_coords)
    panel_data['certainties'].append(current_certainties)
    panel_data['keys'].append(current_keys)
    panel_data['encoders'].append(current_encoders)
    
    # Limit the number of epochs to show
    if len(panel_data['epochs']) > max_epochs_to_show:
        # Remove oldest epochs
        panel_data['epochs'] = panel_data['epochs'][-max_epochs_to_show:]
        panel_data['latent_tuples'] = panel_data['latent_tuples'][-max_epochs_to_show:]
        panel_data['logvars'] = panel_data['logvars'][-max_epochs_to_show:]
        panel_data['tsne_coords'] = panel_data['tsne_coords'][-max_epochs_to_show:]
        panel_data['certainties'] = panel_data['certainties'][-max_epochs_to_show:]
        panel_data['keys'] = panel_data['keys'][-max_epochs_to_show:]
        panel_data['encoders'] = panel_data['encoders'][-max_epochs_to_show:]
    
    # Save panel data
    with open(panel_data_file, 'wb') as f:
        pickle.dump(panel_data, f)
    
    # Create panel visualization
    if wandb_logger and hasattr(wandb_logger, 'is_initialized') and wandb_logger.is_initialized:
        try:
            import wandb
            
            # Create a comprehensive table with all epoch data for panel
            all_data = []
            for epoch_idx, epoch in enumerate(panel_data['epochs']):
                tsne_coords = panel_data['tsne_coords'][epoch_idx]
                certainties = panel_data['certainties'][epoch_idx]
                keys = panel_data['keys'][epoch_idx]
                encoders = panel_data['encoders'][epoch_idx]
                
                for i, (coord, certainty, key, enc) in enumerate(zip(tsne_coords, certainties, keys, encoders)):
                    all_data.append({
                        "tsne_x": coord[0],
                        "tsne_y": coord[1],
                        "certainty": certainty,
                        "key": str(key),
                        "encoder": str(enc),
                        "epoch": epoch,
                        "epoch_idx": epoch_idx
                    })
            
            # Create wandb table for panel
            panel_table = wandb.Table(data=all_data, columns=["tsne_x", "tsne_y", "certainty", "key", "encoder", "epoch", "epoch_idx"])
            
            # Log the panel table
            wandb_logger._safe_log({
                "training_latent_space_panel": panel_table
            }, step_hint=current_epoch+1)
            
            print(f"[ OK ] Created training latent space panel with {len(panel_data['epochs'])} epochs")
            
        except Exception as e:
            print(f"[ WARNING ] Could not create training latent space panel: {e}")
    
    return panel_data

def create_standalone_latent_space_plot(trajectory_info, model, save_dir, epoch, sample_idx, evaluated_key=None, device='cuda', wandb_logger=None, eval_results=None):
    """
    Create a standalone latent space plot for a specific trajectory.
    This replicates exactly the latent space visualization from the trajectory figure.
    
    Args:
        trajectory_info: Dictionary containing trajectory data
        model: The trained model
        save_dir: Directory to save the plot
        epoch: Current epoch number
        sample_idx: Index of the sample (0, 1, 2, etc.)
        evaluated_key: The key being evaluated (optional)
        device: Device to run on
        wandb_logger: WandB logger for uploading plots
        eval_results: Evaluation results containing support/query latents
        
    Returns:
        str: Path to the saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create trajectory plots directory
    trajectory_plots_dir = os.path.join(save_dir, "trajectory_plots")
    os.makedirs(trajectory_plots_dir, exist_ok=True)
    
    # Generate filename with the specified naming convention
    # Extract key hash from evaluated_key if available
    key_hash = ""
    if evaluated_key:
        # Create a hash from the key (first 8 characters of hash)
        import hashlib
        key_hash = hashlib.md5(evaluated_key.encode()).hexdigest()[:8]
    
    filename = f"trajectory_epoch{epoch}_{key_hash}_sample{sample_idx}_latent_space_wandb.png"
    save_path = os.path.join(trajectory_plots_dir, filename)
    
    print(f"Creating standalone latent space plot: {filename}")
    
    # Load unified latent data (training + support + query + trajectory) - same as trajectory figure
    from LPN_reproduction.evaluate_trajectory import load_unified_latent_data_with_trajectory
    training_latent_data, training_tsne_2d, training_labels, training_colors, trajectory_tsne_2d, all_labels, all_tsne_2d = load_unified_latent_data_with_trajectory(
        save_dir, model, device, trajectory_info, eval_results=eval_results
    )
    
    # Get trajectory data
    z_vectors = trajectory_info.get('z_vectors', [])
    losses = trajectory_info.get('losses', [])
    
    if not z_vectors or len(z_vectors) < 2:
        print(f"Warning: No trajectory data found for sample {sample_idx}")
        return None
    
    # Create the plot - replicate exactly the trajectory figure latent space
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectory in latent space using precomputed t-SNE coordinates (same as trajectory figure)
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
                    training_encoders.append(int(encoder))
                else:
                    training_keys.append(label)
                    training_encoders.append(0)  # Default encoder
            
            # Get the key of the sample being evaluated
            evaluated_key = trajectory_info.get('evaluated_key', 'unknown')
            print(f"DEBUG: Using evaluated_key: '{evaluated_key}'")
            
            # Create light colors for encoders (different light colors for each encoder)
            unique_encoders = sorted(list(set(training_encoders)))
            encoder_colors = {
                enc: plt.cm.tab10(enc % 10) for enc in unique_encoders  # More visible colors from tab10
            }
            
            # Create bright colors for the evaluated key (replaced yellow with more visible colors)
            bright_colors = ['red', 'orange', 'darkorange', 'lime', 'cyan', 'magenta', 'pink', 'brown']
            evaluated_color = bright_colors[hash(evaluated_key) % len(bright_colors)]
            
            # Plot background points colored by encoder (light colors)
            for encoder in unique_encoders:
                indices = [i for i, enc in enumerate(training_encoders) if enc == encoder]
                if indices:
                    x_coords = training_tsne_2d[indices, 0]
                    y_coords = training_tsne_2d[indices, 1]
                    color = encoder_colors[encoder]
                    
                    # Use alpha for background effect
                    ax.scatter(x_coords, y_coords, color=color, alpha=0.6, s=35, 
                           edgecolors='none', label=f'Encoder {encoder} (Light)')
            
            # Highlight samples with the same key as evaluated sample (bright colors)
            same_key_indices = [i for i, key in enumerate(training_keys) if key == evaluated_key]
            if same_key_indices:
                x_coords = training_tsne_2d[same_key_indices, 0]
                y_coords = training_tsne_2d[same_key_indices, 1]
                
                # Use bright color for same key samples
                ax.scatter(x_coords, y_coords, color=evaluated_color, alpha=0.9, s=50, 
                       edgecolors='black', linewidth=1.5, 
                       label=f'Same Key: {evaluated_key[:8]} (Bright)')
            
            # Plot support and query samples for the evaluated key only
            if all_labels is not None and all_tsne_2d is not None:
                # Filter support samples for the evaluated key
                support_indices = []
                for i, label in enumerate(all_labels):
                    if 'support' in label and evaluated_key in label:
                        support_indices.append(i)
                
                print(f"DEBUG: Found {len(support_indices)} support samples for key '{evaluated_key}'")
                if support_indices:
                    x_coords = all_tsne_2d[support_indices, 0]
                    y_coords = all_tsne_2d[support_indices, 1]
                    ax.scatter(x_coords, y_coords, color='blue', alpha=0.8, s=35, 
                            marker='s', edgecolors='black', linewidth=1.0, label='Support Samples')
                
                # Filter query samples for the evaluated key
                query_indices = []
                for i, label in enumerate(all_labels):
                    if 'query' in label and evaluated_key in label:
                        query_indices.append(i)
                
                print(f"DEBUG: Found {len(query_indices)} query samples for key '{evaluated_key}'")
                if query_indices:
                    x_coords = all_tsne_2d[query_indices, 0]
                    y_coords = all_tsne_2d[query_indices, 1]
                    ax.scatter(x_coords, y_coords, color='red', alpha=0.8, s=35, 
                            marker='^', edgecolors='black', linewidth=1.0, label='Query Samples')
    
    # Use unified trajectory coordinates (already computed in unified t-SNE) - same as trajectory figure
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
        
        # Plot trajectory in latent space using precomputed t-SNE coordinates (same as trajectory figure)
        trajectory_scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', 
                                     s=50, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Draw arrows between consecutive trajectory points - same as trajectory figure
        for i in range(len(z_2d) - 1):
            ax.annotate('', xy=z_2d[i+1], xytext=z_2d[i],
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=2))
        
        # Mark start and end points - same as trajectory figure
        ax.scatter(z_2d[0, 0], z_2d[0, 1], color='green', s=100, marker='o', 
                label='Trajectory Start', edgecolors='black', linewidth=3, zorder=10, alpha=0.8)
        ax.scatter(z_2d[-1, 0], z_2d[-1, 1], color='red', s=100, marker='s', 
                label='Trajectory End', edgecolors='black', linewidth=3, zorder=10, alpha=0.8)
        
        # Add colorbar - same as trajectory figure
        cbar = plt.colorbar(trajectory_scatter, ax=ax, shrink=0.6)
        cbar.set_label('Loss', rotation=270, labelpad=20)
    
    # Add title
    title = f"Trajectory Latent Space - Sample {sample_idx}"
    if evaluated_key:
        title += f" (Key: {evaluated_key[:20]}{'...' if len(evaluated_key) > 20 else ''})"
    title += f" (Epoch {epoch})"
    ax.set_title(title, fontsize=14)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved standalone latent space plot: {save_path}")
    
    # Upload to WandB if available - using step = epoch
    if wandb_logger is not None and hasattr(wandb_logger, 'is_initialized') and wandb_logger.is_initialized:
        try:
            import wandb
            
            # Use consistent key naming for slider visualization across epochs (same as trajectory plots)
            # Format: trajectory_latent_space_{key}_sample{idx} - this creates sliders
            panel_name = f"trajectory_latent_space_{evaluated_key}_sample{sample_idx}" if evaluated_key else f"trajectory_latent_space_sample_{sample_idx}"
            
            wandb_logger._safe_log({
                panel_name: wandb.Image(save_path, caption=f"Trajectory Latent Space - Sample {sample_idx} - Epoch {epoch}")
            }, step_hint=epoch)  # Using step = epoch as requested
            
            print(f"[ OK ] Uploaded trajectory latent space plot to wandb panel '{panel_name}' (step={epoch})")
            
        except Exception as e:
            print(f"[ WARNING ] Could not upload trajectory latent space plot to wandb: {e}")
    
    return save_path

def extract_latent_data_from_dataloader(dataloader, model, max_batches=3, use_optimization=True):
    from utils.latent_functions import optimize_task_latent
    device = next(model.parameters()).device
    model.eval()
    
    if use_optimization:
        # Group samples by task key and preserve sample info
        task_samples = {}
        sample_to_task = []  # Track (sample_idx, task_key) mapping
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches: break
            
            batch_input, batch_target, batch_keys = batch[:3] if len(batch) >= 3 else (*batch[:2], [f"sample_{i}" for i in range(batch[0].size(0))])
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            
            for i, key in enumerate(batch_keys):
                if key not in task_samples: task_samples[key] = []
                task_samples[key].append((batch_input[i:i+1], batch_target[i:i+1]))
                sample_to_task.append(key)
        
        # Optimize per task and assign to all samples of that task
        latent_optimization = settings.get_latent_optimization()
        task_latents = {}  # Store optimized latents per task
        
        for task_key, support_samples in task_samples.items():
            task_latent, _, _ = optimize_task_latent(
                model, support_samples, task_key,
                num_steps=latent_optimization['training']['num_steps'],
                lr=latent_optimization['training']['learning_rate']
            )
            task_latents[task_key] = task_latent.cpu().numpy().flatten()
        
        # Create result with one entry per task (not per sample)
        result = {'encoder_0': {'latent_zs': [], 'data_type': 'task_optimized'}}
        task_keys_ordered = []  # Keep track of task order
        
        for task_key, task_latent in task_latents.items():
            result['encoder_0']['latent_zs'].append(task_latent)
            task_keys_ordered.append(task_key)
        
        return result, task_keys_ordered
    
    # Regular extraction
    result = {'encoder_0': {'latent_zs': [], 'data_type': 'training'}}
    sample_keys = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches: break
            batch_input, batch_target, batch_keys = batch[:3] if len(batch) >= 3 else (*batch[:2], [f"sample_{i}" for i in range(batch[0].size(0))])
            mu, log_var, _ = model.encoder(batch_input.to(device), batch_target.to(device))
            z = model.reparameterize(mu, log_var)
            result['encoder_0']['latent_zs'].extend([z[i].cpu().numpy() for i in range(z.size(0))])
            sample_keys.extend(batch_keys)
    return result, sample_keys

def plot_evaluation_latent_space_comprehensive(latent_tuples, sample_types, title, save_path=None, key_colors=None, epoch=None, phase=None, infinite_dataloader=False, logvars=None, wandb_logger=None, slider_table=None, upload_slider=False):
    """
    Plot evaluation latent space with different markers for query, support, and task samples.
    Shows task keys with different colors, and query/support samples with different markers.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    if not latent_tuples or len(latent_tuples) == 0:
        print("No latent data to plot.")
        return

    latents = np.array([x[0] for x in latent_tuples])
    keys = [x[1] for x in latent_tuples]
    encoders = [x[2] for x in latent_tuples]
    unique_keys = sorted(list(set(keys)))
    
    if key_colors is None:
        key_colors = {k: cm.tab20(i % 20) for i, k in enumerate(unique_keys)}

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    tsne_coords = tsne.fit_transform(latents)

    # Certainty as -logvar.mean(1)
    if logvars is not None:
        certainties = -np.mean(logvars, axis=1)
        min_c, max_c = np.min(certainties), np.max(certainties)
        sizes = 40 + 160 * (certainties - min_c) / (max_c - min_c + 1e-8)
    else:
        certainties = [1]*len(tsne_coords)
        sizes = 60

    plt.figure(figsize=(14, 10))
    
    # Plot with different markers for different sample types
    for i, (coord, key, enc) in enumerate(zip(tsne_coords, keys, encoders)):
        sample_type = sample_types[i] if i < len(sample_types) else 'unknown'
        
        # Different markers for different sample types
        if sample_type == 'task':
            marker = 's'  # Square for task-level points
            size = sizes[i] if isinstance(sizes, np.ndarray) else sizes
            alpha = 0.8
            edgecolor = 'k'
            linewidth = 1.0
        elif sample_type == 'support':
            marker = 'o'  # Circle for support samples
            size = (sizes[i] if isinstance(sizes, np.ndarray) else sizes) * 0.7
            alpha = 0.6
            edgecolor = 'k'
            linewidth = 0.5
        elif sample_type == 'query':
            marker = '^'  # Triangle for query samples
            size = (sizes[i] if isinstance(sizes, np.ndarray) else sizes) * 0.8
            alpha = 0.7
            edgecolor = 'k'
            linewidth = 0.8
        else:
            marker = 'o'
            size = sizes[i] if isinstance(sizes, np.ndarray) else sizes
            alpha = 0.5
            edgecolor = 'k'
            linewidth = 0.3
        
        plt.scatter(coord[0], coord[1], color=key_colors[key], s=size, alpha=alpha, 
                   edgecolors=edgecolor, linewidths=linewidth, marker=marker)
        
        # Add labels for task-level points only to avoid clutter
        if sample_type == 'task':
            label = f"{str(key)[:4]}"
            if enc is not None:
                label += f"/E{enc}"
            plt.text(coord[0], coord[1], label, fontsize=8, color=key_colors[key], alpha=0.9,
                    ha='center', va='center', weight='bold')
    
    # Create legend with different markers
    legend_elements = []
    
    # Task-level points legend
    for k in unique_keys:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=key_colors[k], 
                                        markersize=8, label=f"Task {str(k)[:4]}"))
    
    # Sample type legend
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                    markersize=8, label='Task Key'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                    markersize=6, label='Support Sample'))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                                    markersize=6, label='Query Sample'))
    
    plt.legend(handles=legend_elements, title="Sample Types", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    full_title = title
    if phase:
        full_title = f"{phase.title()} Latent Space: " + full_title
    if epoch is not None:
        full_title += f" (Epoch {epoch+1})"
    if infinite_dataloader:
        full_title += " [Infinite Dataloader]"
    plt.title(full_title, fontsize=15)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved comprehensive evaluation latent space plot: {save_path}")
    else:
        plt.show()

    # Upload to wandb if requested
    if wandb_logger and upload_slider:
        try:
            upload_latent_space_slider_to_wandb(latent_tuples, tsne_coords, certainties, keys, encoders, wandb_logger, phase, epoch)
        except Exception as e:
            print(f"[ WARNING ] Could not upload latent space data: {e}")

def plot_original_bonnet_latent_space(eval_results, save_dir, epoch=None, wandb_logger=None, slider_table=None, upload_slider=False):
    """
    Original Bonnet approach: Plot latent space with all samples, where latents are directly sampled 
    from the encoder posterior for BOTH training set and evaluation set.
    
    Args:
        eval_results: Results from evaluate_model_original_bonnet_approach
        save_dir: Directory to save plots
        epoch: Current epoch number
        wandb_logger: WandB logger instance
        slider_table: Table for slider visualization
        upload_slider: Whether to upload to slider
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    
    if 'latent_data' not in eval_results:
        print("No latent data found in evaluation results")
        return
    
    latent_data = eval_results['latent_data']
    
    # Extract all latents and keys
    all_latents = []
    all_keys = []
    all_sample_types = []  # 'support' or 'query'
    
    # Add support latents
    if 'support_latents' in latent_data and latent_data['support_latents']:
        support_latents = np.array(latent_data['support_latents'])
        support_keys = latent_data['support_keys']
        
        all_latents.extend(support_latents)
        all_keys.extend(support_keys)
        all_sample_types.extend(['support'] * len(support_latents))
        
        print(f"Added {len(support_latents)} support latents")
    
    # Add query latents
    if 'query_latents' in latent_data and latent_data['query_latents']:
        query_latents = np.array(latent_data['query_latents'])
        query_keys = latent_data['query_keys']
        
        all_latents.extend(query_latents)
        all_keys.extend(query_keys)
        all_sample_types.extend(['query'] * len(query_latents))
        
        print(f"Added {len(query_latents)} query latents")
    
    if not all_latents:
        print("No latent data to visualize")
        return
    
    # Convert to numpy array
    all_latents = np.array(all_latents)
    print(f"Total latents shape: {all_latents.shape}")
    
    # Apply t-SNE for 2D visualization
    print("Applying t-SNE to all latents...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents)//4))
    tsne_coords = tsne.fit_transform(all_latents)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Get unique keys for color mapping
    unique_keys = sorted(list(set(all_keys)))
    key_colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
    
    # Plot support samples
    support_mask = np.array(all_sample_types) == 'support'
    if np.any(support_mask):
        support_coords = tsne_coords[support_mask]
        support_keys_subset = [all_keys[i] for i in range(len(all_keys)) if all_sample_types[i] == 'support']
        
        for i, (coord, key) in enumerate(zip(support_coords, support_keys_subset)):
            plt.scatter(coord[0], coord[1], color=key_colors[key], s=100, alpha=0.7, 
                       marker='o', edgecolors='k', linewidths=0.5, label=f'Support: {key[:4]}' if i == 0 else "")
    
    # Plot query samples
    query_mask = np.array(all_sample_types) == 'query'
    if np.any(query_mask):
        query_coords = tsne_coords[query_mask]
        query_keys_subset = [all_keys[i] for i in range(len(all_keys)) if all_sample_types[i] == 'query']
        
        for i, (coord, key) in enumerate(zip(query_coords, query_keys_subset)):
            plt.scatter(coord[0], coord[1], color=key_colors[key], s=150, alpha=0.8, 
                       marker='s', edgecolors='k', linewidths=1.0, label=f'Query: {key[:4]}' if i == 0 else "")
    
    # Create legend
    legend_elements = []
    for key in unique_keys:
        color = key_colors[key]
        # Add support legend element
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=8, label=f'Support: {key[:8]}'))
        # Add query legend element  
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                                        markersize=10, label=f'Query: {key[:8]}'))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.title(f"Original Bonnet Latent Space (Epoch {epoch+1 if epoch else 'Final'})\n"
              f"Support: {np.sum(support_mask)} samples, Query: {np.sum(query_mask)} samples", fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    latent_plots_dir = os.path.join(save_dir, "latent_space_plots")
    os.makedirs(latent_plots_dir, exist_ok=True)
    save_path = os.path.join(latent_plots_dir, f"original_bonnet_latent_space_epoch_{epoch+1}.png" if epoch else "original_bonnet_latent_space.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] Saved original Bonnet latent space plot: {save_path}")
    
    # Upload to WandB if provided
    if wandb_logger:
        try:
            wandb_logger._safe_log({
                'original_bonnet_latent_space': wandb.Image(save_path)
            }, step_hint=epoch+1 if epoch else None)
            print(f"[ OK ] Uploaded original Bonnet latent space plot to WandB")
        except Exception as e:
            print(f"[ WARNING ] Could not upload original Bonnet latent space plot to wandb: {e}")
    
    return save_path