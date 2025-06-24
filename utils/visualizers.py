from utils.data_preparation import transform_grid_to_sequence

# Import extract_grid_from_sequence from evaluate_trajectory
def extract_grid_from_sequence(sequence, max_rows=30, max_cols=30):
    """Extract grid from sequence using the same logic as evaluate_trajectory.py"""
    try:
        if isinstance(sequence, (list, tuple)):
            sequence = np.array(sequence)
        
        # Shape information is stored at indices 900 and 901
        rows = int(sequence[900]) if len(sequence) > 900 else max_rows
        cols = int(sequence[901]) if len(sequence) > 901 else max_cols
        
        # Grid data is at the beginning of the sequence
        grid_data = sequence[:900]
        grid = grid_data.reshape(30, 30)
        
        # Extract the relevant portion
        actual_grid = grid[:rows, :cols] if rows > 0 and cols > 0 else np.zeros((1, 1))
        
        return actual_grid, (rows, cols)
    except Exception as e:
        print(f"Error extracting grid from sequence: {e}")
        return np.zeros((max_rows, max_cols)), (max_rows, max_cols)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from tabulate import tabulate
import os
import torch
from sklearn.manifold import TSNE
import pickle
from utils.settings_manager import settings
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from utils.model_utils import load_model

# Get settings from settings manager
evaluation_settings = settings.get_evaluation_settings()
DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']

##############################
# MULTI-ENCODER SUPPORT FUNCTIONS
##############################

def get_epoch_accuracies_for_plotting(results):
    """
    Extract epoch accuracy data for plotting, handling both single and multi-encoder formats.
    
    Args:
        results: Training results dictionary
        
    Returns:
        list: List of epoch accuracy dictionaries with standardized format
    """
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
                    'individual_encoder_data': individual_encoders  # Store individual data for detailed plots
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

def extract_multi_encoder_metrics(results):
    """
    Extract and process multi-encoder training metrics for visualization.
    Treats single-encoder (num_encoders=1) as multi-encoder for unified processing.
    Converts both single and multi-encoder metrics to a unified structure.
    
    Args:
        results: Training results dictionary
        
    Returns:
        dict: Processed metrics for unified visualization
    """
    if 'epoch_metrics' not in results or not results['epoch_metrics']:
        # Check if this might be a single-encoder model with standard structure
        if 'model_summary' in results:
            num_encoders = results.get('model_summary', {}).get('NUM_ENCODERS', 1)
            processed_results = results.copy()
            processed_results['is_multi_encoder'] = True  # Unified processing
            processed_results['num_encoders'] = num_encoders
            processed_results['is_actually_single_encoder'] = (num_encoders == 1)
            return processed_results
        return results
    
    # Check if this is multi-encoder training
    first_epoch_metrics = results['epoch_metrics'][0]
    has_multi_encoder_metrics = 'multi_encoder_metrics' in first_epoch_metrics
    
    # Also check model summary for number of encoders
    num_encoders = results.get('model_summary', {}).get('NUM_ENCODERS', 1)
    
    # Unified processing: treat both single and multi-encoder as "multi-encoder"
    if not has_multi_encoder_metrics and num_encoders == 1:
        # Convert single-encoder to multi-encoder format
        print("Converting single-encoder results to unified multi-encoder format...")
        processed_results = results.copy()
        processed_results['is_multi_encoder'] = True
        processed_results['num_encoders'] = 1
        processed_results['is_actually_single_encoder'] = True
        
        # Convert epoch metrics to multi-encoder format
        converted_metrics = []
        for epoch_data in results['epoch_metrics']:
            # Create single encoder entry
            multi_encoder_data = [{
                'encoder_idx': 0,
                'avg_total_loss': epoch_data.get('avg_total_loss', 0.0),
                'avg_shape_loss': epoch_data.get('avg_shape_loss', 0.0),
                'avg_grid_loss': epoch_data.get('avg_grid_loss', 0.0),
                'avg_kl_loss': epoch_data.get('avg_kl_loss', 0.0)
            }]
            
            converted_epoch = epoch_data.copy()
            converted_epoch['multi_encoder_metrics'] = multi_encoder_data
            converted_metrics.append(converted_epoch)
        
        processed_results['epoch_metrics'] = converted_metrics
        return processed_results
    
    elif not has_multi_encoder_metrics:
        # No multi-encoder metrics and num_encoders > 1 (shouldn't happen, but handle gracefully)
        processed_results = results.copy()
        processed_results['is_multi_encoder'] = True
        processed_results['num_encoders'] = num_encoders
        processed_results['is_actually_single_encoder'] = False
        return processed_results
    
    print("Processing multi-encoder metrics for visualization...")
    
    # Create processed results with additional multi-encoder specific data
    processed_results = results.copy()
    
    # Extract per-encoder metrics across epochs
    num_epochs = len(results['epoch_metrics'])
    encoder_indices = set()
    
    # Find all encoder indices
    for epoch_data in results['epoch_metrics']:
        if 'multi_encoder_metrics' in epoch_data:
            for encoder_data in epoch_data['multi_encoder_metrics']:
                encoder_indices.add(encoder_data['encoder_idx'])
    
    num_encoders = len(encoder_indices)
    print(f"Found {num_encoders} encoders across {num_epochs} epochs")
    
    # Initialize per-encoder metrics storage
    encoder_metrics = {
        'per_encoder_losses': {i: [] for i in encoder_indices},
        'per_encoder_shape_losses': {i: [] for i in encoder_indices},
        'per_encoder_grid_losses': {i: [] for i in encoder_indices},
        'per_encoder_kl_losses': {i: [] for i in encoder_indices},
        'epochs': list(range(1, num_epochs + 1))
    }
    
    # Extract metrics for each encoder across epochs
    for epoch_idx, epoch_data in enumerate(results['epoch_metrics']):
        epoch_num = epoch_idx + 1
        
        if 'multi_encoder_metrics' in epoch_data:
            # Create a mapping for this epoch
            epoch_encoder_data = {enc_data['encoder_idx']: enc_data 
                                for enc_data in epoch_data['multi_encoder_metrics']}
            
            # Fill in metrics for each encoder
            for encoder_idx in encoder_indices:
                if encoder_idx in epoch_encoder_data:
                    enc_data = epoch_encoder_data[encoder_idx]
                    encoder_metrics['per_encoder_losses'][encoder_idx].append(enc_data['avg_total_loss'])
                    encoder_metrics['per_encoder_shape_losses'][encoder_idx].append(enc_data['avg_shape_loss'])
                    encoder_metrics['per_encoder_grid_losses'][encoder_idx].append(enc_data['avg_grid_loss'])
                    encoder_metrics['per_encoder_kl_losses'][encoder_idx].append(enc_data['avg_kl_loss'])
                else:
                    # Encoder didn't train in this epoch (shouldn't happen, but handle gracefully)
                    encoder_metrics['per_encoder_losses'][encoder_idx].append(0.0)
                    encoder_metrics['per_encoder_shape_losses'][encoder_idx].append(0.0)
                    encoder_metrics['per_encoder_grid_losses'][encoder_idx].append(0.0)
                    encoder_metrics['per_encoder_kl_losses'][encoder_idx].append(0.0)
    
    # Add multi-encoder specific data
    processed_results['multi_encoder_data'] = encoder_metrics
    processed_results['is_multi_encoder'] = True
    processed_results['num_encoders'] = num_encoders
    
    # Create aggregated single-encoder compatible metrics for existing visualizers
    processed_results['aggregated_epoch_metrics'] = []
    for epoch_idx, epoch_data in enumerate(results['epoch_metrics']):
        if 'multi_encoder_metrics' in epoch_data:
            # Aggregate metrics across encoders
            total_losses = [enc_data['avg_total_loss'] for enc_data in epoch_data['multi_encoder_metrics']]
            shape_losses = [enc_data['avg_shape_loss'] for enc_data in epoch_data['multi_encoder_metrics']]
            grid_losses = [enc_data['avg_grid_loss'] for enc_data in epoch_data['multi_encoder_metrics']]
            kl_losses = [enc_data['avg_kl_loss'] for enc_data in epoch_data['multi_encoder_metrics']]
            
            aggregated_metrics = {
                'epoch': epoch_idx + 1,
                'avg_total_loss': sum(total_losses) / len(total_losses),
                'avg_shape_loss': sum(shape_losses) / len(shape_losses),
                'avg_grid_loss': sum(grid_losses) / len(grid_losses),
                'avg_kl_loss': sum(kl_losses) / len(kl_losses),
                'learning_rate': epoch_data.get('learning_rate', 0.0)
            }
            processed_results['aggregated_epoch_metrics'].append(aggregated_metrics)
    
    print(f"✓ Multi-encoder metrics processed: {num_encoders} encoders, {num_epochs} epochs")
    return processed_results

def plot_multi_encoder_training_losses(results, save_dir=None):
    """
    Plot training losses for each encoder in a multi-encoder setup.
    
    Args:
        results: Processed training results with multi-encoder data
        save_dir: Directory to save plots (optional)
    """
    if not results.get('is_multi_encoder', False):
        print("Not a multi-encoder model, skipping encoder loss plots")
        return
    
    if 'multi_encoder_data' not in results:
        print("No multi-encoder data found for loss plotting")
        return
    
    encoder_data = results['multi_encoder_data']
    epochs = encoder_data['epochs']
    num_encoders = results['num_encoders']
    
    # Create subplots for different loss types
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different encoders
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    # 1. Total losses per encoder
    ax = axs[0, 0]
    for encoder_idx in range(num_encoders):
        losses = encoder_data['per_encoder_losses'][encoder_idx]
        ax.plot(epochs, losses, marker='o', label=f'Encoder {encoder_idx}', 
               color=colors[encoder_idx], linewidth=2, alpha=0.8)
    
    ax.set_title('Total Loss per Encoder', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Shape losses per encoder
    ax = axs[0, 1]
    for encoder_idx in range(num_encoders):
        losses = encoder_data['per_encoder_shape_losses'][encoder_idx]
        ax.plot(epochs, losses, marker='s', label=f'Encoder {encoder_idx}', 
               color=colors[encoder_idx], linewidth=2, alpha=0.8)
    
    ax.set_title('Shape Loss per Encoder', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shape Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Grid losses per encoder
    ax = axs[1, 0]
    for encoder_idx in range(num_encoders):
        losses = encoder_data['per_encoder_grid_losses'][encoder_idx]
        ax.plot(epochs, losses, marker='^', label=f'Encoder {encoder_idx}', 
               color=colors[encoder_idx], linewidth=2, alpha=0.8)
    
    ax.set_title('Grid Loss per Encoder', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Grid Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. KL losses per encoder
    ax = axs[1, 1]
    for encoder_idx in range(num_encoders):
        losses = encoder_data['per_encoder_kl_losses'][encoder_idx]
        ax.plot(epochs, losses, marker='d', label=f'Encoder {encoder_idx}', 
               color=colors[encoder_idx], linewidth=2, alpha=0.8)
    
    ax.set_title('KL Loss per Encoder', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Encoder Training Losses ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multi_encoder_training_losses.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder training losses plot saved to {save_dir}/multi_encoder_training_losses.png")
    else:
        plt.show()

##############################
# LOAD LATENT REPRESENTATIONS
##############################

def load_evaluation_latent_data(run_dir, return_all_components=False):
    """
    Load latent representations generated during evaluation.
    These are created by encoding training sequences with the trained model.
    
    Args:
        run_dir: Directory containing evaluation results
        return_all_components: If True, return dict with all components, else just latent_mus
        
    Returns:
        np.ndarray or dict or None: Latent representations, dict with all components, or None if not found
    """
    # First try to load from the dedicated encoded training latents file
    encoded_file = os.path.join(run_dir, 'encoded_training_latents.pkl')
    if os.path.exists(encoded_file):
        try:
            print(f"Loading encoded training latents from {encoded_file}...")
            with open(encoded_file, 'rb') as f:
                encoded_data = pickle.load(f)
            
            if return_all_components:
                all_components = {
                    'latent_mus': encoded_data['latent_mus'],
                    'latent_log_vars': encoded_data['latent_log_vars'], 
                    'latent_zs': encoded_data['latent_zs'],
                    'encoding_info': encoded_data.get('encoding_info', {})
                }
                
                print(f"✓ Successfully loaded all latent components:")
                print(f"  - Means (μ): {len(all_components['latent_mus'])} samples")
                print(f"  - Log-vars (log σ²): {len(all_components['latent_log_vars'])} samples")
                print(f"  - Sampled Z: {len(all_components['latent_zs'])} samples")
                print(f"  - Encoding device: {all_components['encoding_info'].get('device', 'Unknown')}")
                
                return all_components
            else:
                latent_mus = encoded_data['latent_mus']
                encoding_info = encoded_data.get('encoding_info', {})
                
                print(f"✓ Successfully loaded {len(latent_mus)} encoded training latent vectors")
                print(f"  - Total training samples: {encoding_info.get('total_training_samples', 'Unknown')}")
                print(f"  - Encoded samples: {encoding_info.get('encoded_samples', len(latent_mus))}")
                print(f"  - Encoding device: {encoding_info.get('device', 'Unknown')}")
                
                return latent_mus
            
        except Exception as e:
            print(f"⚠ Error loading encoded training latents: {e}")
    
    # Fallback: try to load from evaluation results
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if os.path.exists(eval_file):
        try:
            print(f"Trying to load training latents from evaluation results...")
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
            
            # Look for encoded training latents in any key's results
            for key, results in eval_results.items():
                if isinstance(results, dict) and 'encoded_training_latents' in results:
                    embedded_data = results['encoded_training_latents']
                    
                    # Handle both numpy arrays and lists
                    if return_all_components:
                        # Convert to numpy arrays if they're lists
                        latent_mus = embedded_data['latent_mus']
                        if isinstance(latent_mus, list):
                            latent_mus = np.array(latent_mus)
                        
                        latent_log_vars = embedded_data.get('latent_log_vars', [])
                        if isinstance(latent_log_vars, list):
                            latent_log_vars = np.array(latent_log_vars)
                        
                        latent_zs = embedded_data.get('latent_zs', [])
                        if isinstance(latent_zs, list):
                            latent_zs = np.array(latent_zs)
                        
                        all_components = {
                            'latent_mus': latent_mus,
                            'latent_log_vars': latent_log_vars,
                            'latent_zs': latent_zs,
                            'encoding_info': embedded_data.get('encoding_info', {})
                        }
                        
                        print(f"✓ Found all latent components in evaluation results for key '{key}'")
                        print(f"  - Means: {len(all_components['latent_mus'])} samples")
                        print(f"  - Log-vars: {len(all_components['latent_log_vars'])} samples")
                        print(f"  - Sampled Z: {len(all_components['latent_zs'])} samples")
                        
                        return all_components
                    else:
                        latent_mus = embedded_data['latent_mus']
                        if isinstance(latent_mus, list):
                            latent_mus = np.array(latent_mus)
                        
                        encoding_info = embedded_data.get('encoding_info', {})
                        
                        print(f"✓ Found {len(latent_mus)} training latent vectors in evaluation results for key '{key}'")
                        print(f"  - Total training samples: {encoding_info.get('total_training_samples', 'Unknown')}")
                        print(f"  - Encoded samples: {encoding_info.get('encoded_samples', len(latent_mus))}")
                        
                        return latent_mus
            
            print("⚠ No encoded training latents found in evaluation results")
            
        except Exception as e:
            print(f"⚠ Error loading evaluation results: {e}")
    
    print("⚠ Warning: No encoded training latent data found.")
    print("  This suggests evaluation hasn't been run with the updated code.")
    print("  Please run evaluation to generate encoded training latents.")
    return None

def load_legacy_latent_data(run_dir):
    """
    Legacy function to load latent data from training results.pkl.
    This is kept for backward compatibility but will show a deprecation warning.
    
    Args:
        run_dir: Directory containing results.pkl
        
    Returns:
        np.ndarray or None: Latent representations or None if not found
    """
    results_file = os.path.join(run_dir, 'results.pkl')
    if not os.path.exists(results_file):
        return None
    
    try:
        print("⚠ DEPRECATION WARNING: Loading latents from legacy training results.pkl")
        print("  Using legacy training data for visualization.")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # Process latent means using the original approach
        def process_latent_var(latent_var_list, var_name):
            try:
                if not latent_var_list:
                    return None
                    
                # If we have a dictionary structure (multiple keys), extract and combine data
                if isinstance(latent_var_list, dict):
                    combined_data = []
                    for key, value in latent_var_list.items():
                        if value:  # If not empty
                            combined_data.extend(value)
                    if not combined_data:
                        return None
                    latent_var_list = combined_data
                
                # Process based on data type
                if isinstance(latent_var_list[0], torch.Tensor):
                    processed_var = torch.cat(latent_var_list, dim=0).numpy()
                elif isinstance(latent_var_list[0], np.ndarray):
                    processed_var = np.concatenate(latent_var_list, axis=0)
                elif isinstance(latent_var_list[0], list):
                    processed_var = np.array(latent_var_list)
                    if processed_var.size == 0 or (len(processed_var.shape) > 1 and processed_var.shape[1] == 0):
                        return None
                    if len(processed_var.shape) > 2:
                        processed_var = processed_var.reshape(-1, processed_var.shape[-1])
                else:
                    return None
                    
                if processed_var.size == 0 or (len(processed_var.shape) > 1 and processed_var.shape[1] == 0):
                    return None
                    
                return processed_var
            except (IndexError, TypeError, ValueError):
                return None
        
        # Process latent means
        all_mus = process_latent_var(results.get('latent_mus', []), 'latent_mus')
        
        if all_mus is None or len(all_mus) < 2:
            return None
        
        print(f"✓ Loaded {len(all_mus)} legacy training latent vectors")
        return all_mus
        
    except Exception as e:
        print(f"⚠ Error loading legacy training results: {e}")
        return None

def load_training_latent_data(run_dir):
    """
    Load training latent data for background visualization in trajectory plots.
    
    Args:
        run_dir: Directory containing training results
        
    Returns:
        numpy.ndarray: Flattened latent vectors from training data, or None if not available
    """
    try:
        # Try to load from evaluation results first (encoded training latents)
        eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
        if os.path.exists(eval_file):
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
            
            # Look for encoded training latents in any key
            for key, results in eval_results.items():
                if 'encoded_training_latents' in results:
                    training_latents = results['encoded_training_latents']
                    if 'latent_zs' in training_latents:
                        latent_zs = training_latents['latent_zs']
                        if isinstance(latent_zs, np.ndarray) and len(latent_zs) > 0:
                            # Flatten to 2D array (samples x latent_dim)
                            if latent_zs.ndim > 2:
                                latent_zs = latent_zs.reshape(latent_zs.shape[0], -1)
                            print(f"✓ Loaded {latent_zs.shape[0]} training latent vectors for background visualization")
                            return latent_zs
        
        # Fallback: Try to load from training results
        results_file = os.path.join(run_dir, 'results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            # Try different possible locations for latent data
            latent_sources = [
                ('latent_zs', 'latent_zs'),
                ('encoder_latent_data', 'encoder_latent_data'),
                ('single_encoder_latent_data', 'single_encoder_latent_data')
            ]
            
            for source_key, desc in latent_sources:
                if source_key in results:
                    latent_data = results[source_key]
                    
                    if source_key == 'latent_zs' and isinstance(latent_data, list) and len(latent_data) > 0:
                        # Convert list to numpy array
                        try:
                            latent_array = np.array(latent_data)
                            if latent_array.ndim > 2:
                                latent_array = latent_array.reshape(latent_array.shape[0], -1)
                            print(f"✓ Loaded {latent_array.shape[0]} training latent vectors from {desc}")
                            return latent_array
                        except:
                            continue
                    
                    elif source_key in ['encoder_latent_data', 'single_encoder_latent_data']:
                        # Extract from structured latent data
                        if isinstance(latent_data, dict):
                            for enc_key, enc_data in latent_data.items():
                                if isinstance(enc_data, dict) and 'latent_zs' in enc_data:
                                    latent_zs = enc_data['latent_zs']
                                    if isinstance(latent_zs, np.ndarray) and len(latent_zs) > 0:
                                        if latent_zs.ndim > 2:
                                            latent_zs = latent_zs.reshape(latent_zs.shape[0], -1)
                                        print(f"✓ Loaded {latent_zs.shape[0]} training latent vectors from {desc}")
                                        return latent_zs
        
        print("⚠ Warning: No training latent data found for background visualization")
        return None
        
    except Exception as e:
        print(f"⚠ Warning: Error loading training latent data: {e}")
        return None

##############################
# VISUALIZERS FOR DATA
##############################

def visualize_full_transformation(input_grid, output_grid,full_seq):
    # Get sequences
    input_seq = transform_grid_to_sequence(np.array(input_grid))
    output_seq = transform_grid_to_sequence(np.array(output_grid))

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Define colormap and normalization to match plot_task
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}

    # Plot original input grid
    plt.subplot(4, 2, 1)
    plt.imshow(input_grid, **args)
    plt.title('Original Input Grid', fontsize=12)
    plt.axis('off')

    # Plot original output grid
    plt.subplot(4, 2, 2)
    plt.imshow(output_grid, **args)
    plt.title('Original Output Grid', fontsize=12)
    plt.axis('off')

    # Plot input sequence
    plt.subplot(4, 2, 3)
    plt.plot(input_seq, '-b')
    plt.title('Input Sequence (shape_info + flattened grid)', fontsize=12)
    plt.axvline(x=2, color='r', linestyle='--', label='Shape info end')
    plt.legend()
    plt.grid(True)

    # Plot output sequence
    plt.subplot(4, 2, 4)
    plt.plot(output_seq, '-b')
    plt.title('Output Sequence (shape_info + flattened grid)', fontsize=12)
    plt.axvline(x=2, color='r', linestyle='--', label='Shape info end')
    plt.legend()
    plt.grid(True)

    # Plot padded input grid (30x30)
    padded_input = np.zeros((30, 30))
    rows, cols = input_grid.shape
    padded_input[:rows, :cols] = input_grid
    plt.subplot(4, 2, 5)
    plt.imshow(padded_input, **args)
    plt.title('Padded Input Grid (30x30)', fontsize=12)
    plt.axis('off')

    # Plot padded output grid (30x30)
    padded_output = np.zeros((30, 30))
    rows, cols = output_grid.shape
    padded_output[:rows, :cols] = output_grid
    plt.subplot(4, 2, 6)
    plt.imshow(padded_output, **args)
    plt.title('Padded Output Grid (30x30)', fontsize=12)
    plt.axis('off')

    # Plot full sequence
    plt.subplot(4, 1, 4)
    plt.plot(full_seq, '-b', alpha=0.6)
    plt.title('Full Combined Sequence (Input + Output + CLS token)', fontsize=12)
    plt.axvline(x=902, color='r', linestyle='--', label='Input End')
    plt.axvline(x=1804, color='g', linestyle='--', label='Output End')
    plt.annotate('Input Sequence', xy=(450, plt.ylim()[1]), ha='center')
    plt.annotate('Output Sequence', xy=(1350, plt.ylim()[1]), ha='center')
    plt.annotate('CLS', xy=(1804, plt.ylim()[1]), ha='left')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print sequence information
    print("\nSequence Information:")
    print(f"Input sequence length: {len(input_seq)}")
    print(f"Output sequence length: {len(output_seq)}")
    print(f"Full sequence length: {len(full_seq)}")
    print("\nSequence breakdown:")
    print(f"- First 2 values (input shape): {full_seq[0:2]}")
    print(f"- Start of input grid values: {full_seq[2:7]}...")
    print(f"- Input/Output boundary values: {full_seq[900:904]}...")
    print(f"- Final values including CLS token: {full_seq[-5:]}")


def print_sequence_info(input_grids, output_grids, sequences):
    table_data = []
    for i in range(len(input_grids)):
        table_data.append([
            f"Sequence {i+1}",
            input_grids[i].shape,
            output_grids[i].shape,
            sequences[i].shape,
            sequences[i][:10]
        ])

    headers = ["Sequence", "Input Grid Shape", "Output Grid Shape", "Transformed Sequence Length", "First Few Values"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("Number of sequences generated:", len(sequences))


##############################
# VISUALIZERS FOR MODELS
##############################

def visualize_sequence_reconstruction(original, reconstructed, epoch, batch_idx, run_dir):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(original[0].cpu().numpy())
    plt.title('Original Sequence')
    plt.subplot(1, 2, 2)
    plt.plot(reconstructed[0].detach().cpu().numpy())
    plt.title('Reconstructed Sequence')
    plt.savefig(os.path.join(run_dir, f'reconstruction_epoch{epoch}_batch{batch_idx}.png'))
    plt.close()

def plot_training_and_latent_DEPRECATED(results, save_dir=None):
    """DEPRECATED: This function has been removed. Use plot_comprehensive_latent_space instead."""
    print("⚠ Warning: plot_training_and_latent is deprecated. Use plot_comprehensive_latent_space instead.")
    return

def plot_latent_analysis_DEPRECATED(results, save_dir=None):
    """DEPRECATED: This function has been removed. Use plot_comprehensive_latent_space instead.""" 
    print("⚠ Warning: plot_latent_analysis is deprecated. Use plot_comprehensive_latent_space instead.")
    return

def plot_epoch_accuracies(results, save_dir=None):
    """
    Plot training accuracy over epochs.
    """
    # Use the helper function to get the right accuracy data for plotting
    epoch_accuracies = get_epoch_accuracies_for_plotting(results)
    
    if not epoch_accuracies:
        print("No epoch accuracy data found for plotting")
        return
    
    epochs = [ep_data['epoch'] for ep_data in epoch_accuracies]
    shape_accuracies = [ep_data['shape_accuracy'] for ep_data in epoch_accuracies]
    grid_accuracies = [ep_data['grid_accuracy'] for ep_data in epoch_accuracies]
    overall_accuracies = [ep_data['overall_accuracy'] for ep_data in epoch_accuracies]
    sample_exact_accuracies = [ep_data['sample_exact_accuracy'] for ep_data in epoch_accuracies]
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, shape_accuracies, 'o-', label='Shape Accuracy', alpha=0.8)
    plt.plot(epochs, grid_accuracies, 's-', label='Grid Accuracy', alpha=0.8)
    plt.plot(epochs, overall_accuracies, '^-', label='Overall Accuracy', alpha=0.8)
    plt.plot(epochs, sample_exact_accuracies, 'd-', label='Pixel Accuracy (Exact)', alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'epoch_accuracies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Epoch accuracies plot saved to {save_dir}/epoch_accuracies.png")
    else:
        plt.show()

def plot_z_optimization_losses(results, save_dir=None):
    """Plot z optimization losses over steps."""
    losses = results['losses_gradient_ascent']
    
    # Flatten the nested list structure
    flattened_losses = []
    for loss_sequence in losses:
        if isinstance(loss_sequence, list):
            flattened_losses.extend(loss_sequence)
        else:
            flattened_losses.append(loss_sequence)
    
    plt.figure(figsize=(10, 6))
    plt.plot(flattened_losses, marker='o', linestyle='-', markersize=2)        
    plt.title("Z Optimization Losses Over Steps", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'z_optimization_losses.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Z optimization losses plot saved to {save_dir}/z_optimization_losses.png")
    else:
        plt.show()


def plot_reconstructions(results, save_dir=None, filename_prefix=""):
    """Plot input, target, reconstruction, and error map for each sample."""
    reconstructions = results['reconstructions']
    
    input_seqs = [item['input'] for item in reconstructions]
    output_seqs = [item['target'] for item in reconstructions]

    shape_logits_list = []
    grid_logits_list = []

    for item in reconstructions:
        shape_logits, grid_logits = item['reconstruction']

        shape_array = np.array(shape_logits)
        grid_array = np.array(grid_logits)

        pred_shapes = np.argmax(shape_array, axis=-1)
        pred_grid = np.argmax(grid_array, axis=-1)

        shape_logits_list.append(torch.tensor(pred_shapes))
        grid_logits_list.append(torch.tensor(pred_grid))

    shape_recons = torch.stack(shape_logits_list).numpy()
    grid_recons = torch.stack(grid_logits_list).numpy()

    num_samples = min(DEFAULT_VISUALIZE_N_VALUES, len(shape_recons))
    for i in range(num_samples):
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        # Input
        input_seq = input_seqs[i]
        in_rows, in_cols = int(input_seq[-2]), int(input_seq[-1])
        input_grid = np.array(input_seq[:900]).reshape(30, 30)[:in_rows, :in_cols]
        axs[0].imshow(input_grid, cmap='viridis')
        axs[0].set_title('Input')
        axs[0].axis('off')

        # Target
        target_seq = output_seqs[i]
        out_rows, out_cols = int(target_seq[-2]), int(target_seq[-1])
        target_grid = np.array(target_seq[:900]).reshape(30, 30)[:out_rows, :out_cols]
        axs[1].imshow(target_grid, cmap='viridis')
        axs[1].set_title('Target')
        axs[1].axis('off')

        # Reconstruction
        recon_rows, recon_cols = int(shape_recons[i][0]), int(shape_recons[i][1])
        recon_grid = grid_recons[i].reshape(30, 30)[:recon_rows, :recon_cols]
        axs[2].imshow(recon_grid, cmap='viridis')
        axs[2].set_title(f'Reconstruction ({recon_rows}x{recon_cols})')
        axs[2].axis('off')

        # Error Map
        common_rows = min(out_rows, recon_rows)
        common_cols = min(out_cols, recon_cols)
        error_map = np.abs(target_grid[:common_rows, :common_cols] - recon_grid[:common_rows, :common_cols])
        axs[3].imshow(error_map, cmap='hot')
        axs[3].set_title('Error Map')
        axs[3].axis('off')

        plt.suptitle(f'Sample {i+1}', fontsize=18)
        plt.tight_layout()
        if save_dir:
            filename = f'{filename_prefix}reconstructions_sample_{i+1}.png' if filename_prefix else f'reconstructions_sample_{i+1}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Reconstruction plot for sample {i+1} saved to {save_dir}/{filename}")
        else:
            plt.show()


def visualize_all_results(results, save_dir=None, eval_results=None):
    """Plot all visualizations for the results. Uses unified multi-encoder processing."""
    
    # Process with unified multi-encoder approach (handles both single and multi-encoder)
    processed_results = extract_multi_encoder_metrics(results)
    
    # Always use multi-encoder approach for unified processing
    is_multi_encoder = processed_results.get('is_multi_encoder', False)
    num_encoders = processed_results.get('num_encoders', 1)
    is_actually_single = processed_results.get('is_actually_single_encoder', False)
    
    if is_multi_encoder:
        encoder_type = f"single encoder (unified)" if is_actually_single else f"multi-encoder ({num_encoders} encoders)"
        print(f"\nUnified multi-encoder processing: {encoder_type}")
        
        # Use multi-encoder visualizations for both single and multi-encoder models
        print("\nPlotting multi-encoder accuracies...")
        plot_multi_encoder_accuracies(processed_results, save_dir)
        
        print("\nPlotting multi-encoder accuracy summary...")
        plot_multi_encoder_accuracy_summary(processed_results, save_dir)
        
        print("\nPlotting multi-encoder training losses...")
        plot_multi_encoder_training_losses(processed_results, save_dir)
    
    print("\nPlotting comprehensive latent space...")
    plot_comprehensive_latent_space(processed_results, eval_results=eval_results, save_dir=save_dir)

    print("\nPlotting epoch accuracies over time...")
    plot_epoch_accuracies(processed_results, save_dir)

    if 'losses_gradient_ascent' in processed_results:
        if len(processed_results['losses_gradient_ascent']) > 0:
            print("\nPlotting z optimization losses...")
            plot_z_optimization_losses(processed_results, save_dir)

    # print("\nPlotting reconstructions...")
    # plot_reconstructions(processed_results, save_dir)


def plot_evaluation_results(eval_results, save_dir=None):
    """
    Plot evaluation metrics and reconstructions for each key.
    Uses unified multi-encoder processing for both single and multi-encoder models.
    
    Args:
        eval_results: Dictionary containing evaluation results for each key
        save_dir: Directory to save plots (optional)
    """
    if not eval_results:
        print("No evaluation results to plot")
        return

    print("Processing evaluation results...")
    
    # Determine evaluation type and number of encoders
    sample_key_results = next(iter(eval_results.values()))
    metrics = sample_key_results.get('metrics', {})
    is_multi_encoder = metrics.get('is_multi_encoder', False)
    num_encoders = metrics.get('num_encoders', 1)
    
    # Use unified processing: treat single-encoder as multi-encoder with 1 encoder
    unified_multi_encoder = is_multi_encoder or (num_encoders >= 1)
    
    if unified_multi_encoder:
        encoder_type = f"single encoder (unified)" if num_encoders == 1 else f"multi-encoder ({num_encoders} encoders)"
        print(f"Unified multi-encoder evaluation processing: {encoder_type}")
        
        # Use multi-encoder processing for all cases
        for key, key_results in eval_results.items():
            plot_multi_encoder_evaluation_results(key, key_results, save_dir)
    else:
        print("Fallback to single-encoder evaluation processing")
        # Fallback for legacy single encoder results without num_encoders info
        for key, key_results in eval_results.items():
            plot_single_encoder_evaluation_results(key, key_results, save_dir)
    
    # Plot overall comparison across keys
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Define metrics to plot
    if is_multi_encoder:
        # Use PoE metrics for multi-encoder comparison
        metrics = [
            'support_loss', 'query_loss',  # Loss metrics
            'shape_accuracy', 'grid_accuracy',  # Accuracy metrics (these are PoE metrics)
            'overall_accuracy', 'sample_exact_accuracy'  # Additional metrics
        ]
    else:
        # Use standard metrics for single encoder
        metrics = [
            'support_loss', 'query_loss',  # Loss metrics
            'shape_accuracy', 'grid_accuracy',  # Accuracy metrics
            'overall_accuracy', 'sample_exact_accuracy'  # Additional metrics
        ]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = []
        keys = []
        for key in eval_results:
            if 'metrics' in eval_results[key] and metric in eval_results[key]['metrics']:
                values.append(eval_results[key]['metrics'][metric])
                keys.append(key)
        
        if values:  # Only plot if we have data
            axs[i].bar(keys, values)
            axs[i].set_title(f'{metric.replace("_", " ").title()}')
            axs[i].set_xticks(range(len(keys)))
            axs[i].set_xticklabels(keys, rotation=45)
            
            # Set y-axis limits for accuracy metrics
            if 'accuracy' in metric:
                axs[i].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_dir:
        filename = 'multi_encoder_evaluation_overview.png' if is_multi_encoder else 'evaluation_metrics.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Evaluation overview plot saved to {save_dir}/{filename}")
    else:
        plt.show()
    
    # Print detailed metrics for each key
    print(f"\nDetailed Evaluation Results ({'Multi-Encoder' if is_multi_encoder else 'Single Encoder'}):")
    for key, key_results in eval_results.items():
        metrics = key_results.get('metrics', {})
        print(f"\n{key}:")
        if is_multi_encoder:
            # Print PoE results (primary)
            print(f"  PoE Results:")
            print(f"    Support loss: {metrics.get('support_loss', 'N/A'):.4f}")
            print(f"    Query loss: {metrics.get('query_loss', 'N/A'):.4f}")
            print(f"    Shape accuracy: {metrics.get('shape_accuracy', 'N/A'):.4f}")
            print(f"    Grid accuracy: {metrics.get('grid_accuracy', 'N/A'):.4f}")
            print(f"    Sample exact accuracy: {metrics.get('sample_exact_accuracy', 'N/A'):.4f}")
            
            # Print individual encoder summary
            individual_accuracies = metrics.get('individual_encoder_accuracies', {})
            if individual_accuracies:
                print(f"  Individual Encoder Results:")
                for enc_name, enc_metrics in individual_accuracies.items():
                    print(f"    {enc_name}: Shape={enc_metrics.get('shape_accuracy', 0):.3f}, "
                          f"Grid={enc_metrics.get('grid_accuracy', 0):.3f}, "
                          f"Exact={enc_metrics.get('sample_exact_accuracy', 0):.3f}")
        else:
            # Print single encoder results
            print(f"    Support loss: {metrics.get('support_loss', 'N/A'):.4f}")
            print(f"    Query loss: {metrics.get('query_loss', 'N/A'):.4f}")
            print(f"    Shape accuracy: {metrics.get('shape_accuracy', 'N/A'):.4f}")
            print(f"    Grid accuracy: {metrics.get('grid_accuracy', 'N/A'):.4f}")
            print(f"    Sample exact accuracy: {metrics.get('sample_exact_accuracy', 'N/A'):.4f}")

def visualize_stored_results(run_dir):
    """
    Load and visualize results from a previous run.
    Makes training results optional - will still work if only evaluation results exist.
    
    Args:
        run_dir: Directory containing the stored results
    """
    print(f"Looking for results in: {run_dir}")
    
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
        
        # Try to create default parameters from current settings
        try:
            from utils.settings_manager import settings
            
            data_settings = settings.get_data_settings()
            model_architecture = settings.get_model_architecture()
            training_settings = settings.get_training_settings()
            latent_optimization = settings.get_latent_optimization()
            evaluation_settings = settings.get_evaluation_settings()
            
            model_params = {
                'TRAINING_KEYS': data_settings.get('training_keys', []),
                'TRAINING_SEED': data_settings.get('training_seed', 42),
                'EVAL_SEED': data_settings.get('eval_seed', 1),
                'LATENT_DIM': model_architecture.get('latent_dim', 64),
                'ENCODER_HIDDEN_DIM': model_architecture.get('encoder_hidden_dim', model_architecture.get('hidden_dim', 128)),
                'DECODER_HIDDEN_DIM': model_architecture.get('decoder_hidden_dim', model_architecture.get('hidden_dim', 128)),
                'ENCODER_LAYERS': model_architecture.get('encoder_layers', model_architecture.get('num_layers', 2)),
                'DECODER_LAYERS': model_architecture.get('decoder_layers', model_architecture.get('num_layers', 2)),
                'ENCODER_HEADS': model_architecture.get('encoder_heads', model_architecture.get('num_heads', 4)),
                'DECODER_HEADS': model_architecture.get('decoder_heads', model_architecture.get('num_heads', 4)),
                'NUM_ENCODERS': model_architecture.get('num_encoders', 1),
                'DROPOUT': model_architecture.get('dropout', 0.1),
                'MAX_LENGTH': model_architecture.get('max_length', 902),
                'ENCODER_MAX_LENGTH': model_architecture.get('encoder_max_length', 1805),
                'DECODER_MAX_LENGTH': model_architecture.get('decoder_max_length', 902),
                'BATCH_SIZE': training_settings.get('batch_size', 16),
                'NUM_EPOCHS': training_settings.get('num_epochs', 10),
                'LEARNING_RATE': training_settings.get('learning_rate', 0.001),
                'BETA': training_settings.get('beta', 0.1),
                'n': data_settings.get('n', 100),
                'OPTIMIZE_Z': latent_optimization.get('training', {}).get('enabled', False),
                'OPTIMIZE_Z_NUM_STEPS': latent_optimization.get('training', {}).get('num_steps', 5),
                'OPTIMIZE_Z_LR': latent_optimization.get('training', {}).get('learning_rate', 0.1),
                'OPTIMIZE_Z_INFERENCE': latent_optimization.get('inference', {}).get('enabled', True),
                'OPTIMIZE_Z_INFERENCE_NUM_STEPS': latent_optimization.get('inference', {}).get('num_steps', 10),
                'OPTIMIZE_Z_INFERENCE_LR': latent_optimization.get('inference', {}).get('learning_rate', 0.1),
                'DEFAULT_EVAL_KEYS': evaluation_settings.get('eval_keys', []),
                'DEFAULT_EVAL_N_SAMPLES': evaluation_settings.get('eval_n_samples', 2),
                'DEFAULT_EVAL_N_QUERIES': evaluation_settings.get('eval_n_queries', 10),
                'DEFAULT_EVAL_EPOCH': evaluation_settings.get('eval_epoch', 1),
                'OPTIMIZATION_METHOD': latent_optimization.get('method', 'gradient')
            }
            print("✓ Created default model parameters from current settings")
        except Exception as e:
            print(f"⚠ Warning: Could not create default model parameters: {e}")
            model_params = {
                'TRAINING_KEYS': ['unknown'],
                'TRAINING_SEED': 42,
                'EVAL_SEED': 1,
                'LATENT_DIM': 64,
                'ENCODER_HIDDEN_DIM': 128,
                'DECODER_HIDDEN_DIM': 128,
                'ENCODER_LAYERS': 2,
                'DECODER_LAYERS': 2,
                'ENCODER_HEADS': 4,
                'DECODER_HEADS': 4,
                'NUM_ENCODERS': 1,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 902,
                'ENCODER_MAX_LENGTH': 1805,
                'DECODER_MAX_LENGTH': 902,
                'BATCH_SIZE': 16,
                'NUM_EPOCHS': 10,
                'LEARNING_RATE': 0.001,
                'BETA': 0.1,
                'n': 100,
                'OPTIMIZE_Z': False,
                'OPTIMIZE_Z_NUM_STEPS': 5,
                'OPTIMIZE_Z_LR': 0.1,
                'OPTIMIZE_Z_INFERENCE': True,
                'OPTIMIZE_Z_INFERENCE_NUM_STEPS': 10,
                'OPTIMIZE_Z_INFERENCE_LR': 0.1,
                'DEFAULT_EVAL_KEYS': ['unknown'],
                'DEFAULT_EVAL_N_SAMPLES': 2,
                'DEFAULT_EVAL_N_QUERIES': 10,
                'DEFAULT_EVAL_EPOCH': 1,
                'OPTIMIZATION_METHOD': 'gradient'
            }
            print("✓ Using minimal default model parameters")
    
    # Plot enhanced model summary if we have parameters (even if no training results)
    if model_params is not None:
        print("\nPlotting enhanced model summary...")
        try:
            plot_enhanced_model_summary(results, model_params, run_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not plot enhanced model summary: {e}")
            # Fallback to original model summary
            try:
                plot_model_summary(results, model_params, run_dir)
            except Exception as e2:
                print(f"⚠ Warning: Could not plot fallback model summary: {e2}")
    
    # Visualize training results if available
    if results is not None:
        print("\nVisualizing training results...")
        try:
            # Check if this is multi-encoder training and add appropriate info
            processed_results = extract_multi_encoder_metrics(results)
            is_multi_encoder = processed_results.get('is_multi_encoder', False)
            
            if is_multi_encoder:
                print(f"Detected multi-encoder training with {processed_results['num_encoders']} encoders")
            
            # First try to load eval results for comprehensive visualization
            eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
            eval_results_for_latent = None
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'rb') as f:
                        eval_results_for_latent = pickle.load(f)
                    print("✓ Loaded evaluation results for comprehensive latent visualization")
                except Exception as e:
                    print(f"⚠ Warning: Could not load evaluation results for latent visualization: {e}")
            
            visualize_all_results(results, run_dir, eval_results_for_latent)
        except Exception as e:
            print(f"⚠ Warning: Could not visualize training results: {e}")
    else:
        print("Skipping training results visualization (no training results available)")
    
    # Try to load and visualize evaluation results
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    eval_results = None
    if os.path.exists(eval_file):
        print("\nFound evaluation results file, loading...")
        try:
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
            print("✓ Evaluation results loaded successfully")
            
            print("Visualizing evaluation results...")
            plot_evaluation_results(eval_results, run_dir)
            
            # Add trajectory reconstructions (unified processing for both single and multi-encoder)
            sample_key_results = next(iter(eval_results.values()))
            metrics = sample_key_results.get('metrics', {})
            is_multi_encoder = metrics.get('is_multi_encoder', False)
            num_encoders = metrics.get('num_encoders', 1)
            
            # Use unified processing for trajectory reconstructions
            if is_multi_encoder or num_encoders >= 1:
                trajectory_type = f"single encoder (unified)" if num_encoders == 1 else f"multi-encoder ({num_encoders} encoders)"
                print(f"Creating trajectory reconstruction plots ({trajectory_type})...")
                try:
                    plot_multi_encoder_trajectory_reconstructions(eval_results, run_dir)
                except Exception as e:
                    print(f"⚠ Warning: Could not create trajectory plots: {e}")
                    print("✓ Trajectory reconstruction analysis placeholder - feature coming soon")
        except Exception as e:
            print(f"⚠ Warning: Could not load/visualize evaluation results: {e}")
    else:
        print("No evaluation results found (evaluation_results.pkl)")
    
    # Generate training data distribution analysis for multi-encoder models (Point 14)
    if model_params is not None and model_params.get('NUM_ENCODERS', 1) > 1:
        print("\nGenerating training data distribution analysis...")
        try:
            # TODO: Implement plot_training_data_distribution_analysis function
            print("✓ Training data distribution analysis placeholder - feature coming soon")
        except Exception as e:
            print(f"⚠ Warning: Could not generate training data distribution analysis: {e}")
    
    # Generate JSON experiment summary
    if model_params is not None:
        print("\nGenerating experiment summary...")
        try:
            create_simple_analysis_report(results, model_params, eval_results, run_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not generate experiment summary: {e}")
    

    
    # Summary of what was found and processed
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Run directory: {run_dir}")
    print(f"Training results: {'✓ Found and processed' if results is not None else '✗ Not found'}")
    print(f"Model parameters: {'✓ Found and processed' if model_params is not None else '✗ Not found'}")
    eval_found = os.path.exists(os.path.join(run_dir, 'evaluation_results.pkl'))
    print(f"Evaluation results: {'✓ Found and processed' if eval_found else '✗ Not found'}")
    
    if results is None and not eval_found:
        print("\n⚠ Warning: No results files found in the specified directory.")
        print("Make sure the directory contains either 'results.pkl' or 'evaluation_results.pkl'")
    elif results is None:
        print("\n✓ Evaluation-only visualization completed successfully")
    elif not eval_found:
        print("\n✓ Training-only visualization completed successfully")
    else:
        print("\n✓ Complete visualization (training + evaluation) completed successfully")

def plot_multi_encoder_trajectory_reconstructions(eval_results, save_dir=None):
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
    
    for key, key_results in eval_results.items():
        print(f"\n=== Processing trajectory reconstructions for key: {key} ===")
        
        # Look for trajectory information
        trajectory_info_list = []
        if 'metrics' in key_results and 'trajectory_info' in key_results['metrics']:
            trajectory_info_list = key_results['metrics']['trajectory_info']
        elif 'trajectory_info' in key_results:
            trajectory_info_list = key_results['trajectory_info']
        
        if not trajectory_info_list:
            print(f"⚠ No trajectory information found for key {key}")
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
                model, _, _, _ = load_model(save_dir, device=device)
                print(f"  ✓ Loaded model from {save_dir}")
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

def create_multi_encoder_trajectory_reconstruction(trajectory_info, model, save_path, device='cuda'):
    """
    Create a comprehensive multi-encoder trajectory reconstruction visualization.
    Shows individual encoder outputs (once) and PoE trajectory evolution (5 steps).
    
    Args:
        trajectory_info: Dictionary containing trajectory data
        model: The trained multi-encoder model
        save_path: Path to save the visualization
        device: Device for computation
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Extract data from trajectory
    z_vectors = trajectory_info.get('z_vectors', [])
    losses = trajectory_info.get('losses', [])
    input_sample = trajectory_info.get('input_sample')
    target_sample = trajectory_info.get('target_sample')
    individual_encoder_trajectories = trajectory_info.get('individual_encoder_trajectories', {})
    num_encoders = trajectory_info.get('num_encoders', 1)
    
    if len(z_vectors) < 2:
        print("⚠ Warning: Insufficient trajectory data for visualization")
        return
    
    # Prepare input/target tensors
    input_tensor = torch.tensor(input_sample, dtype=torch.float32).unsqueeze(0).to(device)
    target_tensor = torch.tensor(target_sample, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Extract target grid dimensions for proper reconstruction visualization
    try:
        target_rows = int(target_sample[900]) if len(target_sample) > 900 else 10
        target_cols = int(target_sample[901]) if len(target_sample) > 901 else 10
    except:
        target_rows, target_cols = 10, 10
    
    # Create figure: Top row for individual encoders + PoE steps, bottom row for grids and analysis
    fig = plt.figure(figsize=(20, 12))
    
    # Layout: 3 rows x 8 columns
    # Row 0: Individual encoder reconstructions + first PoE steps
    # Row 1: Input/Target/Loss plot + remaining PoE steps  
    # Row 2: Latent space trajectory analysis
    gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
    
    model.eval()
    
    # ============= TOP ROW: INDIVIDUAL ENCODERS + POE STEPS =============
    
    # Individual encoder reconstructions (columns 0-3, row 0)
    print(f"  Generating individual encoder reconstructions for {num_encoders} encoders...")
    for enc_idx in range(min(num_encoders, 4)):  # Limit to 4 encoders for layout
        ax = fig.add_subplot(gs[0, enc_idx])
        
        encoder_key = f'encoder_{enc_idx}'
        if encoder_key in individual_encoder_trajectories:
            try:
                with torch.no_grad():
                    # Get individual encoder z (these don't change during optimization)
                    enc_data = individual_encoder_trajectories[encoder_key]
                    enc_z = torch.tensor(enc_data['z'], dtype=torch.float32).to(device)
                    
                    # Generate reconstruction using individual encoder z
                    shape_logits, grid_logits = model.multi_encoder.decoder(
                        enc_z, input_tensor, target_seq=target_tensor
                    )
                    
                    # Extract and reshape predicted grid
                    grid_pred = grid_logits.argmax(dim=-1).cpu().numpy()[0]
                    if target_rows > 0 and target_cols > 0:
                        pred_grid = grid_pred[:target_rows*target_cols].reshape(target_rows, target_cols)
                        im = ax.imshow(pred_grid, cmap='tab20', vmin=0, vmax=9)
                        ax.set_title(f'Encoder {enc_idx}\nStatic Output', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, f'Enc {enc_idx}\nInvalid dims', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Encoder {enc_idx}', fontsize=10)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Enc {enc_idx}\nError:\n{str(e)[:20]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'Encoder {enc_idx}', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'Enc {enc_idx}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Encoder {enc_idx}', fontsize=10)
        
        ax.axis('off')
    
    # PoE trajectory reconstructions (columns 4-7, row 0)
    print(f"  Generating PoE trajectory reconstructions for {len(z_vectors)} steps...")
    if len(z_vectors) >= 5:
        # Select 4 evenly spaced steps from the trajectory
        indices = [0, len(z_vectors)//4, len(z_vectors)//2, len(z_vectors)-1]
        labels = ['PoE Start', 'PoE Step 1', 'PoE Step 2', 'PoE End']
    elif len(z_vectors) >= 3:
        indices = [0, len(z_vectors)//2, len(z_vectors)-1, len(z_vectors)-1]
        labels = ['PoE Start', 'PoE Mid', 'PoE End', 'PoE End (dup)']
    else:
        indices = [0, len(z_vectors)-1, 0, len(z_vectors)-1]  # Pad with duplicates
        labels = ['PoE Start', 'PoE End', 'PoE Start (dup)', 'PoE End (dup)']
    
    for i, (idx, label) in enumerate(zip(indices, labels)):
        ax = fig.add_subplot(gs[0, 4 + i])
        
        try:
            with torch.no_grad():
                # Get PoE z vector for this step
                poe_z = torch.tensor(z_vectors[idx], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Generate reconstruction
                shape_logits, grid_logits = model.multi_encoder.decoder(
                    poe_z, input_tensor, target_seq=target_tensor
                )
                
                # Extract and reshape predicted grid
                grid_pred = grid_logits.argmax(dim=-1).cpu().numpy()[0]
                if target_rows > 0 and target_cols > 0:
                    pred_grid = grid_pred[:target_rows*target_cols].reshape(target_rows, target_cols)
                    im = ax.imshow(pred_grid, cmap='tab20', vmin=0, vmax=9)
                    
                    # Add loss information if available
                    if losses and idx < len(losses):
                        ax.set_title(f'{label}\nLoss: {losses[idx]:.3f}', fontsize=10)
                    else:
                        ax.set_title(f'{label}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, f'{label}\nInvalid dims', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{label}', fontsize=10)
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'{label}\nError:\n{str(e)[:15]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{label}', fontsize=10)
        
        ax.axis('off')
    
    # ============= MIDDLE ROW: INPUT/TARGET/ANALYSIS =============
    
    # Input grid (column 0, row 1)
    ax = fig.add_subplot(gs[1, 0])
    try:
        input_grid = extract_grid_from_sequence(input_sample)
        if input_grid.shape[0] > 0 and input_grid.shape[1] > 0:
            im = ax.imshow(input_grid, cmap='tab20', vmin=0, vmax=9)
            ax.set_title(f'Input\n{input_grid.shape[0]}×{input_grid.shape[1]}', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Empty\nInput', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Input', fontsize=10)
    except Exception as e:
        ax.text(0.5, 0.5, f'Input\nError', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Input', fontsize=10)
    ax.axis('off')
    
    # Target grid (column 1, row 1)  
    ax = fig.add_subplot(gs[1, 1])
    try:
        target_grid = extract_grid_from_sequence(target_sample)
        if target_grid.shape[0] > 0 and target_grid.shape[1] > 0:
            im = ax.imshow(target_grid, cmap='tab20', vmin=0, vmax=9)
            ax.set_title(f'Target\n{target_grid.shape[0]}×{target_grid.shape[1]}', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Empty\nTarget', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Target', fontsize=10)
    except Exception as e:
        ax.text(0.5, 0.5, f'Target\nError', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Target', fontsize=10)
    ax.axis('off')
    
    # Loss trajectory (columns 2-3, row 1)
    ax = fig.add_subplot(gs[1, 2:4])
    if losses and len(losses) > 1:
        ax.plot(range(len(losses)), losses, 'b-o', linewidth=2, markersize=4)
        ax.set_title('PoE Optimization Loss Trajectory', fontsize=11)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Mark the specific steps we're showing
        for i, idx in enumerate(indices[:4]):  # Only mark first 4
            if idx < len(losses):
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.7)
                ax.text(idx, max(losses)*0.9, f'Step {i+1}', rotation=90, 
                       fontsize=8, ha='right', va='top')
    else:
        ax.text(0.5, 0.5, 'No loss data\navailable', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Loss Trajectory (No Data)', fontsize=11)
    
    # Comparative analysis (columns 4-7, row 1)
    ax = fig.add_subplot(gs[1, 4:8])
    ax.axis('off')
    
    # Create analysis text
    analysis_lines = [
        "MULTI-ENCODER TRAJECTORY ANALYSIS",
        "=" * 40,
        f"Number of encoders: {num_encoders}",
        f"Trajectory length: {len(z_vectors)} optimization steps",
    ]
    
    if losses and len(losses) > 1:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = initial_loss - final_loss
        analysis_lines.extend([
            "",
            "OPTIMIZATION PERFORMANCE:",
            f"Initial loss: {initial_loss:.4f}",
            f"Final loss: {final_loss:.4f}",
            f"Improvement: {improvement:.4f} ({improvement/initial_loss*100:.1f}%)",
            f"Optimization {'SUCCESS' if improvement > 0 else 'FAILED'}"
        ])
    
    analysis_lines.extend([
        "",
        "VISUALIZATION DETAILS:",
        "• Individual encoders: Static outputs (optimization independent)",  
        "• PoE trajectory: Shows latent evolution during optimization",
        "• Red lines: Mark specific trajectory steps shown above"
    ])
    
    ax.text(0.05, 0.95, '\n'.join(analysis_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ============= BOTTOM ROW: LATENT SPACE ANALYSIS =============
    
    # Latent space trajectory in 2D (columns 0-3, row 2)
    ax = fig.add_subplot(gs[2, 0:4])
    try:
        # Load training latent data for background
        training_latent_data = load_training_latent_data(os.path.dirname(save_path))
        
        # Create 2D projection of trajectory
        z_array = np.array([z.flatten() if hasattr(z, 'flatten') else z for z in z_vectors])
        
        if training_latent_data is not None and len(training_latent_data) > 10:
            # Combine training and trajectory data for consistent t-SNE/PCA
            try:
                from sklearn.manifold import TSNE
                from sklearn.preprocessing import StandardScaler
                
                # Limit training data for performance
                max_training_samples = min(500, len(training_latent_data))
                training_subset = training_latent_data[:max_training_samples]
                
                # Combine data
                combined_data = np.vstack([training_subset, z_array])
                
                # Standardize
                scaler = StandardScaler()
                combined_data_scaled = scaler.fit_transform(combined_data)
                
                # Apply t-SNE if enough samples, otherwise PCA
                if len(combined_data) > 50:
                    perplexity = min(30, len(combined_data) - 1)
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
                    combined_2d = tsne.fit_transform(combined_data_scaled)
                    method_name = "t-SNE"
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    combined_2d = pca.fit_transform(combined_data_scaled)
                    method_name = f"PCA (var: {sum(pca.explained_variance_ratio_):.1%})"
                
                # Split back
                n_training = len(training_subset)
                training_2d = combined_2d[:n_training]
                z_2d = combined_2d[n_training:]
                
                # Plot training background
                ax.scatter(training_2d[:, 0], training_2d[:, 1],
                          c=np.arange(len(training_2d)), cmap='viridis', 
                          alpha=0.3, s=20, edgecolors='none', label='Training Data')
                
                print(f"  ✓ Using {method_name} with {len(training_subset)} training samples as background")
                
            except Exception as e:
                print(f"  ⚠ Could not use training background ({e}), falling back to trajectory-only")
                # Fallback to trajectory-only PCA
                if z_array.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    z_2d = pca.fit_transform(z_array)
                    method_name = f"PCA (var: {sum(pca.explained_variance_ratio_):.1%})"
                else:
                    z_2d = z_array[:, :2]
                    method_name = "2D Direct"
        else:
            # No training data available, use trajectory-only visualization
            if z_array.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                z_2d = pca.fit_transform(z_array)
                method_name = f"PCA (var: {sum(pca.explained_variance_ratio_):.1%})"
            else:
                z_2d = z_array[:, :2]
                method_name = "2D Direct"
        
        # Plot trajectory path
        if losses:
            # Color by loss values
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=losses, cmap='plasma', s=50, alpha=0.8, label='PoE Trajectory')
            plt.colorbar(scatter, ax=ax, shrink=0.6, label='Loss')
        else:
            ax.plot(z_2d[:, 0], z_2d[:, 1], 'b-o', alpha=0.7, markersize=4, label='PoE Trajectory')
        
        # Mark start and end
        ax.scatter(z_2d[0, 0], z_2d[0, 1], c='green', s=100, marker='o', 
                  label='Start', edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(z_2d[-1, 0], z_2d[-1, 1], c='red', s=100, marker='s', 
                  label='End', edgecolors='black', linewidth=2, zorder=10)
        
        # Mark the specific steps we're showing
        for i, idx in enumerate(indices[:4]):
            if idx < len(z_2d):
                ax.scatter(z_2d[idx, 0], z_2d[idx, 1], c='white', s=80, marker='D',
                          edgecolors='black', linewidth=2, zorder=15)
        
        ax.set_title(f'PoE Latent Space Trajectory ({method_name})', fontsize=11)
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Latent space visualization error:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Latent Space Trajectory (Error)')
    
    # Individual encoder latent positions (columns 4-7, row 2)
    ax = fig.add_subplot(gs[2, 4:8])
    try:
        # Show individual encoder latent positions relative to PoE trajectory
        if individual_encoder_trajectories and z_vectors:
            
            # Collect individual encoder z vectors
            encoder_zs = []
            encoder_labels = []
            for enc_idx in range(num_encoders):
                encoder_key = f'encoder_{enc_idx}'
                if encoder_key in individual_encoder_trajectories:
                    enc_z = individual_encoder_trajectories[encoder_key]['z'].flatten()
                    encoder_zs.append(enc_z)
                    encoder_labels.append(f'Enc {enc_idx}')
            
            if encoder_zs:
                # Combine with trajectory for consistent dimensionality reduction
                all_z = np.vstack([z_array] + encoder_zs)
                
                if all_z.shape[1] > 2:
                    pca = PCA(n_components=2)
                    all_z_2d = pca.fit_transform(all_z)
                else:
                    all_z_2d = all_z[:, :2]
                
                # Split back
                traj_2d = all_z_2d[:len(z_vectors)]
                enc_2d = all_z_2d[len(z_vectors):]
                
                # Plot trajectory in background
                ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'gray', alpha=0.5, linewidth=1, label='PoE Trajectory')
                
                # Plot individual encoder positions
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                for i, (enc_z_2d, label) in enumerate(zip(enc_2d, encoder_labels)):
                    color = colors[i % len(colors)]
                    ax.scatter(enc_z_2d[0], enc_z_2d[1], c=color, s=100, marker='o',
                              label=label, edgecolors='black', linewidth=2, zorder=10)
                
                ax.set_title('Individual Encoder Positions in Latent Space', fontsize=11)
                ax.set_xlabel('Latent Dimension 1')
                ax.set_ylabel('Latent Dimension 2')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No individual\nencoder data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Individual Encoder Positions')
        else:
            ax.text(0.5, 0.5, 'No encoder\nposition data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Individual Encoder Positions')
            
    except Exception as e:
        ax.text(0.5, 0.5, f'Encoder position error:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Individual Encoder Positions (Error)')
    
    plt.suptitle(f'Multi-Encoder Trajectory Reconstruction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated comprehensive multi-encoder trajectory visualization")

def create_trajectory_visualization(trajectory_info, model, save_path, device='cuda'):
    """
    Create a simplified trajectory visualization for a single sample.
    
    Args:
        trajectory_info: Dictionary containing trajectory data
        model: The trained model
        save_path: Path to save the visualization
        device: Device for computation
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data from trajectory
    z_vectors = trajectory_info.get('z_vectors', [])
    losses = trajectory_info.get('losses', [])
    input_sample = trajectory_info.get('input_sample')
    target_sample = trajectory_info.get('target_sample')
    
    if len(z_vectors) < 2:
        print("⚠ Warning: Insufficient trajectory data for visualization")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Loss trajectory
    ax = axes[0, 0]
    if losses:
        ax.plot(range(len(losses)), losses, 'b-o', linewidth=2, markersize=4)
        ax.set_title('Optimization Loss Trajectory')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Loss Trajectory (No Data)')
    
    # 2. Latent space trajectory (2D projection)
    ax = axes[0, 1]
    try:
        z_array = np.array([z.flatten() if hasattr(z, 'flatten') else z for z in z_vectors])
        if z_array.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_array)
        else:
            z_2d = z_array[:, :2]
        
        # Plot trajectory path
        ax.plot(z_2d[:, 0], z_2d[:, 1], 'g-', alpha=0.7, linewidth=2)
        ax.scatter(z_2d[0, 0], z_2d[0, 1], c='red', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(z_2d[-1, 0], z_2d[-1, 1], c='blue', s=100, marker='s', label='End', zorder=5)
        
        ax.set_title('Latent Space Trajectory (2D Projection)')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Could not create 2D projection:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Latent Trajectory (Error)')
    
    # 3. Input visualization
    ax = axes[0, 2]
    if input_sample is not None:
        try:
            input_grid = extract_grid_from_sequence(input_sample)
            if input_grid.shape[0] > 0 and input_grid.shape[1] > 0:
                im = ax.imshow(input_grid, cmap='tab20', vmin=0, vmax=9)
                ax.set_title('Input Grid')
                plt.colorbar(im, ax=ax, shrink=0.6)
            else:
                ax.text(0.5, 0.5, 'Empty input grid', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Input visualization error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Input Sample')
    
    # 4. Target visualization  
    ax = axes[1, 0]
    if target_sample is not None:
        try:
            target_grid = extract_grid_from_sequence(target_sample)
            if target_grid.shape[0] > 0 and target_grid.shape[1] > 0:
                im = ax.imshow(target_grid, cmap='tab20', vmin=0, vmax=9)
                ax.set_title('Target Grid')
                plt.colorbar(im, ax=ax, shrink=0.6)
            else:
                ax.text(0.5, 0.5, 'Empty target grid', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Target visualization error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Target Sample')
    
    # 5. Initial reconstruction
    ax = axes[1, 1]
    try:
        if input_sample is not None and len(z_vectors) > 0:
            # Use initial z vector for reconstruction
            initial_z = torch.tensor(z_vectors[0], dtype=torch.float32).unsqueeze(0).to(device)
            input_tensor = torch.tensor(input_sample, dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(target_sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if hasattr(model, 'multi_encoder'):
                    shape_logits, grid_logits = model.multi_encoder.decoder(initial_z, input_tensor, target_seq=target_tensor)
                else:
                    shape_logits, grid_logits = model.decoder(initial_z, input_tensor, target_seq=target_tensor)
                
                # Extract predicted grid
                grid_pred = grid_logits.argmax(dim=-1).cpu().numpy()[0]
                rows = int(target_sample[900]) if len(target_sample) > 900 else 10
                cols = int(target_sample[901]) if len(target_sample) > 901 else 10
                
                if rows > 0 and cols > 0:
                    pred_grid = grid_pred[:rows*cols].reshape(rows, cols)
                    im = ax.imshow(pred_grid, cmap='tab20', vmin=0, vmax=9)
                    ax.set_title('Initial Reconstruction')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                else:
                    ax.text(0.5, 0.5, 'Invalid grid dimensions', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data for reconstruction', ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f'Reconstruction error:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Initial Reconstruction')
    
    # 6. Final reconstruction
    ax = axes[1, 2]
    try:
        if input_sample is not None and len(z_vectors) > 0:
            # Use final z vector for reconstruction
            final_z = torch.tensor(z_vectors[-1], dtype=torch.float32).unsqueeze(0).to(device)
            input_tensor = torch.tensor(input_sample, dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(target_sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if hasattr(model, 'multi_encoder'):
                    shape_logits, grid_logits = model.multi_encoder.decoder(final_z, input_tensor, target_seq=target_tensor)
                else:
                    shape_logits, grid_logits = model.decoder(final_z, input_tensor, target_seq=target_tensor)
                
                # Extract predicted grid
                grid_pred = grid_logits.argmax(dim=-1).cpu().numpy()[0]
                rows = int(target_sample[900]) if len(target_sample) > 900 else 10
                cols = int(target_sample[901]) if len(target_sample) > 901 else 10
                
                if rows > 0 and cols > 0:
                    pred_grid = grid_pred[:rows*cols].reshape(rows, cols)
                    im = ax.imshow(pred_grid, cmap='tab20', vmin=0, vmax=9)
                    ax.set_title('Final Reconstruction')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                else:
                    ax.text(0.5, 0.5, 'Invalid grid dimensions', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data for reconstruction', ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f'Reconstruction error:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Final Reconstruction')
    
    plt.suptitle('Trajectory Optimization Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_trajectory_summary_analysis(trajectory_samples, eval_results, save_dir):
    """
    Create a summary analysis of trajectory optimization performance across samples.
    """
    if not trajectory_samples:
        return
    
    try:
        # Collect trajectory statistics
        initial_losses = []
        final_losses = []
        loss_improvements = []
        trajectory_lengths = []
        
        for sample in trajectory_samples:
            if sample.get('losses') and len(sample['losses']) > 1:
                initial_loss = sample['losses'][0]
                final_loss = sample['losses'][-1]
                improvement = initial_loss - final_loss
                
                initial_losses.append(initial_loss)
                final_losses.append(final_loss)
                loss_improvements.append(improvement)
                trajectory_lengths.append(len(sample['losses']))
        
        if not initial_losses:
            print("⚠ No trajectory loss data available for summary analysis")
            return
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss improvement distribution
        axes[0, 0].hist(loss_improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(loss_improvements), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(loss_improvements):.4f}')
        axes[0, 0].set_xlabel('Loss Improvement')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Loss Improvements')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Initial vs Final loss scatter
        axes[0, 1].scatter(initial_losses, final_losses, alpha=0.6, color='green')
        min_loss = min(min(initial_losses), min(final_losses))
        max_loss = max(max(initial_losses), max(final_losses))
        axes[0, 1].plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.8, label='No improvement line')
        axes[0, 1].set_xlabel('Initial Loss')
        axes[0, 1].set_ylabel('Final Loss')
        axes[0, 1].set_title('Initial vs Final Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trajectory length distribution
        axes[1, 0].hist(trajectory_lengths, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(trajectory_lengths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(trajectory_lengths):.1f}')
        axes[1, 0].set_xlabel('Trajectory Length (optimization steps)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Trajectory Lengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        
        # Calculate statistics
        stats_text = [
            "TRAJECTORY OPTIMIZATION SUMMARY",
            "=" * 35,
            f"Total samples analyzed: {len(trajectory_samples)}",
            f"Samples with valid trajectories: {len(initial_losses)}",
            "",
            "LOSS STATISTICS:",
            f"Mean initial loss: {np.mean(initial_losses):.4f} ± {np.std(initial_losses):.4f}",
            f"Mean final loss: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}",
            f"Mean improvement: {np.mean(loss_improvements):.4f} ± {np.std(loss_improvements):.4f}",
            f"Success rate (improvement > 0): {sum(1 for x in loss_improvements if x > 0) / len(loss_improvements) * 100:.1f}%",
            "",
            "TRAJECTORY STATISTICS:",
            f"Mean trajectory length: {np.mean(trajectory_lengths):.1f} ± {np.std(trajectory_lengths):.1f}",
            f"Min/Max trajectory length: {min(trajectory_lengths)}/{max(trajectory_lengths)}",
            "",
            "PERFORMANCE ASSESSMENT:",
            f"Best improvement: {max(loss_improvements):.4f}",
            f"Worst case: {min(loss_improvements):.4f}",
            f"Median improvement: {np.median(loss_improvements):.4f}"
        ]
        
        axes[1, 1].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Multi-Encoder Trajectory Optimization Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = os.path.join(save_dir, 'trajectory_optimization_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved trajectory summary analysis: {summary_path}")
        
    except Exception as e:
        print(f"⚠ Error creating trajectory summary analysis: {e}")

def plot_model_summary(results, model_params, save_dir=None):
    """
    Create a comprehensive experiment summary visualization that includes model architecture,
    training configuration, performance metrics, evaluation results, and system information.
    
    Args:
        results: Dictionary containing training results (can be None)
        model_params: Dictionary containing model parameters
        save_dir: Directory to save the plot (optional)
    """
    def format_param_count(value):
        """Helper function to format parameter counts with commas if numeric."""
        if isinstance(value, (int, float)):
            return f"{value:,}"
        else:
            return str(value)
    
    # Create figure with subplots for organized layout
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x2 grid layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                         hspace=0.4, wspace=0.3)
    
    # ============= TOP LEFT: MODEL ARCHITECTURE & CONFIGURATION =============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    summary_lines = []
    summary_lines.append("MODEL ARCHITECTURE & CONFIGURATION")
    summary_lines.append("=" * 45)
    summary_lines.append("")
    
    # Model Architecture
    num_encoders = model_params.get('NUM_ENCODERS', 1)
    is_multi_encoder = num_encoders > 1
    
    summary_lines.append("ARCHITECTURE:")
    summary_lines.append(f"  Type: {'Multi-Encoder' if is_multi_encoder else 'Single Encoder'}")
    if is_multi_encoder:
        summary_lines.append(f"  Number of encoders: {num_encoders}")
    
    summary_lines.append(f"  Latent dimension: {model_params.get('LATENT_DIM', 'N/A')}")
    summary_lines.append(f"  Hidden dimension: {model_params.get('ENCODER_HIDDEN_DIM', 'N/A')}")
    summary_lines.append(f"  Dropout rate: {model_params.get('DROPOUT', 'N/A')}")
    summary_lines.append("")
    
    # Parameters
    summary_lines.append("PARAMETERS:")
    summary_lines.append(f"  Total: {format_param_count(model_params.get('total_params', 'N/A'))}")
    summary_lines.append(f"  Trainable: {format_param_count(model_params.get('trainable_params', 'N/A'))}")
    summary_lines.append(f"  Non-trainable: {format_param_count(model_params.get('non_trainable_params', 'N/A'))}")
    summary_lines.append("")
    
    # Training Configuration
    summary_lines.append("TRAINING CONFIGURATION:")
    summary_lines.append(f"  Tasks: {model_params.get('TRAINING_KEYS', 'N/A')}")
    summary_lines.append(f"  Examples per task: {model_params.get('n', 'N/A')}")
    summary_lines.append(f"  Batch size: {model_params.get('BATCH_SIZE', 'N/A')}")
    summary_lines.append(f"  Epochs: {model_params.get('NUM_EPOCHS', 'N/A')}")
    summary_lines.append(f"  Learning rate: {model_params.get('LEARNING_RATE', 'N/A')}")
    summary_lines.append(f"  Beta (KL weight): {model_params.get('BETA', 'N/A')}")
    summary_lines.append("")
    
    # Latent Optimization
    summary_lines.append("LATENT OPTIMIZATION:")
    summary_lines.append(f"  Training opt: {model_params.get('OPTIMIZE_Z', False)}")
    summary_lines.append(f"  Inference opt: {model_params.get('OPTIMIZE_Z_INFERENCE', True)}")
    if model_params.get('OPTIMIZE_Z_INFERENCE', True):
        summary_lines.append(f"  Inf. steps: {model_params.get('OPTIMIZE_Z_INFERENCE_NUM_STEPS', 'N/A')}")
        summary_lines.append(f"  Inf. LR: {model_params.get('OPTIMIZE_Z_INFERENCE_LR', 'N/A')}")
    
    summary_text = '\n'.join(summary_lines)
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    # ============= TOP RIGHT: TRAINING PERFORMANCE =============
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    perf_lines = []
    perf_lines.append("TRAINING PERFORMANCE")
    perf_lines.append("=" * 30)
    perf_lines.append("")
    
    if results:
        # Training samples info
        training_samples = len(results.get('input_sequences', []))
        perf_lines.append(f"Training samples: {training_samples:,}")
        perf_lines.append("")
        
        # Final training accuracies
        final_acc = None
        if 'epoch_accuracies' in results and results['epoch_accuracies']:
            final_acc = results['epoch_accuracies'][-1]
        
        if final_acc:
            perf_lines.append("FINAL TRAINING ACCURACY:")
            if is_multi_encoder and 'individual_encoders' in final_acc:
                # Multi-encoder format
                perf_lines.append("Individual encoders:")
                for enc_idx, enc_acc in final_acc['individual_encoders'].items():
                    exact_acc = enc_acc.get('sample_exact_accuracy', 0)
                    perf_lines.append(f"  Enc {enc_idx}: {exact_acc:.3f}")
                
                # Calculate average
                if final_acc['individual_encoders']:
                    avg_exact = sum(enc.get('sample_exact_accuracy', 0) 
                                  for enc in final_acc['individual_encoders'].values()) / len(final_acc['individual_encoders'])
                    perf_lines.append(f"  Average: {avg_exact:.3f}")
            else:
                # Single encoder format
                perf_lines.append(f"Shape: {final_acc.get('shape_accuracy', 0):.3f}")
                perf_lines.append(f"Grid: {final_acc.get('grid_accuracy', 0):.3f}")
                perf_lines.append(f"Pixel: {final_acc.get('sample_exact_accuracy', 0):.3f}")
        
        # Final loss
        if 'epoch_losses' in results and results['epoch_losses']:
            final_loss = results['epoch_losses'][-1]
            perf_lines.append("")
            perf_lines.append(f"Final loss: {final_loss:.4f}")
        
        # Training progress
        total_epochs = len(results.get('epoch_losses', []))
        if total_epochs > 0:
            perf_lines.append(f"Epochs completed: {total_epochs}")
        
        # Latent optimization usage
        has_trajectory = len(results.get('losses_gradient_ascent', [])) > 0
        perf_lines.append(f"Used latent opt: {has_trajectory}")
        
    else:
        perf_lines.append("No training data available")
    
    perf_text = '\n'.join(perf_lines)
    ax2.text(0.05, 0.95, perf_text, transform=ax2.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    # ============= BOTTOM LEFT: EVALUATION RESULTS =============
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    eval_lines = []
    eval_lines.append("EVALUATION RESULTS")
    eval_lines.append("=" * 25)
    eval_lines.append("")
    
    # Try to load evaluation results
    eval_results = None
    if save_dir:
        eval_file = os.path.join(save_dir, 'evaluation_results.pkl')
        if os.path.exists(eval_file):
            try:
                import pickle
                with open(eval_file, 'rb') as f:
                    eval_results = pickle.load(f)
            except Exception as e:
                eval_lines.append(f"Error loading eval: {str(e)[:30]}...")
    
    if eval_results:
        eval_lines.append(f"Tasks evaluated: {len(eval_results)}")
        eval_lines.append("")
        
        # Calculate averages
        avg_metrics = {
            'shape_accuracy': 0,
            'grid_accuracy': 0,
            'sample_exact_accuracy': 0,
            'support_loss': 0,
            'query_loss': 0
        }
        
        valid_tasks = 0
        multi_encoder_info = None
        
        for key, key_results in eval_results.items():
            if 'metrics' in key_results and 'error' not in key_results['metrics']:
                metrics = key_results['metrics']
                
                # Store multi-encoder info from first valid task
                if multi_encoder_info is None and metrics.get('is_multi_encoder', False):
                    multi_encoder_info = metrics.get('comparative_analysis', {})
                
                for metric_name in avg_metrics.keys():
                    if metric_name in metrics:
                        avg_metrics[metric_name] += metrics[metric_name]
                valid_tasks += 1
        
        if valid_tasks > 0:
            # Average metrics
            eval_lines.append("AVERAGE PERFORMANCE:")
            eval_lines.append(f"Shape: {avg_metrics['shape_accuracy']/valid_tasks:.3f}")
            eval_lines.append(f"Grid: {avg_metrics['grid_accuracy']/valid_tasks:.3f}")
            eval_lines.append(f"Pixel: {avg_metrics['sample_exact_accuracy']/valid_tasks:.3f}")
            eval_lines.append(f"Support loss: {avg_metrics['support_loss']/valid_tasks:.3f}")
            eval_lines.append(f"Query loss: {avg_metrics['query_loss']/valid_tasks:.3f}")
            
            # Multi-encoder analysis
            if multi_encoder_info:
                eval_lines.append("")
                eval_lines.append("MULTI-ENCODER ANALYSIS:")
                poe_advantage = multi_encoder_info.get('poe_vs_best_advantage', 0)
                spec_range = multi_encoder_info.get('specialization_range', 0)
                eval_lines.append(f"PoE advantage: {poe_advantage:+.3f}")
                eval_lines.append(f"Specialization: {spec_range:.3f}")
                
                # Interpretation
                if poe_advantage > 0.01:
                    eval_lines.append("→ PoE outperforms")
                elif poe_advantage < -0.01:
                    eval_lines.append("→ Individual best")
                else:
                    eval_lines.append("→ Similar performance")
        else:
            eval_lines.append("No successful evaluations")
    else:
        eval_lines.append("No evaluation data available")
    
    eval_text = '\n'.join(eval_lines)
    ax3.text(0.05, 0.95, eval_text, transform=ax3.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    # ============= BOTTOM RIGHT: SYSTEM INFO & EXPERIMENT METADATA =============
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    sys_lines = []
    sys_lines.append("SYSTEM & EXPERIMENT INFO")
    sys_lines.append("=" * 35)
    sys_lines.append("")
    
    # System information
    try:
        import platform
        import sys
        import torch
        import datetime
        
        sys_lines.append("SYSTEM:")
        sys_lines.append(f"  Python: {sys.version.split()[0]}")
        sys_lines.append(f"  PyTorch: {torch.__version__}")
        sys_lines.append(f"  Platform: {platform.system()} {platform.release()}")
        sys_lines.append(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
            # Truncate long GPU names
            if len(gpu_name) > 25:
                gpu_name = gpu_name[:22] + "..."
            sys_lines.append(f"  GPU: {gpu_name}")
        
        sys_lines.append("")
        sys_lines.append("EXPERIMENT:")
        sys_lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if save_dir:
            exp_name = os.path.basename(save_dir)
            if len(exp_name) > 25:
                exp_name = exp_name[:22] + "..."
            sys_lines.append(f"  Name: {exp_name}")
        
        sys_lines.append(f"  Seed: {model_params.get('TRAINING_SEED', 'N/A')}")
        
        # File status
        sys_lines.append("")
        sys_lines.append("FILES:")
        sys_lines.append(f"  Training: {'✓' if results else '✗'}")
        if save_dir:
            eval_exists = os.path.exists(os.path.join(save_dir, 'evaluation_results.pkl'))
            sys_lines.append(f"  Evaluation: {'✓' if eval_exists else '✗'}")
            model_exists = os.path.exists(os.path.join(save_dir, 'model.pth'))
            sys_lines.append(f"  Model: {'✓' if model_exists else '✗'}")
        
    except Exception as e:
        sys_lines.append(f"System info error: {str(e)[:30]}...")
    
    sys_text = '\n'.join(sys_lines)
    ax4.text(0.05, 0.95, sys_text, transform=ax4.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    # ============= OVERALL TITLE =============
    experiment_name = os.path.basename(save_dir) if save_dir else "Unknown Experiment"
    plt.suptitle(f'Experiment Summary: {experiment_name}', fontsize=18, weight='bold', y=0.95)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'model_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Model summary saved to {save_dir}/model_summary.png")
    else:
        plt.show()


def create_simple_analysis_report(results, model_params, eval_results, save_dir):
    """
    Create a comprehensive JSON summary of the experiment (no longer creates .txt file).
    The visual summary is now integrated into model_summary.png.
    
    Args:
        results: Training results dictionary
        model_params: Model parameter information
        eval_results: Evaluation results dictionary  
        save_dir: Directory to save the report
    """
    try:
        import json
        import datetime
        from utils.settings_manager import settings
        
        # Get all settings
        settings_info = settings.get_settings()
        
        # Build comprehensive summary
        summary = {
            "experiment_info": {
                "experiment_name": os.path.basename(save_dir) if save_dir else "Unknown",
                "generation_time": str(datetime.datetime.now()),
                "run_directory": save_dir
            },
            
            "model_architecture": settings_info.get('model_architecture', {}),
            "training_settings": settings_info.get('training_settings', {}),
            "data_settings": settings_info.get('data_settings', {}),
            "latent_optimization": settings_info.get('latent_optimization', {}),
            "evaluation_settings": settings_info.get('evaluation_settings', {}),
            
            "model_parameters": {
                "total_params": model_params.get('total_params', 'N/A') if model_params else 'N/A',
                "trainable_params": model_params.get('trainable_params', 'N/A') if model_params else 'N/A',
                "non_trainable_params": model_params.get('non_trainable_params', 'N/A') if model_params else 'N/A',
                "is_multi_encoder": model_params.get('is_multi_encoder_model', False) if model_params else False,
                "component_breakdown": model_params.get('component_breakdown', {}) if model_params else {}
            },
            
            "training_performance": {},
            "evaluation_performance": {},
            
            "system_info": {}
        }
        
        # Add training performance if available
        if results:
            # Get final training accuracies
            final_acc = None
            if 'epoch_accuracies' in results and results['epoch_accuracies']:
                final_acc = results['epoch_accuracies'][-1]
            
            summary["training_performance"] = {
                "final_accuracy": final_acc,
                "final_loss": results['epoch_losses'][-1] if 'epoch_losses' in results and results['epoch_losses'] else 'N/A',
                "total_epochs": len(results.get('epoch_losses', [])),
                "training_samples": len(results.get('input_sequences', [])),
                "has_trajectory_info": len(results.get('losses_gradient_ascent', [])) > 0
            }
        
        # Add evaluation performance if available
        if eval_results:
            eval_summary = {
                "total_tasks": len(eval_results),
                "successful_evaluations": 0,
                "average_metrics": {},
                "task_results": {}
            }
            
            # Calculate averages
            avg_metrics = {
                'shape_accuracy': 0,
                'grid_accuracy': 0,
                'overall_accuracy': 0,
                'sample_exact_accuracy': 0,
                'support_loss': 0,
                'query_loss': 0
            }
            
            valid_keys = 0
            for key, key_results in eval_results.items():
                if 'metrics' in key_results and 'error' not in key_results['metrics']:
                    metrics = key_results['metrics']
                    eval_summary["task_results"][key] = {
                        "shape_accuracy": metrics.get('shape_accuracy', 'N/A'),
                        "grid_accuracy": metrics.get('grid_accuracy', 'N/A'),
                        "sample_exact_accuracy": metrics.get('sample_exact_accuracy', 'N/A'),
                        "support_loss": metrics.get('support_loss', 'N/A'),
                        "query_loss": metrics.get('query_loss', 'N/A'),
                        "is_multi_encoder": metrics.get('is_multi_encoder', False)
                    }
                    
                    for metric_name in avg_metrics.keys():
                        if metric_name in metrics:
                            avg_metrics[metric_name] += metrics[metric_name]
                    valid_keys += 1
                else:
                    eval_summary["task_results"][key] = {"error": key_results.get('metrics', {}).get('error', 'Unknown error')}
            
            if valid_keys > 0:
                eval_summary["successful_evaluations"] = valid_keys
                eval_summary["average_metrics"] = {k: v/valid_keys for k, v in avg_metrics.items()}
                
                # Add multi-encoder analysis if available
                first_key_results = list(eval_results.values())[0]
                if first_key_results.get('metrics', {}).get('is_multi_encoder', False):
                    comp_analysis = first_key_results['metrics'].get('comparative_analysis', {})
                    eval_summary["multi_encoder_analysis"] = {
                        "num_encoders": first_key_results['metrics'].get('num_encoders', 'N/A'),
                        "poe_vs_best_advantage": comp_analysis.get('poe_vs_best_advantage', 'N/A'),
                        "poe_vs_avg_advantage": comp_analysis.get('poe_vs_avg_advantage', 'N/A'),
                        "specialization_range": comp_analysis.get('specialization_range', 'N/A')
                    }
            
            summary["evaluation_performance"] = eval_summary
        
        # Add system info
        try:
            import platform
            import sys
            import torch
            
            summary["system_info"] = {
                "python_version": sys.version.split()[0],
                "platform": f"{platform.system()} {platform.release()}",
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
            
            if torch.cuda.is_available():
                summary["system_info"].update({
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "current_gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
                })
        except Exception as e:
            summary["system_info"] = {"error": str(e)}
        
        # Save only as JSON (no more .txt file)
        json_path = os.path.join(save_dir, 'experiment_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Saved experiment summary: {json_path}")
        print("✓ Visual summary integrated into model_summary.png")
        
    except Exception as e:
        print(f"⚠ Error creating experiment summary: {e}")
        import traceback
        traceback.print_exc()

def extract_multi_encoder_accuracies(results):
    """
    Extract per-encoder accuracy data from multi-encoder training results.
    Note: PoE is only evaluated during inference, not training.
    
    Args:
        results: Training results dictionary
        
    Returns:
        dict: Processed accuracy data for visualization
    """
    if 'epoch_accuracies' not in results or not results['epoch_accuracies']:
        return None
    
    # Check if this includes detailed multi-encoder accuracy data
    detailed_epochs = []
    for epoch_data in results['epoch_accuracies']:
        if isinstance(epoch_data, dict) and 'individual_encoders' in epoch_data:
            detailed_epochs.append(epoch_data)
    
    if not detailed_epochs:
        print("No detailed multi-encoder accuracy data found")
        return None
    
    print("Processing multi-encoder training accuracy data for visualization...")
    
    # Initialize accuracy storage
    encoder_indices = set()
    
    # Find all encoder indices from the detailed epochs
    for epoch_data in detailed_epochs:
        encoder_indices.update(epoch_data['individual_encoders'].keys())
    
    num_encoders = len(encoder_indices)
    num_epochs = len(detailed_epochs)
    
    print(f"Found detailed accuracy data: {num_encoders} encoders, {num_epochs} epochs")
    print("Note: PoE accuracy only available during evaluation, not training")
    
    # Initialize per-encoder accuracy storage (no PoE during training)
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
        'has_poe_data': False  # No PoE during training
    }
    
    # Extract accuracy data for each epoch
    for epoch_data in detailed_epochs:
        # Individual encoder accuracies
        for enc_idx in encoder_indices:
            if enc_idx in epoch_data['individual_encoders']:
                enc_data = epoch_data['individual_encoders'][enc_idx]
                encoder_accuracies['individual_encoders'][enc_idx]['shape_accuracy'].append(enc_data['shape_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['grid_accuracy'].append(enc_data['grid_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['overall_accuracy'].append(enc_data['overall_accuracy'])
                encoder_accuracies['individual_encoders'][enc_idx]['sample_exact_accuracy'].append(enc_data['sample_exact_accuracy'])
            else:
                # Fill with zeros if encoder data missing
                encoder_accuracies['individual_encoders'][enc_idx]['shape_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['grid_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['overall_accuracy'].append(0.0)
                encoder_accuracies['individual_encoders'][enc_idx]['sample_exact_accuracy'].append(0.0)
    
    print(f"✓ Multi-encoder accuracy data processed: {num_encoders} encoders, {num_epochs} epochs")
    return encoder_accuracies

def plot_multi_encoder_accuracies(results, save_dir=None):
    """
    Plot detailed multi-encoder accuracy curves showing per-encoder performance during training.
    Note: PoE accuracy is only available during evaluation, not training.
    
    Args:
        results: Training results dictionary
        save_dir: Directory to save plots (optional)
    """
    accuracy_data = extract_multi_encoder_accuracies(results)
    
    if accuracy_data is None:
        print("No detailed multi-encoder accuracy data found, skipping accuracy plots")
        return
    
    epochs = accuracy_data['epochs']
    num_encoders = len(accuracy_data['individual_encoders'])
    has_poe_data = accuracy_data.get('has_poe_data', False)
    
    # Create subplots for different accuracy metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different encoders
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    metrics = ['shape_accuracy', 'grid_accuracy', 'overall_accuracy', 'sample_exact_accuracy']
    metric_titles = ['Shape Accuracy', 'Grid Accuracy', 'Overall Accuracy', 'Sample Exact Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axs[idx // 2, idx % 2]
        
        # Plot individual encoder accuracies
        for i, (encoder_idx, enc_data) in enumerate(accuracy_data['individual_encoders'].items()):
            ax.plot(epochs, enc_data[metric], marker='o', label=f'Encoder {encoder_idx}', 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Only plot PoE if data is available (evaluation results)
        if has_poe_data and 'poe_accuracy' in accuracy_data:
            ax.plot(epochs, accuracy_data['poe_accuracy'][metric], 'k-', linewidth=4, 
                   label='Product of Experts (PoE)', alpha=0.9)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # Set consistent y-axis for accuracies
    
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

def plot_multi_encoder_accuracy_summary(results, save_dir=None):
    """
    Plot a summary view of multi-encoder accuracy showing final performance and convergence.
    Note: During training, only individual encoder data is available (no PoE).
    
    Args:
        results: Training results dictionary
        save_dir: Directory to save plots (optional)
    """
    accuracy_data = extract_multi_encoder_accuracies(results)
    
    if accuracy_data is None:
        print("No detailed multi-encoder accuracy data found, skipping accuracy summary")
        return
    
    epochs = accuracy_data['epochs']
    num_encoders = len(accuracy_data['individual_encoders'])
    has_poe_data = accuracy_data.get('has_poe_data', False)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Final accuracy comparison across encoders
    final_overall_accuracies = []
    encoder_labels = []
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders + (1 if has_poe_data else 0)))
    
    for encoder_idx, enc_data in accuracy_data['individual_encoders'].items():
        final_overall_accuracies.append(enc_data['overall_accuracy'][-1])
        encoder_labels.append(f'Encoder {encoder_idx}')
    
    # Add PoE final accuracy only if available
    if has_poe_data and 'poe_accuracy' in accuracy_data:
        final_overall_accuracies.append(accuracy_data['poe_accuracy']['overall_accuracy'][-1])
        encoder_labels.append('PoE')
    
    bars = axs[0].bar(encoder_labels, final_overall_accuracies, color=colors[:len(final_overall_accuracies)])
    title_suffix = " (Training)" if not has_poe_data else " (Including PoE)"
    axs[0].set_title(f'Final Overall Accuracy{title_suffix}', fontsize=14)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks(range(len(encoder_labels)))
    axs[0].set_xticklabels(encoder_labels, rotation=45)
    axs[0].set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_overall_accuracies):
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Accuracy variance across encoders over time
    accuracy_variances = []
    for epoch_idx in range(len(epochs)):
        epoch_accuracies = [accuracy_data['individual_encoders'][enc_idx]['overall_accuracy'][epoch_idx] 
                          for enc_idx in accuracy_data['individual_encoders'].keys()]
        accuracy_variances.append(np.var(epoch_accuracies))
    
    axs[1].plot(epochs, accuracy_variances, marker='o', linewidth=2, color='red')
    axs[1].set_title('Accuracy Variance Across Encoders', fontsize=14)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy Variance')
    axs[1].grid(True, alpha=0.3)
    
    # 3. Overall accuracy progress: Individual encoders (and PoE if available)
    progress_title = 'Overall Accuracy Progress: Individual Encoders'
    if has_poe_data:
        progress_title += ' vs PoE'
    axs[2].set_title(progress_title, fontsize=14)
    
    # Plot individual encoder overall accuracies (thin lines)
    for i, (encoder_idx, enc_data) in enumerate(accuracy_data['individual_encoders'].items()):
        axs[2].plot(epochs, enc_data['overall_accuracy'], color=colors[i], alpha=0.6, linewidth=1.5, 
                   label=f'Encoder {encoder_idx}')
    
    # Plot PoE overall accuracy (thick line) only if available
    if has_poe_data and 'poe_accuracy' in accuracy_data:
        axs[2].plot(epochs, accuracy_data['poe_accuracy']['overall_accuracy'], 'k-', linewidth=3, 
                   label='PoE', alpha=0.9)
    
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Overall Accuracy')
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[2].grid(True, alpha=0.3)
    axs[2].set_ylim(0, 1.05)
    
    summary_title = f'Multi-Encoder Accuracy Summary ({num_encoders} Encoders)'
    if not has_poe_data:
        summary_title += ' - Training Phase'
    plt.suptitle(summary_title, fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        filename = 'multi_encoder_training_accuracy_summary.png' if not has_poe_data else 'multi_encoder_accuracy_summary.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder accuracy summary plot saved to {save_dir}/{filename}")
    else:
        plt.show()

def plot_evaluation_results(eval_results, save_dir=None):
    """
    Plot evaluation results. Handles both single-encoder and multi-encoder evaluation results.
    
    Args:
        eval_results: Dictionary containing evaluation results for each key
        save_dir: Directory to save plots (optional)
    """
    print("Processing evaluation results...")
    
    for key, key_results in eval_results.items():
        print(f"\nProcessing evaluation results for key: {key}")
        
        if 'metrics' not in key_results:
            print(f"No metrics found for key {key}, skipping")
            continue
        
        metrics = key_results['metrics']
        is_multi_encoder = metrics.get('is_multi_encoder', False)
        
        if is_multi_encoder:
            print(f"Multi-encoder evaluation detected with {metrics['num_encoders']} encoders")
            # Use enhanced multi-encoder evaluation with training data analysis
            plot_multi_encoder_evaluation_results(key, key_results, save_dir)
        else:
            print("Single-encoder evaluation detected")
            plot_single_encoder_evaluation_results(key, key_results, save_dir)

def plot_single_encoder_evaluation_results(key, key_results, save_dir=None):
    """
    Plot evaluation results for single-encoder models.
    
    Args:
        key: Problem key
        key_results: Evaluation results for this key
        save_dir: Directory to save plots (optional)
    """
    metrics = key_results['metrics']
    
    # Create evaluation summary plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Accuracy metrics bar chart
    accuracy_names = ['Shape', 'Grid', 'Overall', 'Sample Exact']
    accuracy_values = [
        metrics['shape_accuracy'],
        metrics['grid_accuracy'],
        metrics['overall_accuracy'],
        metrics['sample_exact_accuracy']
    ]
    
    bars = axs[0].bar(accuracy_names, accuracy_values, color=['skyblue', 'lightgreen', 'orange', 'coral'])
    axs[0].set_title(f'Evaluation Accuracy Metrics - {key}', fontsize=14)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracy_values):
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Loss comparison
    loss_names = ['Support Loss', 'Query Loss']
    loss_values = [metrics['support_loss'], metrics['query_loss']]
    
    bars = axs[1].bar(loss_names, loss_values, color=['lightcoral', 'lightsalmon'])
    axs[1].set_title(f'Evaluation Losses - {key}', fontsize=14)
    axs[1].set_ylabel('Loss')
    
    # Add value labels on bars
    for bar, loss in zip(bars, loss_values):
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{loss:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Single-Encoder Evaluation Results: {key}', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        filename = f'evaluation_results_{key}_single_encoder.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Single-encoder evaluation plot saved to {save_dir}/{filename}")
    else:
        plt.show()

def plot_multi_encoder_evaluation_results(key, key_results, save_dir=None):
    """
    Enhanced multi-encoder evaluation visualization with training data distribution analysis.
    
    Args:
        key: Problem key
        key_results: Evaluation results for this key
        save_dir: Directory to save plots (optional)
    """
    metrics = key_results['metrics']
    num_encoders = metrics['num_encoders']
    
    print(f"Creating enhanced multi-encoder evaluation analysis for {num_encoders} encoders...")
    
    # Create comprehensive multi-encoder evaluation plot with limited size
    max_figure_size = (14, 10)  # Reduced to prevent oversized images
    fig, axs = plt.subplots(3, 3, figsize=max_figure_size)
    
    # Extract individual encoder accuracies and PoE accuracies
    individual_accuracies = metrics.get('individual_encoder_accuracies', {})
    poe_metrics = metrics.get('poe_metrics', {})
    
    # Check if we have valid evaluation data
    if not individual_accuracies or len(individual_accuracies) == 0:
        # Fallback to simple single-encoder evaluation
        print(f"⚠ Warning: No multi-encoder evaluation data found for {key}, using simple evaluation")
        plot_single_encoder_evaluation_results(key, key_results, save_dir)
        return
    
    # Colors for encoders
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders + 1))
    colors[-1] = [0, 0, 0, 1]  # Make PoE black
    
    # 1. Training Data Distribution Analysis (top-left)
    ax = axs[0, 0]
    
    # Simulate training data distribution per encoder (since we split the data)
    # This shows how the training data was distributed among encoders
    encoder_names = list(individual_accuracies.keys()) + ['PoE']
    
    # Create a pie chart showing data distribution
    data_per_encoder = [100/num_encoders] * num_encoders  # Equal split
    wedges, texts, autotexts = ax.pie(data_per_encoder, labels=[f'Encoder {i}' for i in range(num_encoders)], 
                                      colors=colors[:-1], autopct='%1.1f%%', startangle=90)
    ax.set_title('Training Data Distribution\n(Multi-Encoder Split)', fontsize=12, weight='bold')
    
    # Add center text
    ax.text(0, 0, f'{num_encoders}\nEncoders', ha='center', va='center', fontsize=10, weight='bold')
    
    # 2. Encoder Specialization Analysis (top-center)
    ax = axs[0, 1]
    
    # Calculate specialization metrics
    shape_accuracies = [individual_accuracies[enc]['shape_accuracy'] for enc in individual_accuracies.keys()]
    grid_accuracies = [individual_accuracies[enc]['grid_accuracy'] for enc in individual_accuracies.keys()]
    exact_accuracies = [individual_accuracies[enc]['sample_exact_accuracy'] for enc in individual_accuracies.keys()]
    
    # Specialization is measured by variance in performance
    shape_var = np.var(shape_accuracies)
    grid_var = np.var(grid_accuracies)
    exact_var = np.var(exact_accuracies)
    
    specialization_metrics = ['Shape Accuracy', 'Grid Accuracy', 'Exact Accuracy']
    variances = [shape_var, grid_var, exact_var]
    
    bars = ax.bar(specialization_metrics, variances, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax.set_title('Encoder Specialization\n(Performance Variance)', fontsize=12, weight='bold')
    ax.set_ylabel('Variance')
    ax.set_xticks(range(len(specialization_metrics)))
    ax.set_xticklabels(specialization_metrics, rotation=45)
    
    # Add interpretation text
    for i, (bar, var) in enumerate(zip(bars, variances)):
        height = bar.get_height()
        interpretation = "High" if var > 0.05 else "Moderate" if var > 0.01 else "Low"
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{var:.3f}\n({interpretation})', ha='center', va='bottom', fontsize=8)
    
    # 3. PoE Benefit Analysis (top-right)
    ax = axs[0, 2]
    
    # Compare PoE vs best individual encoder
    best_individual_shape = max(shape_accuracies)
    best_individual_grid = max(grid_accuracies)
    best_individual_exact = max(exact_accuracies)
    
    comparison_metrics = ['Shape', 'Grid', 'Exact']
    best_individual = [best_individual_shape, best_individual_grid, best_individual_exact]
    poe_results = [poe_metrics.get('shape_accuracy', 0.0), poe_metrics.get('grid_accuracy', 0.0), 
                   poe_metrics.get('sample_exact_accuracy', 0.0)]
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_individual, width, label='Best Individual', alpha=0.8, color='lightblue')
    bars2 = ax.bar(x + width/2, poe_results, width, label='PoE', alpha=0.8, color='darkblue')
    
    ax.set_title('PoE vs Best Individual Encoder', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Metric')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_metrics)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Add improvement indicators
    for i, (best, poe) in enumerate(zip(best_individual, poe_results)):
        ax.text(i - width/2, best + 0.01, f'{best:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, poe + 0.01, f'{poe:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add improvement/degradation indicator
        diff = poe - best
        color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
        symbol = '↑' if diff > 0 else '↓' if diff < 0 else '='
        ax.text(i, max(best, poe) + 0.05, f'{symbol}{abs(diff):.3f}', 
               ha='center', va='bottom', fontsize=8, color=color, weight='bold')
    
    # 4. Individual Encoder Performance Heat Map (middle-left)
    ax = axs[1, 0]
    
    # Create performance matrix
    performance_matrix = []
    encoder_labels = []
    metric_labels = ['Shape', 'Grid', 'Overall', 'Exact']
    
    for enc_name in individual_accuracies.keys():
        acc_data = individual_accuracies[enc_name]
        performance_row = [
            acc_data['shape_accuracy'],
            acc_data['grid_accuracy'], 
            acc_data['overall_accuracy'],
            acc_data['sample_exact_accuracy']
        ]
        performance_matrix.append(performance_row)
        encoder_labels.append(enc_name.replace('encoder_', 'Enc '))
    
    im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(encoder_labels)))
    ax.set_yticklabels(encoder_labels)
    ax.set_title('Individual Encoder Performance Matrix', fontsize=12, weight='bold')
    
    # Add text annotations
    for i in range(len(encoder_labels)):
        for j in range(len(metric_labels)):
            text = ax.text(j, i, f'{performance_matrix[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Accuracy')
    
    # 5. Sample Exact Accuracy Comparison (middle-center)
    ax = axs[1, 1]
    
    encoder_names_plot = list(individual_accuracies.keys()) + ['PoE']
    exact_accuracies_plot = exact_accuracies + [poe_metrics.get('sample_exact_accuracy', 0.0)]
    
    bars = ax.bar(range(len(encoder_names_plot)), exact_accuracies_plot, 
                  color=colors[:len(encoder_names_plot)])
    ax.set_title('Sample Exact Accuracy Comparison', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(range(len(encoder_names_plot)))
    ax.set_xticklabels([name.replace('encoder_', 'Enc ') for name in encoder_names_plot], rotation=45)
    ax.set_ylim(0, 1.05)
    
    # Add value labels and rank indicators
    sorted_accuracies = sorted(enumerate(exact_accuracies_plot), key=lambda x: x[1], reverse=True)
    for rank, (idx, acc) in enumerate(sorted_accuracies):
        bars[idx].set_alpha(0.8)
        ax.text(idx, acc + 0.01, f'{acc:.3f}\n#{rank+1}', ha='center', va='bottom', fontsize=9)
    
    # 6. Evaluation Strategy Analysis (middle-right)
    ax = axs[1, 2]
    ax.axis('off')
    
    # Analysis text
    analysis_text = "EVALUATION STRATEGY ANALYSIS\n" + "="*35 + "\n\n"
    analysis_text += f"Multi-Encoder Setup:\n"
    analysis_text += f"• {num_encoders} encoders trained on separate data subsets\n"
    analysis_text += f"• Each encoder sees {100/num_encoders:.1f}% of training data\n"
    analysis_text += f"• PoE combines all {num_encoders} encoders during inference\n\n"
    
    analysis_text += f"Specialization Assessment:\n"
    if exact_var > 0.05:
        analysis_text += f"• HIGH specialization (variance={exact_var:.3f})\n"
        analysis_text += f"• Encoders learned distinct strategies\n"
    elif exact_var > 0.01:
        analysis_text += f"• MODERATE specialization (variance={exact_var:.3f})\n"
        analysis_text += f"• Some encoder differentiation\n"
    else:
        analysis_text += f"• LOW specialization (variance={exact_var:.3f})\n"
        analysis_text += f"• Encoders learned similar strategies\n"
    
    analysis_text += f"\nPoE Performance:\n"
    poe_exact = poe_metrics.get('sample_exact_accuracy', 0.0)
    best_individual_exact = max(exact_accuracies)
    improvement = poe_exact - best_individual_exact
    
    if improvement > 0.05:
        analysis_text += f"• SIGNIFICANT improvement (+{improvement:.3f})\n"
    elif improvement > 0.01:
        analysis_text += f"• MODEST improvement (+{improvement:.3f})\n"
    elif improvement > -0.01:
        analysis_text += f"• NEUTRAL performance ({improvement:+.3f})\n"
    else:
        analysis_text += f"• DEGRADED performance ({improvement:+.3f})\n"
    
    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7. Training Data Impact Analysis (bottom-left)
    ax = axs[2, 0]
    
    # Simulate effect of data splitting on encoder performance
    # Show how performance might vary with different data split ratios
    split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of data per encoder
    estimated_performance = []
    
    for ratio in split_ratios:
        # Simple model: performance scales with sqrt of data amount, bounded by model capacity
        relative_perf = min(1.0, np.sqrt(ratio) * 1.2)  # Assume current is at 0.5 ratio
        estimated_performance.append(relative_perf * np.mean(exact_accuracies))
    
    ax.plot(split_ratios, estimated_performance, 'o-', linewidth=2, markersize=6)
    ax.axvline(x=1/num_encoders, color='red', linestyle='--', alpha=0.7, 
               label=f'Current Split\n(1/{num_encoders} = {1/num_encoders:.2f})')
    ax.set_xlabel('Data Fraction per Encoder')
    ax.set_ylabel('Estimated Performance')
    ax.set_title('Data Split Impact Analysis', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add current performance point
    current_ratio = 1/num_encoders
    current_perf = np.mean(exact_accuracies)
    ax.plot(current_ratio, current_perf, 'ro', markersize=8, label='Actual')
    
    # 8. Analysis Summary Panel (bottom-center and right)
    ax_combined = plt.subplot2grid((3, 3), (2, 1), colspan=2)
    ax_combined.axis('off')
    
    summary_text = "ANALYSIS SUMMARY\n" + "="*30 + "\n\n"
    
    # Performance metrics
    data_per_encoder = 100 / num_encoders
    
    # Add summary statistics
    summary_text += f"PERFORMANCE METRICS:\n"
    summary_text += f"• Best Individual Encoder: {max(exact_accuracies):.3f}\n"
    summary_text += f"• PoE Performance: {poe_exact:.3f}\n"
    summary_text += f"• PoE Improvement: {improvement:+.3f}\n"
    summary_text += f"• Encoder Specialization: {exact_var:.3f}\n"
    summary_text += f"• Data per Encoder: {data_per_encoder:.1f}%\n"
    
    # Configuration details
    summary_text += f"\nCONFIGURATION:\n"
    summary_text += f"• Number of Encoders: {num_encoders}\n"
    summary_text += f"• Support Samples: {metrics.get('support_samples', 'N/A')}\n"
    summary_text += f"• Query Samples: {metrics.get('query_samples', 'N/A')}\n"
    summary_text += f"• Latent Optimization: {metrics.get('used_latent_optimization', 'N/A')}\n"
    
    # Performance analysis
    summary_text += f"\nPERFORMANCE ANALYSIS:\n"
    if improvement > 0.02:
        summary_text += f"• PoE shows significant benefits\n"
    elif improvement > -0.01:
        summary_text += f"• PoE shows modest/neutral benefits\n"
    else:
        summary_text += f"• PoE underperforming vs individual encoders\n"
        
    if exact_var > 0.02:
        summary_text += f"• High encoder specialization detected\n"
    else:
        summary_text += f"• Low encoder specialization detected\n"
    
    ax_combined.text(0.02, 0.98, summary_text, transform=ax_combined.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle(f'Enhanced Multi-Encoder Evaluation Analysis: {key} ({num_encoders} Encoders)', 
                 fontsize=14, y=0.96)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    if save_dir:
        filename = f'evaluation_enhanced_{key}_multi_encoder.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Enhanced multi-encoder evaluation analysis saved to {save_dir}/{filename}")
    else:
        plt.show()

# Multi-encoder detailed comparison plot removed - point (11) from evaluation

def plot_enhanced_model_summary(results, model_params, save_dir=None):
    """
    Plot a clean, well-organized summary of model parameters, training configuration, and results.
    Disabled version that falls back to simple summary to avoid errors.
    
    Args:
        results: Dictionary containing training and evaluation results (can be None)
        model_params: Dictionary containing model parameters
        save_dir: Directory to save the plot (optional)
    """
    try:
        print("⚠ Warning: Enhanced model summary disabled due to complexity issues, using simple summary")
        # Use the simpler, more reliable model summary instead
        plot_model_summary(results, model_params, save_dir)
        return
        # Create figure with clean layout
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # Create a simple grid layout without complex slicing
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = {
            'arch': '#E3F2FD',      # Light blue
            'train': '#E8F5E8',     # Light green  
            'opt': '#FFF3E0',       # Light orange
            'results': '#F3E5F5',   # Light purple
            'eval': '#E0F2F1',      # Light teal
            'summary': '#FFEBEE'    # Light red
        }
        
        # 1. MODEL ARCHITECTURE (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        num_encoders = model_params.get('NUM_ENCODERS', 1)
        is_multi_encoder = num_encoders > 1
        
        arch_lines = [
            "🏗️ MODEL ARCHITECTURE",
            "─" * 25,
            "",
            f"📐 Latent Dimension: {model_params.get('LATENT_DIM', 'N/A')}",
            f"🔧 Dropout Rate: {model_params.get('DROPOUT', 'N/A')}",
            f"📏 Max Sequence Length: {model_params.get('MAX_LENGTH', 'N/A')}",
            "",
        ]
        
        if is_multi_encoder:
            arch_lines.extend([
                f"🔄 MULTI-ENCODER ({num_encoders} encoders)",
                f"├─ Enc Hidden Dim: {model_params.get('ENCODER_HIDDEN_DIM', 'N/A')}",
                f"├─ Dec Hidden Dim: {model_params.get('DECODER_HIDDEN_DIM', 'N/A')}",
                f"├─ Encoder Layers: {model_params.get('ENCODER_LAYERS', 'N/A')}",
                f"├─ Decoder Layers: {model_params.get('DECODER_LAYERS', 'N/A')}",
                f"├─ Encoder Heads: {model_params.get('ENCODER_HEADS', 'N/A')}",
                f"└─ Decoder Heads: {model_params.get('DECODER_HEADS', 'N/A')}",
                "",
                "⚙️ TRAINING: Individual encoders",
                "🔀 INFERENCE: Product of Experts",
            ])
        else:
            arch_lines.extend([
                f"🔄 SINGLE ENCODER",
                f"├─ Hidden Dimension: {model_params.get('ENCODER_HIDDEN_DIM', model_params.get('HIDDEN_DIM', 'N/A'))}",
                f"├─ Transformer Layers: {model_params.get('ENCODER_LAYERS', model_params.get('NUM_LAYERS', 'N/A'))}",
                f"└─ Attention Heads: {model_params.get('ENCODER_HEADS', model_params.get('NUM_HEADS', 'N/A'))}",
            ])
        
        ax1.text(0.05, 0.95, "\n".join(arch_lines), transform=ax1.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['arch'], alpha=0.8, edgecolor='gray'))
        
        # 2. TRAINING CONFIGURATION (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        training_keys = model_params.get('TRAINING_KEYS', ['N/A'])
        train_lines = [
            "🎯 TRAINING CONFIG",
            "─" * 20,
            "",
            f"📋 Tasks: {len(training_keys)} tasks",
            f"📊 Examples/Task: {model_params.get('n', 'N/A')}",
            f"🔢 Batch Size: {model_params.get('BATCH_SIZE', 'N/A')}",
            f"🔄 Epochs: {model_params.get('NUM_EPOCHS', 'N/A')}",
            f"📈 Learning Rate: {model_params.get('LEARNING_RATE', 'N/A')}",
            f"⚖️ Beta (KL): {model_params.get('BETA', 'N/A')}",
            f"🎲 Seed: {model_params.get('TRAINING_SEED', 'N/A')}",
            "",
            "🚀 OPTIMIZER",
            f"├─ Type: Adam",
            f"├─ Weight Decay: {model_params.get('optimizer_weight_decay', 0.0)}",
            f"├─ Mixed Precision: {model_params.get('use_mixed_precision', False)}",
            f"└─ Grad Accum: {model_params.get('gradient_accumulation_steps', 1)} steps",
        ]
        
        ax2.text(0.05, 0.95, "\n".join(train_lines), transform=ax2.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['train'], alpha=0.8, edgecolor='gray'))
        
        # 3. LATENT OPTIMIZATION (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        opt_method = model_params.get('OPTIMIZATION_METHOD', 'gradient').title()
        opt_lines = [
            "🔍 LATENT OPTIMIZATION",
            "─" * 25,
            "",
            f"🧠 Method: {opt_method}",
            "",
            "🏋️ TRAINING",
            f"├─ Enabled: {model_params.get('OPTIMIZE_Z', 'N/A')}",
            f"├─ Steps: {model_params.get('OPTIMIZE_Z_NUM_STEPS', 'N/A')}",
            f"└─ LR: {model_params.get('OPTIMIZE_Z_LR', 'N/A')}",
            "",
            "🎯 INFERENCE",
            f"├─ Enabled: {model_params.get('OPTIMIZE_Z_INFERENCE', 'N/A')}",
            f"├─ Steps: {model_params.get('OPTIMIZE_Z_INFERENCE_NUM_STEPS', 'N/A')}",
            f"└─ LR: {model_params.get('OPTIMIZE_Z_INFERENCE_LR', 'N/A')}",
        ]
        
        if 'EVOLUTIONARY_POPULATION_SIZE' in model_params:
            opt_lines.extend([
                "",
                "🧬 EVOLUTIONARY",
                f"├─ Population: {model_params.get('EVOLUTIONARY_POPULATION_SIZE', 'N/A')}",
                f"├─ Generations: {model_params.get('EVOLUTIONARY_NUM_GENERATIONS', 'N/A')}",
                f"└─ Mutation σ: {model_params.get('EVOLUTIONARY_MUTATION_STD', 'N/A')}",
            ])
        
        ax3.text(0.05, 0.95, "\n".join(opt_lines), transform=ax3.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['opt'], alpha=0.8, edgecolor='gray'))
        
        # 4. TRAINING RESULTS (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        
        results_lines = ["📊 TRAINING RESULTS", "─" * 20, ""]
        
        if results is not None:
            try:
                processed_results = extract_multi_encoder_metrics(results)
                is_multi_encoder_results = processed_results.get('is_multi_encoder', False)
                
                if is_multi_encoder_results:
                    results_lines.append(f"🔄 Multi-Encoder Training ({processed_results['num_encoders']} encoders)")
                    
                    if 'multi_encoder_data' in processed_results:
                        multi_data = processed_results['multi_encoder_data']
                        
                        # Final losses per encoder
                        final_losses = []
                        for encoder_idx, losses in multi_data['per_encoder_losses'].items():
                            if losses:
                                final_loss = losses[-1]
                                final_losses.append(final_loss)
                                results_lines.append(f"├─ Encoder {encoder_idx}: Loss {final_loss:.4f}")
                        
                        if len(final_losses) > 1:
                            loss_var = np.var(final_losses)
                            avg_loss = np.mean(final_losses)
                            results_lines.extend([
                                f"├─ Average Loss: {avg_loss:.4f}",
                                f"└─ Loss Variance: {loss_var:.4f} {'(High diversity)' if loss_var > 0.01 else '(Low diversity)'}",
                            ])
                        results_lines.append("")
                
                # Training accuracy results
                if 'epoch_accuracies' in results and results['epoch_accuracies']:
                    last_accuracy = None
                    for acc_data in reversed(results['epoch_accuracies']):
                        if isinstance(acc_data, dict) and 'shape_accuracy' in acc_data:
                            last_accuracy = acc_data
                            break
                    
                    if last_accuracy:
                        exact_acc = last_accuracy.get('sample_exact_accuracy', 0)
                        status = ("🟢 Excellent" if exact_acc >= 0.9 else 
                                 "🟡 Good" if exact_acc >= 0.7 else
                                 "🟠 Moderate" if exact_acc >= 0.5 else "🔴 Needs Work")
                        
                        results_lines.extend([
                            f"🎯 Final Accuracy ({status})",
                            f"├─ Shape: {last_accuracy.get('shape_accuracy', 0):.3f}",
                            f"├─ Grid: {last_accuracy.get('grid_accuracy', 0):.3f}",
                            f"├─ Overall: {last_accuracy.get('overall_accuracy', 0):.3f}",
                            f"└─ Exact: {last_accuracy.get('sample_exact_accuracy', 0):.3f}",
                            ""
                        ])
                
                # Convergence info
                if 'epoch_losses' in results and len(results['epoch_losses']) > 1:
                    initial_loss = results['epoch_losses'][0]
                    final_loss = results['epoch_losses'][-1]
                    improvement = (initial_loss - final_loss) / initial_loss * 100
                    conv_status = ("🟢 Strong" if improvement > 50 else 
                                  "🟡 Moderate" if improvement > 20 else "🔴 Weak")
                    
                    results_lines.extend([
                        f"📈 Convergence ({conv_status})",
                        f"├─ Initial Loss: {initial_loss:.4f}",
                        f"├─ Final Loss: {final_loss:.4f}",
                        f"└─ Improvement: {improvement:.1f}%",
                    ])
            except Exception as e:
                results_lines.extend([
                    f"⚠️ Error processing results: {str(e)[:40]}...",
                    "   Results data may be incomplete"
                ])
        else:
            results_lines.extend([
                "❌ No training results available",
                "   Training may not be completed"
            ])
        
        ax4.text(0.05, 0.95, "\n".join(results_lines), transform=ax4.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['results'], alpha=0.8, edgecolor='gray'))
        
        # 5. EVALUATION STATUS (bottom-center)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        eval_lines = [
            "🔬 EVALUATION STATUS",
            "─" * 22,
            "",
        ]
        
        # Try to load evaluation results if save_dir is provided
        has_eval_results = False
        if save_dir:
            eval_file = os.path.join(save_dir, 'evaluation_results.pkl')
            if os.path.exists(eval_file):
                try:
                    import pickle
                    with open(eval_file, 'rb') as f:
                        eval_results = pickle.load(f)
                    has_eval_results = True
                    
                    eval_lines.extend([
                        f"✅ Evaluation completed",
                        f"📋 Tasks evaluated: {len(eval_results)}",
                        "",
                        "📊 AVERAGE PERFORMANCE:",
                    ])
                    
                    # Calculate averages
                    avg_metrics = {
                        'shape_accuracy': 0,
                        'grid_accuracy': 0,
                        'sample_exact_accuracy': 0,
                        'support_loss': 0,
                        'query_loss': 0
                    }
                    
                    valid_tasks = 0
                    for key, key_results in eval_results.items():
                        if 'metrics' in key_results and 'error' not in key_results['metrics']:
                            metrics = key_results['metrics']
                            for metric_name in avg_metrics.keys():
                                if metric_name in metrics:
                                    avg_metrics[metric_name] += metrics[metric_name]
                            valid_tasks += 1
                    
                    if valid_tasks > 0:
                        eval_lines.extend([
                            f"├─ Shape: {avg_metrics['shape_accuracy']/valid_tasks:.3f}",
                            f"├─ Grid: {avg_metrics['grid_accuracy']/valid_tasks:.3f}",
                            f"├─ Pixel: {avg_metrics['sample_exact_accuracy']/valid_tasks:.3f}",
                            f"├─ Support Loss: {avg_metrics['support_loss']/valid_tasks:.3f}",
                            f"└─ Query Loss: {avg_metrics['query_loss']/valid_tasks:.3f}",
                        ])
                        
                        # Check for multi-encoder specific metrics
                        first_key_results = list(eval_results.values())[0]
                        if first_key_results.get('metrics', {}).get('is_multi_encoder', False):
                            eval_lines.extend([
                                "",
                                "🔄 Multi-encoder analysis available",
                                "   See evaluation plots for details"
                            ])
                    
                except Exception as e:
                    eval_lines.extend([
                        f"⚠️ Error loading eval results",
                        f"   {str(e)[:30]}..."
                    ])
        
        if not has_eval_results:
            eval_lines.extend([
                "❌ No evaluation results found",
                "   Run evaluation to see performance",
                "",
                "📋 CONFIGURATION:",
                f"├─ Eval Keys: {len(model_params.get('DEFAULT_EVAL_KEYS', []))}",
                f"├─ Support Samples: {model_params.get('DEFAULT_EVAL_N_SAMPLES', 'N/A')}",
                f"├─ Query Samples: {model_params.get('DEFAULT_EVAL_N_QUERIES', 'N/A')}",
                f"└─ Eval Seed: {model_params.get('EVAL_SEED', 'N/A')}",
            ])
        
        ax5.text(0.05, 0.95, "\n".join(eval_lines), transform=ax5.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['eval'], alpha=0.8, edgecolor='gray'))
        
        # 6. SYSTEM INFO & FILES (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        sys_lines = [
            "💻 SYSTEM & FILES",
            "─" * 20,
            "",
        ]
        
        # System information
        try:
            import platform
            import sys
            import torch
            import datetime
            
            sys_lines.extend([
                "🖥️ SYSTEM:",
                f"├─ Python: {sys.version.split()[0]}",
                f"├─ PyTorch: {torch.__version__}",
                f"├─ Platform: {platform.system()} {platform.release()}",
                f"└─ CUDA: {torch.cuda.is_available()}",
            ])
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
                # Truncate long GPU names
                if len(gpu_name) > 20:
                    gpu_name = gpu_name[:17] + "..."
                sys_lines.append(f"   GPU: {gpu_name}")
            
            sys_lines.extend([
                "",
                "📁 FILES:",
            ])
            
            if save_dir:
                model_exists = os.path.exists(os.path.join(save_dir, 'model.pth'))
                results_exists = os.path.exists(os.path.join(save_dir, 'results.pkl'))
                eval_exists = os.path.exists(os.path.join(save_dir, 'evaluation_results.pkl'))
                params_exists = os.path.exists(os.path.join(save_dir, 'model_params.json'))
                
                sys_lines.extend([
                    f"├─ Model: {'✓' if model_exists else '✗'}",
                    f"├─ Training: {'✓' if results_exists else '✗'}",
                    f"├─ Evaluation: {'✓' if eval_exists else '✗'}",
                    f"└─ Parameters: {'✓' if params_exists else '✗'}",
                ])
            else:
                sys_lines.append("   Directory not specified")
                
            sys_lines.extend([
                "",
                f"🕒 Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ])
            
        except Exception as e:
            sys_lines.extend([
                f"⚠️ System info error:",
                f"   {str(e)[:30]}..."
            ])
        
        ax6.text(0.05, 0.95, "\n".join(sys_lines), transform=ax6.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['summary'], alpha=0.8, edgecolor='gray'))
        
        # Overall title
        experiment_name = os.path.basename(save_dir) if save_dir else "Unknown Experiment"
        plt.suptitle(f'Enhanced Model Summary: {experiment_name}', fontsize=16, weight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'model_summary.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Enhanced model summary saved to {save_dir}/model_summary.png")
        else:
            plt.show()
            
    except Exception as e:
        print(f"⚠ Warning: Enhanced model summary failed ({str(e)}), falling back to simple summary")
        # Fallback to the simple model summary we created
        plot_model_summary(results, model_params, save_dir)

##############################
# COMPREHENSIVE LATENT SPACE VISUALIZATION
##############################

def plot_comprehensive_latent_space(results, eval_results=None, save_dir=None):
    """
    Create a comprehensive latent space visualization using t-SNE with color-coded clusters.
    Handles both training and evaluation data, single and multi-encoder models.
    
    Args:
        results: Training results containing encoder latent data
        eval_results: Evaluation results containing support/query latent data (optional)
        save_dir: Directory to save the plot (optional)
    """
    print("Creating comprehensive latent space visualization...")
    
    # Determine if this is multi-encoder
    is_multi_encoder = 'encoder_latent_data' in results
    
    # Collect all latent data
    all_latent_data = []
    all_labels = []
    all_colors = []
    legend_elements = []
    
    # Color palette for different data types
    color_palette = {
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
        'training_single': '#9370DB',       # Medium Purple
        'support_single': '#DDA0DD',        # Plum
        'query_single': '#4B0082'           # Indigo
    }
    
    if is_multi_encoder:
        print("  Processing multi-encoder latent data...")
        
        # Process training data from encoders
        if 'encoder_latent_data' in results:
            for encoder_key, encoder_data in results['encoder_latent_data'].items():
                if encoder_data['num_samples'] > 0:
                    latent_z = encoder_data['latent_zs']
                    all_latent_data.append(latent_z)
                    
                    data_type = encoder_data['data_type']
                    color = color_palette.get(data_type, '#808080')
                    all_colors.extend([color] * len(latent_z))
                    all_labels.extend([data_type] * len(latent_z))
                    
                    # Add to legend
                    legend_elements.append(mpatches.Patch(color=color, label=f'{data_type} (n={len(latent_z)})'))
                    print(f"    ✓ Added {len(latent_z)} samples from {data_type}")
        
        # Process evaluation data if available
        if eval_results:
            for key, key_results in eval_results.items():
                if 'evaluation_latent_data' in key_results:
                    eval_data = key_results['evaluation_latent_data']
                    
                    # Process support data
                    if 'support' in eval_data:
                        for encoder_key, encoder_data in eval_data['support'].items():
                            if encoder_data['num_samples'] > 0:
                                latent_z = encoder_data['latent_zs']
                                all_latent_data.append(latent_z)
                                
                                data_type = encoder_data['data_type']
                                color = color_palette.get(data_type, '#808080')
                                all_colors.extend([color] * len(latent_z))
                                all_labels.extend([data_type] * len(latent_z))
                                
                                legend_elements.append(mpatches.Patch(color=color, label=f'{data_type} (n={len(latent_z)})'))
                                print(f"    ✓ Added {len(latent_z)} samples from {data_type}")
                    
                    # Process query data
                    if 'query' in eval_data:
                        for encoder_key, encoder_data in eval_data['query'].items():
                            if encoder_data['num_samples'] > 0:
                                latent_z = encoder_data['latent_zs']
                                all_latent_data.append(latent_z)
                                
                                data_type = encoder_data['data_type']
                                color = color_palette.get(data_type, '#808080')
                                all_colors.extend([color] * len(latent_z))
                                all_labels.extend([data_type] * len(latent_z))
                                
                                legend_elements.append(mpatches.Patch(color=color, label=f'{data_type} (n={len(latent_z)})'))
                                print(f"    ✓ Added {len(latent_z)} samples from {data_type}")
                break  # Only process first key for now
        
    else:
        print("  Processing single-encoder latent data...")
        
        # Process training data
        if 'single_encoder_latent_data' in results:
            latent_data = results['single_encoder_latent_data']
            if latent_data['num_samples'] > 0:
                latent_z = latent_data['latent_zs']
                all_latent_data.append(latent_z)
                
                data_type = 'training_single'
                color = color_palette[data_type]
                all_colors.extend([color] * len(latent_z))
                all_labels.extend([data_type] * len(latent_z))
                
                legend_elements.append(mpatches.Patch(color=color, label=f'Training (n={len(latent_z)})'))
                print(f"    ✓ Added {len(latent_z)} samples from training")
        
        # Process evaluation data if available
        if eval_results:
            for key, key_results in eval_results.items():
                if 'evaluation_latent_data' in key_results:
                    eval_data = key_results['evaluation_latent_data']
                    
                    # Process support data
                    if 'support' in eval_data and 'single_encoder' in eval_data['support']:
                        encoder_data = eval_data['support']['single_encoder']
                        if encoder_data['num_samples'] > 0:
                            latent_z = encoder_data['latent_zs']
                            all_latent_data.append(latent_z)
                            
                            data_type = 'support_single'
                            color = color_palette[data_type]
                            all_colors.extend([color] * len(latent_z))
                            all_labels.extend([data_type] * len(latent_z))
                            
                            legend_elements.append(mpatches.Patch(color=color, label=f'Support (n={len(latent_z)})'))
                            print(f"    ✓ Added {len(latent_z)} samples from support")
                    
                    # Process query data
                    if 'query' in eval_data and 'single_encoder' in eval_data['query']:
                        encoder_data = eval_data['query']['single_encoder']
                        if encoder_data['num_samples'] > 0:
                            latent_z = encoder_data['latent_zs']
                            all_latent_data.append(latent_z)
                            
                            data_type = 'query_single'
                            color = color_palette[data_type]
                            all_colors.extend([color] * len(latent_z))
                            all_labels.extend([data_type] * len(latent_z))
                            
                            legend_elements.append(mpatches.Patch(color=color, label=f'Query (n={len(latent_z)})'))
                            print(f"    ✓ Added {len(latent_z)} samples from query")
                break  # Only process first key for now
    
    if not all_latent_data:
        print("⚠ Warning: No latent data found for visualization")
        return
    
    # Combine all latent data
    combined_latents = np.concatenate(all_latent_data, axis=0)
    print(f"  Total latent samples for visualization: {len(combined_latents)}")
    print(f"  Latent dimensionality: {combined_latents.shape[1]}")
    
    # Apply t-SNE for dimensionality reduction
    print("  Applying t-SNE dimensionality reduction...")
    
    # Use PCA preprocessing if dimensionality is very high
    if combined_latents.shape[1] > 50:
        print("    Applying PCA preprocessing (high dimensionality detected)...")
        pca = PCA(n_components=50, random_state=42)
        combined_latents = pca.fit_transform(combined_latents)
        print(f"    Reduced to {combined_latents.shape[1]} dimensions with PCA")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_latents)//4))
    tsne_results = tsne.fit_transform(combined_latents)
    print("    ✓ t-SNE completed")
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create scatter plot with color coding
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                        c=all_colors, alpha=0.6, s=20, edgecolors='black', linewidth=0.1)
    
    # Customize the plot
    model_type = "Multi-Encoder" if is_multi_encoder else "Single Encoder"
    title = f'Comprehensive Latent Space Visualization ({model_type})'
    if eval_results:
        title += '\nTraining + Evaluation Data'
    else:
        title += '\nTraining Data Only'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    # Add summary text
    total_samples = len(combined_latents)
    unique_types = len(set(all_labels))
    summary_text = f'Total Samples: {total_samples}\nData Types: {unique_types}'
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        filename = 'comprehensive_latent_space.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive latent space visualization saved to {save_dir}/{filename}")
    else:
        plt.show()

def get_comprehensive_latent_data_for_trajectory(run_dir):
    """
    Get comprehensive latent data (encoders + PoE) for trajectory visualization background.
    This reuses the same logic as plot_comprehensive_latent_space but returns the data
    instead of creating a plot.
    
    Args:
        run_dir: Directory containing training and evaluation results
        
    Returns:
        tuple: (combined_latents, tsne_2d, labels, colors) or (None, None, None, None) if no data
    """
    try:
        # Load training results
        results_file = os.path.join(run_dir, 'results.pkl')
        results = None
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
        
        # Load evaluation results  
        eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
        eval_results = None
        if os.path.exists(eval_file):
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
        
        if not results and not eval_results:
            return None, None, None, None
        
        # Use the same logic as plot_comprehensive_latent_space
        is_multi_encoder = results and 'encoder_latent_data' in results
        
        # Collect all latent data
        all_latent_data = []
        all_labels = []
        all_colors = []
        
        # Color palette for different data types
        color_palette = {
            'training_encoder_0': '#FF6B6B',    'training_encoder_1': '#4ECDC4',
            'training_encoder_2': '#45B7D1',    'training_encoder_3': '#96CEB4',
            'support_encoder_0': '#FFB6C1',     'support_encoder_1': '#B4E7E1',
            'support_encoder_2': '#B3D9FF',     'support_encoder_3': '#C8E6C9',
            'query_encoder_0': '#8B0000',       'query_encoder_1': '#006666',
            'query_encoder_2': '#0066CC',       'query_encoder_3': '#2E8B57',
            'support_poe': '#FFD700',           'query_poe': '#FF8C00',
            'training_single': '#9370DB',       'support_single': '#DDA0DD',
            'query_single': '#4B0082'
        }
        
        if is_multi_encoder and results:
            # Process training data from encoders
            if 'encoder_latent_data' in results:
                for encoder_key, encoder_data in results['encoder_latent_data'].items():
                    if encoder_data['num_samples'] > 0:
                        latent_z = encoder_data['latent_zs']
                        all_latent_data.append(latent_z)
                        
                        data_type = encoder_data['data_type']
                        color = color_palette.get(data_type, '#808080')
                        all_colors.extend([color] * len(latent_z))
                        all_labels.extend([data_type] * len(latent_z))
            
            # Process evaluation data if available
            if eval_results:
                for key, key_results in eval_results.items():
                    if 'evaluation_latent_data' in key_results:
                        eval_data = key_results['evaluation_latent_data']
                        
                        # Process support and query data
                        for phase in ['support', 'query']:
                            if phase in eval_data:
                                for encoder_key, encoder_data in eval_data[phase].items():
                                    if encoder_data['num_samples'] > 0:
                                        latent_z = encoder_data['latent_zs']
                                        all_latent_data.append(latent_z)
                                        
                                        data_type = encoder_data['data_type']
                                        color = color_palette.get(data_type, '#808080')
                                        all_colors.extend([color] * len(latent_z))
                                        all_labels.extend([data_type] * len(latent_z))
                    break  # Only process first key
        
        elif results:  # Single encoder
            # Process training data
            if 'single_encoder_latent_data' in results:
                latent_data = results['single_encoder_latent_data']
                if latent_data['num_samples'] > 0:
                    latent_z = latent_data['latent_zs']
                    all_latent_data.append(latent_z)
                    
                    data_type = 'training_single'
                    color = color_palette[data_type]
                    all_colors.extend([color] * len(latent_z))
                    all_labels.extend([data_type] * len(latent_z))
            
            # Process evaluation data if available
            if eval_results:
                for key, key_results in eval_results.items():
                    if 'evaluation_latent_data' in key_results:
                        eval_data = key_results['evaluation_latent_data']
                        
                        # Process support and query data
                        for phase in ['support', 'query']:
                            if phase in eval_data and 'single_encoder' in eval_data[phase]:
                                encoder_data = eval_data[phase]['single_encoder']
                                if encoder_data['num_samples'] > 0:
                                    latent_z = encoder_data['latent_zs']
                                    all_latent_data.append(latent_z)
                                    
                                    data_type = f'{phase}_single'
                                    color = color_palette[data_type]
                                    all_colors.extend([color] * len(latent_z))
                                    all_labels.extend([data_type] * len(latent_z))
                    break
        
        if not all_latent_data:
            return None, None, None, None
        
        # Combine all latent data
        combined_latents = np.concatenate(all_latent_data, axis=0)
        
        # Apply dimensionality reduction for 2D visualization
        if combined_latents.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            combined_latents_reduced = pca.fit_transform(combined_latents)
        else:
            combined_latents_reduced = combined_latents
        
        # Apply t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_latents)//4), max_iter=1000)
        tsne_2d = tsne.fit_transform(combined_latents_reduced)
        
        return combined_latents, tsne_2d, all_labels, all_colors
        
    except Exception as e:
        print(f"⚠ Warning: Could not load comprehensive latent data: {e}")
        return None, None, None, None
