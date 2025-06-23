from utils.data_preparation import transform_grid_to_sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from tabulate import tabulate
import os
import torch
from sklearn.manifold import TSNE
import pickle
from utils.settings_manager import settings

# Get settings from settings manager
evaluation_settings = settings.get_evaluation_settings()
DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']

##############################
# MULTI-ENCODER SUPPORT FUNCTIONS
##############################

def extract_multi_encoder_metrics(results):
    """
    Extract and process multi-encoder training metrics for visualization.
    Converts multi-encoder metrics structure to be compatible with existing visualizers.
    
    Args:
        results: Training results dictionary
        
    Returns:
        dict: Processed metrics for visualization
    """
    if 'epoch_metrics' not in results or not results['epoch_metrics']:
        return results
    
    # Check if this is multi-encoder training
    first_epoch_metrics = results['epoch_metrics'][0]
    is_multi_encoder = 'multi_encoder_metrics' in first_epoch_metrics
    
    if not is_multi_encoder:
        return results  # No processing needed for single encoder
    
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
    Plot detailed multi-encoder training losses showing per-encoder and aggregated metrics.
    
    Args:
        results: Processed results dictionary with multi-encoder data
        save_dir: Directory to save plots (optional)
    """
    if not results.get('is_multi_encoder', False):
        print("Not a multi-encoder model, skipping multi-encoder loss plots")
        return
    
    multi_encoder_data = results['multi_encoder_data']
    epochs = multi_encoder_data['epochs']
    num_encoders = results['num_encoders']
    
    # Create subplots for different loss components
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for different encoders
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    # Plot total losses per encoder
    axs[0, 0].set_title('Total Loss per Encoder', fontsize=14)
    for i, (encoder_idx, losses) in enumerate(multi_encoder_data['per_encoder_losses'].items()):
        axs[0, 0].plot(epochs, losses, marker='o', label=f'Encoder {encoder_idx}', 
                      color=colors[i], linewidth=2)
    axs[0, 0].plot(epochs, results['epoch_losses'], 'k--', linewidth=3, 
                  label='Average', alpha=0.8)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot shape losses per encoder
    axs[0, 1].set_title('Shape Loss per Encoder', fontsize=14)
    for i, (encoder_idx, losses) in enumerate(multi_encoder_data['per_encoder_shape_losses'].items()):
        axs[0, 1].plot(epochs, losses, marker='s', label=f'Encoder {encoder_idx}', 
                      color=colors[i], linewidth=2)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Shape Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot grid losses per encoder
    axs[1, 0].set_title('Grid Loss per Encoder', fontsize=14)
    for i, (encoder_idx, losses) in enumerate(multi_encoder_data['per_encoder_grid_losses'].items()):
        axs[1, 0].plot(epochs, losses, marker='^', label=f'Encoder {encoder_idx}', 
                      color=colors[i], linewidth=2)
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Grid Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot KL losses per encoder
    axs[1, 1].set_title('KL Loss per Encoder', fontsize=14)
    for i, (encoder_idx, losses) in enumerate(multi_encoder_data['per_encoder_kl_losses'].items()):
        axs[1, 1].plot(epochs, losses, marker='d', label=f'Encoder {encoder_idx}', 
                      color=colors[i], linewidth=2)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('KL Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Encoder Training Losses ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multi_encoder_training_losses.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder training losses plot saved to {save_dir}/multi_encoder_training_losses.png")
    else:
        plt.show()

def plot_multi_encoder_summary(results, save_dir=None):
    """
    Plot a summary view of multi-encoder training with key statistics.
    
    Args:
        results: Processed results dictionary with multi-encoder data
        save_dir: Directory to save plots (optional)
    """
    if not results.get('is_multi_encoder', False):
        print("Not a multi-encoder model, skipping multi-encoder summary")
        return
    
    multi_encoder_data = results['multi_encoder_data']
    epochs = multi_encoder_data['epochs']
    num_encoders = results['num_encoders']
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Final loss comparison across encoders
    final_losses = []
    encoder_labels = []
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders))
    
    for encoder_idx, losses in multi_encoder_data['per_encoder_losses'].items():
        final_losses.append(losses[-1])  # Last epoch loss
        encoder_labels.append(f'Encoder {encoder_idx}')
    
    bars = axs[0].bar(encoder_labels, final_losses, color=colors[:len(final_losses)])
    axs[0].set_title('Final Loss per Encoder', fontsize=14)
    axs[0].set_ylabel('Final Loss')
    axs[0].set_xticklabels(encoder_labels, rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.3f}', ha='center', va='bottom')
    
    # 2. Loss variance across encoders over time
    loss_variances = []
    for epoch_idx in range(len(epochs)):
        epoch_losses = [multi_encoder_data['per_encoder_losses'][enc_idx][epoch_idx] 
                       for enc_idx in multi_encoder_data['per_encoder_losses'].keys()]
        loss_variances.append(np.var(epoch_losses))
    
    axs[1].plot(epochs, loss_variances, marker='o', linewidth=2, color='red')
    axs[1].set_title('Loss Variance Across Encoders', fontsize=14)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss Variance')
    axs[1].grid(True, alpha=0.3)
    
    # 3. Training progress comparison
    axs[2].set_title('Training Progress: All vs Average', fontsize=14)
    
    # Plot individual encoder losses (thin lines)
    for i, (encoder_idx, losses) in enumerate(multi_encoder_data['per_encoder_losses'].items()):
        axs[2].plot(epochs, losses, color=colors[i], alpha=0.6, linewidth=1, 
                   label=f'Encoder {encoder_idx}')
    
    # Plot average loss (thick line)
    axs[2].plot(epochs, results['epoch_losses'], 'k-', linewidth=3, 
               label='Average', alpha=0.9)
    
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Encoder Training Summary ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multi_encoder_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder summary plot saved to {save_dir}/multi_encoder_summary.png")
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
                if 'encoded_training_latents' in results:
                    if return_all_components:
                        embedded_data = results['encoded_training_latents']
                        all_components = {
                            'latent_mus': np.array(embedded_data['latent_mus']),
                            'latent_log_vars': np.array(embedded_data.get('latent_log_vars', [])),
                            'latent_zs': np.array(embedded_data.get('latent_zs', [])),
                            'encoding_info': embedded_data.get('encoding_info', {})
                        }
                        
                        print(f"✓ Found all latent components in evaluation results for key '{key}'")
                        print(f"  - Means: {len(all_components['latent_mus'])} samples")
                        print(f"  - Log-vars: {len(all_components['latent_log_vars'])} samples")
                        print(f"  - Sampled Z: {len(all_components['latent_zs'])} samples")
                        
                        return all_components
                    else:
                        latent_mus = np.array(results['encoded_training_latents']['latent_mus'])
                        encoding_info = results['encoded_training_latents'].get('encoding_info', {})
                        
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
        print("  Consider running evaluation to generate fresh encoded latents for better accuracy.")
        
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

def plot_training_and_latent(results, save_dir=None):
    """Plot training loss and latent space visualization using evaluation-generated latents."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Subplot 0: Log(Training Loss) Over Epochs
    epoch_losses = np.array(results['epoch_losses'])
    log_losses = np.log(epoch_losses + 1e-8)
    axs[0].plot(log_losses, marker='o', color='tab:blue', linewidth=2)
    axs[0].set_xlabel('Epoch', fontsize=16)
    axs[0].set_ylabel('log(Training Loss)', fontsize=16)
    axs[0].grid(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # Subplot 1: Latent Space Visualization via t-SNE
    # Try to load latents from evaluation first, then fall back to legacy
    if save_dir:
        # When called from visualize_stored_results, save_dir is the run_dir
        run_dir = save_dir
    else:
        # Fallback: try to infer run_dir (this might not always work)
        run_dir = os.getcwd()
    
    # Load evaluation-generated latents
    all_mus = load_evaluation_latent_data(run_dir)
    
    # Fallback to legacy latents if evaluation latents not found
    if all_mus is None:
        all_mus = load_legacy_latent_data(run_dir)
    
    # Final fallback to results latent_mus (original approach)
    if all_mus is None and 'latent_mus' in results and results['latent_mus']:
        print("⚠ Falling back to legacy results processing...")
        
        def process_latent_var(latent_var_list, var_name):
            try:
                if not latent_var_list:
                    return None
                if isinstance(latent_var_list, dict):
                    combined_data = []
                    for key, value in latent_var_list.items():
                        if value:
                            combined_data.extend(value)
                    if not combined_data:
                        return None
                    latent_var_list = combined_data
                
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
                
        all_mus = process_latent_var(results.get('latent_mus', []), 'latent_mus')
    
    if all_mus is None or len(all_mus) < 2:
        print("⚠ Warning: No valid latent means found for t-SNE visualization.")
        axs[1].text(0.5, 0.5, 'No latent data available\n\nRun evaluation to generate\nencoded training latents', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=12)
        axs[1].set_title('Latent space (t-SNE)', fontsize=18)
        axs[1].axis('off')
    else:
        # Choose appropriate perplexity based on number of samples
        perplexity = min(30, max(1, len(all_mus) - 1))
        
        # Apply t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            latent_2d = tsne.fit_transform(all_mus)
            sc1 = axs[1].scatter(latent_2d[:, 0], latent_2d[:, 1],
                                c=np.arange(len(latent_2d)), cmap='viridis', alpha=0.8, s=80)
            axs[1].set_title('Latent space (t-SNE)', fontsize=18)
            axs[1].set_xlabel('Dimension 1', fontsize=16)
            axs[1].set_ylabel('Dimension 2', fontsize=16)
            plt.colorbar(sc1, ax=axs[1], label='Sample Index', pad=0.02)
            axs[1].grid(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            
            print(f"✓ t-SNE visualization created with {len(all_mus)} samples")
        except Exception as e:
            print(f"⚠ Warning: Error during t-SNE: {e}")
            axs[1].text(0.5, 0.5, f"t-SNE Error:\n{str(e)}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axs[1].transAxes)
            axs[1].set_title('Latent space (t-SNE)', fontsize=18)
            axs[1].axis('off')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_and_latent.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training and latent plot saved to {save_dir}/training_and_latent.png")
    else:
        plt.show()

def plot_latent_analysis(results, save_dir=None):
    """Plot detailed latent space analysis using fresh encoded training latents from trained model."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Try to load evaluation-generated latents
    if save_dir:
        run_dir = save_dir
    else:
        run_dir = os.getcwd()
    
    # Load ALL latent components from evaluation-generated encoded training latents
    print("✓ Using fresh encoded training latents from trained model for ALL latent analysis")
    encoded_latents = load_evaluation_latent_data(run_dir, return_all_components=True)
    
    if encoded_latents is not None:
        latent_mu = encoded_latents['latent_mus']
        latent_log_var = encoded_latents['latent_log_vars']
        latent_z = encoded_latents['latent_zs']
        data_source = "fresh encoded (trained model)"
    else:
        print("⚠ WARNING: No fresh encoded latents found, falling back to legacy training data")
        
        # Fallback to legacy data processing
        def process_latent_var(latent_var_list, var_name):
            try:
                if not latent_var_list:
                    return None
                
                if isinstance(latent_var_list, dict):
                    combined_data = []
                    for key, value in latent_var_list.items():
                        if value:
                            combined_data.extend(value)
                    if not combined_data:
                        return None
                    latent_var_list = combined_data
                
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
        
        latent_mu = process_latent_var(results.get('latent_mus', []), 'latent_mus')
        latent_log_var = process_latent_var(results.get('latent_log_vars', []), 'latent_log_vars')
        latent_z = process_latent_var(results.get('latent_zs', []), 'latent_zs')
        data_source = "legacy training data"
    
    # Count how many valid latent variables we have
    valid_count = sum(1 for var in [latent_mu, latent_log_var, latent_z] if var is not None)
    
    if valid_count == 0:
        print("⚠ Warning: No valid latent variables found for analysis.")
        for i, title in enumerate(['Mean (μ)', 'Log-Variance (log σ²)', 'Sampled Z']):
            axs[i].text(0.5, 0.5, f"No data for {title}\n\nRun evaluation to generate\nencoded training latents", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axs[i].transAxes, fontsize=12)
            axs[i].set_title(f't-SNE {title}', fontsize=18)
            axs[i].axis('off')
    else:
        # Function to create t-SNE plot with appropriate perplexity
        def create_tsne_plot(data, ax, title):
            if data is None or len(data) < 2:
                ax.text(0.5, 0.5, f"Insufficient data for {title}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(title, fontsize=18)
                ax.axis('off')
                return
                
            # Choose appropriate perplexity based on number of samples
            perplexity = min(30, max(1, len(data) - 1))
            
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_dims = tsne.fit_transform(data)
                sc = ax.scatter(tsne_dims[:, 0], tsne_dims[:, 1],
                               c=np.arange(tsne_dims.shape[0]), cmap='viridis', alpha=0.8, s=80)
                ax.set_title(title, fontsize=18)
                ax.set_xlabel('Dimension 1', fontsize=16)
                ax.set_ylabel('Dimension 2', fontsize=16)
                plt.colorbar(sc, ax=ax, label='Sample Index', pad=0.02)
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add data source info
                ax.text(0.02, 0.98, f"Source: {data_source}", transform=ax.transAxes, 
                       fontsize=8, alpha=0.7, verticalalignment='top')
                
            except Exception as e:
                print(f"⚠ Warning: Error in t-SNE for {title}: {e}")
                ax.text(0.5, 0.5, f"t-SNE Error:\n{str(e)}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(title, fontsize=18)
                ax.axis('off')
        
        # Create the three plots using fresh encoded latents
        create_tsne_plot(latent_mu, axs[0], 't-SNE Mean (μ)')
        create_tsne_plot(latent_log_var, axs[1], 't-SNE Log-Variance (log σ²)')
        create_tsne_plot(latent_z, axs[2], 't-SNE Sampled Z')

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'latent_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Latent analysis plot saved to {save_dir}/latent_analysis.png")
    else:
        plt.show()

def plot_epoch_accuracies(results, save_dir=None):
    """Plot shape, grid, overall, and sample-level exact accuracy over epochs."""
    # Extract epoch numbers and accuracy metrics from the results.
    epoch_nums = [acc["epoch"] for acc in results["epoch_accuracies"]]
    shape_acc = [acc["shape_accuracy"] for acc in results["epoch_accuracies"]]
    grid_acc = [acc["grid_accuracy"] for acc in results["epoch_accuracies"]]
    overall_acc = [acc["overall_accuracy"] for acc in results["epoch_accuracies"]]
    sample_acc = [acc["sample_exact_accuracy"] for acc in results["epoch_accuracies"]]

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_nums, shape_acc, marker='o', linestyle='-', label="Shape Accuracy")
    plt.plot(epoch_nums, grid_acc, marker='s', linestyle='-', label="Grid Accuracy")
    plt.plot(epoch_nums, overall_acc, marker='^', linestyle='-', label="Overall Pixel Accuracy")
    plt.plot(epoch_nums, sample_acc, marker='d', linestyle='-', label="Sample-Level Exact Accuracy")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Accuracy Over Training Epochs", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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


def visualize_all_results(results, save_dir=None):
    """Plot all visualizations for the results."""
    
    # First, check if this is multi-encoder training and process accordingly
    processed_results = extract_multi_encoder_metrics(results)
    
    # Check if this is multi-encoder training
    is_multi_encoder = processed_results.get('is_multi_encoder', False)
    
    if is_multi_encoder:
        print("\nDetected multi-encoder training results")
        print(f"Number of encoders: {processed_results['num_encoders']}")
        
        # Plot multi-encoder specific visualizations
        print("\nPlotting multi-encoder training losses...")
        plot_multi_encoder_training_losses(processed_results, save_dir)
        
        print("\nPlotting multi-encoder summary...")
        plot_multi_encoder_summary(processed_results, save_dir)
        
        # Plot multi-encoder accuracy visualizations
        print("\nPlotting multi-encoder accuracies...")
        plot_multi_encoder_accuracies(processed_results, save_dir)
        
        print("\nPlotting multi-encoder accuracy summary...")
        plot_multi_encoder_accuracy_summary(processed_results, save_dir)
    
    print("\nPlotting training progress and latent space...")
    plot_training_and_latent(processed_results, save_dir)

    print("\nPlotting latent space analysis...")
    plot_latent_analysis(processed_results, save_dir)

    print("\nPlotting epoch accuracies over time...")
    plot_epoch_accuracies(processed_results, save_dir)

    if 'losses_gradient_ascent' in processed_results:
        if len(processed_results['losses_gradient_ascent']) > 0:
            print("\nPlotting z optimization losses...")
            plot_z_optimization_losses(processed_results, save_dir)

    # print("\nPlotting reconstructions...")
    # plot_reconstructions(processed_results, save_dir)


def plot_evaluation_results(results, save_dir=None):
    """
    Plot evaluation metrics and reconstructions for each key.
    
    Args:
        results: Dictionary containing evaluation results for each key
        save_dir: Directory to save plots (optional)
    """
    # Plot metrics
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Define metrics to plot
    metrics = [
        'support_loss', 'query_loss',  # Loss metrics
        'shape_accuracy', 'grid_accuracy',  # Accuracy metrics
        'overall_accuracy', 'sample_exact_accuracy'  # Additional metrics
    ]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = []
        keys = []
        for key in results:
            if 'metrics' in results[key] and metric in results[key]['metrics']:
                values.append(results[key]['metrics'][metric])
                keys.append(key)
        
        axs[i].bar(keys, values)
        axs[i].set_title(f'{metric.replace("_", " ").title()}')
        axs[i].set_xticklabels(keys, rotation=45)
        
        # Set y-axis limits for accuracy metrics
        if 'accuracy' in metric:
            axs[i].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Evaluation metrics plot saved to {save_dir}/evaluation_metrics.png")
    else:
        plt.show()
    
    # Print detailed metrics for each key
    print("\nDetailed Evaluation Results:")
    for key in results:
        if 'metrics' in results[key]:
            print(f"\nKey {key}:")
            metrics = results[key]['metrics']
            print(f"  Support Loss: {metrics['support_loss']:.4f}")
            print(f"  Query Loss: {metrics['query_loss']:.4f}")
            print(f"  Shape Accuracy: {metrics['shape_accuracy']:.4f}")
            print(f"  Grid Accuracy: {metrics['grid_accuracy']:.4f}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print(f"  Sample Exact Accuracy: {metrics['sample_exact_accuracy']:.4f}")

            if 'losses_gradient_ascent' in results[key]['metrics']:
                print(f"\nPlotting z optimization losses for key {key}")
                plot_z_optimization_losses(results[key]['metrics'], save_dir)        

        if 'reconstruction_results' in results[key]:
            print(f"\nPlotting support reconstructions for key {key}")
            aux = {
                'input_sequences': results[key]['reconstruction_results']['input_samples_sequences'],
                'output_sequences': results[key]['reconstruction_results']['output_samples_sequences'],
                'reconstructions': results[key]['reconstruction_results']['support_reconstructions'],
            }
            plot_reconstructions(aux, save_dir, f"{key}_support_")

            print(f"\nPlotting query reconstructions for key {key}")
            aux = {
                'input_sequences': results[key]['reconstruction_results']['input_queries_sequences'],
                'output_sequences': results[key]['reconstruction_results']['output_queries_sequences'],
                'reconstructions': results[key]['reconstruction_results']['query_reconstructions'],
            }
            plot_reconstructions(aux, save_dir, f"{key}_query_")


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
                'HIDDEN_DIM': model_architecture.get('hidden_dim', 128),
                'NUM_LAYERS': model_architecture.get('num_layers', 2),
                'NUM_HEADS': model_architecture.get('num_heads', 4),
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
                'HIDDEN_DIM': 128,
                'NUM_LAYERS': 2,
                'NUM_HEADS': 4,
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
    
    # Plot model summary if we have parameters (even if no training results)
    if model_params is not None:
        print("\nPlotting model summary...")
        try:
            plot_model_summary(results, model_params, run_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not plot model summary: {e}")
    
    # Visualize training results if available
    if results is not None:
        print("\nVisualizing training results...")
        try:
            # Check if this is multi-encoder training and add appropriate info
            processed_results = extract_multi_encoder_metrics(results)
            is_multi_encoder = processed_results.get('is_multi_encoder', False)
            
            if is_multi_encoder:
                print(f"Detected multi-encoder training with {processed_results['num_encoders']} encoders")
            
            visualize_all_results(results, run_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not visualize training results: {e}")
    else:
        print("Skipping training results visualization (no training results available)")
    
    # Try to load and visualize evaluation results
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if os.path.exists(eval_file):
        print("\nFound evaluation results file, loading...")
        try:
            with open(eval_file, 'rb') as f:
                eval_results = pickle.load(f)
            print("✓ Evaluation results loaded successfully")
            
            print("Visualizing evaluation results...")
            plot_evaluation_results(eval_results, run_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not load/visualize evaluation results: {e}")
    else:
        print("No evaluation results found (evaluation_results.pkl)")
    
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

def plot_model_summary(results, model_params, save_dir=None):
    """
    Plot a summary of model parameters and results.
    
    Args:
        results: Dictionary containing training and evaluation results (can be None)
        model_params: Dictionary containing model parameters
        save_dir: Directory to save the plot (optional)
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Create text box with model parameters
    param_text = "Model Parameters:\n\n"
    param_text += f"Training Keys: {', '.join(model_params.get('TRAINING_KEYS', ['None']))}\n"
    param_text += f"Training Seed: {model_params.get('TRAINING_SEED', 'N/A')}\n"
    param_text += f"Evaluation Seed: {model_params.get('EVAL_SEED', 'N/A')}\n\n"
    
    param_text += "Architecture:\n"
    param_text += f"Latent Dimension: {model_params.get('LATENT_DIM', 'N/A')}\n"
    param_text += f"Hidden Dimension: {model_params.get('HIDDEN_DIM', 'N/A')}\n"
    param_text += f"Number of Layers: {model_params.get('NUM_LAYERS', 'N/A')}\n"
    param_text += f"Number of Heads: {model_params.get('NUM_HEADS', 'N/A')}\n"
    param_text += f"Dropout: {model_params.get('DROPOUT', 'N/A')}\n"
    param_text += f"Max Length: {model_params.get('MAX_LENGTH', 'N/A')}\n"
    param_text += f"Encoder Max Length: {model_params.get('ENCODER_MAX_LENGTH', 'N/A')}\n"
    param_text += f"Decoder Max Length: {model_params.get('DECODER_MAX_LENGTH', 'N/A')}\n"
    
    # Add multi-encoder information if available
    num_encoders = model_params.get('NUM_ENCODERS', 1)
    if num_encoders > 1:
        param_text += f"Number of Encoders: {num_encoders} (Multi-Encoder Mode)\n"
        param_text += f"Training Strategy: Individual encoder training with shared latent space\n"
        param_text += f"Inference Strategy: Product of Experts (PoE)\n"
    else:
        param_text += f"Number of Encoders: {num_encoders} (Single-Encoder Mode)\n"
    
    param_text += "\n"
    
    param_text += "Training Settings:\n"
    param_text += f"Batch Size: {model_params.get('BATCH_SIZE', 'N/A')}\n"
    param_text += f"Number of Epochs: {model_params.get('NUM_EPOCHS', 'N/A')}\n"
    param_text += f"Learning Rate: {model_params.get('LEARNING_RATE', 'N/A')}\n"
    param_text += f"Beta (KL Loss): {model_params.get('BETA', 'N/A')}\n"
    param_text += f"Training Examples per Batch: {model_params.get('n', 'N/A')}\n\n"
    
    param_text += "Latent Optimization:\n"
    
    # Get method if available, otherwise default to "gradient"
    optimization_method = model_params.get('OPTIMIZATION_METHOD', 'gradient')
    param_text += f"Optimization Method: {optimization_method}\n\n"
    
    param_text += "Training Optimization:\n"
    param_text += f"  Enabled: {model_params.get('OPTIMIZE_Z', 'N/A')}\n"
    param_text += f"  Steps: {model_params.get('OPTIMIZE_Z_NUM_STEPS', 'N/A')}\n"
    param_text += f"  Learning Rate: {model_params.get('OPTIMIZE_Z_LR', 'N/A')}\n\n"
    
    param_text += "Inference Optimization:\n"
    param_text += f"  Enabled: {model_params.get('OPTIMIZE_Z_INFERENCE', 'N/A')}\n"
    param_text += f"  Steps: {model_params.get('OPTIMIZE_Z_INFERENCE_NUM_STEPS', 'N/A')}\n"
    param_text += f"  Learning Rate: {model_params.get('OPTIMIZE_Z_INFERENCE_LR', 'N/A')}\n\n"
    
    # Add evolutionary parameters if available
    if 'EVOLUTIONARY_POPULATION_SIZE' in model_params:
        param_text += "Evolutionary Settings:\n"
        param_text += f"  Population Size: {model_params.get('EVOLUTIONARY_POPULATION_SIZE', 'N/A')}\n"
        param_text += f"  Number of Generations: {model_params.get('EVOLUTIONARY_NUM_GENERATIONS', 'N/A')}\n"
        param_text += f"  Mutation Std: {model_params.get('EVOLUTIONARY_MUTATION_STD', 'N/A')}\n\n"
    
    # Add Voronoi parameters if available
    if 'VORONOI_POPULATION_SIZE' in model_params:
        param_text += "Voronoi Search Settings:\n"
        param_text += f"  Population Size: {model_params.get('VORONOI_POPULATION_SIZE', 'N/A')}\n"
        param_text += f"  Number of Generations: {model_params.get('VORONOI_NUM_GENERATIONS', 'N/A')}\n"
        param_text += f"  Diversity Weight: {model_params.get('VORONOI_DIVERSITY_WEIGHT', 'N/A')}\n"
        param_text += f"  Mutation Std: {model_params.get('VORONOI_MUTATION_STD', 'N/A')}\n\n"
    
    param_text += "Evaluation Settings:\n"
    param_text += f"Eval Keys: {model_params.get('DEFAULT_EVAL_KEYS', 'N/A')}\n"
    param_text += f"Eval Samples: {model_params.get('DEFAULT_EVAL_N_SAMPLES', 'N/A')}\n"
    param_text += f"Eval Queries: {model_params.get('DEFAULT_EVAL_N_QUERIES', 'N/A')}\n"
    param_text += f"Eval Epoch: {model_params.get('DEFAULT_EVAL_EPOCH', 'N/A')}\n"
    
    # Add parameters text box
    plt.subplot(1, 2, 1)
    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.axis('off')
    
    # Add results summary
    plt.subplot(1, 2, 2)
    results_text = "Results Summary:\n\n"
    
    if results is not None:
        # Check for multi-encoder training
        processed_results = extract_multi_encoder_metrics(results)
        is_multi_encoder = processed_results.get('is_multi_encoder', False)
        
        if is_multi_encoder:
            results_text += f"Multi-Encoder Training Results ({processed_results['num_encoders']} encoders):\\n\\n"
            
            # Show final losses for each encoder
            if 'multi_encoder_data' in processed_results:
                multi_data = processed_results['multi_encoder_data']
                results_text += "Final Loss per Encoder:\\n"
                for encoder_idx, losses in multi_data['per_encoder_losses'].items():
                    final_loss = losses[-1] if losses else 0.0
                    results_text += f"  Encoder {encoder_idx}: {final_loss:.4f}\\n"
                
                # Average final loss
                avg_final_loss = processed_results['epoch_losses'][-1] if processed_results['epoch_losses'] else 0.0
                results_text += f"  Average: {avg_final_loss:.4f}\\n\\n"
        
        # Training results (accuracy is same for both single and multi-encoder)
        if 'epoch_accuracies' in results and results['epoch_accuracies']:
            last_epoch = results['epoch_accuracies'][-1]
            if is_multi_encoder:
                results_text += "Final Evaluation Results (PoE Inference):\\n"
            else:
                results_text += "Training Results (Last Epoch):\\n"
            results_text += f"Shape Accuracy: {last_epoch['shape_accuracy']:.4f}\\n"
            results_text += f"Grid Accuracy: {last_epoch['grid_accuracy']:.4f}\\n"
            results_text += f"Overall Accuracy: {last_epoch['overall_accuracy']:.4f}\\n"
            results_text += f"Sample Exact Accuracy: {last_epoch['sample_exact_accuracy']:.4f}\\n\\n"
        
        # Evaluation results (if embedded in training results)
        if 'evaluation_results' in results:
            results_text += "Evaluation Results:\\n"
            for key in results['evaluation_results']:
                metrics = results['evaluation_results'][key]['metrics']
                results_text += f"\\nKey {key}:\\n"
                results_text += f"Support Loss: {metrics['support_loss']:.4f}\\n"
                results_text += f"Query Loss: {metrics['query_loss']:.4f}\\n"
                results_text += f"Shape Accuracy: {metrics['shape_accuracy']:.4f}\\n"
                results_text += f"Grid Accuracy: {metrics['grid_accuracy']:.4f}\\n"
                results_text += f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\\n"
                results_text += f"Sample Exact Accuracy: {metrics['sample_exact_accuracy']:.4f}\\n"
    else:
        results_text += "No training results available.\\n"
        results_text += "Only model parameters are shown.\\n\\n"
        results_text += "This could be because:\\n"
        results_text += "• Training hasn't been run yet\\n"
        results_text += "• Results file is missing or corrupted\\n"
        results_text += "• Only evaluation has been performed\\n"
    
    plt.text(0.05, 0.95, results_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.axis('off')
    
    plt.suptitle('Model Summary and Results', fontsize=16, y=0.95)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'model_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Model summary plot saved to {save_dir}/model_summary.png")
    else:
        plt.show()

def extract_multi_encoder_accuracies(results):
    """
    Extract per-encoder and PoE accuracy data from multi-encoder training results.
    
    Args:
        results: Training results dictionary
        
    Returns:
        dict: Processed accuracy data for visualization
    """
    if 'epoch_accuracies' not in results or not results['epoch_accuracies']:
        return None
    
    # Check if this includes detailed multi-encoder accuracy data
    has_detailed_accuracy = False
    for epoch_data in results['epoch_accuracies']:
        if isinstance(epoch_data, dict) and 'individual_encoders' in epoch_data:
            has_detailed_accuracy = True
            break
    
    if not has_detailed_accuracy:
        return None
    
    print("Processing multi-encoder accuracy data for visualization...")
    
    # Initialize accuracy storage
    encoder_indices = set()
    detailed_epochs = []
    
    # Find all encoder indices and detailed epochs
    for epoch_data in results['epoch_accuracies']:
        if isinstance(epoch_data, dict) and 'individual_encoders' in epoch_data:
            detailed_epochs.append(epoch_data)
            encoder_indices.update(epoch_data['individual_encoders'].keys())
    
    num_encoders = len(encoder_indices)
    num_epochs = len(detailed_epochs)
    
    print(f"Found detailed accuracy data: {num_encoders} encoders, {num_epochs} epochs")
    
    # Initialize per-encoder accuracy storage
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
        'poe_accuracy': {
            'shape_accuracy': [],
            'grid_accuracy': [],
            'overall_accuracy': [],
            'sample_exact_accuracy': []
        }
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
        
        # PoE accuracy
        if 'poe_accuracy' in epoch_data:
            poe_data = epoch_data['poe_accuracy']
            encoder_accuracies['poe_accuracy']['shape_accuracy'].append(poe_data['shape_accuracy'])
            encoder_accuracies['poe_accuracy']['grid_accuracy'].append(poe_data['grid_accuracy'])
            encoder_accuracies['poe_accuracy']['overall_accuracy'].append(poe_data['overall_accuracy'])
            encoder_accuracies['poe_accuracy']['sample_exact_accuracy'].append(poe_data['sample_exact_accuracy'])
        else:
            # Fill with zeros if PoE data missing
            encoder_accuracies['poe_accuracy']['shape_accuracy'].append(0.0)
            encoder_accuracies['poe_accuracy']['grid_accuracy'].append(0.0)
            encoder_accuracies['poe_accuracy']['overall_accuracy'].append(0.0)
            encoder_accuracies['poe_accuracy']['sample_exact_accuracy'].append(0.0)
    
    print(f"✓ Multi-encoder accuracy data processed: {num_encoders} encoders, {num_epochs} epochs")
    return encoder_accuracies

def plot_multi_encoder_accuracies(results, save_dir=None):
    """
    Plot detailed multi-encoder accuracy curves showing per-encoder and PoE performance.
    
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
        
        # Plot PoE accuracy (thick line)
        ax.plot(epochs, accuracy_data['poe_accuracy'][metric], 'k-', linewidth=4, 
               label='Product of Experts (PoE)', alpha=0.9)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # Set consistent y-axis for accuracies
    
    plt.suptitle(f'Multi-Encoder Accuracy Comparison ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multi_encoder_accuracies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder accuracy plot saved to {save_dir}/multi_encoder_accuracies.png")
    else:
        plt.show()

def plot_multi_encoder_accuracy_summary(results, save_dir=None):
    """
    Plot a summary view of multi-encoder accuracy showing final performance and convergence.
    
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
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Final accuracy comparison across encoders
    final_overall_accuracies = []
    encoder_labels = []
    colors = plt.cm.Set1(np.linspace(0, 1, num_encoders + 1))  # +1 for PoE
    
    for encoder_idx, enc_data in accuracy_data['individual_encoders'].items():
        final_overall_accuracies.append(enc_data['overall_accuracy'][-1])
        encoder_labels.append(f'Encoder {encoder_idx}')
    
    # Add PoE final accuracy
    final_overall_accuracies.append(accuracy_data['poe_accuracy']['overall_accuracy'][-1])
    encoder_labels.append('PoE')
    
    bars = axs[0].bar(encoder_labels, final_overall_accuracies, color=colors[:len(final_overall_accuracies)])
    axs[0].set_title('Final Overall Accuracy', fontsize=14)
    axs[0].set_ylabel('Accuracy')
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
    
    # 3. Overall accuracy progress: Individual vs PoE
    axs[2].set_title('Overall Accuracy Progress', fontsize=14)
    
    # Plot individual encoder overall accuracies (thin lines)
    for i, (encoder_idx, enc_data) in enumerate(accuracy_data['individual_encoders'].items()):
        axs[2].plot(epochs, enc_data['overall_accuracy'], color=colors[i], alpha=0.6, linewidth=1.5, 
                   label=f'Encoder {encoder_idx}')
    
    # Plot PoE overall accuracy (thick line)
    axs[2].plot(epochs, accuracy_data['poe_accuracy']['overall_accuracy'], 'k-', linewidth=3, 
               label='PoE', alpha=0.9)
    
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Overall Accuracy')
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[2].grid(True, alpha=0.3)
    axs[2].set_ylim(0, 1.05)
    
    plt.suptitle(f'Multi-Encoder Accuracy Summary ({num_encoders} Encoders)', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multi_encoder_accuracy_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Multi-encoder accuracy summary plot saved to {save_dir}/multi_encoder_accuracy_summary.png")
    else:
        plt.show()
