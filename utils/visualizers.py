from utils.data_preparation import transform_grid_to_sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from tabulate import tabulate
import os
import torch
from sklearn.manifold import TSNE
import pickle

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

def plot_training_and_latent(results):
    """Plot training loss and latent space visualization."""
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
    all_mus = torch.cat(results['latent_mus'], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=1, random_state=42)
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

    plt.tight_layout()
    plt.show()

def plot_latent_analysis(results):
    """Plot detailed latent space analysis for the latent means, log variances, and sampled z."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 0: t-SNE of latent means (samples plotted directly)
    latent_mu = torch.cat(results['latent_mus'], dim=0).numpy()
    tsne_dims = TSNE(n_components=2, perplexity=1, random_state=42).fit_transform(latent_mu)
    sc0 = axs[0].scatter(tsne_dims[:, 0], tsne_dims[:, 1],
                         c=np.arange(tsne_dims.shape[0]), cmap='viridis', alpha=0.8, s=80)
    axs[0].set_title('t-SNE Mean', fontsize=18)
    axs[0].set_xlabel('Dimension 1', fontsize=16)
    axs[0].set_ylabel('Dimension 2', fontsize=16)
    plt.colorbar(sc0, ax=axs[0], label='Sample Index', pad=0.02)
    axs[0].grid(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # Subplot 1: t-SNE of log-variance values
    latent_log_var = torch.cat(results['latent_log_vars'], dim=0).numpy()
    tsne_dims = TSNE(n_components=2, perplexity=1, random_state=42).fit_transform(latent_log_var)
    sc1 = axs[1].scatter(tsne_dims[:, 0], tsne_dims[:, 1],
                         c=np.arange(tsne_dims.shape[0]), cmap='viridis', alpha=0.8, s=80)
    axs[1].set_title('t-SNE log-Variance', fontsize=18)
    axs[1].set_xlabel('Dimension 1', fontsize=16)
    axs[1].set_ylabel('Dimension 2', fontsize=16)
    plt.colorbar(sc1, ax=axs[1], label='Sample Index', pad=0.02)
    axs[1].grid(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Subplot 2: t-SNE of sampled z values
    latent_z = torch.cat(results['latent_zs'], dim=0).numpy()
    tsne_dims = TSNE(n_components=2, perplexity=1, random_state=42).fit_transform(latent_z)
    sc2 = axs[2].scatter(tsne_dims[:, 0], tsne_dims[:, 1],
                         c=np.arange(tsne_dims.shape[0]), cmap='viridis', alpha=0.8, s=80)
    axs[2].set_title('t-SNE sampled Z', fontsize=18)
    axs[2].set_xlabel('Dimension 1', fontsize=16)
    axs[2].set_ylabel('Dimension 2', fontsize=16)
    plt.colorbar(sc2, ax=axs[2], label='Sample Index', pad=0.02)
    axs[2].grid(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_epoch_accuracies(results):
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
    plt.show()

def plot_reconstructions(results):
    """Plot grid reconstructions from results along with an error map."""
    # Extract sequences and reconstructions.
    input_seqs = results['input_sequences']
    output_seqs = results['output_sequences']
    reconstructions = results['reconstructions']
    shape_logits_list = []
    grid_logits_list = []

    for shape_logits, grid_logits in reconstructions:
        # Convert logits to predicted values.
        pred_shapes = shape_logits.argmax(dim=-1)
        pred_grid = grid_logits.argmax(dim=-1)
        shape_logits_list.append(pred_shapes)
        grid_logits_list.append(pred_grid)

    shape_recons = torch.cat(shape_logits_list, dim=0).cpu().numpy()  # Shape: (N, 2)
    grid_recons = torch.cat(grid_logits_list, dim=0).cpu().numpy()    # Shape: (N, 900)

    num_samples = min(5, len(shape_recons))
    for i in range(num_samples):
        # Create 4 subplots: Input, Target, Reconstruction, Error Map
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        # Plot input:
        input_seq = input_seqs[i]
        in_rows, in_cols = int(input_seq[-2]), int(input_seq[-1])
        input_grid_full = input_seq[:900].reshape(30, 30)
        input_grid = input_grid_full[:in_rows, :in_cols]
        axs[0].imshow(input_grid, cmap='viridis')
        axs[0].set_title('Input')
        axs[0].axis('off')

        # Plot target:
        target_seq = output_seqs[i]
        out_rows, out_cols = int(target_seq[-2]), int(target_seq[-1])
        target_grid_full = target_seq[:900].reshape(30, 30)
        target_grid = target_grid_full[:out_rows, :out_cols]
        axs[1].imshow(target_grid, cmap='viridis')
        axs[1].set_title('Target')
        axs[1].axis('off')

        # Plot reconstruction:
        shape_pred = shape_recons[i]  # Expected: [predicted_rows, predicted_cols]
        recon_rows, recon_cols = int(shape_pred[0]), int(shape_pred[1])
        grid_recon_full = grid_recons[i].reshape(30, 30)
        recon_grid = grid_recon_full[:recon_rows, :recon_cols]
        axs[2].imshow(recon_grid, cmap='viridis')
        axs[2].set_title(f'Reconstruction ({recon_rows}x{recon_cols})')
        axs[2].axis('off')

        # Plot error map over the overlapping region:
        # Determine the common dimensions.
        common_rows = min(out_rows, recon_rows)
        common_cols = min(out_cols, recon_cols)
        error_region_target = target_grid[:common_rows, :common_cols]
        error_region_recon = recon_grid[:common_rows, :common_cols]
        error_map = np.abs(error_region_target - error_region_recon)
        axs[3].imshow(error_map, cmap='hot')
        axs[3].set_title('Error Map')
        axs[3].axis('off')

        plt.suptitle(f'Sample {i+1}', fontsize=18)
        plt.tight_layout()
        plt.show()




def visualize_all_results(results):
    """Plot all visualizations for the results."""
    print("Plotting training progress and latent space...")
    plot_training_and_latent(results)

    print("\nPlotting latent space analysis...")
    plot_latent_analysis(results)

    print("\nPlotting epoch accuracies over time...")
    plot_epoch_accuracies(results)

    print("\nPlotting reconstructions...")
    plot_reconstructions(results)


def plot_evaluation_results(results):
    """
    Plot evaluation metrics and reconstructions for each key.
    
    Args:
        results: Dictionary containing evaluation results for each key
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
    

    for key in results:
        if 'reconstruction_results' in results[key]:
            print(f"\nPlotting support reconstructions")
            aux = {
                'input_sequences': results[key]['reconstruction_results']['input_samples_sequences'],
                'output_sequences': results[key]['reconstruction_results']['output_samples_sequences'],
                'reconstructions': results[key]['reconstruction_results']['support_reconstructions'],
            }
            plot_reconstructions(aux)

            print(f"\nPlotting query reconstructions")
            aux = {
                'input_sequences': results[key]['reconstruction_results']['input_queries_sequences'],
                'output_sequences': results[key]['reconstruction_results']['output_queries_sequences'],
                'reconstructions': results[key]['reconstruction_results']['query_reconstructions'],
            }
            plot_reconstructions(aux)

    return

    # Plot reconstructions for each key
    for key in results:
        if 'reconstruction_results' in results[key]:
            print(f"\nPlotting reconstructions for Key {key}:")
            recon_results = results[key]['reconstruction_results']
            
            # Create a 2x2 grid for sample and query visualizations
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            
            # Plot sample input and output
            sample_input = recon_results['input_samples_sequences'][0].reshape(30, 30)
            sample_output = recon_results['output_samples_sequences'][0].reshape(30, 30)
            
            axs[0, 0].imshow(sample_input, cmap='viridis')
            axs[0, 0].set_title('Sample Input')
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(sample_output, cmap='viridis')
            axs[0, 1].set_title('Sample Output')
            axs[0, 1].axis('off')
            
            # Plot query input, reconstruction, and error map
            query_input = recon_results['input_queries_sequences'][0].reshape(30, 30)
            query_output = recon_results['output_queries_sequences'][0].reshape(30, 30)
            
            # Get reconstruction from the first query
            shape_logits, grid_logits = recon_results['reconstructions'][0]
            pred_shape = shape_logits.argmax(dim=-1)
            pred_grid = grid_logits.argmax(dim=-1)
            
            # Reshape prediction to match grid size
            pred_grid = pred_grid.reshape(30, 30)
            
            # Calculate error map
            error_map = np.abs(query_output - pred_grid.numpy())
            
            axs[1, 0].imshow(query_input, cmap='viridis')
            axs[1, 0].set_title('Query Input')
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(pred_grid, cmap='viridis')
            axs[1, 1].set_title('Query Reconstruction')
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Plot error map separately
            plt.figure(figsize=(6, 6))
            plt.imshow(error_map, cmap='hot')
            plt.colorbar(label='Error Magnitude')
            plt.title('Reconstruction Error Map')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

def visualize_stored_results(run_dir):
    """
    Load and visualize results from a previous run.
    
    Args:
        run_dir: Directory containing the stored results
    """
    # Load training results
    results_file = os.path.join(run_dir, 'results.pkl')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"No results file found in {run_dir}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # Visualize training results
    print("\nVisualizing training results...")
    #visualize_all_results(results)
    
    # Try to load and visualize evaluation results
    eval_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if os.path.exists(eval_file):
        print("\nVisualizing evaluation results...")
        with open(eval_file, 'rb') as f:
            eval_results = pickle.load(f)
        
        # Plot evaluation metrics and reconstructions
        plot_evaluation_results(eval_results)
    else:
        print("\nNo evaluation results found in the run directory.")
