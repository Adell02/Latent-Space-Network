from utils.data_preparation import transform_grid_to_sequence, prepare_input_output_pair

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from tabulate import tabulate
import os
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
# Visualize Sequence Reconstruction
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