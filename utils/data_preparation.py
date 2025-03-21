import numpy as np
import os 


def transform_grid_to_sequence(grid):
    """
    Transform a 2D grid into a sequence as described in the paper:
    - Pad grid to 30x30
    - Add shape information (2 values for rows and columns)
    - Flatten in raster-scan fashion

    Args:
        grid: 2D numpy array of pixel values
    Returns:
        sequence: 1D numpy array of length 902 (2 shape values + 900 pixel values)
    """
    # Get original shape
    rows, cols = grid.shape

    # Create the shape prefix (2 values)
    shape_info = np.array([rows, cols])

    # Pad the grid to 30x30
    padded_grid = np.zeros((30, 30), dtype=int)
    padded_grid[:rows, :cols] = grid

    # Flatten the padded grid in raster-scan fashion
    flattened_grid = padded_grid.flatten()

    # Concatenate shape info with flattened grid
    sequence = np.concatenate([flattened_grid,shape_info])

    return sequence


def prepare_input_output_pair(input_grids,output_grids):
    """
    Prepare an input-output pair for the encoder as described.
    Total sequence length will be 1805 (902 + 902 + 1 CLS token)

    Args:
        input_grid: 2D numpy array for input
        output_grid: 2D numpy array for output
    Returns:
        full_sequence: Combined sequence for encoder input
    """ 
    full_sequences = []

    for idx in range(len(input_grids)):
        # Assuming that each generated example is a dictionary with keys 'input' and 'output'
        input_grid = input_grids[idx]
        output_grid = output_grids[idx]

        # Transform the grids into sequences (e.g., padded grid flattened + shape info)
        input_seq = transform_grid_to_sequence(input_grid)
        output_seq = transform_grid_to_sequence(output_grid)

        # Add CLS token (represented as -1 here, you can change this as needed)
        cls_token = np.array([-1])
        
        full_sequence = np.concatenate([input_seq, output_seq, cls_token])
        full_sequences.append(full_sequence)


    return np.array(full_sequences)
