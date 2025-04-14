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


def prepare_input_output_pair(input_grid, output_grid):
    """Prepare an input-output pair for the encoder."""
    input_seq = transform_grid_to_sequence(np.array(input_grid))
    output_seq = transform_grid_to_sequence(np.array(output_grid))
    cls_token = np.array([-1])
    full_sequence = np.concatenate([input_seq, output_seq, cls_token])
    return full_sequence
