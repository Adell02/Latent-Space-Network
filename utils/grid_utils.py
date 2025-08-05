import numpy as np

def transform_grid_to_sequence(grid: np.ndarray):
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

    # Validate dimensions (should be <= 30 for ARC)
    if rows > 30 or cols > 30:
        raise ValueError(f"Grid dimensions ({rows}, {cols}) exceed maximum of 30x30")
    
    # Validate pixel values (should be 0-9 for ARC)
    if grid.min() < 0 or grid.max() > 9:
        raise ValueError(f"Grid contains invalid pixel values. Range: [{grid.min()}, {grid.max()}], expected: [0, 9]")

    # Create the shape prefix (2 values)
    shape_info = np.array([rows, cols])

    # Pad the grid to 30x30
    padded_grid = np.zeros((30, 30), dtype=int)
    padded_grid[:rows, :cols] = grid

    # Flatten the padded grid in raster-scan fashion
    flattened_grid = padded_grid.flatten()

    # Concatenate shape info with flattened grid
    sequence = np.concatenate([flattened_grid, shape_info])

    return sequence 