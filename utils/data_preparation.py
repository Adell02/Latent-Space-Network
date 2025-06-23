import numpy as np
import os 


def transform_grid_to_sequence(grid:np.ndarray):
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


def split_dataset_for_multi_encoder(input_sequences, output_sequences, num_encoders, shuffle=True, seed=42):
    """
    Split dataset into num_encoders subsets for individual encoder training.
    
    Args:
        input_sequences: List of input sequences
        output_sequences: List of output sequences  
        num_encoders: Number of encoders (and thus number of splits)
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of tuples: [(inputs_0, outputs_0), (inputs_1, outputs_1), ...]
    """
    if shuffle:
        # Create indices and shuffle them
        indices = np.arange(len(input_sequences))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # Apply shuffled indices
        input_sequences = [input_sequences[i] for i in indices]
        output_sequences = [output_sequences[i] for i in indices]
    
    # Calculate split sizes
    total_samples = len(input_sequences)
    base_size = total_samples // num_encoders
    remainder = total_samples % num_encoders
    
    # Create splits
    splits = []
    start_idx = 0
    
    for i in range(num_encoders):
        # Some splits get one extra sample if there's a remainder
        split_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + split_size
        
        encoder_inputs = input_sequences[start_idx:end_idx]
        encoder_outputs = output_sequences[start_idx:end_idx]
        splits.append((encoder_inputs, encoder_outputs))
        
        start_idx = end_idx
    
    return splits
