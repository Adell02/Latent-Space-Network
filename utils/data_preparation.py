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
    
    DEPRECATED: Use split_dataset_by_keys_for_multi_encoder for key-based splitting.
    Kept for backward compatibility only.
    
    Args:
        input_sequences: List of input sequences
        output_sequences: List of output sequences  
        num_encoders: Number of encoders (and thus number of splits)
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of tuples: [(inputs_0, outputs_0), (inputs_1, outputs_1), ...]
    """
    print("WARNING: Using deprecated even-split method. Consider using key-based splitting.")
    
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

def extract_grid_from_sequence(sequence, max_rows=30, max_cols=30):
    """Extract the 2D grid and its shape from an ARC sequence."""
    sequence = np.array(sequence)

    if len(sequence) >= 902:
        rows = int(sequence[-2])
        cols = int(sequence[-1])

        grid_flat = sequence[:900]
        grid_full = grid_flat.reshape(30, 30)
        actual_grid = grid_full[:rows, :cols]
        return actual_grid, (rows, cols)

    grid_size = int(np.sqrt(len(sequence)))
    if grid_size * grid_size == len(sequence):
        grid = sequence.reshape(grid_size, grid_size)
        return grid, (grid_size, grid_size)

    return sequence, getattr(sequence, 'shape', (max_rows, max_cols))

def split_dataset_by_keys_for_multi_encoder(task_keys, num_encoders, n_examples_per_task, generate_func):
    """
    Split dataset by keys for multi-encoder training with comprehensive logging.
    """
    num_keys = len(task_keys)
    print(f"\n=== KEY-BASED DATASET SPLITTING ===")
    print(f"Configuration: {num_keys} keys, {num_encoders} encoders, {n_examples_per_task} examples per task")
    print(f"Task keys: {task_keys}")
    
    # Initialize encoder datasets
    encoder_datasets = [[] for _ in range(num_encoders)]
    key_to_encoder_mapping = {}
    data_statistics = {
        'num_keys': num_keys,
        'num_encoders': num_encoders,
        'n_examples_per_task': n_examples_per_task,
        'splitting_strategy': '',
        'samples_per_encoder': {},
        'keys_per_encoder': {},
        'task_distribution_details': {}  # NEW: Detailed per-task information
    }
    
    print(f"\n--- DETERMINING SPLITTING STRATEGY ---")
    
    if num_keys <= num_encoders:
        if num_keys < num_encoders:
            data_statistics['splitting_strategy'] = 'even_split_with_empty_encoders'
            print(f"Strategy: Even split with redistribution ({num_keys} keys < {num_encoders} encoders)")
            print(f"  - Each of the first {num_keys} encoders gets 1 unique task")
            print(f"  - Remaining {num_encoders - num_keys} encoders will receive redistributed samples")
        else:
            data_statistics['splitting_strategy'] = 'one_key_per_encoder'
            print(f"Strategy: Perfect one-to-one mapping ({num_keys} keys = {num_encoders} encoders)")
            print(f"  - Each encoder gets exactly 1 unique task")
            print(f"  - Perfect encoder specialization expected")
        
        print(f"\n--- ASSIGNING TASKS TO ENCODERS ---")
        
        for i, key in enumerate(task_keys):
            encoder_idx = i % num_encoders
            print(f"  Processing task '{key}' for Encoder {encoder_idx}...")
            
            all_input_sequences, all_output_sequences = _generate_key_data(key, n_examples_per_task, generate_func)
            
            if all_input_sequences and all_output_sequences:
                encoder_datasets[encoder_idx].extend(list(zip(all_input_sequences, all_output_sequences)))
                key_to_encoder_mapping[key] = encoder_idx
                
                # Store detailed task information
                data_statistics['task_distribution_details'][key] = {
                    'encoder_idx': encoder_idx,
                    'generated_samples': len(all_input_sequences),
                    'generation_successful': True
                }
                
                print(f"    ✓ Task '{key}' → Encoder {encoder_idx} ({len(all_input_sequences)} samples generated)")
            else:
                print(f"    ❌ Task '{key}' → Failed to generate data")
                data_statistics['task_distribution_details'][key] = {
                    'encoder_idx': None,
                    'generated_samples': 0,
                    'generation_successful': False,
                    'error': 'Data generation failed'
                }
        
        if num_keys < num_encoders:
            print(f"\n--- REDISTRIBUTING SAMPLES TO EMPTY ENCODERS ---")
            _distribute_extra_samples(encoder_datasets, num_keys, num_encoders)
    
    else:
        # (c): More keys than encoders
        data_statistics['splitting_strategy'] = 'clustered_keys_per_encoder'
        print(f"Strategy: Clustered assignment ({num_keys} keys > {num_encoders} encoders)")
        
        keys_per_encoder = num_keys // num_encoders
        remaining_keys = num_keys % num_encoders
        
        print(f"  - Base keys per encoder: {keys_per_encoder}")
        print(f"  - Extra keys for first {remaining_keys} encoders: 1 each")
        print(f"  - Expected samples per encoder: ~{keys_per_encoder * n_examples_per_task} to {(keys_per_encoder + 1) * n_examples_per_task}")
        
        print(f"\n--- ASSIGNING CLUSTERED TASKS TO ENCODERS ---")
        
        key_idx = 0
        for encoder_idx in range(num_encoders):
            num_keys_for_encoder = keys_per_encoder + (1 if encoder_idx < remaining_keys else 0)
            encoder_keys = task_keys[key_idx:key_idx + num_keys_for_encoder]
            
            print(f"  Encoder {encoder_idx} assigned tasks: {encoder_keys} ({len(encoder_keys)} tasks)")
            
            encoder_total_samples = 0
            for key in encoder_keys:
                print(f"    Processing task '{key}' for Encoder {encoder_idx}...")
                
                all_input_sequences, all_output_sequences = _generate_key_data(key, n_examples_per_task, generate_func)
                
                if all_input_sequences and all_output_sequences:
                    encoder_datasets[encoder_idx].extend(list(zip(all_input_sequences, all_output_sequences)))
                    key_to_encoder_mapping[key] = encoder_idx
                    encoder_total_samples += len(all_input_sequences)
                    
                    data_statistics['task_distribution_details'][key] = {
                        'encoder_idx': encoder_idx,
                        'generated_samples': len(all_input_sequences),
                        'generation_successful': True
                    }
                    
                    print(f"      ✓ Task '{key}' → Encoder {encoder_idx} ({len(all_input_sequences)} samples)")
                else:
                    print(f"      ❌ Task '{key}' → Failed to generate data")
                    data_statistics['task_distribution_details'][key] = {
                        'encoder_idx': encoder_idx,
                        'generated_samples': 0,
                        'generation_successful': False,
                        'error': 'Data generation failed'
                    }
            
            print(f"    Encoder {encoder_idx} total: {encoder_total_samples} samples from {len(encoder_keys)} tasks")
            key_idx += num_keys_for_encoder
    
    print(f"\n--- FINALIZING ENCODER DATASETS ---")
    
    # Convert to separate input/output lists and collect statistics
    encoder_splits = []
    for encoder_idx, dataset in enumerate(encoder_datasets):
        if dataset:
            inputs, outputs = zip(*dataset)
            encoder_splits.append((list(inputs), list(outputs)))
            data_statistics['samples_per_encoder'][encoder_idx] = len(inputs)
            
            encoder_keys = [key for key, enc_idx in key_to_encoder_mapping.items() if enc_idx == encoder_idx]
            data_statistics['keys_per_encoder'][encoder_idx] = encoder_keys
            
            print(f"  Encoder {encoder_idx}: {len(inputs)} samples from {len(encoder_keys)} tasks {encoder_keys}")
        else:
            encoder_splits.append(([], []))
            data_statistics['samples_per_encoder'][encoder_idx] = 0
            data_statistics['keys_per_encoder'][encoder_idx] = []
            print(f"  Encoder {encoder_idx}: 0 samples (no tasks assigned)")
    
    # Enhanced summary statistics
    total_samples = sum(data_statistics['samples_per_encoder'].values())
    successful_tasks = sum(1 for task_info in data_statistics['task_distribution_details'].values() 
                          if task_info['generation_successful'])
    failed_tasks = len(data_statistics['task_distribution_details']) - successful_tasks
    
    print(f"\n=== DATASET SPLITTING SUMMARY ===")
    print(f"Strategy used: {data_statistics['splitting_strategy']}")
    print(f"Total samples generated: {total_samples}")
    print(f"Successful task generations: {successful_tasks}/{len(task_keys)}")
    if failed_tasks > 0:
        print(f"Failed task generations: {failed_tasks}")
        failed_task_keys = [key for key, info in data_statistics['task_distribution_details'].items() 
                           if not info['generation_successful']]
        print(f"  Failed tasks: {failed_task_keys}")
    
    print(f"\nDetailed encoder distribution:")
    for encoder_idx in range(num_encoders):
        samples = data_statistics['samples_per_encoder'][encoder_idx]
        keys = data_statistics['keys_per_encoder'][encoder_idx]
        if samples > 0:
            avg_samples_per_task = samples / len(keys) if keys else 0
            print(f"  Encoder {encoder_idx}: {samples} samples ({len(keys)} tasks, avg {avg_samples_per_task:.1f} samples/task)")
            for key in keys:
                task_samples = data_statistics['task_distribution_details'].get(key, {}).get('generated_samples', 0)
                print(f"    - {key}: {task_samples} samples")
        else:
            print(f"  Encoder {encoder_idx}: 0 samples (empty)")
    
    print(f"=" * 50)
    
    return encoder_splits, key_to_encoder_mapping, data_statistics

def _generate_key_data(key, n_examples, generate_func):
    """Generate data for a single key efficiently with detailed logging."""
    try:
        _, _, _, input_sequences, output_sequences = generate_func(key, n_examples)
        if not input_sequences or not output_sequences:
            print(f"      Warning: Empty sequences generated for key '{key}'")
            return [], []
        return input_sequences, output_sequences
    except Exception as e:
        print(f"      Error generating data for key '{key}': {e}")
        return [], []

def _distribute_extra_samples(encoder_datasets, num_keys, num_encoders):
    """Distribute samples from populated encoders to empty ones with detailed logging."""
    if num_keys >= num_encoders:
        return
    
    populated_encoders = [i for i in range(num_keys) if encoder_datasets[i]]
    empty_encoders = [i for i in range(num_keys, num_encoders)]
    
    if not populated_encoders or not empty_encoders:
        print("  No redistribution needed - no populated or empty encoders found")
        return
    
    print(f"  Redistributing samples from {len(populated_encoders)} populated to {len(empty_encoders)} empty encoders")
    
    total_redistributed = 0
    for empty_idx in empty_encoders:
        source_idx = empty_idx % len(populated_encoders)
        source_encoder = populated_encoders[source_idx]
        
        if encoder_datasets[source_encoder]:
            samples_to_move = len(encoder_datasets[source_encoder]) // 2
            if samples_to_move > 0:
                moved_samples = encoder_datasets[source_encoder][:samples_to_move]
                encoder_datasets[empty_idx].extend(moved_samples)
                encoder_datasets[source_encoder] = encoder_datasets[source_encoder][samples_to_move:]
                total_redistributed += samples_to_move
                print(f"    ✓ Moved {samples_to_move} samples: Encoder {source_encoder} → Encoder {empty_idx}")
            else:
                print(f"    ⚠ Cannot move samples from Encoder {source_encoder} (insufficient samples)")
    
    print(f"  Total samples redistributed: {total_redistributed}")

def safe_extract_reconstruction_grid(shape_logits, grid_logits):
    """
    Safely extract grid from reconstruction logits, handling scalar conversion errors.
    
    Args:
        shape_logits: Shape predictions logits
        grid_logits: Grid predictions logits
        
    Returns:
        tuple: (recon_grid, recon_rows, recon_cols) or (None, rows, cols) if invalid
    """
    import numpy as np
    
    try:
        pred_shapes = np.argmax(shape_logits, axis=-1)
        pred_grid_flat = np.argmax(grid_logits, axis=-1)
        
        # Handle both scalar and array cases for shape predictions  
        if np.isscalar(pred_shapes):
            recon_rows = recon_cols = int(pred_shapes)
        elif hasattr(pred_shapes, '__len__') and len(pred_shapes) >= 2:
            recon_rows, recon_cols = int(pred_shapes[0]), int(pred_shapes[1])
        elif hasattr(pred_shapes, '__len__') and len(pred_shapes) == 1:
            # Handle nested array case: pred_shapes[0] might be [rows, cols]
            inner_shape = pred_shapes[0]
            if hasattr(inner_shape, '__len__') and len(inner_shape) >= 2:
                recon_rows, recon_cols = int(inner_shape[0]), int(inner_shape[1])
            elif hasattr(inner_shape, '__len__') and len(inner_shape) == 1:
                recon_rows = recon_cols = int(inner_shape[0])
            else:
                recon_rows = recon_cols = int(inner_shape)
        else:
            return None, 0, 0
        
        # Validate dimensions
        if recon_rows <= 0 or recon_cols <= 0 or recon_rows > 30 or recon_cols > 30:
            return None, recon_rows, recon_cols
        
        # Handle grid prediction data - fix the shape issue
        # pred_grid_flat might be (1, 900) instead of (900,), so flatten it
        if hasattr(pred_grid_flat, 'flatten'):
            grid_predictions = pred_grid_flat.flatten()
        else:
            grid_predictions = pred_grid_flat
        
        if hasattr(grid_predictions, '__len__') and len(grid_predictions) >= 900:
            # Full 900 predictions - reshape and crop
            full_grid = grid_predictions[:900].reshape(30, 30)
            recon_grid = full_grid[:recon_rows, :recon_cols]
            return recon_grid, recon_rows, recon_cols
        elif hasattr(grid_predictions, '__len__'):
            # Limited predictions - use what we have
            available_pixels = len(grid_predictions)
            needed_pixels = recon_rows * recon_cols
            
            if available_pixels >= needed_pixels:
                recon_grid = grid_predictions[:needed_pixels].reshape(recon_rows, recon_cols)
                return recon_grid, recon_rows, recon_cols
            else:
                return None, recon_rows, recon_cols
        else:
            return None, recon_rows, recon_cols
            
    except Exception as e:
        return None, 0, 0