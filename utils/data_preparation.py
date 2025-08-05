import numpy as np
import os
import json
import random
from utils.settings_manager import settings 


from utils.grid_utils import transform_grid_to_sequence


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
        'key_lists_per_encoder': {},
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
                encoder_datasets[encoder_idx].extend(
                    [(inp, out, key) for inp, out in zip(all_input_sequences, all_output_sequences)]
                )
                key_to_encoder_mapping[key] = encoder_idx
                
                # Store detailed task information
                data_statistics['task_distribution_details'][key] = {
                    'encoder_idx': encoder_idx,
                    'generated_samples': len(all_input_sequences),
                    'generation_successful': True,
                    'data_type': 'synthetic'  # ✅ Track data type
                }
                
                print(f"    [ OK ] Task '{key}' → Encoder {encoder_idx} ({len(all_input_sequences)} synthetic samples)")
            else:
                print(f"    ❌ Task '{key}' → Failed to generate synthetic data")
                data_statistics['task_distribution_details'][key] = {
                    'encoder_idx': None,
                    'generated_samples': 0,
                    'generation_successful': False,
                    'data_type': 'synthetic',
                    'error': 'Synthetic data generation failed'
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
                    encoder_datasets[encoder_idx].extend(
                        [(inp, out, key) for inp, out in zip(all_input_sequences, all_output_sequences)]
                    )
                    key_to_encoder_mapping[key] = encoder_idx
                    encoder_total_samples += len(all_input_sequences)
                    
                    data_statistics['task_distribution_details'][key] = {
                        'encoder_idx': encoder_idx,
                        'generated_samples': len(all_input_sequences),
                        'generation_successful': True,
                        'data_type': 'synthetic'  # ✅ Track data type
                    }
                    
                    print(f"      [ OK ] Task '{key}' → Encoder {encoder_idx} ({len(all_input_sequences)} synthetic samples)")
                else:
                    print(f"      ❌ Task '{key}' → Failed to generate synthetic data")
                    data_statistics['task_distribution_details'][key] = {
                        'encoder_idx': encoder_idx,
                        'generated_samples': 0,
                        'generation_successful': False,
                        'data_type': 'synthetic',
                        'error': 'Synthetic data generation failed'
                    }
            
            print(f"    Encoder {encoder_idx} total: {encoder_total_samples} samples from {len(encoder_keys)} tasks")
            key_idx += num_keys_for_encoder
    
    print(f"\n--- FINALIZING ENCODER DATASETS ---")
    
    # Convert to separate input/output lists and collect statistics
    encoder_splits = []
    for encoder_idx, dataset in enumerate(encoder_datasets):
        if dataset:
            inputs, outputs, keys = zip(*dataset)
            encoder_splits.append((list(inputs), list(outputs)))
            data_statistics['samples_per_encoder'][encoder_idx] = len(inputs)
            
            encoder_keys = [key for key, enc_idx in key_to_encoder_mapping.items() if enc_idx == encoder_idx]
            data_statistics['keys_per_encoder'][encoder_idx] = encoder_keys
            data_statistics['key_lists_per_encoder'][encoder_idx] = list(keys)
            
            print(f"  Encoder {encoder_idx}: {len(inputs)} samples from {len(encoder_keys)} tasks {encoder_keys}")
        else:
            encoder_splits.append(([], []))
            data_statistics['samples_per_encoder'][encoder_idx] = 0
            data_statistics['keys_per_encoder'][encoder_idx] = []
            data_statistics['key_lists_per_encoder'][encoder_idx] = []
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
    """Generate TRAINING data for a single key using synthetic generators."""
    # ✅ ALWAYS use synthetic generation for training
    try:
        _, _, _, input_sequences, output_sequences = generate_func(key, n_examples)
        if not input_sequences or not output_sequences:
            print(f"      Warning: Empty sequences generated for key '{key}'")
            return [], []
        print(f"      [OK] Generated {len(input_sequences)} synthetic training samples for key '{key}'")
        return input_sequences, output_sequences
    except Exception as e:
        print(f"      Error generating synthetic data for key '{key}': {e}")
        return [], []

def _generate_ood_evaluation_data(key, n_examples):
    """Generate EVALUATION data for a single key using original ARC tasks."""
    # ✅ Use original ARC tasks ONLY for evaluation
    try:
        # Get available original ARC evaluation tasks
        available_tasks = get_available_evaluation_tasks()
        if not available_tasks:
            print(f"      Warning: No original ARC evaluation tasks found for key '{key}'")
            return [], []
        
        # Use original ARC evaluation tasks for OOD evaluation
        _, _, _, input_sequences, output_sequences = load_and_process_original_arc_evaluation_tasks(
            available_tasks, n_examples_per_task=n_examples
        )
        
        if not input_sequences or not output_sequences:
            print(f"      Warning: No out-of-distribution samples loaded for key '{key}'")
            return [], []
        
        # Randomly sample the required number of examples
        if len(input_sequences) >= n_examples:
            indices = np.random.choice(len(input_sequences), n_examples, replace=False)
            input_sequences = [input_sequences[i] for i in indices]
            output_sequences = [output_sequences[i] for i in indices]
            print(f"      [OK] Generated {len(input_sequences)} OOD evaluation samples for key '{key}'")
            return input_sequences, output_sequences
        else:
            print(f"      Warning: Only {len(input_sequences)} OOD samples available for key '{key}'")
            return input_sequences, output_sequences
            
    except Exception as e:
        print(f"      Error generating OOD data for key '{key}': {e}")
        return [], []

def get_available_evaluation_tasks():
    """
    Get list of available evaluation task keys.
    
    Returns:
        List[str]: List of task keys available for out-of-distribution evaluation
    """
    eval_dir = 're_arc/arc_original/evaluation'
    if not os.path.exists(eval_dir):
        return []
    
    task_files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
    task_keys = [f.replace('.json', '') for f in task_files]
    return sorted(task_keys)

def load_original_arc_evaluation_task(task_key):
    """
    Load an original ARC evaluation task from the evaluation directory.
    
    Args:
        task_key (str): The task key (filename without .json extension)
        
    Returns:
        dict: The task data with 'train' and 'test' examples
    """
    file_path = f're_arc/arc_original/evaluation/{task_key}.json'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation task file not found: {file_path}")
    
    with open(file_path, 'r') as fp:
        task_data = json.load(fp)
    
    return task_data

def load_and_process_original_arc_evaluation_tasks(task_keys, n_examples_per_task=10):
    """
    Load and process original ARC evaluation tasks for out-of-distribution sampling.
    
    Args:
        task_keys: List of task keys to load from original ARC evaluation
        n_examples_per_task: Number of examples to use per task
        
    Returns:
        Tuple of (generated_examples, input_grids, output_grids, input_sequences, output_sequences)
    """
    generated_examples = []
    input_grids = []
    output_grids = []
    input_sequences = []
    output_sequences = []
    
    for task_key in task_keys:
        try:
            # Load the original ARC evaluation task
            task_data = load_original_arc_evaluation_task(task_key)
            
            # Combine train and test examples
            all_examples = task_data.get('train', []) + task_data.get('test', [])
            
            # Sample n_examples_per_task examples (or all if fewer available)
            n_available = len(all_examples)
            n_to_use = min(n_examples_per_task, n_available)
            
            if n_to_use == 0:
                print(f"Warning: No examples found for task {task_key}")
                continue
                
            # Randomly sample examples
            selected_examples = random.sample(all_examples, n_to_use)
            
            for example in selected_examples:
                # Original ARC format has 'input' and 'output' as 2D arrays
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                # Convert to the format expected by the rest of the pipeline
                processed_example = {
                    'input': input_grid.tolist(),
                    'output': output_grid.tolist()
                }
                
                generated_examples.append(processed_example)
                input_grids.append(input_grid)
                output_grids.append(output_grid)
                
                # Transform to sequences
                input_seq = transform_grid_to_sequence(input_grid)
                output_seq = transform_grid_to_sequence(output_grid)
                
                input_sequences.append(input_seq)
                output_sequences.append(output_seq)
                
            print(f"Loaded {n_to_use} examples from original ARC task {task_key}")
            
        except Exception as e:
            print(f"Error loading original ARC task {task_key}: {e}")
            continue
    
    print(f"Total loaded {len(generated_examples)} out-of-distribution examples from {len(task_keys)} tasks")
    
    return generated_examples, input_grids, output_grids, input_sequences, output_sequences

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
                print(f"    [ OK ] Moved {samples_to_move} samples: Encoder {source_encoder} → Encoder {empty_idx}")
            else:
                print(f"    [ WARNING ] Cannot move samples from Encoder {source_encoder} (insufficient samples)")
    
    print(f"  Total samples redistributed: {total_redistributed}")

def safe_extract_grid_from_sequence(sequence, max_rows=30, max_cols=30):
    """Safely extract grid from sequence, handling different input formats."""
    try:
        sequence = np.array(sequence)
        
        # Handle 902-length ARC format
        if len(sequence) >= 902:
            rows = int(sequence[-2])
            cols = int(sequence[-1])
            grid_flat = sequence[:900]
            grid_full = grid_flat.reshape(30, 30)
            actual_grid = grid_full[:rows, :cols]
            return actual_grid, (rows, cols)
        
        # Handle square grid format
        grid_size = int(np.sqrt(len(sequence)))
        if grid_size * grid_size == len(sequence):
            grid = sequence.reshape(grid_size, grid_size)
            return grid, (grid_size, grid_size)
        
        # Handle other formats - try to reshape to reasonable dimensions
        if len(sequence) <= 900:
            # Try to make it square
            grid_size = int(np.sqrt(len(sequence)))
            if grid_size * grid_size <= len(sequence):
                grid = sequence[:grid_size*grid_size].reshape(grid_size, grid_size)
                return grid, (grid_size, grid_size)
        
        # Fallback: return as 1D array
        return sequence.reshape(1, -1), (1, len(sequence))
        
    except Exception as e:
        print(f"Warning: Could not extract grid from sequence of length {len(sequence)}: {e}")
        # Return a default grid
        return np.zeros((max_rows, max_cols)), (max_rows, max_cols)

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
        
        # Validate dimensions – if invalid, fall back to best effort based on available pixels
        if recon_rows <= 0 or recon_cols <= 0 or recon_rows > 30 or recon_cols > 30:
            # Fallback: try to infer square grid size from available predictions (≤30)
            total_pred = len(pred_grid_flat.flatten() if hasattr(pred_grid_flat, 'flatten') else pred_grid_flat)
            if total_pred > 0:
                side = int(np.sqrt(total_pred))
                side = max(1, min(30, side))
                recon_rows = recon_cols = side
            else:
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
                # Fallback: reshape using whatever pixels are available into square grid
                side = int(np.sqrt(available_pixels))
                if side > 0:
                    recon_grid = grid_predictions[:side*side].reshape(side, side)
                    return recon_grid, side, side
                return None, recon_rows, recon_cols
        else:
            return None, recon_rows, recon_cols
            
    except Exception as e:
        return None, 0, 0

def generate_per_key_ood_samples(evaluation_keys, n_samples_per_key, n_queries_per_key, seed=42):
    """
    Generate per-key out-of-distribution sample sets with fixed support/query samples for each evaluation key.
    
    Args:
        evaluation_keys: List of evaluation keys that need OOD samples
        n_samples_per_key: Number of support samples per key
        n_queries_per_key: Number of query samples per key
        seed: Random seed for reproducible sampling
        
    Returns:
        dict: {key: {'support': [samples], 'query': [samples]}}
    """
    # Set random seed for reproducible sampling
    random.seed(seed)
    
    # Get all available original ARC evaluation tasks
    available_tasks = get_available_evaluation_tasks()
    if not available_tasks:
        print("Warning: No original ARC evaluation tasks found")
        return {}
    
    print(f"Generating per-key OOD samples for {len(evaluation_keys)} evaluation keys")
    print(f"Using {len(available_tasks)} available original ARC evaluation tasks")
    
    # Create a mapping from evaluation keys to OOD tasks
    # Use consistent assignment based on key hash for reproducibility
    key_to_ood_tasks = {}
    for i, eval_key in enumerate(evaluation_keys):
        # Assign 2-3 OOD tasks per evaluation key for variety
        num_ood_tasks = min(3, len(available_tasks) // len(evaluation_keys))
        start_idx = (hash(eval_key) % len(available_tasks))
        assigned_tasks = []
        for j in range(num_ood_tasks):
            task_idx = (start_idx + j) % len(available_tasks)
            assigned_tasks.append(available_tasks[task_idx])
        key_to_ood_tasks[eval_key] = assigned_tasks
        print(f"  Key '{eval_key}' assigned to OOD tasks: {assigned_tasks}")
    
    # Generate per-key sample sets
    per_key_samples = {}
    total_needed_per_key = n_samples_per_key + n_queries_per_key
    
    for eval_key in evaluation_keys:
        try:
            # Load examples from assigned OOD tasks for this key
            all_examples = []
            for ood_task in key_to_ood_tasks[eval_key]:
                try:
                    task_data = load_original_arc_evaluation_task(ood_task)
                    task_examples = task_data.get('train', []) + task_data.get('test', [])
                    all_examples.extend(task_examples)
                except Exception as e:
                    print(f"Error loading OOD task {ood_task} for key {eval_key}: {e}")
                    continue
            
            if not all_examples:
                print(f"Warning: No examples available for key '{eval_key}'")
                continue
            
            # Use key-specific seed for reproducible sampling
            key_seed = seed + hash(eval_key)
            random.seed(key_seed)
            
            # Sample total needed examples for this key
            n_available = len(all_examples)
            n_to_sample = min(total_needed_per_key, n_available)
            
            if n_to_sample < total_needed_per_key:
                print(f"Warning: Only {n_to_sample} examples available for key '{eval_key}', requested {total_needed_per_key}")
            
            # Sample examples for this key
            selected_examples = random.sample(all_examples, n_to_sample)
            
            # Split into support and query samples
            support_samples = selected_examples[:n_samples_per_key]
            query_samples = selected_examples[n_samples_per_key:n_samples_per_key + n_queries_per_key]
            
            # Process support samples
            support_input_sequences = []
            support_output_sequences = []
            for example in support_samples:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                input_seq = transform_grid_to_sequence(input_grid)
                output_seq = transform_grid_to_sequence(output_grid)
                support_input_sequences.append(input_seq)
                support_output_sequences.append(output_seq)
            
            # Process query samples
            query_input_sequences = []
            query_output_sequences = []
            for example in query_samples:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                input_seq = transform_grid_to_sequence(input_grid)
                output_seq = transform_grid_to_sequence(output_grid)
                query_input_sequences.append(input_seq)
                query_output_sequences.append(output_seq)
            
            per_key_samples[eval_key] = {
                'support': {
                    'input_sequences': support_input_sequences,
                    'output_sequences': support_output_sequences,
                    'samples': support_samples
                },
                'query': {
                    'input_sequences': query_input_sequences,
                    'output_sequences': query_output_sequences,
                    'samples': query_samples
                },
                # ✅ ADD: Store the actual OOD task keys used
                'ood_task_keys': key_to_ood_tasks[eval_key]
            }
            
            print(f"  Key '{eval_key}': {len(support_input_sequences)} support, {len(query_input_sequences)} query samples")
            
        except Exception as e:
            print(f"Error generating OOD samples for key '{eval_key}': {e}")
            continue
    
    print(f"Generated per-key OOD samples for {len(per_key_samples)} keys")
    return per_key_samples

def generate_ood_evaluation_dataset(evaluation_keys, n_samples_per_key, n_queries_per_key, seed=42):
    """
    Generate OOD evaluation dataset using original ARC tasks.
    ✅ This should ONLY be called during evaluation, never during training.
    """
    print(f"\n=== GENERATING OOD EVALUATION DATASET ===")
    print(f"Evaluation keys: {evaluation_keys}")
    print(f"Samples per key: {n_samples_per_key}")
    print(f"Queries per key: {n_queries_per_key}")
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    
    # Get all available original ARC evaluation tasks
    available_tasks = get_available_evaluation_tasks()
    if not available_tasks:
        print("Warning: No original ARC evaluation tasks found")
        return {}
    
    print(f"Using {len(available_tasks)} available original ARC evaluation tasks")
    
    # Create a mapping from evaluation keys to OOD tasks
    key_to_ood_tasks = {}
    for i, eval_key in enumerate(evaluation_keys):
        # Assign 2-3 OOD tasks per evaluation key for variety
        num_ood_tasks = min(3, len(available_tasks) // len(evaluation_keys))
        start_idx = (hash(eval_key) % len(available_tasks))
        assigned_tasks = []
        for j in range(num_ood_tasks):
            task_idx = (start_idx + j) % len(available_tasks)
            assigned_tasks.append(available_tasks[task_idx])
        key_to_ood_tasks[eval_key] = assigned_tasks
        print(f"  Key '{eval_key}' assigned to OOD tasks: {assigned_tasks}")
    
    # Generate per-key OOD sample sets
    ood_evaluation_data = {}
    total_needed_per_key = n_samples_per_key + n_queries_per_key
    
    for eval_key in evaluation_keys:
        try:
            # Load examples from assigned OOD tasks for this key
            all_examples = []
            for ood_task in key_to_ood_tasks[eval_key]:
                try:
                    task_data = load_original_arc_evaluation_task(ood_task)
                    task_examples = task_data.get('train', []) + task_data.get('test', [])
                    all_examples.extend(task_examples)
                except Exception as e:
                    print(f"Error loading OOD task {ood_task} for key {eval_key}: {e}")
                    continue
            
            if not all_examples:
                print(f"Warning: No examples available for key '{eval_key}'")
                continue
            
            # Use key-specific seed for reproducible sampling
            key_seed = seed + hash(eval_key)
            random.seed(key_seed)
            
            # Sample total needed examples for this key
            n_available = len(all_examples)
            n_to_sample = min(total_needed_per_key, n_available)
            
            if n_to_sample < total_needed_per_key:
                print(f"Warning: Only {n_to_sample} examples available for key '{eval_key}', requested {total_needed_per_key}")
            
            # Sample examples for this key
            selected_examples = random.sample(all_examples, n_to_sample)
            
            # Split into support and query samples
            support_samples = selected_examples[:n_samples_per_key]
            query_samples = selected_examples[n_samples_per_key:n_samples_per_key + n_queries_per_key]
            
            # Process support samples
            support_input_sequences = []
            support_output_sequences = []
            for example in support_samples:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                input_seq = transform_grid_to_sequence(input_grid)
                output_seq = transform_grid_to_sequence(output_grid)
                support_input_sequences.append(input_seq)
                support_output_sequences.append(output_seq)
            
            # Process query samples
            query_input_sequences = []
            query_output_sequences = []
            for example in query_samples:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                input_seq = transform_grid_to_sequence(input_grid)
                output_seq = transform_grid_to_sequence(output_grid)
                query_input_sequences.append(input_seq)
                query_output_sequences.append(output_seq)
            
            ood_evaluation_data[eval_key] = {
                'support': {
                    'input_sequences': support_input_sequences,
                    'output_sequences': support_output_sequences,
                    'samples': support_samples
                },
                'query': {
                    'input_sequences': query_input_sequences,
                    'output_sequences': query_output_sequences,
                    'samples': query_samples
                },
                'ood_task_keys': key_to_ood_tasks[eval_key],
                'data_type': 'ood_original_arc'  # ✅ Track data type
            }
            
            print(f"  Key '{eval_key}': {len(support_input_sequences)} support, {len(query_input_sequences)} query OOD samples")
            
        except Exception as e:
            print(f"Error generating OOD samples for key '{eval_key}': {e}")
            continue
    
    print(f"Generated OOD evaluation data for {len(ood_evaluation_data)} keys")
    return ood_evaluation_data