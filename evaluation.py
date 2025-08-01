import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import numpy as np

from utils.model_utils import set_seed, prepare_dataloader, collect_latent_data
from torch.utils.data import TensorDataset, DataLoader
from re_arc.main import generate_and_process_tasks
from utils.settings_manager import settings
from utils.latent_functions import get_optimized_z, get_optimized_z_from_initial, optimize_task_latent, optimize_latent_z
from models.base_model import compute_loss

# Maximum batch size to avoid GPU memory issues
MAX_BATCH_SIZE = 16

def collect_support_samples_by_task(samples_dataloader, key_mapping=None, device='cuda'):
    """
    Collect all support samples grouped by task key for task-level optimization.
    
    Args:
        samples_dataloader: DataLoader with (input, target, key_index) batches
        key_mapping: List mapping key_index to actual key string
        device: Device to use
    
    Returns:
        dict: {task_key: [(input_tensor, target_tensor), ...]}
    """
    task_samples = {}
    
    print("  Collecting support samples by task...")
    
    for batch in samples_dataloader:
        # Handle batch structure with keys
        if len(batch) >= 3:
            batch_input, batch_target, batch_key_indices = batch[:3]
        else:
            batch_input, batch_target = batch[:2]
            batch_key_indices = None
        
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        # Group samples by key
        if batch_key_indices is not None and key_mapping is not None:
            for i, key_idx in enumerate(batch_key_indices):
                key = key_mapping[key_idx.item()]  # Get actual key from mapping
                if key not in task_samples:
                    task_samples[key] = []
                task_samples[key].append((batch_input[i:i+1], batch_target[i:i+1]))
        else:
            # If no keys provided, treat each sample as its own task
            for i in range(batch_input.size(0)):
                key = f"sample_{i}"
                if key not in task_samples:
                    task_samples[key] = []
                task_samples[key].append((batch_input[i:i+1], batch_target[i:i+1]))
    
    print(f"  Found {len(task_samples)} unique tasks with support samples:")
    for key, samples in task_samples.items():
        print(f"    Task '{key}': {len(samples)} support samples")
    
    return task_samples

def prepare_dataloader_with_keys(input_seqs, output_seqs, keys, batch_size, shuffle=True):
    """Create a dataloader that includes keys for task-level optimization."""
    # Convert to tensors
    if isinstance(input_seqs, list) and len(input_seqs) > 0 and isinstance(input_seqs[0], np.ndarray):
        try:
            max_len_input = max(len(s) for s in input_seqs) if input_seqs else 0
            max_len_output = max(len(s) for s in output_seqs) if output_seqs else 0
            
            padded_input_seqs = [np.pad(s, (0, max_len_input - len(s)), 'constant') if len(s) < max_len_input else s for s in input_seqs]
            padded_output_seqs = [np.pad(s, (0, max_len_output - len(s)), 'constant') if len(s) < max_len_output else s for s in output_seqs]

            input_tensor = torch.tensor(np.array(padded_input_seqs), dtype=torch.float32)
            output_tensor = torch.tensor(np.array(padded_output_seqs), dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Could not convert with np.array. Error: {e}. Using slow method.")
            input_tensor = torch.FloatTensor(input_seqs) 
            output_tensor = torch.FloatTensor(output_seqs)
    else:
        input_tensor = torch.FloatTensor(input_seqs)
        output_tensor = torch.FloatTensor(output_seqs)

    # Create string tensor for keys - convert to indices for TensorDataset
    key_indices = torch.tensor([i for i in range(len(keys))], dtype=torch.long)
    dataset = TensorDataset(input_tensor, output_tensor, key_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Return dataloader and key mapping
    return dataloader, keys

##############################
# Evaluation Latent Data Collection
##############################

def collect_evaluation_latent_data(model, samples_dataloader, queries_dataloader, device, is_multi_encoder, num_encoders):
    """
    Collect latent representations from support and query samples efficiently.
    Avoids data duplication by reusing shared input/output data.
    Always uses mean vectors (mu) for consistency.
    """
    model.eval()
    evaluation_latent_data = {}
    
    print("  Collecting support samples latent data...")
    support_data = collect_unified_evaluation_latents(
        model, samples_dataloader, device, is_multi_encoder, num_encoders, data_type='support'
    )
    evaluation_latent_data['support'] = support_data
    
    print("  Collecting query samples latent data...")
    query_data = collect_unified_evaluation_latents(
        model, queries_dataloader, device, is_multi_encoder, num_encoders, data_type='query'
    )
    evaluation_latent_data['query'] = query_data
    
    return evaluation_latent_data

def collect_unified_evaluation_latents(model, dataloader, device, is_multi_encoder, num_encoders, data_type='support', max_samples=100, batch_size_limit=16):
    """
    Unified evaluation latent collection using batch-split approach.
    Processes data in smaller batches to avoid memory issues.
    Always uses mean vectors (mu) for consistency and efficiency.
    """
    print(f"    Collecting {data_type} latents from {num_encoders}-encoder model...")
    print(f"    Using mean (mu) vectors for visualization")
    
    # Get the data samples once (shared across all encoders)
    input_samples = []
    output_samples = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input, batch_target, batch_keys = batch[:3]
            else:
                batch_input, batch_target = batch[:2]
                batch_keys = None
            
            if sample_count >= max_samples:
                break
                
            batch_size = min(batch_input.size(0), max_samples - sample_count)
            input_samples.append(batch_input[:batch_size])
            output_samples.append(batch_target[:batch_size])
            sample_count += batch_size
    
    if not input_samples:
        print(f"      [WARNING] No {data_type} samples found")
        return {}
    
    # Combine all batches - this is the shared dataset
    all_inputs = torch.cat(input_samples, dim=0)
    all_outputs = torch.cat(output_samples, dim=0)
    
    print(f"      Processing {len(all_inputs)} {data_type} samples in batches of {batch_size_limit}...")
    
    # Store shared input/output data (avoid duplicating across encoders)
    shared_data = {
        'input_samples': all_inputs.cpu().numpy(),
        'output_samples': all_outputs.cpu().numpy(),
        'num_samples': len(all_inputs),
        'latent_type': 'reparameterized'
    }
    
    encoder_latent_data = {}
    
    # Process data in batches to avoid memory issues
    def process_in_batches(inputs, outputs, processing_func, func_name):
        """Process data in batches and concatenate results."""
        all_mus = []
        all_log_vars = []
        all_zs = []
        
        total_batches = (len(inputs) + batch_size_limit - 1) // batch_size_limit
        print(f"          Processing {total_batches} batches...")
        
        for i in range(0, len(inputs), batch_size_limit):
            end_idx = min(i + batch_size_limit, len(inputs))
            batch_inputs = inputs[i:end_idx].to(device)
            batch_outputs = outputs[i:end_idx].to(device)
            
            with torch.no_grad():
                mu, log_var = processing_func(batch_inputs, batch_outputs)
                # Always use mean vectors (mu) for consistency
                z = model.reparameterize(mu, log_var)
                
                all_mus.append(mu.cpu().numpy())
                all_log_vars.append(log_var.cpu().numpy())
                all_zs.append(z.cpu().numpy())
        
        # Concatenate all batch results
        final_mus = np.concatenate(all_mus, axis=0)
        final_log_vars = np.concatenate(all_log_vars, axis=0)
        final_zs = np.concatenate(all_zs, axis=0)
        
        return final_mus, final_log_vars, final_zs
    
    # For single encoder: only collect once as encoder_0
    if not is_multi_encoder or num_encoders == 1:
        print(f"        Single encoder processing...")
        
        def single_encoder_processing(batch_inputs, batch_outputs):
            mu, log_var,_ =model.encoder(batch_inputs, batch_outputs)
            return mu, log_var
        
        final_mus, final_log_vars, final_zs = process_in_batches(
            all_inputs, all_outputs, single_encoder_processing, "single encoder"
        )
        
        encoder_latent_data['encoder_0'] = {
            'latent_mus': final_mus,
            'latent_log_vars': final_log_vars,
            'latent_zs': final_zs,  # This will be mu or z based on use_mean_for_viz
            'data_type': f"{data_type}_encoder_0",
            'encoder_idx': 0,
            **shared_data
        }
        print(f"          [OK] Collected {len(final_zs)} samples from Encoder 0 (z = mu + sigma * epsilon)")
    
    # For true multi-encoder: collect individual encoder latents + PoE
    else:
        # Individual encoders
        for encoder_idx in range(num_encoders):
            print(f"        Encoder {encoder_idx}...")
            
            def individual_encoder_processing(batch_inputs, batch_outputs):
                mu, log_var = model(batch_inputs, batch_outputs, encoder_idx=encoder_idx)[1:3]
                return mu, log_var
            
            final_mus, final_log_vars, final_zs = process_in_batches(
                all_inputs, all_outputs, individual_encoder_processing, f"encoder_{encoder_idx}"
            )
            
            encoder_latent_data[f"encoder_{encoder_idx}"] = {
                'latent_mus': final_mus,
                'latent_log_vars': final_log_vars,
                'latent_zs': final_zs,
                'data_type': f"{data_type}_encoder_{encoder_idx}",
                'encoder_idx': encoder_idx,
                **shared_data
            }
            print(f"          [OK] Collected {len(final_zs)} samples from Encoder {encoder_idx} (z = mu + sigma * epsilon)")

            # PoE (Product of Experts) - ONLY for evaluation data, not training data
            if data_type in ['support', 'query']:
                print(f"      PoE (Product of Experts)...")
                
                def poe_processing(batch_inputs, batch_outputs):
                    mu, log_var = model(batch_inputs, batch_outputs)[1:3]  # No encoder_idx = PoE
                    return mu, log_var
                
                final_mus, final_log_vars, final_zs = process_in_batches(
                    all_inputs, all_outputs, poe_processing, "PoE"
                )
                
                encoder_latent_data['poe'] = {
                    'latent_mus': final_mus,
                    'latent_log_vars': final_log_vars,
                    'latent_zs': final_zs,
                    'data_type': f"{data_type}_poe",
                    **shared_data
                }
                print(f"          [OK] Collected {len(final_zs)} samples from PoE (z = mu + sigma * epsilon)")
    
    print(f"      [OK] {num_encoders}-encoder {data_type} collection complete")
    return encoder_latent_data

##############################
# Training Latent Data Collection
##############################

def collect_training_latent_representations(model, run_dir, device='cuda'):
    """
    Collect training latent representations efficiently without data duplication.
    Always uses mean vectors (mu) for consistency.
    """
    print("  Loading training data from results.pkl...")
    
    results_file = os.path.join(run_dir, 'results.pkl')
    if not os.path.exists(results_file):
        print(f"  Warning: No training results file found at {results_file}")
        return None
    
    try:
        with open(results_file, 'rb') as f:
            training_results = pickle.load(f)
        
        input_sequences = training_results.get('input_sequences', [])
        output_sequences = training_results.get('output_sequences', [])
        
        if not input_sequences or not output_sequences:
            print("  Warning: No training sequences found in results.pkl")
            return None
        
        print(f"  Found {len(input_sequences)} training sequences")
        
        # Check model type
        is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
        num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
        
        print(f"  Model type: {num_encoders}-encoder ({'Multi' if is_multi_encoder else 'Single'})")
        
        # Limit samples for memory efficiency
        max_samples = 200
        if len(input_sequences) > max_samples:
            input_sequences = input_sequences[:max_samples]
            output_sequences = output_sequences[:max_samples]
            print(f"  Limited to {max_samples} samples for memory efficiency")
        
        # Encode using the trained model
        model.eval()
        batch_size = 16
        print(f"  Encoding {len(input_sequences)} training samples...")
        
        training_latent_data = collect_unified_training_latents(
            model, input_sequences, output_sequences, device, is_multi_encoder, num_encoders, batch_size
        )
        
        if training_latent_data:
            # Add metadata without duplicating data
            training_latent_data['collection_info'] = {
                'total_available_samples': len(training_results.get('input_sequences', [])),
                'collected_samples': len(input_sequences),
                'max_samples_limit': max_samples,
                'batch_size': batch_size,
                'device': str(device),
                'is_multi_encoder': is_multi_encoder,
                'num_encoders': num_encoders
            }
            print(f"  [OK] Successfully collected training latent representations")
        
        return training_latent_data
        
    except Exception as e:
        print(f"  Error collecting training latent representations: {e}")
        return None

def collect_unified_training_latents(model, input_sequences, output_sequences, device, is_multi_encoder, num_encoders, batch_size=16):
    """
    Unified training latent collection using batch-split approach.
    Processes the training data efficiently with memory-safe batching.
    Always uses mean vectors (mu) for consistency.
    """
    print(f"    Collecting training latents from {num_encoders}-encoder model...")
    print(f"    Using mean (mu) vectors for visualization")
    
    # Create dataloader for the shared training data
    dataloader = prepare_dataloader(input_sequences, output_sequences, batch_size, shuffle=False)
    
    # Convert dataloader to tensors for batch processing
    all_inputs = []
    all_outputs = []
    
    for batch in dataloader:
        # Handle batch structure with keys
        if len(batch) >= 3:
            batch_input, batch_output, batch_keys = batch[:3]
        else:
            batch_input, batch_output = batch[:2]
            batch_keys = None
        
        all_inputs.append(batch_input)
        all_outputs.append(batch_output)
    
    if not all_inputs:
        print(f"      [WARNING] No training data found")
        return {}
    
    # Combine all batches
    combined_inputs = torch.cat(all_inputs, dim=0)
    combined_outputs = torch.cat(all_outputs, dim=0)
    
    print(f"      Processing {len(combined_inputs)} training samples in batches of {batch_size}...")
    
    # Store shared input/output data
    shared_data = {
        'input_samples': combined_inputs.cpu().numpy(),
        'output_samples': combined_outputs.cpu().numpy(),
        'num_samples': len(combined_inputs),
        'latent_type': 'reparameterized'
    }
    
    encoder_latent_data = {}
    
    # Process data in batches to avoid memory issues
    def process_training_in_batches(inputs, outputs, processing_func, func_name):
        """Process training data in batches and concatenate results."""
        all_mus = []
        all_log_vars = []
        all_zs = []
        
        total_batches = (len(inputs) + batch_size - 1) // batch_size
        print(f"        Processing {total_batches} batches...")
        
        for i in range(0, len(inputs), batch_size):
            end_idx = min(i + batch_size, len(inputs))
            batch_inputs = inputs[i:end_idx].to(device)
            batch_outputs = outputs[i:end_idx].to(device)
            
            with torch.no_grad():
                mu, log_var = processing_func(batch_inputs, batch_outputs)
                # Always use mean vectors (mu) for consistency
                z = model.reparameterize(mu, log_var)
                
                all_mus.append(mu.cpu().numpy())
                all_log_vars.append(log_var.cpu().numpy())
                all_zs.append(z.cpu().numpy())
        
        # Concatenate all batch results
        final_mus = np.concatenate(all_mus, axis=0)
        final_log_vars = np.concatenate(all_log_vars, axis=0)
        final_zs = np.concatenate(all_zs, axis=0)
        
        return {
            'latent_mus': final_mus,
            'latent_log_vars': final_log_vars,
            'latent_zs': final_zs,  # Always mu (mean) vectors
            'data_type': func_name,
            **shared_data
        }
    
    # For single encoder: only collect once as encoder_0
    if not is_multi_encoder or num_encoders == 1:
        print(f"      Single encoder processing...")
        
        def single_encoder_processing(batch_inputs, batch_outputs):
            mu, log_var,_ =model.encoder(batch_inputs, batch_outputs)
            return mu, log_var
        
        latent_data = process_training_in_batches(
            combined_inputs, combined_outputs, single_encoder_processing, "training_encoder_0"
        )
        latent_data['encoder_idx'] = 0
        encoder_latent_data['encoder_0'] = latent_data
        print(f"        [OK] Collected {latent_data['num_samples']} samples from Encoder 0 (z = mu + sigma * epsilon)")
    
    # For true multi-encoder: collect individual encoder latents + PoE
    else:
        # Individual encoders
        for encoder_idx in range(num_encoders):
            print(f"      Encoder {encoder_idx}...")
            
            def individual_encoder_processing(batch_inputs, batch_outputs):
                mu, log_var = model(batch_inputs, batch_outputs, encoder_idx=encoder_idx)[1:3]
                return mu, log_var
            
            latent_data = process_training_in_batches(
                combined_inputs, combined_outputs, individual_encoder_processing, f"training_encoder_{encoder_idx}"
            )
            latent_data['encoder_idx'] = encoder_idx
            encoder_latent_data[f"encoder_{encoder_idx}"] = latent_data
            print(f"        [OK] Collected {latent_data['num_samples']} samples from Encoder {encoder_idx} (z = mu + sigma * epsilon)")
    
    return encoder_latent_data

##############################
# Legacy Encode Training Sequences (kept for compatibility)
##############################

def encode_training_sequences(model, run_dir, device='cuda', max_samples=500, batch_size=16):
    """
    Load training sequences from results.pkl and encode them with the trained model
    to generate latent representations for background visualization.
    
    Args:
        model: Trained model to use for encoding
        run_dir: Directory containing results.pkl
        device: Device to run encoding on
        max_samples: Maximum number of samples to encode (None for all)
        batch_size: Batch size for encoding
        
    Returns:
        dict: Contains encoded latent representations and metadata
    """
    results_file = os.path.join(run_dir, 'results.pkl')
    
    if not os.path.exists(results_file):
        print(f"Warning: No training results file found at {results_file}")
        return None
    
    try:
        print(f"Loading training sequences from {results_file}...")
        with open(results_file, 'rb') as f:
            training_results = pickle.load(f)
        
        # Extract training sequences
        if 'input_sequences' not in training_results or 'output_sequences' not in training_results:
            print("Warning: No training sequences found in results.pkl")
            return None
            
        input_sequences = training_results['input_sequences']
        output_sequences = training_results['output_sequences']
        
        if not input_sequences or not output_sequences:
            print("Warning: Empty training sequences in results.pkl")
            return None
        
        print(f"Found {len(input_sequences)} training sequences")
        
        # Limit samples if requested
        if max_samples is not None and len(input_sequences) > max_samples:
            input_sequences = input_sequences[:max_samples]
            output_sequences = output_sequences[:max_samples]
            print(f"Limited to {max_samples} samples for memory efficiency")
        
        # Encode sequences using the trained model
        model.eval()
        latent_mus = []
        latent_log_vars = []
        latent_zs = []
        initial_losses = []
        
        print(f"Encoding {len(input_sequences)} training samples using trained model...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(input_sequences), batch_size), desc="Encoding training samples"):
                end_idx = min(i + batch_size, len(input_sequences))
                
                # Convert to tensors
                batch_input = torch.tensor(input_sequences[i:end_idx]).float().to(device)
                batch_output = torch.tensor(output_sequences[i:end_idx]).float().to(device)
                
                # Encode (equivalent to initial step of trajectory)
                # Handle both single and multi-encoder models - always use mean vectors
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    # Multi-encoder: use PoE inference
                    mu, log_var = model(batch_input, batch_output)[1:3]
                    z = model.reparameterize(mu, log_var)
                else:
                    # Single encoder
                    mu, log_var,_ =model.encoder(batch_input, batch_output)
                    z = model.reparameterize(mu, log_var)
                
                # Compute loss for this encoded latent (equivalent to initial trajectory loss)
                # Handle both single and multi-encoder models for decoding
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    # Multi-encoder: use shared decoder
                    shape_logits, grid_logits = model.multi_encoder.decoder(z, batch_input, target_seq=batch_output)
                else:
                    # Single encoder
                    shape_logits, grid_logits = model.decoder(z, batch_input, target_seq=batch_output)
                shape_targets = batch_output[:, 900:902].long()
                shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
                
                # Compute grid loss
                grid_loss_list = []
                for j in range(batch_input.size(0)):
                    tgt_rows = int(batch_output[j, 900].item())
                    tgt_cols = int(batch_output[j, 901].item())
                    active_pixels = tgt_rows * tgt_cols
                    if active_pixels > 0:
                        loss_j = F.cross_entropy(grid_logits[j, :active_pixels],
                                               batch_output[j, :active_pixels].long())
                        grid_loss_list.append(loss_j)

                grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else \
                           torch.tensor(0.0, device=batch_input.device)
                
                batch_losses = (shape_loss + grid_loss).item()
                
                # Store equivalent information to trajectory
                latent_mus.append(mu.cpu().numpy())
                latent_log_vars.append(log_var.cpu().numpy())
                latent_zs.append(z.cpu().numpy())
                initial_losses.extend([batch_losses] * batch_input.size(0))  # Loss per sample
        
        # Concatenate all batches
        all_latent_mus = np.concatenate(latent_mus, axis=0)
        all_latent_log_vars = np.concatenate(latent_log_vars, axis=0)
        all_latent_zs = np.concatenate(latent_zs, axis=0)
        
        print(f"Successfully encoded {len(all_latent_mus)} training samples")
        
        encoded_data = {
            'latent_mus': all_latent_mus,
            'latent_log_vars': all_latent_log_vars,
            'latent_zs': all_latent_zs,
            'initial_losses': initial_losses,
            'input_sequences': input_sequences,
            'output_sequences': output_sequences,
            'encoding_info': {
                'total_training_samples': len(training_results['input_sequences']),
                'encoded_samples': len(all_latent_mus),
                'max_samples_limit': max_samples,
                'batch_size': batch_size,
                'device': str(device),
                'has_initial_losses': True
            }
        }
        
        return encoded_data
        
    except Exception as e:
        print(f"Error encoding training sequences: {e}")
        return None

##############################
# Run Inference
##############################

def main_test(model, keys, run_dir, n_samples, n_queries, seed, device='cuda', 
              encoder_idx=None, use_independent_decoder=False, use_task_optimization=True, use_original_bonnet=False):
    """
    Generate new data and evaluate the model on it with key-specific evaluation.
    Uses task-level optimization by default for proper task clustering.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: Number of input-output pairs to generate for support (should match eval_n_samples)
        n_queries: Number of queries to do inference (should match eval_n_queries)
        device: Device to run evaluation on
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
        use_task_optimization: Whether to use task-level optimization (default: True)
        use_original_bonnet: Whether to use original Bonnet approach (per-sample optimization)
    
    Returns:
        dict: Dictionary containing evaluation results structured by key   
    """

    set_seed(seed)
    # Create evaluation mode description
    eval_mode = "PoE" if encoder_idx is None else f"Encoder_{encoder_idx}"
    decoder_mode = "independent" if use_independent_decoder else "shared"
    
    results = {
        'evaluation_metadata': {
            'keys': keys,
            'n_samples_per_key': n_samples,
            'n_queries_per_key': n_queries,
            'max_batch_size': MAX_BATCH_SIZE,
            'device': str(device),
            'seed': seed,
            'evaluation_strategy': 'key_specific_separate_evaluation',
            'latent_type': 'reparameterized',
            'encoder_idx': encoder_idx,
            'use_independent_decoder': use_independent_decoder,
            'evaluation_mode': f"{eval_mode}+{decoder_mode}_decoder"
        },
        'key_results': {},
        'aggregated_metrics': {}
    }
    
    print(f"=== KEY-SPECIFIC EVALUATION CONFIGURATION ===")
    print(f"Keys to evaluate: {keys}")
    print(f"Support samples per key: {n_samples}")
    print(f"Query samples per key: {n_queries}")
    print(f"Evaluation strategy: Separate support->optimization->queries for each key")
    print(f"Maximum batch size: {MAX_BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print(f"Latent representation: Reparameterized (z = mu + sigma * epsilon)")
    print("=" * 50)
    
    # Collect training latent representations once at the beginning
    print(f"\n>>> COLLECTING TRAINING LATENT REPRESENTATIONS <<<")
    training_latent_data = collect_training_latent_representations(model, run_dir, device)
    if training_latent_data:
        print(f"[OK] Collected training latent data: {training_latent_data.get('collection_info', {})}")
        results['training_latent_data'] = training_latent_data
    else:
        print("[WARNING] Warning: Could not collect training latent representations")
    
    # Initialize aggregated metrics
    aggregated_metrics = {
        'total_keys': len(keys),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'average_metrics': {},
        'per_key_summary': {}
    }
    
    # Evaluate each key separately
    for key_idx, key in enumerate(keys):
        print(f"\n[KEY {key_idx+1}/{len(keys)}] EVALUATING '{key}'")
        print("-" * 50)
        
        try:
            # Generate key-specific support samples
            print(f"  Generating {n_samples} support samples and {n_queries} query samples for key '{key}'...")
            all_needed = n_samples + n_queries
            _, _, _, all_input_sequences, all_output_sequences = generate_and_process_tasks(key, all_needed)
            input_samples_sequences = all_input_sequences[:n_samples]
            output_samples_sequences = all_output_sequences[:n_samples]
            input_queries_sequences = all_input_sequences[n_samples:]
            output_queries_sequences = all_output_sequences[n_samples:]
            
            if not input_samples_sequences or not input_queries_sequences:
                print(f"  [ERROR] Failed to generate data for key '{key}' - skipping")
                results['key_results'][key] = {
                    'error': f'Data generation failed for key {key}',
                    'support_samples': 0,
                    'query_samples': 0
                }
                aggregated_metrics['failed_evaluations'] += 1
                continue
            
            # Create dataloaders for this key with actual keys
            support_batch_size = min(MAX_BATCH_SIZE, n_samples)
            query_batch_size = min(MAX_BATCH_SIZE, n_queries)
            
            # Use the actual key for all samples of this key
            support_keys = [key] * len(input_samples_sequences)
            query_keys = [key] * len(input_queries_sequences)
            
            samples_dataloader, support_key_mapping = prepare_dataloader_with_keys(
                input_samples_sequences, output_samples_sequences, support_keys,
                batch_size=support_batch_size, shuffle=False)
            queries_dataloader, query_key_mapping = prepare_dataloader_with_keys(
                input_queries_sequences, output_queries_sequences, query_keys,
                batch_size=query_batch_size, shuffle=False)
            
            print(f"  Support: {len(input_samples_sequences)} samples, batch size: {support_batch_size}")
            print(f"  Query: {len(input_queries_sequences)} samples, batch size: {query_batch_size}")
            
            # Evaluate model on this specific key
            print(f"  Running evaluation for key '{key}' ({eval_mode} + {decoder_mode} decoder)...")
            if use_original_bonnet:
                print(f"  Using original Bonnet approach (per-sample optimization)")
                key_metrics = evaluate_model_original_bonnet_approach(model, samples_dataloader, queries_dataloader, device=device,
                                                                   encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder,
                                                                   support_key_mapping=support_key_mapping, query_key_mapping=query_key_mapping)
            elif use_task_optimization:
                print(f"  Using task-level optimization")
                key_metrics = evaluate_model_with_task_optimization(model, samples_dataloader, queries_dataloader, device=device,
                                                                   encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder,
                                                                   support_key_mapping=support_key_mapping, query_key_mapping=query_key_mapping)
            else:
                print(f"  Using regular evaluation (no optimization)")
                key_metrics = evaluate_model(model, samples_dataloader, queries_dataloader, device=device,
                                           encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder,
                                           evaluated_key=key)
            
            # Store key-specific results with structured information
            results['key_results'][key] = {
                'key': key,
                'support_samples': len(input_samples_sequences),
                'query_samples': len(input_queries_sequences),
                'metrics': key_metrics.get('metrics', {}),
                'reconstruction_results': key_metrics.get('reconstruction_results', []),
                'evaluation_latent_data': key_metrics.get('evaluation_latent_data', {}),
                'trajectory_info': key_metrics.get('trajectory_info', []),
                'task_latent_data': key_metrics.get('task_latent_data', {}),
                'latent_data': key_metrics.get('latent_data', {}),  # For original Bonnet approach
                'evaluation_method': key_metrics.get('evaluation_method', 'unknown'),
                'raw_data': {
                    'input_samples_sequences': input_samples_sequences,
                    'output_samples_sequences': output_samples_sequences,
                    'input_queries_sequences': input_queries_sequences,
                    'output_queries_sequences': output_queries_sequences
                }
            }
            
            # Add training latent data reference to each key
            if training_latent_data:
                results['key_results'][key]['training_latent_data'] = training_latent_data
            
            aggregated_metrics['successful_evaluations'] += 1
            
            # Collect metrics for aggregation
            if 'error' not in key_metrics.get('metrics', {}):
                metrics = key_metrics.get('metrics', {})
                key_summary = {
                    'support_loss': metrics.get('support_loss', 0.0),
                    'query_loss': metrics.get('query_loss', 0.0),
                    'shape_accuracy': metrics.get('shape_accuracy', 0.0),
                    'grid_accuracy': metrics.get('grid_accuracy', 0.0),
                    'sample_exact_accuracy': metrics.get('sample_exact_accuracy', 0.0),
                    'exact_accuracy': metrics.get('exact_accuracy', 0.0),  # For original Bonnet approach
                    'trajectory_samples': len(key_metrics.get('trajectory_info', []))
                }
                aggregated_metrics['per_key_summary'][key] = key_summary
                
                # Print key summary
                print(f"  [OK] Key '{key}' Results:")
                print(f"    Support loss: {key_summary['support_loss']:.4f}")
                print(f"    Query loss: {key_summary['query_loss']:.4f}")
                print(f"    Shape accuracy: {key_summary['shape_accuracy']:.4f}")
                print(f"    Grid accuracy: {key_summary['grid_accuracy']:.4f}")
                if 'exact_accuracy' in key_summary:
                    print(f"    Exact accuracy: {key_summary['exact_accuracy']:.4f}")
                else:
                    print(f"    Sample exact accuracy: {key_summary['sample_exact_accuracy']:.4f}")
                print(f"    Trajectory samples: {key_summary['trajectory_samples']}")
            else:
                print(f"  [ERROR] Key '{key}' Error: {key_metrics['metrics']['error']}")
                aggregated_metrics['failed_evaluations'] += 1
                
        except Exception as e:
            print(f"  [ERROR] Exception during evaluation of key '{key}': {e}")
            results['key_results'][key] = {
                'error': f'Exception during evaluation: {str(e)}',
                'support_samples': 0,
                'query_samples': 0
            }
            aggregated_metrics['failed_evaluations'] += 1
    
    # Calculate aggregated metrics across all successful keys
    if aggregated_metrics['per_key_summary']:
        successful_keys = list(aggregated_metrics['per_key_summary'].keys())
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['support_loss', 'query_loss', 'shape_accuracy', 'grid_accuracy', 'sample_exact_accuracy']:
            values = [aggregated_metrics['per_key_summary'][key][metric] for key in successful_keys]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
            avg_metrics[f'min_{metric}'] = min(values)
            avg_metrics[f'max_{metric}'] = max(values)
            avg_metrics[f'std_{metric}'] = (sum((v - avg_metrics[f'avg_{metric}'])**2 for v in values) / len(values))**0.5
        
        aggregated_metrics['average_metrics'] = avg_metrics
        aggregated_metrics['successful_keys'] = successful_keys
    
    results['aggregated_metrics'] = aggregated_metrics
    
    print(f"\n=== KEY-SPECIFIC EVALUATION COMPLETE ===")
    print(f"Total keys processed: {aggregated_metrics['total_keys']}")
    print(f"Successful evaluations: {aggregated_metrics['successful_evaluations']}")
    print(f"Failed evaluations: {aggregated_metrics['failed_evaluations']}")
    
    if aggregated_metrics['average_metrics']:
        print(f"\nAggregated Results Across {len(aggregated_metrics['successful_keys'])} Keys:")
        avg_metrics = aggregated_metrics['average_metrics']
        print(f"  Average shape accuracy: {avg_metrics['avg_shape_accuracy']:.4f} ± {avg_metrics['std_shape_accuracy']:.4f}")
        print(f"  Average grid accuracy: {avg_metrics['avg_grid_accuracy']:.4f} ± {avg_metrics['std_grid_accuracy']:.4f}")
        print(f"  Average sample exact accuracy: {avg_metrics['avg_sample_exact_accuracy']:.4f} ± {avg_metrics['std_sample_exact_accuracy']:.4f}")
        print(f"  Average support loss: {avg_metrics['avg_support_loss']:.4f} ± {avg_metrics['std_support_loss']:.4f}")
        print(f"  Average query loss: {avg_metrics['avg_query_loss']:.4f} ± {avg_metrics['std_query_loss']:.4f}")
    
    # Store evaluation latent data at the top level for visualization
    if use_task_optimization:
        # Collect all task latent data from key results
        all_task_latent_data = {}
        all_task_trajectories = {}
        
        for key, key_results in results['key_results'].items():
            if 'task_latent_data' in key_results:
                all_task_latent_data.update(key_results['task_latent_data'].get('task_latents', {}))
            if 'task_trajectories' in key_results:
                all_task_trajectories.update(key_results['task_trajectories'])
        
        # Store at top level for visualization
        results['task_latent_data'] = {
            'task_latents': all_task_latent_data
        }
        results['task_trajectories'] = all_task_trajectories
        
        # Also store evaluation_latent_data at top level
        # Check if any key has evaluation_latent_data
        first_key = list(results['key_results'].keys())[0]
        if 'evaluation_latent_data' in results['key_results'][first_key]:
            results['evaluation_latent_data'] = results['key_results'][first_key]['evaluation_latent_data']
    
    return results


def evaluate_model(model, samples_dataloader, queries_dataloader, device='cuda',
                  encoder_idx=None, use_independent_decoder=False, evaluated_key=None):
    """
    Evaluate model performance on dataloaders.
    For multi-encoder models: each encoder processes the same sample, then PoE combines them.
    
    Args:
        model: The trained model to evaluate (single or multi-encoder)
        samples_dataloader: DataLoader containing support samples for latent optimization
        queries_dataloader: DataLoader containing query samples for evaluation
        device: Device to run evaluation on
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
        
    Returns:
        dict: Dictionary containing evaluation metrics and reconstruction results
    """
    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']
    
    # Check if this is a multi-encoder model
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
    
    # Create evaluation mode description
    eval_mode = "PoE" if encoder_idx is None else f"Encoder_{encoder_idx}"
    decoder_mode = "independent" if use_independent_decoder else "shared"
    
    print(f"=== EVALUATION SETUP ===")
    print(f"Model type: {'Multi-encoder' if is_multi_encoder else 'Single encoder'}")
    if is_multi_encoder:
        print(f"Number of encoders: {num_encoders}")
    print(f"Evaluation mode: {eval_mode} + {decoder_mode} decoder")
    print(f"Latent optimization enabled: {optimize_z_inference}")
    if optimize_z_inference:
        print(f"Optimization steps: {latent_optimization['inference']['num_steps']}")
        print(f"Optimization learning rate: {latent_optimization['inference']['learning_rate']}")

    model.eval()
    
    # Initialize metrics storage
    if is_multi_encoder:
        # For multi-encoder: track individual encoder metrics + PoE metrics
        individual_encoder_metrics = {
            f'encoder_{i}': {
                'shape_correct': 0, 'shape_tokens': 0,
                'grid_correct': 0, 'grid_tokens': 0,
                'sample_exact_correct': 0, 'total_samples': 0
            } for i in range(num_encoders)
        }
        poe_metrics = {
            'shape_correct': 0, 'shape_tokens': 0,
            'grid_correct': 0, 'grid_tokens': 0,
            'sample_exact_correct': 0, 'total_samples': 0
        }
        # Encoder covariance traces storage
        encoder_covariance_traces = []
    else:
        # For single encoder: use existing structure
        single_metrics = {
            'shape_correct': 0, 'shape_tokens': 0,
            'grid_correct': 0, 'grid_tokens': 0,
            'sample_exact_correct': 0, 'total_samples': 0
        }
    
    support_losses = []
    query_losses = []
    support_reconstructions = []
    query_reconstructions = []
    z_optimization_logs = []
    trajectory_info = []
    
    # Store encoder-specific results for multi-encoder
    if is_multi_encoder:
        individual_support_reconstructions = {f'encoder_{i}': [] for i in range(num_encoders)}
        individual_query_reconstructions = {f'encoder_{i}': [] for i in range(num_encoders)}
        poe_support_reconstructions = []
        poe_query_reconstructions = []

    # Collect all support samples and their optimized z vectors
    all_support_z_vectors = []
    
    print("\n>>> PROCESSING SUPPORT SAMPLES <<<")
    total_support_samples = sum(len(batch[0]) if isinstance(batch, (list, tuple)) and len(batch) >= 2 else batch[0].size(0) for batch in samples_dataloader)
    processed_support_samples = 0
    
    # Create progress bar for support batches
    support_pbar = tqdm(samples_dataloader, desc="Support batches", unit="batch")
    
    # Store support sample latent vectors for trajectory consistency
    support_latent_vectors = []
    
    for batch_idx, batch in enumerate(support_pbar):
        # Handle batch structure with keys
        if len(batch) >= 3:
            batch_input_s, batch_target_s, batch_keys = batch[:3]
        else:
            batch_input_s, batch_target_s = batch[:2]
            batch_keys = None
        
        batch_input_s = batch_input_s.to(device)
        batch_target_s = batch_target_s.to(device)
        batch_size = batch_input_s.size(0)
        
        # Update progress bar description
        support_pbar.set_description(f"Support batch {batch_idx+1}/{len(samples_dataloader)} (size: {batch_size})")
        
        # FIRST: Compute support sample latent vectors with consistent random seed
        print(f"  Computing support sample latent vectors for batch {batch_idx+1}...")
        batch_support_latents = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Set consistent random seed for reparameterization
                torch.manual_seed(42 + processed_support_samples + i)  # Consistent seed
                
                if is_multi_encoder:
                    # For multi-encoder, compute PoE latent
                    if encoder_idx is not None:
                        # Use specific encoder
                        mu, log_var, _ = model.multi_encoder.encoders[encoder_idx](
                            batch_input_s[i:i+1], batch_target_s[i:i+1]
                        )
                    else:
                        # Use PoE inference mode
                        result = model(batch_input_s[i:i+1], batch_target_s[i:i+1])
                        (shape_logits, grid_logits), mu, log_var, _ = result
                else:
                    # For single encoder
                    mu, log_var, _ = model.encoder(batch_input_s[i:i+1], batch_target_s[i:i+1])
                
                # CRITICAL FIX: Ensure mu and log_var have correct shape [1, latent_dim]
                if mu.dim() > 2:
                    mu = mu.squeeze()  # Remove extra dimensions
                if log_var.dim() > 2:
                    log_var = log_var.squeeze()  # Remove extra dimensions
                
                # Ensure they have batch dimension
                if mu.dim() == 1:
                    mu = mu.unsqueeze(0)  # Add batch dimension
                if log_var.dim() == 1:
                    log_var = log_var.unsqueeze(0)  # Add batch dimension
                
                # Use reparameterization trick with consistent seed
                z = model.reparameterize(mu, log_var)
                
                # CRITICAL FIX: Ensure z has correct shape [1, latent_dim]
                if z.dim() > 2:
                    z = z.squeeze()  # Remove extra dimensions
                if z.dim() == 1:
                    z = z.unsqueeze(0)  # Add batch dimension
                
                batch_support_latents.append(z.detach().clone())
        
        support_latent_vectors.extend(batch_support_latents)
        
        current_z_for_this_sample_batch = None
        
        if optimize_z_inference:
            print(f"  Optimizing latent z for support batch {batch_idx+1}...")
            
            # CRITICAL FIX: Use the exact same support sample latent vectors as starting points
            # Convert stored latent vectors to the format expected by optimization
            print(f"DEBUG: batch_support_latents type: {type(batch_support_latents)}")
            print(f"DEBUG: batch_support_latents length: {len(batch_support_latents)}")
            for i, latent in enumerate(batch_support_latents):
                print(f"DEBUG: batch_support_latents[{i}] type: {type(latent)}, shape: {latent.shape if hasattr(latent, 'shape') else 'no shape'}")
                if hasattr(latent, 'cpu'):
                    print(f"DEBUG: batch_support_latents[{i}] is tensor")
                else:
                    print(f"DEBUG: batch_support_latents[{i}] is NOT tensor!")
            
            support_z_tensor = torch.cat(batch_support_latents, dim=0)  # Use cat instead of stack
            print(f"DEBUG: support_z_tensor type: {type(support_z_tensor)}, shape: {support_z_tensor.shape}")
            
            # Ensure support_z_tensor is a proper tensor
            if not isinstance(support_z_tensor, torch.Tensor):
                print(f"ERROR: support_z_tensor is not a tensor! Type: {type(support_z_tensor)}")
                # Try to convert to tensor
                support_z_tensor = torch.tensor(support_z_tensor, dtype=torch.float32, device=device)
                print(f"DEBUG: Converted to tensor, shape: {support_z_tensor.shape}")
            
            # Latent optimization starting from the exact support sample latent vectors
            z_optimized, losses, trajectory = get_optimized_z_from_initial(
                model, batch_input_s, batch_target_s, 
                initial_z=support_z_tensor,  # Use exact support sample latents
                num_steps=latent_optimization['inference']['num_steps'],
                lr=latent_optimization['inference']['learning_rate'],
                context='inference',
                return_trajectory=True,
                encoder_idx=encoder_idx,
                use_independent_decoder=use_independent_decoder
            )
            current_z_for_this_sample_batch = z_optimized
            z_optimization_logs.append(trajectory.get('losses', []) if trajectory else losses)
            
            # Store trajectory information for each sample
            if trajectory:
                for i in range(batch_size):
                    sample_trajectory = {
                        'input_sample': batch_input_s[i].detach().cpu().numpy(),
                        'target_sample': batch_target_s[i].detach().cpu().numpy(),
                        'z_vectors': [],
                        'losses': [],
                        'is_multi_encoder': is_multi_encoder,
                        'num_encoders': num_encoders if is_multi_encoder else 1,
                        'individual_encoder_trajectories': {},
                        'support_latent_vector': batch_support_latents[i].cpu().numpy(),  # Store the exact support latent
                        'evaluated_key': evaluated_key  # Store the key being evaluated
                    }
                    
                    # Always create individual encoder trajectories for unified processing
                    sample_trajectory['individual_encoder_trajectories'] = {}
                    
                    if is_multi_encoder:
                        # Get individual encoder z vectors for this sample during optimization
                        with torch.no_grad():
                            for enc_idx in range(num_encoders):
                                # Use the exact same support latent for individual encoders too
                                enc_z = batch_support_latents[i]  # Use the same support latent
                                
                                sample_trajectory['individual_encoder_trajectories'][f'encoder_{enc_idx}'] = {
                                    'mu': batch_support_latents[i].cpu().numpy(),  # Store the support latent
                                    'log_var': batch_support_latents[i].cpu().numpy(),  # Placeholder
                                    'z': enc_z.cpu().numpy()
                                }
                        
                        # Store individual encoder reconstructions for comparison
                        sample_trajectory['individual_encoder_reconstructions'] = {}
                        
                        # Get trajectory decoder type setting (using global settings import)
                        eval_settings = settings.get_evaluation_settings()
                        decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
                        use_independent_decoder = (decoder_type == 'independent')
                        
                        # Add decoder type metadata to trajectory
                        sample_trajectory['trajectory_decoder_type'] = decoder_type
                        sample_trajectory['used_independent_decoder'] = use_independent_decoder
                        
                        with torch.no_grad():
                            for enc_idx in range(num_encoders):
                                enc_data = sample_trajectory['individual_encoder_trajectories'][f'encoder_{enc_idx}']
                                enc_z_tensor = torch.tensor(enc_data['z']).to(device)
                                
                                try:
                                    # Generate reconstruction using appropriate decoder based on setting
                                    if use_independent_decoder:
                                        # Use individual encoder's independent decoder
                                        enc_shape_logits, enc_grid_logits = model.multi_encoder.independent_decoders[enc_idx](
                                            enc_z_tensor, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                        )
                                    else:
                                        # Use shared decoder (default behavior)
                                        enc_shape_logits, enc_grid_logits = model.multi_encoder.shared_decoder(
                                            enc_z_tensor, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                        )
                                    
                                    sample_trajectory['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = {
                                        'shape_logits': enc_shape_logits.cpu().numpy(),
                                        'grid_logits': enc_grid_logits.cpu().numpy()
                                    }
                                except Exception as e:
                                    decoder_name = f"independent decoder {enc_idx}" if use_independent_decoder else "shared decoder"
                                    print(f"    Warning: Could not generate reconstruction for encoder {enc_idx} with {decoder_name}: {e}")
                                    sample_trajectory['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = None
                    else:
                        # Single encoder: create encoder_0 entry for unified processing
                        # IMPORTANT: For single encoder, individual encoder should match PoE start
                        if 'z_vectors' in trajectory and len(trajectory['z_vectors']) > 0:
                            # Use the exact same initial z that started the PoE optimization
                            initial_poe_z = trajectory['z_vectors'][0][i].detach().cpu().numpy()
                            
                            # Get encoder mu/log_var for metadata (but use PoE's z)
                            with torch.no_grad():
                                enc_mu, enc_log_var, _ = model.encoder(batch_input_s[i:i+1], batch_target_s[i:i+1])
                            
                            sample_trajectory['individual_encoder_trajectories']['encoder_0'] = {
                                'mu': enc_mu.cpu().numpy(),
                                'log_var': enc_log_var.cpu().numpy(), 
                                'z': initial_poe_z  # Use SAME z as PoE trajectory start
                            }
                        else:
                            # Fallback: independent z (should not happen normally)
                            with torch.no_grad():
                                enc_mu, enc_log_var, _ = model.encoder(batch_input_s[i:i+1], batch_target_s[i:i+1])
                                enc_z = model.reparameterize(enc_mu, enc_log_var)
                                
                                sample_trajectory['individual_encoder_trajectories']['encoder_0'] = {
                                    'mu': enc_mu.cpu().numpy(),
                                    'log_var': enc_log_var.cpu().numpy(),
                                    'z': enc_z.cpu().numpy()
                                }
                        
                        # Store individual encoder reconstructions for single encoder too
                        sample_trajectory['individual_encoder_reconstructions'] = {}
                        
                        # Get trajectory decoder type setting
                        eval_settings = settings.get_evaluation_settings()
                        decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
                        
                        # Add decoder type metadata to trajectory
                        sample_trajectory['trajectory_decoder_type'] = decoder_type
                        
                        # Generate individual encoder reconstruction for single encoder
                        with torch.no_grad():
                            enc_data = sample_trajectory['individual_encoder_trajectories']['encoder_0']
                            enc_z_tensor = torch.tensor(enc_data['z']).to(device)
                            
                            try:
                                # Generate reconstruction using the decoder
                                enc_shape_logits, enc_grid_logits = model.decoder(
                                    enc_z_tensor, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                )
                                
                                sample_trajectory['individual_encoder_reconstructions']['encoder_0'] = {
                                    'shape_logits': enc_shape_logits.cpu().numpy(),
                                    'grid_logits': enc_grid_logits.cpu().numpy()
                                }
                            except Exception as e:
                                print(f"    Warning: Could not generate reconstruction for encoder 0: {e}")
                                sample_trajectory['individual_encoder_reconstructions']['encoder_0'] = None
                    
                    # Convert PoE z vectors to numpy (main trajectory)
                    if 'z_vectors' in trajectory:
                        print(f"DEBUG: Processing trajectory z_vectors for sample {i}")
                        for z_step in trajectory['z_vectors']:
                            # Ensure z_step[i] is a tensor before converting to numpy
                            if hasattr(z_step[i], 'detach'):
                                # It's a tensor
                                z_numpy = z_step[i].detach().cpu().numpy()
                            else:
                                # If it's already a numpy array, ensure it's the right format
                                z_numpy = np.array(z_step[i], dtype=np.float32)
                            sample_trajectory['z_vectors'].append(z_numpy)
                        print(f"  Sample {processed_support_samples + i + 1}: {len(sample_trajectory['z_vectors'])} trajectory steps stored")
                        
                        # Store trajectory reconstructions at key points (PoE for multi-encoder, main trajectory for single-encoder)
                        if len(sample_trajectory['z_vectors']) > 1:
                            print(f"DEBUG: Starting trajectory reconstruction for sample {i}")
                            sample_trajectory['poe_trajectory_reconstructions'] = {}
                            key_indices = [0, len(sample_trajectory['z_vectors'])//2, len(sample_trajectory['z_vectors'])-1]
                            key_labels = ['initial', 'mid', 'final']
                            
                            with torch.no_grad():
                                for idx, label in zip(key_indices, key_labels):
                                    if idx < len(sample_trajectory['z_vectors']):
                                        print(f"DEBUG: Processing trajectory step {idx} ({label}) for sample {i}")
                                        try:
                                            # Ensure the trajectory z vector is properly formatted
                                            z_vector = sample_trajectory['z_vectors'][idx]
                                            print(f"DEBUG: z_vector type: {type(z_vector)}, shape: {z_vector.shape if hasattr(z_vector, 'shape') else 'no shape'}")
                                            
                                            if isinstance(z_vector, np.ndarray):
                                                # Ensure it's float32 and has the right shape
                                                z_vector = z_vector.astype(np.float32)
                                                if z_vector.ndim == 1:
                                                    z_vector = z_vector.reshape(1, -1)
                                            else:
                                                # Convert to numpy array if it's not already
                                                z_vector = np.array(z_vector, dtype=np.float32)
                                                if z_vector.ndim == 1:
                                                    z_vector = z_vector.reshape(1, -1)
                                            
                                            print(f"DEBUG: Converting z_vector to tensor, shape: {z_vector.shape}")
                                            traj_z = torch.tensor(z_vector, dtype=torch.float32).to(device)
                                            print(f"DEBUG: Successfully converted to tensor, shape: {traj_z.shape}")
                                            
                                            # Ensure correct shape for decoder input
                                            if traj_z.dim() == 1:
                                                traj_z = traj_z.unsqueeze(0)  # Add batch dimension
                                            elif traj_z.dim() > 2:
                                                traj_z = traj_z.squeeze()  # Remove extra dimensions
                                                if traj_z.dim() == 1:
                                                    traj_z = traj_z.unsqueeze(0)
                                            print(f"DEBUG: Final tensor shape for decoder: {traj_z.shape}")
                                            
                                            if is_multi_encoder:
                                                # Multi-encoder: use decoder based on setting
                                                if use_independent_decoder:
                                                    # Note: For PoE trajectory, we use the shared decoder even with independent setting
                                                    # since PoE latent is a fusion that doesn't belong to any specific encoder
                                                    print(f"    Note: Using shared decoder for PoE trajectory (PoE latent is encoder-agnostic)")
                                                    traj_shape_logits, traj_grid_logits = model.multi_encoder.shared_decoder(
                                                        traj_z, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                                    )
                                                else:
                                                    # Use shared decoder (default)
                                                    traj_shape_logits, traj_grid_logits = model.multi_encoder.shared_decoder(
                                                        traj_z, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                                    )
                                            else:
                                                # Single encoder: use regular decoder
                                                traj_shape_logits, traj_grid_logits = model.decoder(
                                                    traj_z, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                                )
                                            
                                            sample_trajectory['poe_trajectory_reconstructions'][label] = {
                                                'step': idx,
                                                'shape_logits': traj_shape_logits.cpu().numpy(),
                                                'grid_logits': traj_grid_logits.cpu().numpy()
                                            }
                                            print(f"DEBUG: Successfully stored {label} reconstruction")
                                        except Exception as e:
                                            reconstruction_type = "PoE" if is_multi_encoder else "trajectory"
                                            decoder_info = f" (decoder: {decoder_type})" if is_multi_encoder else ""
                                            print(f"    Warning: Could not generate {reconstruction_type} reconstruction at step {idx}{decoder_info}: {e}")
                                            sample_trajectory['poe_trajectory_reconstructions'][label] = None
                    
                    # Use individual trajectory losses if available, otherwise fall back to batch average
                    if 'individual_losses' in trajectory and trajectory['individual_losses']:
                        # Use individual sample losses for this specific sample
                        sample_trajectory['losses'] = [step_losses[i] for step_losses in trajectory['individual_losses']]
                        
                        # Add loss improvement statistics for this specific sample
                        sample_losses = sample_trajectory['losses']
                        if len(sample_losses) > 1:
                            initial_loss = sample_losses[0]
                            final_loss = sample_losses[-1]
                            sample_trajectory['loss_improvement'] = initial_loss - final_loss
                            sample_trajectory['loss_improvement_percent'] = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0.0
                    elif 'losses' in trajectory:
                        # Fallback to batch average (for compatibility with older code)
                        sample_trajectory['losses'] = trajectory['losses'] if isinstance(trajectory['losses'], list) else [trajectory['losses']]
                        
                        # Add loss improvement statistics
                        if len(sample_trajectory['losses']) > 1:
                            initial_loss = sample_trajectory['losses'][0]
                            final_loss = sample_trajectory['losses'][-1]
                            sample_trajectory['loss_improvement'] = initial_loss - final_loss
                            sample_trajectory['loss_improvement_percent'] = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0.0
                    
                    trajectory_info.append(sample_trajectory)
                    print(f"DEBUG: Added sample {i} trajectory to trajectory_info")
                    
                    processed_support_samples += 1
                    
                    # Store the optimized z vector for this sample
                    if current_z_for_this_sample_batch is not None:
                        print(f"DEBUG: current_z_for_this_sample_batch shape: {current_z_for_this_sample_batch.shape}")
                        print(f"DEBUG: Extracting z vector for sample {i}")
                        
                        # Check if it's already a numpy array or a tensor
                        if hasattr(current_z_for_this_sample_batch[i], 'detach'):
                            # It's a tensor
                            z_np = current_z_for_this_sample_batch[i].detach().cpu().numpy()
                            print(f"DEBUG: Storing optimized z[{i}] (tensor->numpy) shape: {z_np.shape}")
                            all_support_z_vectors.append(z_np)
                        else:
                            # It's already a numpy array
                            z_data = current_z_for_this_sample_batch[i]
                            print(f"DEBUG: Storing optimized z[{i}] (already numpy) shape: {z_data.shape}, dtype: {z_data.dtype}")
                            all_support_z_vectors.append(z_data)
                    else:
                        # If no optimization, use the support sample latent
                        z_np = batch_support_latents[i].cpu().numpy()
                        print(f"DEBUG: Storing support z[{i}] (no optimization) shape: {z_np.shape}")
                        all_support_z_vectors.append(z_np)
        
        print(f"DEBUG: Finished processing support batch {batch_idx+1}")
        print(f"DEBUG: trajectory_info length: {len(trajectory_info)}")
        
        # Calculate support loss
        print(f"DEBUG: Calculating support loss")
        s_loss_val = compute_loss(model, batch_input_s, batch_target_s)
        support_losses.append(s_loss_val.item())
        print(f"DEBUG: Support loss calculated: {s_loss_val.item()}")

        # Store reconstructions - handle both single and multi-encoder
        if current_z_for_this_sample_batch is not None:
            with torch.no_grad():
                if is_multi_encoder:
                    # PoE reconstruction using the combined latent
                    shape_logits_s, grid_logits_s = model.multi_encoder.decoder(
                        current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s
                    )
                    
                    # Store PoE reconstruction
                    for j in range(batch_input_s.size(0)):
                        poe_reconstruction = {
                            'input': batch_input_s[j].detach().cpu().numpy().tolist(),
                            'target': batch_target_s[j].detach().cpu().numpy().tolist(),
                            'reconstruction': (shape_logits_s[j].detach().cpu().numpy().tolist(), 
                                             grid_logits_s[j].detach().cpu().numpy().tolist())
                        }
                        poe_support_reconstructions.append(poe_reconstruction)
                    
                    # Also store individual encoder reconstructions for analysis
                    for enc_idx in range(num_encoders):
                        enc_mu, enc_log_var,_ = model.multi_encoder.encoders[enc_idx](batch_input_s, batch_target_s)
                        enc_z = model.reparameterize(enc_mu, enc_log_var)
                        enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                            enc_z, batch_input_s, target_seq=batch_target_s
                        )
                        
                        for j in range(batch_input_s.size(0)):
                            enc_reconstruction = {
                                'input': batch_input_s[j].detach().cpu().numpy().tolist(),
                                'target': batch_target_s[j].detach().cpu().numpy().tolist(),
                                'reconstruction': (enc_shape_logits[j].detach().cpu().numpy().tolist(),
                                                 enc_grid_logits[j].detach().cpu().numpy().tolist())
                            }
                            individual_support_reconstructions[f'encoder_{enc_idx}'].append(enc_reconstruction)
                
                else:
                    # Single encoder reconstruction
                    shape_logits_s, grid_logits_s = model.decoder(
                        current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s
                    )
                    
                    for j in range(batch_input_s.size(0)):
                        reconstruction = {
                            'input': batch_input_s[j].detach().cpu().numpy().tolist(),
                            'target': batch_target_s[j].detach().cpu().numpy().tolist(),
                            'reconstruction': (shape_logits_s[j].detach().cpu().numpy().tolist(), 
                                             grid_logits_s[j].detach().cpu().numpy().tolist())
                        }
                        support_reconstructions.append(reconstruction)
        
        processed_support_samples += batch_size
        
        # Update progress bar postfix
        support_pbar.set_postfix({
            'samples': f'{processed_support_samples}/{total_support_samples}',
            'loss': f'{s_loss_val.item():.4f}',
            'z_stored': len(all_support_z_vectors)
        })
    
    support_pbar.close()
    
    print(f"\nSupport processing complete:")
    print(f"  Total samples processed: {processed_support_samples}")
    print(f"  Total batches processed: {len(samples_dataloader)}")
    print(f"  Z vectors collected: {len(all_support_z_vectors)}")
    print(f"  Trajectory info samples: {len(trajectory_info)}")

    # Create prototype z for queries by averaging all support z vectors
    if all_support_z_vectors:
        print(f"DEBUG: Processing {len(all_support_z_vectors)} support z vectors")
        
        # Convert numpy arrays to tensors before concatenating
        support_z_tensors = []
        for i, z_vec in enumerate(all_support_z_vectors):
            if isinstance(z_vec, np.ndarray):
                print(f"DEBUG: z_vec[{i}] (numpy) shape: {z_vec.shape}, dtype: {z_vec.dtype}")
                z_tensor = torch.tensor(z_vec, dtype=torch.float32, device=device)
                # Ensure 2D shape [1, latent_dim] for proper stacking
                if z_tensor.dim() == 1:
                    z_tensor = z_tensor.unsqueeze(0)  # Add batch dimension
                print(f"DEBUG: z_tensor[{i}] (after conversion and reshaping) shape: {z_tensor.shape}")
                support_z_tensors.append(z_tensor)
            else:
                print(f"DEBUG: z_vec[{i}] (tensor) shape: {z_vec.shape}")
                z_tensor = z_vec.to(device)
                # Ensure 2D shape [1, latent_dim] for proper stacking
                if z_tensor.dim() == 1:
                    z_tensor = z_tensor.unsqueeze(0)  # Add batch dimension
                support_z_tensors.append(z_tensor)
        
        print(f"DEBUG: About to concatenate {len(support_z_tensors)} tensors")
        for i, tensor in enumerate(support_z_tensors):
            print(f"DEBUG: support_z_tensor[{i}] shape: {tensor.shape}")
        
        # Concatenate all support z vectors (each is now [1, latent_dim]) and take mean
        combined_support_z = torch.cat(support_z_tensors, dim=0)
        print(f"DEBUG: combined_support_z shape after cat: {combined_support_z.shape}")
        
        z_for_queries_prototype = combined_support_z.mean(dim=0, keepdim=True).to(device)
        print(f"  Created prototype z from {combined_support_z.size(0)} support samples")
        print(f"  Prototype z shape: {z_for_queries_prototype.shape}")
    else:
        print("ERROR: No z vectors obtained from support samples")
        return {
            'metrics': {
                'error': 'No z from support samples for query evaluation',
                'support_loss': sum(support_losses)/len(support_losses) if support_losses else 0,
                'query_loss': 0, 'shape_accuracy': 0, 'grid_accuracy': 0, 'overall_accuracy': 0,
                'sample_exact_accuracy': 0, 'losses_gradient_ascent': z_optimization_logs,
                'used_latent_optimization': optimize_z_inference,
                'trajectory_info': trajectory_info
            },
            'reconstruction_results': {'support_reconstructions': support_reconstructions, 'query_reconstructions': []}
        }

    print("\n>>> PROCESSING QUERY SAMPLES <<<")
    total_query_samples = sum(len(batch[0]) if isinstance(batch, (list, tuple)) and len(batch) >= 2 else batch[0].size(0) for batch in queries_dataloader)
    processed_query_samples = 0
    
    # Create progress bar for query batches
    query_pbar = tqdm(queries_dataloader, desc="Query batches", unit="batch")
    
    # Store query data for visualization
    query_data_for_trajectory = []
    
    for batch_idx, batch in enumerate(query_pbar):
        # Handle batch structure with keys
        if len(batch) >= 3:
            batch_input_q, batch_target_q, batch_keys = batch[:3]
        else:
            batch_input_q, batch_target_q = batch[:2]
            batch_keys = None
        
        batch_input_q = batch_input_q.to(device)
        batch_target_q = batch_target_q.to(device)
        query_batch_size = batch_input_q.size(0)
        
        # Update progress bar description
        query_pbar.set_description(f"Query batch {batch_idx+1}/{len(queries_dataloader)} (size: {query_batch_size})")
            
        with torch.no_grad():
            # Expand the prototype z to match query batch size
            z_query_expanded = z_for_queries_prototype.expand(query_batch_size, -1)
            
            if is_multi_encoder:
                # Multi-encoder evaluation: individual encoders + PoE
                
                # 1. PoE evaluation using the prototype z
                poe_shape_logits, poe_grid_logits = model.multi_encoder.decoder(
                    z_query_expanded, batch_input_q, target_seq=batch_target_q
                )
                
                # Calculate PoE metrics
                poe_shape_pred = poe_shape_logits.argmax(dim=-1)
                poe_grid_pred = poe_grid_logits.argmax(dim=-1)
                shape_tgt = batch_target_q[:, 900:902].long()
                grid_tgt = batch_target_q[:, :900].long()

                batch_poe_shape_correct = (poe_shape_pred == shape_tgt).sum().item()
                batch_poe_shape_tokens = shape_tgt.numel()
                batch_poe_grid_correct = 0
                batch_poe_grid_tokens = 0
                batch_poe_exact_correct = 0

                poe_metrics['shape_correct'] += batch_poe_shape_correct
                poe_metrics['shape_tokens'] += batch_poe_shape_tokens

                for i in range(query_batch_size):
                    tgt_rows = int(batch_target_q[i, 900].item())
                    tgt_cols = int(batch_target_q[i, 901].item())
                    active_pixels = tgt_rows * tgt_cols
                    if active_pixels > 0:
                        sample_poe_grid_correct = (poe_grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                        batch_poe_grid_correct += sample_poe_grid_correct
                        batch_poe_grid_tokens += active_pixels
                        if torch.all(poe_shape_pred[i] == shape_tgt[i]) and sample_poe_grid_correct == active_pixels:
                            batch_poe_exact_correct += 1
                    elif torch.all(poe_shape_pred[i] == shape_tgt[i]):
                         batch_poe_exact_correct += 1

                poe_metrics['grid_correct'] += batch_poe_grid_correct
                poe_metrics['grid_tokens'] += batch_poe_grid_tokens
                poe_metrics['sample_exact_correct'] += batch_poe_exact_correct
                poe_metrics['total_samples'] += query_batch_size
                
                # Store PoE query reconstructions
                for i in range(query_batch_size):
                    poe_query_reconstructions.append({
                        'input': batch_input_q[i].cpu().numpy().tolist(),
                        'target': batch_target_q[i].cpu().numpy().tolist(),
                        'reconstruction': (poe_shape_logits[i].cpu().numpy().tolist(), 
                                         poe_grid_logits[i].cpu().numpy().tolist())
                    })
                    
                    # Store query data for trajectory visualization using global index
                    global_query_idx = processed_query_samples + i
                    if global_query_idx < len(trajectory_info):  # Ensure we have a corresponding trajectory entry
                        trajectory_info[global_query_idx]['query_input'] = batch_input_q[i].detach().cpu().numpy()
                        trajectory_info[global_query_idx]['query_target'] = batch_target_q[i].detach().cpu().numpy()
                        
                        # Store query POE reconstruction
                        trajectory_info[global_query_idx]['query_poe_reconstructions'] = {
                            'initial': {
                                'shape_logits': poe_shape_logits[i].detach().cpu().numpy(),
                                'grid_logits': poe_grid_logits[i].detach().cpu().numpy()
                            }
                        }

                # 2. Individual encoder evaluations + Influence Metrics
                individual_batch_accuracies = {}  # Track batch performance for comparison
                
                # Collect all encoder outputs for influence calculation
                all_enc_mus = []
                all_enc_logvars = []
                
                for enc_idx in range(num_encoders):
                    # Get individual encoder output
                    enc_mu, enc_log_var,_ = model.multi_encoder.encoders[enc_idx](batch_input_q, batch_target_q)
                    all_enc_mus.append(enc_mu)
                    all_enc_logvars.append(enc_log_var)
                    
                    enc_z = model.reparameterize(enc_mu, enc_log_var)
                    enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                        enc_z, batch_input_q, target_seq=batch_target_q
                    )
                    
                    # Calculate individual encoder metrics
                    enc_shape_pred = enc_shape_logits.argmax(dim=-1)
                    enc_grid_pred = enc_grid_logits.argmax(dim=-1)

                    batch_enc_shape_correct = (enc_shape_pred == shape_tgt).sum().item()
                    batch_enc_grid_correct = 0
                    batch_enc_grid_tokens = 0
                    batch_enc_exact_correct = 0

                    individual_encoder_metrics[f'encoder_{enc_idx}']['shape_correct'] += batch_enc_shape_correct
                    individual_encoder_metrics[f'encoder_{enc_idx}']['shape_tokens'] += batch_poe_shape_tokens

                    for i in range(query_batch_size):
                        tgt_rows = int(batch_target_q[i, 900].item())
                        tgt_cols = int(batch_target_q[i, 901].item())
                        active_pixels = tgt_rows * tgt_cols
                        if active_pixels > 0:
                            sample_enc_grid_correct = (enc_grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                            batch_enc_grid_correct += sample_enc_grid_correct
                            batch_enc_grid_tokens += active_pixels
                            if torch.all(enc_shape_pred[i] == shape_tgt[i]) and sample_enc_grid_correct == active_pixels:
                                batch_enc_exact_correct += 1
                        elif torch.all(enc_shape_pred[i] == shape_tgt[i]):
                             batch_enc_exact_correct += 1

                    individual_encoder_metrics[f'encoder_{enc_idx}']['grid_correct'] += batch_enc_grid_correct
                    individual_encoder_metrics[f'encoder_{enc_idx}']['grid_tokens'] += batch_enc_grid_tokens
                    individual_encoder_metrics[f'encoder_{enc_idx}']['sample_exact_correct'] += batch_enc_exact_correct
                    individual_encoder_metrics[f'encoder_{enc_idx}']['total_samples'] += query_batch_size
                    
                    # Store individual encoder query reconstructions for trajectory visualization using global index
                    for i in range(query_batch_size):
                        global_query_idx = processed_query_samples + i
                        if global_query_idx < len(trajectory_info):  # Ensure we have a corresponding trajectory entry
                            if 'query_encoder_reconstructions' not in trajectory_info[global_query_idx]:
                                trajectory_info[global_query_idx]['query_encoder_reconstructions'] = {}
                            
                            trajectory_info[global_query_idx]['query_encoder_reconstructions'][f'encoder_{enc_idx}'] = {
                                'shape_logits': enc_shape_logits[i].detach().cpu().numpy(),
                                'grid_logits': enc_grid_logits[i].detach().cpu().numpy()
                            }
                    
                    # Calculate batch-level accuracies for comparison
                    batch_enc_shape_acc = batch_enc_shape_correct / batch_poe_shape_tokens if batch_poe_shape_tokens > 0 else 0
                    batch_enc_grid_acc = batch_enc_grid_correct / batch_enc_grid_tokens if batch_enc_grid_tokens > 0 else 0
                    batch_enc_exact_acc = batch_enc_exact_correct / query_batch_size
                    
                    individual_batch_accuracies[f'encoder_{enc_idx}'] = {
                        'shape_acc': batch_enc_shape_acc,
                        'grid_acc': batch_enc_grid_acc,
                        'exact_acc': batch_enc_exact_acc
                    }
                    
                    # Store individual encoder query reconstructions
                    for i in range(query_batch_size):
                                                    individual_query_reconstructions[f'encoder_{enc_idx}'].append({
                                'input': batch_input_q[i].cpu().numpy().tolist(),
                                'target': batch_target_q[i].cpu().numpy().tolist(),
                                'reconstruction': (enc_shape_logits[i].cpu().numpy().tolist(),
                                                 enc_grid_logits[i].cpu().numpy().tolist())
                            })
                
                # 3. Calculate Encoder Covariance Traces
                if len(all_enc_mus) > 1:  # Only calculate for true multi-encoder models
                    from models.base_model import compute_encoder_covariance_traces
                    
                    # Stack encoder outputs: shape (K, B, D)
                    mu_stack = torch.stack(all_enc_mus, dim=0)  # (num_encoders, batch_size, latent_dim)
                    logvar_stack = torch.stack(all_enc_logvars, dim=0)  # (num_encoders, batch_size, latent_dim)
                    
                    # Compute covariance traces (sum of variances): shape (K, B)
                    covariance_traces = compute_encoder_covariance_traces(mu_stack, logvar_stack)
                    
                    # Store covariance traces for each sample in this batch
                    for i in range(query_batch_size):
                        sample_traces = {}
                        for enc_idx in range(num_encoders):
                            sample_traces[f'encoder_{enc_idx}'] = covariance_traces[enc_idx, i].item()
                        
                        # Add to global covariance traces storage
                        encoder_covariance_traces.append(sample_traces)
                
                # Log comparative performance every 10 batches for detailed analysis
                if (batch_idx + 1) % 10 == 0:
                    print(f"\n    Batch {batch_idx + 1} Comparative Analysis:")
                    print(f"      PoE: Shape={batch_poe_shape_acc:.3f}, Grid={batch_poe_grid_acc:.3f}, Exact={batch_poe_exact_acc:.3f}")
                    
                    # Find best performing encoder for this batch
                    best_encoder = max(individual_batch_accuracies.keys(), 
                                     key=lambda x: individual_batch_accuracies[x]['exact_acc'])
                    best_exact_acc = individual_batch_accuracies[best_encoder]['exact_acc']
                    
                    print(f"      Best Individual: {best_encoder} (Exact={best_exact_acc:.3f})")
                    
                    # Show performance difference
                    poe_vs_best = batch_poe_exact_acc - best_exact_acc
                    if poe_vs_best > 0.01:
                        print(f"      PoE Advantage: +{poe_vs_best:.3f} (PoE outperforming)")
                    elif poe_vs_best < -0.01:
                        print(f"      Individual Advantage: {abs(poe_vs_best):.3f} ({best_encoder} outperforming)")
                    else:
                        print(f"      Performance: Similar (diff={poe_vs_best:.3f})")
                
                # Compute query loss using PoE results
                q_loss_val = compute_loss(model, batch_input_q, batch_target_q)
                query_losses.append(q_loss_val.item())

                # Calculate batch accuracies for display (using PoE results)
                batch_poe_shape_acc = batch_poe_shape_correct / batch_poe_shape_tokens if batch_poe_shape_tokens > 0 else 0
                batch_poe_grid_acc = batch_poe_grid_correct / batch_poe_grid_tokens if batch_poe_grid_tokens > 0 else 0
                batch_poe_exact_acc = batch_poe_exact_correct / query_batch_size

                # Enhanced progress bar postfix with encoder comparison
                best_individual_exact = max(acc['exact_acc'] for acc in individual_batch_accuracies.values())
                
                # Update progress bar postfix
                query_pbar.set_postfix({
                    'samples': f'{processed_query_samples + query_batch_size}/{total_query_samples}',
                    'loss': f'{q_loss_val.item():.4f}',
                    'poe_exact': f'{batch_poe_exact_acc:.3f}',
                    'best_enc': f'{best_individual_exact:.3f}',
                    'poe_adv': f'{batch_poe_exact_acc - best_individual_exact:+.3f}'
                })
                
            else:
                # Single encoder evaluation (existing logic)
                shape_logits, grid_logits = model.decoder(z_query_expanded, batch_input_q, target_seq=batch_target_q)
                
                # Compute query loss
                q_loss_val = compute_loss(model, batch_input_q, batch_target_q)
                query_losses.append(q_loss_val.item())
                
                # Store each query's reconstruction with its input and target
                for i in range(query_batch_size):
                    query_reconstructions.append({
                        'input': batch_input_q[i].cpu().numpy().tolist(),
                        'target': batch_target_q[i].cpu().numpy().tolist(),
                        'reconstruction': (shape_logits[i].cpu().numpy().tolist(), grid_logits[i].cpu().numpy().tolist())
                    })
                    
                    # Store query data for trajectory visualization (single encoder) using global index
                    global_query_idx = processed_query_samples + i
                    if global_query_idx < len(trajectory_info):  # Ensure we have a corresponding trajectory entry
                        trajectory_info[global_query_idx]['query_input'] = batch_input_q[i].detach().cpu().numpy()
                        trajectory_info[global_query_idx]['query_target'] = batch_target_q[i].detach().cpu().numpy()
                        
                        # Store query encoder reconstruction (single encoder = encoder_0)
                        trajectory_info[global_query_idx]['query_encoder_reconstructions'] = {
                            'encoder_0': {
                                'shape_logits': shape_logits[i].detach().cpu().numpy(),
                                'grid_logits': grid_logits[i].detach().cpu().numpy()
                            }
                        }
                        
                        # Store query POE reconstruction (same as encoder for single encoder)
                        trajectory_info[global_query_idx]['query_poe_reconstructions'] = {
                            'initial': {
                                'shape_logits': shape_logits[i].detach().cpu().numpy(),
                                'grid_logits': grid_logits[i].detach().cpu().numpy()
                            }
                        }
                
                # Calculate metrics for single encoder
                shape_pred = shape_logits.argmax(dim=-1)
                grid_pred = grid_logits.argmax(dim=-1)
                shape_tgt = batch_target_q[:, 900:902].long()
                grid_tgt = batch_target_q[:, :900].long()

                # Calculate accuracies for this batch
                batch_shape_correct = (shape_pred == shape_tgt).sum().item()
                batch_shape_tokens = shape_tgt.numel()
                batch_grid_correct = 0
                batch_grid_tokens = 0
                batch_exact_correct = 0

                single_metrics['shape_correct'] += batch_shape_correct
                single_metrics['shape_tokens'] += batch_shape_tokens

                for i in range(query_batch_size):
                    tgt_rows = int(batch_target_q[i, 900].item())
                    tgt_cols = int(batch_target_q[i, 901].item())
                    active_pixels = tgt_rows * tgt_cols
                    if active_pixels > 0:
                        sample_grid_correct = (grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                        batch_grid_correct += sample_grid_correct
                        batch_grid_tokens += active_pixels
                        if torch.all(shape_pred[i] == shape_tgt[i]) and sample_grid_correct == active_pixels:
                            batch_exact_correct += 1
                    elif torch.all(shape_pred[i] == shape_tgt[i]):
                         batch_exact_correct += 1

                single_metrics['grid_correct'] += batch_grid_correct
                single_metrics['grid_tokens'] += batch_grid_tokens
                single_metrics['sample_exact_correct'] += batch_exact_correct
                single_metrics['total_samples'] += query_batch_size

                # Calculate batch accuracies for display
                batch_shape_acc = batch_shape_correct / batch_shape_tokens if batch_shape_tokens > 0 else 0
                batch_grid_acc = batch_grid_correct / batch_grid_tokens if batch_grid_tokens > 0 else 0
                batch_exact_acc = batch_exact_correct / query_batch_size

                # Update progress bar postfix
                query_pbar.set_postfix({
                    'samples': f'{processed_query_samples + query_batch_size}/{total_query_samples}',
                    'loss': f'{q_loss_val.item():.4f}',
                    'shape_acc': f'{batch_shape_acc:.3f}',
                    'exact_acc': f'{batch_exact_acc:.3f}'
                })
        
        processed_query_samples += query_batch_size
    
    query_pbar.close()

    print(f"\nQuery processing complete:")
    print(f"  Total samples processed: {processed_query_samples}")
    print(f"  Total batches processed: {len(queries_dataloader)}")

    # Calculate final metrics
    avg_support_loss = sum(support_losses) / len(support_losses) if support_losses else 0.0
    avg_query_loss = sum(query_losses) / len(query_losses) if query_losses else 0.0

    if is_multi_encoder:
        # Multi-encoder final metrics
        print(f"\n>>> MULTI-ENCODER EVALUATION RESULTS <<<")
        
        # PoE metrics
        poe_shape_acc = poe_metrics['shape_correct'] / poe_metrics['shape_tokens'] if poe_metrics['shape_tokens'] > 0 else 0.0
        poe_grid_acc = poe_metrics['grid_correct'] / poe_metrics['grid_tokens'] if poe_metrics['grid_tokens'] > 0 else 0.0
        poe_overall_acc = (poe_metrics['shape_correct'] + poe_metrics['grid_correct']) / (poe_metrics['shape_tokens'] + poe_metrics['grid_tokens']) if (poe_metrics['shape_tokens'] + poe_metrics['grid_tokens']) > 0 else 0.0
        poe_exact_acc = poe_metrics['sample_exact_correct'] / poe_metrics['total_samples'] if poe_metrics['total_samples'] > 0 else 0.0
        
        print(f"\n🎯 PRODUCT OF EXPERTS (PoE) RESULTS:")
        print(f"  Shape accuracy: {poe_shape_acc:.4f} ({poe_metrics['shape_correct']}/{poe_metrics['shape_tokens']})")
        print(f"  Grid accuracy: {poe_grid_acc:.4f} ({poe_metrics['grid_correct']}/{poe_metrics['grid_tokens']})")
        print(f"  Overall accuracy: {poe_overall_acc:.4f}")
        print(f"  Sample exact accuracy: {poe_exact_acc:.4f} ({poe_metrics['sample_exact_correct']}/{poe_metrics['total_samples']})")
        
        # Individual encoder metrics
        individual_accuracies = {}
        encoder_exact_accs = []
        
        print(f"\n👥 INDIVIDUAL ENCODER RESULTS:")
        for enc_name, metrics in individual_encoder_metrics.items():
            enc_shape_acc = metrics['shape_correct'] / metrics['shape_tokens'] if metrics['shape_tokens'] > 0 else 0.0
            enc_grid_acc = metrics['grid_correct'] / metrics['grid_tokens'] if metrics['grid_tokens'] > 0 else 0.0
            enc_overall_acc = (metrics['shape_correct'] + metrics['grid_correct']) / (metrics['shape_tokens'] + metrics['grid_tokens']) if (metrics['shape_tokens'] + metrics['grid_tokens']) > 0 else 0.0
            enc_exact_acc = metrics['sample_exact_correct'] / metrics['total_samples'] if metrics['total_samples'] > 0 else 0.0
            
            individual_accuracies[enc_name] = {
                'shape_accuracy': enc_shape_acc,
                'grid_accuracy': enc_grid_acc,
                'overall_accuracy': enc_overall_acc,
                'sample_exact_accuracy': enc_exact_acc
            }
            
            encoder_exact_accs.append(enc_exact_acc)
            
            print(f"  {enc_name.replace('_', ' ').title()}:")
            print(f"    Shape accuracy: {enc_shape_acc:.4f} ({metrics['shape_correct']}/{metrics['shape_tokens']})")
            print(f"    Grid accuracy: {enc_grid_acc:.4f} ({metrics['grid_correct']}/{metrics['grid_tokens']})")
            print(f"    Sample exact accuracy: {enc_exact_acc:.4f} ({metrics['sample_exact_correct']}/{metrics['total_samples']})")

        # Comparative Analysis
        print(f"\n📊 COMPARATIVE ANALYSIS:")
        
        # Best individual encoder
        best_encoder_idx = encoder_exact_accs.index(max(encoder_exact_accs))
        best_encoder_name = f"encoder_{best_encoder_idx}"
        best_individual_acc = max(encoder_exact_accs)
        worst_individual_acc = min(encoder_exact_accs)
        avg_individual_acc = sum(encoder_exact_accs) / len(encoder_exact_accs)
        
        print(f"  Best individual encoder: {best_encoder_name} ({best_individual_acc:.4f})")
        print(f"  Worst individual encoder: encoder_{encoder_exact_accs.index(worst_individual_acc)} ({worst_individual_acc:.4f})")
        print(f"  Average individual performance: {avg_individual_acc:.4f}")
        print(f"  Individual encoder variance: {sum((acc - avg_individual_acc)**2 for acc in encoder_exact_accs) / len(encoder_exact_accs):.6f}")
        
        # PoE vs Individual comparison
        poe_advantage = poe_exact_acc - best_individual_acc
        poe_vs_avg = poe_exact_acc - avg_individual_acc
        
        print(f"\n⚖️  PoE PERFORMANCE COMPARISON:")
        print(f"  PoE vs Best Individual: {poe_advantage:+.4f} ({poe_exact_acc:.4f} vs {best_individual_acc:.4f})")
        print(f"  PoE vs Average Individual: {poe_vs_avg:+.4f} ({poe_exact_acc:.4f} vs {avg_individual_acc:.4f})")
        
        # Encoder specialization analysis
        print(f"\n🔍 ENCODER SPECIALIZATION ANALYSIS:")
        if len(encoder_exact_accs) > 1:
            performance_range = max(encoder_exact_accs) - min(encoder_exact_accs)
            print(f"  Specialization range: {performance_range:.4f}")
            if performance_range > 0.05:
                print(f"  Classification: HIGH specialization")
            elif performance_range > 0.02:
                print(f"  Classification: MODERATE specialization")
            else:
                print(f"  Classification: LOW specialization")
        
        # Data distribution impact
        print(f"\n📈 PERFORMANCE METRICS SUMMARY:")
        print(f"  Total query samples evaluated: {poe_metrics['total_samples']}")
        print(f"  PoE exact match samples: {poe_metrics['sample_exact_correct']}")
        print(f"  Best individual exact matches: {individual_encoder_metrics[best_encoder_name]['sample_exact_correct']}")
        print(f"  Additional samples PoE got right: {poe_metrics['sample_exact_correct'] - individual_encoder_metrics[best_encoder_name]['sample_exact_correct']}")

        # Collect latent representations for visualization
        print(f"\n>>> COLLECTING LATENT REPRESENTATIONS FOR VISUALIZATION <<<")
        
        # Create evaluation latent data in the format expected by visualizers
        evaluation_latent_data = {}
        
        # Collect support sample latents
        support_latent_zs = []
        support_log_vars = []
        if support_latent_vectors:
            for support_latent in support_latent_vectors:
                latent_np = support_latent.cpu().numpy() if hasattr(support_latent, 'cpu') else support_latent
                # Ensure 2D shape [1, latent_dim]
                if latent_np.ndim == 1:
                    latent_np = latent_np.reshape(1, -1)
                support_latent_zs.append(latent_np[0])  # Store as 1D array
                # Create dummy log_var (not used in visualization but expected)
                support_log_vars.append(np.zeros_like(latent_np[0]))
            
            print(f"[OK] Collected {len(support_latent_zs)} support samples with key '{evaluated_key}'")
        
        # Collect query sample latents
        query_latent_zs = []
        query_log_vars = []
        for batch in queries_dataloader:
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            # Ensure tensors are on the correct device
            batch_input_q = batch_input_q.to(device)
            batch_target_q = batch_target_q.to(device)
            
            # Get query latents for this batch
            with torch.no_grad():
                if is_multi_encoder:
                    if encoder_idx is not None:
                        mu_q, log_var_q, _ = model.multi_encoder.encoders[encoder_idx](batch_input_q, batch_target_q)
                    else:
                        result = model(batch_input_q, batch_target_q)
                        (shape_logits, grid_logits), mu_q, log_var_q, _ = result
                else:
                    mu_q, log_var_q, _ = model.encoder(batch_input_q, batch_target_q)
                
                query_z = model.reparameterize(mu_q, log_var_q)
                
                for i in range(batch_input_q.size(0)):
                    query_latent_zs.append(query_z[i].cpu().numpy())  # Store as 1D array
                    query_log_vars.append(log_var_q[i].cpu().numpy())  # Store log_var too
        
        print(f"[OK] Collected {len(query_latent_zs)} query samples with key '{evaluated_key}'")
        
        # Structure data in the format expected by visualizers
        if is_multi_encoder:
            # Multi-encoder: provide both PoE and individual encoder format
            evaluation_latent_data['support'] = {
                'poe': {
                    'latent_zs': support_latent_zs,
                    'latent_log_vars': support_log_vars
                }
            }
            evaluation_latent_data['query'] = {
                'poe': {
                    'latent_zs': query_latent_zs,
                    'latent_log_vars': query_log_vars
                }
            }
            # Add individual encoder entries (same data since we're using PoE)
            for enc_idx in range(num_encoders):
                evaluation_latent_data['support'][f'encoder_{enc_idx}'] = {
                    'latent_zs': support_latent_zs,
                    'latent_log_vars': support_log_vars
                }
                evaluation_latent_data['query'][f'encoder_{enc_idx}'] = {
                    'latent_zs': query_latent_zs,
                    'latent_log_vars': query_log_vars
                }
        else:
            # Single encoder: provide encoder_0 format
            evaluation_latent_data['support'] = {
                'encoder_0': {
                    'latent_zs': support_latent_zs,
                    'latent_log_vars': support_log_vars
                }
            }
            evaluation_latent_data['query'] = {
                'encoder_0': {
                    'latent_zs': query_latent_zs,
                    'latent_log_vars': query_log_vars
                }
            }
        
        print(f"[OK] Structured evaluation latent data with {len(evaluation_latent_data)} data types")

        # After support sample trajectory_info is built, store the first query sample's input/output in the first trajectory_info entry
        # Only do this if there is at least one query sample and at least one trajectory_info entry
        # This is done after the query evaluation loop, before returning results
        # Find the first query sample (from the first batch in queries_dataloader)
        first_query_input = None
        first_query_output = None
        for batch in queries_dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            if batch_input_q.size(0) > 0:
                first_query_input = batch_input_q[0].detach().cpu().numpy()
                first_query_output = batch_target_q[0].detach().cpu().numpy()
                break
        if trajectory_info and first_query_input is not None and first_query_output is not None:
            trajectory_info[0]['query_input'] = first_query_input
            trajectory_info[0]['query_output'] = first_query_output

        # Store 'visualize_n_values' from evaluation settings in each trajectory_info entry (if present)
        eval_settings = settings.get_evaluation_settings()
        visualize_n_values = eval_settings.get('visualize_n_values', None)
        if visualize_n_values is not None and trajectory_info:
            for entry in trajectory_info:
                entry['visualize_n_values'] = visualize_n_values

        print(f"DEBUG: Finished evaluation function")
        print(f"DEBUG: trajectory_info length: {len(trajectory_info)}")
        print(f"DEBUG: trajectory_info type: {type(trajectory_info)}")
        if trajectory_info:
            print(f"DEBUG: First trajectory_info entry keys: {list(trajectory_info[0].keys())}")
        
        # Store evaluation latent data for visualization
        evaluation_latent_data = {
            'task_latents': task_latent_data,
            'support_samples': task_samples,
            'query_samples': query_reconstructions,
            'evaluation_type': 'task_optimization',
            'task_keys': list(task_latents.keys()),
            'evaluated_keys': list(task_latents.keys())  # ← Add this
        }

        # Also store at the top level for visualization
        results['evaluation_latent_data'] = evaluation_latent_data
        results['task_latent_data'] = task_latent_data
        
        return {
            'metrics': {
                'support_loss': avg_support_loss,
                'query_loss': avg_query_loss,
                'losses_gradient_ascent': z_optimization_logs,
                'used_latent_optimization': optimize_z_inference,
                'trajectory_info': trajectory_info,
                'is_multi_encoder': True,
                'num_encoders': num_encoders,
                # PoE metrics (primary results)
                'shape_accuracy': poe_shape_acc,
                'grid_accuracy': poe_grid_acc,
                'overall_accuracy': poe_overall_acc,
                'sample_exact_accuracy': poe_exact_acc,
                # Individual encoder metrics (for analysis)
                'individual_encoder_accuracies': individual_accuracies,
                # PoE metrics (detailed)
                'poe_metrics': {
                    'shape_accuracy': poe_shape_acc,
                    'grid_accuracy': poe_grid_acc,
                    'overall_accuracy': poe_overall_acc,
                    'sample_exact_accuracy': poe_exact_acc
                },
                # Comparative analysis results
                'comparative_analysis': {
                    'best_individual_accuracy': best_individual_acc,
                    'worst_individual_accuracy': worst_individual_acc,
                    'average_individual_accuracy': avg_individual_acc,
                    'poe_vs_best_advantage': poe_advantage,
                    'poe_vs_avg_advantage': poe_vs_avg,
                    'encoder_performance_variance': sum((acc - avg_individual_acc)**2 for acc in encoder_exact_accs) / len(encoder_exact_accs),
                    'specialization_range': max(encoder_exact_accs) - min(encoder_exact_accs) if len(encoder_exact_accs) > 1 else 0.0
                },
                # Encoder covariance traces for PoE analysis
                'encoder_covariance_traces': encoder_covariance_traces
            },
            'reconstruction_results': {
                'support_reconstructions': poe_support_reconstructions,
                'query_reconstructions': poe_query_reconstructions,
                'individual_support_reconstructions': individual_support_reconstructions,
                'individual_query_reconstructions': individual_query_reconstructions
            },
            'evaluation_latent_data': evaluation_latent_data
        }
    else:
        # Single encoder final metrics (existing logic)
        print(f"DEBUG: Processing single encoder results")
        final_shape_acc = single_metrics['shape_correct'] / single_metrics['shape_tokens'] if single_metrics['shape_tokens'] > 0 else 0.0
        final_grid_acc = single_metrics['grid_correct'] / single_metrics['grid_tokens'] if single_metrics['grid_tokens'] > 0 else 0.0
        final_overall_acc = (single_metrics['shape_correct'] + single_metrics['grid_correct']) / (single_metrics['shape_tokens'] + single_metrics['grid_tokens']) if (single_metrics['shape_tokens'] + single_metrics['grid_tokens']) > 0 else 0.0
        final_exact_acc = single_metrics['sample_exact_correct'] / single_metrics['total_samples'] if single_metrics['total_samples'] > 0 else 0.0

        print(f"\n>>> SINGLE ENCODER EVALUATION RESULTS <<<")
        print(f"Support samples: {len(support_reconstructions)}")
        print(f"Query samples: {len(query_reconstructions)}")
        print(f"Support loss: {avg_support_loss:.4f}")
        print(f"Query loss: {avg_query_loss:.4f}")
        print(f"Shape accuracy: {final_shape_acc:.4f} ({single_metrics['shape_correct']}/{single_metrics['shape_tokens']})")
        print(f"Grid accuracy: {final_grid_acc:.4f} ({single_metrics['grid_correct']}/{single_metrics['grid_tokens']})")
        print(f"Overall accuracy: {final_overall_acc:.4f}")
        print(f"Sample exact accuracy: {final_exact_acc:.4f} ({single_metrics['sample_exact_correct']}/{single_metrics['total_samples']})")

        # Collect latent representations for visualization
        print(f"\n>>> COLLECTING LATENT REPRESENTATIONS FOR VISUALIZATION <<<")
        
        # Create evaluation latent data in the format expected by visualizers (single encoder)
        evaluation_latent_data = {}
        
        # Collect support sample latents
        support_latent_zs = []
        support_log_vars = []
        if support_latent_vectors:
            for support_latent in support_latent_vectors:
                latent_np = support_latent.cpu().numpy() if hasattr(support_latent, 'cpu') else support_latent
                # Ensure 2D shape [1, latent_dim]
                if latent_np.ndim == 1:
                    latent_np = latent_np.reshape(1, -1)
                support_latent_zs.append(latent_np[0])  # Store as 1D array
                # Create dummy log_var (not used in visualization but expected)
                support_log_vars.append(np.zeros_like(latent_np[0]))
            
            print(f"[OK] Collected {len(support_latent_zs)} support samples with key '{evaluated_key}'")
        
        # Collect query sample latents
        query_latent_zs = []
        query_log_vars = []
        for batch in queries_dataloader:
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            # Ensure tensors are on the correct device
            batch_input_q = batch_input_q.to(device)
            batch_target_q = batch_target_q.to(device)
            
            # Get query latents for this batch (single encoder)
            with torch.no_grad():
                mu_q, log_var_q, _ = model.encoder(batch_input_q, batch_target_q)
                query_z = model.reparameterize(mu_q, log_var_q)
                
                for i in range(batch_input_q.size(0)):
                    query_latent_zs.append(query_z[i].cpu().numpy())  # Store as 1D array
                    query_log_vars.append(log_var_q[i].cpu().numpy())  # Store log_var too
        
        print(f"[OK] Collected {len(query_latent_zs)} query samples with key '{evaluated_key}'")
        
        # Structure data in the format expected by visualizers (single encoder format)
        evaluation_latent_data['support'] = {
            'encoder_0': {
                'latent_zs': support_latent_zs,
                'latent_log_vars': support_log_vars
            }
        }
        evaluation_latent_data['query'] = {
            'encoder_0': {
                'latent_zs': query_latent_zs,
                'latent_log_vars': query_log_vars
            }
        }
        
        print(f"[OK] Structured evaluation latent data with {len(evaluation_latent_data)} data types")

        # Convert single-encoder results to unified multi-encoder format for consistent processing
        individual_accuracies = {
            'encoder_0': {
                'shape_accuracy': final_shape_acc,
                'grid_accuracy': final_grid_acc,
                'overall_accuracy': final_overall_acc,
                'sample_exact_accuracy': final_exact_acc
            }
        }
        
        print(f"DEBUG: Finished single encoder evaluation function")
        print(f"DEBUG: trajectory_info length: {len(trajectory_info)}")
        print(f"DEBUG: trajectory_info type: {type(trajectory_info)}")
        if trajectory_info:
            print(f"DEBUG: First trajectory_info entry keys: {list(trajectory_info[0].keys())}")
        
        return {
            'metrics': {
                'support_loss': avg_support_loss,
                'query_loss': avg_query_loss,
                'losses_gradient_ascent': z_optimization_logs,
                'used_latent_optimization': optimize_z_inference,
                'trajectory_info': trajectory_info,
                'is_multi_encoder': True,  # Always True for unified processing
                'num_encoders': 1,  # Single encoder treated as multi-encoder with 1 encoder
                'is_actually_single_encoder': True,  # Track original state
                # Single encoder metrics (primary results - same as PoE for 1 encoder)
                'shape_accuracy': final_shape_acc,
                'grid_accuracy': final_grid_acc,
                'overall_accuracy': final_overall_acc,
                'sample_exact_accuracy': final_exact_acc,
                # Individual encoder metrics (for unified processing)
                'individual_encoder_accuracies': individual_accuracies,
                # PoE metrics (same as single encoder for 1 encoder)
                'poe_metrics': {
                    'shape_accuracy': final_shape_acc,
                    'grid_accuracy': final_grid_acc,
                    'overall_accuracy': final_overall_acc,
                    'sample_exact_accuracy': final_exact_acc
                },
                # Comparative analysis results (trivial for single encoder)
                'comparative_analysis': {
                    'best_individual_accuracy': final_exact_acc,
                    'worst_individual_accuracy': final_exact_acc,
                    'average_individual_accuracy': final_exact_acc,
                    'poe_vs_best_advantage': 0.0,
                    'poe_vs_avg_advantage': 0.0,
                    'encoder_performance_variance': 0.0,
                    'specialization_range': 0.0
                }
            },
            'reconstruction_results': {
                'support_reconstructions': support_reconstructions,
                'query_reconstructions': query_reconstructions,
                # Add individual reconstructions for unified processing (same as main for single encoder)
                'individual_support_reconstructions': {'encoder_0': support_reconstructions},
                'individual_query_reconstructions': {'encoder_0': query_reconstructions}
            },
            'evaluation_latent_data': evaluation_latent_data
        }

def evaluate_model_with_task_optimization(model, samples_dataloader, queries_dataloader, device='cuda',
                                        encoder_idx=None, use_independent_decoder=False, support_key_mapping=None, query_key_mapping=None):
    """
    Evaluate model using task-level latent optimization instead of per-sample optimization.
    This optimizes ONE latent per task to explain ALL support samples of that task.
    
    Args:
        model: The trained model to evaluate
        samples_dataloader: DataLoader containing support samples grouped by task
        queries_dataloader: DataLoader containing query samples
        device: Device to run evaluation on
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
        
    Returns:
        dict: Dictionary containing evaluation metrics and task-level latent data
    """
    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']
    
    if not optimize_z_inference:
        print("WARNING: Task-level optimization requires latent optimization to be enabled!")
        print("Falling back to regular evaluation...")
        return evaluate_model(model, samples_dataloader, queries_dataloader, device, 
                            encoder_idx, use_independent_decoder)
    
    # Check if this is a multi-encoder model
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
    
    print(f"=== TASK-LEVEL OPTIMIZATION EVALUATION ===")
    print(f"Model type: {'Multi-encoder' if is_multi_encoder else 'Single encoder'}")
    if is_multi_encoder:
        print(f"Number of encoders: {num_encoders}")
    print(f"Optimization steps: {latent_optimization['inference']['num_steps']}")
    print(f"Optimization learning rate: {latent_optimization['inference']['learning_rate']}")
    
    model.eval()
    
    # Step 1: Collect support samples grouped by task
    task_samples = collect_support_samples_by_task(samples_dataloader, support_key_mapping, device)
    
    if not task_samples:
        print("ERROR: No task samples found!")
        return {}
    
    # Step 2: Optimize one latent per task
    print(f"\n=== TASK OPTIMIZATION ===")
    task_latents = {}
    task_trajectories = {}
    
    for task_key, support_samples in task_samples.items():
        print(f"\nOptimizing task '{task_key}' with {len(support_samples)} support samples...")
        
        task_latent, final_loss, trajectory = optimize_task_latent(
            model, support_samples, task_key,
            num_steps=latent_optimization['inference']['num_steps'],
            lr=latent_optimization['inference']['learning_rate'],
            encoder_idx=encoder_idx,
            use_independent_decoder=use_independent_decoder
        )
        
        task_latents[task_key] = task_latent
        task_trajectories[task_key] = trajectory
        
        print(f"Task '{task_key}' final loss: {final_loss:.4f}")
    
    # Step 3: Evaluate on query samples using task latents
    print(f"\n=== QUERY EVALUATION ===")
    total_shape_correct = 0
    total_shape_tokens = 0
    total_grid_correct = 0
    total_grid_tokens = 0
    total_exact_correct = 0
    total_samples = 0
    
    query_reconstructions = []
    
    with torch.no_grad():
        for batch in queries_dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            batch_input_q = batch_input_q.to(device)
            batch_target_q = batch_target_q.to(device)
            
            # For each query sample, use the corresponding task latent
            for i in range(batch_input_q.size(0)):
                query_input = batch_input_q[i:i+1]
                query_target = batch_target_q[i:i+1]
                
                # Determine which task latent to use
                if batch_keys is not None and query_key_mapping is not None:
                    key_idx = batch_keys[i].item()
                    task_key = query_key_mapping[key_idx]
                    if task_key in task_latents:
                        task_latent = task_latents[task_key]
                    else:
                        # Use first available task latent if key not found
                        task_key = list(task_latents.keys())[0]
                        task_latent = task_latents[task_key]
                        print(f"    Warning: No task latent found for query key '{query_key_mapping[key_idx]}', using '{task_key}'")
                else:
                    # Use first available task latent if no keys
                    task_key = list(task_latents.keys())[0]
                    task_latent = task_latents[task_key]
                    print(f"    Warning: No query keys provided, using '{task_key}'")
                
                # Decode using task latent
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    if use_independent_decoder and encoder_idx is not None:
                        shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](
                            task_latent, query_input, target_seq=query_target)
                    else:
                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                            task_latent, query_input, target_seq=query_target)
                else:
                    shape_logits, grid_logits = model.decoder(task_latent, query_input, target_seq=query_target)
                
                # Compute accuracy
                shape_pred = torch.argmax(shape_logits, dim=-1)
                shape_tgt = query_target[:, 900:902].long()
                shape_correct = (shape_pred == shape_tgt).sum().item()
                shape_tokens = shape_tgt.numel()
                
                # Grid accuracy
                tgt_rows = int(query_target[0, 900].item())
                tgt_cols = int(query_target[0, 901].item())
                active_pixels = tgt_rows * tgt_cols
                
                if active_pixels > 0:
                    grid_pred = torch.argmax(grid_logits[:, :active_pixels], dim=-1)
                    grid_tgt = query_target[:, :active_pixels].long()
                    grid_correct = (grid_pred == grid_tgt).sum().item()
                    grid_tokens = active_pixels
                else:
                    grid_correct = 0
                    grid_tokens = 0
                
                # Sample exact match
                exact_correct = 1 if (shape_correct == shape_tokens and grid_correct == grid_tokens) else 0
                
                # Accumulate metrics
                total_shape_correct += shape_correct
                total_shape_tokens += shape_tokens
                total_grid_correct += grid_correct
                total_grid_tokens += grid_tokens
                total_exact_correct += exact_correct
                total_samples += 1
                
                # Store reconstruction for visualization
                query_reconstructions.append({
                    'input': query_input[0].cpu().numpy(),
                    'target': query_target[0].cpu().numpy(),
                    'reconstruction': {
                        'shape_logits': shape_logits[0].cpu().numpy(),
                        'grid_logits': grid_logits[0].cpu().numpy()
                    },
                    'task_key': task_key,
                    'exact_match': exact_correct == 1
                })
    
    # Calculate final metrics
    shape_accuracy = (total_shape_correct / total_shape_tokens) if total_shape_tokens > 0 else 0.0
    grid_accuracy = (total_grid_correct / total_grid_tokens) if total_grid_tokens > 0 else 0.0
    overall_accuracy = ((total_shape_correct + total_grid_correct) / 
                       (total_shape_tokens + total_grid_tokens)) if (total_shape_tokens + total_grid_tokens) > 0 else 0.0
    exact_accuracy = (total_exact_correct / total_samples) if total_samples > 0 else 0.0
    
    print(f"\n=== TASK-LEVEL EVALUATION RESULTS ===")
    print(f"Tasks evaluated: {len(task_latents)}")
    print(f"Query samples: {total_samples}")
    print(f"Shape accuracy: {shape_accuracy:.4f}")
    print(f"Grid accuracy: {grid_accuracy:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Exact match accuracy: {exact_accuracy:.4f}")
    
    # Prepare latent data for visualization (task-level latents only)
    task_latent_data = {
        'task_latents': {
            task_key: {
                'latent_z': task_latent.cpu().numpy(),
                'final_loss': task_trajectories[task_key]['losses'][-1],
                'num_support_samples': task_trajectories[task_key]['num_support_samples'],
                'trajectory': task_trajectories[task_key]
            }
            for task_key, task_latent in task_latents.items()
        }
    }
    
    # Calculate support loss (average of all task optimization final losses)  
    support_loss = sum(task_trajectories[task_key]['losses'][-1] for task_key in task_trajectories) / len(task_trajectories) if task_trajectories else 0.0
    
    # Calculate query loss (average reconstruction loss on queries)
    query_loss = 0.0
    query_loss_count = 0
    for recon in query_reconstructions:
        if recon.get('exact_match', False):
            query_loss += 0.0  # Perfect reconstruction
        else:
            query_loss += 1.0  # Imperfect reconstruction (simplified)
        query_loss_count += 1
    query_loss = query_loss / query_loss_count if query_loss_count > 0 else 0.0
    
    trajectory_info = []

    if not task_samples:
        print("WARNING: No support samples found for trajectory visualization")
        
    else:
        # Create trajectory_info for visualization compatibility
        for task_key, task_trajectory in task_trajectories.items():
            # Get the first support sample for visualization
            support_samples = task_samples.get(task_key, [])
            if support_samples:
                first_support_input, first_support_target = support_samples[0]
                
                # Convert tensors to proper sequence format for visualization
                input_seq = first_support_input.cpu().numpy().flatten()  # Flatten to 1D
                target_seq = first_support_target.cpu().numpy().flatten()  # Flatten to 1D
                
                # Ensure proper sequence format (if not already 902-length)
                if len(input_seq) != 902:
                    # Pad or truncate to 902 length for visualization compatibility
                    if len(input_seq) > 902:
                        input_seq = input_seq[:902]
                    else:
                        # Pad with zeros to 902 length
                        padding = np.zeros(902 - len(input_seq))
                        input_seq = np.concatenate([input_seq, padding])
                
                if len(target_seq) != 902:
                    if len(target_seq) > 902:
                        target_seq = target_seq[:902]
                    else:
                        padding = np.zeros(902 - len(target_seq))
                        target_seq = np.concatenate([target_seq, padding])
                
                trajectory_info.append({
                    'task_optimization': True,
                    'task_key': task_key,
                    'evaluated_key': task_key,  # ← Ensure this is set
                    'z_vectors': task_trajectory['z_vectors'],  # Required for trajectory plots
                    'losses': task_trajectory['losses'],        # Required for trajectory plots
                    'input_sample': input_seq,     # ← Proper sequence format
                    'target_sample': target_seq,   # ← Proper sequence format
                    'trajectory_type': 'task_optimization',
                    'num_support_samples': task_trajectory['num_support_samples'],
                    'final_loss': task_trajectory['losses'][-1] if task_trajectory['losses'] else 0.0,
                    'is_multi_encoder': is_multi_encoder
                })
    
                # Generate support reconstructions for trajectory visualization
                print(f"DEBUG: Generating support reconstructions for task '{task_key}' trajectory")
                
                # Store trajectory reconstructions at key points
                if len(task_trajectory['z_vectors']) > 1:
                    print(f"DEBUG: Starting trajectory reconstruction for task '{task_key}'")
                    trajectory_info[-1]['poe_trajectory_reconstructions'] = {}
                    key_indices = [0, len(task_trajectory['z_vectors'])//2, len(task_trajectory['z_vectors'])-1]
                    key_labels = ['initial', 'mid', 'final']
                    
                    with torch.no_grad():
                        for idx, label in zip(key_indices, key_labels):
                            if idx < len(task_trajectory['z_vectors']):
                                print(f"DEBUG: Processing trajectory step {idx} ({label}) for task '{task_key}'")
                                try:
                                    # Get the trajectory z vector
                                    z_vector = task_trajectory['z_vectors'][idx]
                                    print(f"DEBUG: z_vector type: {type(z_vector)}, shape: {z_vector.shape if hasattr(z_vector, 'shape') else 'no shape'}")
                                    
                                    # Ensure proper tensor format
                                    if isinstance(z_vector, torch.Tensor):
                                        traj_z = z_vector.to(device)
                                    else:
                                        # Convert to tensor if it's not already
                                        z_vector = np.array(z_vector, dtype=np.float32)
                                        if z_vector.ndim == 1:
                                            z_vector = z_vector.reshape(1, -1)
                                        traj_z = torch.tensor(z_vector, dtype=torch.float32).to(device)
                                    
                                    # Ensure correct shape for decoder input
                                    if traj_z.dim() == 1:
                                        traj_z = traj_z.unsqueeze(0)  # Add batch dimension
                                    elif traj_z.dim() > 2:
                                        traj_z = traj_z.squeeze()  # Remove extra dimensions
                                        if traj_z.dim() == 1:
                                            traj_z = traj_z.unsqueeze(0)
                                    
                                    print(f"DEBUG: Final tensor shape for decoder: {traj_z.shape}")
                                    
                                    # Convert input/target to proper tensor format for decoder
                                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                                    target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
                                    
                                    print(f"DEBUG: input_tensor shape: {input_tensor.shape}")
                                    print(f"DEBUG: target_tensor shape: {target_tensor.shape}")
                                    
                                    # Generate reconstruction
                                    if is_multi_encoder:
                                        # Multi-encoder: use shared decoder (default for task optimization)
                                        traj_shape_logits, traj_grid_logits = model.multi_encoder.shared_decoder(
                                            traj_z, input_tensor, target_seq=target_tensor
                                        )
                                    else:
                                        # Single encoder: use regular decoder
                                        traj_shape_logits, traj_grid_logits = model.decoder(
                                            traj_z, input_tensor, target_seq=target_tensor
                                        )
                                    
                                    print(f"DEBUG: Reconstruction successful - shape_logits shape: {traj_shape_logits.shape}, grid_logits shape: {traj_grid_logits.shape}")
                                    
                                    trajectory_info[-1]['poe_trajectory_reconstructions'][label] = {
                                        'step': idx,
                                        'shape_logits': traj_shape_logits.cpu().numpy(),
                                        'grid_logits': traj_grid_logits.cpu().numpy()
                                    }
                                    print(f"DEBUG: Successfully stored {label} reconstruction for task '{task_key}'")
                                except Exception as e:
                                    reconstruction_type = "PoE" if is_multi_encoder else "trajectory"
                                    print(f"    Warning: Could not generate {reconstruction_type} reconstruction at step {idx} for task '{task_key}': {e}")
                                    print(f"    DEBUG: Exception details - {type(e).__name__}: {str(e)}")
                                    import traceback
                                    print(f"    DEBUG: Full traceback:")
                                    traceback.print_exc()
                                    trajectory_info[-1]['poe_trajectory_reconstructions'][label] = None
                
                # Generate individual encoder reconstructions for comparison
                print(f"DEBUG: Generating individual encoder reconstructions for task '{task_key}'")
                trajectory_info[-1]['individual_encoder_reconstructions'] = {}
                
                # Get trajectory decoder type setting
                eval_settings = settings.get_evaluation_settings()
                decoder_type = eval_settings.get('trajectory_decoder_type', 'shared')
                use_independent_decoder = (decoder_type == 'independent')
                
                # Add decoder type metadata to trajectory
                trajectory_info[-1]['trajectory_decoder_type'] = decoder_type
                trajectory_info[-1]['used_independent_decoder'] = use_independent_decoder
                
                with torch.no_grad():
                    # For single encoder, create encoder_0 entry
                    if not is_multi_encoder:
                        # Use the initial z vector from trajectory
                        initial_z = task_trajectory['z_vectors'][0]
                        if isinstance(initial_z, torch.Tensor):
                            enc_z_tensor = initial_z.to(device)
                        else:
                            enc_z_tensor = torch.tensor(initial_z, dtype=torch.float32).to(device)
                        
                        try:
                            print(f"DEBUG: Generating single encoder reconstruction for task '{task_key}'")
                            print(f"DEBUG: enc_z_tensor shape: {enc_z_tensor.shape}")
                            
                            # Convert input/target to proper tensor format
                            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                            target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            print(f"DEBUG: input_tensor shape: {input_tensor.shape}")
                            print(f"DEBUG: target_tensor shape: {target_tensor.shape}")
                            
                            # Generate reconstruction using the decoder
                            enc_shape_logits, enc_grid_logits = model.decoder(
                                enc_z_tensor, input_tensor, target_seq=target_tensor
                            )
                            
                            print(f"DEBUG: Single encoder reconstruction successful - shape_logits shape: {enc_shape_logits.shape}, grid_logits shape: {enc_grid_logits.shape}")
                            
                            trajectory_info[-1]['individual_encoder_reconstructions']['encoder_0'] = {
                                'shape_logits': enc_shape_logits.cpu().numpy(),
                                'grid_logits': enc_grid_logits.cpu().numpy()
                            }
                        except Exception as e:
                            print(f"    Warning: Could not generate reconstruction for encoder 0 for task '{task_key}': {e}")
                            print(f"    DEBUG: Exception details - {type(e).__name__}: {str(e)}")
                            import traceback
                            print(f"    DEBUG: Full traceback:")
                            traceback.print_exc()
                            trajectory_info[-1]['individual_encoder_reconstructions']['encoder_0'] = None
                    else:
                        # Multi-encoder: generate reconstructions for each encoder
                        for enc_idx in range(num_encoders):
                            # Use the initial z vector from trajectory
                            initial_z = task_trajectory['z_vectors'][0]
                            if isinstance(initial_z, torch.Tensor):
                                enc_z_tensor = initial_z.to(device)
                            else:
                                enc_z_tensor = torch.tensor(initial_z, dtype=torch.float32).to(device)
                            
                            try:
                                print(f"DEBUG: Generating individual encoder {enc_idx} reconstruction for task '{task_key}'")
                                print(f"DEBUG: enc_z_tensor shape: {enc_z_tensor.shape}")
                                
                                # Convert input/target to proper tensor format
                                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                                target_tensor = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
                                
                                print(f"DEBUG: input_tensor shape: {input_tensor.shape}")
                                print(f"DEBUG: target_tensor shape: {target_tensor.shape}")
                                
                                # Generate reconstruction using appropriate decoder based on setting
                                if use_independent_decoder:
                                    # Use individual encoder's independent decoder
                                    enc_shape_logits, enc_grid_logits = model.multi_encoder.independent_decoders[enc_idx](
                                        enc_z_tensor, input_tensor, target_seq=target_tensor
                                    )
                                else:
                                    # Use shared decoder (default behavior)
                                    enc_shape_logits, enc_grid_logits = model.multi_encoder.shared_decoder(
                                        enc_z_tensor, input_tensor, target_seq=target_tensor
                                    )
                                
                                print(f"DEBUG: Individual encoder {enc_idx} reconstruction successful - shape_logits shape: {enc_shape_logits.shape}, grid_logits shape: {enc_grid_logits.shape}")
                                
                                trajectory_info[-1]['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = {
                                    'shape_logits': enc_shape_logits.cpu().numpy(),
                                    'grid_logits': enc_grid_logits.cpu().numpy()
                                }
                            except Exception as e:
                                decoder_name = f"independent decoder {enc_idx}" if use_independent_decoder else "shared decoder"
                                print(f"    Warning: Could not generate reconstruction for encoder {enc_idx} with {decoder_name} for task '{task_key}': {e}")
                                print(f"    DEBUG: Exception details - {type(e).__name__}: {str(e)}")
                                import traceback
                                print(f"    DEBUG: Full traceback:")
                                traceback.print_exc()
                                trajectory_info[-1]['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = None
    
    # Store evaluation latent data for visualization
    evaluation_latent_data = {
        'task_latents': task_latent_data,
        'support_samples': task_samples,
        'query_samples': query_reconstructions,
        'evaluation_type': 'task_optimization',
        'task_keys': list(task_latents.keys()),
        'evaluated_keys': list(task_latents.keys())
    }
    
    # Add query reconstructions to trajectory_info for visualization
    print(f"DEBUG: Adding query reconstructions to trajectory_info")
    for task_key, task_trajectory in task_trajectories.items():
        # Find the corresponding trajectory_info entry
        for trajectory_entry in trajectory_info:
            if trajectory_entry.get('task_key') == task_key:
                print(f"DEBUG: Adding query reconstructions for task '{task_key}'")
                
                # Find query samples for this task
                task_query_reconstructions = [q for q in query_reconstructions if q.get('task_key') == task_key]
                
                if task_query_reconstructions:
                    # Use the first query sample for visualization
                    query_recon = task_query_reconstructions[0]
                    query_input = query_recon['input']
                    query_target = query_recon['target']
                    
                    # Store query input and target
                    trajectory_entry['query_input'] = query_input
                    trajectory_entry['query_target'] = query_target
                    
                    # Generate query reconstructions
                    print(f"DEBUG: Generating query reconstructions for task '{task_key}'")
                    trajectory_entry['query_encoder_reconstructions'] = {}
                    trajectory_entry['query_poe_reconstructions'] = {}
                    
                    with torch.no_grad():
                        # Convert query data to tensors
                        query_input_tensor = torch.tensor(query_input, dtype=torch.float32).unsqueeze(0).to(device)
                        query_target_tensor = torch.tensor(query_target, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # 1. Individual encoder reconstruction (initial encoder belief)
                        if not is_multi_encoder:
                            # Single encoder: use encoder_0
                            try:
                                print(f"DEBUG: Generating query encoder_0 reconstruction for task '{task_key}'")
                                
                                # Get initial z vector from trajectory (encoder belief)
                                initial_z = task_trajectory['z_vectors'][0]
                                if isinstance(initial_z, torch.Tensor):
                                    enc_z_tensor = initial_z.to(device)
                                else:
                                    enc_z_tensor = torch.tensor(initial_z, dtype=torch.float32).to(device)
                                
                                # Generate reconstruction using the decoder
                                enc_shape_logits, enc_grid_logits = model.decoder(
                                    enc_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                )
                                
                                print(f"DEBUG: Query encoder_0 reconstruction successful")
                                
                                trajectory_entry['query_encoder_reconstructions']['encoder_0'] = {
                                    'shape_logits': enc_shape_logits.cpu().numpy(),
                                    'grid_logits': enc_grid_logits.cpu().numpy()
                                }
                            except Exception as e:
                                print(f"    Warning: Could not generate query encoder_0 reconstruction for task '{task_key}': {e}")
                                trajectory_entry['query_encoder_reconstructions']['encoder_0'] = None
                        else:
                            # Multi-encoder: generate for each encoder
                            for enc_idx in range(num_encoders):
                                try:
                                    print(f"DEBUG: Generating query encoder {enc_idx} reconstruction for task '{task_key}'")
                                    
                                    # Get initial z vector from trajectory (encoder belief)
                                    initial_z = task_trajectory['z_vectors'][0]
                                    if isinstance(initial_z, torch.Tensor):
                                        enc_z_tensor = initial_z.to(device)
                                    else:
                                        enc_z_tensor = torch.tensor(initial_z, dtype=torch.float32).to(device)
                                    
                                    # Generate reconstruction using appropriate decoder
                                    if use_independent_decoder:
                                        enc_shape_logits, enc_grid_logits = model.multi_encoder.independent_decoders[enc_idx](
                                            enc_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                        )
                                    else:
                                        enc_shape_logits, enc_grid_logits = model.multi_encoder.shared_decoder(
                                            enc_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                        )
                                    
                                    print(f"DEBUG: Query encoder {enc_idx} reconstruction successful")
                                    
                                    trajectory_entry['query_encoder_reconstructions'][f'encoder_{enc_idx}'] = {
                                        'shape_logits': enc_shape_logits.cpu().numpy(),
                                        'grid_logits': enc_grid_logits.cpu().numpy()
                                    }
                                except Exception as e:
                                    decoder_name = f"independent decoder {enc_idx}" if use_independent_decoder else "shared decoder"
                                    print(f"    Warning: Could not generate query encoder {enc_idx} reconstruction with {decoder_name} for task '{task_key}': {e}")
                                    trajectory_entry['query_encoder_reconstructions'][f'encoder_{enc_idx}'] = None
                        
                        # 2. Initial POE reconstruction (joint encoder belief before optimization)
                        try:
                            print(f"DEBUG: Generating query initial POE reconstruction for task '{task_key}'")
                            
                            # Use initial z vector (joint encoder belief before optimization)
                            initial_z = task_trajectory['z_vectors'][0]
                            if isinstance(initial_z, torch.Tensor):
                                poe_z_tensor = initial_z.to(device)
                            else:
                                poe_z_tensor = torch.tensor(initial_z, dtype=torch.float32).to(device)
                            
                            # Generate POE reconstruction
                            if is_multi_encoder:
                                poe_shape_logits, poe_grid_logits = model.multi_encoder.shared_decoder(
                                    poe_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                )
                            else:
                                poe_shape_logits, poe_grid_logits = model.decoder(
                                    poe_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                )
                            
                            print(f"DEBUG: Query initial POE reconstruction successful")
                            
                            trajectory_entry['query_poe_reconstructions']['initial'] = {
                                'shape_logits': poe_shape_logits.cpu().numpy(),
                                'grid_logits': poe_grid_logits.cpu().numpy()
                            }
                        except Exception as e:
                            print(f"    Warning: Could not generate query initial POE reconstruction for task '{task_key}': {e}")
                            trajectory_entry['query_poe_reconstructions']['initial'] = None
                        
                        # 3. Final POE reconstruction (optimized latent with support samples)
                        try:
                            print(f"DEBUG: Generating query final POE reconstruction for task '{task_key}'")
                            
                            # Use final optimized z vector (optimized with support samples)
                            final_z = task_trajectory['z_vectors'][-1]
                            if isinstance(final_z, torch.Tensor):
                                poe_z_tensor = final_z.to(device)
                            else:
                                poe_z_tensor = torch.tensor(final_z, dtype=torch.float32).to(device)
                            
                            # Generate POE reconstruction
                            if is_multi_encoder:
                                poe_shape_logits, poe_grid_logits = model.multi_encoder.shared_decoder(
                                    poe_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                )
                            else:
                                poe_shape_logits, poe_grid_logits = model.decoder(
                                    poe_z_tensor, query_input_tensor, target_seq=query_target_tensor
                                )
                            
                            print(f"DEBUG: Query final POE reconstruction successful")
                            
                            trajectory_entry['query_poe_reconstructions']['final'] = {
                                'shape_logits': poe_shape_logits.cpu().numpy(),
                                'grid_logits': poe_grid_logits.cpu().numpy()
                            }
                        except Exception as e:
                            print(f"    Warning: Could not generate query final POE reconstruction for task '{task_key}': {e}")
                            trajectory_entry['query_poe_reconstructions']['final'] = None
                        
                        break  # Found the matching trajectory entry
                else:
                    print(f"DEBUG: No query reconstructions found for task '{task_key}'")
    
    # Generate latent representations for support and query samples for visualization
    print(f"DEBUG: Generating latent representations for support and query samples")

    # Generate support sample latents
    support_latents = []
    for task_key, support_samples in task_samples.items():
        for i, (input_tensor, target_tensor) in enumerate(support_samples):
            try:
                # Generate latent representation
                if is_multi_encoder:
                    # Multi-encoder: generate for each encoder
                    for enc_idx in range(num_encoders):
                        if use_independent_decoder:
                            enc_z = model.multi_encoder.encoders[enc_idx](input_tensor, target_tensor)
                        else:
                            enc_output = model.multi_encoder.encoders[enc_idx](input_tensor, target_tensor)
                        
                        # Extract mu from VAE output (mu, logvar)
                        if isinstance(enc_output, tuple):
                            enc_z = enc_output[0]  # mu
                        else:
                            enc_z = enc_output
                        
                        support_latents.append({
                            'latent': enc_z.cpu().detach().numpy(),
                            'key': task_key,
                            'encoder': f'encoder_{enc_idx}',
                            'sample_type': 'support',
                            'sample_idx': i
                        })
                else:
                    # Single encoder
                    enc_output = model.encoder(input_tensor, target_tensor)

                    # Extract mu from VAE output (mu, logvar)
                    if isinstance(enc_output, tuple):
                        enc_z = enc_output[0]  # mu
                    else:
                        enc_z = enc_output
                
                    support_latents.append({
                        'latent': enc_z.cpu().detach().numpy(),
                        'key': task_key,
                        'encoder': 'encoder_0',
                        'sample_type': 'support',
                        'sample_idx': i
                    })
            except Exception as e:
                print(f"Warning: Could not generate support latent for task '{task_key}' sample {i}: {e}")

    # Generate query sample latents
    query_latents = []
    for query_recon in query_reconstructions:
        try:
            task_key = query_recon['task_key']
            query_input = query_recon['input']
            query_target = query_recon['target']
            
            # Convert to tensors
            query_input_tensor = torch.tensor(query_input, dtype=torch.float32).unsqueeze(0).to(device)
            query_target_tensor = torch.tensor(query_target, dtype=torch.float32).unsqueeze(0).to(device)
            
            if is_multi_encoder:
                # Multi-encoder: generate for each encoder
                for enc_idx in range(num_encoders):
                    if use_independent_decoder:
                        enc_output = model.multi_encoder.encoders[enc_idx](query_input_tensor, query_target_tensor)
                    else:
                        enc_output = model.multi_encoder.encoders[enc_idx](query_input_tensor, query_target_tensor)
                    
                    # Extract mu from VAE output (mu, logvar)
                    if isinstance(enc_output, tuple):
                        enc_z = enc_output[0]  # mu
                    else:
                        enc_z = enc_output
                    
                    query_latents.append({
                        'latent': enc_z.cpu().detach().numpy(),
                        'key': task_key,
                        'encoder': f'encoder_{enc_idx}',
                        'sample_type': 'query',
                        'sample_idx': 0
                    })
            else:
                # Single encoder
                enc_output = model.encoder(query_input_tensor, query_target_tensor)
                
                # Extract mu from VAE output (mu, logvar)
                if isinstance(enc_output, tuple):
                    enc_z = enc_output[0]  # mu
                else:
                    enc_z = enc_output
                
                query_latents.append({
                    'latent': enc_z.cpu().detach().numpy(),
                    'key': task_key,
                    'encoder': 'encoder_0',
                    'sample_type': 'query',
                    'sample_idx': 0
                })
        except Exception as e:
            print(f"Warning: Could not generate query latent for task '{task_key}': {e}")

    # Update evaluation_latent_data with the generated latents
    evaluation_latent_data['support_latents'] = support_latents
    evaluation_latent_data['query_latents'] = query_latents

    print(f"DEBUG: Generated {len(support_latents)} support latents and {len(query_latents)} query latents")

    return {
        'metrics': {
            'shape_accuracy': shape_accuracy,
            'grid_accuracy': grid_accuracy,
            'overall_accuracy': overall_accuracy,
            'sample_exact_accuracy': exact_accuracy,
            'total_samples': total_samples,
            'num_tasks': len(task_latents),
            'used_task_optimization': True,
            'support_loss': support_loss,
            'query_loss': query_loss
        },
        'reconstruction_results': {
            'query_reconstructions': query_reconstructions
        },
        'task_latent_data': task_latent_data,
        'evaluation_latent_data': evaluation_latent_data,  # Added for latent space visualization
        'task_trajectories': task_trajectories,
        'trajectory_info': trajectory_info
    }

def evaluate_model_original_bonnet_approach(model, samples_dataloader, queries_dataloader, device='cuda',
                                          encoder_idx=None, use_independent_decoder=False, support_key_mapping=None, query_key_mapping=None):
    """
    Original Bonnet approach: per-sample optimization averaged over supports to create unique latent for query.
    Latent space plots show all samples with latents directly sampled from encoder posterior.
    
    Args:
        model: The trained model to evaluate
        samples_dataloader: DataLoader containing support samples grouped by task
        queries_dataloader: DataLoader containing query samples
        device: Device to run evaluation on
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
        
    Returns:
        dict: Dictionary containing evaluation metrics and per-sample latent data
    """
    from utils.latent_functions import optimize_latent_z
    
    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']
    
    if not optimize_z_inference:
        print("WARNING: Original approach requires latent optimization to be enabled!")
        print("Falling back to regular evaluation...")
        return evaluate_model(model, samples_dataloader, queries_dataloader, device, 
                            encoder_idx, use_independent_decoder)
    
    # Check if this is a multi-encoder model
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
    
    print(f"=== ORIGINAL BONNET APPROACH EVALUATION ===")
    print(f"Model type: {'Multi-encoder' if is_multi_encoder else 'Single encoder'}")
    if is_multi_encoder:
        print(f"Number of encoders: {num_encoders}")
    print(f"Optimization steps: {latent_optimization['inference']['num_steps']}")
    print(f"Optimization learning rate: {latent_optimization['inference']['learning_rate']}")
    
    model.eval()
    
    # Step 1: Collect support samples grouped by task
    task_samples = collect_support_samples_by_task(samples_dataloader, support_key_mapping, device)
    
    if not task_samples:
        print("ERROR: No task samples found!")
        return {}
    
    # Step 2: Per-sample optimization for each support sample, then average for query
    print(f"\n=== PER-SAMPLE OPTIMIZATION ===")
    task_optimized_latents = {}  # Store optimized latents for each support sample per task
    task_averaged_latents = {}   # Store averaged latents for query inference
    task_trajectories = {}       # Store trajectory information for each task
    
    for task_key, support_samples in task_samples.items():
        print(f"\nProcessing task '{task_key}' with {len(support_samples)} support samples...")
        
        # Optimize each support sample individually (original Bonnet approach)
        optimized_latents = []
        task_trajectories[task_key] = []  # Store trajectories for this task
        
        for i, (input_seq, target_seq) in enumerate(support_samples):
            print(f"  Optimizing support sample {i+1}/{len(support_samples)}...")
            
            # Per-sample optimization (original approach)
            optimized_z, final_loss, trajectory = optimize_latent_z(
                model, input_seq, target_seq,
                num_steps=latent_optimization['inference']['num_steps'],
                lr=latent_optimization['inference']['learning_rate'],
                return_trajectory=True,
                encoder_idx=encoder_idx,
                use_independent_decoder=use_independent_decoder
            )
            
            optimized_latents.append(optimized_z)
            
            # Add input/target sample data to trajectory for reconstruction visualization
            if trajectory:
                trajectory['input_sample'] = input_seq.detach().cpu().numpy().flatten()
                trajectory['target_sample'] = target_seq.detach().cpu().numpy().flatten()
                
                # Generate reconstructions at key trajectory points for visualization
                print(f"    Generating reconstructions for trajectory visualization...")
                try:
                    with torch.no_grad():
                        z_vectors = trajectory.get('z_vectors', [])
                        if len(z_vectors) >= 3:
                            # Get initial, mid, and final z vectors
                            initial_z = z_vectors[0]  # First optimization step
                            mid_z = z_vectors[len(z_vectors)//2]  # Middle step
                            final_z = z_vectors[-1]  # Final optimization step
                            
                            # Generate reconstructions using decoder
                            poe_trajectory_reconstructions = {}
                            individual_encoder_reconstructions = {}
                            
                            # POE reconstructions (using the combined latent) - map to expected key names
                            stage_mapping = {'initial': 'initial', 'mid': 'mid', 'final': 'final'}
                            
                            # For 1 encoder case, use original encoder latent for initial reconstruction to match encoder reconstruction
                            if 'encoder_mu' in trajectory and 'encoder_log_var' in trajectory:
                                encoder_mu = trajectory['encoder_mu']
                                encoder_log_var = trajectory['encoder_log_var']
                                original_encoder_z = model.reparameterize(encoder_mu, encoder_log_var)
                                # Use original encoder latent for initial, trajectory latents for mid/final
                                stage_latents = [('initial', original_encoder_z), ('mid', mid_z), ('final', final_z)]
                            else:
                                # Fallback to trajectory latents
                                stage_latents = [('initial', initial_z), ('mid', mid_z), ('final', final_z)]
                            
                            for stage, z_vec in stage_latents:
                                try:
                                    if use_independent_decoder:
                                        shape_logits, grid_logits = model.multi_encoder.decoders[encoder_idx or 0](z_vec, input_seq, target_seq=target_seq)
                                    else:
                                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(z_vec, input_seq, target_seq=target_seq)
                                    
                                    shape_pred = torch.argmax(shape_logits, dim=-1)
                                    grid_pred = torch.argmax(grid_logits, dim=-1)
                                    
                                    # Use the correct key names that visualization expects
                                    expected_key = stage_mapping[stage]
                                    poe_trajectory_reconstructions[expected_key] = {
                                        'shape_pred': shape_pred.detach().cpu().numpy(),
                                        'grid_pred': grid_pred.detach().cpu().numpy(),
                                        'shape_logits': shape_logits.detach().cpu().numpy(),
                                        'grid_logits': grid_logits.detach().cpu().numpy()
                                    }
                                except Exception as e:
                                    print(f"      Warning: Could not generate {stage} reconstruction: {e}")
                            
                            # Individual encoder reconstructions (for compatibility)
                            if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
                                # Generate reconstructions for ALL encoders
                                for enc_idx in range(num_encoders):
                                    try:
                                        # Use the original encoder latent (non-optimized) for encoder reconstruction
                                        if 'encoder_mu' in trajectory and 'encoder_log_var' in trajectory:
                                            encoder_mu = trajectory['encoder_mu']
                                            encoder_log_var = trajectory['encoder_log_var']
                                            # Generate non-optimized latent using reparameterization
                                            original_encoder_z = model.reparameterize(encoder_mu, encoder_log_var)
                                        else:
                                            # Fallback to initial_z if encoder mu/log_var not available
                                            original_encoder_z = initial_z
                                        
                                        if use_independent_decoder and hasattr(model.multi_encoder, 'decoders'):
                                            shape_logits, grid_logits = model.multi_encoder.decoders[enc_idx](original_encoder_z, input_seq, target_seq=target_seq)
                                        else:
                                            shape_logits, grid_logits = model.multi_encoder.shared_decoder(original_encoder_z, input_seq, target_seq=target_seq)
                                        
                                        shape_pred = torch.argmax(shape_logits, dim=-1)
                                        grid_pred = torch.argmax(grid_logits, dim=-1)
                                        
                                        individual_encoder_reconstructions[f'encoder_{enc_idx}'] = {
                                            'shape_pred': shape_pred.detach().cpu().numpy(),
                                            'grid_pred': grid_pred.detach().cpu().numpy(),
                                            'shape_logits': shape_logits.detach().cpu().numpy(),
                                            'grid_logits': grid_logits.detach().cpu().numpy()
                                        }
                                    except Exception as e:
                                        print(f"      Warning: Could not generate encoder {enc_idx} reconstruction: {e}")
                            
                            # Store reconstructions in trajectory with correct key names
                            trajectory['poe_trajectory_reconstructions'] = poe_trajectory_reconstructions
                            trajectory['individual_encoder_reconstructions'] = individual_encoder_reconstructions
                            
                            print(f"      Generated {len(poe_trajectory_reconstructions)} trajectory reconstructions")
                            print(f"      POE reconstruction keys: {list(poe_trajectory_reconstructions.keys())}")
                            print(f"      Generated {len(individual_encoder_reconstructions)} encoder reconstructions")
                            print(f"      Encoder reconstruction keys: {list(individual_encoder_reconstructions.keys())}")
                        
                except Exception as e:
                    print(f"    Warning: Could not generate trajectory reconstructions: {e}")
            
            task_trajectories[task_key].append(trajectory)  # Store trajectory
            print(f"    Sample {i+1} final loss: {final_loss:.4f}")
        
        # Average the optimized latents to create unique latent for query (original approach)
        averaged_latent = torch.stack(optimized_latents).mean(dim=0)
        
        task_optimized_latents[task_key] = optimized_latents
        task_averaged_latents[task_key] = averaged_latent
        
        print(f"  Task '{task_key}' averaged latent created from {len(optimized_latents)} support samples")
    
    # Step 3: Evaluate on query samples using averaged latents
    print(f"\n=== QUERY EVALUATION ===")
    total_shape_correct = 0
    total_shape_tokens = 0
    total_grid_correct = 0
    total_grid_tokens = 0
    total_exact_correct = 0
    total_samples = 0
    
    query_reconstructions = []
    
    with torch.no_grad():
        for batch in queries_dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            batch_input_q = batch_input_q.to(device)
            batch_target_q = batch_target_q.to(device)
            
            # For each query sample, use the corresponding averaged task latent
            for i in range(batch_input_q.size(0)):
                query_input = batch_input_q[i:i+1]
                query_target = batch_target_q[i:i+1]
                
                # Determine which task latent to use
                if batch_keys is not None and query_key_mapping is not None:
                    key_idx = batch_keys[i].item()
                    task_key = query_key_mapping[key_idx]
                    if task_key in task_averaged_latents:
                        query_latent = task_averaged_latents[task_key]
                    else:
                        print(f"WARNING: Task key '{task_key}' not found in optimized latents")
                        continue
                else:
                    # Fallback: use first available task latent
                    task_key = list(task_averaged_latents.keys())[0]
                    query_latent = task_averaged_latents[task_key]
                
                # Decode using the averaged latent
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    if use_independent_decoder and encoder_idx is not None:
                        shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](
                            query_latent, query_input, target_seq=query_target)
                    else:
                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                            query_latent, query_input, target_seq=query_target)
                else:
                    shape_logits, grid_logits = model.decoder(query_latent, query_input, target_seq=query_target)
                
                # Evaluate reconstruction
                shape_pred = shape_logits.argmax(dim=-1)
                grid_pred = grid_logits.argmax(dim=-1)
                
                shape_target = query_target[0, 900:902].long()
                grid_target = query_target[0, :900].long()
                
                # Calculate metrics
                shape_correct = (shape_pred == shape_target).sum().item()
                grid_correct = (grid_pred == grid_target).sum().item()
                
                total_shape_correct += shape_correct
                total_shape_tokens += 2  # Shape has 2 tokens
                total_grid_correct += grid_correct
                total_grid_tokens += 900  # Grid has 900 tokens
                
                # Check exact match
                exact_match = (shape_correct == 2 and grid_correct == 900)
                total_exact_correct += exact_match
                total_samples += 1
                
                # Store reconstruction for analysis
                query_reconstructions.append({
                    'input': query_input.cpu().numpy(),
                    'target': query_target.cpu().numpy(),
                    'shape_pred': shape_pred.cpu().numpy(),
                    'grid_pred': grid_pred.cpu().numpy(),
                    'shape_logits': shape_logits.cpu().numpy(),
                    'grid_logits': grid_logits.cpu().numpy(),
                    'exact_match': exact_match,
                    'task_key': task_key
                })
    
    # Calculate final metrics
    shape_accuracy = total_shape_correct / total_shape_tokens if total_shape_tokens > 0 else 0.0
    grid_accuracy = total_grid_correct / total_grid_tokens if total_grid_tokens > 0 else 0.0
    exact_accuracy = total_exact_correct / total_samples if total_samples > 0 else 0.0
    
    print(f"Shape accuracy: {shape_accuracy:.4f}")
    print(f"Grid accuracy: {grid_accuracy:.4f}")
    print(f"Exact match accuracy: {exact_accuracy:.4f}")
    
    # Prepare latent data for visualization (per-sample latents directly from encoder)
    print(f"\n=== COLLECTING LATENT DATA FOR VISUALIZATION ===")
    
    # Collect all support and query samples for latent space visualization
    all_support_latents = []
    all_query_latents = []
    all_support_keys = []
    all_query_keys = []
    
    # Extract support latents directly from encoder (no optimization for visualization)
    for task_key, support_samples in task_samples.items():
        for input_seq, target_seq in support_samples:
            with torch.no_grad():
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    if encoder_idx is not None:
                        mu, logvar, _ = model.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                    else:
                        result = model(input_seq, target_seq)
                        (shape_logits, grid_logits), mu, logvar, _ = result
                else:
                    _, mu, logvar = model(input_seq, target_seq)
                
                # Sample from encoder posterior (original Bonnet approach)
                z = model.reparameterize(mu, logvar)
                all_support_latents.append(z[0].cpu().numpy())
                all_support_keys.append(task_key)
    
    # Extract query latents directly from encoder
    with torch.no_grad():
        for batch in queries_dataloader:
            if len(batch) >= 3:
                batch_input_q, batch_target_q, batch_keys = batch[:3]
            else:
                batch_input_q, batch_target_q = batch[:2]
                batch_keys = None
            
            batch_input_q = batch_input_q.to(device)
            batch_target_q = batch_target_q.to(device)
            
            for i in range(batch_input_q.size(0)):
                query_input = batch_input_q[i:i+1]
                query_target = batch_target_q[i:i+1]
                
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    if encoder_idx is not None:
                        mu, logvar, _ = model.multi_encoder.encoders[encoder_idx](query_input, query_target)
                    else:
                        result = model(query_input, query_target)
                        (shape_logits, grid_logits), mu, logvar, _ = result
                else:
                    _, mu, logvar = model(query_input, query_target)
                
                # Sample from encoder posterior (original Bonnet approach)
                z = model.reparameterize(mu, logvar)
                all_query_latents.append(z[0].cpu().numpy())
                
                # Determine task key
                if batch_keys is not None and query_key_mapping is not None:
                    key_idx = batch_keys[i].item()
                    task_key = query_key_mapping[key_idx]
                else:
                    task_key = "unknown"
                all_query_keys.append(task_key)
    
    # Prepare results in original Bonnet format
    latent_data = {
        'support_latents': all_support_latents,
        'query_latents': all_query_latents,
        'support_keys': all_support_keys,
        'query_keys': all_query_keys,
        'task_optimized_latents': task_optimized_latents,
        'task_averaged_latents': task_averaged_latents
    }
    
    # Prepare trajectory information for plotting
    trajectory_info = {}
    for task_key, trajectories in task_trajectories.items():
        if trajectories and len(trajectories) > 0:
            # Use the first trajectory as representative for this task
            trajectory_info[task_key] = trajectories[0]
            # Add task key to trajectory for identification
            trajectory_info[task_key]['evaluated_key'] = task_key
            
            # Add query reconstructions for this task in the format expected by visualization
            task_query_reconstructions = [q for q in query_reconstructions if q['task_key'] == task_key]
            if task_query_reconstructions:
                # Store original list format
                trajectory_info[task_key]['query_reconstructions'] = task_query_reconstructions
                
                # Also format for visualization - use first query reconstruction as representative
                first_query = task_query_reconstructions[0]
                
                # Add query input and target data for visualization (flatten like input_sample/target_sample)
                trajectory_info[task_key]['query_input'] = first_query['input'].flatten()
                trajectory_info[task_key]['query_target'] = first_query['target'].flatten()
                
                # Format for encoder reconstructions (expecting encoder_0 key)
                trajectory_info[task_key]['query_encoder_reconstructions'] = {
                    'encoder_0': {
                        'shape_pred': first_query['shape_pred'],
                        'grid_pred': first_query['grid_pred'],
                        'shape_logits': first_query['shape_logits'],
                        'grid_logits': first_query['grid_logits']
                    }
                }
                
                # Format for POE reconstructions (expecting initial/final keys)
                trajectory_info[task_key]['query_poe_reconstructions'] = {
                    'initial': {
                        'shape_pred': first_query['shape_pred'],
                        'grid_pred': first_query['grid_pred'],
                        'shape_logits': first_query['shape_logits'],
                        'grid_logits': first_query['grid_logits']
                    },
                    'final': {
                        'shape_pred': first_query['shape_pred'],
                        'grid_pred': first_query['grid_pred'],
                        'shape_logits': first_query['shape_logits'],
                        'grid_logits': first_query['grid_logits']
                    }
                }
    
    # Also create evaluation_latent_data in the expected format for plotting
    evaluation_latent_data = {
        'support': {
            'poe': {
                'latent_zs': all_support_latents,
                'keys': all_support_keys
            },
            'task_keys': list(set(all_support_keys))
        },
        'query': {
            'poe': {
                'latent_zs': all_query_latents,
                'keys': all_query_keys
            },
            'task_keys': list(set(all_query_keys))
        }
    }
    
    results = {
        'shape_accuracy': shape_accuracy,
        'grid_accuracy': grid_accuracy,
        'exact_accuracy': exact_accuracy,
        'total_samples': total_samples,
        'query_reconstructions': query_reconstructions,
        'latent_data': latent_data,
        'evaluation_latent_data': evaluation_latent_data,  # Add for plotting compatibility
        'trajectory_info': trajectory_info,  # Add trajectory information
        'evaluation_method': 'original_bonnet_approach'
    }
    
    return results