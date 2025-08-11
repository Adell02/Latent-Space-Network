import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import numpy as np

from utils.model_utils import set_seed, prepare_dataloader, collect_latent_data
from torch.utils.data import TensorDataset, DataLoader
from re_arc.main import generate_and_process_tasks
from utils.data_preparation import load_and_process_original_arc_evaluation_tasks, get_available_evaluation_tasks, load_original_arc_evaluation_task, transform_grid_to_sequence
import random

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
            if data_type == 'query':
                dummy = torch.zeros_like(batch_outputs)
                mu, log_var, _ = model.encoder(batch_inputs, dummy)
            else:
                mu, log_var,_ = model.encoder(batch_inputs, batch_outputs)
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
                if data_type == 'query':
                    dummy = torch.zeros_like(batch_outputs)
                    mu, log_var, _ = model.multi_encoder.encoders[encoder_idx](batch_inputs, dummy)
                else:
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
                    if data_type == 'query':
                        dummy = torch.zeros_like(batch_outputs)
                        mu, log_var = model(batch_inputs, dummy)[1:3]
                    else:
                        mu, log_var = model(batch_inputs, batch_outputs)[1:3]
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
              encoder_idx=None, use_independent_decoder=False):
    """
    Generate new data and evaluate the model using Bonnet approach (per-sample optimization + averaging).
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: Number of input-output pairs to generate for support
        n_queries: Number of queries to do inference
        device: Device to run evaluation on
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
    
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
            'evaluation_strategy': 'bonnet_per_sample_optimization',
            'latent_type': 'reparameterized',
            'encoder_idx': encoder_idx,
            'use_independent_decoder': use_independent_decoder,
            'evaluation_mode': f"{eval_mode}+{decoder_mode}_decoder"
        },
        'key_results': {},
        'aggregated_metrics': {}
    }
    
    print(f"=== BONNET EVALUATION CONFIGURATION ===")
    print(f"Keys to evaluate: {keys}")
    print(f"Support samples per key: {n_samples}")
    print(f"Query samples per key: {n_queries}")
    print(f"Evaluation strategy: Bonnet approach (per-sample optimization + averaging)")
    print(f"Maximum batch size: {MAX_BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print("=" * 50)
    
    # Collect training latent representations once at the beginning
    print(f"\n>>> COLLECTING TRAINING LATENT REPRESENTATIONS <<<")
    training_latent_data = collect_training_latent_representations(model, run_dir, device)
    if training_latent_data:
        print(f"[OK] Collected training latent data: {training_latent_data.get('collection_info', {})}")
        results['training_latent_data'] = training_latent_data
    else:
        print("[WARNING] Could not collect training latent representations")
    
    # Initialize aggregated metrics
    aggregated_metrics = {
        'total_keys': len(keys),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'average_metrics': {},
        'per_key_summary': {}
    }
    
    # Evaluate each key separately using Bonnet approach
    for key_idx, key in enumerate(keys):
        print(f"\n[KEY {key_idx+1}/{len(keys)}] EVALUATING '{key}'")
        print("-" * 50)
        
        try:
            # Initialize OOD variables
            ood_task_keys = []  # ✅ FIX: Initialize ood_task_keys before conditional blocks
            
            # Check if OOD evaluation is enabled
            eval_data_settings = settings.get_evaluation_data_settings()
            use_ood_for_evaluation = eval_data_settings.get('use_ood_for_evaluation', True)
            
            if use_ood_for_evaluation:
                print(f"  Using OOD evaluation data from original ARC tasks...")
                
                # Generate OOD evaluation dataset using original ARC tasks
                from utils.data_preparation import generate_ood_evaluation_dataset
                ood_evaluation_data = generate_ood_evaluation_dataset(
                    keys, n_samples, n_queries, seed=seed
                )
                
                if not ood_evaluation_data or key not in ood_evaluation_data:
                    print(f"  [WARNING] No OOD evaluation data available for key '{key}', falling back to synthetic samples")
                    use_ood_for_evaluation = False
                else:
                    # Use OOD evaluation data
                    key_ood_data = ood_evaluation_data[key]
                    input_samples_sequences = key_ood_data['support']['input_sequences']
                    output_samples_sequences = key_ood_data['support']['output_sequences']
                    input_queries_sequences = key_ood_data['query']['input_sequences']
                    output_queries_sequences = key_ood_data['query']['output_sequences']
                    
                    # Store the actual OOD task keys used
                    ood_task_keys = key_ood_data.get('ood_task_keys', [])
                    
                    print(f"  Using {len(input_samples_sequences)} support and {len(input_queries_sequences)} query OOD samples for key '{key}'")
                    print(f"  OOD samples from tasks: {ood_task_keys}")
            else:
                # Use synthetic samples for evaluation
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
            
            # Create dataloaders for this key
            support_batch_size = min(MAX_BATCH_SIZE, n_samples)
            query_batch_size = min(MAX_BATCH_SIZE, n_queries)
            
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
            
            # Always use Bonnet approach for evaluation
            print(f"  Running Bonnet evaluation for key '{key}' ({eval_mode} + {decoder_mode} decoder)...")
            key_metrics = evaluate_model_original_bonnet_approach(
                model, samples_dataloader, queries_dataloader, device=device,
                encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder,
                support_key_mapping=support_key_mapping, query_key_mapping=query_key_mapping
            )
            
            # Store key-specific results
            results['key_results'][key] = {
                'key': key,
                'support_samples': len(input_samples_sequences),
                'query_samples': len(input_queries_sequences),
                'metrics': key_metrics.get('metrics', {}),
                'reconstruction_results': key_metrics.get('reconstruction_results', []),
                'evaluation_latent_data': key_metrics.get('evaluation_latent_data', {}),
                'trajectory_info': key_metrics.get('trajectory_info', []),
                'task_latent_data': key_metrics.get('task_latent_data', {}),
                'evaluation_method': 'bonnet_per_sample_optimization',
                'raw_data': {
                    'input_samples_sequences': input_samples_sequences,
                    'output_samples_sequences': output_samples_sequences,
                    'input_queries_sequences': input_queries_sequences,
                    'output_queries_sequences': output_queries_sequences,
                    # ✅ ADD: Store the actual OOD task keys used
                    'ood_task_keys': ood_task_keys
                }
            }
            
            # Add training latent data reference
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
                    'trajectory_samples': len(key_metrics.get('trajectory_info', []))
                }
                aggregated_metrics['per_key_summary'][key] = key_summary
                
                print(f"  [OK] Key '{key}' Results:")
                print(f"    Support loss: {key_summary['support_loss']:.4f}")
                print(f"    Query loss: {key_summary['query_loss']:.4f}")
                print(f"    Shape accuracy: {key_summary['shape_accuracy']:.4f}")
                print(f"    Grid accuracy: {key_summary['grid_accuracy']:.4f}")
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
    
    # Calculate aggregated metrics
    if aggregated_metrics['per_key_summary']:
        successful_keys = list(aggregated_metrics['per_key_summary'].keys())
        
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
    
    print(f"\n=== BONNET EVALUATION COMPLETE ===")
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
    
    # Store evaluation data at top level for visualization
    all_task_latent_data = {}
    all_task_trajectories = {}
    
    for key, key_results in results['key_results'].items():
        if 'task_latent_data' in key_results:
            all_task_latent_data.update(key_results['task_latent_data'].get('task_latents', {}))
        if 'task_trajectories' in key_results:
            all_task_trajectories.update(key_results['task_trajectories'])
    
    results['task_latent_data'] = {'task_latents': all_task_latent_data}
    results['task_trajectories'] = all_task_trajectories
    
    # Store evaluation_latent_data at top level
    first_key = list(results['key_results'].keys())[0]
    if 'evaluation_latent_data' in results['key_results'][first_key]:
        results['evaluation_latent_data'] = results['key_results'][first_key]['evaluation_latent_data']
    
    return results


def evaluate_model_original_bonnet_approach(model, samples_dataloader, queries_dataloader, device='cuda',
                                          encoder_idx=None, use_independent_decoder=False, support_key_mapping=None, query_key_mapping=None):
    """
    Original Bonnet approach: Per-sample optimization, then average latents for query inference.
    """
    print(f"\n=== ORIGINAL BONNET APPROACH EVALUATION ===")
    print(f"DEBUG: Function evaluate_model_original_bonnet_approach called")
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Model is_multi_encoder: {getattr(model, 'is_multi_encoder', 'Not set')}")
    
    # Get model configuration
    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
        num_encoders = len(model.multi_encoder.encoders)
        print(f"DEBUG: Multi-encoder model with {num_encoders} encoders")
    else:
        num_encoders = 1
        print(f"DEBUG: Single encoder model")
    
    # Get latent optimization settings from settings
    from utils.settings_manager import settings
    latent_optimization = settings.get_latent_optimization()
    print(f"DEBUG: Loaded latent optimization settings: {latent_optimization}")
    
    # Check if this is a multi-encoder model
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
    
    print(f"=== ORIGINAL BONNET APPROACH EVALUATION ===")
    print(f"Model type: {'Multi-encoder' if is_multi_encoder else 'Single encoder'}")
    print(f"is_multi_encoder: {is_multi_encoder}")
    print(f"num_encoders: {num_encoders}")
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
                trajectory['input_sample'] = input_seq.detach().cpu().numpy().squeeze()
                trajectory['target_sample'] = target_seq.detach().cpu().numpy().squeeze()
                
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
                                        shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx or 0](z_vec, input_seq, target_seq=target_seq)
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
                                        
                                        if use_independent_decoder and hasattr(model.multi_encoder, 'independent_decoders'):
                                            shape_logits, grid_logits = model.multi_encoder.independent_decoders[enc_idx](original_encoder_z, input_seq, target_seq=target_seq)
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
                
                # Ensure query_target has correct shape (900 grid + 2 shape = 902)
                if query_target.shape[-1] != 902:
                    if query_target.shape[-1] == 900:
                        # Add shape information if missing
                        shape_info = torch.tensor([30, 30], dtype=torch.float32, device=device).unsqueeze(0)
                        query_target = torch.cat([query_target, shape_info], dim=-1)
                    elif query_target.shape[-1] > 902:
                        query_target = query_target[..., :902]
                
                # Decode using the averaged latent
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    if use_independent_decoder and encoder_idx is not None:
                        shape_logits, grid_logits = model.multi_encoder.independent_decoders[encoder_idx](
                            query_latent, query_input, target_seq=None)
                    else:
                        shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                            query_latent, query_input, target_seq=None)
                else:
                    shape_logits, grid_logits = model.decoder(query_latent, query_input, target_seq=None)
                
                # Evaluate reconstruction
                shape_pred = shape_logits.argmax(dim=-1)
                grid_pred = grid_logits.argmax(dim=-1)
                
                shape_target = query_target[0, 900:902].long()
                grid_target = query_target[0, :900].long()
                
                # Calculate metrics using only active pixels from target
                rows = int(query_target[0, 900].item())
                cols = int(query_target[0, 901].item())
                active = max(rows * cols, 0)

                shape_correct = (shape_pred == shape_target).sum().item()
                total_shape_correct += shape_correct
                total_shape_tokens += 2

                if active > 0:
                    grid_correct_active = (grid_pred[0, :active] == grid_target[:active]).sum().item()
                    total_grid_correct += grid_correct_active
                    total_grid_tokens += active
                else:
                    # no active pixels; treat as exact shape-only sample
                    total_grid_tokens += 0

                exact_match = (shape_correct == 2 and (active == 0 or grid_correct_active == active))
                total_exact_correct += 1 if exact_match else 0
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
    # === ADDING QUERY SAMPLE INFORMATION TO TRAJECTORY DATA ===
    print(f"\n=== DEBUG: Adding query sample information to trajectory data ===")
    print(f"DEBUG: Number of task_trajectories: {len(task_trajectories)}")
    print(f"DEBUG: Number of query_reconstructions: {len(query_reconstructions)}")
    print(f"DEBUG: Model is_multi_encoder: {getattr(model, 'is_multi_encoder', 'Not set')}")
    print(f"DEBUG: num_encoders variable: {num_encoders if 'num_encoders' in locals() else 'Not defined'}")
    
    for task_key, trajectories in task_trajectories.items():
        print(f"DEBUG: Processing task_key: {task_key}")
        if trajectories:
            trajectory = trajectories[0]
            query_samples_for_task = [q for q in query_reconstructions if q['task_key'] == task_key]
            print(f"DEBUG: Found {len(query_samples_for_task)} query samples for task {task_key}")
            if query_samples_for_task:
                first_query = query_samples_for_task[0]
                print(f"DEBUG: first_query['target'] shape: {first_query['target'].shape}")
                print(f"DEBUG: first_query['target'] type: {type(first_query['target'])}")
                # Ensure the target has the correct shape before storing
                target_data = first_query['target'].squeeze()
                if len(target_data.shape) == 1 and target_data.shape[0] == 900:
                    # Add shape information if missing
                    shape_info = np.array([30, 30])  # Default shape
                    target_data = np.concatenate([target_data, shape_info])
                    print(f"DEBUG: Fixed target_data shape from 900 to {len(target_data)}")
                
                trajectory['query_input'] = first_query['input'].squeeze()
                trajectory['query_target'] = target_data
                query_encoder_reconstructions = {}
                query_poe_reconstructions = {}
                
                # Prepare input tensor for query (NEVER use GT for model inputs)
                input_tensor = torch.tensor(first_query['input'], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Ensure input has correct shape (should be 902 for complete sequence including shape tokens)
                if input_tensor.shape[-1] != 902:
                    if input_tensor.shape[-1] > 902:
                        input_tensor = input_tensor[..., :902]
                    else:
                        padding = torch.zeros(1, 902 - input_tensor.shape[-1], device=device)
                        input_tensor = torch.cat([input_tensor, padding], dim=-1)
                
                # Ensure input is 2D: [batch, seq_len]
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.squeeze(1)  # Remove middle dimension
                
                # Create a dummy target tensor to satisfy interfaces that expect two sequences
                dummy_target_tensor = torch.zeros_like(input_tensor)
                
                # Verify input sequence has the correct length (902)
                if input_tensor.shape[1] != 902:
                    raise ValueError(f"Sequence length mismatch: input={input_tensor.shape}. Input should be [batch, 902]")
                
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    print(f"DEBUG: Model is multi-encoder, num_encoders: {num_encoders}")
                    
                    # Generate individual encoder latents from query sample (non-optimized)
                    encoder_mus = []
                    encoder_log_vars = []
                    encoder_zs = []
                    
                    for enc_idx in range(num_encoders):
                        print(f"DEBUG: Generating latent for encoder {enc_idx}")
                        try:
                            # Check if input tensor is valid
                            print(f"DEBUG: input_tensor shape: {input_tensor.shape}, target_tensor shape: {target_tensor.shape}")
                            if input_tensor.numel() == 0 or target_tensor.numel() == 0:
                                raise ValueError("Empty input or target tensor")
                            
                            # Check for empty dimensions
                            if 0 in input_tensor.shape or 0 in target_tensor.shape:
                                raise ValueError(f"Tensor has empty dimension: input_shape={input_tensor.shape}, target_shape={target_tensor.shape}")
                            
                            # Ensure tensors have the expected shape (batch_size, seq_len)
                            if len(input_tensor.shape) != 2 or len(target_tensor.shape) != 2:
                                raise ValueError(f"Unexpected tensor shapes: input_shape={input_tensor.shape}, target_shape={target_tensor.shape}")
                            
                            # Check if sequence length is valid
                            if input_tensor.shape[1] == 0 or target_tensor.shape[1] == 0:
                                raise ValueError(f"Empty sequence length: input_shape={input_tensor.shape}, target_shape={target_tensor.shape}")
                            
                            # Handle tensor shape mismatch - ensure both tensors have the same sequence length
                            if input_tensor.shape[1] != target_tensor.shape[1]:
                                print(f"DEBUG: Shape mismatch detected: input={input_tensor.shape}, target={target_tensor.shape}")
                                # Use the minimum length to avoid index errors
                                min_seq_len = min(input_tensor.shape[1], target_tensor.shape[1])
                                if min_seq_len == 0:
                                    raise ValueError(f"Both tensors have zero sequence length after truncation")
                                
                                # Truncate both tensors to the same length
                                input_tensor = input_tensor[:, :min_seq_len]
                                target_tensor = target_tensor[:, :min_seq_len]
                                print(f"DEBUG: Truncated to shape: input={input_tensor.shape}, target={target_tensor.shape}")
                            
                            # Get encoder latent (non-optimized) from query sample (use dummy target)
                            mu, log_var, _ = model.multi_encoder.encoders[enc_idx](input_tensor, dummy_target_tensor)
                            z = model.reparameterize(mu, log_var)
                            
                            encoder_mus.append(mu)
                            encoder_log_vars.append(log_var)
                            encoder_zs.append(z)
                            
                            print(f"DEBUG: Successfully generated latent for encoder {enc_idx}")
                        except Exception as e:
                            print(f"      Warning: Could not generate latent for encoder {enc_idx}: {e}")
                            # Use zeros as fallback
                            latent_dim = getattr(model, 'latent_dim', 16)
                            fallback_z = torch.zeros(1, latent_dim, device=device)
                            fallback_mu = torch.zeros(1, latent_dim, device=device)
                            fallback_log_var = torch.zeros(1, latent_dim, device=device)
                            encoder_zs.append(fallback_z)
                            encoder_mus.append(fallback_mu)
                            encoder_log_vars.append(fallback_log_var)
                    
                    # Generate individual encoder reconstructions (using each encoder's own latent)
                    for enc_idx in range(num_encoders):
                        print(f"DEBUG: Generating reconstruction for encoder {enc_idx}")
                        try:
                            if use_independent_decoder and hasattr(model.multi_encoder, 'independent_decoders'):
                                print(f"DEBUG: Using independent decoder for encoder {enc_idx}")
                                shape_logits, grid_logits = model.multi_encoder.independent_decoders[enc_idx](
                                    encoder_zs[enc_idx], input_tensor, target_seq=None)
                            else:
                                print(f"DEBUG: Using shared decoder for encoder {enc_idx}")
                                shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                                    encoder_zs[enc_idx], input_tensor, target_seq=None)
                            
                            shape_pred = torch.argmax(shape_logits, dim=-1)
                            grid_pred = torch.argmax(grid_logits, dim=-1)
                            query_encoder_reconstructions[f'encoder_{enc_idx}'] = {
                                'shape_pred': shape_pred.detach().cpu().numpy(),
                                'grid_pred': grid_pred.detach().cpu().numpy(),
                                'shape_logits': shape_logits.detach().cpu().numpy(),
                                'grid_logits': grid_logits.detach().cpu().numpy()
                            }
                            print(f"DEBUG: Successfully generated reconstruction for encoder {enc_idx}")
                        except Exception as e:
                            print(f"      Warning: Could not generate query encoder {enc_idx} reconstruction: {e}")
                    
                    # Generate POE initial (using PoE of all encoders' latents - non-optimized)
                    if len(encoder_mus) >= 1:
                        try:
                            print(f"DEBUG: Generating POE initial reconstruction with {len(encoder_mus)} encoders")
                            from models.base_model import gaussian_poe
                            
                            # Stack encoder latents for PoE
                            mu_stack = torch.stack(encoder_mus)
                            logvar_stack = torch.stack(encoder_log_vars)
                            poe_mu, poe_logvar = gaussian_poe(mu_stack, logvar_stack)
                            
                            # Generate PoE latent (non-optimized)
                            poe_z_initial = model.reparameterize(poe_mu, poe_logvar)
                            
                            # Generate reconstruction using PoE latent (autoregressive)
                            shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                                poe_z_initial, input_tensor, target_seq=None)
                            
                            shape_pred = torch.argmax(shape_logits, dim=-1)
                            grid_pred = torch.argmax(grid_logits, dim=-1)
                            query_poe_reconstructions['initial'] = {
                                'shape_pred': shape_pred.detach().cpu().numpy(),
                                'grid_pred': grid_pred.detach().cpu().numpy(),
                                'shape_logits': shape_logits.detach().cpu().numpy(),
                                'grid_logits': grid_logits.detach().cpu().numpy()
                            }
                            print(f"DEBUG: Successfully generated POE initial reconstruction")
                        except Exception as e:
                            print(f"      Warning: Could not generate POE initial reconstruction: {e}")
                    else:
                        print(f"DEBUG: Skipping POE initial (no encoder latents available)")
                    
                    # Generate POE final (using optimized latent from support samples)
                    if task_key in task_averaged_latents:
                        try:
                            print(f"DEBUG: Generating POE final reconstruction using optimized latent")
                            optimized_latent = task_averaged_latents[task_key]
                            
                            shape_logits, grid_logits = model.multi_encoder.shared_decoder(
                                optimized_latent, input_tensor, target_seq=None)
                            
                            shape_pred = torch.argmax(shape_logits, dim=-1)
                            grid_pred = torch.argmax(grid_logits, dim=-1)
                            query_poe_reconstructions['final'] = {
                                'shape_pred': shape_pred.detach().cpu().numpy(),
                                'grid_pred': grid_pred.detach().cpu().numpy(),
                                'shape_logits': shape_logits.detach().cpu().numpy(),
                                'grid_logits': grid_logits.detach().cpu().numpy()
                            }
                            print(f"DEBUG: Successfully generated POE final reconstruction")
                        except Exception as e:
                            print(f"      Warning: Could not generate POE final reconstruction: {e}")
                else:
                    print(f"DEBUG: Model is single encoder")
                    # For single encoder, generate reconstruction for encoder_0 (no GT)
                    print(f"DEBUG: Generating reconstruction for single encoder")
                    try:
                        # Get encoder latent (non-optimized) from query sample (use dummy target)
                        mu, log_var, _ = model.encoder(input_tensor, dummy_target_tensor)
                        z = model.reparameterize(mu, log_var)
                        
                        shape_logits, grid_logits = model.decoder(z, input_tensor, target_seq=None)
                        shape_pred = torch.argmax(shape_logits, dim=-1)
                        grid_pred = torch.argmax(grid_logits, dim=-1)
                        query_encoder_reconstructions['encoder_0'] = {
                            'shape_pred': shape_pred.detach().cpu().numpy(),
                            'grid_pred': grid_pred.detach().cpu().numpy(),
                            'shape_logits': shape_logits.detach().cpu().numpy(),
                            'grid_logits': grid_logits.detach().cpu().numpy()
                        }
                        print(f"DEBUG: Successfully generated reconstruction for single encoder")
                    except Exception as e:
                        print(f"      Warning: Could not generate query single encoder reconstruction: {e}")
                
                # Store query reconstructions in trajectory
                trajectory['query_encoder_reconstructions'] = query_encoder_reconstructions
                trajectory['query_poe_reconstructions'] = query_poe_reconstructions
                print(f"    Added query sample information to trajectory for task '{task_key}'")
                print(f"    Query encoder reconstructions: {list(query_encoder_reconstructions.keys())}")
                print(f"    Query POE reconstructions: {list(query_poe_reconstructions.keys())}")
            else:
                print(f"    Warning: No query samples found for task '{task_key}'")
    
    # Calculate final metrics
    shape_accuracy = total_shape_correct / total_shape_tokens if total_shape_tokens > 0 else 0.0
    grid_accuracy = total_grid_correct / total_grid_tokens if total_grid_tokens > 0 else 0.0
    exact_accuracy = total_exact_correct / total_samples if total_samples > 0 else 0.0
    
    print(f"Shape accuracy: {shape_accuracy:.4f}")
    print(f"Grid accuracy: {grid_accuracy:.4f}")
    print(f"Exact match accuracy: {exact_accuracy:.4f}")
    
    # Prepare latent data for visualization (per-sample latents directly from encoder)
    print(f"\n=== COLLECTING LATENT DATA FOR VISUALIZATION ===")
    
    # Initialize trajectory_info and transfer trajectory data from task_trajectories
    trajectory_info = []
    for task_key, trajectories in task_trajectories.items():
        for trajectory in trajectories:
            trajectory_info.append(trajectory)
    print(f"  [OK] Collected {len(trajectory_info)} trajectory samples for visualization")
    
    # Collect all support and query samples for latent space visualization
    all_support_latents = []
    all_query_latents = []
    all_support_keys = []
    all_query_keys = []
    all_support_log_vars = []  # Add logvar storage
    all_query_log_vars = []    # Add logvar storage
    
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
                all_support_log_vars.append(logvar[0].cpu().numpy())  # Store logvar
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
                all_query_log_vars.append(logvar[0].cpu().numpy())  # Store logvar
                
                # Determine task key
                if batch_keys is not None and query_key_mapping is not None:
                    key_idx = batch_keys[i].item()
                    task_key = query_key_mapping[key_idx]
                else:
                    task_key = "unknown"
                all_query_keys.append(task_key)
    
    # Also create evaluation_latent_data in the expected format for plotting with logvars
    evaluation_latent_data = {
        'support': {
            'poe': {
                'latent_zs': all_support_latents,
                'latent_log_vars': all_support_log_vars,  # Add logvars
                'keys': all_support_keys
            },
            'task_keys': list(set(all_support_keys))
        },
        'query': {
            'poe': {
                'latent_zs': all_query_latents,
                'latent_log_vars': all_query_log_vars,    # Add logvars
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
        'latent_data': evaluation_latent_data,
        'evaluation_latent_data': evaluation_latent_data,  # Add for plotting compatibility
        'trajectory_info': trajectory_info,  # Add trajectory information
        'evaluation_method': 'original_bonnet_approach'
    }
    
    return results