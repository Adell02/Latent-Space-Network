import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import numpy as np

from utils.model_utils import set_seed, prepare_dataloader, collect_latent_data
from re_arc.main import generate_and_process_tasks
from utils.settings_manager import settings
from utils.latent_functions import get_optimized_z
from models.base_model import compute_loss

# Maximum batch size to avoid GPU memory issues
MAX_BATCH_SIZE = 16

##############################
# Evaluation Latent Data Collection
##############################

def collect_evaluation_latent_data(model, samples_dataloader, queries_dataloader, device, is_multi_encoder, num_encoders):
    """
    Collect latent representations from support and query samples efficiently.
    Avoids data duplication by reusing shared input/output data.
    Always uses mean vectors (μ) for consistency.
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
    Always uses mean vectors (μ) for consistency and efficiency.
    """
    print(f"    Collecting {data_type} latents from {num_encoders}-encoder model...")
    print(f"    Using mean (μ) vectors for visualization")
    
    # Get the data samples once (shared across all encoders)
    input_samples = []
    output_samples = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_input, batch_target in dataloader:
            if sample_count >= max_samples:
                break
                
            batch_size = min(batch_input.size(0), max_samples - sample_count)
            input_samples.append(batch_input[:batch_size])
            output_samples.append(batch_target[:batch_size])
            sample_count += batch_size
    
    if not input_samples:
        print(f"      ⚠ No {data_type} samples found")
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
        'latent_type': 'mean'
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
                # Always use mean vectors (μ) for consistency
                z = mu
                
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
            mu, log_var = model.encoder(batch_inputs, batch_outputs)
            return mu, log_var
        
        final_mus, final_log_vars, final_zs = process_in_batches(
            all_inputs, all_outputs, single_encoder_processing, "single encoder"
        )
        
        encoder_latent_data['encoder_0'] = {
            'latent_mus': final_mus,
            'latent_log_vars': final_log_vars,
            'latent_zs': final_zs,  # This will be μ or z based on use_mean_for_viz
            'data_type': f"{data_type}_encoder_0",
            'encoder_idx': 0,
            **shared_data
        }
        print(f"          ✓ Collected {len(final_zs)} samples from Encoder 0 (μ)")
    
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
            print(f"          ✓ Collected {len(final_zs)} samples from Encoder {encoder_idx} (μ)")

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
                print(f"          ✓ Collected {len(final_zs)} samples from PoE (μ)")
    
    print(f"      ✓ {num_encoders}-encoder {data_type} collection complete")
    return encoder_latent_data

##############################
# Training Latent Data Collection
##############################

def collect_training_latent_representations(model, run_dir, device='cuda'):
    """
    Collect training latent representations efficiently without data duplication.
    Always uses mean vectors (μ) for consistency.
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
            print(f"  ✓ Successfully collected training latent representations")
        
        return training_latent_data
        
    except Exception as e:
        print(f"  Error collecting training latent representations: {e}")
        return None

def collect_unified_training_latents(model, input_sequences, output_sequences, device, is_multi_encoder, num_encoders, batch_size=16):
    """
    Unified training latent collection using batch-split approach.
    Processes the training data efficiently with memory-safe batching.
    Always uses mean vectors (μ) for consistency.
    """
    print(f"    Collecting training latents from {num_encoders}-encoder model...")
    print(f"    Using mean (μ) vectors for visualization")
    
    # Create dataloader for the shared training data
    dataloader = prepare_dataloader(input_sequences, output_sequences, batch_size, shuffle=False)
    
    # Convert dataloader to tensors for batch processing
    all_inputs = []
    all_outputs = []
    
    for batch_input, batch_output in dataloader:
        all_inputs.append(batch_input)
        all_outputs.append(batch_output)
    
    if not all_inputs:
        print(f"      ⚠ No training data found")
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
        'latent_type': 'mean'
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
                # Always use mean vectors (μ) for consistency
                z = mu
                
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
            'latent_zs': final_zs,  # Always μ (mean) vectors
            'data_type': func_name,
            **shared_data
        }
    
    # For single encoder: only collect once as encoder_0
    if not is_multi_encoder or num_encoders == 1:
        print(f"      Single encoder processing...")
        
        def single_encoder_processing(batch_inputs, batch_outputs):
            mu, log_var = model.encoder(batch_inputs, batch_outputs)
            return mu, log_var
        
        latent_data = process_training_in_batches(
            combined_inputs, combined_outputs, single_encoder_processing, "training_encoder_0"
        )
        latent_data['encoder_idx'] = 0
        encoder_latent_data['encoder_0'] = latent_data
        print(f"        ✓ Collected {latent_data['num_samples']} samples from Encoder 0 (μ)")
    
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
            print(f"        ✓ Collected {latent_data['num_samples']} samples from Encoder {encoder_idx} (μ)")
    
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
                    z = mu  # Use mean for consistency
                else:
                    # Single encoder
                    mu, log_var = model.encoder(batch_input, batch_output)
                    z = mu  # Use mean for consistency
                
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

def main_test(model, keys, run_dir, n_samples, n_queries, seed, device='cuda'):
    """
    Generate new data and evaluate the model on it with key-specific evaluation.
    Each key is evaluated separately with its own support->optimization->queries cycle.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: Number of input-output pairs to generate for support (should match eval_n_samples)
        n_queries: Number of queries to do inference (should match eval_n_queries)
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation results structured by key   
    """

    set_seed(seed)
    results = {
        'evaluation_metadata': {
            'keys': keys,
            'n_samples_per_key': n_samples,
            'n_queries_per_key': n_queries,
            'max_batch_size': MAX_BATCH_SIZE,
            'device': str(device),
            'seed': seed,
            'evaluation_strategy': 'key_specific_separate_evaluation',
            'latent_type': 'mean'
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
    print(f"Latent representation: Mean (μ)")
    print("=" * 50)
    
    # Collect training latent representations once at the beginning
    print(f"\n>>> COLLECTING TRAINING LATENT REPRESENTATIONS <<<")
    training_latent_data = collect_training_latent_representations(model, run_dir, device)
    if training_latent_data:
        print(f"✓ Collected training latent data: {training_latent_data.get('collection_info', {})}")
        results['training_latent_data'] = training_latent_data
    else:
        print("⚠ Warning: Could not collect training latent representations")
    
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
            print(f"  Generating {n_samples} support samples for key '{key}'...")
            _, _, _, input_samples_sequences, output_samples_sequences = generate_and_process_tasks(key, n_samples)
            
            # Generate key-specific query samples  
            print(f"  Generating {n_queries} query samples for key '{key}'...")
            _, _, _, input_queries_sequences, output_queries_sequences = generate_and_process_tasks(key, n_queries)
            
            if not input_samples_sequences or not input_queries_sequences:
                print(f"  ❌ Failed to generate data for key '{key}' - skipping")
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
            
            samples_dataloader = prepare_dataloader(input_samples_sequences, output_samples_sequences, 
                                                   batch_size=support_batch_size, shuffle=False)
            queries_dataloader = prepare_dataloader(input_queries_sequences, output_queries_sequences, 
                                                   batch_size=query_batch_size, shuffle=False)
            
            print(f"  Support: {len(input_samples_sequences)} samples, batch size: {support_batch_size}")
            print(f"  Query: {len(input_queries_sequences)} samples, batch size: {query_batch_size}")
            
            # Evaluate model on this specific key
            print(f"  Running evaluation for key '{key}'...")
            key_metrics = evaluate_model(model, samples_dataloader, queries_dataloader, device=device)
            
            # Store key-specific results with structured information
            results['key_results'][key] = {
                'key': key,
                'support_samples': len(input_samples_sequences),
                'query_samples': len(input_queries_sequences),
                'metrics': key_metrics['metrics'],
                'reconstruction_results': key_metrics['reconstruction_results'],
                'evaluation_latent_data': key_metrics.get('evaluation_latent_data', {}),
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
            if 'error' not in key_metrics['metrics']:
                key_summary = {
                    'support_loss': key_metrics['metrics']['support_loss'],
                    'query_loss': key_metrics['metrics']['query_loss'],
                    'shape_accuracy': key_metrics['metrics']['shape_accuracy'],
                    'grid_accuracy': key_metrics['metrics']['grid_accuracy'],
                    'sample_exact_accuracy': key_metrics['metrics']['sample_exact_accuracy'],
                    'trajectory_samples': len(key_metrics['metrics']['trajectory_info'])
                }
                aggregated_metrics['per_key_summary'][key] = key_summary
                
                # Print key summary
                print(f"  ✅ Key '{key}' Results:")
                print(f"    Support loss: {key_summary['support_loss']:.4f}")
                print(f"    Query loss: {key_summary['query_loss']:.4f}")
                print(f"    Shape accuracy: {key_summary['shape_accuracy']:.4f}")
                print(f"    Grid accuracy: {key_summary['grid_accuracy']:.4f}")
                print(f"    Sample exact accuracy: {key_summary['sample_exact_accuracy']:.4f}")
                print(f"    Trajectory samples: {key_summary['trajectory_samples']}")
            else:
                print(f"  ❌ Key '{key}' Error: {key_metrics['metrics']['error']}")
                aggregated_metrics['failed_evaluations'] += 1
                
        except Exception as e:
            print(f"  ❌ Exception during evaluation of key '{key}': {e}")
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
    
    return results


def evaluate_model(model, samples_dataloader, queries_dataloader, device='cuda'):
    """
    Evaluate model performance on dataloaders.
    For multi-encoder models: each encoder processes the same sample, then PoE combines them.
    
    Args:
        model: The trained model to evaluate (single or multi-encoder)
        samples_dataloader: DataLoader containing support samples for latent optimization
        queries_dataloader: DataLoader containing query samples for evaluation
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics and reconstruction results
    """
    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']
    
    # Check if this is a multi-encoder model
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    num_encoders = getattr(model, 'num_encoders', 1) if is_multi_encoder else 1
    
    print(f"=== EVALUATION SETUP ===")
    print(f"Model type: {'Multi-encoder' if is_multi_encoder else 'Single encoder'}")
    if is_multi_encoder:
        print(f"Number of encoders: {num_encoders}")
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
    total_support_samples = sum(batch_input.size(0) for batch_input, _ in samples_dataloader)
    processed_support_samples = 0
    
    # Create progress bar for support batches
    support_pbar = tqdm(samples_dataloader, desc="Support batches", unit="batch")
    
    for batch_idx, (batch_input_s, batch_target_s) in enumerate(support_pbar):
        batch_input_s = batch_input_s.to(device)
        batch_target_s = batch_target_s.to(device)
        batch_size = batch_input_s.size(0)
        
        # Update progress bar description
        support_pbar.set_description(f"Support batch {batch_idx+1}/{len(samples_dataloader)} (size: {batch_size})")
        
        current_z_for_this_sample_batch = None
        
        if optimize_z_inference:
            print(f"  Optimizing latent z for support batch {batch_idx+1}...")
            # Latent optimization - works for both single and multi-encoder
            z_optimized, losses, trajectory = get_optimized_z(
                model, batch_input_s, batch_target_s, 
                num_steps=latent_optimization['inference']['num_steps'],
                lr=latent_optimization['inference']['learning_rate'],
                context='inference',
                return_trajectory=True
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
                        'is_multi_encoder': True,  # Always True for unified processing
                        'num_encoders': num_encoders if is_multi_encoder else 1,
                        'is_actually_single_encoder': not is_multi_encoder,
                        'batch_idx': batch_idx,
                        'sample_idx_in_batch': i
                    }
                    
                    # Always create individual encoder trajectories for unified processing
                    sample_trajectory['individual_encoder_trajectories'] = {}
                    
                    if is_multi_encoder:
                        # Get individual encoder z vectors for this sample during optimization
                        with torch.no_grad():
                            for enc_idx in range(num_encoders):
                                enc_mu, enc_log_var = model.multi_encoder.encoders[enc_idx](
                                    batch_input_s[i:i+1], batch_target_s[i:i+1]
                                )
                                enc_z = enc_mu  # Use mean for consistency
                                
                                sample_trajectory['individual_encoder_trajectories'][f'encoder_{enc_idx}'] = {
                                    'mu': enc_mu.cpu().numpy(),
                                    'log_var': enc_log_var.cpu().numpy(),
                                    'z': enc_z.cpu().numpy()
                                }
                        
                        # Store individual encoder reconstructions for comparison
                        sample_trajectory['individual_encoder_reconstructions'] = {}
                        with torch.no_grad():
                            for enc_idx in range(num_encoders):
                                enc_data = sample_trajectory['individual_encoder_trajectories'][f'encoder_{enc_idx}']
                                enc_z_tensor = torch.tensor(enc_data['z']).to(device)
                                
                                try:
                                    # Generate reconstruction using individual encoder z
                                    enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                                        enc_z_tensor, batch_input_s[i:i+1], target_seq=batch_target_s[i:i+1]
                                    )
                                    
                                    sample_trajectory['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = {
                                        'shape_logits': enc_shape_logits.cpu().numpy(),
                                        'grid_logits': enc_grid_logits.cpu().numpy()
                                    }
                                except Exception as e:
                                    print(f"    Warning: Could not generate reconstruction for encoder {enc_idx}: {e}")
                                    sample_trajectory['individual_encoder_reconstructions'][f'encoder_{enc_idx}'] = None
                    else:
                        # Single encoder: create encoder_0 entry for unified processing
                        # IMPORTANT: For single encoder, individual encoder should match PoE start
                        if 'z_vectors' in trajectory and len(trajectory['z_vectors']) > 0:
                            # Use the exact same initial z that started the PoE optimization
                            initial_poe_z = trajectory['z_vectors'][0][i].detach().cpu().numpy()
                            
                            # Get encoder mu/log_var for metadata (but use PoE's z)
                            with torch.no_grad():
                                enc_mu, enc_log_var = model.encoder(batch_input_s[i:i+1], batch_target_s[i:i+1])
                            
                            sample_trajectory['individual_encoder_trajectories']['encoder_0'] = {
                                'mu': enc_mu.cpu().numpy(),
                                'log_var': enc_log_var.cpu().numpy(), 
                                'z': initial_poe_z  # Use SAME z as PoE trajectory start
                            }
                        else:
                            # Fallback: independent z (should not happen normally)
                            with torch.no_grad():
                                enc_mu, enc_log_var = model.encoder(batch_input_s[i:i+1], batch_target_s[i:i+1])
                                enc_z = enc_mu  # Use mean for consistency
                                
                                sample_trajectory['individual_encoder_trajectories']['encoder_0'] = {
                                    'mu': enc_mu.cpu().numpy(),
                                    'log_var': enc_log_var.cpu().numpy(),
                                    'z': enc_z.cpu().numpy()
                                }
                    
                    # Convert PoE z vectors to numpy (main trajectory)
                    if 'z_vectors' in trajectory:
                        for z_step in trajectory['z_vectors']:
                            sample_trajectory['z_vectors'].append(z_step[i].detach().cpu().numpy())
                        print(f"  Sample {processed_support_samples + i + 1}: {len(sample_trajectory['z_vectors'])} trajectory steps stored")
                        
                        # Store trajectory reconstructions at key points (PoE for multi-encoder, main trajectory for single-encoder)
                        if len(sample_trajectory['z_vectors']) > 1:
                            sample_trajectory['poe_trajectory_reconstructions'] = {}
                            key_indices = [0, len(sample_trajectory['z_vectors'])//2, len(sample_trajectory['z_vectors'])-1]
                            key_labels = ['initial', 'middle', 'final']
                            
                            with torch.no_grad():
                                for idx, label in zip(key_indices, key_labels):
                                    if idx < len(sample_trajectory['z_vectors']):
                                        try:
                                            traj_z = torch.tensor(sample_trajectory['z_vectors'][idx]).unsqueeze(0).to(device)
                                            
                                            if is_multi_encoder:
                                                # Multi-encoder: use PoE decoder
                                                traj_shape_logits, traj_grid_logits = model.multi_encoder.decoder(
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
                                        except Exception as e:
                                            reconstruction_type = "PoE" if is_multi_encoder else "trajectory"
                                            print(f"    Warning: Could not generate {reconstruction_type} reconstruction at step {idx}: {e}")
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
            else:
                print(f"Warning: No optimization losses returned for batch {batch_idx+1}")
        else:
            # Use encoder inference - handle both single and multi-encoder
            with torch.no_grad():
                if is_multi_encoder:
                    # For multi-encoder, use PoE inference (each encoder sees the same input)
                    mu, log_var = model.multi_encoder(
                        [(batch_input_s, batch_target_s) for _ in range(num_encoders)],
                        training=False, sample_latent=False, use_poe=True
                    )[1:3]  # Get mu, log_var from PoE
                    current_z_for_this_sample_batch = mu  # Use mean for deterministic inference
                else:
                    # Single encoder
                    mu, log_var = model.encoder(batch_input_s, batch_target_s)
                    current_z_for_this_sample_batch = mu  # Use mean for consistency
                print(f"Using encoder z (no optimization) for batch {batch_idx+1}")
        
        # Store z vectors from all support samples
        if current_z_for_this_sample_batch is not None:
            all_support_z_vectors.append(current_z_for_this_sample_batch)

        # Calculate support loss
        s_loss_val = compute_loss(model, batch_input_s, batch_target_s)
        support_losses.append(s_loss_val.item())

        # Store reconstructions - handle both single and multi-encoder
        if current_z_for_this_sample_batch is not None:
            with torch.no_grad():
                if is_multi_encoder:
                    # PoE reconstruction using the combined latent
                    shape_logits_s, grid_logits_s = model.multi_encoder.decoder(
                        current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s
                    )
                    
                    # Store PoE reconstruction
                    for i in range(batch_input_s.size(0)):
                        poe_reconstruction = {
                            'input': batch_input_s[i].detach().cpu().numpy().tolist(),
                            'target': batch_target_s[i].detach().cpu().numpy().tolist(),
                            'reconstruction': (shape_logits_s[i].detach().cpu().numpy().tolist(), 
                                             grid_logits_s[i].detach().cpu().numpy().tolist())
                        }
                        poe_support_reconstructions.append(poe_reconstruction)
                    
                    # Also store individual encoder reconstructions for analysis
                    for enc_idx in range(num_encoders):
                        enc_mu, enc_log_var = model.multi_encoder.encoders[enc_idx](batch_input_s, batch_target_s)
                        enc_z = enc_mu  # Use mean for deterministic inference
                        enc_shape_logits, enc_grid_logits = model.multi_encoder.decoder(
                            enc_z, batch_input_s, target_seq=batch_target_s
                        )
                        
                        for i in range(batch_input_s.size(0)):
                            enc_reconstruction = {
                                'input': batch_input_s[i].detach().cpu().numpy().tolist(),
                                'target': batch_target_s[i].detach().cpu().numpy().tolist(),
                                'reconstruction': (enc_shape_logits[i].detach().cpu().numpy().tolist(),
                                                 enc_grid_logits[i].detach().cpu().numpy().tolist())
                            }
                            individual_support_reconstructions[f'encoder_{enc_idx}'].append(enc_reconstruction)
                
                else:
                    # Single encoder reconstruction
                    shape_logits_s, grid_logits_s = model.decoder(
                        current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s
                    )
                    
                    for i in range(batch_input_s.size(0)):
                        reconstruction = {
                            'input': batch_input_s[i].detach().cpu().numpy().tolist(),
                            'target': batch_target_s[i].detach().cpu().numpy().tolist(),
                            'reconstruction': (shape_logits_s[i].detach().cpu().numpy().tolist(), 
                                             grid_logits_s[i].detach().cpu().numpy().tolist())
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
        # Concatenate all support z vectors and take mean
        combined_support_z = torch.cat(all_support_z_vectors, dim=0)
        z_for_queries_prototype = combined_support_z.mean(dim=0, keepdim=True)
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
    total_query_samples = sum(batch_input.size(0) for batch_input, _ in queries_dataloader)
    processed_query_samples = 0
    
    # Create progress bar for query batches
    query_pbar = tqdm(queries_dataloader, desc="Query batches", unit="batch")
    
    for batch_idx, (batch_input_q, batch_target_q) in enumerate(query_pbar):
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

                # 2. Individual encoder evaluations 
                individual_batch_accuracies = {}  # Track batch performance for comparison
                
                for enc_idx in range(num_encoders):
                    # Get individual encoder output
                    enc_mu, enc_log_var = model.multi_encoder.encoders[enc_idx](batch_input_q, batch_target_q)
                    enc_z = enc_mu  # Use mean for deterministic inference
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
        evaluation_latent_data = collect_evaluation_latent_data(
            model, samples_dataloader, queries_dataloader, device, is_multi_encoder, num_encoders
        )
        print(f"✓ Collected evaluation latent data with {len(evaluation_latent_data)} data types")

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
                }
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
        evaluation_latent_data = collect_evaluation_latent_data(
            model, samples_dataloader, queries_dataloader, device, is_multi_encoder, num_encoders
        )
        print(f"✓ Collected evaluation latent data with {len(evaluation_latent_data)} data types")

        # Convert single-encoder results to unified multi-encoder format for consistent processing
        individual_accuracies = {
            'encoder_0': {
                'shape_accuracy': final_shape_acc,
                'grid_accuracy': final_grid_acc,
                'overall_accuracy': final_overall_acc,
                'sample_exact_accuracy': final_exact_acc
            }
        }
        
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