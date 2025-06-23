import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import numpy as np
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import set_seed, prepare_dataloader
from re_arc.main import generate_and_process_tasks
# Use local multiencoder settings manager
import Multiencoder_LPN.settings_manager as settings
from models.multi_encoder_lpn import multinomial_loss, gaussian_poe

# Get the local settings instance
# settings is already imported directly

# Maximum batch size to avoid GPU memory issues
MAX_BATCH_SIZE = 16

##############################
# Multi-Encoder Latent Optimization Functions
##############################

def multi_encoder_optimize_latent_z(multi_encoder_model, input_seq, target_seq, num_steps=None, lr=None, return_trajectory=False):
    """
    Optimize latent z for multi-encoder model using gradient ascent.
    
    Args:
        multi_encoder_model: MultiEncoderLPN model
        input_seq: Input sequence tensor
        target_seq: Target sequence tensor
        num_steps: Number of optimization steps
        lr: Learning rate
        return_trajectory: Whether to return optimization trajectory
        
    Returns:
        z: Optimized latent vector
        losses: List of losses during optimization
        trajectory: Optimization trajectory (if requested)
    """
    latent_optimization = settings.settings.get_latent_optimization()
    if num_steps is None:
        num_steps = latent_optimization['inference']['num_steps']
    if lr is None:
        lr = latent_optimization['inference']['learning_rate']
    
    num_encoders = len(multi_encoder_model.encoders)
    batch_size = input_seq.size(0)
    
    # Create identical views for all encoders (inference behavior)
    encoder_views = [(input_seq, target_seq) for _ in range(num_encoders)]
    
    # Get initial latent parameters from the multi-encoder model
    with torch.no_grad():
        (_, _), mu_star, logvar_star = multi_encoder_model(encoder_views, training=False, sample_latent=False)
        initial_z = mu_star.detach().clone()  # Use mu* as initial z for inference
    
    # Compute initial loss before optimization (step 0)
    with torch.no_grad():
        shape_logits_init, grid_logits_init = multi_encoder_model.decoder(initial_z, input_seq, target_seq=target_seq)
        shape_targets = target_seq[:, 900:902].long()
        shape_loss_init = F.cross_entropy(shape_logits_init.reshape(-1, 31), shape_targets.reshape(-1))
        
        # Compute initial grid loss
        grid_loss_list_init = []
        for i in range(batch_size):
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits_init[i, :active_pixels],
                                       target_seq[i, :active_pixels].long())
                grid_loss_list_init.append(loss_i)

        grid_loss_init = sum(grid_loss_list_init) / len(grid_loss_list_init) if grid_loss_list_init else \
                       torch.tensor(0.0, device=input_seq.device)
        initial_loss = (shape_loss_init + grid_loss_init).item()

    # Detach z from the graph and enable gradients on it
    z = initial_z.detach().requires_grad_(True)

    # Create an optimizer for z
    optimizer_z = torch.optim.Adam([z], lr=lr)

    # Track losses and z changes - start with initial loss
    losses = [initial_loss]
    z_changes = []
    
    # Track trajectory information if requested
    trajectory = {
        'z_vectors': [initial_z.detach().clone()],
        'losses': [initial_loss],
        'encoder_mu': mu_star.detach().clone(),
        'encoder_log_var': logvar_star.detach().clone(),
        'initial_z': initial_z.detach().clone()
    } if return_trajectory else None

    print(f"    Multi-encoder gradient ascent: {num_steps} steps, LR: {lr}, Batch size: {batch_size}")
    print(f"    Initial loss: {initial_loss:.4f}")

    # Create progress bar for gradient ascent steps
    pbar = tqdm(range(num_steps), desc="Multi-encoder gradient ascent", unit="step", leave=False)

    for step in pbar:
        optimizer_z.zero_grad()
        # Decode using the current z
        shape_logits, grid_logits = multi_encoder_model.decoder(z, input_seq, target_seq=target_seq)

        # Compute loss on the shape tokens
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

        # Compute grid loss
        grid_loss_list = []
        for i in range(batch_size):
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits[i, :active_pixels],
                                       target_seq[i, :active_pixels].long())
                grid_loss_list.append(loss_i)

        grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else \
                   torch.tensor(0.0, device=input_seq.device, requires_grad=True)

        reconstruction_loss = shape_loss + grid_loss
        losses.append(reconstruction_loss.item())

        # Track how much z has changed
        z_delta = torch.norm(z - initial_z).item()
        z_changes.append(z_delta)

        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{reconstruction_loss.item():.4f}',
            'Δz': f'{z_delta:.4f}',
            'shape': f'{shape_loss.item():.4f}',
            'grid': f'{grid_loss.item():.4f}'
        })

        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(z, 1.0)
        optimizer_z.step()

        # Store trajectory information after optimization step
        if return_trajectory:
            trajectory['z_vectors'].append(z.detach().clone())
            trajectory['losses'].append(reconstruction_loss.item())

    pbar.close()

    # Final change in Z
    final_z_change = torch.norm(z - initial_z).item()
    loss_improvement = losses[0] - losses[-1]
    
    print(f"    ✓ Multi-encoder optimization complete: "
          f"Loss {losses[0]:.4f} → {losses[-1]:.4f} "
          f"(Δ: {loss_improvement:+.4f}), "
          f"Z change: {final_z_change:.4f}")

    if return_trajectory:
        return z, losses, trajectory
    else:
        return z, losses


def multi_encoder_get_optimized_z(multi_encoder_model, input_seq, target_seq, num_steps=None, lr=None, context='inference', return_trajectory=False):
    """
    Get optimized z for multi-encoder model - wrapper function to handle different optimization methods.
    
    Args:
        multi_encoder_model: MultiEncoderLPN model
        input_seq: Input sequence tensor
        target_seq: Target sequence tensor
        num_steps: Number of optimization steps
        lr: Learning rate
        context: 'training' or 'inference' context
        return_trajectory: Whether to return optimization trajectory
        
    Returns:
        z: Optimized latent vector
        losses: List of losses during optimization
        trajectory: Optimization trajectory (if requested)
    """
    latent_optimization = settings.settings.get_latent_optimization()
    
    # Use context-specific parameters if not provided
    if context == 'training':
        context_settings = latent_optimization['training']
    else:
        context_settings = latent_optimization['inference']
    
    if num_steps is None:
        num_steps = context_settings['num_steps']
    if lr is None:
        lr = context_settings['learning_rate']
    
    # For now, only implement gradient-based optimization for multi-encoder
    # Could add evolutionary and voronoi methods later if needed
    method = latent_optimization.get('method', 'gradient_ascent')
    
    if method == 'gradient_ascent':
        return multi_encoder_optimize_latent_z(
            multi_encoder_model, input_seq, target_seq, 
            num_steps=num_steps, lr=lr, return_trajectory=return_trajectory
        )
    else:
        # Fallback to gradient ascent for unsupported methods
        print(f"Warning: Method '{method}' not implemented for multi-encoder, using gradient_ascent")
        return multi_encoder_optimize_latent_z(
            multi_encoder_model, input_seq, target_seq, 
            num_steps=num_steps, lr=lr, return_trajectory=return_trajectory
        )

##############################
# Multi-Encoder Data Preparation for Inference
##############################

def create_multi_encoder_inference_batches(input_sequences, output_sequences, num_encoders, batch_size):
    """
    Create batches for multi-encoder inference where all encoders get identical samples.
    For inference: K identical copies of the single (x,y₀) pair for each encoder.
    """
    total_samples = len(input_sequences)
    batches = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        
        batch_inputs = torch.tensor(input_sequences[i:end_idx]).float()
        batch_outputs = torch.tensor(output_sequences[i:end_idx]).float()
        
        # Create identical views for all encoders (same data for inference)
        encoder_views = [(batch_inputs, batch_outputs) for _ in range(num_encoders)]
        
        batches.append((encoder_views, end_idx - i))
    
    return batches

##############################
# Encode Training Sequences
##############################

def encode_training_sequences(model, run_dir, device='cuda', max_samples=500, batch_size=16):
    """
    Load training sequences from results.pkl and encode them with the trained model
    to generate latent representations for background visualization.
    
    Args:
        model: Trained multi-encoder model to use for encoding
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
        
        # Get number of encoders from model
        num_encoders = len(model.encoders)
        print(f"Using {num_encoders} encoders for inference")
        
        # Create inference batches for multi-encoder model
        inference_batches = create_multi_encoder_inference_batches(
            input_sequences, output_sequences, num_encoders, batch_size
        )
        
        # Encode sequences using the trained multi-encoder model
        model.eval()
        latent_mus = []
        latent_log_vars = []
        latent_zs = []
        initial_losses = []
        
        print(f"Encoding {len(input_sequences)} training samples using trained multi-encoder model...")
        
        with torch.no_grad():
            for encoder_views, actual_batch_size in tqdm(inference_batches, desc="Encoding training samples"):
                # Move all encoder views to device
                encoder_views_gpu = []
                for input_seq, target_seq in encoder_views:
                    encoder_views_gpu.append((input_seq.to(device), target_seq.to(device)))
                
                # Forward pass through multi-encoder model (inference mode)
                (shape_logits, grid_logits), mu, log_var = model(encoder_views_gpu, training=False, sample_latent=False)
                z = mu  # Use mean for deterministic encoding
                
                # Use the first encoder's target for loss computation (they're all identical in inference)
                target_seq = encoder_views_gpu[0][1]
                
                # Compute loss for this encoded latent (equivalent to initial trajectory loss)
                batch_losses = multinomial_loss(
                    (shape_logits, grid_logits), 
                    target_seq, 
                    beta=0.0,  # Don't include KL term for loss computation
                    mu=mu, 
                    logvar=log_var
                ).item()
                
                # Store equivalent information to trajectory
                latent_mus.append(mu.cpu().numpy())
                latent_log_vars.append(log_var.cpu().numpy())
                latent_zs.append(z.cpu().numpy())
                initial_losses.extend([batch_losses / actual_batch_size] * actual_batch_size)  # Loss per sample
        
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
                'has_initial_losses': True,
                'num_encoders': num_encoders
            }
        }
        
        return encoded_data
        
    except Exception as e:
        print(f"Error encoding training sequences: {e}")
        return None

##############################
# Run Inference
##############################

def main_test(model, keys, n_samples, n_queries, seed, device='cuda'):
    """
    Generate new data and evaluate the model on it.
    Also encodes training sequences for latent space visualization.
    
    Args:
        model: The trained model
        keys: List of problem keys
        n_samples: Number of input-output pairs to generate for support (should match eval_n_samples)
        n_queries: Number of queries to do inference (should match eval_n_queries)
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation results for each key   
    """

    set_seed(seed)
    results = {}
    
    print(f"=== EVALUATION CONFIGURATION ===")
    print(f"Keys to evaluate: {keys}")
    print(f"Support samples per key: {n_samples}")
    print(f"Query samples per key: {n_queries}")
    print(f"Maximum batch size: {MAX_BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print("=" * 50)
    
    for key_idx, key in enumerate(keys):
        results[key] = {}
        print(f"\n[KEY {key_idx+1}/{len(keys)}] Evaluating '{key}'")
        print("-" * 40)
        
        # Generate support samples (used for latent space optimization)
        print(f"Generating {n_samples} support samples...")
        _, _, _, input_samples_sequences, output_samples_sequences = generate_and_process_tasks(key, n_samples)
        
        # Use appropriate batch size for support samples - should fit in one batch if possible
        support_batch_size = min(MAX_BATCH_SIZE, n_samples)
        samples_dataloader = prepare_dataloader(input_samples_sequences, output_samples_sequences, 
                                               batch_size=support_batch_size, shuffle=False)
        
        print(f"Support samples: {len(input_samples_sequences)} generated")
        print(f"Support batch size: {support_batch_size}")
        print(f"Support batches: {len(samples_dataloader)}")

        # Generate query samples (used for evaluation)
        print(f"Generating {n_queries} query samples...")
        _, _, _, input_queries_sequences, output_queries_sequences = generate_and_process_tasks(key, n_queries)
        
        # Use reasonable batch size for queries to avoid GPU memory issues
        query_batch_size = min(MAX_BATCH_SIZE, n_queries)
        queries_dataloader = prepare_dataloader(input_queries_sequences, output_queries_sequences, 
                                               batch_size=query_batch_size, shuffle=False)
        
        print(f"Query samples: {len(input_queries_sequences)} generated")
        print(f"Query batch size: {query_batch_size}")
        print(f"Query batches: {len(queries_dataloader)}")

        # Evaluate overall performance
        metrics = evaluate_model(model, samples_dataloader, queries_dataloader, device=device)

        results[key] = metrics
        results[key]['reconstruction_results']['input_samples_sequences'] = input_samples_sequences
        results[key]['reconstruction_results']['output_samples_sequences'] = output_samples_sequences  
        results[key]['reconstruction_results']['input_queries_sequences'] = input_queries_sequences
        results[key]['reconstruction_results']['output_queries_sequences'] = output_queries_sequences
        
        # Print summary for this key
        print(f"\n[KEY {key_idx+1}/{len(keys)}] '{key}' RESULTS:")
        if 'error' not in metrics['metrics']:
            print(f"  Support loss: {metrics['metrics']['support_loss']:.4f}")
            print(f"  Query loss: {metrics['metrics']['query_loss']:.4f}")
            print(f"  Shape accuracy: {metrics['metrics']['shape_accuracy']:.4f}")
            print(f"  Grid accuracy: {metrics['metrics']['grid_accuracy']:.4f}")
            print(f"  Sample exact accuracy: {metrics['metrics']['sample_exact_accuracy']:.4f}")
            print(f"  Support reconstructions: {len(metrics['reconstruction_results']['support_reconstructions'])}")
            print(f"  Query reconstructions: {len(metrics['reconstruction_results']['query_reconstructions'])}")
            print(f"  Trajectory info samples: {len(metrics['metrics']['trajectory_info'])}")
        else:
            print(f"  ERROR: {metrics['metrics']['error']}")

    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Processed {len(keys)} keys successfully")
    return results


def evaluate_model(model, samples_dataloader, queries_dataloader, device='cuda'):
    """
    Evaluate multi-encoder model performance on dataloaders.
    
    Args:
        model: The trained multi-encoder model to evaluate
        samples_dataloader: DataLoader containing support samples for latent optimization
        queries_dataloader: DataLoader containing query samples for evaluation
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics and reconstruction results
    """
    latent_optimization = settings.settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']
    num_encoders = len(model.encoders)
    
    print(f"Multi-encoder evaluation with {num_encoders} encoders")
    print(f"Latent optimization enabled: {optimize_z_inference}")
    if optimize_z_inference:
        print(f"Optimization steps: {latent_optimization['inference']['num_steps']}")
        print(f"Optimization learning rate: {latent_optimization['inference']['learning_rate']}")

    model.eval()
    shape_correct, shape_tokens = 0, 0
    grid_correct, grid_tokens = 0, 0
    sample_exact_correct = 0
    total_samples = 0
    
    support_losses = []
    query_losses = []
    support_reconstructions = []
    query_reconstructions = []
    z_optimization_logs = []
    trajectory_info = []  # Store trajectory information for each sample

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
            # Create progress bar for gradient ascent steps for this batch
            print(f"\nOptimizing latent space for support batch {batch_idx+1} ({batch_size} samples)...")
            
            # Request trajectory information during optimization
            current_z_for_this_sample_batch, losses_opt, trajectory = multi_encoder_get_optimized_z(
                model, batch_input_s, batch_target_s, context='inference', return_trajectory=True
            )
            
            if losses_opt is not None:
                z_optimization_logs.extend(losses_opt if isinstance(losses_opt, list) else [losses_opt])
                print(f"Gradient ascent completed. Final loss: {losses_opt[-1] if isinstance(losses_opt, list) else losses_opt:.4f}")
                
                # Store trajectory information for each sample in the batch
                for i in range(batch_input_s.size(0)):
                    sample_trajectory = {
                        'input_sample': batch_input_s[i].detach().cpu().numpy(),
                        'target_sample': batch_target_s[i].detach().cpu().numpy(),
                        'z_vectors': [],
                        'losses': trajectory.get('losses', losses_opt) if trajectory else losses_opt,
                        'encoder_mu': None,
                        'encoder_log_var': None,
                        'initial_z': None
                    }
                    
                    # Process trajectory if available
                    if trajectory is not None:
                        # Store encoder information for equivalent data to training
                        if 'encoder_mu' in trajectory:
                            sample_trajectory['encoder_mu'] = trajectory['encoder_mu'][i].detach().cpu().numpy()
                        if 'encoder_log_var' in trajectory:
                            sample_trajectory['encoder_log_var'] = trajectory['encoder_log_var'][i].detach().cpu().numpy()
                        if 'initial_z' in trajectory:
                            sample_trajectory['initial_z'] = trajectory['initial_z'][i].detach().cpu().numpy()
                        
                        # Convert z vectors to numpy
                        if 'z_vectors' in trajectory:
                            for z_step in trajectory['z_vectors']:
                                sample_trajectory['z_vectors'].append(z_step[i].detach().cpu().numpy())
                            print(f"  Sample {processed_support_samples + i + 1}: {len(sample_trajectory['z_vectors'])} trajectory steps stored")
                        
                        # Use trajectory losses if available
                        if 'losses' in trajectory:
                            sample_trajectory['losses'] = trajectory['losses']
                    
                    trajectory_info.append(sample_trajectory)
            else:
                print(f"Warning: No optimization losses returned for batch {batch_idx+1}")
        else:
            with torch.no_grad():
                # Create identical encoder views for multi-encoder model (inference mode)
                encoder_views = [(batch_input_s, batch_target_s) for _ in range(num_encoders)]
                (_, _), mu, log_var = model(encoder_views, training=False, sample_latent=False)
                current_z_for_this_sample_batch = mu  # Use mean for deterministic inference
                print(f"Using encoder z (no optimization) for batch {batch_idx+1}")
        
        # Store z vectors from all support samples
        if current_z_for_this_sample_batch is not None:
            all_support_z_vectors.append(current_z_for_this_sample_batch)

        # Calculate support loss using multi-encoder model
        with torch.no_grad():
            encoder_views = [(batch_input_s, batch_target_s) for _ in range(num_encoders)]
            (shape_logits, grid_logits), mu, log_var = model(encoder_views, training=False, sample_latent=False)
            s_loss_val = multinomial_loss(
                (shape_logits, grid_logits), 
                batch_target_s, 
                beta=0.0,  # Don't include KL term for evaluation
                mu=mu, 
                logvar=log_var
            )
            support_losses.append(s_loss_val.item())

        # Store reconstructions with their corresponding inputs/outputs
        if current_z_for_this_sample_batch is not None:
            with torch.no_grad():
                # Use the decoder directly with the optimized z
                shape_logits_s, grid_logits_s = model.decoder(current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s)
                # Store each sample's reconstruction with its input and target
                for i in range(batch_input_s.size(0)):
                    reconstruction = {
                        'input': batch_input_s[i].detach().cpu().numpy().tolist(),
                        'target': batch_target_s[i].detach().cpu().numpy().tolist(),
                        'reconstruction': (shape_logits_s[i].detach().cpu().numpy().tolist(), grid_logits_s[i].detach().cpu().numpy().tolist())
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
            
            shape_logits, grid_logits = model.decoder(z_query_expanded, batch_input_q, target_seq=batch_target_q)
            
            # Compute query loss using multi-encoder model
            encoder_views = [(batch_input_q, batch_target_q) for _ in range(num_encoders)]
            (_, _), mu_q, log_var_q = model(encoder_views, training=False, sample_latent=False)
            q_loss_val = multinomial_loss(
                (shape_logits, grid_logits), 
                batch_target_q, 
                beta=0.0,  # Don't include KL term for evaluation
                mu=mu_q, 
                logvar=log_var_q
            )
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

            shape_correct += batch_shape_correct
            shape_tokens += batch_shape_tokens

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

            grid_correct += batch_grid_correct
            grid_tokens += batch_grid_tokens
            sample_exact_correct += batch_exact_correct
            total_samples += query_batch_size
            processed_query_samples += query_batch_size

            # Calculate batch accuracies for display
            batch_shape_acc = batch_shape_correct / batch_shape_tokens if batch_shape_tokens > 0 else 0
            batch_grid_acc = batch_grid_correct / batch_grid_tokens if batch_grid_tokens > 0 else 0
            batch_exact_acc = batch_exact_correct / query_batch_size

            # Update progress bar postfix
            query_pbar.set_postfix({
                'samples': f'{processed_query_samples}/{total_query_samples}',
                'loss': f'{q_loss_val.item():.4f}',
                'shape_acc': f'{batch_shape_acc:.3f}',
                'exact_acc': f'{batch_exact_acc:.3f}'
            })
    
    query_pbar.close()

    print(f"\nQuery processing complete:")
    print(f"  Total samples processed: {processed_query_samples}")
    print(f"  Total batches processed: {len(queries_dataloader)}")

    avg_support_loss = sum(support_losses) / len(support_losses) if support_losses else 0.0
    avg_query_loss = sum(query_losses) / len(query_losses) if query_losses else 0.0

    final_shape_acc = shape_correct / shape_tokens if shape_tokens > 0 else 0.0
    final_grid_acc = grid_correct / grid_tokens if grid_tokens > 0 else 0.0
    final_overall_acc = (shape_correct + grid_correct) / (shape_tokens + grid_tokens) if (shape_tokens + grid_tokens) > 0 else 0.0
    final_exact_acc = sample_exact_correct / total_samples if total_samples > 0 else 0.0

    print(f"\n>>> FINAL EVALUATION METRICS <<<")
    print(f"Support samples: {len(support_reconstructions)}")
    print(f"Query samples: {len(query_reconstructions)}")
    print(f"Support loss: {avg_support_loss:.4f}")
    print(f"Query loss: {avg_query_loss:.4f}")
    print(f"Shape accuracy: {final_shape_acc:.4f} ({shape_correct}/{shape_tokens})")
    print(f"Grid accuracy: {final_grid_acc:.4f} ({grid_correct}/{grid_tokens})")
    print(f"Overall accuracy: {final_overall_acc:.4f}")
    print(f"Sample exact accuracy: {final_exact_acc:.4f} ({sample_exact_correct}/{total_samples})")

    return {
        'metrics': {
            'support_loss': avg_support_loss,
            'query_loss': avg_query_loss,
            'shape_accuracy': final_shape_acc,
            'grid_accuracy': final_grid_acc,
            'overall_accuracy': final_overall_acc,
            'sample_exact_accuracy': final_exact_acc,
            'losses_gradient_ascent': z_optimization_logs,
            'used_latent_optimization': optimize_z_inference,
            'trajectory_info': trajectory_info
        },
        'reconstruction_results': {
            'support_reconstructions': support_reconstructions,
            'query_reconstructions': query_reconstructions
        }
    }