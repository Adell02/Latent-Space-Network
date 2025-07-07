import torch
from torch.optim import Adam
import torch.nn.functional as F
import json
import numpy as np
import os

from models.base_model import LatentProgramNetwork, compute_loss, gaussian_poe
from utils.settings_manager import settings
from re_arc.main import generate_and_process_tasks
from utils.latent_functions import get_optimized_z
from utils.data_preparation import split_dataset_by_keys_for_multi_encoder

from utils.model_utils import (
    set_seed,
    create_run_directory,
    setup_logging,
    prepare_dataloader,
    save_checkpoint,
    save_results,
    count_model_parameters,
    save_model_params,
    collect_latent_data,
)

from utils.wandb_logger import init_wandb_for_mode, get_wandb_logger
from utils.evaluation_utils import run_quick_evaluation, should_run_evaluation, log_evaluation_to_wandb

##############################
# Latent Data Collection Functions
##############################

def collect_multi_encoder_latent_data(model, encoder_dataloaders, device, max_samples_per_encoder=50):
    """Collect latent representations from each encoder for visualization."""    
    encoder_latent_data = {}
    
    for encoder_idx, dataloader in enumerate(encoder_dataloaders):
        print(f"  Collecting from Encoder {encoder_idx}...")

        latent_data = collect_latent_data(
            model,
            dataloader,
            device,
            encoder_idx=encoder_idx,
            max_samples=max_samples_per_encoder,
            data_type=f"training_encoder_{encoder_idx}",
        )

        if latent_data.get('num_samples', 0) > 0:
            print(
                f"    ✓ Collected {latent_data['num_samples']} samples from Encoder {encoder_idx}"
            )

        encoder_latent_data[f"encoder_{encoder_idx}"] = latent_data
    
    return encoder_latent_data

def collect_single_encoder_latent_data(model, dataloader, device, max_samples=100, data_type='training'):
    """
    Collect latent representations from single encoder for visualization.
    
    Args:
        model: Single encoder model
        dataloader: Data loader
        device: Device to run on
        max_samples: Maximum samples to collect
        data_type: Type of data being collected
        
    Returns:
        dict: Latent data
    """
    model.eval()
    latent_data = {
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_samples': [],
        'output_samples': [],
        'data_type': data_type,
        'num_samples': 0
    }
    
    with torch.no_grad():
        sample_count = 0
        for batch_input, batch_target in dataloader:
            if sample_count >= max_samples:
                break
                
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            # Get latent representations - always use mean vectors
            mu, log_var = model.encoder(batch_input, batch_target)
            z = mu  # Use mean for consistency
            
            # Store data
            batch_size = min(batch_input.size(0), max_samples - sample_count)
            latent_data['latent_mus'].append(mu[:batch_size].cpu().numpy())
            latent_data['latent_log_vars'].append(log_var[:batch_size].cpu().numpy())
            latent_data['latent_zs'].append(z[:batch_size].cpu().numpy())
            latent_data['input_samples'].append(batch_input[:batch_size].cpu().numpy())
            latent_data['output_samples'].append(batch_target[:batch_size].cpu().numpy())
            
            sample_count += batch_size
    
    # Concatenate all batches
    if latent_data['latent_mus']:
        latent_data['latent_mus'] = np.concatenate(latent_data['latent_mus'], axis=0)
        latent_data['latent_log_vars'] = np.concatenate(latent_data['latent_log_vars'], axis=0)
        latent_data['latent_zs'] = np.concatenate(latent_data['latent_zs'], axis=0)
        latent_data['input_samples'] = np.concatenate(latent_data['input_samples'], axis=0)
        latent_data['output_samples'] = np.concatenate(latent_data['output_samples'], axis=0)
        latent_data['num_samples'] = len(latent_data['latent_mus'])
    
    return latent_data

##############################
# Main Training Function
##############################

def evaluate_accuracy(model, dataloader, device, is_multi_encoder=False, encoder_idx=None, optimize_z=False, logger=None):
    """
    Evaluate model accuracy on a given dataloader.
    
    Args:
        model: The trained model
        dataloader: DataLoader to evaluate on
        device: Device to run evaluation on
        is_multi_encoder: Whether this is a multi-encoder model
        encoder_idx: Which encoder to use (None for PoE inference)
        optimize_z: Whether to use latent optimization
        logger: Logger instance
    
    Returns:
        dict: Dictionary with accuracy metrics
    """
    model.eval()
    
    epoch_shape_correct = 0
    epoch_shape_tokens = 0
    epoch_grid_correct = 0
    epoch_grid_tokens = 0
    sample_exact_correct = 0
    total_samples_eval = 0
    
    evaluation_name = f"Encoder {encoder_idx}" if encoder_idx is not None else "PoE" if is_multi_encoder else "Model"
    
    with torch.no_grad():
        for batch_input_eval, batch_target_eval in dataloader:
            total_samples_eval += batch_input_eval.size(0)
            batch_input_eval = batch_input_eval.to(device)
            batch_target_eval = batch_target_eval.to(device)

            # Get latent representation
            if optimize_z:
                # Use latent optimization
                z_eval, _ = get_optimized_z(model, batch_input_eval, batch_target_eval, context='training')
            else:
                if is_multi_encoder:
                    if encoder_idx is not None:
                        # Individual encoder evaluation
                        mu_eval, log_var_eval = model(batch_input_eval, batch_target_eval, encoder_idx=encoder_idx)[1:3]
                    else:
                        # PoE inference
                        mu_eval, log_var_eval = model(batch_input_eval, batch_target_eval)[1:3]
                    z_eval = mu_eval  # Use mean for consistency
                else:
                    # Single encoder
                    mu_eval, log_var_eval = model.encoder(batch_input_eval, batch_target_eval)
                    z_eval = mu_eval  # Use mean for consistency

            # Decode
            shape_logits_eval, grid_logits_eval = model.decoder(z_eval, batch_input_eval, target_seq=batch_target_eval)
            
            shape_pred_eval = shape_logits_eval.argmax(dim=-1)
            grid_pred_eval = grid_logits_eval.argmax(dim=-1)
            shape_tgt_eval = batch_target_eval[:, 900:902].long()
            grid_tgt_eval = batch_target_eval[:, :900].long()

            epoch_shape_correct += (shape_pred_eval == shape_tgt_eval).sum().item()
            epoch_shape_tokens += shape_tgt_eval.numel()
            
            for i in range(batch_input_eval.size(0)):
                tgt_rows_eval = int(batch_target_eval[i, 900].item())
                tgt_cols_eval = int(batch_target_eval[i, 901].item())
                active_pixels_eval = tgt_rows_eval * tgt_cols_eval
                if active_pixels_eval > 0:
                    epoch_grid_correct += (grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]).sum().item()
                    epoch_grid_tokens += active_pixels_eval
                    if torch.all(shape_pred_eval[i] == shape_tgt_eval[i]) and \
                       torch.all(grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]):
                        sample_exact_correct += 1
                elif torch.all(shape_pred_eval[i] == shape_tgt_eval[i]):
                    sample_exact_correct += 1

    # Calculate accuracies
    shape_accuracy = epoch_shape_correct / epoch_shape_tokens if epoch_shape_tokens > 0 else 0.0
    grid_accuracy = epoch_grid_correct / epoch_grid_tokens if epoch_grid_tokens > 0 else 0.0
    overall_accuracy = (epoch_shape_correct + epoch_grid_correct) / (epoch_shape_tokens + epoch_grid_tokens) if (epoch_shape_tokens + epoch_grid_tokens) > 0 else 0.0
    sample_exact_accuracy = sample_exact_correct / total_samples_eval if total_samples_eval > 0 else 0.0
    
    accuracy_metrics = {
        'shape_accuracy': shape_accuracy,
        'grid_accuracy': grid_accuracy,
        'overall_accuracy': overall_accuracy,
        'sample_exact_accuracy': sample_exact_accuracy,
        'evaluation_name': evaluation_name
    }
    
    if logger:
        logger.info(f"{evaluation_name} Accuracy -- Shape: {shape_accuracy:.4f}, Grid: {grid_accuracy:.4f}, Overall: {overall_accuracy:.4f}, Sample Exact: {sample_exact_accuracy:.4f}")
    
    return accuracy_metrics

def train_model(model, dataloader, optimizer, run_dir, logger, scaler, use_mixed_precision, gradient_accumulation_steps, current_epoch_num, total_epochs, encoder_idx=None, joint_training=False):
    model.train()
    epoch_total_loss = 0
    epoch_shape_loss_sum = 0
    epoch_grid_loss_sum = 0
    epoch_kl_loss_sum = 0
    epoch_repulsion_loss_sum = 0  # New: track repulsion loss
    
    # Track the repulsion lambda used in this epoch (joint training)
    current_lambda_rep = 0.0  # default in case repulsion not used

    optimizer.zero_grad() # Ensure gradients are zeroed at the start of accumulation cycle / epoch

    logger.info("-" * 60)
    if joint_training:
        logger.info(f"Starting joint training batch loop with repulsion loss - Epoch {current_epoch_num}/{total_epochs}...")
    elif encoder_idx is not None:
        logger.info(f"Starting training batch loop for Encoder {encoder_idx} - Epoch {current_epoch_num}/{total_epochs}...")
    else:
        logger.info(f"Starting training batch loop for Epoch {current_epoch_num}/{total_epochs}...")
    total_batches = len(dataloader)

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        device = next(model.parameters()).device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            # Joint training mode with repulsion loss
            if joint_training:
                # Get repulsion loss settings and optional schedule
                rep_cfg = settings.get_repulsion_loss_settings()
                base_lambda = rep_cfg.get('lambda', 0.1)
                schedule_cfg = rep_cfg.get('schedule', None)
                λ_rep = 0.0
                if schedule_cfg:
                    sched_type = schedule_cfg.get('type', 'linear').lower()
                    warmup_epochs = schedule_cfg.get('warmup_epochs', 0)
                    if current_epoch_num <= warmup_epochs:
                        λ_rep = 0.0  # No repulsion during warm-up
                    else:
                        epoch_idx = current_epoch_num - warmup_epochs  # 1-indexed inside schedule window
                        effective_total = max(total_epochs - warmup_epochs, 1)
                        if sched_type == 'linear':
                            lam_start = schedule_cfg.get('start', 0.0)
                            lam_end = schedule_cfg.get('end', base_lambda)
                            denom = max(effective_total - 1, 1)
                            progress = (epoch_idx - 1) / denom
                            λ_rep = lam_start + progress * (lam_end - lam_start)
                        elif sched_type == 'exponential':
                            lam_start = schedule_cfg.get('start', 0.01)
                            rate = schedule_cfg.get('rate', 1.05)
                            λ_rep = lam_start * (rate ** (epoch_idx - 1))
                        else:
                            λ_rep = base_lambda  # unknown schedule
                else:
                    λ_rep = base_lambda
                
                # Collect latent distributions from all encoders
                K = model.num_encoders
                mus, logvars = [], []
                for enc_idx in range(K):
                    mu, logvar = model.multi_encoder.encoders[enc_idx](input_seq, target_seq)
                    mus.append(mu)
                    logvars.append(logvar)
                
                # Compute PoE fusion
                mu_stack = torch.stack(mus)                 # [K, B, D]
                logvar_stack = torch.stack(logvars)         # [K, B, D]
                mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
                
                # 1) Stabilise variance (exp(logvar) stays in a safe range)
                logvar_star = logvar_star.clamp(min=-8.0, max=4.0)

                # 2) Sample latent with re-parameterisation so σ² receives gradient
                eps = torch.randn_like(mu_star)
                z = mu_star + eps * torch.exp(0.5 * logvar_star)
                
                # Decode
                shape_logits, grid_logits = model.multi_encoder.decoder(z, input_seq, target_seq=target_seq)
                
                # Compute reconstruction loss
                shape_targets = target_seq[:, 900:902].long()
                shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))
                
                batch_size = target_seq.size(0)
                grid_loss_sum = torch.tensor(0.0, device=target_seq.device)
                active = 0
                for i in range(batch_size):
                    r, c = map(int, target_seq[i, 900:902])
                    n_pix = r * c
                    if n_pix > 0:
                        grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
                        active += n_pix
                grid_loss = grid_loss_sum / active if active > 0 else torch.tensor(0.0, device=target_seq.device)
                rec_loss = shape_loss + grid_loss
                
                # KL to prior for every encoder
                kl_prior = 0.5 * torch.stack([
                    (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)
                    for mu, logvar in zip(mus, logvars)
                ]).mean()
                
                # Pairwise repulsive KL (closed form for Gaussians)
                repulsion_loss = torch.tensor(0.0, device=target_seq.device)
                if K > 1:
                    kl_matrix = []
                    z_dim = mus[0].size(1)  # latent dimension
                    for j in range(K):
                        for k in range(j+1, K):
                            mu_j, lv_j = mus[j], logvars[j]
                            mu_k, lv_k = mus[k], logvars[k]
                            var_j, var_k = lv_j.exp(), lv_k.exp()
                            
                            kl_jk = 0.5 * (
                                (var_j / var_k).sum(1)
                              + ((mu_k - mu_j).pow(2) / var_k).sum(1)
                              - z_dim
                              + (lv_k - lv_j).sum(1)
                            )
                            kl_matrix.append(kl_jk)
                    
                    repulsion_loss = torch.stack(kl_matrix).mean() if kl_matrix else torch.tensor(0.0, device=input_seq.device)
                
                # Total loss with repulsion term
                loss = rec_loss + BETA * kl_prior - λ_rep * repulsion_loss
                loss = loss / gradient_accumulation_steps
                
                # Store components for logging
                shape_loss_comp = shape_loss
                grid_loss_comp = grid_loss
                kl_loss_comp = kl_prior
                repulsion_comp = repulsion_loss
                current_lambda_rep = λ_rep
            else:
                # Original training modes (single encoder or individual encoder training)
                if encoder_idx is not None:
                    # Multi-encoder: train specific encoder
                    loss, shape_loss_comp, grid_loss_comp, kl_loss_comp = compute_loss(
                        model, input_seq, target_seq, beta=BETA, return_components=True, encoder_idx=encoder_idx
                    )
                else:
                    # Single encoder or inference mode
                    loss, shape_loss_comp, grid_loss_comp, kl_loss_comp = compute_loss(
                        model, input_seq, target_seq, beta=BETA, return_components=True
                    )
                loss = loss / gradient_accumulation_steps
                repulsion_comp = torch.tensor(0.0, device=input_seq.device)  # No repulsion in original modes
        
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_total_loss += loss.item() * gradient_accumulation_steps # Unscale for logging
        epoch_shape_loss_sum += shape_loss_comp.item()
        epoch_grid_loss_sum += grid_loss_comp.item()
        epoch_kl_loss_sum += kl_loss_comp.item()
        epoch_repulsion_loss_sum += repulsion_comp.item()
        
        progress = (batch_idx + 1) / total_batches * 100
        # Log less frequently if accumulating gradients
        log_frequency = gradient_accumulation_steps * 5 
        if (batch_idx + 1) % log_frequency == 0 or (batch_idx + 1) == total_batches:
            # Log individual unscaled losses for the current batch/step
            if joint_training:
                logger.info(f"Joint Training - Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
                logger.info(
                    f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f} "
                    f"(Shape: {shape_loss_comp.item():.4f}, Grid: {grid_loss_comp.item():.4f}, "
                    f"KL: {kl_loss_comp.item():.4f}, Repulsion: {repulsion_comp.item():.4f}, λ: {current_lambda_rep:.4f})"
                )
            elif encoder_idx is not None:
                logger.info(f"Encoder {encoder_idx} - Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
                logger.info(f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f} (Shape: {shape_loss_comp.item():.4f}, Grid: {grid_loss_comp.item():.4f}, KL: {kl_loss_comp.item():.4f})")
            else:
                logger.info(f"Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
                logger.info(f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f} (Shape: {shape_loss_comp.item():.4f}, Grid: {grid_loss_comp.item():.4f}, KL: {kl_loss_comp.item():.4f})")

    avg_loss_for_epoch = epoch_total_loss / total_batches
    avg_shape_loss = epoch_shape_loss_sum / total_batches
    avg_grid_loss = epoch_grid_loss_sum / total_batches
    avg_kl_loss = epoch_kl_loss_sum / total_batches
    avg_repulsion_loss = epoch_repulsion_loss_sum / total_batches
    
    logger.info("=" * 60)
    if joint_training:
        logger.info(f"Joint Training - Epoch {current_epoch_num} Summary:")
        logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
        logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
        logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
        logger.info(f"  Final Avg Repulsion Loss: {avg_repulsion_loss:.4f}")
        logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
        logger.info(f"  Repulsion λ (epoch): {current_lambda_rep:.4f}")
    elif encoder_idx is not None:
        logger.info(f"Encoder {encoder_idx} - Epoch {current_epoch_num} Summary:")
        logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
        logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
        logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
        logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
    else:
        logger.info(f"Epoch {current_epoch_num} Summary:")
        logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
        logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
        logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
        logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
    logger.info("=" * 60)

    return avg_loss_for_epoch, avg_shape_loss, avg_grid_loss, avg_kl_loss, avg_repulsion_loss, current_lambda_rep


def main_training(file_store_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get current settings (don't reload from file - preserve sweep configurations)
    global data_settings, model_architecture, training_settings, latent_optimization
    global TRAINING_KEYS, N_EXAMPLES_PER_TASK, N_PAIRS_PER_EXAMPLE
    global DROPOUT
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, BETA
    global OPTIMIZE_Z, OPTIMIZE_Z_NUM_STEPS, OPTIMIZE_Z_LR
    global OPTIMIZE_Z_INFERENCE, OPTIMIZE_Z_INFERENCE_NUM_STEPS, OPTIMIZE_Z_INFERENCE_LR

    # Use current in-memory settings (important for sweeps - don't reload from file!)
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    repulsion_loss_settings = settings.get_repulsion_loss_settings()
    wandb_settings = settings.get_wandb_settings()

    # Initialize wandb for training mode (will be done after run_dir is created)
    wandb_logger = None

    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings after reload.")
    N_EXAMPLES_PER_TASK = data_settings['n']
    N_PAIRS_PER_EXAMPLE = data_settings.get('n_pairs',1)
    
    DROPOUT = model_architecture['dropout']
    
    BATCH_SIZE = training_settings['batch_size']
    NUM_EPOCHS = training_settings['num_epochs']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']

    OPTIMIZE_Z = latent_optimization['training']['enabled']
    OPTIMIZE_Z_NUM_STEPS = latent_optimization['training']['num_steps']
    OPTIMIZE_Z_LR = latent_optimization['training']['learning_rate']
    OPTIMIZE_Z_INFERENCE = latent_optimization['inference']['enabled']
    OPTIMIZE_Z_INFERENCE_NUM_STEPS = latent_optimization['inference']['num_steps']
    OPTIMIZE_Z_INFERENCE_LR = latent_optimization['inference']['learning_rate']
    
    set_seed(data_settings['training_seed'])

    SPLIT_ACROSS_ENCODERS = data_settings.get('split_across_encoders', True)
    # Check if multi-encoder training is enabled
    NUM_ENCODERS = model_architecture.get('num_encoders', 1)
    is_multi_encoder = NUM_ENCODERS > 1

    run_dir = create_run_directory(file_store_name)
    logger = setup_logging(run_dir)
    logger.info(f"Starting training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)

    # Initialize wandb for training mode (now that run_dir is available)
    if wandb_settings.get('enabled', False):
        # Don't override WANDB_PROJECT_NAME - let it be set by the sweep or environment
        wandb_logger = init_wandb_for_mode('train', run_dir)
        if wandb_logger:
            logger.info(f"✓ Wandb logging enabled: {wandb_logger.run.name}")
        else:
            logger.info("⚠ Wandb initialization failed, continuing without wandb")

    logger.info("Generating and preparing data...")
    print("Generating and preparing data...")

    if is_multi_encoder:
        logger.info(f"Multi-encoder training enabled with {NUM_ENCODERS} encoders")
        print(f"Multi-encoder training enabled with {NUM_ENCODERS} encoders")
        
        if SPLIT_ACROSS_ENCODERS:
            # ---------------------- KEY-BASED SPLITTING ----------------------
            logger.info("Using key-based dataset splitting for multi-encoder training (split_across_encoders = True)")
            print("Using key-based dataset splitting for multi-encoder training (split_across_encoders = True)")
            
            dataset_splits, key_to_encoder_mapping, splitting_statistics = split_dataset_by_keys_for_multi_encoder(
                TRAINING_KEYS, NUM_ENCODERS, N_EXAMPLES_PER_TASK, generate_and_process_tasks
            )
            
            # Store splitting information for later use
            training_metadata = {
                'key_to_encoder_mapping': key_to_encoder_mapping,
                'splitting_statistics': splitting_statistics,
                'training_keys': TRAINING_KEYS,
                'num_encoders': NUM_ENCODERS,
                'split_across_encoders': True
            }
            
            # Create dataloaders for each encoder
            encoder_dataloaders = []
            for i, (enc_inputs, enc_outputs) in enumerate(dataset_splits):
                if enc_inputs and enc_outputs:
                    dataloader = prepare_dataloader(enc_inputs, enc_outputs, BATCH_SIZE)
                    encoder_dataloaders.append(dataloader)
                    
                    # Log which keys are trained by this encoder
                    encoder_keys = splitting_statistics['keys_per_encoder'][i]
                    logger.info(f"Encoder {i}: {len(enc_inputs)} samples from keys {encoder_keys}")
                    print(f"Encoder {i}: {len(enc_inputs)} samples from keys {encoder_keys}")
                else:
                    # Create empty dataloader for consistency
                    encoder_dataloaders.append(prepare_dataloader([], [], BATCH_SIZE))
                    logger.info(f"Encoder {i}: No data assigned")
                    print(f"Encoder {i}: No data assigned")
        else:
            # ---------------------- MIXED DATASET (NO SPLIT) ------------------
            logger.info("split_across_encoders = False → using mixed dataset for all encoders")
            print("split_across_encoders = False → using mixed dataset for all encoders")
            
            all_input_sequences = []
            all_output_sequences = []
            for task_key in TRAINING_KEYS:
                try:
                    _, _, _, task_input_sequences, task_output_sequences = generate_and_process_tasks(task_key, N_EXAMPLES_PER_TASK)
                    all_input_sequences.extend(task_input_sequences)
                    all_output_sequences.extend(task_output_sequences)
                    logger.info(f"Generated {len(task_input_sequences)} pairs for task {task_key}")
                except Exception as e:
                    logger.error(f"Error generating data for task {task_key}: {e}")
                    continue
            if not all_input_sequences:
                logger.error("No data generated for mixed dataset. Exiting training.")
                print("No data generated for mixed dataset. Exiting training.")
                return None, None
            mixed_dataloader = prepare_dataloader(all_input_sequences, all_output_sequences, BATCH_SIZE)
            # Use the same dataloader reference for each encoder to keep downstream code intact
            encoder_dataloaders = [mixed_dataloader for _ in range(NUM_ENCODERS)]
            
            training_metadata = {
                'training_keys': TRAINING_KEYS,
                'num_encoders': NUM_ENCODERS,
                'split_across_encoders': False,
                'mixed_dataset_samples': len(all_input_sequences)
            }
    else:
        # Single encoder training - generate data normally
        logger.info("Generating data for single encoder training...")
        print("Generating data for single encoder training...")

        all_input_sequences = []
        all_output_sequences = []
        logger.info(f"Generating data for tasks: {TRAINING_KEYS}")
        print(f"Generating data for tasks: {TRAINING_KEYS}")

        for task_key in TRAINING_KEYS:
            logger.info(f"Processing task: {task_key} with {N_EXAMPLES_PER_TASK} examples")
            print(f"Processing task: {task_key} with {N_EXAMPLES_PER_TASK} examples")
            try:
                _, _, _, task_input_sequences, task_output_sequences = generate_and_process_tasks(task_key, N_EXAMPLES_PER_TASK)
                all_input_sequences.extend(task_input_sequences)
                all_output_sequences.extend(task_output_sequences)
                logger.info(f"Generated {len(task_input_sequences)} pairs for task {task_key}")
                print(f"Generated {len(task_input_sequences)} pairs for task {task_key}")
            except Exception as e:
                logger.error(f"Error generating data for task {task_key}: {e}")
                print(f"Error generating data for task {task_key}: {e}")
                continue 
        
        if not all_input_sequences:
            logger.error("No data generated from any task. Exiting training.")
            print("No data generated from any task. Exiting training.")
            return None, None

        input_sequences = all_input_sequences
        output_sequences = all_output_sequences
        logger.info(f"Total generated {len(input_sequences)} pairs of sequences from {len(TRAINING_KEYS)} tasks.")
        print(f"Total generated {len(input_sequences)} pairs of sequences from {len(TRAINING_KEYS)} tasks.")
        
        dataloader = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)
        training_metadata = {
            'training_keys': TRAINING_KEYS,
            'num_encoders': 1,
            'single_encoder_training': True
        }

    logger.info("Initializing model...")
    print("Initializing model...")
    model = LatentProgramNetwork().to(device)
    logger.info(f"Model initialized: {type(model)}")

    optimizer_weight_decay = training_settings.get('optimizer_weight_decay', 0.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=optimizer_weight_decay)
    print(f"Model and optimizer initialized. Optimizer weight decay: {optimizer_weight_decay}")

    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    logger.info(f"Mixed precision training enabled: {use_mixed_precision}")

    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Learning rate scheduler
    lr_scheduler_config = training_settings.get('learning_rate_scheduler', {'type': 'none'})
    scheduler = None
    if lr_scheduler_config['type'] == 'cosine':
        # CosineAnnealingLR needs T_max which is total number of steps.
        # Total steps = (num_epochs - warmup_epochs) * len(dataloader) / gradient_accumulation_steps
        # This is a bit tricky if warmup is per epoch. Let's simplify:
        # If using warmup, scheduler starts after warmup.
        # For now, let's assume T_max is for the entire training duration after warmup.
        # A common setup for cosine annealing with warmup is to use warmup for N epochs,
        # then cosine anneal for M-N epochs.
        # Or, simpler: a warmup phase, then a fixed scheduler.
        # For this integration, let's stick to a simple CosineAnnealingLR without complex warmup logic
        # directly tied to step count, or use a simpler StepLR if cosine is too complex here.
        # A more robust implementation would wrap this in a custom scheduler class.
        # For now, this is a basic setup.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS * len(dataloader) // gradient_accumulation_steps, # Approximation of total steps
            eta_min=lr_scheduler_config.get('lr_min', 1e-6)
        )
        logger.info(f"Using CosineAnnealingLR scheduler. T_max={scheduler.T_max}, eta_min={scheduler.eta_min}")
    elif lr_scheduler_config['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_config.get('step_size', 30), gamma=lr_scheduler_config.get('gamma', 0.1))
        logger.info(f"Using StepLR scheduler. Step_size={scheduler.step_size}, gamma={scheduler.gamma}")


    param_info = count_model_parameters(model)
    print("Model parameter count completed.")

    results = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_metrics': [],
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'losses_gradient_ascent': [],
        'training_metadata': training_metadata  # Add training metadata
    }
    
    # Store training sequences for visualization (both single and multi-encoder)
    if not is_multi_encoder:
        results['input_sequences'] = [seq.tolist() for seq in input_sequences]
        results['output_sequences'] = [seq.tolist() for seq in output_sequences]
    else:
        # For multi-encoder, combine all training data for visualization
        all_inputs = []
        all_outputs = []
        for encoder_idx in range(NUM_ENCODERS):
            for batch_input, batch_output in encoder_dataloaders[encoder_idx]:
                all_inputs.extend(batch_input.tolist())
                all_outputs.extend(batch_output.tolist())
        
        results['input_sequences'] = all_inputs
        results['output_sequences'] = all_outputs
        print(f"Saved {len(all_inputs)} combined training sequences from {NUM_ENCODERS} encoders for visualization")

    # Save initial settings and model parameters
    logger.info("Saving initial model parameters and settings...")
    print("Saving initial model parameters and settings...")
    
    # Store parameter info in results for later use
    results['model_parameter_info'] = param_info
    save_results(results, run_dir)

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        logger.info("\n" + "=" * 80)
        logger.info(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} started. LR: {current_lr}")
        logger.info("=" * 80)

        if is_multi_encoder:
            # Check if we should use joint training with repulsion loss
            repulsion_loss_settings = settings.get_repulsion_loss_settings()
            use_joint_training = repulsion_loss_settings.get('enabled', True)
            
            if use_joint_training:
                # Joint training: train all encoders together with repulsion loss
                logger.info(f"\n--- Joint Training with Repulsion Loss ---")
                print(f"Joint training all encoders with repulsion loss...")
                
                # Build (or reuse) combined dataloader
                if SPLIT_ACROSS_ENCODERS:
                    combined_input_sequences = []
                    combined_output_sequences = []
                    for encoder_dataloader in encoder_dataloaders:
                        for batch_input, batch_output in encoder_dataloader:
                            combined_input_sequences.extend(batch_input.tolist())
                            combined_output_sequences.extend(batch_output.tolist())
                    import random
                    combined_indices = list(range(len(combined_input_sequences)))
                    random.shuffle(combined_indices)
                    shuffled_input_sequences = [combined_input_sequences[i] for i in combined_indices]
                    shuffled_output_sequences = [combined_output_sequences[i] for i in combined_indices]
                    combined_dataloader = prepare_dataloader(
                        shuffled_input_sequences,
                        shuffled_output_sequences,
                        BATCH_SIZE
                    )
                else:
                    # No split: reuse the first encoder dataloader (all data already mixed)
                    combined_dataloader = encoder_dataloaders[0]

                # Train with joint training
                avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss, avg_repulsion_loss, current_lambda_rep = train_model(
                    model, combined_dataloader, optimizer, run_dir, logger, 
                    scaler, use_mixed_precision, gradient_accumulation_steps,
                    current_epoch_num=epoch+1, total_epochs=NUM_EPOCHS, 
                    joint_training=True
                )
                
                avg_epoch_loss = avg_loss  # Ensure variable exists for downstream checkpoint/save logic

                # Store metrics
                results['epoch_losses'].append(avg_loss)
                results['epoch_metrics'].append({
                    'epoch': epoch + 1,
                    'training_mode': 'joint',
                    'avg_shape_loss': avg_shape_loss,
                    'avg_grid_loss': avg_grid_loss,
                    'avg_kl_loss': avg_kl_loss,
                    'avg_repulsion_loss': avg_repulsion_loss,
                    'repulsion_lambda': current_lambda_rep,
                    'avg_total_loss': avg_loss,
                    'learning_rate': current_lr
                })
                
                # Log to console
                logger.info(f"Joint training completed for epoch {epoch+1}")
                print(f"Epoch {epoch+1} completed. Joint training loss: {avg_loss:.4f}")
            else:
                # Original multi-encoder training: train each encoder individually
                epoch_losses = []
                epoch_metrics_list = []
                
                for encoder_idx in range(NUM_ENCODERS):
                    logger.info(f"\n--- Training Encoder {encoder_idx} ---")
                    print(f"Training Encoder {encoder_idx}...")
                    
                    avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss, avg_repulsion_loss, current_lambda_rep = train_model(
                        model, encoder_dataloaders[encoder_idx], optimizer, run_dir, logger, 
                        scaler, use_mixed_precision, gradient_accumulation_steps,
                        current_epoch_num=epoch+1, total_epochs=NUM_EPOCHS, encoder_idx=encoder_idx
                    )
                    
                    epoch_losses.append(avg_loss)
                    epoch_metrics_list.append({
                        'encoder_idx': encoder_idx,
                        'avg_shape_loss': avg_shape_loss,
                        'avg_grid_loss': avg_grid_loss,
                        'avg_kl_loss': avg_kl_loss,
                        'avg_total_loss': avg_loss
                    })
                
                # Average losses across all encoders for epoch summary
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                results['epoch_losses'].append(avg_epoch_loss)
                results['epoch_metrics'].append({
                    'epoch': epoch + 1,
                    'training_mode': 'individual',
                    'multi_encoder_metrics': epoch_metrics_list,
                    'avg_total_loss': avg_epoch_loss,
                    'learning_rate': current_lr
                })
        else:
            # Single encoder training
            avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss, avg_repulsion_loss, current_lambda_rep = train_model(
                model, dataloader, optimizer, run_dir, logger, 
                scaler, use_mixed_precision, gradient_accumulation_steps,
                current_epoch_num=epoch+1, total_epochs=NUM_EPOCHS
            )
            results['epoch_losses'].append(avg_loss)
            results['epoch_metrics'].append({
                'epoch': epoch + 1,
                'avg_shape_loss': avg_shape_loss,
                'avg_grid_loss': avg_grid_loss,
                'avg_kl_loss': avg_kl_loss,
                'avg_total_loss': avg_loss,
                'learning_rate': current_lr
            })
        
        if scheduler:
            scheduler.step() # Step the scheduler each epoch

        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed.")
        if is_multi_encoder:
            if use_joint_training:
                logger.info(f"Joint Training Loss: {avg_loss:.4f}")
                print(f"Epoch {epoch+1} completed. Joint Training Loss: {avg_loss:.4f}")
            else:
                logger.info(f"Average Loss across encoders: {avg_epoch_loss:.4f}")
                print(f"Epoch {epoch+1} completed. Average Loss across encoders: {avg_epoch_loss:.4f}")
        else:
            logger.info(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate accuracy at the end of each epoch.
        logger.info("\n" + "=" * 40)
        logger.info(f"Evaluating accuracy at end of Epoch {epoch+1}")
        logger.info("=" * 40)
        
        if is_multi_encoder:
            # Multi-encoder: evaluate each encoder individually + PoE
            epoch_accuracy_data = {
                'epoch': epoch + 1,
                'individual_encoders': {},
                'poe_accuracy': {}
            }
            
            # Evaluate each encoder individually on its own data
            for eval_encoder_idx in range(NUM_ENCODERS):
                logger.info(f"\n--- Evaluating Encoder {eval_encoder_idx} ---")
                print(f"Evaluating Encoder {eval_encoder_idx}...")
                
                # Use the specific encoder's dataloader for individual evaluation
                encoder_dataloader = encoder_dataloaders[eval_encoder_idx]
                encoder_accuracy = evaluate_accuracy(
                    model, encoder_dataloader, device, 
                    is_multi_encoder=True, encoder_idx=eval_encoder_idx, 
                    optimize_z=OPTIMIZE_Z, logger=logger
                )
                
                epoch_accuracy_data['individual_encoders'][eval_encoder_idx] = encoder_accuracy
                print(f"Encoder {eval_encoder_idx} - Shape: {encoder_accuracy['shape_accuracy']:.4f}, "
                      f"Grid: {encoder_accuracy['grid_accuracy']:.4f}, "
                      f"Overall: {encoder_accuracy['overall_accuracy']:.4f}, "
                      f"Sample Exact: {encoder_accuracy['sample_exact_accuracy']:.4f}")
            
            logger.info("\n--- Multi-encoder Training: Individual Encoder Evaluation Only ---")
            print("Note: PoE evaluation removed from training. PoE will be evaluated during inference/evaluation phase.")
            
            # Store both detailed and summary accuracy data
            results['epoch_accuracies'].append(epoch_accuracy_data)
            
            # Remove duplicate legacy format - visualizers will extract what they need from detailed data
            # results['epoch_accuracies'].append({
            #     'epoch': epoch + 1,
            #     'shape_accuracy': poe_accuracy['shape_accuracy'],
            #     'grid_accuracy': poe_accuracy['grid_accuracy'],
            #     'overall_accuracy': poe_accuracy['overall_accuracy'],
            #     'sample_exact_accuracy': poe_accuracy['sample_exact_accuracy']
            # })
        else:
            # Single encoder: standard evaluation
            logger.info("\n--- Evaluating Single Encoder ---")
            print("Evaluating model...")
            
            single_accuracy = evaluate_accuracy(
                model, dataloader, device,
                is_multi_encoder=False, encoder_idx=None,
                optimize_z=OPTIMIZE_Z, logger=logger
            )
            
            # Store accuracy data (add epoch info)
            single_accuracy['epoch'] = epoch + 1
            results['epoch_accuracies'].append(single_accuracy)
            
            print(f"Model - Shape: {single_accuracy['shape_accuracy']:.4f}, "
                  f"Grid: {single_accuracy['grid_accuracy']:.4f}, "
                  f"Overall: {single_accuracy['overall_accuracy']:.4f}, "
                  f"Sample Exact: {single_accuracy['sample_exact_accuracy']:.4f}")

        model.train() # Already called at the start of train_model function for the next epoch

        # Log training metrics to wandb
        if wandb_logger:
            if is_multi_encoder:
                if use_joint_training:
                    # Log joint training metrics including repulsion loss
                    wandb_logger.log_training_metrics(epoch + 1, {
                        'avg_shape_loss': avg_shape_loss,
                        'avg_grid_loss': avg_grid_loss,
                        'avg_kl_loss': avg_kl_loss,
                        'avg_repulsion_loss': avg_repulsion_loss,
                        'repulsion_lambda': current_lambda_rep,
                        'avg_total_loss': avg_loss,
                        'training_mode': 'joint'
                    })
                else:
                    # Log average metrics across encoders
                    wandb_logger.log_training_metrics(epoch + 1, {
                        'avg_shape_loss': sum(m['avg_shape_loss'] for m in epoch_metrics_list) / len(epoch_metrics_list),
                        'avg_grid_loss': sum(m['avg_grid_loss'] for m in epoch_metrics_list) / len(epoch_metrics_list),
                        'avg_kl_loss': sum(m['avg_kl_loss'] for m in epoch_metrics_list) / len(epoch_metrics_list),
                        'avg_total_loss': avg_epoch_loss,
                        'training_mode': 'individual'
                    })
                # Log accuracy metrics
                wandb_logger.log_accuracy_metrics(epoch + 1, epoch_accuracy_data)
            else:
                # Single encoder logging
                wandb_logger.log_training_metrics(epoch + 1, {
                    'avg_shape_loss': avg_shape_loss,
                    'avg_grid_loss': avg_grid_loss,
                    'avg_kl_loss': avg_kl_loss,
                    'avg_total_loss': avg_loss
                })
                wandb_logger.log_accuracy_metrics(epoch + 1, single_accuracy)

        # Run evaluation and log visualizations every N epochs
        if wandb_logger and should_run_evaluation(epoch + 1, wandb_settings.get('log_interval', 1), NUM_EPOCHS):
            logger.info(f"Running evaluation and visualization logging for epoch {epoch+1}...")
            eval_results = run_quick_evaluation(model, run_dir, epoch + 1)
            if eval_results:
                # Pass the current in-memory model to avoid loading from disk
                log_evaluation_to_wandb(eval_results, run_dir, epoch + 1, wandb_logger, current_model=model)
            else:
                # Log visualizations without evaluation results (training-only visualizations)
                wandb_logger.log_visualizations(run_dir, epoch + 1)

        # Save checkpoint and results at regular intervals or at the end
        save_interval = training_settings.get('save_checkpoint_interval', 50)
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == NUM_EPOCHS
        
        if should_save:
            logger.info(f"Saving checkpoint and results at epoch {epoch+1}...")
            print(f"Saving checkpoint and results at epoch {epoch+1}...")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1, avg_loss if not is_multi_encoder else avg_epoch_loss, run_dir)
            
            # Save updated results (this will overwrite previous results.pkl with current progress)
            save_results(results, run_dir)
            
            logger.info(f"Checkpoint and results saved at epoch {epoch+1}.")
            print(f"Checkpoint and results saved at epoch {epoch+1}.")

    print("Training complete.")
    
    # Final save of complete results
    logger.info("Saving final complete results...")
    print("Saving final complete results...")
    save_results(results, run_dir)
    
    # Save final model and upload to WandB
    if wandb_logger:
        logger.info("Saving and uploading final model...")
        print("Saving and uploading final model...")
        
        # Save the final model state
        final_model_path = os.path.join(run_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'training_metadata': training_metadata,
            'final_epoch': NUM_EPOCHS,
            'final_loss': avg_loss if not is_multi_encoder else avg_epoch_loss
        }, final_model_path)
        
        # Upload final model and config to WandB
        config_path = os.path.join(run_dir, 'model_settings.json')
        wandb_logger.upload_final_model(final_model_path, config_path)

    # Finish wandb run
    if wandb_logger:
        wandb_logger.finish()

    print("Results saved in:", run_dir)
    return results, model