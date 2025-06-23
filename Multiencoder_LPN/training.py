import torch
from torch.optim import Adam
import json
import random
import numpy as np
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import set_seed, create_run_directory, setup_logging, prepare_dataloader, save_checkpoint, save_results, count_model_parameters
from models.multi_encoder_lpn import MultiEncoderLPN, multinomial_loss
# Use local multiencoder settings manager
import Multiencoder_LPN.settings_manager as multiencoder_settings
from re_arc.main import generate_and_process_tasks
from utils.latent_functions import get_optimized_z

# Get the local settings instance
settings = multiencoder_settings.settings

##############################
# Data Splitting for Multi-Encoder Training
##############################

def create_multi_encoder_batches(input_sequences, output_sequences, num_encoders, batch_size):
    """
    Create batches where each encoder gets different samples from the same task.
    For training: K different (x,y) pairs from the same ARC task for each encoder.
    """
    total_samples = len(input_sequences)
    batches = []
    
    # Group samples to ensure each batch has enough samples for all encoders
    samples_per_batch = batch_size * num_encoders
    
    for i in range(0, total_samples, samples_per_batch):
        end_idx = min(i + samples_per_batch, total_samples)
        batch_samples = end_idx - i
        
        if batch_samples < num_encoders:
            # Not enough samples for a complete batch
            break
            
        # Actual batch size for this iteration
        actual_batch_size = batch_samples // num_encoders
        
        # Create input views for each encoder
        encoder_views = []
        for enc_idx in range(num_encoders):
            start_enc = i + enc_idx * actual_batch_size
            end_enc = start_enc + actual_batch_size
            
            enc_inputs = torch.tensor(input_sequences[start_enc:end_enc]).float()
            enc_outputs = torch.tensor(output_sequences[start_enc:end_enc]).float()
            encoder_views.append((enc_inputs, enc_outputs))
        
        batches.append((encoder_views, actual_batch_size))
    
    return batches

def create_inference_batches(input_sequences, output_sequences, num_encoders, batch_size):
    """
    Create batches for inference where all encoders get the same (x,y) pair.
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
# Multi-Encoder Training Function
##############################

def train_model(model, dataloader, optimizer, run_dir, logger, scaler, use_mixed_precision, gradient_accumulation_steps, current_epoch_num, total_epochs):
    model.train()
    epoch_total_loss = 0
    epoch_shape_loss_sum = 0
    epoch_grid_loss_sum = 0
    epoch_kl_loss_sum = 0
    
    optimizer.zero_grad() # Ensure gradients are zeroed at the start of accumulation cycle / epoch

    logger.info("-" * 60)
    logger.info(f"Starting training batch loop for Epoch {current_epoch_num}/{total_epochs}...")
    total_batches = len(dataloader)

    for batch_idx, (encoder_views, actual_batch_size) in enumerate(dataloader):
        device = next(model.parameters()).device
        
        # Move all encoder views to device
        encoder_views_gpu = []
        for input_seq, target_seq in encoder_views:
            encoder_views_gpu.append((input_seq.to(device), target_seq.to(device)))

        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            # Forward pass through multi-encoder model
            (shape_logits, grid_logits), mu, log_var = model(encoder_views_gpu, training=True, sample_latent=True)
            
            # Use the first encoder's target for loss computation (they should all be equivalent)
            target_seq = encoder_views_gpu[0][1]
            
            # Compute loss using the multi-encoder loss function
            loss = multinomial_loss(
                (shape_logits, grid_logits), 
                target_seq, 
                beta=BETA, 
                mu=mu, 
                logvar=log_var
            )
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_total_loss += loss.item() * gradient_accumulation_steps # Unscale for logging
        
        progress = (batch_idx + 1) / total_batches * 100
        # Log less frequently if accumulating gradients
        log_frequency = gradient_accumulation_steps * 5 
        if (batch_idx + 1) % log_frequency == 0 or (batch_idx + 1) == total_batches:
            # Log individual unscaled losses for the current batch/step
            logger.info(f"Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
            logger.info(f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f}")

    avg_loss_for_epoch = epoch_total_loss / total_batches
    
    logger.info("=" * 60)
    logger.info(f"Epoch {current_epoch_num} Summary:")
    logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
    logger.info("=" * 60)

    return avg_loss_for_epoch, 0.0, 0.0, 0.0  # Return zeros for shape, grid, kl losses for now


def main_training(file_store_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reload settings variables to ensure they are current, especially if an optimization function was called
    global data_settings, model_architecture, training_settings, latent_optimization
    global TRAINING_KEYS, N_EXAMPLES_PER_TASK, N_PAIRS_PER_EXAMPLE
    global NUM_ENCODERS, LATENT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, BETA
    global OPTIMIZE_Z, OPTIMIZE_Z_NUM_STEPS, OPTIMIZE_Z_LR
    global OPTIMIZE_Z_INFERENCE, OPTIMIZE_Z_INFERENCE_NUM_STEPS, OPTIMIZE_Z_INFERENCE_LR

    settings.load_settings() # Force reload from file to ensure all parts of the code use updated settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()

    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings after reload.")
    N_EXAMPLES_PER_TASK = data_settings['n']
    N_PAIRS_PER_EXAMPLE = data_settings.get('n_pairs',1)
    
    NUM_ENCODERS = model_architecture['num_encoders']
    LATENT_DIM = model_architecture['latent_dim']
    HIDDEN_DIM = model_architecture['hidden_dim']
    NUM_LAYERS = model_architecture['num_layers']
    NUM_HEADS = model_architecture['num_heads']
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

    run_dir = create_run_directory(file_store_name)
    logger = setup_logging(run_dir)
    logger.info(f"Starting multi-encoder training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Number of encoders: {NUM_ENCODERS}")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)

    logger.info("Generating and preparing data...")
    print("Generating and preparing data...")

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

    # Create multi-encoder training batches
    logger.info(f"Creating multi-encoder training batches with {NUM_ENCODERS} encoders...")
    print(f"Creating multi-encoder training batches with {NUM_ENCODERS} encoders...")
    
    batches = create_multi_encoder_batches(input_sequences, output_sequences, NUM_ENCODERS, BATCH_SIZE)
    logger.info(f"Created {len(batches)} training batches")
    print(f"Created {len(batches)} training batches")

    logger.info("Initializing multi-encoder model...")
    print("Initializing multi-encoder model...")
    
    # Initialize the multi-encoder model
    model = MultiEncoderLPN(
        num_encoders=NUM_ENCODERS,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        encoder_max_length=model_architecture['encoder_max_length'],
        decoder_max_length=model_architecture['decoder_max_length']
    ).to(device)

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS * len(batches) // gradient_accumulation_steps,
            eta_min=lr_scheduler_config.get('lr_min', 1e-6)
        )
        logger.info(f"Using CosineAnnealingLR scheduler. T_max={scheduler.T_max}, eta_min={scheduler.eta_min}")
    elif lr_scheduler_config['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_config.get('step_size', 30), gamma=lr_scheduler_config.get('gamma', 0.1))
        logger.info(f"Using StepLR scheduler. Step_size={scheduler.step_size}, gamma={scheduler.gamma}")

    # Count model parameters (function prints the results)
    count_model_parameters(model)
    logger.info("Model parameter count completed")

    # Training results storage - match original LPN structure
    training_results = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_metrics': [],  # To store shape, grid, kl losses per epoch
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_sequences': [seq.tolist() for seq in input_sequences],  # Convert to list for JSON
        'output_sequences': [seq.tolist() for seq in output_sequences],  # Convert to list for JSON
        'losses_gradient_ascent': []
    }

    save_checkpoint_interval = training_settings.get('save_checkpoint_interval', 10)
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Starting Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Starting Epoch {epoch}/{NUM_EPOCHS}")

        # Shuffle batches for each epoch
        random.shuffle(batches)
        
        avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss = train_model(
            model, batches, optimizer, run_dir, logger, scaler, use_mixed_precision, 
            gradient_accumulation_steps, epoch, NUM_EPOCHS
        )

        # Store results - match original LPN structure
        training_results['epoch_losses'].append(avg_loss)
        training_results['epoch_metrics'].append({
            'epoch': epoch,
            'avg_shape_loss': avg_shape_loss,
            'avg_grid_loss': avg_grid_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_total_loss': avg_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Evaluate accuracy at the end of each epoch - match original LPN
        logger.info(f"Evaluating epoch {epoch} accuracy...")
        model.eval()
        epoch_shape_correct = 0
        epoch_shape_tokens = 0
        epoch_grid_correct = 0
        epoch_grid_tokens = 0
        sample_exact_correct = 0
        total_samples_eval = 0

        with torch.no_grad():
            # Use a subset of batches for evaluation to save time
            eval_batches = batches[:min(10, len(batches))]  # Evaluate on first 10 batches
            for batch_idx, (encoder_views, actual_batch_size) in enumerate(eval_batches):
                total_samples_eval += actual_batch_size
                
                # Move encoder views to device
                encoder_views_gpu = []
                for input_seq, target_seq in encoder_views:
                    encoder_views_gpu.append((input_seq.to(device), target_seq.to(device)))
                
                # Forward pass through multi-encoder model
                (shape_logits_eval, grid_logits_eval), mu_eval, log_var_eval = model(encoder_views_gpu, training=False)
                
                # Use the first encoder's target for evaluation (they should all be equivalent in training)
                target_seq = encoder_views_gpu[0][1]
                
                shape_pred_eval = shape_logits_eval.argmax(dim=-1)
                grid_pred_eval = grid_logits_eval.argmax(dim=-1)
                shape_tgt_eval = target_seq[:, 900:902].long()
                grid_tgt_eval = target_seq[:, :900].long()

                epoch_shape_correct += (shape_pred_eval == shape_tgt_eval).sum().item()
                epoch_shape_tokens += shape_tgt_eval.numel()
                
                for i in range(target_seq.size(0)):
                    tgt_rows_eval = int(target_seq[i, 900].item())
                    tgt_cols_eval = int(target_seq[i, 901].item())
                    active_pixels_eval = tgt_rows_eval * tgt_cols_eval
                    if active_pixels_eval > 0:
                        epoch_grid_correct += (grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]).sum().item()
                        epoch_grid_tokens += active_pixels_eval
                        if torch.all(shape_pred_eval[i] == shape_tgt_eval[i]) and \
                           torch.all(grid_pred_eval[i, :active_pixels_eval] == grid_tgt_eval[i, :active_pixels_eval]):
                            sample_exact_correct += 1
                    elif torch.all(shape_pred_eval[i] == shape_tgt_eval[i]):
                        sample_exact_correct += 1

        epoch_shape_accuracy = epoch_shape_correct / epoch_shape_tokens if epoch_shape_tokens > 0 else 0.0
        epoch_grid_accuracy = epoch_grid_correct / epoch_grid_tokens if epoch_grid_tokens > 0 else 0.0
        epoch_overall_accuracy = (epoch_shape_correct + epoch_grid_correct) / (epoch_shape_tokens + epoch_grid_tokens) if (epoch_shape_tokens + epoch_grid_tokens) > 0 else 0.0
        sample_level_accuracy = sample_exact_correct / total_samples_eval if total_samples_eval > 0 else 0.0

        training_results['epoch_accuracies'].append({
            'epoch': epoch,
            'shape_accuracy': epoch_shape_accuracy,
            'grid_accuracy': epoch_grid_accuracy,
            'overall_accuracy': epoch_overall_accuracy,
            'sample_exact_accuracy': sample_level_accuracy
        })

        logger.info(f"Epoch {epoch} Accuracy -- Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")
        print(f"Epoch {epoch} Accuracy: Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")

        model.train()  # Set back to training mode

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                # For cosine scheduler, step after each epoch
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate after epoch {epoch}: {current_lr:.6f}")

        # Save checkpoint and results
        if epoch % save_checkpoint_interval == 0:
            logger.info(f"Saving checkpoint and results at epoch {epoch}")
            save_checkpoint(model, optimizer, epoch, avg_loss, run_dir)
            save_results(training_results, run_dir)

    # Final evaluation loop to collect latent representations and reconstructions
    logger.info("Starting final evaluation to collect latent representations...")
    print("Training complete. Starting final evaluation to collect latent representations...")
    model.eval()
    
    final_eval_batch_count = 0
    max_eval_batches = 20  # Limit to save memory
    
    with torch.no_grad():
        for batch_idx, (encoder_views, actual_batch_size) in enumerate(batches):
            final_eval_batch_count += 1
            
            # Move encoder views to device
            encoder_views_gpu = []
            for input_seq, target_seq in encoder_views:
                encoder_views_gpu.append((input_seq.to(device), target_seq.to(device)))
            
            # Forward pass through multi-encoder model
            (shape_logits, grid_logits), mu, log_var = model(encoder_views_gpu, training=False)
            z = model._reparam(mu, log_var, sample=False)  # Use deterministic z for final evaluation
            
            # Use the first encoder's data for storing reconstructions
            target_seq = encoder_views_gpu[0][1]
            input_seq = encoder_views_gpu[0][0]
            
            # Store only a subset of these potentially large tensors if memory is an issue
            if isinstance(training_results.get('latent_mus'), list):
                mu_np = mu.cpu().numpy() if isinstance(mu, torch.Tensor) else mu
                training_results['latent_mus'].append(mu_np.tolist())
            if isinstance(training_results.get('latent_log_vars'), list):
                log_var_np = log_var.cpu().numpy() if isinstance(log_var, torch.Tensor) else log_var
                training_results['latent_log_vars'].append(log_var_np.tolist())
            if isinstance(training_results.get('latent_zs'), list):
                z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
                training_results['latent_zs'].append(z_np.tolist())
            
            # Store only one reconstruction per batch to save memory
            if isinstance(training_results.get('reconstructions'), list) and len(training_results['reconstructions']) < 20:
                training_results['reconstructions'].append({
                    'input': input_seq[0].cpu().numpy().tolist(),  # First sample
                    'target': target_seq[0].cpu().numpy().tolist(),
                    'reconstruction': (shape_logits[0].cpu().numpy().tolist(), grid_logits[0].cpu().numpy().tolist())
                })
            
            print(f"Final evaluation: Processed batch {final_eval_batch_count}/{len(batches)}")
            
            # Limit stored reconstructions for large datasets
            if final_eval_batch_count >= max_eval_batches:
                print(f"Limiting stored latent vars/reconstructions to first {final_eval_batch_count} batches to save memory.")
                break

    # Final save
    logger.info("Saving final checkpoint and results")
    save_checkpoint(model, optimizer, NUM_EPOCHS, avg_loss, run_dir)
    save_results(training_results, run_dir)

    logger.info("Training completed successfully!")
    print("Training completed successfully!")
    print("Results saved in:", run_dir)
    
    return training_results, model