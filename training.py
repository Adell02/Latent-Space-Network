import torch
from torch.optim import Adam
import json

from utils.model_utils import set_seed, create_run_directory, setup_logging, prepare_dataloader, save_checkpoint, save_results, count_model_parameters
from models.base_model import LatentProgramNetwork, compute_loss
from utils.settings_manager import settings
from re_arc.main import generate_and_process_tasks
from utils.latent_functions import get_optimized_z

##############################
# Main Training Function
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

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        device = next(model.parameters()).device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            # Pass beta from global settings (or training_settings directly)
            loss, shape_loss_comp, grid_loss_comp, kl_loss_comp = compute_loss(model, input_seq, target_seq, beta=BETA, return_components=True)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_total_loss += loss.item() * gradient_accumulation_steps # Unscale for logging
        epoch_shape_loss_sum += shape_loss_comp.item()
        epoch_grid_loss_sum += grid_loss_comp.item()
        epoch_kl_loss_sum += kl_loss_comp.item()
        
        progress = (batch_idx + 1) / total_batches * 100
        # Log less frequently if accumulating gradients
        log_frequency = gradient_accumulation_steps * 5 
        if (batch_idx + 1) % log_frequency == 0 or (batch_idx + 1) == total_batches:
            # Log individual unscaled losses for the current batch/step
            logger.info(f"Epoch [{current_epoch_num}/{total_epochs}] Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")
            logger.info(f"  Step Loss: {loss.item() * gradient_accumulation_steps:.4f} (Shape: {shape_loss_comp.item():.4f}, Grid: {grid_loss_comp.item():.4f}, KL: {kl_loss_comp.item():.4f})")


    avg_loss_for_epoch = epoch_total_loss / total_batches
    avg_shape_loss = epoch_shape_loss_sum / total_batches
    avg_grid_loss = epoch_grid_loss_sum / total_batches
    avg_kl_loss = epoch_kl_loss_sum / total_batches
    
    logger.info("=" * 60)
    logger.info(f"Epoch {current_epoch_num} Summary:")
    logger.info(f"  Final Avg Shape Loss: {avg_shape_loss:.4f}")
    logger.info(f"  Final Avg Grid Loss: {avg_grid_loss:.4f}")
    logger.info(f"  Final Avg KL Loss: {avg_kl_loss:.4f}")
    logger.info(f"  Final Avg Total Loss: {avg_loss_for_epoch:.4f}")
    logger.info("=" * 60)

    return avg_loss_for_epoch, avg_shape_loss, avg_grid_loss, avg_kl_loss


def main_training(file_store_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reload settings variables to ensure they are current, especially if an optimization function was called
    global data_settings, model_architecture, training_settings, latent_optimization
    global TRAINING_KEYS, N_EXAMPLES_PER_TASK, N_PAIRS_PER_EXAMPLE
    global LATENT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT
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
    logger.info(f"Starting training for ARC problems: {TRAINING_KEYS}")
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

    dataloader = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)

    logger.info("Initializing model...")
    print("Initializing model...")
    # Model instantiation will pick up global LATENT_DIM, HIDDEN_DIM etc. which are now reloaded
    model = LatentProgramNetwork().to(device) 

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


    count_model_parameters(model)
    print("Model parameter count completed.")

    results = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_metrics': [], # To store shape, grid, kl losses per epoch
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_sequences': [seq.tolist() for seq in input_sequences], # Convert to list for JSON
        'output_sequences': [seq.tolist() for seq in output_sequences], # Convert to list for JSON
        'losses_gradient_ascent': []
    }

    # Save initial settings and model parameters
    logger.info("Saving initial model parameters and settings...")
    print("Saving initial model parameters and settings...")
    save_results(results, run_dir)

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        logger.info("\\n" + "=" * 80)
        logger.info(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS} started. LR: {current_lr}")
        logger.info("=" * 80)

        avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss = train_model(
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

        logger.info(f"\\nEpoch {epoch+1}/{NUM_EPOCHS} completed.")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate accuracy at the end of each epoch.
        model.eval()
        epoch_shape_correct = 0
        epoch_shape_tokens = 0
        epoch_grid_correct = 0
        epoch_grid_tokens = 0
        sample_exact_correct = 0
        total_samples_eval = 0 # Use a different variable for clarity during evaluation

        # We now leave the no_grad block for the latent optimization step if it's enabled for inference.
        # The get_optimized_z function handles its own grad context.
        for batch_input_eval, batch_target_eval in dataloader: # Can use a validation dataloader here if available
            total_samples_eval += batch_input_eval.size(0)
            batch_input_eval = batch_input_eval.to(device)
            batch_target_eval = batch_target_eval.to(device)

            # Use the appropriate latent optimization method for training
            if OPTIMIZE_Z: # Check training specific flag
                # Pass context='training' to use training-specific parameters
                z_eval, losses_eval = get_optimized_z(model, batch_input_eval, batch_target_eval, 
                                                      context='training')
                if losses_eval is not None and isinstance(results.get('losses_gradient_ascent'), list):
                    if not isinstance(losses_eval, list):
                        losses_eval = [losses_eval]
                    results['losses_gradient_ascent'].extend(losses_eval)
            else:
                with torch.no_grad(): # Ensure no grads if not optimizing z
                    mu_eval, log_var_eval = model.encoder(batch_input_eval, batch_target_eval)
                    z_eval = model.reparameterize(mu_eval, log_var_eval)

            # Now, perform decoding with no_grad.
            with torch.no_grad():
                # The decoder's target_seq argument is for teacher forcing. 
                # For true autoregressive generation during eval, it should be None
                # or handle it differently if evaluating reconstruction of target.
                # Current model.decoder uses target_seq for teacher-forced decoding.
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
                elif torch.all(shape_pred_eval[i] == shape_tgt_eval[i]): # If grid is empty, only shape matters for exact
                    sample_exact_correct += 1


        epoch_shape_accuracy = epoch_shape_correct / epoch_shape_tokens if epoch_shape_tokens > 0 else 0.0
        epoch_grid_accuracy = epoch_grid_correct / epoch_grid_tokens if epoch_grid_tokens > 0 else 0.0
        epoch_overall_accuracy = (epoch_shape_correct + epoch_grid_correct) / (epoch_shape_tokens + epoch_grid_tokens) if (epoch_shape_tokens + epoch_grid_tokens) > 0 else 0.0
        sample_level_accuracy = sample_exact_correct / total_samples_eval if total_samples_eval > 0 else 0.0

        results['epoch_accuracies'].append({
            'epoch': epoch + 1,
            'shape_accuracy': epoch_shape_accuracy,
            'grid_accuracy': epoch_grid_accuracy,
            'overall_accuracy': epoch_overall_accuracy,
            'sample_exact_accuracy': sample_level_accuracy
        })

        logger.info(f"Epoch {epoch+1} Accuracy -- Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")
        print(f"Epoch {epoch+1} Accuracy: Shape: {epoch_shape_accuracy:.4f}, Grid: {epoch_grid_accuracy:.4f}, Overall: {epoch_overall_accuracy:.4f}, Sample Exact: {sample_level_accuracy:.4f}")

        model.train() # Already called at the start of train_model function for the next epoch

        # Save checkpoint and results at regular intervals or at the end
        save_interval = training_settings.get('save_checkpoint_interval', 50)
        should_save = (epoch + 1) % save_interval == 0 or (epoch + 1) == NUM_EPOCHS
        
        if should_save:
            logger.info(f"Saving checkpoint and results at epoch {epoch+1}...")
            print(f"Saving checkpoint and results at epoch {epoch+1}...")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, run_dir)
            
            # Save updated results (this will overwrite previous results.pkl with current progress)
            save_results(results, run_dir)
            
            logger.info(f"Checkpoint and results saved at epoch {epoch+1}.")
            print(f"Checkpoint and results saved at epoch {epoch+1}.")

    print("Training complete. Starting final evaluation on the dataloader (can be train or val)...")
    model.eval()
    
    # Final evaluation loop to collect latent representations and reconstructions
    # This is for collecting final latent variables and reconstructions.
    # Limit the number of batches processed to avoid memory issues
    final_eval_batch_count = 0
    max_eval_batches = 20  # Limit to save memory
    
    for batch_input, batch_target in dataloader:
        final_eval_batch_count += 1
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        # Consistent z retrieval for final eval
        if OPTIMIZE_Z_INFERENCE:
            z, losses = get_optimized_z(model, batch_input, batch_target, 
                                        context='inference')
            # Get mu and log_var for storage
            with torch.no_grad():
                mu, log_var = model.encoder(batch_input, batch_target)
        else:
            with torch.no_grad():
                mu, log_var = model.encoder(batch_input, batch_target)
                z = model.reparameterize(mu, log_var)
        
        with torch.no_grad(): # Ensure no grads for decoder pass
            shape_logits, grid_logits = model.decoder(z, batch_input, target_seq=batch_target) # Using target_seq for reconstruction eval
            
            # Store only a subset of these potentially large tensors if memory is an issue
            if isinstance(results.get('latent_mus'), list):
                mu_np = mu.cpu().numpy() if isinstance(mu, torch.Tensor) else mu
                results['latent_mus'].append(mu_np.tolist())
            if isinstance(results.get('latent_log_vars'), list):
                log_var_np = log_var.cpu().numpy() if isinstance(log_var, torch.Tensor) else log_var
                results['latent_log_vars'].append(log_var_np.tolist())
            if isinstance(results.get('latent_zs'), list):
                z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
                results['latent_zs'].append(z_np.tolist())
            
            # Store only one reconstruction per batch to save memory
            if isinstance(results.get('reconstructions'), list) and len(results['reconstructions']) < 20:
                results['reconstructions'].append({
                    'input': batch_input[0].cpu().numpy().tolist(),
                    'target': batch_target[0].cpu().numpy().tolist(),
                    'reconstruction': (shape_logits[0].cpu().numpy().tolist(), grid_logits[0].cpu().numpy().tolist())
                })
        
        print(f"Final evaluation: Processed batch {final_eval_batch_count}/{len(dataloader)}")
        
        # Limit stored reconstructions for large datasets
        if final_eval_batch_count >= max_eval_batches:
            print(f"Limiting stored latent vars/reconstructions to first {final_eval_batch_count} batches to save memory.")
            break

    # Final save of complete results
    logger.info("Saving final complete results...")
    print("Saving final complete results...")
    save_results(results, run_dir)

    print("Results saved in:", run_dir)
    return results, model