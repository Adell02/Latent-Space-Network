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
from utils.training_helpers import create_mixed_domains_dataloader, create_infinite_dataloader

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
from utils.visualizers import plot_training_latent_space_per_epoch


def build_model(device, wandb_logger=None, global_step=None):
    """Build and return LatentProgramNetwork with architecture visualization."""
    from utils.model_architecture_viz import generate_architecture_visualizations, log_model_summary
    
    model = LatentProgramNetwork().to(device)
    
    # Generate architecture visualizations and upload to wandb
    if wandb_logger:
        print("[BUILDING] Generating model architecture visualizations...")
        generate_architecture_visualizations(model, wandb_logger, device, global_step)
        log_model_summary(model, wandb_logger, global_step)
    
    return model

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
                f"    [OK] Collected {latent_data['num_samples']} samples from Encoder {encoder_idx}"
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
        for batch in dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input, batch_target, batch_keys = batch[:3]
            else:
                batch_input, batch_target = batch[:2]
                batch_keys = None
            
            if sample_count >= max_samples:
                break
                
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            # Get latent representations - always use mean vectors
            mu, log_var,_ = model.encoder(batch_input, batch_target)
            z = model.reparameterize(mu, log_var)
            
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
        for batch in dataloader:
            # Handle batch structure with keys
            if len(batch) >= 3:
                batch_input_eval, batch_target_eval, batch_keys = batch[:3]
            else:
                batch_input_eval, batch_target_eval = batch[:2]
                batch_keys = None
            
            total_samples_eval += batch_input_eval.size(0)
            batch_input_eval = batch_input_eval.to(device)
            batch_target_eval = batch_target_eval.to(device)

            # Get latent representation
            if optimize_z:
                # Use latent optimization with correct context and settings
                latent_optimization = settings.get_latent_optimization()
                inference_settings = latent_optimization['inference']
                num_steps = inference_settings.get('num_steps', 10)
                lr = inference_settings.get('learning_rate', 0.1)
                
                z_eval, _ = get_optimized_z(
                    model, batch_input_eval, batch_target_eval, 
                    context='evaluation',  # Use evaluation context
                    num_steps=num_steps,
                    lr=lr
                )
            else:
                if is_multi_encoder:
                    if encoder_idx is not None:
                        # Individual encoder evaluation
                        mu_eval, log_var_eval = model(batch_input_eval, batch_target_eval, encoder_idx=encoder_idx)[1:3]
                    else:
                        # PoE inference
                        mu_eval, log_var_eval = model(batch_input_eval, batch_target_eval)[1:3]
                    z_eval = model.reparameterize(mu_eval, log_var_eval)
                else:
                    # Single encoder
                    mu_eval, log_var_eval,_ = model.encoder(batch_input_eval, batch_target_eval)
                    z_eval = model.reparameterize(mu_eval, log_var_eval)

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

def train_model(model, dataloader, optimizer, run_dir, logger, scaler,
                use_mixed_precision, gradient_accumulation_steps,
                current_epoch_num, total_epochs,
                encoder_idx=None, joint_training=False):
    """
    Training loop with unified loss computation for ALL training types.
    ALL training now uses the same compute_loss function with appropriate settings.
    """
    # Put model in train mode
    model.train()
    device = next(model.parameters()).device

    # Get unified settings for ALL training types
    enhanced_training = settings.get_enhanced_training()
    training_settings = settings.get_training_settings()
    repulsion_loss_settings = settings.get_repulsion_loss_settings()
    
    # ENABLE LEAVE-ONE-OUT TRAINING FOR ALL TRAINING TYPES
    cross_pair_enabled = training_settings.get('cross_pair_loss', {}).get('enabled', True)
    cross_pair_num_pairs = training_settings.get('cross_pair_loss', {}).get('num_pairs', 4)
    
    # Repulsion parameters (for joint training)
    use_repulsion_loss = repulsion_loss_settings.get('enabled', False) and joint_training
    repulsion_lambda = repulsion_loss_settings.get('lambda', 0.1)
    repulsion_margin = repulsion_loss_settings.get('margin', 0.5)
    repulsion_logvar_min = repulsion_loss_settings.get('logvar_min', -8.0)
    
    # Schedule repulsion weight if enabled
    if use_repulsion_loss:
        sched_cfg = repulsion_loss_settings.get('schedule', None)
        sched_type = sched_cfg.get('type', 'none') if sched_cfg else 'none'
        if sched_cfg and sched_type != 'none':
            warmup = sched_cfg.get('warmup_epochs', 0)
            if current_epoch_num <= warmup:
                repulsion_lambda = 0.0
            else:
                epoch_idx = current_epoch_num - warmup
                T_eff = max(total_epochs - warmup, 1)
                typ = sched_cfg.get('type','linear').lower()
                start, end = sched_cfg.get('start',0.0), sched_cfg.get('end',repulsion_lambda)
                frac = min(max((epoch_idx-1)/(T_eff-1),0.0),1.0)
                if typ == 'linear':
                    repulsion_lambda = start + frac*(end-start)
                elif typ == 'exponential':
                    repulsion_lambda = start * ((end/start)**frac) if start>0 else end
                else:
                    repulsion_lambda = repulsion_loss_settings.get('lambda', 0.1)

    # Accumulators
    epoch_total_loss = 0.0
    epoch_shape = epoch_grid = epoch_kl = epoch_repulsion = 0.0

    # Zero gradients before epoch
    optimizer.zero_grad()

    total_batches = len(dataloader)
    BETA = settings.get_training_settings()['beta']

    for batch_idx, batch in enumerate(dataloader):
        # MODIFIED: Handle keys in batch
        if len(batch) >= 3:
            input_seq, target_seq, batch_keys = batch[:3]
        else:
            input_seq, target_seq = batch[:2]
            batch_keys = None  # Fallback for non-key dataloaders
        
        input_seq  = input_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            # UNIFIED LOSS COMPUTATION FOR ALL TRAINING TYPES
            comp = compute_loss(
                model, input_seq, target_seq,
                beta=BETA, return_components=True,
                encoder_idx=encoder_idx if not joint_training else None,
                use_independent_decoder=(encoder_idx is not None and not joint_training),
                # ENABLE LEAVE-ONE-OUT TRAINING FOR ALL
                cross_pair_enabled=cross_pair_enabled,
                cross_pair_num_pairs=cross_pair_num_pairs,
                # Enhanced mechanisms
                current_epoch=current_epoch_num,
                **enhanced_training,
                # Repulsion loss parameters (only for joint training)
                use_repulsion_loss=use_repulsion_loss,
                repulsion_lambda=repulsion_lambda,
                repulsion_margin=repulsion_margin,
                repulsion_logvar_min=repulsion_logvar_min
            )
            
            loss = comp['total_loss']
            shape_comp = comp.get('shape_loss', torch.tensor(0.0, device=device))
            grid_comp  = comp.get('grid_loss' , torch.tensor(0.0, device=device))
            kl_comp    = comp.get('kl_loss'   , torch.tensor(0.0, device=device))
            repulsion_comp = comp.get('repulsion_loss', torch.tensor(0.0, device=device))

            # normalize for accumulation
            loss = loss / gradient_accumulation_steps

        # backward + step
        scaler.scale(loss).backward()
        if batch_idx % gradient_accumulation_steps == 0 or batch_idx==total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_total_loss += loss.item() * gradient_accumulation_steps
        epoch_shape += shape_comp.item()
        epoch_grid  += grid_comp.item()
        epoch_kl    += kl_comp.item()
        epoch_repulsion += repulsion_comp.item()

    # return averages
    avg_loss = epoch_total_loss/total_batches
    return avg_loss, epoch_shape/total_batches, epoch_grid/total_batches,epoch_kl/total_batches, epoch_repulsion/total_batches, repulsion_lambda


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
    n_max_keys = data_settings.get('n_max_keys', None)
    if isinstance(TRAINING_KEYS, str) and TRAINING_KEYS.lower() == 'all':
        tasks_dir = os.path.join(os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
        all_keys = [fname[:-5] for fname in os.listdir(tasks_dir) if fname.endswith('.json')]
        all_keys.sort()
        if n_max_keys is not None:
            try:
                n_max_keys = int(n_max_keys)
                all_keys = all_keys[:n_max_keys]
            except Exception:
                pass
        TRAINING_KEYS = all_keys
    if TRAINING_KEYS is None or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings after reload.")
    N_EXAMPLES_PER_TASK = data_settings['n']
    N_PAIRS_PER_EXAMPLE = data_settings.get('n_pairs',1)
    
    DROPOUT = model_architecture['dropout']
    
    BATCH_SIZE = training_settings['batch_size']
    NUM_EPOCHS = training_settings['num_epochs']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    INFINITE_DATALOADER = training_settings.get('infinite_dataloader', False)
    BATCHES_PER_EPOCH = training_settings.get(
        'batches_per_epoch', max(1, (N_EXAMPLES_PER_TASK * len(TRAINING_KEYS)) // BATCH_SIZE))

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
    logger.info(f"Starting training for ARC problems {len(TRAINING_KEYS)} keys.")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)

    # Initialize results dictionary early
    results = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_metrics': [],
        'reconstructions': [],
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'losses_gradient_ascent': [],
        'training_metadata': None  # Add training metadata
    }

    # Initialize wandb for training mode (now that run_dir is available)
    if wandb_settings.get('enabled', False):
        # Don't override WANDB_PROJECT_NAME - let it be set by the sweep or environment
        wandb_logger = init_wandb_for_mode('train', run_dir)
        if wandb_logger:
            logger.info(f"[OK] Wandb logging enabled: {wandb_logger.run.name}")
        else:
            logger.info("[WARNING] Wandb initialization failed, continuing without wandb")

    logger.info("Generating and preparing data...")
    print("Generating and preparing data...")

    if INFINITE_DATALOADER:
        logger.info("INFINITE_DATALOADER = True")
        print("INFINITE_DATALOADER = True")
    else:
        logger.info("INFINITE_DATALOADER = False")
        print("INFINITE_DATALOADER = False")

    dataloader = None

    if is_multi_encoder:
        logger.info(f"Multi-encoder training enabled with {NUM_ENCODERS} encoders")
        print(f"Multi-encoder training enabled with {NUM_ENCODERS} encoders")
        
        if SPLIT_ACROSS_ENCODERS:
            # ---------------------- KEY-BASED SPLITTING ----------------------
            logger.info("Using key-based dataset splitting for multi-encoder training (split_across_encoders = True)")
            print("Using key-based dataset splitting for multi-encoder training (split_across_encoders = True)")
            if INFINITE_DATALOADER:
                encoder_dataloaders = []
                key_to_encoder_mapping = {}
                tasks_dir = os.path.join(os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
                
                # Use proper key-based splitting instead of round-robin
                num_keys = len(TRAINING_KEYS)
                keys_per_encoder = num_keys // NUM_ENCODERS
                remaining_keys = num_keys % NUM_ENCODERS
                
                key_idx = 0
                for enc_idx in range(NUM_ENCODERS):
                    # Calculate how many keys this encoder should get
                    keys_for_this_encoder = keys_per_encoder + (1 if enc_idx < remaining_keys else 0)
                    
                    # Assign keys to this encoder
                    for _ in range(keys_for_this_encoder):
                        if key_idx < len(TRAINING_KEYS):
                            key = TRAINING_KEYS[key_idx]
                    key_to_encoder_mapping.setdefault(enc_idx, []).append(key)
                            key_idx += 1
                
                # Create dataloaders for each encoder
                for enc_idx in range(NUM_ENCODERS):
                    enc_keys = key_to_encoder_mapping.get(enc_idx, [])
                    if enc_keys:
                        dl = create_infinite_dataloader(enc_keys, BATCH_SIZE, BATCHES_PER_EPOCH,
                                                        seed=data_settings['training_seed'], data_dir=tasks_dir)
                        encoder_dataloaders.append(dl)
                        logger.info(f"Encoder {enc_idx}: infinite loader for keys {enc_keys}")
                        print(f"Encoder {enc_idx}: infinite loader for keys {enc_keys}")
                    else:
                        encoder_dataloaders.append(None)
                        logger.info(f"Encoder {enc_idx}: No keys assigned")
                        print(f"Encoder {enc_idx}: No keys assigned")

                training_metadata = {
                    'training_keys': TRAINING_KEYS,
                    'num_encoders': NUM_ENCODERS,
                    'split_across_encoders': True,
                    'infinite_dataloader': True,
                    'key_to_encoder_mapping': key_to_encoder_mapping
                }
            else:
                dataset_splits, key_to_encoder_mapping, splitting_statistics = split_dataset_by_keys_for_multi_encoder(
                    TRAINING_KEYS, NUM_ENCODERS, N_EXAMPLES_PER_TASK, generate_and_process_tasks
                )

                training_metadata = {
                    'key_to_encoder_mapping': key_to_encoder_mapping,
                    'splitting_statistics': splitting_statistics,
                    'training_keys': TRAINING_KEYS,
                    'num_encoders': NUM_ENCODERS,
                    'split_across_encoders': True
                }

                encoder_dataloaders = []
                for i, (enc_inputs, enc_outputs) in enumerate(dataset_splits):
                    if enc_inputs and enc_outputs:
                        dataloader = prepare_dataloader(enc_inputs, enc_outputs, BATCH_SIZE)
                        encoder_dataloaders.append(dataloader)

                        encoder_keys = splitting_statistics['keys_per_encoder'][i]
                        logger.info(f"Encoder {i}: {len(enc_inputs)} samples from keys {encoder_keys}")
                        print(f"Encoder {i}: {len(enc_inputs)} samples from keys {encoder_keys}")
                    else:
                        encoder_dataloaders.append(None)
                        logger.info(f"Encoder {i}: No data assigned")
                        print(f"Encoder {i}: No data assigned")
            
        else:
            # ---------------------- MIXED DATASET (NO SPLIT) ------------------
            logger.info("split_across_encoders = False → using mixed dataset for all encoders")
            print("split_across_encoders = False → using mixed dataset for all encoders")
            
            if INFINITE_DATALOADER:
                tasks_dir = os.path.join(os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
                mixed_dataloader = create_infinite_dataloader(
                    TRAINING_KEYS, BATCH_SIZE, BATCHES_PER_EPOCH,
                    seed=data_settings['training_seed'], data_dir=tasks_dir)
                encoder_dataloaders = [mixed_dataloader for _ in range(NUM_ENCODERS)]

                training_metadata = {
                    'training_keys': TRAINING_KEYS,
                    'num_encoders': NUM_ENCODERS,
                    'split_across_encoders': False,
                    'infinite_dataloader': True
                }
            else:
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

        if INFINITE_DATALOADER:
            tasks_dir = os.path.join(os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
            dataloader = create_infinite_dataloader(
                TRAINING_KEYS, BATCH_SIZE, BATCHES_PER_EPOCH,
                seed=data_settings['training_seed'], data_dir=tasks_dir)
            training_metadata = {
                'training_keys': TRAINING_KEYS,
                'num_encoders': 1,
                'single_encoder_training': True,
                'infinite_dataloader': True
            }
            input_sequences = output_sequences = None
            encoder_dataloaders = [dataloader]
            # For infinite dataloader, we can't pre-compute sequences
            results['input_sequences'] = None
            results['output_sequences'] = None
            results['key_list'] = None
        else:
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
            encoder_dataloaders = [dataloader]

    logger.info("Initializing model...")
    print("Initializing model...")
    model = build_model(device, wandb_logger, global_step=0)
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

    
    results['training_metadata'] = training_metadata
    
    # Store training metadata in model for later use in plotting
    model.training_metadata = training_metadata
    
    # Store training sequences for visualization (both single and multi-encoder)
    if not is_multi_encoder and not INFINITE_DATALOADER:
        # --- Track key for each sample ---
        key_list = []
        for task_key in TRAINING_KEYS:
            n_for_key = N_EXAMPLES_PER_TASK
            key_list.extend([task_key] * n_for_key)
        key_list.extend([TRAINING_KEYS[-1]] * (len(input_sequences) - len(key_list)))
        results['input_sequences'] = [seq.tolist() for seq in input_sequences]
        results['output_sequences'] = [seq.tolist() for seq in output_sequences]
        results['key_list'] = key_list
        # Collect and save training latent data with keys
        from utils.model_utils import collect_latent_data
        dataloader_for_latents = prepare_dataloader(input_sequences, output_sequences, BATCH_SIZE)
        training_latent_data = collect_latent_data(model, dataloader_for_latents, device, encoder_idx=0, max_samples=len(input_sequences), data_type='training', key_list=key_list)
        results['training_latent_data'] = {'encoder_0': training_latent_data}
    else:
        # For multi-encoder, combine all training data for visualization
        if not INFINITE_DATALOADER:
            all_inputs = []
            all_outputs = []
            all_keys = []
            key_lists_per_encoder = splitting_statistics.get('key_lists_per_encoder', {})
            for encoder_idx in range(NUM_ENCODERS):
                enc_inputs, enc_outputs = dataset_splits[encoder_idx]
                if not enc_inputs:
                    continue
                keys_for_enc = key_lists_per_encoder.get(encoder_idx, [])
                all_inputs.extend([seq.tolist() for seq in enc_inputs])
                all_outputs.extend([seq.tolist() for seq in enc_outputs])
                all_keys.extend(list(keys_for_enc))
            results['input_sequences'] = all_inputs
            results['output_sequences'] = all_outputs
            results['key_list'] = all_keys
            print(f"Saved {len(all_inputs)} combined training sequences from {NUM_ENCODERS} encoders for visualization")
            # Collect and save training latent data with keys for each encoder
            from utils.model_utils import collect_latent_data
            results['training_latent_data'] = {}
            start_idx = 0
            for encoder_idx in range(NUM_ENCODERS):
                enc_inputs, enc_outputs = dataset_splits[encoder_idx]
                keys_for_enc = key_lists_per_encoder.get(encoder_idx, [])
                if not enc_inputs:
                    continue
                dataloader_for_latents = prepare_dataloader(enc_inputs, enc_outputs, BATCH_SIZE)
                training_latent_data = collect_latent_data(model, dataloader_for_latents, device, encoder_idx=encoder_idx, max_samples=len(enc_inputs), data_type=f'training_encoder_{encoder_idx}', key_list=keys_for_enc)
                results['training_latent_data'][f'encoder_{encoder_idx}'] = training_latent_data
        else:
            # For infinite dataloader, we can't pre-compute all sequences, but we can store sample sequences for evaluation
            print("[WARNING] Infinite dataloader enabled - storing sample sequences for evaluation")
            
            # Generate sample sequences for evaluation (first few samples from each key)
            sample_inputs = []
            sample_outputs = []
            sample_keys = []
            
            # Generate a few samples from each training key for evaluation
            for task_key in TRAINING_KEYS:
                try:
                    _, _, _, task_input_sequences, task_output_sequences = generate_and_process_tasks(task_key, min(5, N_EXAMPLES_PER_TASK))
                    sample_inputs.extend([seq.tolist() for seq in task_input_sequences])
                    sample_outputs.extend([seq.tolist() for seq in task_output_sequences])
                    sample_keys.extend([task_key] * len(task_input_sequences))
                except Exception as e:
                    logger.error(f"Error generating sample data for task {task_key}: {e}")
                    continue
            
            if sample_inputs:
                results['input_sequences'] = sample_inputs
                results['output_sequences'] = sample_outputs
                results['key_list'] = sample_keys
                print(f"Saved {len(sample_inputs)} sample training sequences for evaluation")
            else:
                # Fallback: set empty sequences to force dataloader-based approach in plotting
                results['input_sequences'] = None
                results['output_sequences'] = None
                results['key_list'] = None
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
                        if encoder_dataloader is None:
                            continue # Skip encoders with no data
                        for batch in encoder_dataloader:
                            # Handle batch structure with keys
                            if len(batch) >= 3:
                                batch_input, batch_output, batch_keys = batch[:3]
                            else:
                                batch_input, batch_output = batch[:2]
                                batch_keys = None
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
                dataloader = combined_dataloader

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
                    dataloader = encoder_dataloaders[encoder_idx]
                    if dataloader is None:
                        continue # Skip encoders with no data
                    logger.info(f"\n--- Training Encoder {encoder_idx} ---")
                    print(f"Training Encoder {encoder_idx}...")
                    
                    avg_loss, avg_shape_loss, avg_grid_loss, avg_kl_loss, avg_repulsion_loss, current_lambda_rep = train_model(
                        model, dataloader, optimizer, run_dir, logger, 
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
            dataloader = encoder_dataloaders[0] # Assuming single encoder training uses the first dataloader
            if dataloader is None:
                logger.warning("Skipping single encoder training as no data is assigned.")
                print("Skipping single encoder training as no data is assigned.")
                continue # Skip this epoch if no data

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
        
        # --- COMPREHENSIVE EVALUATION AFTER EACH EPOCH ---
        if dataloader is not None:
            input_seqs = results.get('input_sequences', None)
            output_seqs = results.get('output_sequences', None)
            key_list = results.get('key_list', None)
            
            # Run comprehensive evaluation that implements all three requirements:
            # 1. Plot training latents on t-SNE (colored by key + by encoder)
            # 2. Evaluate with sample-level optimization, generating trajectory figures
            # 3. Plot latent space with support and query samples colored by keys
            try:
                comprehensive_evaluation_after_epoch(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    epoch=epoch,
                    run_dir=run_dir,
                    wandb_logger=wandb_logger,
                    training_keys=TRAINING_KEYS,
                    input_sequences=input_seqs,
                    output_sequences=output_seqs,
                    key_list=key_list,
                    is_multi_encoder=is_multi_encoder,
                    num_encoders=NUM_ENCODERS,
                    infinite_dataloader=INFINITE_DATALOADER,
                    encoder_dataloaders=encoder_dataloaders
                )
                print(f"[OK] Comprehensive evaluation completed for epoch {epoch+1}")
                
            except Exception as e:
                logger.error(f"Error during comprehensive evaluation at epoch {epoch+1}: {e}")
                print(f"[WARNING] Error during comprehensive evaluation at epoch {epoch+1}: {e}")
                
                # Fallback to original training latent space visualization
                try:
                plot_training_latent_space_per_epoch(
                    model, dataloader, device, epoch, run_dir, key_list, wandb_logger,
                    infinite_dataloader=INFINITE_DATALOADER, 
                    max_batches=10,
                    input_sequences=input_seqs,
                    output_sequences=output_seqs,
                    training_keys=TRAINING_KEYS,
                    upload_slider=True
                )
                except Exception as fallback_e:
                    logger.error(f"Fallback visualization also failed: {fallback_e}")
                    print(f"[WARNING] Fallback visualization also failed: {fallback_e}")
        
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
                dataloader = encoder_dataloaders[eval_encoder_idx]
                if dataloader is None:
                    continue # Skip encoders with no data
                logger.info(f"\n--- Evaluating Encoder {eval_encoder_idx} ---")
                print(f"Evaluating Encoder {eval_encoder_idx}...")
                
                # Use the specific encoder's dataloader for individual evaluation
                encoder_accuracy = evaluate_accuracy(
                    model, dataloader, device, 
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
                log_dict = {
                    'avg_shape_loss': avg_shape_loss,
                    'avg_grid_loss': avg_grid_loss,
                    'avg_kl_loss': avg_kl_loss,
                    'avg_total_loss': avg_loss
                }
                
                # Add VQ-VAE metrics if enabled
                if hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae():
                    vq_metrics = model.get_vq_metrics()
                    if vq_metrics:
                        log_dict.update({
                            'vq_codebook_perplexity': vq_metrics.get('codebook_perplexity', 0.0),
                            'vq_num_embeddings': vq_metrics.get('num_embeddings', 0),
                        })
                        # Log codebook usage histogram
                        codebook_usage = vq_metrics.get('codebook_usage', None)
                        if codebook_usage is not None:
                            log_dict['vq_codebook_usage_entropy'] = -torch.sum(codebook_usage * torch.log(codebook_usage + 1e-10)).item()
                            log_dict['vq_codebook_usage_max'] = torch.max(codebook_usage).item()
                            log_dict['vq_codebook_usage_min'] = torch.min(codebook_usage).item()
                
                wandb_logger.log_training_metrics(epoch + 1, log_dict)
                wandb_logger.log_accuracy_metrics(epoch + 1, single_accuracy)

        # Note: Comprehensive evaluation is now done after each epoch above
        # The old interval-based evaluation has been replaced with the comprehensive evaluation
        # that runs after every epoch and includes all three requirements:
        # 1. Training latent space plots (colored by key + encoder)
        # 2. Sample-level optimization with trajectory plots
        # 3. Evaluation latent space plots (support + query samples colored by keys)

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

def comprehensive_evaluation_after_epoch(model, dataloader, device, epoch, run_dir, wandb_logger, 
                                       training_keys, input_sequences=None, output_sequences=None, 
                                       key_list=None, is_multi_encoder=False, num_encoders=1, 
                                       infinite_dataloader=False, encoder_dataloaders=None):
    """
    Comprehensive evaluation after each epoch that implements all three requirements:
    1. Plot training latents on t-SNE (colored by key + by encoder) 
    2. Evaluate with sample-level optimization, generating trajectory figures
    3. Plot latent space with support and query samples colored by keys
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
            from evaluation import main_test
    from utils.visualizers import create_standalone_latent_space_plot
    from utils.latent_functions import optimize_latent_z
    
    print(f"\n=== COMPREHENSIVE EVALUATION FOR EPOCH {epoch+1} ===")
    
    training_settings = settings.get_training_settings()
    batches_per_epoch = training_settings.get('batches_per_epoch', 10)
    batch_size = training_settings.get('batch_size', 4)
    samples_per_epoch = batches_per_epoch * batch_size
    print(f"  Dataloader config: {batches_per_epoch} batches x {batch_size} samples = {samples_per_epoch} samples per epoch")

    # 1. PLOT TRAINING LATENTS ON T-SNE (COLORED BY KEY + BY ENCODER)
    print("1. Plotting training latents on t-SNE...")
    
    # Collect training latents from all encoders
    all_training_latents = []
    all_training_keys = []
    all_encoder_indices = []
    
    model.eval()
    with torch.no_grad():
        if is_multi_encoder:
            # Multi-encoder: collect from each encoder using encoder-specific dataloaders
            for encoder_idx in range(num_encoders):
                print(f"  Collecting latents from Encoder {encoder_idx}...")
                
                # Use encoder-specific dataloader if available
                if encoder_dataloaders and encoder_idx < len(encoder_dataloaders):
                    encoder_dataloader = encoder_dataloaders[encoder_idx]
                    if encoder_dataloader is None:
                        print(f"    [WARNING] No dataloader for Encoder {encoder_idx}, skipping...")
                        continue
                else:
                    # Fallback to shared dataloader
                    encoder_dataloader = dataloader
                    print(f"    [WARNING] Using shared dataloader for Encoder {encoder_idx}")
                
                # Process training data for this encoder
                sample_count = 0
                # Use actual dataloader configuration for infinite dataloader
                if infinite_dataloader:
                    # Use the configuration already loaded from settings
                    max_samples_per_encoder = samples_per_epoch
                    print(f"    [INFO] Using infinite dataloader config: {batches_per_epoch} batches × {batch_size} batch_size = {max_samples_per_encoder} samples per encoder")
                else:
                    max_samples_per_encoder = float('inf')
                
                for batch in encoder_dataloader:
                    # Check if we've reached the limit for infinite dataloader
                    if infinite_dataloader and sample_count >= max_samples_per_encoder:
                        print(f"    [INFO] Reached limit of {max_samples_per_encoder} samples for infinite dataloader")
                        break
                    
                    # Handle batch structure with keys
                    if len(batch) >= 3:
                        batch_input, batch_target, batch_keys = batch[:3]
                    else:
                        batch_input, batch_target = batch[:2]
                        batch_keys = None
                    
                    batch_input = batch_input.to(device)
                    batch_target = batch_target.to(device)
                    
                    # Get latent representations from specific encoder
                    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                        mu, log_var = model(batch_input, batch_target, encoder_idx=encoder_idx)[1:3]
                    else:
                        mu, log_var, _ = model.encoder(batch_input, batch_target)
                    
                    z = model.reparameterize(mu, log_var)
                    
                    # Store latents and metadata for ALL samples in batch
                    batch_size = batch_input.size(0)
                    for i in range(batch_size):
                        all_training_latents.append(z[i].cpu().numpy())
                        if batch_keys is not None and i < len(batch_keys):
                            all_training_keys.append(batch_keys[i])
                        else:
                            all_training_keys.append('unknown')
                        all_encoder_indices.append(encoder_idx)
                    
                    sample_count += batch_size
                
                print(f"    [OK] Collected {sample_count} samples from Encoder {encoder_idx}")
                
                # Debug: Show the keys collected for this encoder
                if sample_count > 0:
                    encoder_keys = [key for key, enc_idx in zip(all_training_keys[-sample_count:], all_encoder_indices[-sample_count:]) if enc_idx == encoder_idx]
                    unique_keys = list(set(encoder_keys))
                    print(f"    [DEBUG] Encoder {encoder_idx} collected keys: {unique_keys[:5]}{'...' if len(unique_keys) > 5 else ''} ({len(unique_keys)} unique keys)")
        else:
            # Single encoder
            print("  Collecting latents from single encoder...")
            sample_count = 0
            # Use actual dataloader configuration for infinite dataloader
            if infinite_dataloader:
                # Use the configuration already loaded from settings
                max_samples = samples_per_epoch
                print(f"    [INFO] Using infinite dataloader config: {batches_per_epoch} batches × {batch_size} batch_size = {max_samples} samples for single encoder")
            else:
                max_samples = float('inf')
            
            for batch in dataloader:
                # Check if we've reached the limit for infinite dataloader
                if infinite_dataloader and sample_count >= max_samples:
                    print(f"    [INFO] Reached limit of {max_samples} samples for infinite dataloader")
                    break
                
                # Handle batch structure with keys
                if len(batch) >= 3:
                    batch_input, batch_target, batch_keys = batch[:3]
                else:
                    batch_input, batch_target = batch[:2]
                    batch_keys = None
                
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                # Get latent representations
                mu, log_var, _ = model.encoder(batch_input, batch_target)
                z = model.reparameterize(mu, log_var)
                
                # Store latents and metadata for ALL samples in batch
                batch_size = batch_input.size(0)
                for i in range(batch_size):
                    all_training_latents.append(z[i].cpu().numpy())
                    if batch_keys is not None and i < len(batch_keys):
                        all_training_keys.append(batch_keys[i])
                    else:
                        all_training_keys.append('unknown')
                    all_encoder_indices.append(0)  # Single encoder = index 0
                
                sample_count += batch_size
            
            print(f"    [OK] Collected {sample_count} samples from single encoder")
    
    # Create t-SNE visualization of training latents
    if all_training_latents:
        all_training_latents = np.array(all_training_latents)
        print(f"Training latents shape: {all_training_latents.shape}")
        
        # Apply t-SNE
        print("  Applying t-SNE to training latents...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_training_latents)//4))
        tsne_coords = tsne.fit_transform(all_training_latents)
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        # Get unique keys and encoders for color mapping
        unique_keys = sorted(list(set(all_training_keys)))
        unique_encoders = sorted(list(set(all_encoder_indices)))
        
        # Create color maps
        key_colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
        encoder_markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+']
        
        # Plot each point
        for i, (coord, key, encoder_idx) in enumerate(zip(tsne_coords, all_training_keys, all_encoder_indices)):
            color = key_colors[key]
            marker = encoder_markers[encoder_idx % len(encoder_markers)]
            plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.7, 
                       marker=marker, edgecolors='k', linewidths=0.5)
        
        # Create legend
        legend_elements = []
        for key in unique_keys:
            color = key_colors[key]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=8, label=f'Key: {key[:8]}'))
        
        # Only add encoder legend if multi-encoder
        if is_multi_encoder and len(unique_encoders) > 1:
            for encoder_idx in unique_encoders:
                marker = encoder_markers[encoder_idx % len(encoder_markers)]
                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                                markersize=8, label=f'Encoder {encoder_idx}'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        sample_text = "Sample of Training Data" if infinite_dataloader else "ALL Training Samples"
        plt.title(f'Training Latent Space - Epoch {epoch+1}\n({sample_text} - Colored by Key, Markers by Encoder)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Save plot
        plot_path = os.path.join(run_dir, 'latent_space_plots', f'training_latent_space_epoch_{epoch+1}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Training latent space plot saved: {plot_path}")
        
        # Upload to wandb
        if wandb_logger:
            try:
                import wandb
                wandb_logger._safe_log({
                    f'training_latent_space_epoch_{epoch+1}': wandb.Image(plot_path)
                }, step_hint=epoch+1)
                print(f"  [OK] Training latent space plot uploaded to wandb")
            except Exception as e:
                print(f"  [WARNING] Could not upload training latent space plot to wandb: {e}")
        
        # Create encoder-specific training plot (showing which tasks were used by which encoder)
        # Generate this plot for all models to show task assignment
        print(f"  DEBUG: is_multi_encoder={is_multi_encoder}, num_encoders={num_encoders}")
        if True:  # Always generate encoder-specific plot
            print("  Creating encoder-specific training plot...")
            
            # Group tasks by encoder - use intended key assignment instead of actual sampled keys
            encoder_task_mapping = {}
            
            # Try to get the intended key assignment from training metadata
            if hasattr(model, 'training_metadata') and model.training_metadata and 'key_to_encoder_mapping' in model.training_metadata:
                # Use the intended key assignment from training
                intended_mapping = model.training_metadata['key_to_encoder_mapping']
                for encoder_idx, keys in intended_mapping.items():
                    encoder_task_mapping[encoder_idx] = keys
                print(f"    DEBUG: Using intended key assignment: {encoder_task_mapping}")
            else:
                # Fallback to actual sampled keys
                for key, encoder_idx in zip(all_training_keys, all_encoder_indices):
                    if encoder_idx not in encoder_task_mapping:
                        encoder_task_mapping[encoder_idx] = []
                    if key not in encoder_task_mapping[encoder_idx]:
                        encoder_task_mapping[encoder_idx].append(key)
                print(f"    DEBUG: Using actual sampled keys (fallback): {encoder_task_mapping}")
            
            # Create visualization
            plt.figure(figsize=(16, 12))
            
            # Create color map for encoders (using more visible colors)
            encoder_colors = {enc: plt.cm.tab10(enc % 10) for enc in unique_encoders}
            
            # Plot each point colored by encoder
            for i, (coord, key, encoder_idx) in enumerate(zip(tsne_coords, all_training_keys, all_encoder_indices)):
                color = encoder_colors[encoder_idx]
                plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.7, 
                           edgecolors='k', linewidths=0.5)
            
            # Create legend
            legend_elements = []
            for encoder_idx in unique_encoders:
                color = encoder_colors[encoder_idx]
                tasks = encoder_task_mapping.get(encoder_idx, [])
                task_list = ', '.join([t[:4] for t in tasks[:3]])  # Show first 3 tasks
                if len(tasks) > 3:
                    task_list += f'... (+{len(tasks)-3} more)'
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                                markersize=8, label=f'Encoder {encoder_idx}: {task_list}'))
            
            # Add debug information about key distribution
            print(f"    DEBUG: Legend shows {len(encoder_task_mapping)} encoders")
            for enc_idx, tasks in encoder_task_mapping.items():
                print(f"    DEBUG: Encoder {enc_idx}: {len(tasks)} keys assigned")
                if len(tasks) > 0:
                    print(f"    DEBUG: Encoder {enc_idx} keys: {tasks[:5]}{'...' if len(tasks) > 5 else ''}")
            
            # Debug: Compare intended vs actual key collection
            if hasattr(model, 'training_metadata') and model.training_metadata and 'key_to_encoder_mapping' in model.training_metadata:
                intended_mapping = model.training_metadata['key_to_encoder_mapping']
                print(f"    DEBUG: Intended key assignment:")
                for enc_idx, intended_keys in intended_mapping.items():
                    actual_keys = [key for key, enc_idx_actual in zip(all_training_keys, all_encoder_indices) if enc_idx_actual == enc_idx]
                    actual_unique = list(set(actual_keys))
                    print(f"      Encoder {enc_idx}:")
                    print(f"        Intended: {len(intended_keys)} keys {intended_keys[:3]}{'...' if len(intended_keys) > 3 else ''}")
                    print(f"        Actual: {len(actual_unique)} keys {actual_unique[:3]}{'...' if len(actual_unique) > 3 else ''}")
                    if set(intended_keys) != set(actual_unique):
                        print(f"        ⚠️ MISMATCH: Intended and actual keys differ!")
                    else:
                        print(f"        ✅ MATCH: Intended and actual keys match")
            
            plt.legend(handles=legend_elements, loc='upper right')
            sample_text = "Sample of Training Data" if infinite_dataloader else "ALL Training Samples"
            plt.title(f'Encoder-Specific Training Latent Space - Epoch {epoch+1}\n({sample_text} - Colored by Encoder, Tasks Listed in Legend)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            
            # Save plot
            encoder_plot_path = os.path.join(run_dir, 'latent_space_plots', f'encoder_training_latent_space_epoch_{epoch+1}.png')
            os.makedirs(os.path.dirname(encoder_plot_path), exist_ok=True)
            plt.savefig(encoder_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [OK] Encoder-specific training latent space plot saved: {encoder_plot_path}")
            
            # Upload to wandb
            if wandb_logger:
                try:
                    import wandb
                    wandb_logger._safe_log({
                        f'encoder_training_latent_space_epoch_{epoch+1}': wandb.Image(encoder_plot_path)
                    }, step_hint=epoch+1)
                    print(f"  [OK] Encoder-specific training latent space plot uploaded to wandb")
                except Exception as e:
                    print(f"  [WARNING] Could not upload encoder-specific training latent space plot to wandb: {e}")
        else:
            print("  [WARNING] Skipping encoder-specific training plot")
    
    # 2. EVALUATE WITH SAMPLE-LEVEL OPTIMIZATION, GENERATING TRAJECTORY FIGURES
    print("2. Running evaluation with sample-level optimization...")
    
    # Use subset of training keys for evaluation
    eval_keys = training_keys[:2] if len(training_keys) >= 2 else training_keys

    # Get correct evaluation settings from configuration
    evaluation_settings = settings.get_evaluation_settings()
    n_samples = evaluation_settings.get('eval_n_samples', 2)  # Number of support samples per key
    n_queries = evaluation_settings.get('eval_n_queries', 10)  # Number of query samples per key

    print(f"  Using evaluation settings: {n_samples} support samples, {n_queries} query samples per key")
            
            try:
                # Run evaluation using original Bonnet approach
                eval_results = main_test(
                    model=model,
                    keys=eval_keys,
                    run_dir=run_dir,
                    n_samples=n_samples,
                    n_queries=n_queries,
            seed=42,  # Use fixed seed for reproducibility
                    device=device,
                    encoder_idx=None,  # Use PoE inference
                    use_independent_decoder=False,  # Use shared decoder
                    use_task_optimization=False,  # Disable task-level optimization
                    use_original_bonnet=True  # Enable original Bonnet approach
                )
                
                if eval_results and 'key_results' in eval_results:
            print("  [OK] Evaluation completed successfully")
            
            # Generate trajectory plots for each key
                    for key, key_results in eval_results['key_results'].items():
                if 'trajectory_info' in key_results and key_results['trajectory_info']:
                    print(f"  Generating trajectory plot for key '{key}'...")
                    
                    # Create standalone latent space plot for trajectory
                    trajectory_info = key_results['trajectory_info']
                    
                    # Debug: Print trajectory info structure
                    print(f"    DEBUG: Trajectory info keys: {list(trajectory_info.keys())}")
                    print(f"    DEBUG: Trajectory info type: {type(trajectory_info)}")
                    
                    # Extract the actual trajectory data from the nested structure
                    if key in trajectory_info:
                        actual_trajectory = trajectory_info[key]
                        print(f"    DEBUG: Found trajectory for key '{key}'")
                        print(f"    DEBUG: Actual trajectory keys: {list(actual_trajectory.keys())}")
                        if 'z_vectors' in actual_trajectory:
                            print(f"    DEBUG: z_vectors length: {len(actual_trajectory['z_vectors'])}")
                        if 'losses' in actual_trajectory:
                            print(f"    DEBUG: losses length: {len(actual_trajectory['losses'])}")
                        
                        # Create trajectory plot with the actual trajectory data
                        trajectory_plot_path = create_standalone_latent_space_plot(
                            trajectory_info=actual_trajectory,
                            model=model,
                        save_dir=run_dir,
                        epoch=epoch,
                            sample_idx=0,  # First sample
                            evaluated_key=key,
                            device=device,
                            wandb_logger=wandb_logger,
                            eval_results=eval_results
                        )
                        
                        if trajectory_plot_path:
                            print(f"    [OK] Trajectory plot saved: {trajectory_plot_path}")
                            
                            # Upload to wandb
                            if wandb_logger:
                                try:
                                    import wandb
                            wandb_logger._safe_log({
                                        f'trajectory_plot_{key}_epoch_{epoch+1}': wandb.Image(trajectory_plot_path)
                            }, step_hint=epoch+1)
                                    print(f"    [OK] Trajectory plot uploaded to wandb")
                                except Exception as e:
                                    print(f"    [WARNING] Could not upload trajectory plot to wandb: {e}")
                        else:
                            print(f"    [WARNING] Failed to create trajectory plot for key '{key}'")
                            
                        # Generate main reconstruction plot with trajectory
                        print(f"    Generating main reconstruction plot with trajectory for key '{key}'...")
                        try:
                            from LPN_reproduction.evaluate_trajectory import visualize_comprehensive_trajectory
                            
                            # Create main reconstruction plot path
                            main_reconstruction_path = os.path.join(run_dir, 'trajectory_plots', f'main_reconstruction_{key}_epoch_{epoch+1}.png')
                            os.makedirs(os.path.dirname(main_reconstruction_path), exist_ok=True)
                            
                            # Debug what's in the trajectory_info before passing to visualization
                            print(f"    DEBUG: actual_trajectory keys: {list(actual_trajectory.keys())}")
                            if 'poe_trajectory_reconstructions' in actual_trajectory:
                                print(f"    DEBUG: poe_trajectory_reconstructions keys: {list(actual_trajectory['poe_trajectory_reconstructions'].keys())}")
                else:
                                print(f"    DEBUG: poe_trajectory_reconstructions NOT FOUND in actual_trajectory")
                            
                            # Generate main reconstruction plot
                            visualize_comprehensive_trajectory(
                                trajectory_info=actual_trajectory,
                                model=model,
                                save_path=main_reconstruction_path,
                                run_dir=run_dir,
                                device=device
                            )
                            
                            if os.path.exists(main_reconstruction_path):
                                print(f"    [OK] Main reconstruction plot saved: {main_reconstruction_path}")
                                
                                # Upload to wandb
                                if wandb_logger:
                                    try:
                                        import wandb
                                        wandb_logger._safe_log({
                                            f'main_reconstruction_{key}_epoch_{epoch+1}': wandb.Image(main_reconstruction_path)
                                        }, step_hint=epoch+1)
                                        print(f"    [OK] Main reconstruction plot uploaded to wandb")
            except Exception as e:
                                        print(f"    [WARNING] Could not upload main reconstruction plot to wandb: {e}")
                else:
                                print(f"    [WARNING] Failed to create main reconstruction plot for key '{key}'")
                        except Exception as e:
                            print(f"    [WARNING] Error generating main reconstruction plot for key '{key}': {e}")
                    else:
                        print(f"    [WARNING] Key '{key}' not found in trajectory_info")
                else:
                    print(f"  [WARNING] No trajectory info found for key '{key}'")
                    
            # Also check for trajectory info in the main eval_results
            if 'trajectory_info' in eval_results and eval_results['trajectory_info']:
                print(f"  Generating trajectory plots from main evaluation results...")
                for key, trajectory_info in eval_results['trajectory_info'].items():
                    print(f"    Generating trajectory plot for key '{key}'...")
                    
                    # Debug: Print trajectory info structure
                    print(f"      DEBUG: Trajectory info keys: {list(trajectory_info.keys())}")
                    print(f"      DEBUG: Trajectory info type: {type(trajectory_info)}")
                    
                    # Extract the actual trajectory data from the nested structure
                    if key in trajectory_info:
                        actual_trajectory = trajectory_info[key]
                        print(f"      DEBUG: Found trajectory for key '{key}'")
                        print(f"      DEBUG: Actual trajectory keys: {list(actual_trajectory.keys())}")
                        if 'z_vectors' in actual_trajectory:
                            print(f"      DEBUG: z_vectors length: {len(actual_trajectory['z_vectors'])}")
                        if 'losses' in actual_trajectory:
                            print(f"      DEBUG: losses length: {len(actual_trajectory['losses'])}")
                        
                        # Create trajectory plot with the actual trajectory data
                        trajectory_plot_path = create_standalone_latent_space_plot(
                            trajectory_info=actual_trajectory,
                            model=model,
                            save_dir=run_dir,
                            epoch=epoch,
                            sample_idx=0,  # First sample
                            evaluated_key=key,
                            device=device,
                            wandb_logger=wandb_logger,
                            eval_results=eval_results
                        )
                        
                        if trajectory_plot_path:
                            print(f"      [OK] Trajectory plot saved: {trajectory_plot_path}")
                            
                            # Upload to wandb
                            if wandb_logger:
                                try:
                                    import wandb
                                    wandb_logger._safe_log({
                                        f'trajectory_plot_{key}_epoch_{epoch+1}': wandb.Image(trajectory_plot_path)
                                    }, step_hint=epoch+1)
                                    print(f"      [OK] Trajectory plot uploaded to wandb")
                                except Exception as e:
                                    print(f"      [WARNING] Could not upload trajectory plot to wandb: {e}")
                        else:
                            print(f"      [WARNING] Failed to create trajectory plot for key '{key}'")
                            
                        # Generate main reconstruction plot with trajectory
                        print(f"      Generating main reconstruction plot with trajectory for key '{key}'...")
                        try:
                            from LPN_reproduction.evaluate_trajectory import visualize_comprehensive_trajectory
                            
                            # Create main reconstruction plot path
                            main_reconstruction_path = os.path.join(run_dir, 'trajectory_plots', f'main_reconstruction_{key}_epoch_{epoch+1}.png')
                            os.makedirs(os.path.dirname(main_reconstruction_path), exist_ok=True)
                            
                            # Generate main reconstruction plot
                            visualize_comprehensive_trajectory(
                                trajectory_info=actual_trajectory,
                                model=model,
                                save_path=main_reconstruction_path,
                                run_dir=run_dir,
                                device=device
                            )
                            
                            if os.path.exists(main_reconstruction_path):
                                print(f"      [OK] Main reconstruction plot saved: {main_reconstruction_path}")
                                
                                # Upload to wandb
    if wandb_logger:
                                    try:
                                        import wandb
                                        wandb_logger._safe_log({
                                            f'main_reconstruction_{key}_epoch_{epoch+1}': wandb.Image(main_reconstruction_path)
                                        }, step_hint=epoch+1)
                                        print(f"      [OK] Main reconstruction plot uploaded to wandb")
                                    except Exception as e:
                                        print(f"      [WARNING] Could not upload main reconstruction plot to wandb: {e}")
                            else:
                                print(f"      [WARNING] Failed to create main reconstruction plot for key '{key}'")
                        except Exception as e:
                            print(f"      [WARNING] Error generating main reconstruction plot for key '{key}': {e}")
                    else:
                        print(f"      [WARNING] Key '{key}' not found in trajectory_info")
        else:
            print("  [WARNING] Evaluation failed or no results returned")
            
    except Exception as e:
        print(f"  [ WARNING ] Error during evaluation: {e}")
    
    # 3. PLOT LATENT SPACE WITH SUPPORT AND QUERY SAMPLES COLORED BY KEYS
    print("3. Plotting evaluation latent space with support and query samples...")
    
    try:
        # Collect evaluation latent data
        all_eval_latents = []
        all_eval_keys = []
        all_sample_types = []  # 'support' or 'query'
        
        if eval_results and 'key_results' in eval_results:
            for key, key_results in eval_results['key_results'].items():
                if 'latent_data' in key_results and key_results['latent_data']:
                    latent_data = key_results['latent_data']
                    
                    # Add support latents
                    if 'support_latents' in latent_data and latent_data['support_latents']:
                        support_latents = np.array(latent_data['support_latents'])
                        support_keys = latent_data['support_keys']
                        
                        all_eval_latents.extend(support_latents)
                        all_eval_keys.extend(support_keys)
                        all_sample_types.extend(['support'] * len(support_latents))
                        
                        print(f"    Added {len(support_latents)} support latents for key '{key}'")
                    
                    # Add query latents
                    if 'query_latents' in latent_data and latent_data['query_latents']:
                        query_latents = np.array(latent_data['query_latents'])
                        query_keys = latent_data['query_keys']
                        
                        all_eval_latents.extend(query_latents)
                        all_eval_keys.extend(query_keys)
                        all_sample_types.extend(['query'] * len(query_latents))
                        
                        print(f"    Added {len(query_latents)} query latents for key '{key}'")
        
        # Create evaluation latent space visualization
        if all_eval_latents:
            all_eval_latents = np.array(all_eval_latents)
            print(f"  Evaluation latents shape: {all_eval_latents.shape}")
            
            # Apply t-SNE
            print("  Applying t-SNE to evaluation latents...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_eval_latents)//4))
            tsne_coords = tsne.fit_transform(all_eval_latents)
            
            # Create visualization
            plt.figure(figsize=(16, 12))
            
            # Get unique keys for color mapping
            unique_keys = sorted(list(set(all_eval_keys)))
            key_colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(unique_keys)}
            
            # Plot support samples
            support_mask = np.array(all_sample_types) == 'support'
            if np.any(support_mask):
                support_coords = tsne_coords[support_mask]
                support_keys_subset = [all_eval_keys[i] for i in range(len(all_eval_keys)) if all_sample_types[i] == 'support']
                
                for i, (coord, key) in enumerate(zip(support_coords, support_keys_subset)):
                    plt.scatter(coord[0], coord[1], color=key_colors[key], s=100, alpha=0.7, 
                               marker='o', edgecolors='k', linewidths=0.5, label=f'Support: {key[:4]}' if i == 0 else "")
            
            # Plot query samples
            query_mask = np.array(all_sample_types) == 'query'
            if np.any(query_mask):
                query_coords = tsne_coords[query_mask]
                query_keys_subset = [all_eval_keys[i] for i in range(len(all_eval_keys)) if all_sample_types[i] == 'query']
                
                for i, (coord, key) in enumerate(zip(query_coords, query_keys_subset)):
                    plt.scatter(coord[0], coord[1], color=key_colors[key], s=150, alpha=0.8, 
                               marker='s', edgecolors='k', linewidths=1.0, label=f'Query: {key[:4]}' if i == 0 else "")
            
            # Create legend
            legend_elements = []
            for key in unique_keys:
                color = key_colors[key]
                # Add support legend element
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                                markersize=8, label=f'Support: {key[:8]}'))
                # Add query legend element  
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                                                markersize=10, label=f'Query: {key[:8]}'))
            
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title(f'Evaluation Latent Space - Epoch {epoch+1}\n(Support: circles, Query: squares)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            
            # Save plot
            plot_path = os.path.join(run_dir, 'latent_space_plots', f'evaluation_latent_space_epoch_{epoch+1}.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [ OK ] Evaluation latent space plot saved: {plot_path}")
            
            # Upload to wandb
    if wandb_logger:
                try:
                    import wandb
                    wandb_logger._safe_log({
                        f'evaluation_latent_space_epoch_{epoch+1}': wandb.Image(plot_path)
                    }, step_hint=epoch+1)
                    print(f"  [ OK ] Evaluation latent space plot uploaded to wandb")
                except Exception as e:
                    print(f"  [ WARNING ] Could not upload evaluation latent space plot to wandb: {e}")
        else:
            print("  [ WARNING ] No evaluation latent data available")
            
    except Exception as e:
        print(f"  [ WARNING ] Error during evaluation latent space plotting: {e}")
    
    print(f"=== COMPREHENSIVE EVALUATION COMPLETED FOR EPOCH {epoch+1} ===")
    return True