#!/usr/bin/env python3
"""
Specialist Training Script for Multi-Encoder Models

Implements 3-phase training approach:
- Phase A: Pre-train each encoder on its key subset individually
- Phase B: Freeze encoders, train shared decoder on mixed data
- Phase C: Joint fine-tuning with reduced encoder learning rate

Usage:
    python train_specialist.py [--phases A,B,C] [--resume_from_phase PHASE]
"""

import torch
from torch.optim import Adam
import json
import numpy as np
import os
import argparse
from tqdm import tqdm

from models.base_model import LatentProgramNetwork, compute_loss
from utils.settings_manager import settings
from re_arc.main import generate_and_process_tasks
from utils.data_preparation import split_dataset_by_keys_for_multi_encoder
from utils.training_helpers import (
    create_mixed_domains_dataloader,
    setup_phase_training,
    create_phase_optimizer,
    save_encoder_checkpoint,
    load_all_encoder_checkpoints,
    save_decoder_checkpoint,
    load_decoder_checkpoint,
    save_full_model_checkpoint,
    save_phase_checkpoint,
    print_parameter_status,
    count_trainable_parameters,
)

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


def build_model(device):
    """Build and return LatentProgramNetwork."""
    return LatentProgramNetwork().to(device)


def train_phase_a_pretraining(model, encoder_datasets, device, logger, wandb_logger, run_dir, phase_epochs=None):
    """
    Phase A: Pre-train each encoder individually on its domain data.
    
    Args:
        model: Multi-encoder model
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to train on
        logger: Logger instance
        wandb_logger: WandB logger
        run_dir: Run directory for saving checkpoints
        phase_epochs: Number of epochs to train each encoder
    
    Returns:
        dict: Training results for Phase A
    """
    logger.info("=" * 80)
    logger.info("PHASE A: ENCODER PRE-TRAINING")
    logger.info("=" * 80)
    
    # Get training settings
    training_settings = settings.get_training_settings()
    specialist_settings = settings.get_specialist_training_settings()
    
    BATCH_SIZE = training_settings['batch_size']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    
    # Use settings for phase epochs if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_a']['epochs']
    
    # Phase A setup
    setup_phase_training(model, 'pretrain')
    print_parameter_status(model, 'pretrain')
    
    # Use gradient accumulation and mixed precision
    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    
    num_encoders = len(encoder_datasets)
    phase_a_results = {
        'encoder_losses': {i: [] for i in range(num_encoders)},
        'encoder_epochs': phase_epochs,
        'total_encoders': num_encoders
    }
    
    for encoder_idx in range(num_encoders):
        inputs, outputs = encoder_datasets[encoder_idx]
        
        if not inputs or not outputs:
            logger.info(f"Encoder {encoder_idx}: No data available, skipping...")
            continue
            
        logger.info(f"\n--- Training Encoder {encoder_idx} ---")
        logger.info(f"Data: {len(inputs)} training samples")
        print(f"Training Encoder {encoder_idx} ({len(inputs)} samples)...")
        
        # Create dataloader for this encoder
        dataloader = prepare_dataloader(inputs, outputs, BATCH_SIZE)
        
        # Create optimizer for this specific encoder
        # Freeze all other encoders
        for other_idx, other_encoder in enumerate(model.multi_encoder.encoders):
            if other_idx != encoder_idx:
                for param in other_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in other_encoder.parameters():
                    param.requires_grad = True
        
        # Freeze decoder for pre-training
        for param in model.multi_encoder.decoder.parameters():
            param.requires_grad = False
        
        optimizer = create_phase_optimizer(model, 'pretrain', LEARNING_RATE)
        
        # Training loop for this encoder
        encoder_losses = []
        
        for epoch in range(phase_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = len(dataloader)
            
            # Progress bar for this encoder's training
            pbar = tqdm(dataloader, desc=f"Encoder {encoder_idx} Epoch {epoch+1}/{phase_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                    # Train only this specific encoder
                    loss = compute_loss(
                        model, input_seq, target_seq, 
                        beta=BETA, encoder_idx=encoder_idx
                    )
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
            
            avg_epoch_loss = epoch_loss / num_batches
            encoder_losses.append(avg_epoch_loss)
            
            logger.info(f"Encoder {encoder_idx} Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
            
            # Log to wandb
            if wandb_logger:
                wandb_logger.log_training_metrics(epoch + 1, {
                    f'phase_a/encoder_{encoder_idx}_loss': avg_epoch_loss,
                    'phase': 'A',
                    'current_encoder': encoder_idx
                })
        
        # Save encoder checkpoint
        encoder_checkpoint_path = save_encoder_checkpoint(model, encoder_idx, run_dir)
        logger.info(f"✓ Encoder {encoder_idx} saved to {encoder_checkpoint_path}")
        
        phase_a_results['encoder_losses'][encoder_idx] = encoder_losses
        
        # Log final encoder performance
        final_loss = encoder_losses[-1] if encoder_losses else float('inf')
        logger.info(f"Encoder {encoder_idx} final loss: {final_loss:.4f}")
        print(f"✓ Encoder {encoder_idx} training complete (final loss: {final_loss:.4f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE A COMPLETE - All encoders pre-trained")
    logger.info("=" * 60)
    
    return phase_a_results


def train_phase_b_decoder(model, encoder_datasets, device, logger, wandb_logger, run_dir, phase_epochs=None):
    """
    Phase B: Freeze encoders, train shared decoder on mixed data.
    
    Args:
        model: Multi-encoder model  
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to train on
        logger: Logger instance
        wandb_logger: WandB logger
        run_dir: Run directory
        phase_epochs: Number of epochs for decoder training
    
    Returns:
        dict: Training results for Phase B
    """
    logger.info("=" * 80)
    logger.info("PHASE B: DECODER TRAINING")
    logger.info("=" * 80)
    
    # Load pre-trained encoders
    logger.info("Loading pre-trained encoders...")
    load_all_encoder_checkpoints(model, run_dir, device)
    
    # Get training settings
    training_settings = settings.get_training_settings()
    specialist_settings = settings.get_specialist_training_settings()
    
    BATCH_SIZE = training_settings['batch_size']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    
    # Use settings for phase epochs if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_b']['epochs']
    
    # Phase B setup - freeze encoders, unfreeze decoder
    setup_phase_training(model, 'decoder')
    print_parameter_status(model, 'decoder')
    
    # Create mixed domains dataloader
    num_encoders = len(encoder_datasets)
    mixed_dataloader = create_mixed_domains_dataloader(
        encoder_datasets, num_encoders, BATCH_SIZE, shuffle=True
    )
    
    logger.info(f"Mixed dataloader created with {len(mixed_dataloader)} batches")
    print(f"Training decoder on mixed data ({len(mixed_dataloader)} batches)...")
    
    # Create optimizer for decoder only
    optimizer = create_phase_optimizer(model, 'decoder', LEARNING_RATE)
    
    # Use gradient accumulation and mixed precision  
    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    
    phase_b_results = {
        'decoder_losses': [],
        'decoder_epochs': phase_epochs
    }
    
    # Training loop for decoder
    for epoch in range(phase_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(mixed_dataloader)
        
        pbar = tqdm(mixed_dataloader, desc=f"Phase B Epoch {epoch+1}/{phase_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (input_seq, target_seq, encoder_indices) in enumerate(pbar):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            encoder_indices = encoder_indices.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                # Use PoE inference (encoder_idx=None) to combine all encoders
                loss = compute_loss(
                    model, input_seq, target_seq,
                    beta=BETA, encoder_idx=None  # PoE inference
                )
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        phase_b_results['decoder_losses'].append(avg_epoch_loss)
        
        logger.info(f"Phase B Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
        
        # Log to wandb
        if wandb_logger:
            wandb_logger.log_training_metrics(epoch + 1, {
                'phase_b/decoder_loss': avg_epoch_loss,
                'phase': 'B'
            })
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == phase_epochs:
            checkpoint_path = save_phase_checkpoint(
                model, optimizer, 'decoder', epoch + 1, avg_epoch_loss, run_dir
            )
            logger.info(f"Phase B checkpoint saved: {checkpoint_path}")
    
    # Save final decoder checkpoint
    decoder_checkpoint_path = save_decoder_checkpoint(model, run_dir)
    logger.info(f"✓ Decoder saved to {decoder_checkpoint_path}")
    
    final_loss = phase_b_results['decoder_losses'][-1] if phase_b_results['decoder_losses'] else float('inf')
    logger.info(f"\n" + "=" * 60)
    logger.info(f"PHASE B COMPLETE - Decoder training finished (final loss: {final_loss:.4f})")
    logger.info("=" * 60)
    
    return phase_b_results


def train_phase_c_joint_finetuning(model, encoder_datasets, device, logger, wandb_logger, run_dir, phase_epochs=None, encoder_lr_mult=None):
    """
    Phase C: Joint fine-tuning with reduced encoder learning rate.
    
    Args:
        model: Multi-encoder model
        encoder_datasets: List of (inputs, outputs) for each encoder
        device: Device to train on  
        logger: Logger instance
        wandb_logger: WandB logger
        run_dir: Run directory
        phase_epochs: Number of epochs for joint fine-tuning
        encoder_lr_mult: Learning rate multiplier for encoders
    
    Returns:
        dict: Training results for Phase C
    """
    logger.info("=" * 80)
    logger.info("PHASE C: JOINT FINE-TUNING")
    logger.info("=" * 80)
    
    # Load all pre-trained components
    logger.info("Loading pre-trained encoders and decoder...")
    load_all_encoder_checkpoints(model, run_dir, device)
    load_decoder_checkpoint(model, run_dir, device)
    
    # Get training settings
    training_settings = settings.get_training_settings()
    specialist_settings = settings.get_specialist_training_settings()
    
    BATCH_SIZE = training_settings['batch_size']
    LEARNING_RATE = training_settings['learning_rate']
    BETA = training_settings['beta']
    
    # Use settings for phase epochs and encoder LR multiplier if not provided
    if phase_epochs is None:
        phase_epochs = specialist_settings['phase_c']['epochs']
    if encoder_lr_mult is None:
        encoder_lr_mult = specialist_settings['phase_c']['encoder_lr_multiplier']
    
    # Phase C setup - unfreeze everything
    setup_phase_training(model, 'joint_ft')
    print_parameter_status(model, 'joint_ft')
    
    # Create mixed domains dataloader
    num_encoders = len(encoder_datasets)
    mixed_dataloader = create_mixed_domains_dataloader(
        encoder_datasets, num_encoders, BATCH_SIZE, shuffle=True
    )
    
    logger.info(f"Joint fine-tuning on mixed data ({len(mixed_dataloader)} batches)...")
    logger.info(f"Encoder LR multiplier: {encoder_lr_mult}")
    print(f"Joint fine-tuning with encoder LR multiplier {encoder_lr_mult}...")
    
    # Create optimizer with different learning rates for encoders and decoder
    optimizer = create_phase_optimizer(model, 'joint_ft', LEARNING_RATE, encoder_lr_mult)
    
    # Print optimizer groups info
    logger.info("Optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        logger.info(f"  Group {i}: {num_params:,} parameters, LR = {group['lr']:.6f}")
    
    # Use gradient accumulation and mixed precision
    use_mixed_precision = training_settings.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    gradient_accumulation_steps = training_settings.get('gradient_accumulation_steps', 1)
    
    phase_c_results = {
        'joint_losses': [],
        'joint_epochs': phase_epochs,
        'encoder_lr_mult': encoder_lr_mult
    }
    
    # Training loop for joint fine-tuning
    for epoch in range(phase_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(mixed_dataloader)
        
        pbar = tqdm(mixed_dataloader, desc=f"Phase C Epoch {epoch+1}/{phase_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (input_seq, target_seq, encoder_indices) in enumerate(pbar):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            encoder_indices = encoder_indices.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                # Use PoE inference for joint training
                loss = compute_loss(
                    model, input_seq, target_seq,
                    beta=BETA, encoder_idx=None  # PoE inference
                )
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        phase_c_results['joint_losses'].append(avg_epoch_loss)
        
        logger.info(f"Phase C Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
        
        # Log to wandb
        if wandb_logger:
            wandb_logger.log_training_metrics(epoch + 1, {
                'phase_c/joint_loss': avg_epoch_loss,
                'phase': 'C'
            })
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == phase_epochs:
            checkpoint_path = save_phase_checkpoint(
                model, optimizer, 'joint_ft', epoch + 1, avg_epoch_loss, run_dir
            )
            logger.info(f"Phase C checkpoint saved: {checkpoint_path}")
    
    # Save final complete model
    final_loss = phase_c_results['joint_losses'][-1] if phase_c_results['joint_losses'] else float('inf')
    final_model_path = save_full_model_checkpoint(model, optimizer, phase_epochs, final_loss, run_dir)
    logger.info(f"✓ Final model saved to {final_model_path}")
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"PHASE C COMPLETE - Joint fine-tuning finished (final loss: {final_loss:.4f})")
    logger.info("=" * 60)
    
    # Save final model as a WandB artifact
    if wandb_logger:
        import wandb
        artifact = wandb.Artifact('final_model', type='model')
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)
        print("✓ Final model uploaded to WandB as an artifact")
    
    return phase_c_results


def run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, phase_name):
    """Run evaluation between training phases."""
    logger.info(f"\n--- Evaluation after {phase_name} ---")
    print(f"Running evaluation after {phase_name}...")
    
    try:
        eval_results = run_quick_evaluation(model, run_dir, epoch=f"{phase_name}_final")
        if eval_results and wandb_logger:
            log_evaluation_to_wandb(eval_results, run_dir, f"{phase_name}_final", wandb_logger)
            logger.info(f"✓ Evaluation results logged to wandb for {phase_name}")
        return eval_results
    except Exception as e:
        logger.warning(f"Evaluation failed after {phase_name}: {e}")
        return None


def main_specialist_training(file_store_name, phases_to_run=None, resume_from_phase=None):
    """
    Main specialist training function implementing 3-phase training.
    
    Args:
        file_store_name: Name for run directory
        phases_to_run: List of phases to run ('A', 'B', 'C') - uses settings default if None
        resume_from_phase: Phase to resume from (if any)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get current settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    repulsion_loss_settings = settings.get_repulsion_loss_settings()
    wandb_settings = settings.get_wandb_settings()
    specialist_settings = settings.get_specialist_training_settings()
    
    # Use specialist settings for defaults
    if phases_to_run is None:
        phases_to_run = specialist_settings['phases_to_run']
    
    evaluation_between_phases = specialist_settings.get('evaluation_between_phases', True)
    
    # Validate multi-encoder configuration
    NUM_ENCODERS = model_architecture.get('num_encoders', 1)
    if NUM_ENCODERS <= 1:
        raise ValueError("Specialist training requires num_encoders > 1. Please update model_architecture settings.")
    
    TRAINING_KEYS = data_settings.get('training_keys', [data_settings.get('key', None)])
    if not TRAINING_KEYS or not TRAINING_KEYS[0]:
        raise ValueError("No training keys specified in data_settings.")
    
    N_EXAMPLES_PER_TASK = data_settings['n']
    
    print(f"Specialist training configuration:")
    print(f"- Number of encoders: {NUM_ENCODERS}")
    print(f"- Training keys: {TRAINING_KEYS}")
    print(f"- Examples per task: {N_EXAMPLES_PER_TASK}")
    print(f"- Phases to run: {phases_to_run}")
    print(f"- Evaluation between phases: {evaluation_between_phases}")
    print(f"- Phase A epochs: {specialist_settings['phase_a']['epochs']}")
    print(f"- Phase B epochs: {specialist_settings['phase_b']['epochs']}")
    print(f"- Phase C epochs: {specialist_settings['phase_c']['epochs']}")
    print(f"- Phase C encoder LR multiplier: {specialist_settings['phase_c']['encoder_lr_multiplier']}")
    
    set_seed(data_settings['training_seed'])
    
    # Create run directory and setup logging
    run_dir = create_run_directory(file_store_name)
    logger = setup_logging(run_dir)
    logger.info(f"Starting specialist training for ARC problems: {TRAINING_KEYS}")
    logger.info(f"Full settings dump: {json.dumps(settings.get_settings(), indent=2)}")
    print("Run directory created:", run_dir)
    
    # Initialize wandb
    wandb_logger = None
    if wandb_settings.get('enabled', False):
        wandb_logger = init_wandb_for_mode('specialist_train', run_dir)
        if wandb_logger:
            logger.info(f"✓ Wandb logging enabled: {wandb_logger.run.name}")
            # Log specialist training configuration
            wandb_logger.log_training_metrics(0, {
                'specialist_training/num_encoders': NUM_ENCODERS,
                'specialist_training/num_keys': len(TRAINING_KEYS),
                'specialist_training/phases_planned': ','.join(phases_to_run)
            })
        else:
            logger.info("⚠ Wandb initialization failed, continuing without wandb")
    
    # Generate and split data for multi-encoder training
    logger.info("Generating and splitting data for specialist training...")
    print("Generating and splitting data...")
    
    dataset_splits, key_to_encoder_mapping, splitting_statistics = split_dataset_by_keys_for_multi_encoder(
        TRAINING_KEYS, NUM_ENCODERS, N_EXAMPLES_PER_TASK, generate_and_process_tasks
    )
    
    # Log data splitting info
    logger.info(f"Data splitting complete:")
    for encoder_idx, (inputs, outputs) in enumerate(dataset_splits):
        encoder_keys = splitting_statistics['keys_per_encoder'][encoder_idx]
        logger.info(f"  Encoder {encoder_idx}: {len(inputs)} samples from keys {encoder_keys}")
    
    # Initialize model first so param_info is available
    logger.info("Initializing multi-encoder model...")
    model = build_model(device)
    param_info = count_model_parameters(model)
    logger.info(f"Model initialized with {param_info['total_params']:,} parameters")
    
    # Collect **all** training sequences (needed for latent visualisation later)
    all_inputs, all_outputs = [], []
    for enc_inputs, enc_outputs in dataset_splits:
        all_inputs.extend(enc_inputs)
        all_outputs.extend(enc_outputs)

    results = {
        'specialist_training': True,
        'phases_completed': [],
        'training_metadata': {
            'key_to_encoder_mapping': key_to_encoder_mapping,
            'splitting_statistics': splitting_statistics,
            'training_keys': TRAINING_KEYS,
            'num_encoders': NUM_ENCODERS,
            'phases_planned': phases_to_run
        },
        'model_parameter_info': param_info,
        # Flatten sequences to plain python lists for pickle safety
        'input_sequences': [seq.tolist() if hasattr(seq, 'tolist') else seq for seq in all_inputs],
        'output_sequences': [seq.tolist() if hasattr(seq, 'tolist') else seq for seq in all_outputs]
    }
    
    # Run phases
    phase_a_epochs = specialist_settings['phase_a']['epochs']
    phase_b_epochs = specialist_settings['phase_b']['epochs']
    phase_c_epochs = specialist_settings['phase_c']['epochs']
    phase_c_encoder_lr_mult = specialist_settings['phase_c']['encoder_lr_multiplier']
    try:
        if 'A' in phases_to_run:
            logger.info("\n" + "=" * 100)
            logger.info("STARTING PHASE A: ENCODER PRE-TRAINING")
            logger.info("=" * 100)
            
            phase_a_results = train_phase_a_pretraining(
                model, dataset_splits, device, logger, wandb_logger, run_dir,phase_a_epochs
            )
            results['phase_a'] = phase_a_results
            results['phases_completed'].append('A')
            
            # Evaluation after Phase A (if enabled)
            if evaluation_between_phases:
                eval_results_a = run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, "Phase A")
                if eval_results_a:
                    results['phase_a']['evaluation'] = eval_results_a
            
            # Add PoE accuracy snapshot to results
            poe_accuracy = run_quick_evaluation(model, run_dir, epoch=f"Phase A Epoch {phase_a_epochs}")
            results['phase_a']['poe_accuracies'] = [poe_accuracy]
            # Save results after each epoch
            save_results(results, run_dir)
        
        if 'B' in phases_to_run:
            logger.info("\n" + "=" * 100)
            logger.info("STARTING PHASE B: DECODER TRAINING")
            logger.info("=" * 100)
            
            phase_b_results = train_phase_b_decoder(
                model, dataset_splits, device, logger, wandb_logger, run_dir,phase_b_epochs
            )
            results['phase_b'] = phase_b_results
            results['phases_completed'].append('B')
            
            # Evaluation after Phase B (if enabled)
            if evaluation_between_phases:
                eval_results_b = run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, "Phase B")
                if eval_results_b:
                    results['phase_b']['evaluation'] = eval_results_b
            
            # Add PoE accuracy snapshot to results
            poe_accuracy = run_quick_evaluation(model, run_dir, epoch=f"Phase B Epoch {phase_b_epochs}")
            results['phase_b']['poe_accuracies'] = [poe_accuracy]
            # Save results after each epoch
            save_results(results, run_dir)
        
        if 'C' in phases_to_run:
            logger.info("\n" + "=" * 100)
            logger.info("STARTING PHASE C: JOINT FINE-TUNING")
            logger.info("=" * 100)
            
            phase_c_results = train_phase_c_joint_finetuning(
                model, dataset_splits, device, logger, wandb_logger, run_dir,phase_c_epochs,phase_c_encoder_lr_mult
            )
            results['phase_c'] = phase_c_results
            results['phases_completed'].append('C')
            
            # Final evaluation after Phase C (if enabled)
            if evaluation_between_phases:
                eval_results_c = run_evaluation_between_phases(model, device, logger, wandb_logger, run_dir, "Phase C")
                if eval_results_c:
                    results['phase_c']['evaluation'] = eval_results_c
            
            # Add PoE accuracy snapshot to results
            poe_accuracy = run_quick_evaluation(model, run_dir, epoch=f"Phase C Epoch {phase_c_epochs}")
            results['phase_c']['poe_accuracies'] = [poe_accuracy]
            # Save results after each epoch
            save_results(results, run_dir)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Save final results
        logger.info("Saving final specialist training results...")
        save_results(results, run_dir)
        
        # Save model parameters
        save_model_params(run_dir, param_info)
        
        if wandb_logger:
            # Log final summary
            wandb_logger.log_training_metrics(-1, {
                'specialist_training/phases_completed': ','.join(results['phases_completed']),
                'specialist_training/final_status': 'completed'
            })
            wandb_logger.finish()
        
        logger.info("=" * 80)
        logger.info("SPECIALIST TRAINING COMPLETE")
        logger.info(f"Phases completed: {results['phases_completed']}")
        logger.info(f"Results saved in: {run_dir}")
        logger.info("=" * 80)
        
        print("\nSpecialist training complete!")
        print(f"Phases completed: {results['phases_completed']}")
        print(f"Results saved in: {run_dir}")
    
    return results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist Multi-Encoder Training")
    parser.add_argument("--phases", type=str, default="A,B,C",
                       help="Comma-separated phases to run (A,B,C)")
    parser.add_argument("--resume_from_phase", type=str, default=None,
                       help="Phase to resume from (A,B,C)")
    parser.add_argument("--run_name", type=str, default="specialist_training",
                       help="Name for the run directory")
    
    args = parser.parse_args()
    
    # Parse phases
    phases_to_run = [p.strip().upper() for p in args.phases.split(',')]
    valid_phases = ['A', 'B', 'C']
    
    if not all(p in valid_phases for p in phases_to_run):
        print(f"Error: Invalid phases. Valid phases are: {valid_phases}")
        exit(1)
    
    print(f"Starting specialist training with phases: {phases_to_run}")
    
    try:
        results, model = main_specialist_training(
            args.run_name, 
            phases_to_run=phases_to_run,
            resume_from_phase=args.resume_from_phase
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 