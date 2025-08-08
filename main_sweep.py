#!/usr/bin/env python3
"""
Hyperparameter Sweep System for Multi-Encoder Latent Program Network

This script runs comprehensive hyperparameter sweeps using WandB, testing different
configurations across model architecture, training settings, and data configurations.
Each run is named sequentially (run_1, run_2, etc.) and configurations are stored
in a dedicated folder for easy reference.
"""

import os
import json
import wandb
import argparse
import subprocess
import shutil
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch

# Import existing modules
from training import main_training
from train_specialist import main_specialist_training  # Import specialist training
from evaluation import main_test
from utils.model_utils import load_model, save_evaluation_results
from utils.visualizers import visualize_stored_results
from utils.settings_manager import settings

# Constants
SWEEP_CONFIG_DIR = "sweep_configs"
WANDB_PROJECT = "LARGE_multiencoder_sweep"
WANDB_OPT_PROJECT = "LPN_opt_sweep"  # Add this new constant
BASE_DIR = "runs_re_arc"

def create_sweep_configurations(specialist_mode: bool = False) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Define comprehensive hyperparameter configurations for the sweep.
    
    Args:
        specialist_mode: If True, create specialist training configurations
        
    Returns:
        List of configuration dictionaries.
    """
    configurations = []
    
    if specialist_mode:
        # Load base configuration from specialist settings
        with open("model_specialist_settings.json", "r") as f:
            base_config = json.load(f)
        
        print("Base config loaded from model_specialist_settings.json for specialist training")
        print(f"  Base num_encoders: {base_config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
        print(f"  Base phase_a epochs: {base_config.get('specialist_training', {}).get('phase_a', {}).get('epochs', 'NOT FOUND')}")
        print(f"  Base phase_b epochs: {base_config.get('specialist_training', {}).get('phase_b', {}).get('epochs', 'NOT FOUND')}")
        print(f"  Base beta: {base_config.get('training_settings', {}).get('beta', 'NOT FOUND')}")
        
        # Different beta values (small values for KL term)
        beta_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        # Different phase epoch combinations
        phase_combinations = [
            (50, 50),   # Short phases
            (75, 75),   # Medium phases  
            (100, 100), # Base phases
            (150, 100), # Longer phase A, standard phase B
            (100, 150), # Standard phase A, longer phase B
            (200, 100), # Much longer phase A
            (100, 200), # Much longer phase B
        ]
        
        # Create configurations for beta sweep
        for beta in beta_values:
            config_beta = copy.deepcopy(base_config)
            config_beta["training_settings"]["beta"] = beta
            config_name = f"specialist_beta_{str(beta).replace('.', '_')}"
            print(f"Config {config_name}: beta={beta}")
            configurations.append((config_name, config_beta))
        
        # Create configurations for phase epoch sweep
        for phase_a_epochs, phase_b_epochs in phase_combinations:
            config_phases = copy.deepcopy(base_config)
            config_phases["specialist_training"]["phase_a"]["epochs"] = phase_a_epochs
            config_phases["specialist_training"]["phase_b"]["epochs"] = phase_b_epochs
            config_name = f"specialist_phaseA_{phase_a_epochs}_phaseB_{phase_b_epochs}"
            print(f"Config {config_name}: phase_a={phase_a_epochs}, phase_b={phase_b_epochs}")
            configurations.append((config_name, config_phases))
        
        # Create configurations combining different beta and phase combinations (selected few)
        selected_combos = [
            (1e-5, 50, 50),  # Low beta, standard phases
            (1e-4, 100, 100),   # Medium beta, longer phase A
        ]
        
        for beta, phase_a_epochs, phase_b_epochs in selected_combos:
            config_combo = copy.deepcopy(base_config)
            config_combo["training_settings"]["beta"] = beta
            config_combo["specialist_training"]["phase_a"]["epochs"] = phase_a_epochs
            config_combo["specialist_training"]["phase_b"]["epochs"] = phase_b_epochs
            config_name = f"specialist_beta_{str(beta).replace('.', '_')}_phaseA_{phase_a_epochs}_phaseB_{phase_b_epochs}"
            print(f"Config {config_name}: beta={beta}, phase_a={phase_a_epochs}, phase_b={phase_b_epochs}")
            configurations.append((config_name, config_combo))
            
    else:
        # Load base configuration from model_settings.json
        with open("model_settings.json", "r") as f:
            base_config = json.load(f)
        
        print("Base config loaded from model_settings.json")
        print(f"  Base num_encoders: {base_config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
        print(f"  Base encoder_layers: {base_config.get('model_architecture', {}).get('encoder_layers', 'NOT FOUND')}")
        print(f"  Base latent_dim: {base_config.get('model_architecture', {}).get('latent_dim', 'NOT FOUND')}")
        
        # Training configurations: 3 different latent dimensions
        training_latent_dims = [64, 256, 512]  # 3 models to train
        
        # Create training configurations
        for latent_dim in training_latent_dims:
            config_train = copy.deepcopy(base_config)
            
            # Update latent dimension for training
            config_train["model_architecture"]["latent_dim"] = latent_dim
            
            # Set default optimization method for training (will be overridden during evaluation)
            config_train["latent_optimization"]["method"] = "gradient"
            
            config_name = f"train_latent_dim_{latent_dim}"
            print(f"Config {config_name}: latent_dim={latent_dim} (training)")
            configurations.append((config_name, config_train))
        
        # Evaluation configurations: Each trained model with both optimization methods
        eval_latent_dims = [64, 256, 512]  # Same as training
        optimization_methods = ['gradient', 'evolutionary']
        
        # Base parameters for gradient ascent
        base_learning_rate = 0.1
        base_steps = 100
        
        for latent_dim in eval_latent_dims:
            for opt_method in optimization_methods:
                config_eval = copy.deepcopy(base_config)
                
                # Set latent dimension (should match trained model)
                config_eval["model_architecture"]["latent_dim"] = latent_dim
                
                # Set optimization method for evaluation
                config_eval["latent_optimization"]["method"] = opt_method
                
                # Configure optimization-specific settings using mathematical relationships
                if opt_method == 'gradient':
                    config_eval["latent_optimization"]["inference"] = {
                        "enabled": True,
                        "num_steps": base_steps,
                        "learning_rate": base_learning_rate
                    }
                elif opt_method == 'evolutionary':
                    # Apply mathematical relationships from cheat-sheet:
                    # σ ≈ √(2η), G ≈ S, λ ≥ c·d/σ²
                    mutation_std = (2 * base_learning_rate) ** 0.5  # σ ≈ √(2η)
                    num_generations = base_steps  # G ≈ S
                    
                    # Population size: λ ≥ c·d/σ² where c≈4-10, d=latent_dim
                    c = 6  # middle value between 4-10
                    min_population = int(c * latent_dim / (mutation_std ** 2))
                    population_size = max(min_population, 25)  # ensure minimum reasonable size
                    
                    config_eval["latent_optimization"]["evolutionary"] = {
                        "population_size": population_size,
                        "num_generations": num_generations,
                        "mutation_std": mutation_std
                    }
                    
                    # Debug print for verification
                    print(f"  Evolutionary params for {latent_dim}D: σ={mutation_std:.3f}, G={num_generations}, λ={population_size}")
                
                config_name = f"eval_latent_dim_{latent_dim}_opt_{opt_method}"
                print(f"Config {config_name}: latent_dim={latent_dim}, optimization={opt_method} (evaluation)")
                configurations.append((config_name, config_eval))

    return configurations

def create_sweep_opt_configurations() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create configurations for optimization method comparison across different latent dimensions.
    
    Returns:
        List of (config_name, config_dict) tuples for training and evaluation runs.
    """
    configurations = []
    
    # Load base configuration
    with open("model_settings.json", "r") as f:
        base_config = json.load(f)
    
    print("Base config loaded from model_settings.json for optimization sweep")
    
    # Reduced latent dimensions to test
    latent_dims = [32, 256, 1024]  # Reduced from [32, 64, 128, 256, 512, 1024]
    
    # Training configurations (one per latent dimension)
    for latent_dim in latent_dims:
        config_train = copy.deepcopy(base_config)
        
        # Update latent dimension
        config_train["model_architecture"]["latent_dim"] = latent_dim
        
        # Use epochs from config (don't override)
        num_epochs = config_train["training_settings"]["num_epochs"]
        print(f"Training config for latent_dim={latent_dim}: using {num_epochs} epochs from config")
        
        # Set checkpoint interval to save at 25%, 50%, 75%, and 100%
        checkpoint_interval = max(1, num_epochs // 4)  # Save every 25% of epochs
        config_train["training_settings"]["save_checkpoint_interval"] = checkpoint_interval
        
        # Set default optimization method (will be overridden during evaluation)
        config_train["latent_optimization"]["method"] = "gradient"
        
        config_name = f"train_latent_dim_{latent_dim}"
        print(f"Training config {config_name}: latent_dim={latent_dim}, epochs={num_epochs}, checkpoint_interval={checkpoint_interval}")
        configurations.append((config_name, config_train))
    
    # Evaluation configurations for each trained model
    optimization_methods = ['gradient', 'evolutionary']
    
    # Base parameters for optimization
    base_learning_rate = 0.1
    base_steps = 50  # Shorter for sweep
    base_generations = 25  # Shorter for sweep
    
    for latent_dim in latent_dims:
        for opt_method in optimization_methods:
            config_eval = copy.deepcopy(base_config)
            
            # Set latent dimension (should match trained model)
            config_eval["model_architecture"]["latent_dim"] = latent_dim
            
            # Set optimization method
            config_eval["latent_optimization"]["method"] = opt_method
            
            # Configure optimization-specific settings
            if opt_method == 'gradient':
                # Reduced step counts: only 25 and 100
                step_counts = [25, 100]  # Reduced from [25, 50, 100]
                
                for steps in step_counts:
                    config_gradient = copy.deepcopy(config_eval)
                    config_gradient["latent_optimization"]["inference"] = {
                        "enabled": True,
                        "num_steps": steps,
                        "learning_rate": base_learning_rate
                    }
                    
                    config_name = f"eval_latent_dim_{latent_dim}_gradient_steps_{steps}"
                    print(f"Evaluation config {config_name}: latent_dim={latent_dim}, gradient, steps={steps}")
                    configurations.append((config_name, config_gradient))
                    
            elif opt_method == 'evolutionary':
                # Apply mathematical relationships for evolutionary parameters
                mutation_std = (2 * base_learning_rate) ** 0.5  # σ ≈ √(2η)
                
                # Reduced population sizes and generations
                # Calculate minimum population size based on latent dimension
                c = 6  # middle value between 4-10
                min_population = int(c * latent_dim / (mutation_std ** 2))
                
                # Reduced population sizes: only 25 and 100
                if latent_dim <= 64:
                    population_sizes = [max(min_population, 25), max(min_population, 100)]  # Reduced from [25, 50, 100]
                elif latent_dim <= 256:
                    population_sizes = [max(min_population, 25), max(min_population, 100)]  # Reduced from [50, 100, 200]
                else:  # 1024
                    population_sizes = [max(min_population, 25), max(min_population, 100)]  # Reduced from [100, 200, 400]
                
                # Reduced generation counts: only 25 and 100
                generation_counts = [25, 100]  # Reduced from [25, 50, 100]
                
                for pop_size in population_sizes:
                    for generations in generation_counts:
                        config_evolutionary = copy.deepcopy(config_eval)
                        config_evolutionary["latent_optimization"]["evolutionary"] = {
                            "population_size": pop_size,
                            "num_generations": generations,
                            "mutation_std": mutation_std
                        }
                        
                        config_name = f"eval_latent_dim_{latent_dim}_evolutionary_pop_{pop_size}_gen_{generations}"
                        print(f"Evaluation config {config_name}: latent_dim={latent_dim}, evolutionary, pop={pop_size}, gen={generations}")
                        configurations.append((config_name, config_evolutionary))
    
    return configurations

def save_configuration(config_name: str, config: Dict[str, Any], run_number: int):
    """Save configuration to the sweep_configs folder."""
    config_path = Path(SWEEP_CONFIG_DIR) / f"run_{run_number}_{config_name}.json"
    config_path.parent.mkdir(exist_ok=True)
    
    # Debug: Print what's being saved
    print(f"Saving config for run {run_number}:")
    print(f"  num_encoders: {config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
    print(f"  encoder_layers: {config.get('model_architecture', {}).get('encoder_layers', 'NOT FOUND')}")
    print(f"  latent_dim: {config.get('model_architecture', {}).get('latent_dim', 'NOT FOUND')}")
    print(f"  optimization_method: {config.get('latent_optimization', {}).get('method', 'NOT FOUND')}")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def update_settings_with_config(config: Dict[str, Any]):
    """Update the global settings with the new configuration by creating a temporary settings file."""
    import tempfile
    import shutil
    
    # Debug: Print what's being applied
    print(f"Applying config to settings:")
    print(f"  num_encoders: {config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
    print(f"  encoder_layers: {config.get('model_architecture', {}).get('encoder_layers', 'NOT FOUND')}")
    print(f"  latent_dim: {config.get('model_architecture', {}).get('latent_dim', 'NOT FOUND')}")
    print(f"  optimization_method: {config.get('latent_optimization', {}).get('method', 'NOT FOUND')}")
    
    # Create a temporary settings file with the new configuration
    temp_settings_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    
    try:
        # Write the new configuration to the temporary file
        json.dump(config, temp_settings_file, indent=2)
        temp_settings_file.close()
        
        # Temporarily replace the settings file
        original_settings_file = settings.settings_file
        backup_settings_file = f"{original_settings_file}.backup"
        
        # Backup original settings
        if os.path.exists(original_settings_file):
            shutil.copy2(original_settings_file, backup_settings_file)
        
        # Replace with new settings
        shutil.copy2(temp_settings_file.name, original_settings_file)
        
        # Reload settings
        settings.load_settings()
        
        # Debug: Verify what was loaded
        print(f"Settings after update:")
        print(f"  num_encoders: {settings.get_model_architecture().get('num_encoders', 'NOT FOUND')}")
        print(f"  encoder_layers: {settings.get_model_architecture().get('encoder_layers', 'NOT FOUND')}")
        print(f"  latent_dim: {settings.get_model_architecture().get('latent_dim', 'NOT FOUND')}")
        print(f"  optimization_method: {settings.get_latent_optimization().get('method', 'NOT FOUND')}")
        
        # Clean up temporary file
        os.unlink(temp_settings_file.name)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_settings_file.name):
            os.unlink(temp_settings_file.name)
        raise e

def restore_original_settings():
    """Restore the original settings file from backup."""
    import shutil
    
    original_settings_file = settings.settings_file
    backup_settings_file = f"{original_settings_file}.backup"
    
    if os.path.exists(backup_settings_file):
        shutil.copy2(backup_settings_file, original_settings_file)
        os.unlink(backup_settings_file)
        settings.load_settings()
        print("[ OK ] Original settings restored")
    else:
        print("[ WARNING ] No backup settings file found to restore")

def generate_run_notes(config_name: str, config: Dict[str, Any], specialist_mode: bool = False) -> str:
    """
    Generate descriptive notes for a sweep run.
    
    Args:
        config_name: Name of the configuration
        config: Configuration dictionary
        specialist_mode: Whether this is a specialist training run
        
    Returns:
        str: Descriptive notes for the WandB run
    """
    notes_parts = [f"Config: {config_name}"]
    
    if specialist_mode:
        # Specialist training notes
        specialist_config = config.get('specialist_training', {})
        phase_a_epochs = specialist_config.get('phase_a', {}).get('epochs', 'N/A')
        phase_b_epochs = specialist_config.get('phase_b', {}).get('epochs', 'N/A')
        beta = config.get('training_settings', {}).get('beta', 'N/A')
        
        notes_parts.extend([
            f"Specialist Training",
            f"Phase A epochs: {phase_a_epochs}",
            f"Phase B epochs: {phase_b_epochs}",
            f"Beta: {beta}"
        ])
    else:
        # Standard training notes
        model_arch = config.get('model_architecture', {})
        training_settings = config.get('training_settings', {})
        latent_opt = config.get('latent_optimization', {})
        
        latent_dim = model_arch.get('latent_dim', 'N/A')
        num_encoders = model_arch.get('num_encoders', 'N/A')
        encoder_layers = model_arch.get('encoder_layers', 'N/A')
        learning_rate = training_settings.get('learning_rate', 'N/A')
        beta = training_settings.get('beta', 'N/A')
        num_epochs = training_settings.get('num_epochs', 'N/A')
        opt_method = latent_opt.get('method', 'N/A')
        
        notes_parts.extend([
            f"Standard Training",
            f"Latent dim: {latent_dim}",
            f"Num encoders: {num_encoders}",
            f"Encoder layers: {encoder_layers}",
            f"Learning rate: {learning_rate}",
            f"Beta: {beta}",
            f"Num epochs: {num_epochs}",
            f"Optimization: {opt_method}"
        ])
        
        # Add repulsion loss info if present
        if 'repulsion_loss_settings' in config:
            repulsion_config = config['repulsion_loss_settings']
            if 'schedule' in repulsion_config:
                start_val = repulsion_config['schedule'].get('start', 'N/A')
                end_val = repulsion_config['schedule'].get('end', 'N/A')
                notes_parts.append(f"Repulsion loss: {start_val} -> {end_val}")
    
    return " | ".join(notes_parts)

def run_single_experiment(run_number: int, config_name: str, config: Dict[str, Any], 
                         modes: List[str], device: str, specialist_mode: bool = False) -> bool:
    """
    Run a single experiment with the given configuration.
    
    Args:
        run_number: Sequential run number
        config_name: Name of the configuration
        config: Configuration dictionary
        modes: List of modes to run (train, eval, visualize)
        device: Device to use (cuda/cpu)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting Run {run_number}: {config_name}")
        print(f"{'='*60}")
        
        # Save configuration
        config_path = save_configuration(config_name, config, run_number)
        print(f"Configuration saved to: {config_path}")
        
        # Update settings
        settings.set_settings(config)
        
        # Create run name
        # Get WandB project name from settings
        wandb_project = config.get("wandb_settings", {}).get("project_name", "default_project")
        run_name = f"{wandb_project}_run_{run_number}"

        # Create run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = BASE_DIR if not specialist_mode else "runs_specialist_sweep"
        run_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created run directory: {run_dir}")

        if 'train' in modes or 'all' in modes:
            print(f"\n--- Training Mode ---")
            print(f"Run directory: {run_dir}")
            
            if specialist_mode:
                print("Using specialist training mode")
                # Extract specialist training parameters
                phases_to_run = config.get('specialist_training', {}).get('phases_to_run', ['A', 'B'])
                
                # Train the model using specialist training
                results, model = main_specialist_training(
                    run_name, 
                    phases_to_run=phases_to_run,
                    resume_from_phase=None,
                    run_dir=run_dir,  # Pass the exact directory
                    notes=generate_run_notes(config_name, config, specialist_mode) # Pass notes
                )
                if results is None:
                    print(f"Specialist training failed for run {run_number}")
                    return False
                
                print(f"Specialist training completed successfully for run {run_number}")
            else:
                print("Using standard training mode")
                # Train the model using standard training
                results, model = main_training(run_name, run_dir=run_dir, notes=generate_run_notes(config_name, config, specialist_mode))  # Pass the exact directory
                if results is None:
                    print(f"Training failed for run {run_number}")
                    return False
                
                print(f"Training completed successfully for run {run_number}")
        
        if 'eval' in modes or 'all' in modes:
            print(f"\n--- Evaluation Mode ---")
            
            # Load the model
            if specialist_mode:
                # For specialist models, use phase-specific epoch naming
                eval_epoch = config.get('evaluation_settings', {}).get('eval_epoch', 'phase_b_final')
            else:
                eval_epoch = config.get('evaluation_settings', {}).get('eval_epoch', 35)
            
            model, _, _, _ = load_model(run_dir, epoch=eval_epoch, device=device)
            
            # Run evaluation
            eval_keys = config.get('evaluation_settings', {}).get('eval_keys', ['00d62c1b'])
            eval_n_samples = config.get('evaluation_settings', {}).get('eval_n_samples', 2)
            eval_n_queries = config.get('evaluation_settings', {}).get('eval_n_queries', 100)
            eval_seed = config.get('data_settings', {}).get('eval_seed', 1)
            
            from utils.evaluation_utils import get_evaluation_keys_with_all_support
            # Handle "all" evaluation keys similar to training keys
            eval_keys = get_evaluation_keys_with_all_support(eval_keys, evaluation_settings.get('n_max_eval_keys', 10))
            eval_results = main_test(model, eval_keys, run_dir, eval_n_samples, 
                                   eval_n_queries, eval_seed, device)
            
            # Save evaluation results
            save_evaluation_results(eval_results, run_dir)
            print(f"Evaluation completed for run {run_number}")
        
        if 'visualize' in modes or 'all' in modes:
            print(f"\n--- Visualization Mode ---")
            
            eval_epoch = config.get('evaluation_settings', {}).get('eval_epoch', 35)
            visualize_stored_results(run_dir, epoch=eval_epoch)
            print(f"Visualization completed for run {run_number}")
        
        print(f"\nRun {run_number} completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in run {run_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always restore original settings
        restore_original_settings()

def run_optimization_evaluation(model, config: Dict[str, Any], run_dir: str, device: str, checkpoint_epoch: str = 'final') -> Dict[str, Any]:
    """
    Run optimization evaluation and return metrics.
    
    Args:
        model: Trained model
        config: Configuration dictionary
        run_dir: Run directory
        device: Device to use
        checkpoint_epoch: Which checkpoint to evaluate ('25%', '50%', '75%', 'final')
        
    Returns:
        Dictionary with evaluation metrics
    """
    import time
    import torch
    
    # Get evaluation settings
    eval_settings = config.get('evaluation_settings', {})
    eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
    eval_n_samples = eval_settings.get('eval_n_samples', 2)
    eval_n_queries = eval_settings.get('eval_n_queries', 10)
    eval_seed = config.get('data_settings', {}).get('eval_seed', 1)
    
    # Get optimization settings
    opt_config = config.get('latent_optimization', {})
    opt_method = opt_config.get('method', 'gradient')
    
    # Prepare metrics dictionary
    metrics = {
        'latent_dim': config['model_architecture']['latent_dim'],
        'method': opt_method,
        'checkpoint': checkpoint_epoch,
        'initial_loss': None,
        'final_loss': None,
        'delta_loss': None,
        'time_seconds': None,
        'loss_per_second': None,
        'forward_passes': None,
        'best_latent_norm': None,
        'non_improving_steps': None
    }
    
    # Add method-specific parameters
    if opt_method == 'gradient':
        inference_config = opt_config.get('inference', {})
        metrics.update({
            'steps': inference_config.get('num_steps', 50),
            'learning_rate': inference_config.get('learning_rate', 0.1),
            'generations': None,
            'population': None,
            'mutation_std': None
        })
    elif opt_method == 'evolutionary':
        evolutionary_config = opt_config.get('evolutionary', {})
        metrics.update({
            'steps': None,
            'learning_rate': None,
            'generations': evolutionary_config.get('num_generations', 25),
            'population': evolutionary_config.get('population_size', 100),
            'mutation_std': evolutionary_config.get('mutation_std', 0.3)
        })
    
    try:
        # Generate evaluation data
        from re_arc.main import generate_and_process_tasks
        from utils.data_preparation import prepare_dataloader
        
        # Generate sample data for evaluation
        input_sequences = []
        target_sequences = []
        
        for key in eval_keys[:1]:  # Use first key for evaluation
            try:
                _, _, _, task_input_sequences, task_output_sequences = generate_and_process_tasks(key, eval_n_samples)
                input_sequences.extend(task_input_sequences)
                target_sequences.extend(task_output_sequences)
            except Exception as e:
                print(f"Error generating data for key {key}: {e}")
                continue
        
        if not input_sequences:
            print("No evaluation data generated")
            return metrics
        
        # Create dataloader
        dataloader = prepare_dataloader(input_sequences, target_sequences, batch_size=1)
        
        # Get one sample for optimization
        sample_input, sample_target = next(iter(dataloader))
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)
        
        # Get initial latent and loss
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                # Multi-encoder model
                mu_initial, logvar_initial, _ = model.encoder(sample_input, sample_target)
                z_initial = model.reparameterize(mu_initial, logvar_initial)
            else:
                # Single encoder model
                mu_initial, logvar_initial, _ = model.encoder(sample_input, sample_target)
                z_initial = model.reparameterize(mu_initial, logvar_initial)
            
            # Get initial loss
            shape_logits_initial, grid_logits_initial = model.decoder(z_initial, sample_input, sample_target)
            initial_loss = compute_loss(model, sample_input, sample_target, return_components=True)['total_loss'].item()
            metrics['initial_loss'] = initial_loss
            metrics['best_latent_norm'] = z_initial.norm().item()
        
        # Run optimization
        start_time = time.time()
        
        if opt_method == 'gradient':
            # Gradient ascent optimization
            z_optimized = z_initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z_optimized], lr=metrics['learning_rate'])
            
            non_improving_steps = 0
            best_loss = initial_loss
            
            for step in range(metrics['steps']):
                optimizer.zero_grad()
                
                # Forward pass
                shape_logits, grid_logits = model.decoder(z_optimized, sample_input, sample_target)
                loss = compute_loss(model, sample_input, sample_target, return_components=True)['total_loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                if current_loss >= best_loss:
                    non_improving_steps += 1
                else:
                    best_loss = current_loss
                    non_improving_steps = 0
                
                # Update best latent norm
                current_norm = z_optimized.norm().item()
                if current_norm > metrics['best_latent_norm']:
                    metrics['best_latent_norm'] = current_norm
            
            final_loss = best_loss
            metrics['forward_passes'] = metrics['steps']
            
        elif opt_method == 'evolutionary':
            # Evolutionary optimization
            from utils.latent_functions import optimize_latent_evolutionary
            
            z_optimized, optimization_history = optimize_latent_evolutionary(
                model, sample_input, sample_target,
                population_size=metrics['population'],
                num_generations=metrics['generations'],
                mutation_std=metrics['mutation_std'],
                device=device
            )
            
            final_loss = optimization_history['best_loss']
            metrics['forward_passes'] = metrics['generations'] * metrics['population']
            metrics['non_improving_steps'] = optimization_history.get('non_improving_generations', 0)
            
            # Update best latent norm
            current_norm = z_optimized.norm().item()
            if current_norm > metrics['best_latent_norm']:
                metrics['best_latent_norm'] = current_norm
        
        # Calculate final metrics
        end_time = time.time()
        metrics['time_seconds'] = end_time - start_time
        metrics['final_loss'] = final_loss
        metrics['delta_loss'] = initial_loss - final_loss
        metrics['loss_per_second'] = metrics['delta_loss'] / metrics['time_seconds'] if metrics['time_seconds'] > 0 else 0
        
        print(f"Optimization completed: {opt_method}, checkpoint={checkpoint_epoch}, ΔLoss: {metrics['delta_loss']:.4f}, Time: {metrics['time_seconds']:.2f}s")
        
    except Exception as e:
        print(f"Error during optimization evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics

def save_optimization_metrics(metrics: Dict[str, Any], run_dir: str, config_name: str):
    """Save optimization metrics to JSON file."""
    import json
    
    metrics_file = os.path.join(run_dir, f"optimization_metrics_{config_name}.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Optimization metrics saved to: {metrics_file}")

def run_single_optimization_experiment(run_number: int, config_name: str, config: Dict[str, Any], 
                                     modes: List[str], device: str) -> bool:
    """
    Run a single optimization experiment with the given configuration.
    
    Args:
        run_number: Sequential run number
        config_name: Name of the configuration
        config: Configuration dictionary
        modes: List of modes to run (train, eval, visualize)
        device: Device to use (cuda/cpu)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting Optimization Run {run_number}: {config_name}")
        print(f"{'='*60}")
        
        # Save configuration
        config_path = save_configuration(config_name, config, run_number)
        print(f"Configuration saved to: {config_path}")
        
        # Update settings
        settings.set_settings(config)
        
        # Create run name
        wandb_project = config.get("wandb_settings", {}).get("project_name", "default_project")
        run_name = f"{wandb_project}_opt_run_{run_number}"

        # Create run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(BASE_DIR, f"{run_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created run directory: {run_dir}")

        if 'train' in modes or 'all' in modes:
            print(f"\n--- Training Mode ---")
            
            # Train the model
            results, model = main_training(run_name, run_dir=run_dir, notes=f"Optimization sweep: {config_name}")
            if results is None:
                print(f"Training failed for run {run_number}")
                return False
            
            print(f"Training completed successfully for run {run_number}")
        
        if 'eval' in modes or 'all' in modes:
            print(f"\n--- Optimization Evaluation Mode ---")
            
            # Get number of epochs from config
            num_epochs = config["training_settings"]["num_epochs"]
            checkpoint_epochs = ['25%', '50%', '75%', 'final']
            
            # Run evaluation at each checkpoint
            for checkpoint_epoch in checkpoint_epochs:
                print(f"\n--- Evaluating at {checkpoint_epoch} checkpoint ---")
                
                # Load the model at specific checkpoint
                if checkpoint_epoch == 'final':
                    model, _, _, _ = load_model(run_dir, epoch='final', device=device)
                else:
                    # Calculate epoch number for percentage checkpoints
                    if checkpoint_epoch == '25%':
                        epoch_num = max(1, num_epochs // 4)
                    elif checkpoint_epoch == '50%':
                        epoch_num = max(1, num_epochs // 2)
                    elif checkpoint_epoch == '75%':
                        epoch_num = max(1, 3 * num_epochs // 4)
                    else:
                        epoch_num = num_epochs
                    
                    model, _, _, _ = load_model(run_dir, epoch=epoch_num, device=device)
                
                # Run optimization evaluation
                metrics = run_optimization_evaluation(model, config, run_dir, device, checkpoint_epoch)
                
                # Save metrics with checkpoint identifier
                checkpoint_suffix = checkpoint_epoch.replace('%', 'pct')
                save_optimization_metrics(metrics, run_dir, f"{config_name}_{checkpoint_suffix}")
                
                print(f"Optimization evaluation completed for {checkpoint_epoch} checkpoint")
        
        if 'visualize' in modes or 'all' in modes:
            print(f"\n--- Visualization Mode ---")
            
            # Load evaluation results and create visualizations
            eval_epoch = config.get('evaluation_settings', {}).get('eval_epoch', 'final')
            visualize_stored_results(run_dir, epoch=eval_epoch)
            print(f"Visualization completed for run {run_number}")
        
        print(f"\nRun {run_number} completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in run {run_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always restore original settings
        restore_original_settings()

def run_sweep(modes: List[str], start_run: int = 1, end_run: int = None, 
              device: str = None, parallel: bool = False, specialist_mode: bool = False):
    """
    Run the hyperparameter sweep.
    
    Args:
        modes: List of modes to run (train, eval, visualize, all)
        start_run: Starting run number
        end_run: Ending run number (None for all)
        device: Device to use (None for auto-detect)
        parallel: Whether to run experiments in parallel
        specialist_mode: Whether to use specialist training configurations
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set WandB project name at the very beginning to ensure all runs use the same project
    os.environ['WANDB_PROJECT_NAME'] = WANDB_PROJECT
    
    print(f"Starting hyperparameter sweep on device: {device}")
    print(f"Modes: {modes}")
    print(f"Specialist mode: {specialist_mode}")
    print(f"WandB Project: {WANDB_PROJECT}")
    
    # Get configurations
    configurations = create_sweep_configurations(specialist_mode=specialist_mode)
    
    # Debug: Print all configurations after creation
    print(f"\nDEBUG: All configurations after creation:")
    for i, (name, config) in enumerate(configurations):
        latent_dim = config['model_architecture'].get('latent_dim', 'NOT FOUND')
        opt_method = config.get('latent_optimization', {}).get('method', 'NOT FOUND')
        print(f"  {i}: {name} - num_encoders={config['model_architecture']['num_encoders']}, encoder_layers={config['model_architecture']['encoder_layers']}, latent_dim={latent_dim}, opt_method={opt_method}")
    
    if end_run is None:
        end_run = len(configurations)
    
    # Filter configurations based on run range
    configurations = configurations[start_run-1:end_run]
    
    # Debug: Print filtered configurations
    print(f"\nDEBUG: Filtered configurations (runs {start_run}-{end_run}):")
    for i, (name, config) in enumerate(configurations):
        latent_dim = config['model_architecture'].get('latent_dim', 'NOT FOUND')
        opt_method = config.get('latent_optimization', {}).get('method', 'NOT FOUND')
        print(f"  {i}: {name} - num_encoders={config['model_architecture']['num_encoders']}, encoder_layers={config['model_architecture']['encoder_layers']}, latent_dim={latent_dim}, opt_method={opt_method}")
    
    print(f"Running {len(configurations)} experiments (runs {start_run}-{end_run})")
    
    # Create sweep configs directory
    Path(SWEEP_CONFIG_DIR).mkdir(exist_ok=True)
    
    # Run experiments
    successful_runs = 0
    failed_runs = []
    
    try:
        for i, (config_name, config) in enumerate(configurations):
            run_number = start_run + i
            
            # Debug: Print which configuration is being used
            print(f"\nDEBUG: Run {run_number} using config '{config_name}'")
            print(f"  Config num_encoders: {config['model_architecture']['num_encoders']}")
            print(f"  Config encoder_layers: {config['model_architecture']['encoder_layers']}")
            print(f"  Config latent_dim: {config['model_architecture'].get('latent_dim', 'NOT FOUND')}")
            print(f"  Config optimization_method: {config.get('latent_optimization', {}).get('method', 'NOT FOUND')}")
            
            if parallel:
                # For parallel execution, we'd need to implement subprocess calls
                # For now, we'll run sequentially
                pass
            
            success = run_single_experiment(run_number, config_name, config, modes, device, specialist_mode)
            
            if success:
                successful_runs += 1
            else:
                failed_runs.append(run_number)
    finally:
        # Final cleanup to ensure settings are restored
        restore_original_settings()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total runs: {len(configurations)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
    print(f"Configurations saved in: {SWEEP_CONFIG_DIR}")
    print(f"WandB project: {WANDB_PROJECT}")

def run_optimization_sweep(modes: List[str], start_run: int = 1, end_run: int = None, 
                          device: str = None, parallel: bool = False):
    """
    Run the optimization method comparison sweep.
    
    Args:
        modes: List of modes to run (train, eval, visualize, all)
        start_run: Starting run number
        end_run: Ending run number (None for all)
        device: Device to use (None for auto-detect)
        parallel: Whether to run experiments in parallel
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set WandB project name for optimization sweep
    os.environ['WANDB_PROJECT_NAME'] = WANDB_OPT_PROJECT  # Use the new constant
    
    print(f"Starting optimization sweep on device: {device}")
    print(f"Modes: {modes}")
    print(f"WandB Project: {WANDB_OPT_PROJECT}")  # Use the new constant
    
    # Get configurations
    configurations = create_sweep_opt_configurations()
    
    # Debug: Print all configurations
    print(f"\nDEBUG: All optimization configurations:")
    for i, (name, config) in enumerate(configurations):
        latent_dim = config['model_architecture'].get('latent_dim', 'NOT FOUND')
        opt_method = config.get('latent_optimization', {}).get('method', 'NOT FOUND')
        print(f"  {i}: {name} - latent_dim={latent_dim}, opt_method={opt_method}")
    
    if end_run is None:
        end_run = len(configurations)
    
    # Filter configurations based on run range
    configurations = configurations[start_run-1:end_run]
    
    print(f"Running {len(configurations)} optimization experiments (runs {start_run}-{end_run})")
    
    # Create sweep configs directory
    Path(SWEEP_CONFIG_DIR).mkdir(exist_ok=True)
    
    # Run experiments
    successful_runs = 0
    failed_runs = []
    
    try:
        for i, (config_name, config) in enumerate(configurations):
            run_number = start_run + i
            
            print(f"\nDEBUG: Run {run_number} using config '{config_name}'")
            print(f"  Config latent_dim: {config['model_architecture'].get('latent_dim', 'NOT FOUND')}")
            print(f"  Config optimization_method: {config.get('latent_optimization', {}).get('method', 'NOT FOUND')}")
            
            success = run_single_optimization_experiment(run_number, config_name, config, modes, device)
            
            if success:
                successful_runs += 1
            else:
                failed_runs.append(run_number)
    finally:
        # Final cleanup
        restore_original_settings()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total runs: {len(configurations)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
    print(f"Configurations saved in: {SWEEP_CONFIG_DIR}")
    print(f"WandB project: {WANDB_OPT_PROJECT}")  # Use the new constant

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for Multi-Encoder LPN')
    parser.add_argument('--mode', nargs='+', choices=['train', 'eval', 'visualize', 'all'], 
                       default=['all'], help='Modes to run')
    parser.add_argument('--start_run', type=int, default=1, help='Starting run number')
    parser.add_argument('--end_run', type=int, default=None, help='Ending run number (None for all)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None, 
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--specialist', action='store_true', help='Use specialist training configurations and methods')
    parser.add_argument('--optimization', action='store_true', help='Run optimization method comparison sweep')
    parser.add_argument('--list_configs', action='store_true', help='List all configurations and exit')
    
    args = parser.parse_args()
    
    # Get device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # List configurations and exit if requested
    if args.list_configs:
        if args.optimization:
            configs = create_sweep_opt_configurations()
            print("Available optimization configurations:")
        elif args.specialist:
            configs = create_sweep_configurations(specialist_mode=True)
            print("Available specialist configurations:")
        else:
            configs = create_sweep_configurations(specialist_mode=False)
            print("Available standard configurations:")
            
        for i, (name, config) in enumerate(configs):
            print(f"  {i+1}. {name}")
            if args.optimization:
                print(f"     - Latent Dim: {config['model_architecture'].get('latent_dim', 'N/A')}")
                print(f"     - Method: {config.get('latent_optimization', {}).get('method', 'N/A')}")
                if config.get('latent_optimization', {}).get('method') == 'gradient':
                    steps = config.get('latent_optimization', {}).get('inference', {}).get('num_steps', 'N/A')
                    print(f"     - Steps: {steps}")
                elif config.get('latent_optimization', {}).get('method') == 'evolutionary':
                    pop = config.get('latent_optimization', {}).get('evolutionary', {}).get('population_size', 'N/A')
                    gen = config.get('latent_optimization', {}).get('evolutionary', {}).get('num_generations', 'N/A')
                    print(f"     - Population: {pop}, Generations: {gen}")
            else:
                print(f"     - Encoders: {config['model_architecture']['num_encoders']}")
                print(f"     - LR: {config['training_settings']['learning_rate']}")
                print(f"     - Beta: {config['training_settings']['beta']}")
                print(f"     - Latent Dim: {config['model_architecture'].get('latent_dim', 'N/A')}")
            print()
        return
    
    # Run appropriate sweep
    if args.optimization:
        run_optimization_sweep(args.mode, args.start_run, args.end_run, device, args.parallel)
    else:
        run_sweep(args.mode, args.start_run, args.end_run, device, args.parallel, args.specialist)

if __name__ == "__main__":
    main() 