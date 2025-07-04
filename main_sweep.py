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
from typing import Dict, List, Any
import torch

# Import existing modules
from training import main_training
from evaluation import main_test
from utils.model_utils import load_model, save_evaluation_results
from utils.visualizers import visualize_stored_results
from utils.settings_manager import settings

# Constants
SWEEP_CONFIG_DIR = "sweep_configs"
WANDB_PROJECT = "LARGE_multiencoder_sweep"
BASE_DIR = "runs_re_arc"

def create_sweep_configurations() -> List[Dict[str, Any]]:
    """
    Define comprehensive hyperparameter configurations for the sweep.
    Returns a list of configuration dictionaries.
    """
    configurations = []
    
    # Base configuration
    # Load base configuration from model_settings.json
    with open("model_settings.json", "r") as f:
        base_config = json.load(f)
    
    print("Base config loaded from model_settings.json")
    print(f"  Base num_encoders: {base_config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
    print(f"  Base encoder_layers: {base_config.get('model_architecture', {}).get('encoder_layers', 'NOT FOUND')}")
    
    # Configuration 1: Single Encoder Baseline ~ 1.2 M params
    # config_1 = copy.deepcopy(base_config)
    # config_1['model_architecture']['num_encoders'] = 1
    # config_1['model_architecture']['encoder_layers'] = 8
    
    # # Remove any project_name from wandb config to ensure environment variable is used
    # if 'training_settings' in config_1 and 'wandb' in config_1['training_settings']:
    #    config_1['training_settings']['wandb'].pop('project_name', None)
    
    # print(f"Config 1: num_encoders={config_1['model_architecture']['num_encoders']}, encoder_layers={config_1['model_architecture']['encoder_layers']}")
    # configurations.append(("1200k_param_1_enc", config_1))
    
    # # Configuration 2: Multi-Encoder (2 encoders) ~ 1.2 M params
    # config_2 = copy.deepcopy(base_config)
    # config_2["model_architecture"]["num_encoders"] = 2
    # config_2["model_architecture"]["encoder_layers"] = 4
    
    # # Remove any project_name from wandb config to ensure environment variable is used
    # if 'training_settings' in config_2 and 'wandb' in config_2['training_settings']:
    #     config_2['training_settings']['wandb'].pop('project_name', None)
    
    # print(f"Config 2: num_encoders={config_2['model_architecture']['num_encoders']}, encoder_layers={config_2['model_architecture']['encoder_layers']}")
    # configurations.append(("1200k_param_2_enc", config_2))
    
    # # Configuration 3: Multi-Encoder (4 encoders) ~ 1.2 M params
    # config_3 = copy.deepcopy(base_config)
    # config_3["model_architecture"]["num_encoders"] = 4
    # config_3["model_architecture"]["encoder_layers"] = 2
    
    # # Remove any project_name from wandb config to ensure environment variable is used
    # if 'training_settings' in config_3 and 'wandb' in config_3['training_settings']:
    #     config_3['training_settings']['wandb'].pop('project_name', None)
    
    # print(f"Config 3: num_encoders={config_3['model_architecture']['num_encoders']}, encoder_layers={config_3['model_architecture']['encoder_layers']}")
    # configurations.append(("1200k_param_4_enc", config_3))

    # Different repulsion loss values
    for i in range(0,10):
        config_i_rl = copy.deepcopy(base_config)
        config_i_rl["training_settings"]["repulsion_loss"]["schedule"]["warmup_epochs"] = i
        print(f"Config {i}_rl: warmup_epochs={i}")
        configurations.append(f"{i}_repulsion_loss", config_i_rl)

    # config_1_rl = copy.deepcopy(base_config)
    # config_1_rl["training_settings"]["repulsion_loss"]["lambda"] = 1

    # print(f"Config 1_rl: num_encoders={config_1_rl['model_architecture']['num_encoders']}, encoder_layers={config_1_rl['model_architecture']['encoder_layers']}")
    # configurations.append(("1_repulsion_loss", config_1_rl))

    # config_0_5_rl = copy.deepcopy(base_config)
    # config_0_5_rl["training_settings"]["repulsion_loss"]["lambda"] = 0.5
    
    # print(f"Config 0_5_rl: num_encoders={config_0_5_rl['model_architecture']['num_encoders']}, encoder_layers={config_0_5_rl['model_architecture']['encoder_layers']}")
    # configurations.append(("0_5_repulsion_loss", config_0_5_rl))

    # config_0_rl = copy.deepcopy(base_config)
    # config_0_rl["training_settings"]["repulsion_loss"]["lambda"] = 0

    # print(f"Config 0_rl: num_encoders={config_0_rl['model_architecture']['num_encoders']}, encoder_layers={config_0_rl['model_architecture']['encoder_layers']}")
    # configurations.append(("0_repulsion_loss", config_0_rl))


    # Different Model Sizes
    # config_24_hd = copy.deepcopy(base_config)
    # config_24_hd["model_architecture"]["encoder_hidden_dim"] = 24

    # print(f"Config 24_hd: num_encoders={config_24_hd['model_architecture']['num_encoders']}, encoder_layers={config_24_hd['model_architecture']['encoder_layers']}")
    # configurations.append(("24_encoder_hidden_dim", config_24_hd))

    # # Different Model Sizes
    # config_96_hd = copy.deepcopy(base_config)
    # config_96_hd["model_architecture"]["encoder_hidden_dim"] = 96

    # print(f"Config 96_hd: num_encoders={config_96_hd['model_architecture']['num_encoders']}, encoder_layers={config_96_hd['model_architecture']['encoder_layers']}")
    # configurations.append(("96_encoder_hidden_dim", config_96_hd))

    # # Different Model Sizes
    # config_180_hd = copy.deepcopy(base_config)
    # config_180_hd["model_architecture"]["encoder_hidden_dim"] = 180

    # print(f"Config 180_hd: num_encoders={config_180_hd['model_architecture']['num_encoders']}, encoder_layers={config_180_hd['model_architecture']['encoder_layers']}")
    # configurations.append(("180_encoder_hidden_dim", config_180_hd))

    # # Train with different latent dimensions
    # config_16_ld = copy.deepcopy(base_config)
    # config_16_ld["model_architecture"]["latent_dim"] = 16

    # print(f"Config 16_ld: num_encoders={config_16_ld['model_architecture']['num_encoders']}, encoder_layers={config_16_ld['model_architecture']['encoder_layers']}")
    # configurations.append(("16_latent_dim", config_16_ld))

    # # Train with different latent dimensions
    # config_64_ld = copy.deepcopy(base_config)
    # config_64_ld["model_architecture"]["latent_dim"] = 64

    # print(f"Config 64_ld: num_encoders={config_64_ld['model_architecture']['num_encoders']}, encoder_layers={config_64_ld['model_architecture']['encoder_layers']}")
    # configurations.append(("64_latent_dim", config_64_ld))

    # # Train with different latent dimensions
    # config_1024_ld = copy.deepcopy(base_config)
    # config_1024_ld["model_architecture"]["latent_dim"] = 1024

    # print(f"Config 1024_ld: num_encoders={config_1024_ld['model_architecture']['num_encoders']}, encoder_layers={config_1024_ld['model_architecture']['encoder_layers']}")
    # configurations.append(("1024_latent_dim", config_1024_ld))    

    return configurations

def save_configuration(config_name: str, config: Dict[str, Any], run_number: int):
    """Save configuration to the sweep_configs folder."""
    config_path = Path(SWEEP_CONFIG_DIR) / f"run_{run_number}_{config_name}.json"
    config_path.parent.mkdir(exist_ok=True)
    
    # Debug: Print what's being saved
    print(f"Saving config for run {run_number}:")
    print(f"  num_encoders: {config.get('model_architecture', {}).get('num_encoders', 'NOT FOUND')}")
    print(f"  encoder_layers: {config.get('model_architecture', {}).get('encoder_layers', 'NOT FOUND')}")
    
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
        print("✓ Original settings restored")
    else:
        print("⚠ No backup settings file found to restore")

def run_single_experiment(run_number: int, config_name: str, config: Dict[str, Any], 
                         modes: List[str], device: str) -> bool:
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
        run_name = f"run_{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create run directory
        run_dir = os.path.join(BASE_DIR, run_name)
        
        if 'train' in modes or 'all' in modes:
            print(f"\n--- Training Mode ---")
            print(f"Run directory: {run_dir}")
            
            # Train the model
            results, model = main_training(run_name)
            if results is None:
                print(f"Training failed for run {run_number}")
                return False
            
            print(f"Training completed successfully for run {run_number}")
        
        if 'eval' in modes or 'all' in modes:
            print(f"\n--- Evaluation Mode ---")
            
            # Load the model
            eval_epoch = config.get('evaluation_settings', {}).get('eval_epoch', 35)
            model, _, _, _ = load_model(run_dir, epoch=eval_epoch, device=device)
            
            # Run evaluation
            eval_keys = config.get('evaluation_settings', {}).get('eval_keys', ['00d62c1b'])
            eval_n_samples = config.get('evaluation_settings', {}).get('eval_n_samples', 2)
            eval_n_queries = config.get('evaluation_settings', {}).get('eval_n_queries', 100)
            eval_seed = config.get('data_settings', {}).get('eval_seed', 1)
            
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

def run_sweep(modes: List[str], start_run: int = 1, end_run: int = None, 
              device: str = None, parallel: bool = False):
    """
    Run the hyperparameter sweep.
    
    Args:
        modes: List of modes to run (train, eval, visualize, all)
        start_run: Starting run number
        end_run: Ending run number (None for all)
        device: Device to use (None for auto-detect)
        parallel: Whether to run experiments in parallel
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set WandB project name at the very beginning to ensure all runs use the same project
    os.environ['WANDB_PROJECT_NAME'] = WANDB_PROJECT
    
    print(f"Starting hyperparameter sweep on device: {device}")
    print(f"Modes: {modes}")
    print(f"WandB Project: {WANDB_PROJECT}")
    
    # Get configurations
    configurations = create_sweep_configurations()
    
    # Debug: Print all configurations after creation
    print(f"\nDEBUG: All configurations after creation:")
    for i, (name, config) in enumerate(configurations):
        print(f"  {i}: {name} - num_encoders={config['model_architecture']['num_encoders']}, encoder_layers={config['model_architecture']['encoder_layers']}")
    
    if end_run is None:
        end_run = len(configurations)
    
    # Filter configurations based on run range
    configurations = configurations[start_run-1:end_run]
    
    # Debug: Print filtered configurations
    print(f"\nDEBUG: Filtered configurations (runs {start_run}-{end_run}):")
    for i, (name, config) in enumerate(configurations):
        print(f"  {i}: {name} - num_encoders={config['model_architecture']['num_encoders']}, encoder_layers={config['model_architecture']['encoder_layers']}")
    
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
            
            if parallel:
                # For parallel execution, we'd need to implement subprocess calls
                # For now, we'll run sequentially
                pass
            
            success = run_single_experiment(run_number, config_name, config, modes, device)
            
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

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for Multi-Encoder LPN')
    parser.add_argument('--mode', nargs='+', choices=['train', 'eval', 'visualize', 'all'], 
                       default=['all'], help='Modes to run')
    parser.add_argument('--start_run', type=int, default=1, help='Starting run number')
    parser.add_argument('--end_run', type=int, default=None, help='Ending run number (None for all)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None, 
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--list_configs', action='store_true', help='List all configurations and exit')
    
    args = parser.parse_args()
    
    # Get device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # List configurations and exit if requested
    if args.list_configs:
        configs = create_sweep_configurations()
        print("Available configurations:")
        for i, (name, config) in enumerate(configs):
            print(f"  {i+1}. {name}")
            print(f"     - Encoders: {config['model_architecture']['num_encoders']}")
            print(f"     - Epochs: {config['training_settings']['num_epochs']}")
            print(f"     - LR: {config['training_settings']['learning_rate']}")
            
            # Check if repulsion_loss exists in training_settings
            if 'repulsion_loss' in config['training_settings']:
                print(f"     - Repulsion Loss: {config['training_settings']['repulsion_loss']['enabled']}")
                if config['training_settings']['repulsion_loss']['enabled']:
                    print(f"     - Repulsion Lambda: {config['training_settings']['repulsion_loss']['lambda']}")
            
            # Check if solo_loss exists in training_settings before accessing it
            if 'solo_loss' in config['training_settings']:
                print(f"     - Solo Loss: {config['training_settings']['solo_loss']['enabled']}")
            
            print()
        return
    
    # Run sweep
    run_sweep(args.mode, args.start_run, args.end_run, device, args.parallel)

if __name__ == "__main__":
    main() 