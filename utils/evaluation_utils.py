#!/usr/bin/env python3
"""
Lean evaluation utilities for running evaluation during training every N epochs.
Reuses existing evaluation functions to avoid code duplication.
"""

import os
import pickle
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple
from utils.settings_manager import settings

def generate_trajectory_plots(eval_results: Dict[str, Any], run_dir: str, epoch: int, max_samples: int = 3, current_model=None) -> Dict[str, str]:
    """
    Generate trajectory plots from evaluation results efficiently.
    
    Args:
        eval_results: Results from evaluation containing trajectory info
        run_dir: Run directory
        epoch: Current epoch
        max_samples: Maximum number of trajectory samples to visualize (for efficiency)
        current_model: Optional in-memory model to use instead of loading from disk
        
    Returns:
        dict: Dictionary mapping plot names to file paths
    """
    trajectory_plots = {}
    
    # Check if trajectory plot logging is enabled
    wandb_settings = settings.get_wandb_settings()
    if not wandb_settings.get('log_trajectory_plots', False):
        print(f"Trajectory plot logging disabled, skipping...")
        return trajectory_plots
    
    # Check if latent optimization is enabled (required for trajectory data)
    latent_opt = settings.get_latent_optimization()
    if not latent_opt.get('inference', {}).get('enabled', False):
        print(f"Latent optimization disabled - no trajectory data available, skipping plots...")
        return trajectory_plots
    
    # Get max samples from settings
    max_samples = wandb_settings.get('trajectory_max_samples', max_samples)
    
    try:
        # Import trajectory visualization functions
        from LPN_reproduction.evaluate_trajectory import (
            visualize_comprehensive_trajectory,
            visualize_multi_encoder_comprehensive_trajectory
        )
        
        # Get the model - use in-memory model if provided, otherwise try to load from disk
        model = current_model
        if model is None:
            try:
                from utils.model_utils import load_model
                model, _, _, _ = load_model(run_dir, epoch=epoch, device='cuda')
                print(f"Loaded model from disk for epoch {epoch}")
            except Exception as e:
                print(f"⚠ Could not load model for epoch {epoch}, skipping trajectory plots: {e}")
                return trajectory_plots
        else:
            print(f"Using provided in-memory model for trajectory plots")
        
        # Process each key's trajectory info
        for key, key_results in eval_results.get('key_results', {}).items():
            if 'metrics' not in key_results:
                continue
                
            trajectory_info_list = key_results['metrics'].get('trajectory_info', [])
            if not trajectory_info_list:
                continue
                
            print(f"Generating trajectory plots for key '{key}' (epoch {epoch})...")
            
            # Limit number of samples for efficiency
            limited_trajectory_list = trajectory_info_list[:max_samples]
            
            # Generate plots for each sample
            for sample_idx, trajectory_info in enumerate(limited_trajectory_list):
                try:
                    # Create descriptive filename - include epoch for file uniqueness
                    plot_filename = f'trajectory_epoch{epoch}_{key}_sample{sample_idx}.png'
                    plot_path = os.path.join(run_dir, plot_filename)
                    
                    # Check if this is multi-encoder
                    is_multi_encoder = trajectory_info.get('is_multi_encoder', False)
                    
                    if is_multi_encoder:
                        visualize_multi_encoder_comprehensive_trajectory(
                            trajectory_info, model, plot_path, run_dir, device='cuda'
                        )
                    else:
                        visualize_comprehensive_trajectory(
                            trajectory_info, model, plot_path, run_dir, device='cuda'
                        )
                    
                    # Store plot path with consistent key for slider visualization
                    # Use key_sample format without epoch in the key for consistent naming across epochs
                    plot_key = f'{key}_sample{sample_idx}'
                    trajectory_plots[plot_key] = plot_path
                    
                    print(f"  ✓ Generated trajectory plot: {plot_filename}")
                    
                except Exception as e:
                    print(f"  ⚠ Failed to generate trajectory plot for {key} sample {sample_idx}: {e}")
                    continue
                    
        if trajectory_plots:
            print(f"✓ Generated {len(trajectory_plots)} trajectory plots for epoch {epoch}")
        else:
            print(f"No trajectory data available for epoch {epoch}")
            
    except Exception as e:
        print(f"⚠ Error generating trajectory plots: {e}")
        import traceback
        traceback.print_exc()
    
    return trajectory_plots

def run_quick_evaluation(model, run_dir: str, epoch: int) -> Optional[Dict[str, Any]]:
    """
    Run a quick evaluation using existing evaluation functions.
    Uses the current model state (which should be the latest trained state).
    Returns evaluation results for wandb logging.
    """
    try:
        # Import evaluation functions
        from evaluation import main_test
        
        # Get evaluation settings
        eval_settings = settings.get_evaluation_settings()
        eval_keys = eval_settings.get('eval_keys', ['00d62c1b'])
        n_samples = eval_settings.get('eval_n_samples', 2)
        n_queries = eval_settings.get('eval_n_queries', 10)  # Reduced for quick eval
        eval_seed = settings.get_data_settings().get('eval_seed', 42)
        
        print(f"Running quick evaluation for epoch {epoch}...")
        print(f"  Keys: {eval_keys}")
        print(f"  Samples: {n_samples}, Queries: {n_queries}")
        
        # Use the provided model directly (it should be the latest trained state)
        device = next(model.parameters()).device
        eval_results = main_test(model, eval_keys, run_dir, n_samples, n_queries, eval_seed, device)
        
        if eval_results:
            print(f"✓ Quick evaluation completed for epoch {epoch}")
            # Add epoch metadata to results
            if 'evaluation_metadata' not in eval_results:
                eval_results['evaluation_metadata'] = {}
            eval_results['evaluation_metadata']['epoch'] = epoch
            # Capture reconstruction results
            if 'reconstruction_results' in eval_results:
                eval_results['trajectory_reconstruction'] = eval_results['reconstruction_results']
            return eval_results
        else:
            print(f"⚠ Quick evaluation failed for epoch {epoch}")
            return None
            
    except Exception as e:
        print(f"⚠ Quick evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def should_run_evaluation(epoch: int, log_interval: int, total_epochs: int) -> bool:
    """
    Determine if evaluation should run for this epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        log_interval: Wandb log interval
        total_epochs: Total number of epochs
        
    Returns:
        bool: True if evaluation should run
    """
    # Always evaluate on first and last epoch
    if epoch == 1 or epoch == total_epochs:
        return True
    
    # Evaluate every log_interval epochs
    if epoch % log_interval == 0:
        return True
    
    return False

def log_evaluation_to_wandb(eval_results: Dict[str, Any], run_dir: str, epoch: int, wandb_logger, current_model=None):
    """
    Log evaluation results and visualizations to wandb.
    
    Args:
        eval_results: Results from evaluation
        run_dir: Run directory
        epoch: Current epoch
        wandb_logger: Wandb logger instance
        current_model: Optional in-memory model to use instead of loading from disk
    """
    if not wandb_logger or not wandb_logger.is_initialized:
        return
    
    try:
        # Log evaluation metrics
        if 'aggregated_metrics' in eval_results:
            agg_metrics = eval_results['aggregated_metrics']
            avg_metrics = agg_metrics.get('average_metrics', {})
            
            eval_metrics = {
                'eval_shape_accuracy': avg_metrics.get('avg_shape_accuracy', 0.0),
                'eval_grid_accuracy': avg_metrics.get('avg_grid_accuracy', 0.0),
                'eval_sample_exact_accuracy': avg_metrics.get('avg_sample_exact_accuracy', 0.0),
                'eval_support_loss': avg_metrics.get('avg_support_loss', 0.0),
                'eval_query_loss': avg_metrics.get('avg_query_loss', 0.0),
            }
            wandb_logger.log_training_metrics(epoch, eval_metrics)
            print(f"✓ Logged evaluation metrics to wandb")
        else:
            print("⚠ No aggregated metrics found in evaluation results")
        
        # Save evaluation results to file for visualization
        try:
            eval_results_file = os.path.join(run_dir, 'evaluation_results.pkl')
            with open(eval_results_file, 'wb') as f:
                pickle.dump(eval_results, f)
            print(f"✓ Saved evaluation results to {eval_results_file}")
        except Exception as e:
            print(f"⚠ Could not save evaluation results: {e}")
        
        # Generate and log trajectory plots
        wandb_settings = settings.get_wandb_settings()
        max_samples = wandb_settings.get('trajectory_max_samples', 3)
        trajectory_plots = generate_trajectory_plots(eval_results, run_dir, epoch, max_samples, current_model=current_model)
        
        # Log all visualizations – ensure epoch is int for modulo check
        epoch_step = epoch if isinstance(epoch, int) else 0
        wandb_logger.log_visualizations(run_dir, epoch_step, eval_results, trajectory_plots)
        
    except Exception as e:
        print(f"⚠ Error logging evaluation to wandb: {e}")
        import traceback
        traceback.print_exc() 