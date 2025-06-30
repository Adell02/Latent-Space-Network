#!/usr/bin/env python3
"""
Lean evaluation utilities for running evaluation during training every N epochs.
Reuses existing evaluation functions to avoid code duplication.
"""

import os
import pickle
from typing import Dict, Any, Optional, Tuple
from utils.settings_manager import settings

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

def log_evaluation_to_wandb(eval_results: Dict[str, Any], run_dir: str, epoch: int, wandb_logger):
    """
    Log evaluation results and visualizations to wandb.
    
    Args:
        eval_results: Results from evaluation
        run_dir: Run directory
        epoch: Current epoch
        wandb_logger: Wandb logger instance
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
        
        # Log all visualizations – ensure epoch is int for modulo check
        epoch_step = epoch if isinstance(epoch, int) else 0
        wandb_logger.log_visualizations(run_dir, epoch_step, eval_results)
        
    except Exception as e:
        print(f"⚠ Error logging evaluation to wandb: {e}")
        import traceback
        traceback.print_exc() 