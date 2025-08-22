#!/usr/bin/env python3
"""
Evaluate multiple model checkpoints downloaded from WandB artifacts on the same dataset.

This script ensures fair comparison by:
1. Generating unified training and evaluation datasets ONCE
2. Processing models one by one using evaluate_artifact.py
3. Freeing memory after each model for efficiency
4. Computing unified metrics across all models

Produces:
  - Individual model evaluations (same as evaluate_artifact.py)
  - Comparative metrics across all models
  - Unified visualizations (t-SNE plots, distance metrics)
  - Summary statistics and rankings

QUICK START:
  # Evaluate different models on same dataset with separate configs
  python evaluate_all_artifacts.py --artifacts \
    ga624-imperial-college-london/LPN_specialist_paper/LPN_specialist_paper_20250821_101933_main_20250821_101933_checkpoint_epoch_20:v0 \
    ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v7 \
    --configs \
    data_evaluation_1ENC_settings.json:model_1ENC_evaluation_settings.json \
    data_evaluation_2ENC_settings.json:model_2ENC_evaluation_settings.json \
    --specialist_flags 0 1

Usage:
  python evaluate_all_artifacts.py --artifacts ARTIFACT1 ARTIFACT2 ... [--configs CONFIG1:CONFIG2 ...] [--specialist_flags 0 1 ...] [--run_dir out_dir]

Notes:
  - All models are evaluated on the same dataset for fair comparison
  - Results are logged to a single WandB run with model-specific prefixes
  - Memory is freed after each model for efficiency
  - Use --configs for separate data/model configs per model
  - Use --specialist_flags to specify which models are specialist models (1) vs single encoder (0)
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import torch
import json
import subprocess
import shutil
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def _init_settings(config_path: str):
    from utils.settings_manager import init_settings
    return init_settings(config_path)


def _generate_unified_datasets(settings, n_samples: int = 5, n_queries: int = 10, n_training_samples: int = 100):
    """Generate unified training and evaluation datasets that will be used for all models"""
    from re_arc.main import generate_and_process_tasks
    from utils.evaluation_utils import get_evaluation_keys_with_all_support
    
    print(f"[OK] Generating unified datasets...")
    
    # Generate training dataset
    print(f"[OK] Generating unified training dataset ({n_training_samples} samples)...")
    training_dataset = _generate_training_dataset(settings, n_training_samples)
    
    # Generate evaluation dataset
    print(f"[OK] Generating unified evaluation dataset ({n_samples} support + {n_queries} query per key)...")
    evaluation_dataset = _generate_evaluation_dataset(settings, n_samples, n_queries)
    
    return training_dataset, evaluation_dataset


def _generate_training_dataset(settings, n_samples: int):
    """Generate unified training dataset"""
    from re_arc.main import generate_and_process_tasks
    
    data_settings = settings.get_data_settings()
    training_keys = data_settings.get('training_keys', [data_settings.get('key')])
    
    # Resolve 'all'
    if isinstance(training_keys, str) and training_keys.lower() == 'all':
        tasks_dir = os.path.join(os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
        all_keys = [fname[:-5] for fname in os.listdir(tasks_dir) if fname.endswith('.json')]
        all_keys.sort()
        n_max_keys = data_settings.get('n_max_keys', None)
        if n_max_keys is not None:
            try:
                n_max_keys = int(n_max_keys)
                all_keys = all_keys[:n_max_keys]
            except Exception:
                pass
        training_keys = all_keys
    
    print(f"[DEBUG] Training keys resolved: {len(training_keys)} keys")
    
    # Generate samples from all keys
    samples_per_key = max(1, n_samples // len(training_keys))
    print(f"[DEBUG] Samples per key: {samples_per_key}")
    
    training_data = {}
    for key in training_keys:
        try:
            _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, samples_per_key)
            if in_seqs and out_seqs:
                training_data[key] = {
                    'inputs': in_seqs,
                    'outputs': out_seqs
                }
        except Exception as e:
            print(f"[WARNING] Could not generate training data for key {key}: {e}")
    
    print(f"[OK] Generated training dataset with {len(training_data)} keys")
    return training_data


def _generate_evaluation_dataset(settings, n_samples: int, n_queries: int):
    """Generate unified evaluation dataset"""
    from re_arc.main import generate_and_process_tasks
    from utils.evaluation_utils import get_evaluation_keys_with_all_support
    
    eval_settings = settings.get_evaluation_settings()
    keys = eval_settings.get('eval_keys', ['00d62c1b'])
    
    # Resolve "all" using helper
    try:
        keys = get_evaluation_keys_with_all_support(keys, eval_settings.get('n_max_eval_keys', 10))
    except Exception:
        pass
    
    # Generate fixed dataset for all models
    evaluation_data = {}
    for key in keys:
        try:
            _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, n_samples + n_queries)
            if in_seqs and out_seqs:
                evaluation_data[key] = {
                    'support_inputs': in_seqs[:n_samples],
                    'support_outputs': out_seqs[:n_samples],
                    'query_inputs': in_seqs[n_samples:n_samples + n_queries],
                    'query_outputs': out_seqs[n_samples:n_samples + n_queries]
                }
        except Exception as e:
            print(f"[WARNING] Could not generate evaluation data for key {key}: {e}")
    
    print(f"[OK] Generated evaluation dataset with {len(evaluation_data)} keys")
    return evaluation_data


def _save_datasets_to_files(training_dataset, evaluation_dataset, output_dir: str):
    """Save datasets to temporary files that can be used by evaluate_artifact.py"""
    import pickle
    
    # Save training dataset
    training_file = os.path.join(output_dir, 'unified_training_dataset.pkl')
    with open(training_file, 'wb') as f:
        pickle.dump(training_dataset, f)
    
    # Save evaluation dataset
    evaluation_file = os.path.join(output_dir, 'unified_evaluation_dataset.pkl')
    with open(evaluation_file, 'wb') as f:
        pickle.dump(evaluation_dataset, f)
    
    # Save dataset info
    dataset_info = {
        'training_keys': list(training_dataset.keys()),
        'evaluation_keys': list(evaluation_dataset.keys()),
        'training_samples_per_key': len(next(iter(training_dataset.values()))['inputs']) if training_dataset else 0,
        'evaluation_support_samples': len(next(iter(evaluation_dataset.values()))['support_inputs']) if evaluation_dataset else 0,
        'evaluation_query_samples': len(next(iter(evaluation_dataset.values()))['query_inputs']) if evaluation_dataset else 0
    }
    
    info_file = os.path.join(output_dir, 'unified_dataset_info.json')
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"[OK] Saved unified datasets to {output_dir}")
    return training_file, evaluation_file, info_file


def _run_evaluate_artifact(artifact_id: str, config_data: str, config_model: str, 
                          is_specialist: bool, output_dir: str, device: str, 
                          no_wandb: bool = False, unified_training_file: str = None,
                          unified_evaluation_file: str = None) -> Dict[str, Any]:
    """Run evaluate_artifact.py for a single model and return results"""
    
    print(f"[OK] Running evaluate_artifact.py for {artifact_id}")
    
    # Build command
    cmd = [
        'python3', 'evaluate_artifact.py',
        '--artifact', artifact_id,
        '--config_data', config_data,
        '--config_model', config_model,
        '--run_dir', output_dir,
        '--device', device
    ]
    
    if is_specialist:
        cmd.append('--specialist')
    
    if no_wandb:
        cmd.append('--no_wandb')
    
    # Add unified dataset paths if available
    if unified_training_file and os.path.exists(unified_training_file):
        cmd.extend(['--unified_training_dataset', unified_training_file])
    
    if unified_evaluation_file and os.path.exists(unified_evaluation_file):
        cmd.extend(['--unified_evaluation_dataset', unified_evaluation_file])
    
    print(f"[DEBUG] Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[OK] evaluate_artifact.py completed successfully")
        
        # Try to load results
        results_file = os.path.join(output_dir, 'eval_aggregated_metrics.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            print(f"[WARNING] Results file not found: {results_file}")
            return {}
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] evaluate_artifact.py failed with exit code {e.returncode}")
        print(f"[ERROR] stdout: {e.stdout}")
        print(f"[ERROR] stderr: {e.stderr}")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to run evaluate_artifact.py: {e}")
        return {}


def _collect_model_results(output_dir: str, model_name: str) -> Dict[str, Any]:
    """Collect results from a model's evaluation output directory"""
    results = {}
    
    # Look for evaluation results
    eval_file = os.path.join(output_dir, 'eval_aggregated_metrics.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            results['evaluation_metrics'] = json.load(f)
    
    # Look for training results
    train_file = os.path.join(output_dir, 'training_latent_distance_metrics.json')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            results['training_metrics'] = json.load(f)
    
    # Look for latent space plots
    plot_dir = os.path.join(output_dir, 'latent_space_plots')
    if os.path.exists(plot_dir):
        results['plots'] = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    
    return results


def _compute_unified_metrics(all_model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute unified metrics across all models for comparison"""
    unified_metrics = {}
    
    for model_name, model_results in all_model_results.items():
        # Extract evaluation metrics
        eval_metrics = model_results.get('evaluation_metrics', {})
        if eval_metrics:
            # Support metrics
            support_metrics = eval_metrics.get('support_metrics', {})
            if support_metrics:
                post_opt = support_metrics.get('post_opt', {})
                unified_metrics[f"{model_name}/support"] = {
                    'shape_accuracy': post_opt.get('shape_accuracy', 0.0),
                    'grid_accuracy': post_opt.get('grid_accuracy', 0.0),
                    'exact_accuracy': post_opt.get('exact_accuracy', 0.0),
                    'final_opt_loss': post_opt.get('final_opt_loss', 0.0)
                }
            
            # Query metrics
            query_metrics = eval_metrics.get('query_metrics', {})
            if query_metrics:
                unified_metrics[f"{model_name}/query"] = {
                    'shape_accuracy': query_metrics.get('shape_accuracy', 0.0),
                    'grid_accuracy': query_metrics.get('grid_accuracy', 0.0),
                    'exact_accuracy': query_metrics.get('exact_accuracy', 0.0)
                }
        
        # Training metrics
        train_metrics = model_results.get('training_metrics', {})
        if train_metrics:
            unified_metrics[f"{model_name}/training"] = {
                'separation_ratio': train_metrics.get('separation_ratio', 0.0),
                'within_task_mean': train_metrics.get('within_task_mean', 0.0),
                'between_task_mean': train_metrics.get('between_task_mean', 0.0)
            }
    
    return unified_metrics


def _create_comparative_visualizations(all_model_results: Dict[str, Any], save_dir: str, wandb_logger=None):
    """Create visualizations comparing all models"""
    try:
        import matplotlib.pyplot as plt
        
        # 1. Comparative accuracy bar chart
        model_names = list(all_model_results.keys())
        metrics = ['shape_accuracy', 'grid_accuracy', 'exact_accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, metric in enumerate(metrics):
            values = []
            for model_name in model_names:
                query_metrics = all_model_results[model_name].get('evaluation_metrics', {}).get('query_metrics', {})
                if query_metrics:
                    values.append(query_metrics.get(metric, 0.0))
                else:
                    values.append(0.0)
            
            axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Accuracy')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        acc_path = os.path.join(save_dir, 'comparative_accuracies.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Training latent space comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        model_names = list(all_model_results.keys())
        separation_ratios = []
        
        for model_name in model_names:
            train_metrics = all_model_results[model_name].get('training_metrics', {})
            separation_ratios.append(train_metrics.get('separation_ratio', 0.0))
        
        ax.bar(model_names, separation_ratios, alpha=0.7)
        ax.set_title('Training Latent Space Separation Ratio')
        ax.set_ylabel('Separation Ratio')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        train_path = os.path.join(save_dir, 'training_latent_comparison.png')
        plt.savefig(train_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Upload to WandB
        if wandb_logger:
            try:
                wandb_logger.log({
                    "comparative_visualizations/accuracies": wandb.Image(acc_path),
                    "comparative_visualizations/training_latent": wandb.Image(train_path)
                })
                print(f"[OK] Uploaded comparative visualizations to WandB")
            except Exception as e:
                print(f"[WARNING] Failed to upload comparative visualizations: {e}")
        
        return [acc_path, train_path]
        
    except Exception as e:
        print(f"[WARNING] Could not create comparative visualizations: {e}")
        return []


def _save_json(obj, path):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple WandB artifact models on unified dataset')
    parser.add_argument('--artifacts', nargs='+', required=True, 
                       help='List of WandB artifact IDs, e.g., entity/project/artifact:version')
    parser.add_argument('--configs', nargs='+', default=None,
                       help='List of config pairs in format "data_config:model_config" for each model')
    parser.add_argument('--specialist_flags', nargs='+', type=int, default=None,
                       help='List of specialist flags (1) or single encoder (0) for each model')
    parser.add_argument('--run_dir', type=str, default=None, 
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_entity', type=str, default='ga624-imperial-college-london', 
                       help='WandB entity name')
    parser.add_argument('--wandb_project', type=str, default='evaluation_lpn', 
                       help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', 
                       help='Disable WandB logging')
    parser.add_argument('--n_samples', type=int, default=5, 
                       help='Number of support samples per task')
    parser.add_argument('--n_queries', type=int, default=10, 
                       help='Number of query samples per task')
    parser.add_argument('--n_training', type=int, default=100, 
                       help='Number of training samples total')
    args = parser.parse_args()

    # Validate arguments
    if args.configs and len(args.configs) != len(args.artifacts):
        print(f"[ERROR] Number of configs ({len(args.configs)}) must match number of artifacts ({len(args.artifacts)})")
        return
    
    if args.specialist_flags and len(args.specialist_flags) != len(args.artifacts):
        print(f"[ERROR] Number of specialist flags ({len(args.specialist_flags)}) must match number of artifacts ({len(args.artifacts)})")
        return

    device = args.device
    
    # Initialize WandB
    wandb_logger = None
    if not args.no_wandb:
        try:
            import wandb
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"multi_artifact_comparison_{timestamp}"
            
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                config={
                    "artifact_ids": args.artifacts,
                    "configs": args.configs,
                    "specialist_flags": args.specialist_flags,
                    "device": str(device),
                    "n_samples": args.n_samples,
                    "n_queries": args.n_queries,
                    "n_training": args.n_training,
                    "evaluation_timestamp": timestamp
                }
            )
            wandb_logger = wandb
            print(f"[OK] WandB run initialized: {args.wandb_entity}/{args.wandb_project}/{run_name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            wandb_logger = None

    # Create output directory
    if args.run_dir:
        out_dir = args.run_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        from utils.model_utils import create_run_directory
        out_dir = create_run_directory('multi_artifact_eval')

    # Initialize settings for dataset generation
    if args.configs:
        # Parse first config pair for dataset generation
        first_config_pair = args.configs[0].split(':')
        if len(first_config_pair) == 2:
            data_config_path, model_config_path = first_config_pair[0], first_config_pair[1]
            print(f"[OK] Using first config pair for dataset generation: {data_config_path}")
            settings = _init_settings(data_config_path)
        else:
            print(f"[ERROR] Invalid config format: {args.configs[0]}. Expected 'data_config:model_config'")
            return
    else:
        print(f"[ERROR] --configs is required for this script")
        return

    # Generate unified datasets ONCE
    print(f"[OK] Generating unified datasets (this will be done once and reused)...")
    training_dataset, evaluation_dataset = _generate_unified_datasets(
        settings, args.n_samples, args.n_queries, args.n_training
    )
    
    # Save datasets to files
    training_file, evaluation_file, info_file = _save_datasets_to_files(
        training_dataset, evaluation_dataset, out_dir
    )
    
    # Save dataset info
    dataset_info = {
        'n_samples': args.n_samples,
        'n_queries': args.n_queries,
        'n_training': args.n_training,
        'training_keys': list(training_dataset.keys()),
        'evaluation_keys': list(evaluation_dataset.keys()),
        'total_training_samples': sum(len(data['inputs']) for data in training_dataset.values()),
        'total_evaluation_samples': sum(len(data['support_inputs']) + len(data['query_inputs']) for data in evaluation_dataset.values())
    }
    _save_json(dataset_info, os.path.join(out_dir, 'unified_dataset_info.json'))

    # Evaluate each model one by one
    all_model_results = {}
    model_metadata = {}
    
    for i, artifact_id in enumerate(args.artifacts):
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL {i+1}/{len(args.artifacts)}: {artifact_id}")
        print(f"{'='*60}")
        
        try:
            # Extract model name from artifact ID
            model_name = artifact_id.split('/')[-1].split(':')[0]
            if model_name in all_model_results:
                model_name = f"{model_name}_{i}"  # Avoid duplicates
            
            # Determine if this is a specialist model
            is_specialist_model = False
            if args.specialist_flags and i < len(args.specialist_flags):
                is_specialist_model = bool(args.specialist_flags[i])
                print(f"[INFO] Using specialist flag for model {i+1}: {is_specialist_model}")
            else:
                print(f"[INFO] No specialist flag provided for model {i+1}, assuming single encoder")
            
            # Create model-specific output directory
            model_output_dir = os.path.join(out_dir, f"model_{i+1}_{model_name}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Get configs for this model
            config_data_path, config_model_path = args.configs[i].split(':')
            
            # Run evaluate_artifact.py for this model
            print(f"[OK] Running evaluation for {model_name}...")
            eval_results = _run_evaluate_artifact(
                artifact_id, config_data_path, config_model_path,
                is_specialist_model, model_output_dir, device, args.no_wandb,
                training_file, evaluation_file
            )
            
            # Collect results
            model_results = _collect_model_results(model_output_dir, model_name)
            all_model_results[model_name] = model_results
            
            # Store metadata
            model_metadata[model_name] = {
                'artifact_id': artifact_id,
                'model_type': 'specialist' if is_specialist_model else 'single_encoder',
                'configs': args.configs[i],
                'output_directory': model_output_dir
            }
            
            print(f"[OK] Completed evaluation of {model_name}")
            
            # Clear GPU memory if using CUDA
            if device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[OK] Cleared GPU memory for {model_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {artifact_id}: {e}")
            continue

    if not all_model_results:
        print("[ERROR] No models were successfully evaluated")
        return

    # Compute unified metrics
    print(f"\n[OK] Computing unified metrics across {len(all_model_results)} models...")
    unified_metrics = _compute_unified_metrics(all_model_results)
    
    # Save results
    _save_json(all_model_results, os.path.join(out_dir, 'all_model_results.json'))
    _save_json(unified_metrics, os.path.join(out_dir, 'unified_metrics.json'))
    _save_json(model_metadata, os.path.join(out_dir, 'model_metadata.json'))

    # Create comparative visualizations
    print(f"\n[OK] Creating comparative visualizations...")
    viz_paths = _create_comparative_visualizations(all_model_results, out_dir, wandb_logger)

    # Log unified metrics to WandB
    if wandb_logger:
        try:
            # Log dataset info
            wandb_logger.log({
                "unified_dataset/num_training_keys": len(training_dataset),
                "unified_dataset/num_evaluation_keys": len(evaluation_dataset),
                "unified_dataset/total_training_samples": dataset_info['total_training_samples'],
                "unified_dataset/total_evaluation_samples": dataset_info['total_evaluation_samples']
            })
            
            # Log unified metrics for each model
            for metric_path, metric_value in unified_metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric, value in metric_value.items():
                        wandb_logger.log({f"unified_metrics/{metric_path}/{sub_metric}": value})
                else:
                    wandb_logger.log({f"unified_metrics/{metric_path}": metric_value})
            
            print(f"[OK] Logged unified metrics to WandB")
        except Exception as e:
            print(f"[WARNING] Failed to log unified metrics to WandB: {e}")

    # Generate summary report
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Models evaluated: {len(all_model_results)}")
    print(f"Training keys: {len(training_dataset)}")
    print(f"Evaluation keys: {len(evaluation_dataset)}")
    print(f"Support samples per key: {args.n_samples}")
    print(f"Query samples per key: {args.n_queries}")
    print(f"Training samples total: {args.n_training}")
    
    # Show top performers
    if unified_metrics:
        print(f"\nTop performers by exact accuracy:")
        exact_accuracies = []
        for model_name, metrics in unified_metrics.items():
            if 'query' in metrics and 'exact_accuracy' in metrics['query']:
                exact_accuracies.append((model_name, metrics['query']['exact_accuracy']))
        
        exact_accuracies.sort(key=lambda x: x[1], reverse=True)
        for i, (model_name, acc) in enumerate(exact_accuracies[:3]):
            print(f"  {i+1}. {model_name}: {acc:.4f}")

    print(f"\n[OK] Multi-artifact evaluation complete. Outputs in: {out_dir}")
    
    # Finish WandB run
    if wandb_logger:
        try:
            wandb_logger.finish()
            print(f"[OK] WandB run completed successfully")
        except Exception as e:
            print(f"[WARNING] Failed to finish WandB run: {e}")


if __name__ == '__main__':
    main()
