#!/usr/bin/env python3
"""
Evaluate multiple model checkpoints downloaded from WandB artifacts on the same dataset.

This script ensures fair comparison by:
1. Using the same evaluation dataset for all models
2. Computing unified metrics with consistent naming
3. Generating comparative analysis and visualizations
4. Logging all results to a single WandB run for easy comparison

Produces:
  - Individual model evaluations (same as evaluate_artifact.py)
  - Comparative metrics across all models
  - Unified visualizations (t-SNE plots, distance metrics)
  - Summary statistics and rankings

QUICK START:
  # Evaluate multiple checkpoints of the same model
  python evaluate_all_artifacts.py --artifacts \
    ga624-imperial-college-london/LPN_specialist_paper/model:epoch_10 \
    ga624-imperial-college-london/LPN_specialist_paper/model:epoch_20 \
    ga624-imperial-college-london/LPN_specialist_paper/model:final
  
  # Evaluate different models on same dataset
  python evaluate_all_artifacts.py --artifacts \
    entity1/project1/model1:latest \
    entity2/project2/model2:v1 \
    --config model_settings.json
  
  # Specialist models with custom settings
  python evaluate_all_artifacts.py --artifacts \
    entity/project/specialist_model:latest \
    --specialist --config model_specialist_settings.json

Usage:
  python evaluate_all_artifacts.py --artifacts ARTIFACT1 ARTIFACT2 ... [--specialist] [--config config.json] [--run_dir out_dir]

Notes:
  - All models are evaluated on the same dataset for fair comparison
  - Results are logged to a single WandB run with model-specific prefixes
  - Comparative metrics and visualizations are generated
  - Use --config to specify settings for all models
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import torch
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def _init_settings(config_path: str):
    from utils.settings_manager import init_settings
    return init_settings(config_path)


def _resolve_settings_path(is_specialist: bool, config_override: str = None) -> str:
    if config_override and os.path.exists(config_override):
        return config_override
    default = 'model_specialist_settings.json' if is_specialist else 'model_settings.json'
    if not os.path.exists(default):
        raise FileNotFoundError(f"Settings file not found: {default}. Use --config to provide a path.")
    return default


def _download_artifact(artifact_id: str, dest_dir: str) -> str:
    import wandb
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[OK] Downloading artifact '{artifact_id}' → {dest_dir}")
    art = wandb.use_artifact(artifact_id, type='model')
    path = art.download(root=dest_dir)
    print(f"[OK] Artifact downloaded to {path}")
    return path


def _find_checkpoint_path(artifact_dir: str) -> str:
    # Prefer checkpoint_epoch*.pt, else final_model.pt
    candidates = []
    for root, _, files in os.walk(artifact_dir):
        for f in files:
            if f.startswith('checkpoint_epoch') and f.endswith('.pt'):
                candidates.append(os.path.join(root, f))
    if candidates:
        # Pick the highest epoch
        def _epoch_from_name(p):
            name = os.path.basename(p)
            try:
                e = int(name.split('checkpoint_epoch')[1].split('.pt')[0])
            except Exception:
                e = -1
            return e
        best = max(candidates, key=_epoch_from_name)
        print(f"[OK] Selected checkpoint: {best}")
        return best
    # Fallback
    for root, _, files in os.walk(artifact_dir):
        for f in files:
            if f == 'final_model.pt':
                p = os.path.join(root, f)
                print(f"[OK] Selected final model: {p}")
                return p
    raise FileNotFoundError(f"No checkpoint file found in artifact at {artifact_dir}")


def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    from models.base_model import LatentProgramNetwork
    model = LatentProgramNetwork().to(device)
    print(f"[OK] Loading state dict from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _generate_unified_evaluation_dataset(settings, n_samples: int = 5, n_queries: int = 10):
    """Generate a fixed dataset that will be used for all model evaluations"""
    from re_arc.main import generate_and_process_tasks
    from utils.evaluation_utils import get_evaluation_keys_with_all_support
    
    eval_settings = settings.get_evaluation_settings()
    keys = eval_settings.get('eval_keys', ['00d62c1b'])
    
    # Use config file settings if available
    if hasattr(settings, 'evaluation_config'):
        config = settings.evaluation_config
        if 'evaluation_settings' in config:
            eval_config = config['evaluation_settings']
            keys = eval_config.get('eval_keys', keys)
            n_samples = eval_config.get('eval_n_samples', n_samples)
            n_queries = eval_config.get('eval_n_queries', n_queries)
    
    # Resolve "all" using helper
    try:
        keys = get_evaluation_keys_with_all_support(keys, eval_settings.get('n_max_eval_keys', 10))
    except Exception:
        pass
    
    # Generate fixed dataset for all models
    unified_dataset = {}
    for key in keys:
        try:
            _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, n_samples + n_queries)
            if in_seqs and out_seqs:
                unified_dataset[key] = {
                    'support_inputs': in_seqs[:n_samples],
                    'support_outputs': out_seqs[:n_samples],
                    'query_inputs': in_seqs[n_samples:n_samples + n_queries],
                    'query_outputs': out_seqs[n_samples:n_samples + n_queries]
                }
        except Exception as e:
            print(f"[WARNING] Could not generate data for key {key}: {e}")
    
    print(f"[OK] Generated unified dataset with {len(unified_dataset)} keys")
    return unified_dataset


def _evaluate_model_on_unified_dataset(model, unified_dataset, device, encoder_idx=None, use_independent_decoder=False):
    """Evaluate a single model on the unified dataset"""
    from evaluation import evaluate_model_original_bonnet_approach
    from utils.model_utils import prepare_dataloader_with_keys
    
    results = {}
    support_metrics = defaultdict(list)
    query_metrics = defaultdict(list)
    
    for key, data in unified_dataset.items():
        try:
            # Prepare support dataloader
            sup_dl, sup_key_map = prepare_dataloader_with_keys(
                data['support_inputs'], data['support_outputs'], 
                [key] * len(data['support_inputs']), 
                batch_size=min(16, len(data['support_inputs'])), 
                shuffle=False
            )
            
            # Prepare query dataloader
            qry_dl, qry_key_map = prepare_dataloader_with_keys(
                data['query_inputs'], data['query_outputs'], 
                [key] * len(data['query_inputs']), 
                batch_size=min(16, len(data['query_inputs'])), 
                shuffle=False
            )
            
            # Evaluate
            eval_result = evaluate_model_original_bonnet_approach(
                model, sup_dl, qry_dl, device, 
                encoder_idx=encoder_idx, 
                use_independent_decoder=use_independent_decoder,
                support_key_mapping=sup_key_map,
                query_key_mapping=qry_key_map
            )
            
            # Extract metrics
            if 'support_metrics' in eval_result:
                support_metrics[key] = eval_result['support_metrics']
            if 'query_metrics' in eval_result:
                query_metrics[key] = eval_result['query_metrics']
                
            results[key] = eval_result
            
        except Exception as e:
            print(f"[WARNING] Evaluation failed for key {key}: {e}")
            continue
    
    return {
        'per_key_results': results,
        'support_metrics': dict(support_metrics),
        'query_metrics': dict(query_metrics)
    }


def _compute_unified_metrics(all_model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute unified metrics across all models for comparison"""
    unified_metrics = {}
    
    for model_name, model_results in all_model_results.items():
        # Aggregate support metrics
        support_metrics = model_results.get('support_metrics', {})
        if support_metrics:
            pre_opt_metrics = defaultdict(list)
            post_opt_metrics = defaultdict(list)
            
            for key, metrics in support_metrics.items():
                pre = metrics.get('pre_opt', {})
                post = metrics.get('post_opt', {})
                
                for metric_name in ['shape_accuracy', 'grid_accuracy', 'exact_accuracy', 'shape_loss', 'grid_loss', 'kl_loss']:
                    if metric_name in pre:
                        pre_opt_metrics[metric_name].append(pre[metric_name])
                    if metric_name in post:
                        post_opt_metrics[metric_name].append(post[metric_name])
            
            # Compute averages
            unified_metrics[f"{model_name}/support/pre_opt"] = {
                metric: np.mean(values) if values else 0.0 
                for metric, values in pre_opt_metrics.items()
            }
            unified_metrics[f"{model_name}/support/post_opt"] = {
                metric: np.mean(values) if values else 0.0 
                for metric, values in post_opt_metrics.items()
            }
        
        # Aggregate query metrics
        query_metrics = model_results.get('query_metrics', {})
        if query_metrics:
            query_agg = defaultdict(list)
            for key, metrics in query_metrics.items():
                for metric_name in ['shape_accuracy', 'grid_accuracy', 'exact_accuracy', 'shape_loss', 'grid_loss']:
                    if metric_name in metrics:
                        query_agg[metric_name].append(metrics[metric_name])
            
            unified_metrics[f"{model_name}/query"] = {
                metric: np.mean(values) if values else 0.0 
                for metric, values in query_agg.items()
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
                query_metrics = all_model_results[model_name].get('query_metrics', {})
                if query_metrics:
                    # Average across all keys
                    metric_values = [v.get(metric, 0.0) for v in query_metrics.values()]
                    values.append(np.mean(metric_values) if metric_values else 0.0)
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
        
        # 2. Support vs Query comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Support metrics (post-optimization)
        support_metrics = ['shape_accuracy', 'grid_accuracy', 'exact_accuracy']
        support_values = defaultdict(list)
        for model_name in model_names:
            support_data = all_model_results[model_name].get('support_metrics', {})
            if support_data:
                for key, metrics in support_data.items():
                    post = metrics.get('post_opt', {})
                    for metric in support_metrics:
                        if metric in post:
                            support_values[metric].append(post[metric])
        
        for i, metric in enumerate(support_metrics):
            values = [np.mean(support_values[metric]) if support_values[metric] else 0.0]
            axes[0].bar([metric.replace('_', ' ').title()], values, alpha=0.7)
        axes[0].set_title('Support Set Performance (Post-Optimization)')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        
        # Query metrics
        query_metrics = ['shape_accuracy', 'grid_accuracy', 'exact_accuracy']
        query_values = defaultdict(list)
        for model_name in model_names:
            query_data = all_model_results[model_name].get('query_metrics', {})
            if query_data:
                for key, metrics in query_data.items():
                    for metric in query_metrics:
                        if metric in metrics:
                            query_values[metric].append(metrics[metric])
        
        for i, metric in enumerate(query_metrics):
            values = [np.mean(query_values[metric]) if query_values[metric] else 0.0]
            axes[1].bar([metric.replace('_', ' ').title()], values, alpha=0.7)
        axes[1].set_title('Query Set Performance')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        comp_path = os.path.join(save_dir, 'support_vs_query_comparison.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Upload to WandB
        if wandb_logger:
            try:
                wandb_logger.log({
                    "comparative_visualizations/accuracies": wandb.Image(acc_path),
                    "comparative_visualizations/support_vs_query": wandb.Image(comp_path)
                })
                print(f"[OK] Uploaded comparative visualizations to WandB")
            except Exception as e:
                print(f"[WARNING] Failed to upload comparative visualizations: {e}")
        
        return [acc_path, comp_path]
        
    except Exception as e:
        print(f"[WARNING] Could not create comparative visualizations: {e}")
        return []


def _save_json(obj, path):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] Saved: {path}")


def _load_evaluation_config(config_path: str) -> Dict[str, Any]:
    """Load and validate evaluation configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['evaluation_settings', 'data_settings', 'wandb_settings']
        for section in required_sections:
            if section not in config:
                print(f"[WARNING] Missing required section '{section}' in config")
        
        # Set defaults for missing values
        if 'evaluation_settings' not in config:
            config['evaluation_settings'] = {}
        if 'data_settings' not in config:
            config['data_settings'] = {}
        if 'wandb_settings' not in config:
            config['wandb_settings'] = {}
        
        # Set defaults
        config['evaluation_settings'].setdefault('eval_keys', ['00d62c1b'])
        config['evaluation_settings'].setdefault('eval_n_samples', 5)
        config['evaluation_settings'].setdefault('eval_n_queries', 10)
        config['evaluation_settings'].setdefault('eval_seed', 42)
        config['wandb_settings'].setdefault('project', 'evaluation_lpn')
        config['wandb_settings'].setdefault('entity', 'ga624-imperial-college-london')
        
        print(f"[OK] Loaded and validated evaluation config: {config_path}")
        return config
        
    except Exception as e:
        print(f"[ERROR] Failed to load evaluation config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple WandB artifact models on unified dataset')
    parser.add_argument('--artifacts', nargs='+', required=True, 
                       help='List of WandB artifact IDs, e.g., entity/project/artifact:version')
    parser.add_argument('--specialist', action='store_true', 
                       help='Use specialist settings and multi-encoder visualizations')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to settings JSON (overrides default)')
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
    args = parser.parse_args()

    device = torch.device(args.device)
    
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
                    "is_specialist": args.specialist,
                    "config_file": args.config,
                    "device": str(device),
                    "n_samples": args.n_samples,
                    "n_queries": args.n_queries,
                    "evaluation_timestamp": timestamp
                }
            )
            wandb_logger = wandb
            print(f"[OK] WandB run initialized: {args.wandb_entity}/{args.wandb_project}/{run_name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            wandb_logger = None

    # Initialize settings
    settings_path = _resolve_settings_path(args.specialist, args.config)
    print(f"[OK] Using settings: {settings_path}")
    settings = _init_settings(settings_path)
    
    # Load evaluation config if provided
    evaluation_config = None
    if args.config and os.path.exists(args.config):
        evaluation_config = _load_evaluation_config(args.config)
        if evaluation_config:
            # Attach config to settings for use in functions
            settings.evaluation_config = evaluation_config
            # Override command line args with config values
            if 'evaluation_settings' in evaluation_config:
                eval_config = evaluation_config['evaluation_settings']
                args.n_samples = eval_config.get('eval_n_samples', args.n_samples)
                args.n_queries = eval_config.get('eval_n_queries', args.n_queries)
            if 'wandb_settings' in evaluation_config:
                wandb_config = evaluation_config['wandb_settings']
                args.wandb_project = wandb_config.get('project', args.wandb_project)
                args.wandb_entity = wandb_config.get('entity', args.wandb_entity)

    # Create output directory
    if args.run_dir:
        out_dir = args.run_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        from utils.model_utils import create_run_directory
        out_dir = create_run_directory('multi_artifact_eval')

    # Generate unified dataset
    print(f"[OK] Generating unified evaluation dataset...")
    unified_dataset = _generate_unified_evaluation_dataset(
        settings, args.n_samples, args.n_queries
    )
    
    # Save unified dataset info
    dataset_info = {
        'n_samples': args.n_samples,
        'n_queries': args.n_queries,
        'keys': list(unified_dataset.keys()),
        'total_support_samples': sum(len(data['support_inputs']) for data in unified_dataset.values()),
        'total_query_samples': sum(len(data['query_inputs']) for data in unified_dataset.values())
    }
    _save_json(dataset_info, os.path.join(out_dir, 'unified_dataset_info.json'))

    # Evaluate each model
    all_model_results = {}
    model_metadata = {}
    
    for i, artifact_id in enumerate(args.artifacts):
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL {i+1}/{len(args.artifacts)}: {artifact_id}")
        print(f"{'='*60}")
        
        try:
            # Download and load model
            with tempfile.TemporaryDirectory() as tmpdir:
                art_dir = _download_artifact(artifact_id, tmpdir)
                ckpt_path = _find_checkpoint_path(art_dir)
                model = _load_model_from_checkpoint(ckpt_path, device)
            
            # Extract model name from artifact ID
            model_name = artifact_id.split('/')[-1].split(':')[0]
            if model_name in all_model_results:
                model_name = f"{model_name}_{i}"  # Avoid duplicates
            
            # Evaluate on unified dataset
            print(f"[OK] Evaluating {model_name} on unified dataset...")
            eval_results = _evaluate_model_on_unified_dataset(
                model, unified_dataset, device,
                encoder_idx=None,  # Use PoE for multi-encoder models
                use_independent_decoder=False
            )
            
            all_model_results[model_name] = eval_results
            model_metadata[model_name] = {
                'artifact_id': artifact_id,
                'checkpoint_path': ckpt_path,
                'model_type': 'specialist' if args.specialist else 'single_encoder'
            }
            
            print(f"[OK] Completed evaluation of {model_name}")
            
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
                "unified_dataset/num_keys": len(unified_dataset),
                "unified_dataset/total_support_samples": dataset_info['total_support_samples'],
                "unified_dataset/total_query_samples": dataset_info['total_query_samples']
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
    print(f"Dataset keys: {len(unified_dataset)}")
    print(f"Support samples per key: {args.n_samples}")
    print(f"Query samples per key: {args.n_queries}")
    
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
