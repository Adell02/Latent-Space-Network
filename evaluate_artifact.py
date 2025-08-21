#!/usr/bin/env python3
"""
Evaluate a model checkpoint downloaded from a WandB artifact.

Produces:
  (a) Single latent plot using one epoch worth of training samples; logs task distance metrics
  (b) Full evaluation with support/query per evaluation settings; logs accuracies, before/after optimization
      distance metrics, and a latent space plot with support and query samples
  (c) A reconstruction figure

QUICK START:
  # Regular model evaluation (with WandB logging)
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest
  
  # Specialist model evaluation (uses model_specialist_settings.json)
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --specialist
  
  # Custom settings and output directory
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --config my_settings.json --run_dir ./my_outputs
  
  # Custom WandB project and entity
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --wandb_project my_project --wandb_entity my_entity
  
  # Disable WandB logging
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --no_wandb

Usage:
  python evaluate_artifact.py --artifact ENTITY/PROJECT/ARTIFACT:VERSION [--specialist] [--config model_settings.json] [--run_dir out_dir]

Notes:
  - The script relies on settings JSON (model_settings.json or model_specialist_settings.json) to initialize
    the model architecture and data/evaluation settings. Use --config to override.
  - The artifact is expected to contain one of: checkpoint_epoch*.pt or final_model.pt
"""

import os
import sys
import argparse
import shutil
import tempfile
import numpy as np
import torch


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
                # checkpoint_epoch{E}.pt
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


def _generate_one_epoch_training_dataset(settings, total_samples: int):
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
    # Distribute samples roughly evenly across keys
    per_key = max(1, total_samples // max(1, len(training_keys)))
    inputs, outputs, keys = [], [], []
    for key in training_keys:
        try:
            _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, per_key)
            if not in_seqs or not out_seqs:
                continue
            inputs.extend(in_seqs)
            outputs.extend(out_seqs)
            keys.extend([key] * len(in_seqs))
        except Exception as e:
            print(f"[WARNING] Could not generate training samples for key {key}: {e}")
    # Trim to requested total
    if len(inputs) > total_samples:
        inputs = inputs[:total_samples]
        outputs = outputs[:total_samples]
        keys = keys[:total_samples]
    return inputs, outputs, keys


def _compute_distance_metrics(latents: np.ndarray, keys: list):
    from utils.latent_metrics import compute_task_distance_metrics
    return compute_task_distance_metrics(latent_data=latents, task_keys=keys, encoder_indices=None, distance_metric='cosine', normalize=True)


def _save_json(obj, path):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a WandB artifact model')
    parser.add_argument('--artifact', required=True, help='WandB artifact id, e.g., entity/project/artifact:version')
    parser.add_argument('--specialist', action='store_true', help='Use specialist settings and multi-encoder visualizations')
    parser.add_argument('--config', type=str, default=None, help='Path to settings JSON (overrides default)')
    parser.add_argument('--run_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_entity', type=str, default='ga624-imperial-college-london', help='WandB entity name')
    parser.add_argument('--wandb_project', type=str, default='evaluation_lpn', help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Initialize WandB
    wandb_logger = None
    if not args.no_wandb:
        try:
            import wandb
            # Create unique run name with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"checkpoint_{timestamp}"
            
            # Initialize WandB run
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                config={
                    "artifact_id": args.artifact,
                    "is_specialist": args.specialist,
                    "config_file": args.config,
                    "device": str(device),
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

    # Create output directory
    if args.run_dir:
        out_dir = args.run_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        from utils.model_utils import create_run_directory
        out_dir = create_run_directory('artifact_eval')

    # Download artifact and load model
    # Note: We need to initialize WandB first to download artifacts, even if --no_wandb is used
    # The artifact download will be logged to the temporary run, then we'll finish it
    temp_wandb_run = None
    if not args.no_wandb:
        # Use the main wandb_logger
        temp_wandb_run = wandb_logger
    else:
        # Create a temporary WandB run just for artifact download
        try:
            import wandb
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_run_name = f"temp_download_{timestamp}"
            temp_wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=temp_run_name,
                config={"purpose": "artifact_download_only"},
                mode="disabled"  # Don't actually log anything
            )
            print(f"[OK] Created temporary WandB run for artifact download: {temp_run_name}")
        except Exception as e:
            print(f"[WARNING] Failed to create temporary WandB run: {e}")
            temp_wandb_run = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if temp_wandb_run:
            # Download using WandB
            art_dir = _download_artifact(args.artifact, tmpdir)
        else:
            # Fallback: try to download without WandB (may fail)
            print(f"[WARNING] Attempting artifact download without WandB (may fail)")
            try:
                art_dir = _download_artifact(args.artifact, tmpdir)
            except Exception as e:
                print(f"[ERROR] Artifact download failed: {e}")
                print(f"[INFO] Please ensure WandB is available or provide the model file manually")
                return
        
        ckpt_path = _find_checkpoint_path(art_dir)
        model = _load_model_from_checkpoint(ckpt_path, device)
    
    # Clean up temporary WandB run if it was created
    if temp_wandb_run and args.no_wandb and temp_wandb_run != wandb_logger:
        try:
            temp_wandb_run.finish()
            print(f"[OK] Cleaned up temporary WandB run")
        except Exception as e:
            print(f"[WARNING] Failed to clean up temporary WandB run: {e}")

    # (a) One-epoch training latent plot + distances
    try:
        train_settings = settings.get_training_settings()
        batches_per_epoch = int(train_settings.get('batches_per_epoch', 10))
        batch_size = int(train_settings.get('batch_size', 16))
        total_samples = max(1, batches_per_epoch * batch_size)
        print(f"[A] Collecting one-epoch training samples: {batches_per_epoch} x {batch_size} = {total_samples}")
        in_seqs, out_seqs, key_list = _generate_one_epoch_training_dataset(settings, total_samples)

        from utils.model_utils import prepare_dataloader
        dl = prepare_dataloader(in_seqs, out_seqs, batch_size=min(64, max(4, batch_size)), shuffle=False)

        # Collect latents (means) and keys
        model.eval()
        all_latents = []
        all_keys = []
        with torch.no_grad():
            for batch in dl:
                x, y = batch[:2]
                x = x.to(device)
                y = y.to(device)
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    mu, log_var = model(x, y)[1:3]
                else:
                    mu, log_var,_ = model.encoder(x, y)
                z = model.reparameterize(mu, log_var)
                all_latents.append(z.detach().cpu().numpy())
        if all_latents:
            all_latents = np.concatenate(all_latents, axis=0)
            # Align keys length if needed
            if len(key_list) != len(all_latents):
                if len(key_list) > len(all_latents):
                    key_list = key_list[:len(all_latents)]
                else:
                    key_list = key_list + [key_list[-1]] * (len(all_latents) - len(key_list))
            dist_metrics = _compute_distance_metrics(all_latents, key_list)
            _save_json(dist_metrics, os.path.join(out_dir, 'training_latent_distance_metrics.json'))

            # Log training latent distance metrics to WandB
            if wandb_logger:
                try:
                    # Log key distance metrics
                    wandb_logger.log({
                        "training_latent_distances/within_task_mean": dist_metrics.get('within_task_mean', 0.0),
                        "training_latent_distances/within_task_std": dist_metrics.get('within_task_std', 0.0),
                        "training_latent_distances/between_task_mean": dist_metrics.get('between_task_mean', 0.0),
                        "training_latent_distances/between_task_std": dist_metrics.get('between_task_std', 0.0),
                        "training_latent_distances/separation_ratio": dist_metrics.get('separation_ratio', 0.0),
                        "training_latent_distances/num_tasks": dist_metrics.get('num_tasks', 0),
                        "training_latent_distances/total_samples": dist_metrics.get('total_samples', 0)
                    })
                    print(f"[OK] Logged training latent distance metrics to WandB")
                except Exception as e:
                    print(f"[WARNING] Failed to log training metrics to WandB: {e}")

            # Plot t-SNE of training latents
            try:
                import matplotlib.pyplot as plt
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(all_latents)//4)))
                coords = tsne.fit_transform(all_latents)
                plt.figure(figsize=(12, 10))
                uniq = sorted(list(set(key_list)))
                colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(uniq)}
                for (xv, yv), k in zip(coords, key_list):
                    plt.scatter(xv, yv, c=[colors[k]], s=20, alpha=0.7, edgecolors='k', linewidths=0.2)
                plt.title('Training Latent Space (one epoch)')
                plt.savefig(os.path.join(out_dir, 'training_latent_space_epoch.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Upload training latent space plot to WandB
                if wandb_logger:
                    try:
                        wandb_logger.log({"training_latent_space": wandb.Image(os.path.join(out_dir, 'training_latent_space_epoch.png'))})
                        print(f"[OK] Uploaded training latent space plot to WandB")
                    except Exception as e:
                        print(f"[WARNING] Failed to upload training plot to WandB: {e}")
            except Exception as e:
                print(f"[WARNING] Could not plot training latent t-SNE: {e}")
        else:
            print("[INFO] No training latents collected")
    except Exception as e:
        print(f"[WARNING] Section (a) failed: {e}")

    # (b) Evaluation respecting evaluation settings
    try:
        from evaluation import main_test
        eval_settings = settings.get_evaluation_settings()
        keys = eval_settings.get('eval_keys', ['00d62c1b'])
        # Resolve "all" using helper
        try:
            from utils.evaluation_utils import get_evaluation_keys_with_all_support
            keys = get_evaluation_keys_with_all_support(keys, eval_settings.get('n_max_eval_keys', 10))
        except Exception:
            pass
        n_samples = int(eval_settings.get('eval_n_samples', 5))
        n_queries = int(eval_settings.get('eval_n_queries', 10))
        seed = settings.get_data_settings().get('eval_seed', 1)

        print(f"[B] Running evaluation on {len(keys)} keys (samples={n_samples}, queries={n_queries})")
        eval_results = main_test(model, keys, out_dir, n_samples, n_queries, seed, device=device,
                                 encoder_idx=None, use_independent_decoder=False)
        # Save raw results
        _save_json(eval_results.get('aggregated_metrics', {}), os.path.join(out_dir, 'eval_aggregated_metrics.json'))
        
        # Log evaluation metrics to WandB
        if wandb_logger and 'aggregated_metrics' in eval_results:
            try:
                agg_metrics = eval_results['aggregated_metrics']
                avg_metrics = agg_metrics.get('average_metrics', {})
                
                # Log key evaluation metrics
                wandb_logger.log({
                    "evaluation/shape_accuracy": avg_metrics.get('avg_shape_accuracy', 0.0),
                    "evaluation/grid_accuracy": avg_metrics.get('avg_grid_accuracy', 0.0),
                    "evaluation/sample_exact_accuracy": avg_metrics.get('avg_sample_exact_accuracy', 0.0),
                    "evaluation/support_loss": avg_metrics.get('avg_support_loss', 0.0),
                    "evaluation/query_loss": avg_metrics.get('avg_query_loss', 0.0),
                    "evaluation/total_keys": agg_metrics.get('total_keys', 0),
                    "evaluation/successful_evaluations": agg_metrics.get('successful_evaluations', 0),
                    "evaluation/failed_evaluations": agg_metrics.get('failed_evaluations', 0)
                })
                print(f"[OK] Logged evaluation metrics to WandB")
            except Exception as e:
                print(f"[WARNING] Failed to log evaluation metrics to WandB: {e}")

        # Split logging for support and query with consistent grouping
        if wandb_logger:
            try:
                # Determine grouping: enc_{i} or poe
                eval_md = eval_results.get('evaluation_metadata', {})
                enc_idx_used = eval_md.get('encoder_idx', None)
                group = f"enc_{enc_idx_used}" if enc_idx_used is not None else "poe"

                # Support metrics
                support_metrics = eval_results.get('support_metrics', {})
                if support_metrics:
                    num_samples_sup = int(support_metrics.get('num_samples', 0))
                    pre = support_metrics.get('pre_opt', {})
                    post = support_metrics.get('post_opt', {})
                    wandb_logger.log({
                        f"support/{group}/num_samples": num_samples_sup,
                        f"support/{group}/pre-opt/shape_accuracy": float(pre.get('shape_accuracy', 0.0)),
                        f"support/{group}/pre-opt/grid_accuracy": float(pre.get('grid_accuracy', 0.0)),
                        f"support/{group}/pre-opt/exact_accuracy": float(pre.get('exact_accuracy', 0.0)),
                        f"support/{group}/pre-opt/shape_loss": float(pre.get('shape_loss', 0.0)),
                        f"support/{group}/pre-opt/grid_loss": float(pre.get('grid_loss', 0.0)),
                        f"support/{group}/pre-opt/kl_loss": float(pre.get('kl_loss', 0.0)),
                        f"support/{group}/post-opt/shape_accuracy": float(post.get('shape_accuracy', 0.0)),
                        f"support/{group}/post-opt/grid_accuracy": float(post.get('grid_accuracy', 0.0)),
                        f"support/{group}/post-opt/exact_accuracy": float(post.get('exact_accuracy', 0.0)),
                        f"support/{group}/post-opt/shape_loss": float(post.get('shape_loss', 0.0)),
                        f"support/{group}/post-opt/grid_loss": float(post.get('grid_loss', 0.0)),
                        f"support/{group}/post-opt/final_opt_loss": float(post.get('final_opt_loss', 0.0)),
                    })

                # Query metrics
                query_metrics = eval_results.get('query_metrics', {})
                if query_metrics:
                    wandb_logger.log({
                        f"query/{group}/num_samples": int(query_metrics.get('num_samples', 0)),
                        f"query/{group}/shape_accuracy": float(query_metrics.get('shape_accuracy', 0.0)),
                        f"query/{group}/grid_accuracy": float(query_metrics.get('grid_accuracy', 0.0)),
                        f"query/{group}/exact_accuracy": float(query_metrics.get('exact_accuracy', 0.0)),
                        f"query/{group}/shape_loss": float(query_metrics.get('shape_loss', 0.0)),
                        f"query/{group}/grid_loss": float(query_metrics.get('grid_loss', 0.0)),
                    })
                print(f"[OK] Logged split support/query metrics to WandB")
            except Exception as e:
                print(f"[WARNING] Failed split metrics logging: {e}")

        # Compute before (encoder means) vs after (optimized) distance metrics for support
        try:
            # Build support dataset again (same keys, n_samples)
            from evaluation import prepare_dataloader_with_keys
            from re_arc.main import generate_and_process_tasks
            support_inputs = []
            support_outputs = []
            support_keys = []
            for key in keys:
                all_needed = n_samples
                _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, all_needed)
                if not in_seqs or not out_seqs:
                    continue
                support_inputs.extend(in_seqs[:n_samples])
                support_outputs.extend(out_seqs[:n_samples])
                support_keys.extend([key] * min(len(in_seqs), n_samples))
            if support_inputs:
                sup_dl, sup_key_map = prepare_dataloader_with_keys(support_inputs, support_outputs, support_keys, batch_size=min(16, n_samples), shuffle=False)
                # Before: encoder means
                pre_latents = []
                pre_keys = []
                with torch.no_grad():
                    for batch in sup_dl:
                        x, y, kidx = batch
                        x = x.to(device); y = y.to(device)
                        if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                            mu, log_var = model(x, y)[1:3]
                        else:
                            mu, log_var,_ = model.encoder(x, y)
                        z = model.reparameterize(mu, log_var)
                        pre_latents.append(z.detach().cpu().numpy())
                        for idx in kidx:
                            pre_keys.append(sup_key_map[idx.item()])
                pre_latents = np.concatenate(pre_latents, axis=0) if pre_latents else np.zeros((0, getattr(model, 'latent_dim', 64)))

                # After: optimized latents from eval results (support)
                post_latents = []
                post_keys = []
                try:
                    # eval_results['evaluation_latent_data']['support']['poe']['latent_zs'] are collected during optimization
                    s_data = eval_results.get('evaluation_latent_data', {}).get('support', {})
                    poe = s_data.get('poe', {})
                    post_latents = np.array(poe.get('latent_zs', []))
                    post_keys = list(poe.get('keys', []))
                except Exception:
                    post_latents = np.zeros((0, getattr(model, 'latent_dim', 64)))
                    post_keys = []

                pre_metrics = _compute_distance_metrics(pre_latents, pre_keys) if len(pre_latents) > 0 else {}
                post_metrics = _compute_distance_metrics(post_latents, post_keys) if len(post_latents) > 0 else {}
                _save_json({'pre_optimization': pre_metrics, 'post_optimization': post_metrics}, os.path.join(out_dir, 'support_distance_metrics_pre_post.json'))

                # Log support distance metrics to WandB with consistent grouping
                if wandb_logger:
                    try:
                        eval_md = eval_results.get('evaluation_metadata', {})
                        enc_idx_used = eval_md.get('encoder_idx', None)
                        group = f"enc_{enc_idx_used}" if enc_idx_used is not None else "poe"
                        # Log pre-optimization metrics
                        if pre_metrics:
                            wandb_logger.log({
                                f"support/{group}/pre-opt/latent_distances/within_task_mean": pre_metrics.get('within_task_mean', 0.0),
                                f"support/{group}/pre-opt/latent_distances/within_task_std": pre_metrics.get('within_task_std', 0.0),
                                f"support/{group}/pre-opt/latent_distances/between_task_mean": pre_metrics.get('between_task_mean', 0.0),
                                f"support/{group}/pre-opt/latent_distances/between_task_std": pre_metrics.get('between_task_std', 0.0),
                                f"support/{group}/pre-opt/latent_distances/separation_ratio": pre_metrics.get('separation_ratio', 0.0)
                            })
                        # Log post-optimization metrics
                        if post_metrics:
                            wandb_logger.log({
                                f"support/{group}/post-opt/latent_distances/within_task_mean": post_metrics.get('within_task_mean', 0.0),
                                f"support/{group}/post-opt/latent_distances/within_task_std": post_metrics.get('within_task_std', 0.0),
                                f"support/{group}/post-opt/latent_distances/between_task_mean": post_metrics.get('between_task_mean', 0.0),
                                f"support/{group}/post-opt/latent_distances/between_task_std": post_metrics.get('between_task_std', 0.0),
                                f"support/{group}/post-opt/latent_distances/separation_ratio": post_metrics.get('separation_ratio', 0.0)
                            })
                        print(f"[OK] Logged support distance metrics to WandB")
                    except Exception as e:
                        print(f"[WARNING] Failed to log support distance metrics to WandB: {e}")
        except Exception as e:
            print(f"[WARNING] Could not compute pre/post distance metrics: {e}")

        # Latent space plot with support and query samples
        try:
            from utils.visualizers import plot_evaluation_latent_space_by_key_and_encoder
            plot_evaluation_latent_space_by_key_and_encoder(
                eval_results=eval_results,
                save_dir=out_dir,
                epoch=1,
                wandb_logger=None,
                use_task_optimization=False
            )
            # Compute and log query latent distances (poe or per-encoder)
            if wandb_logger:
                try:
                    from utils.latent_metrics import compute_task_distance_metrics
                    eval_md = eval_results.get('evaluation_metadata', {})
                    enc_idx_used = eval_md.get('encoder_idx', None)
                    group = f"enc_{enc_idx_used}" if enc_idx_used is not None else "poe"
                    q = eval_results.get('evaluation_latent_data', {}).get('query', {})
                    # Prefer poe, fallback to any available encoder_* key
                    if 'poe' in q:
                        q_latents = np.array(q['poe'].get('latent_zs', []))
                        q_keys = list(q['poe'].get('keys', []))
                    else:
                        # pick first encoder_* entry
                        enc_keys = [k for k in q.keys() if k.startswith('encoder_')]
                        q_latents = np.array(q[enc_keys[0]].get('latent_zs', [])) if enc_keys else np.zeros((0, getattr(model, 'latent_dim', 64)))
                        q_keys = list(q[enc_keys[0]].get('keys', [])) if enc_keys else []
                    if q_latents.size > 0 and len(q_keys) == len(q_latents):
                        q_metrics = compute_task_distance_metrics(latent_data=q_latents, task_keys=q_keys, encoder_indices=None, distance_metric='cosine', normalize=True)
                        wandb_logger.log({
                            f"query/{group}/latent_distances/within_task_mean": q_metrics.get('within_task_mean', 0.0),
                            f"query/{group}/latent_distances/within_task_std": q_metrics.get('within_task_std', 0.0),
                            f"query/{group}/latent_distances/between_task_mean": q_metrics.get('between_task_mean', 0.0),
                            f"query/{group}/latent_distances/between_task_std": q_metrics.get('between_task_std', 0.0),
                            f"query/{group}/latent_distances/separation_ratio": q_metrics.get('separation_ratio', 0.0)
                        })
                        print(f"[OK] Logged query latent distance metrics to WandB")
                except Exception as e:
                    print(f"[WARNING] Failed to log query distances: {e}")
        except Exception as e:
            print(f"[WARNING] Could not create evaluation latent space plot: {e}")

    except Exception as e:
        print(f"[WARNING] Section (b) failed: {e}")

    # (c) Simple reconstruction figure
    try:
        # Generate one training sample and reconstruct
        from re_arc.main import generate_and_process_tasks
        data_settings = settings.get_data_settings()
        training_keys = data_settings.get('training_keys', [data_settings.get('key')])
        if isinstance(training_keys, str):
            training_keys = [training_keys]
        sample_key = training_keys[0]
        _, _, _, in_seqs, out_seqs = generate_and_process_tasks(sample_key, 1)
        if in_seqs and out_seqs:
            x = torch.tensor(in_seqs[0]).float().unsqueeze(0).to(device)
            y = torch.tensor(out_seqs[0]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    mu, log_var = model(x, y)[1:3]
                    z = model.reparameterize(mu, log_var)
                    shape_logits, grid_logits = model.multi_encoder.shared_decoder(z, x, target_seq=y)
                else:
                    mu, log_var,_ = model.encoder(x, y)
                    z = model.reparameterize(mu, log_var)
                    shape_logits, grid_logits = model.decoder(z, x, target_seq=y)
                shape_pred = torch.argmax(shape_logits, dim=-1)[0].cpu().numpy()
                grid_pred = torch.argmax(grid_logits, dim=-1)[0].cpu().numpy()
            # Rebuild grids
            from utils.data_preparation import extract_grid_from_sequence
            input_grid, input_shape = extract_grid_from_sequence(in_seqs[0])
            target_grid, target_shape = extract_grid_from_sequence(out_seqs[0])
            recon_seq = out_seqs[0].copy()
            recon_seq[900:902] = shape_pred
            if len(shape_pred) >= 2 and int(shape_pred[0]) > 0 and int(shape_pred[1]) > 0:
                recon_seq[:min(len(grid_pred), 900)] = grid_pred[:min(len(grid_pred), 900)]
            recon_grid, recon_shape = extract_grid_from_sequence(recon_seq)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(input_grid, cmap='viridis', interpolation='nearest'); axes[0].set_title(f'Input\n{input_shape[0]}×{input_shape[1]}'); axes[0].axis('off')
            axes[1].imshow(target_grid, cmap='viridis', interpolation='nearest'); axes[1].set_title(f'Target\n{target_shape[0]}×{target_shape[1]}'); axes[1].axis('off')
            axes[2].imshow(recon_grid, cmap='viridis', interpolation='nearest'); axes[2].set_title(f'Reconstruction\n{recon_shape[0]}×{recon_shape[1]}'); axes[2].axis('off')
            plt.tight_layout()
            out_path = os.path.join(out_dir, 'single_reconstruction.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved reconstruction: {out_path}")
            
            # Upload reconstruction figure to WandB
            if wandb_logger:
                try:
                    wandb_logger.log({"reconstruction_figure": wandb.Image(out_path)})
                    print(f"[OK] Uploaded reconstruction figure to WandB")
                except Exception as e:
                    print(f"[WARNING] Failed to upload reconstruction to WandB: {e}")
        else:
            print(f"[INFO] Could not generate a training sample for reconstruction")
    except Exception as e:
        print(f"[WARNING] Section (c) failed: {e}")

    print(f"\n[OK] Artifact evaluation complete. Outputs in: {out_dir}")
    
    # Finish WandB run
    if wandb_logger:
        try:
            wandb_logger.finish()
            print(f"[OK] WandB run completed successfully")
        except Exception as e:
            print(f"[WARNING] Failed to finish WandB run: {e}")


if __name__ == '__main__':
    main()


