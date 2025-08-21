#!/usr/bin/env python3
"""
Evaluate a model checkpoint downloaded from a WandB artifact.

Produces:
  (a) Single latent plot using one epoch worth of training samples; logs task distance metrics
  (b) Full evaluation with support/query per evaluation settings; logs accuracies, before/after optimization
      distance metrics, and a latent space plot with support and query samples
  (c) A reconstruction figure

QUICK START:
  # Regular model evaluation
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest
  
  # Specialist model evaluation (uses model_specialist_settings.json)
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --specialist
  
  # Custom settings and output directory
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --config my_settings.json --run_dir ./my_outputs

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
    args = parser.parse_args()

    device = torch.device(args.device)

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
    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = _download_artifact(args.artifact, tmpdir)
        ckpt_path = _find_checkpoint_path(art_dir)
        model = _load_model_from_checkpoint(ckpt_path, device)

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
        else:
            print(f"[INFO] Could not generate a training sample for reconstruction")
    except Exception as e:
        print(f"[WARNING] Section (c) failed: {e}")

    print(f"\n[OK] Artifact evaluation complete. Outputs in: {out_dir}")


if __name__ == '__main__':
    main()


