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
  
  # Separate data and model configs (recommended)
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --config_data data_evaluation_settings.json --config_model model_evaluation_settings.json
  
  # Legacy single config file
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --config my_settings.json --run_dir ./my_outputs
  
  # Custom WandB project and entity
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --wandb_project my_project --wandb_entity my_entity
  
  # Disable WandB logging
  python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/model:latest --no_wandb

Usage:
  python evaluate_artifact.py --artifact ENTITY/PROJECT/ARTIFACT:VERSION [--specialist] [--config_data DATA_CONFIG.json] [--config_model MODEL_CONFIG.json] [--config LEGACY_CONFIG.json] [--run_dir out_dir]

Notes:
  - The script now supports separate configuration files for data/evaluation settings and model architecture.
  - Use --config_data for data, evaluation, and training settings.
  - Use --config_model for model architecture and latent optimization settings.
  - Legacy --config flag is still supported for backward compatibility.
  - The artifact is expected to contain one of: checkpoint_epoch*.pt or final_model.pt


COMMAND:
 - 1ENC: python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/LPN_specialist_paper_20250821_101933_main_20250821_101933_checkpoint_epoch_20:v0 --config_model model_1ENC_evaluation_settings.json --config_data data_evaluation_1ENC_settings.json

 - 2ENC: python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v7 --config_model model_2ENC_evaluation_settings.json --config_data data_evaluation_2ENC_settings.json --specialist

        python evaluate_artifact.py --artifact ga624-imperial-college-london/LPN_specialist_paper/final_specialist_model:v8 --config_model model_2ENC_evaluation_settings.json --config_data data_evaluation_2ENC_settings.json --specialist

 - 4ENC: 
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


def _resolve_settings_path(is_specialist: bool, config_override: str = None, config_data: str = None, config_model: str = None) -> tuple:
    """
    Resolve settings paths for data and model configurations.
    Returns (data_config_path, model_config_path) tuple.
    """
    # If both data and model configs are provided, use them
    if config_data and config_model:
        if not os.path.exists(config_data):
            raise FileNotFoundError(f"Data config file not found: {config_data}")
        if not os.path.exists(config_model):
            raise FileNotFoundError(f"Model config file not found: {config_model}")
        return config_data, config_model
    
    # If only one config is provided, use it for both (fallback)
    if config_data and not config_model:
        if not os.path.exists(config_data):
            raise FileNotFoundError(f"Data config file not found: {config_data}")
        return config_data, config_data
    elif config_model and not config_data:
        if not os.path.exists(config_model):
            raise FileNotFoundError(f"Model config file not found: {config_model}")
        return config_model, config_model
    
    # If config_override is provided, use it for both
    if config_override and os.path.exists(config_override):
        return config_override, config_override
    
    # Default fallback
    default = 'model_specialist_settings.json' if is_specialist else 'model_settings.json'
    if not os.path.exists(default):
        raise FileNotFoundError(f"Settings file not found: {default}. Use --config_data and --config_model to provide separate paths.")
    return default, default


def _download_artifact(artifact_id: str, dest_dir: str) -> str:
    import wandb
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[OK] Downloading artifact '{artifact_id}' → {dest_dir}")
    art = wandb.use_artifact(artifact_id, type='model')
    path = art.download(root=dest_dir)
    print(f"[OK] Artifact downloaded to {path}")
    return path





def _find_checkpoint_path(artifact_dir: str) -> str:
    # Debug: List all files in artifact directory
    print(f"[DEBUG] Searching for checkpoint files in: {artifact_dir}")
    all_files = []
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, artifact_dir)
            all_files.append(rel_path)
            print(f"[DEBUG] Found file: {rel_path}")
    
    # Prefer checkpoint_epoch*.pt, else final_model.pt, else any .pt file
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
    
    # Fallback to final_model.pt
    for root, _, files in os.walk(artifact_dir):
        for f in files:
            if f == 'final_model.pt':
                p = os.path.join(root, f)
                print(f"[OK] Selected final model: {p}")
                return p
    
    # Last resort: look for any checkpoint file (.pt, .ckpt, .pth, etc.)
    checkpoint_files = []
    for root, _, files in os.walk(artifact_dir):
        for f in files:
            if f.endswith(('.pt', '.ckpt', '.pth', '.pkl')):
                checkpoint_files.append(os.path.join(root, f))
    
    if checkpoint_files:
        # Prefer files with 'model' in the name
        model_files = [f for f in checkpoint_files if 'model' in os.path.basename(f).lower()]
        if model_files:
            selected = model_files[0]
            print(f"[OK] Selected model file (fallback): {selected}")
            return selected
        else:
            selected = checkpoint_files[0]
            print(f"[OK] Selected checkpoint file (fallback): {selected}")
            return selected
    
    # If we get here, no checkpoint files found
    print(f"[ERROR] No checkpoint files found. Available files:")
    for f in all_files:
        print(f"  - {f}")
    raise FileNotFoundError(f"No checkpoint file found in artifact at {artifact_dir}")


def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    from models.base_model import LatentProgramNetwork
    model = LatentProgramNetwork().to(device)
    print(f"[OK] Loading state dict from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    
    # Try strict loading first
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[OK] Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"[WARNING] Strict loading failed: {e}")
        print(f"[INFO] Attempting flexible loading (strict=False)...")
        
        # Try flexible loading
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Model loaded successfully with strict=False")
            if missing_keys:
                print(f"[WARNING] Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    print(f"  - First 10: {missing_keys[:10]}")
                    print(f"  - ... and {len(missing_keys) - 10} more")
            
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    print(f"  - First 10: {unexpected_keys[:10]}")
                    print(f"  - ... and {len(unexpected_keys) - 10} more")
                    
        except Exception as e2:
            print(f"[ERROR] Flexible loading also failed: {e2}")
            raise e2
    
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
    
    print(f"[DEBUG] Training keys resolved: {len(training_keys)} keys")
    print(f"[DEBUG] First few keys: {training_keys[:5]}")
    
    # Ensure we get samples from ALL keys by calculating minimum samples per key
    min_samples_per_key = max(1, total_samples // len(training_keys))
    print(f"[DEBUG] Minimum samples per key: {min_samples_per_key}")
    
    inputs, outputs, keys = [], [], []
    successful_keys = 0
    
    for key in training_keys:
        try:
            # Generate at least min_samples_per_key samples for each key
            samples_to_generate = min_samples_per_key
            _, _, _, in_seqs, out_seqs = generate_and_process_tasks(key, samples_to_generate)
            
            if not in_seqs or not out_seqs:
                print(f"[WARNING] No data generated for key {key}")
                continue
                
            # Add all generated samples for this key
            inputs.extend(in_seqs)
            outputs.extend(out_seqs)
            keys.extend([key] * len(in_seqs))
            successful_keys += 1
            
            print(f"[DEBUG] Key {key}: generated {len(in_seqs)} samples")
            
        except Exception as e:
            print(f"[WARNING] Could not generate training samples for key {key}: {e}")
    
    print(f"[DEBUG] Successfully generated samples from {successful_keys}/{len(training_keys)} keys")
    print(f"[DEBUG] Total samples: {len(inputs)}")
    print(f"[DEBUG] Unique keys in output: {len(set(keys))}")
    
    # Trim to requested total if we have more than needed
    if len(inputs) > total_samples:
        print(f"[DEBUG] Trimming from {len(inputs)} to {total_samples} samples")
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
    parser.add_argument('--config', type=str, default=None, help='Path to settings JSON (overrides default, legacy support)')
    parser.add_argument('--config_data', type=str, default=None, help='Path to data/evaluation settings JSON')
    parser.add_argument('--config_model', type=str, default=None, help='Path to model architecture settings JSON')
    parser.add_argument('--run_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_entity', type=str, default='ga624-imperial-college-london', help='WandB entity name')
    parser.add_argument('--wandb_project', type=str, default='evaluation_lpn', help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--unified_training_dataset', type=str, default=None, help='Path to unified training dataset pickle file')
    parser.add_argument('--unified_evaluation_dataset', type=str, default=None, help='Path to unified evaluation dataset pickle file')
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
    data_config_path, model_config_path = _resolve_settings_path(
        args.specialist, args.config, args.config_data, args.config_model
    )
    
    print(f"[OK] Using data config: {data_config_path}")
    print(f"[OK] Using model config: {model_config_path}")
    
    # For now, use the data config as the primary config
    # The model config will be merged later when needed
    settings = _init_settings(data_config_path)
    print(f"[OK] Initialized settings with data configuration")
    
    # Store the model config path for later use if needed
    if data_config_path != model_config_path:
        print(f"[INFO] Model config available at: {model_config_path}")
        print(f"[INFO] Note: Model architecture settings may need to be manually verified")

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
        
        # Check if unified training dataset is provided
        if args.unified_training_dataset and os.path.exists(args.unified_training_dataset):
            print(f"[A] Using unified training dataset from: {args.unified_training_dataset}")
            import pickle
            with open(args.unified_training_dataset, 'rb') as f:
                unified_training_data = pickle.load(f)
            
            # Convert unified data to sequences
            in_seqs, out_seqs, key_list = [], [], []
            for key, data in unified_training_data.items():
                in_seqs.extend(data['inputs'])
                out_seqs.extend(data['outputs'])
                key_list.extend([key] * len(data['inputs']))
            
            print(f"[A] Loaded {len(in_seqs)} training samples from unified dataset")
        else:
            # For multi-encoder models, we need samples for each encoder + PoE
            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                num_encoders = getattr(model, 'num_encoders', 4)
                # Total samples needed: batches × batch_size × (encoders + PoE)
                total_samples = max(1, batches_per_epoch * batch_size * (num_encoders + 1))
                print(f"[A] Multi-encoder model: collecting {batches_per_epoch} x {batch_size} x ({num_encoders} + 1) = {total_samples} samples")
            else:
                total_samples = max(1, batches_per_epoch * batch_size)
                print(f"[A] Single encoder: collecting {batches_per_epoch} x {batch_size} = {total_samples} samples")
            
            in_seqs, out_seqs, key_list = _generate_one_epoch_training_dataset(settings, total_samples)
        print(f"[A] Generated {len(in_seqs)} training samples from {len(set(key_list))} unique keys")

        from utils.model_utils import prepare_dataloader
        dl = prepare_dataloader(in_seqs, out_seqs, batch_size=min(64, max(4, batch_size)), shuffle=False)

        # Collect latents using the SAME method as train_specialist.py
        # This ensures consistency between training and evaluation latent space visualizations
        print(f"[INFO] Collecting training latents using train_specialist.py method...")
        
        # Check if we have stored optimized latents from training (like train_specialist.py)
        if hasattr(model, 'epoch_optimized_latents') and model.epoch_optimized_latents:
            print(f"[INFO] Using stored optimized latents from training: {len(model.epoch_optimized_latents['latents'])} samples")
            all_latents = model.epoch_optimized_latents['latents']
            all_keys = model.epoch_optimized_latents['keys']
            # Note: We don't have encoder indices in evaluation, so we'll use the stored ones if available
            encoder_indices = model.epoch_optimized_latents.get('encoder_indices', [])
        else:
            print(f"[INFO] No stored optimized latents found, collecting fresh latents with optimization...")
            
            # Use the SAME latent collection method as train_specialist.py
            from utils.latent_functions import get_optimized_z
            
            # Get latent optimization settings from model settings
            try:
                from utils.settings_manager import settings
                latent_opt_settings = settings.get_latent_optimization()
                training_opt_steps = latent_opt_settings.get('training', {}).get('num_steps', 0)
                training_opt_lr = latent_opt_settings.get('training', {}).get('learning_rate', 0.1)
                print(f"[INFO] Using latent optimization: {training_opt_steps} steps, LR={training_opt_lr}")
            except Exception as e:
                print(f"[WARNING] Could not load latent optimization settings: {e}")
                training_opt_steps = 0
                training_opt_lr = 0.1
            
            all_latents = []
            all_keys = []
            encoder_indices = []
            
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    # For multi-encoder models, we need to process samples for each encoder + PoE
                    # The key_list now contains samples × (encoders + PoE), so we need to distribute them properly
                    num_encoders = model.num_encoders
                    samples_per_encoder = len(key_list) // (num_encoders + 1)  # +1 for PoE
                    print(f"[DEBUG] Multi-encoder: {len(key_list)} total samples, {samples_per_encoder} per encoder+PoE")
                    
                    # Process samples in chunks for each encoder + PoE
                    for enc_idx in range(-1, num_encoders):  # -1 represents PoE, 0-3 are encoders
                        if enc_idx == -1:
                            # PoE
                            start_idx = 0
                            end_idx = samples_per_encoder
                            encoder_name = "PoE"
                            use_independent_decoder = False
                        else:
                            # Individual encoder
                            start_idx = (enc_idx + 1) * samples_per_encoder
                            end_idx = (enc_idx + 2) * samples_per_encoder
                            encoder_name = f"Encoder {enc_idx}"
                            use_independent_decoder = True
                        
                        # Get the samples for this encoder
                        encoder_samples = list(zip(in_seqs[start_idx:end_idx], out_seqs[start_idx:end_idx]))
                        encoder_keys = key_list[start_idx:end_idx]
                        
                        print(f"[DEBUG] Processing {encoder_name}: samples {start_idx}-{end_idx}, keys: {encoder_keys[:5]}...")
                        
                        # Process in batches
                        batch_size_actual = min(32, len(encoder_samples))
                        for batch_start in range(0, len(encoder_samples), batch_size_actual):
                            batch_end = min(batch_start + batch_size_actual, len(encoder_samples))
                            batch_samples = encoder_samples[batch_start:batch_end]
                            batch_keys = encoder_keys[batch_start:batch_end]
                            
                            # Prepare batch tensors
                            x = torch.tensor([sample[0] for sample in batch_samples]).float().to(device)
                            y = torch.tensor([sample[1] for sample in batch_samples]).float().to(device)
                            
                            try:
                                if enc_idx == -1:
                                    # PoE
                                    z_opt, _ = get_optimized_z(
                                        model, x, y,
                                        num_steps=training_opt_steps,
                                        lr=training_opt_lr,
                                        context='training',
                                        encoder_idx=None,
                                        use_independent_decoder=False
                                    )
                                    all_latents.append(z_opt.detach().cpu().numpy())
                                    all_keys.extend(batch_keys)
                                    encoder_indices.extend([None] * len(batch_keys))
                                    print(f"[DEBUG] Collected PoE optimized latent, batch {batch_start//batch_size_actual}, keys: {batch_keys}")
                                else:
                                    # Individual encoder
                                    z_opt, _ = get_optimized_z(
                                        model, x, y,
                                        num_steps=training_opt_steps,
                                        lr=training_opt_lr,
                                        context='training',
                                        encoder_idx=enc_idx,
                                        use_independent_decoder=True
                                    )
                                    all_latents.append(z_opt.detach().cpu().numpy())
                                    all_keys.extend(batch_keys)
                                    encoder_indices.extend([enc_idx] * len(batch_keys))
                                    print(f"[DEBUG] Collected {encoder_name} optimized latent, batch {batch_start//batch_size_actual}, keys: {batch_keys}")
                            except Exception as e:
                                print(f"[WARNING] Failed to get {encoder_name} optimized latent: {e}")
                else:
                    # Single encoder model - use original logic
                    max_batches = max(20, len(set(key_list)) // 2)
                    print(f"[DEBUG] Single encoder: processing up to {max_batches} batches")
                    
                    for batch_idx, batch in enumerate(dl):
                        if batch_idx >= max_batches:
                            break
                            
                        x, y = batch[:2]
                        x = x.to(device)
                        y = y.to(device)
                        
                        try:
                            z_opt, _ = get_optimized_z(
                                model, x, y,
                                num_steps=training_opt_steps,
                                lr=training_opt_lr,
                                context='training'
                            )
                            all_latents.append(z_opt.detach().cpu().numpy())
                            # Calculate the correct batch indices based on batch_idx and batch_size
                            batch_size_actual = x.size(0)
                            start_idx = batch_idx * batch_size_actual
                            end_idx = start_idx + batch_size_actual
                            batch_keys = key_list[start_idx:end_idx]
                            all_keys.extend(batch_keys)
                            encoder_indices.extend([0] * x.size(0))  # Single encoder = 0
                        except Exception as e:
                            print(f"[WARNING] Failed to get single encoder optimized latent: {e}")
                    else:
                        # Single encoder model
                        try:
                            z_opt, _ = get_optimized_z(
                                model, x, y,
                                num_steps=training_opt_steps,
                                lr=training_opt_lr,
                                context='training'
                            )
                            all_latents.append(z_opt.detach().cpu().numpy())
                            # Calculate the correct batch indices based on batch_idx and batch_size
                            batch_size_actual = x.size(0)
                            start_idx = batch_idx * batch_size_actual
                            end_idx = start_idx + batch_size_actual
                            batch_keys = key_list[start_idx:end_idx]
                            all_keys.extend(batch_keys)
                            encoder_indices.extend([0] * x.size(0))  # Single encoder = 0
                        except Exception as e:
                            print(f"[WARNING] Failed to get single encoder optimized latent: {e}")
            
            print(f"[INFO] Collected {len(all_latents)} optimized latents using train_specialist.py method")
            print(f"[DEBUG] Key distribution: {len(set(all_keys))} unique keys out of {len(all_keys)} total keys")
            print(f"[DEBUG] First 20 keys: {all_keys[:20]}")
            print(f"[DEBUG] Encoder indices: {len(encoder_indices)} indices, unique: {set(encoder_indices)}")
        if all_latents:
            all_latents = np.concatenate(all_latents, axis=0)
            print(f"[A] Collected {len(all_latents)} total latents")
            
            # For multi-encoder models, all_keys already contains the correct number of keys
            # For single-encoder models, align keys length if needed
            if not (hasattr(model, 'is_multi_encoder') and model.is_multi_encoder):
                if len(key_list) != len(all_latents):
                    if len(key_list) > len(all_latents):
                        key_list = key_list[:len(all_latents)]
                    else:
                        key_list = key_list + [key_list[-1]] * (len(all_latents) - len(key_list))
            else:
                # Use all_keys for multi-encoder models
                key_list = all_keys
                print(f"[A] Multi-encoder model: using {len(key_list)} keys for {len(all_latents)} latents")
            
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
                
                if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                    # Specialist model: create t-SNE plots using the SAME method as train_specialist.py
                    # This ensures consistency with training visualizations
                    
                    # Check if we have encoder indices (from stored optimized latents or fresh collection)
                    if 'encoder_indices' in locals() and encoder_indices:
                        print(f"[INFO] Creating t-SNE plots with encoder indices: {len(encoder_indices)} samples")
                        
                        # Create t-SNE of all optimized latents (encoders + PoE)
                        tsne_combined = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(all_latents) - 1))))
                        coords_combined = tsne_combined.fit_transform(all_latents)
                        
                        # Create visualization matching train_specialist.py style
                        unique_keys = sorted(list(set(all_keys)))
                        unique_encoders = sorted(list(set([idx for idx in encoder_indices if idx is not None])))
                        
                        # Create color map for keys (same as train_specialist.py)
                        if len(unique_keys) <= 400:
                            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
                            colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
                            colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
                            colors4 = plt.cm.Paired(np.linspace(0, 1, 12))
                            all_colors = np.vstack([colors1, colors2, colors3, colors4])
                            while len(all_colors) < len(unique_keys):
                                all_colors = np.vstack([all_colors, all_colors])
                            key_colors = {k: all_colors[i % len(all_colors)] for i, k in enumerate(unique_keys)}
                        else:
                            key_colors = {k: plt.cm.viridis(i / len(unique_keys)) for i, k in enumerate(unique_keys)}
                        
                        # Encoder markers (same as train_specialist.py)
                        encoder_markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+']
                        
                        plt.figure(figsize=(16, 12))
                        
                        # Plot by key (colored) and encoder (markers) - EXACTLY like train_specialist.py
                        for coord, key, enc_idx in zip(coords_combined, all_keys, encoder_indices):
                            color = key_colors.get(key, 'gray')
                            if enc_idx is not None and len(unique_encoders) > 1:
                                marker = encoder_markers[enc_idx % len(encoder_markers)]
                            else:
                                marker = 'o'
                            
                            plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.7,
                                        marker=marker, edgecolors='k', linewidths=0.5)
                        
                        # Create legend matching train_specialist.py style
                        legend_elements = []
                        keys_to_show = unique_keys[:20]  # Show only first 20 keys
                        for key in keys_to_show:
                            color = key_colors[key]
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                           markersize=8, label=f'{key[:8]}'))
                        
                        if len(unique_keys) > 20:
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                                           markersize=8, label=f'... and {len(unique_keys)-20} more keys'))
                        
                        # Add encoder legend
                        for enc_idx in unique_encoders:
                            if enc_idx is not None:
                                marker = encoder_markers[enc_idx % len(encoder_markers)]
                                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='k', linestyle='',
                                                               markersize=8, label=f'Encoder {enc_idx}'))
                        
                        plt.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
                        plt.title(f'Training Latent Space - Epoch 1\n(Optimized latents using train_specialist.py method - Colored by Key, Markers by Encoder)', fontsize=12)
                        plt.xlabel('t-SNE Dimension 1')
                        plt.ylabel('t-SNE Dimension 2')
                        
                        # Save combined plot
                        plt.savefig(os.path.join(out_dir, 'training_latent_space_epoch.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"[OK] Created training latent space plot matching train_specialist.py style")
                        
                    else:
                        print(f"[WARNING] No encoder indices available, falling back to simple plotting")
                        # Fallback to simple plotting
                        tsne_combined = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(all_latents) - 1))))
                        coords_combined = tsne_combined.fit_transform(all_latents)
                        
                        plt.figure(figsize=(12, 10))
                        uniq = sorted(list(set(all_keys)))
                        colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(uniq)}
                        for (xv, yv), k in zip(coords_combined, all_keys):
                            plt.scatter(xv, yv, c=[colors[k]], s=20, alpha=0.7, edgecolors='k', linewidths=0.2)
                        plt.title('Training Latent Space (one epoch) - Fallback Mode')
                        plt.savefig(os.path.join(out_dir, 'training_latent_space_epoch.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                    
                else:
                    # Single encoder: plot by task key
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(all_latents) - 1))))
                    coords = tsne.fit_transform(all_latents)
                    plt.figure(figsize=(12, 10))
                    uniq = sorted(list(set(all_keys)))
                    colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(uniq)}
                    for (xv, yv), k in zip(coords, all_keys):
                        plt.scatter(xv, yv, c=[colors[k]], s=20, alpha=0.7, edgecolors='k', linewidths=0.2)
                    plt.title('Training Latent Space (one epoch)')
                    plt.savefig(os.path.join(out_dir, 'training_latent_space_epoch.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Upload combined training latent space plot to WandB (for both cases)
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
        
        # Print summary of aggregated support metrics after optimization
        agg_metrics = eval_results.get('aggregated_metrics', {})
        if agg_metrics and 'support_metrics' in agg_metrics:
            support_agg = agg_metrics['support_metrics']
            print(f"\n=== AGGREGATED SUPPORT METRICS AFTER OPTIMIZATION (Across All Keys) ===")
            print(f"  Average Shape Accuracy: {support_agg.get('avg_shape_accuracy', 0.0):.4f}")
            print(f"  Average Grid Accuracy: {support_agg.get('avg_grid_accuracy', 0.0):.4f}")
            print(f"  Average Exact Accuracy: {support_agg.get('avg_exact_accuracy', 0.0):.4f}")
            print(f"  Average Shape Loss: {support_agg.get('avg_shape_loss', 0.0):.4f}")
            print(f"  Average Grid Loss: {support_agg.get('avg_grid_loss', 0.0):.4f}")
            print(f"  Average Final Optimization Loss: {support_agg.get('avg_final_opt_loss', 0.0):.4f}")
            print(f"  Total Support Samples: {support_agg.get('total_samples', 0)}")
            print(f"==================================================================\n")
        
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
                
                # Print clear summary of support metrics after optimization
                if support_metrics:
                    post = support_metrics.get('post_opt', {})
                    print(f"\n=== SUPPORT METRICS AFTER OPTIMIZATION ===")
                    print(f"  Shape Accuracy: {post.get('shape_accuracy', 0.0):.4f}")
                    print(f"  Grid Accuracy: {post.get('grid_accuracy', 0.0):.4f}")
                    print(f"  Exact Accuracy: {post.get('exact_accuracy', 0.0):.4f}")
                    print(f"  Shape Loss: {post.get('shape_loss', 0.0):.4f}")
                    print(f"  Grid Loss: {post.get('grid_loss', 0.0):.4f}")
                    print(f"  Final Optimization Loss: {post.get('final_opt_loss', 0.0):.4f}")
                    print(f"  Number of Support Samples: {num_samples_sup}")
                    print(f"==========================================\n")
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
        
        # Additional custom evaluation t-SNE: Query vs Support samples
        try:
            print(f"[B] Creating custom evaluation t-SNE: Query vs Support samples")
            from utils.latent_metrics import compute_task_distance_metrics
            
            # Get support and query latents from evaluation results
            eval_latent_data = eval_results.get('evaluation_latent_data', {})
            support_data = eval_latent_data.get('support', {})
            query_data = eval_latent_data.get('query', {})
            
            # Debug: Check what's in the evaluation data
            print(f"[DEBUG] Evaluation latent data keys: {list(eval_latent_data.keys())}")
            print(f"[DEBUG] Support data keys: {list(support_data.keys())}")
            print(f"[DEBUG] Query data keys: {list(query_data.keys())}")
            
            # Debug: Check the structure more deeply
            if support_data:
                print(f"[DEBUG] Support data structure: {type(support_data)}")
                for key, value in support_data.items():
                    if isinstance(value, (list, np.ndarray)):
                        print(f"[DEBUG] Support[{key}]: {type(value)} with length {len(value)}")
                    else:
                        print(f"[DEBUG] Support[{key}]: {type(value)} = {value}")
                        
            if query_data:
                print(f"[DEBUG] Query data structure: {type(query_data)}")
                for key, value in query_data.items():
                    if isinstance(value, (list, np.ndarray)):
                        print(f"[DEBUG] Query[{key}]: {type(value)} with length {len(value)}")
                    else:
                        print(f"[DEBUG] Query[{key}]: {type(value)} = {value}")
            
            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                # For specialist models, create COMBINED plot with all encoders + PoE
                print(f"[B] Creating combined evaluation t-SNE with all encoders + PoE")
                
                # The new structure has flattened support/query data
                # Check if we have the new structure or old structure
                if 'latent_zs' in support_data and 'latent_zs' in query_data:
                    # New structure: flattened data
                    print(f"[DEBUG] Using new flattened data structure")
                    
                    support_latents = np.array(support_data.get('latent_zs', []))
                    query_latents = np.array(query_data.get('latent_zs', []))
                    support_keys = support_data.get('keys', [])
                    query_keys = query_data.get('keys', [])
                    
                    print(f"[DEBUG] Support latents shape: {support_latents.shape if len(support_latents) > 0 else 'empty'}")
                    print(f"[DEBUG] Query latents shape: {query_latents.shape if len(query_latents) > 0 else 'empty'}")
                    print(f"[DEBUG] Support keys length: {len(support_keys)}")
                    print(f"[DEBUG] Query keys length: {len(query_keys)}")
                    
                    if len(support_latents) > 0 and len(query_latents) > 0:
                        # Combine support and query latents
                        combined_latents = np.vstack([support_latents, query_latents])
                        
                        # Create t-SNE
                        import matplotlib.pyplot as plt
                        from sklearn.manifold import TSNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(combined_latents) - 1))))
                        coords = tsne.fit_transform(combined_latents)
                        
                        plt.figure(figsize=(16, 12))
                        
                        # SIMPLIFIED EVALUATION ENCODING SCHEME:
                        # 1. Support latents: Color by KEY, Shape by SAMPLE TYPE (circle for PoE before optimization)
                        # 2. Query latents: Color by KEY, Shape by SAMPLE TYPE (square for optimized PoE)
                        
                        # Get keys from the evaluation results metadata
                        eval_metadata = eval_results.get('evaluation_metadata', {})
                        all_keys = eval_metadata.get('task_keys', [])
                        print(f"[DEBUG] Found {len(all_keys)} keys from evaluation metadata: {all_keys[:10]}...")
                        
                        if all_keys:
                            # Create color mapping for keys
                            if len(all_keys) <= 400:
                                colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
                                colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
                                colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
                                colors4 = plt.cm.Paired(np.linspace(0, 1, 12))
                                all_colors = np.vstack([colors1, colors2, colors3, colors4])
                                while len(all_colors) < len(all_keys):
                                    all_colors = np.vstack([all_colors, all_colors])
                                key_colors = {k: all_colors[i % len(all_colors)] for i, k in enumerate(all_keys)}
                            else:
                                key_colors = {k: plt.cm.viridis(i / len(all_keys)) for i, k in enumerate(all_keys)}
                            
                            # Plot support samples: Color by KEY, Shape by SAMPLE TYPE (circle for PoE before optimization)
                            sup_coords = coords[:len(support_latents)]
                            print(f"[DEBUG] Plotting {len(sup_coords)} support samples")
                            
                            # Assign keys to support samples based on evaluation order
                            support_keys_assigned = []
                            for i in range(len(sup_coords)):
                                key_idx = i % len(all_keys)
                                support_keys_assigned.append(all_keys[key_idx])
                            
                            for i, coord in enumerate(sup_coords):
                                # For support samples, use circle marker (PoE before optimization)
                                marker = 'o'  # Circle for all support samples
                                
                                # Get the key for this sample
                                sample_key = support_keys_assigned[i] if i < len(support_keys_assigned) else 'unknown'
                                
                                color = key_colors.get(sample_key, 'gray')
                                plt.scatter(coord[0], coord[1], color=color, s=60, alpha=0.8,
                                            marker=marker, edgecolors='k', linewidths=0.5)
                            
                            # Plot query samples: Color by KEY, Shape by SAMPLE TYPE (square for optimized PoE)
                            qry_coords = coords[len(support_latents):]
                            
                            # Assign keys to query samples based on evaluation order
                            query_keys_assigned = []
                            for i in range(len(qry_coords)):
                                key_idx = i % len(all_keys)
                                query_keys_assigned.append(all_keys[key_idx])
                            
                            for i, coord in enumerate(qry_coords):
                                # For query samples, use square marker (optimized PoE)
                                marker = 's'  # Square for all query samples
                                
                                # Get the key for this sample
                                sample_key = query_keys_assigned[i] if i < len(query_keys_assigned) else 'unknown'
                                
                                color = key_colors.get(sample_key, 'gray')
                                plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.9,
                                            marker=marker, edgecolors='k', linewidths=0.5)
                            
                            # Create simplified legend: Color by KEY, Shape by SAMPLE TYPE
                            legend_elements = []
                            
                            # Key legend (color) - show first 20 keys
                            keys_to_show = all_keys[:20]
                            for key in keys_to_show:
                                color = key_colors[key]
                                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                               markersize=8, label=f'Key: {key[:8]}'))
                            
                            if len(all_keys) > 20:
                                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                                               markersize=8, label=f'... and {len(all_keys)-20} more keys'))
                            
                            # Sample type legend (shape)
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', linestyle='', markersize=8, label='Support (PoE before optimization)'))
                            legend_elements.append(plt.Line2D([0], [0], marker='s', color='k', linestyle='', markersize=8, label='Query (Optimized PoE)'))
                            
                            plt.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
                            plt.title('Evaluation Latent Space - Support vs Query\n(Color: Key, Shape: Support/Query)', fontsize=12)
                            plt.xlabel('t-SNE Dimension 1')
                            plt.ylabel('t-SNE Dimension 2')
                            plt.tight_layout()
                            plt.savefig(os.path.join(out_dir, 'eval_latent_space_combined_all_encoders_poe.png'), dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            print(f"[OK] Saved combined evaluation t-SNE: {len(support_latents)} support + {len(query_latents)} query samples across all encoders + PoE")
                            
                            # Upload to WandB
                            if wandb_logger:
                                try:
                                    wandb_logger.log({"evaluation_latent_space_combined": wandb.Image(os.path.join(out_dir, 'eval_latent_space_combined_all_encoders_poe.png'))})
                                    print(f"[OK] Uploaded combined evaluation t-SNE to WandB")
                                except Exception as e:
                                    print(f"[WARNING] Failed to upload combined evaluation t-SNE to WandB: {e}")
                        else:
                            print(f"[WARNING] No keys found in evaluation metadata")
                    else:
                        print(f"[WARNING] No support or query latents found in new structure")
                else:
                    # Old structure: separate encoder/poe keys
                    print(f"[DEBUG] Using old encoder/poe structure")
                    
                    # Collect all support and query latents with proper key tracking
                    all_support_latents = []
                    all_query_latents = []
                    all_support_labels = []
                    all_query_labels = []
                    all_support_keys = []
                    all_query_keys = []
                    
                    # Collect PoE latents first
                    if 'poe' in support_data and 'poe' in query_data:
                        poe_sup = np.array(support_data['poe'].get('latent_zs', []))
                        poe_qry = np.array(query_data['poe'].get('latent_zs', []))
                        poe_sup_keys = support_data['poe'].get('keys', [])
                        poe_qry_keys = query_data['poe'].get('keys', [])
                        
                        if len(poe_sup) > 0 and len(poe_qry) > 0:
                            all_support_latents.append(poe_sup)
                            all_query_latents.append(poe_qry)
                            all_support_labels.extend(['PoE'] * len(poe_sup))
                            all_query_labels.extend(['PoE'] * len(poe_qry))
                            all_support_keys.extend(poe_sup_keys[:len(poe_sup)])
                            all_query_keys.extend(poe_qry_keys[:len(poe_qry)])
                    
                    # Collect encoder latents
                    for enc_idx in range(model.num_encoders):
                        enc_key = f'encoder_{enc_idx}'
                        if enc_key in support_data and enc_key in query_data:
                            enc_sup = np.array(support_data[enc_key].get('latent_zs', []))
                            enc_qry = np.array(query_data[enc_key].get('latent_zs', []))
                            enc_sup_keys = support_data[enc_key].get('keys', [])
                            enc_qry_keys = query_data[enc_key].get('keys', [])
                            
                            if len(enc_sup) > 0 and len(enc_qry) > 0:
                                all_support_latents.append(enc_sup)
                                all_query_latents.append(enc_qry)
                                all_support_labels.extend([f'Encoder {enc_idx}'] * len(enc_sup))
                                all_query_labels.extend([f'Encoder {enc_idx}'] * len(enc_qry))
                                all_support_keys.extend(enc_sup_keys[:len(enc_sup)])
                                all_query_keys.extend(enc_qry_keys[:len(enc_qry)])
                    
                    if all_support_latents and all_query_latents:
                        # Combine all support and query latents
                        combined_support = np.vstack(all_support_latents)
                        combined_query = np.vstack(all_query_latents)
                        combined_latents = np.vstack([combined_support, combined_query])
                        
                        # Create t-SNE
                        import matplotlib.pyplot as plt
                        from sklearn.manifold import TSNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(combined_latents) - 1))))
                        coords = tsne.fit_transform(combined_latents)
                        
                        plt.figure(figsize=(16, 12))
                        
                        # SIMPLIFIED EVALUATION ENCODING SCHEME:
                        # 1. Support latents: Color by KEY, Shape by SAMPLE TYPE (circle for PoE before optimization)
                        # 2. Query latents: Color by KEY, Shape by SAMPLE TYPE (square for optimized PoE)
                        
                        # First, try to get keys from the evaluation results directly
                        # The evaluation processes different keys sequentially, so we should be able to get them
                        all_keys = []
                        
                        # Try to get keys from the evaluation results metadata
                        eval_metadata = eval_results.get('evaluation_metadata', {})
                        if 'task_keys' in eval_metadata:
                            all_keys = eval_metadata['task_keys']
                            print(f"[DEBUG] Found keys from evaluation metadata: {all_keys}")
                        else:
                            # Fallback: try to get keys from the evaluation results structure
                            # Look for keys in the evaluation results
                            for key in eval_results.keys():
                                if isinstance(eval_results[key], dict) and 'keys' in eval_results[key]:
                                    all_keys.extend(eval_results[key]['keys'])
                                    print(f"[DEBUG] Found keys in {key}: {eval_results[key]['keys']}")
                            
                            # If still no keys, try to get them from the support/query data
                            if not all_keys:
                                for enc_key in support_data.keys():
                                    if enc_key in support_data and support_data[enc_key].get('keys'):
                                        all_keys.extend(support_data[enc_key]['keys'])
                                for enc_key in query_data.keys():
                                    if enc_key in query_data and query_data[enc_key].get('keys'):
                                        all_keys.extend(query_data[enc_key]['keys'])
                        
                        unique_keys = sorted(list(set(all_keys)))
                        print(f"[DEBUG] Found {len(unique_keys)} unique keys: {unique_keys[:10]}...")
                        
                        if len(unique_keys) <= 400:
                            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
                            colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
                            colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
                            colors4 = plt.cm.Paired(np.linspace(0, 1, 12))
                            all_colors = np.vstack([colors1, colors2, colors3, colors4])
                            while len(all_colors) < len(unique_keys):
                                all_colors = np.vstack([all_colors, all_colors])
                            key_colors = {k: all_colors[i % len(all_colors)] for i, k in enumerate(unique_keys)}
                        else:
                            key_colors = {k: plt.cm.viridis(i / len(unique_keys)) for i, k in enumerate(unique_keys)}
                        
                        # Plot support samples: Color by KEY, Shape by SAMPLE TYPE (circle for PoE before optimization)
                        sup_coords = coords[:len(combined_support)]
                        print(f"[DEBUG] Plotting {len(sup_coords)} support samples")
                        
                        # Assign keys to support samples based on evaluation order
                        # Since the evaluation processes keys sequentially, we can assign keys in order
                        support_keys_assigned = []
                        if unique_keys:
                            # Distribute keys across support samples
                            for i in range(len(sup_coords)):
                                key_idx = i % len(unique_keys)
                                support_keys_assigned.append(unique_keys[key_idx])
                        else:
                            support_keys_assigned = ['unknown'] * len(sup_coords)
                        
                        for i, (coord, label) in enumerate(zip(sup_coords, all_support_labels)):
                            # For support samples, use circle marker regardless of encoder (all are PoE before optimization)
                            marker = 'o'  # Circle for all support samples (PoE before optimization)
                            
                            # Get the key for this sample
                            sample_key = support_keys_assigned[i] if i < len(support_keys_assigned) else 'unknown'
                            
                            color = key_colors.get(sample_key, 'gray')
                            plt.scatter(coord[0], coord[1], color=color, s=60, alpha=0.8,
                                        marker=marker, edgecolors='k', linewidths=0.5)
                        
                        # Plot query samples: Color by KEY, Shape by SAMPLE TYPE (square for optimized PoE)
                        qry_coords = coords[len(combined_support):]
                        
                        # Assign keys to query samples based on evaluation order
                        query_keys_assigned = []
                        if unique_keys:
                            # Distribute keys across query samples
                            for i in range(len(qry_coords)):
                                key_idx = i % len(unique_keys)
                                query_keys_assigned.append(unique_keys[key_idx])
                        else:
                            query_keys_assigned = ['unknown'] * len(qry_coords)
                    
                        for i, (coord, label) in enumerate(zip(qry_coords, all_query_labels)):
                            # For query samples, use square marker (optimized PoE)
                            marker = 's'  # Square for all query samples (optimized PoE)
                            
                            # Get the key for this sample
                            sample_key = query_keys_assigned[i] if i < len(query_keys_assigned) else 'unknown'
                            
                            color = key_colors.get(sample_key, 'gray')
                            plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.9,
                                        marker=marker, edgecolors='k', linewidths=0.5)
                        
                        # Create simplified legend: Color by KEY, Shape by SAMPLE TYPE
                        legend_elements = []
                        
                        # Key legend (color) - show first 20 keys
                        keys_to_show = unique_keys[:20]
                        for key in keys_to_show:
                            color = key_colors[key]
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                           markersize=8, label=f'Key: {key[:8]}'))
                        
                        if len(unique_keys) > 20:
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                                           markersize=8, label=f'... and {len(unique_keys)-20} more keys'))
                        
                        # Sample type legend (shape)
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', linestyle='', markersize=8, label='Support (PoE before optimization)'))
                        legend_elements.append(plt.Line2D([0], [0], marker='s', color='k', linestyle='', markersize=8, label='Query (Optimized PoE)'))
                        
                        plt.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
                        plt.title('Evaluation Latent Space - Support vs Query\n(Color: Key, Shape: Support/Query)', fontsize=12)
                        plt.xlabel('t-SNE Dimension 1')
                        plt.ylabel('t-SNE Dimension 2')
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, 'eval_latent_space_combined_all_encoders_poe.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"[OK] Saved combined evaluation t-SNE: {len(combined_support)} support + {len(combined_query)} query samples across all encoders + PoE")
                        
                        # Upload to WandB
                        if wandb_logger:
                            try:
                                wandb_logger.log({"evaluation_latent_space_combined": wandb.Image(os.path.join(out_dir, 'eval_latent_space_combined_all_encoders_poe.png'))})
                                print(f"[OK] Uploaded combined evaluation t-SNE to WandB")
                            except Exception as e:
                                print(f"[WARNING] Failed to upload combined evaluation t-SNE to WandB: {e}")
                

            else:
                # For single encoder models, create one combined plot with consistent encoding
                if 'poe' in support_data and 'poe' in query_data:
                    sup_latents = np.array(support_data['poe'].get('latent_zs', []))
                    qry_latents = np.array(query_data['poe'].get('latent_zs', []))
                    sup_keys = support_data['poe'].get('keys', [])
                    qry_keys = query_data['poe'].get('keys', [])
                    
                    if len(sup_latents) > 0 and len(qry_latents) > 0:
                        combined_latents = np.vstack([sup_latents, qry_latents])
                        
                        import matplotlib.pyplot as plt
                        from sklearn.manifold import TSNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, min(10, len(combined_latents) - 1))))
                        coords = tsne.fit_transform(combined_latents)
                        
                        plt.figure(figsize=(14, 10))
                        
                        # CONSISTENT ENCODING SCHEME: Color by KEY, Shape by SAMPLE TYPE
                        # Collect all keys
                        all_keys = sup_keys + qry_keys
                        unique_keys = sorted(list(set(all_keys)))
                        
                        # Create color map for keys (same as train_specialist.py)
                        if len(unique_keys) <= 400:
                            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
                            colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
                            colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
                            colors4 = plt.cm.Paired(np.linspace(0, 1, 12))
                            all_colors = np.vstack([colors1, colors2, colors3, colors4])
                            while len(all_colors) < len(unique_keys):
                                all_colors = np.vstack([all_colors, all_colors])
                            key_colors = {k: all_colors[i % len(all_colors)] for i, k in enumerate(unique_keys)}
                        else:
                            key_colors = {k: plt.cm.viridis(i / len(unique_keys)) for i, k in enumerate(unique_keys)}
                        
                        # Plot support samples: Color by KEY, Shape by SAMPLE TYPE (circle)
                        sup_coords = coords[:len(sup_latents)]
                        for i, coord in enumerate(sup_coords):
                            sample_key = sup_keys[i] if i < len(sup_keys) else 'unknown'
                            color = key_colors.get(sample_key, 'gray')
                            plt.scatter(coord[0], coord[1], color=color, s=60, alpha=0.8,
                                        marker='o', edgecolors='k', linewidths=0.5)
                        
                        # Plot query samples: Color by KEY, Shape by SAMPLE TYPE (square)
                        qry_coords = coords[len(sup_latents):]
                        for i, coord in enumerate(qry_coords):
                            sample_key = qry_keys[i] if i < len(qry_keys) else 'unknown'
                            color = key_colors.get(sample_key, 'gray')
                            plt.scatter(coord[0], coord[1], color=color, s=80, alpha=0.9,
                                        marker='s', edgecolors='k', linewidths=0.5)
                        
                        # Create legend: Color by KEY, Shape by SAMPLE TYPE
                        legend_elements = []
                        
                        # Key legend (color)
                        keys_to_show = unique_keys[:20]  # Show only first 20 keys
                        for key in keys_to_show:
                            color = key_colors[key]
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                           markersize=8, label=f'Key: {key[:8]}'))
                        
                        if len(unique_keys) > 20:
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                                           markersize=8, label=f'... and {len(unique_keys)-20} more keys'))
                        
                        # Sample type legend
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', linestyle='', markersize=8, label='Support'))
                        legend_elements.append(plt.Line2D([0], [0], marker='s', color='k', linestyle='', markersize=8, label='Query (Optimized PoE)'))
                        
                        plt.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
                        plt.title('Evaluation Latent Space - Single Encoder\n(Color: Key, Shape: Support/Query)', fontsize=12)
                        plt.xlabel('t-SNE Dimension 1')
                        plt.ylabel('t-SNE Dimension 2')
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, 'eval_latent_space_support_vs_query.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"[OK] Saved evaluation t-SNE: {len(sup_latents)} support + {len(qry_latents)} query samples")
                        
                        # Upload to WandB
                        if wandb_logger:
                            try:
                                wandb_logger.log({"evaluation_latent_space": wandb.Image(os.path.join(out_dir, 'eval_latent_space_support_vs_query.png'))})
                                print(f"[OK] Uploaded evaluation t-SNE to WandB")
                            except Exception as e:
                                print(f"[WARNING] Failed to upload evaluation t-SNE to WandB: {e}")
        except Exception as e:
            print(f"[WARNING] Could not create custom evaluation t-SNE: {e}")
            import traceback
            traceback.print_exc()

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
        
        # Try to find a working key for reconstruction
        working_key = None
        in_seqs = None
        out_seqs = None
        
        # First try the first training key
        if training_keys:
            try:
                sample_key = training_keys[0]
                print(f"  [INFO] Attempting reconstruction with key: {sample_key}")
                _, _, _, in_seqs, out_seqs = generate_and_process_tasks(sample_key, 1)
                if in_seqs and out_seqs:
                    working_key = sample_key
                    print(f"  [OK] Successfully generated reconstruction data for key: {sample_key}")
            except Exception as e:
                print(f"  [WARNING] Failed to generate data for key {sample_key}: {e}")
        
        # If first key failed, try to find a working key from available generators
        if not working_key:
            try:
                from re_arc.main import get_generators
                available_generators = get_generators()
                if available_generators:
                    # Try the first available generator
                    fallback_key = list(available_generators.keys())[0]
                    print(f"  [INFO] Trying fallback key: {fallback_key}")
                    _, _, _, in_seqs, out_seqs = generate_and_process_tasks(fallback_key, 1)
                    if in_seqs and out_seqs:
                        working_key = fallback_key
                        print(f"  [OK] Successfully generated reconstruction data with fallback key: {fallback_key}")
            except Exception as e:
                print(f"  [WARNING] Fallback key generation failed: {e}")
        
        if in_seqs and out_seqs and working_key:
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
            axes[0].imshow(input_grid, cmap='viridis', interpolation='nearest')
            axes[0].set_title(f'Input\n{input_shape[0]}×{input_shape[1]}')
            axes[0].axis('off')
            axes[1].imshow(target_grid, cmap='viridis', interpolation='nearest')
            axes[1].set_title(f'Target\n{target_shape[0]}×{target_shape[1]}')
            axes[1].axis('off')
            axes[2].imshow(recon_grid, cmap='viridis', interpolation='nearest')
            axes[2].set_title(f'Reconstruction\n{recon_shape[0]}×{recon_shape[1]}')
            axes[2].axis('off')
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
            print(f"[INFO] Could not generate reconstruction data - skipping reconstruction figure")
    except Exception as e:
        print(f"[WARNING] Section (c) failed: {e}")
        print(f"  [DEBUG] Error details: {str(e)}")
        import traceback
        traceback.print_exc()

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


