import argparse
import torch
import os
import pickle
import numpy as np
import datetime


# --- Initialize settings FIRST so subsequent imports see the correct config ---
from utils.settings_manager import init_settings
settings = init_settings('model_specialist_settings.json')

# --- Load frozen config if it exists in the run directory ---
import json
run_dir_env = None
import sys
if '--file_name' in sys.argv:
    idx = sys.argv.index('--file_name')
    if idx + 1 < len(sys.argv):
        run_dir_env = sys.argv[idx + 1]
if run_dir_env is not None:
    import os
    base_dir = settings.get_data_settings()['run_base_dir'] if hasattr(settings, 'get_data_settings') else '.'
    run_dir = os.path.join(base_dir, run_dir_env)
    frozen_config_path = os.path.join(run_dir, 'frozen_config.json')
    if os.path.exists(frozen_config_path):
        with open(frozen_config_path, 'r') as f:
            frozen_config = json.load(f)
        settings.set_settings(frozen_config)
        print(f"[ OK ] Loaded frozen config from {frozen_config_path} for all modes")

# Now import modules that rely on the global `settings`
from train_specialist import main_specialist_training
from evaluation import main_test
from utils.model_utils import (
    load_model,
    save_evaluation_results
)
from utils.visualizers import visualize_stored_results, generate_samples_used_folder
from utils.wandb_logger import init_wandb_for_mode, get_wandb_logger

# Efficient latent validation import (graceful fallback)
try:
    from latent_validation import run_latent_validation_for_specialist
    LATENT_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"[ WARNING ] Latent validation not available: {e}")
    LATENT_VALIDATION_AVAILABLE = False

# --------------------------------------------------
# Load settings-driven defaults for CLI parameters
# --------------------------------------------------

data_settings = settings.get_data_settings()
evaluation_settings = settings.get_evaluation_settings()
specialist_settings = settings.get_specialist_training_settings()

BASE_DIR = data_settings['run_base_dir']

from utils.evaluation_utils import get_evaluation_keys_with_all_support
DEFAULT_EVAL_KEYS = evaluation_settings['eval_keys']
# Handle "all" evaluation keys similar to training keys
DEFAULT_EVAL_KEYS = get_evaluation_keys_with_all_support(DEFAULT_EVAL_KEYS, evaluation_settings.get('n_max_eval_keys', 10))

DEFAULT_EVAL_N_SAMPLES = evaluation_settings['eval_n_samples']
DEFAULT_EVAL_N_QUERIES = evaluation_settings['eval_n_queries']
DEFAULT_EVAL_EPOCH = evaluation_settings['eval_epoch']
DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']
DEFAULT_PHASES = specialist_settings['phases_to_run']

# Ensure default phases are valid for the new approach
if DEFAULT_PHASES and any(p not in ['A', 'B'] for p in DEFAULT_PHASES):
    DEFAULT_PHASES = ['A', 'B']

EVAL_SEED = data_settings['eval_seed']


def parse_args():
    parser = argparse.ArgumentParser(description='Train, evaluate, or visualize the Specialist Multi-Encoder Network')
    parser.add_argument('--mode', choices=['train', 'visualize', 'eval', 'all'], nargs='+', required=True,
                      help='Mode to run: train, visualize, evaluate, or all')
    parser.add_argument('--file_name', type=str, help='Directory storing/containing model checkpoints and results', required=True)
    parser.add_argument('--phases', type=str, default=','.join(DEFAULT_PHASES),
                      help='Comma-separated phases to run for training (A,B)')
    parser.add_argument('--resume_from_phase', type=str, default=None,
                      help='Phase to resume training from (A,B)')
    parser.add_argument('--keys', type=str, nargs='+', default=DEFAULT_EVAL_KEYS,
                      help='Problem keys for evaluation (space-separated)')
    parser.add_argument('--n_eval_samples', type=int, default=DEFAULT_EVAL_N_SAMPLES,
                      help='Numbers of input-output pairs to generate for Z optimisation during evaluation')
    parser.add_argument('--n_eval_queries', type=int, default=DEFAULT_EVAL_N_QUERIES,
                      help='Numbers of queries to do inference')
    parser.add_argument('--epoch', type=str, default=DEFAULT_EVAL_EPOCH,
                      help='Specific epoch to load for evaluation (e.g., "phase_c_final", "phase_a_final", or epoch number)')
    parser.add_argument('--visualize_n_values', type=int, default=DEFAULT_VISUALIZE_N_VALUES,
                      help='Numbers of input-output pairs to generate for visualization')
    return parser.parse_args()


def main_args():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not args.file_name:
        raise ValueError("--file_name must be specified")

    # Create run directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BASE_DIR, f"{args.file_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    # Generate notes for this run
    notes = f"Specialist run: {args.file_name} | Specialist training"
    print(f"Run notes: {notes}")

    # ----------------------
    # TRAINING
    # ----------------------
    if 'train' in args.mode or 'all' in args.mode:
        phases_to_run = [p.strip().upper() for p in args.phases.split(',')]
        valid_phases = ['A', 'B']
        if not all(p in valid_phases for p in phases_to_run):
            raise ValueError(f"Invalid phases. Valid phases are: {valid_phases}")

        print(f"Starting specialist training with phases: {phases_to_run}")
        if args.resume_from_phase:
            print(f"Resuming from phase: {args.resume_from_phase}")

        # Train the specialist model
        results, _ = main_specialist_training(
            args.file_name,
            phases_to_run=phases_to_run,
            resume_from_phase=args.resume_from_phase,
            run_dir=run_dir,  # Pass the exact directory
            notes=notes
        )
        print("Specialist training complete. Results saved in the run directory.")

        # Generate samples_used and upload to WandB immediately after training (before any runs are closed)
        try:
            print("\n=== GENERATING SAMPLES_USED FOLDER (main_specialist) ===")
            # Resolve keys from settings
            evaluation_settings_current = settings.get_evaluation_settings()
            data_settings_current = settings.get_data_settings()
            from utils.evaluation_utils import get_evaluation_keys_with_all_support
            eval_keys = evaluation_settings_current.get('eval_keys')
            eval_keys = get_evaluation_keys_with_all_support(
                eval_keys, evaluation_settings_current.get('n_max_eval_keys', 10)
            )
            training_keys = (
                settings.get_training_settings().get('task_keys')
                or data_settings_current.get('task_keys')
                or data_settings_current.get('training_keys')
            )
            if isinstance(training_keys, str) and training_keys.lower() == 'all':
                import os as _os
                tasks_dir = _os.path.join(_os.path.dirname(__file__), 're_arc', 're_arc', 'tasks')
                all_keys = [fname[:-5] for fname in _os.listdir(tasks_dir) if fname.endswith('.json')]
                all_keys.sort()
                n_max_keys = data_settings_current.get('n_max_keys', None)
                if n_max_keys is not None:
                    try:
                        n_max_keys = int(n_max_keys)
                        all_keys = all_keys[:n_max_keys]
                    except Exception:
                        pass
                training_keys = all_keys

            if training_keys or eval_keys:
                generate_samples_used_folder(
                    run_dir=run_dir,
                    training_keys=training_keys,
                    eval_keys=eval_keys,
                    n_samples=5
                )
                print("[ OK ] Samples_used generated (main_specialist)")

                # Upload to current WandB run if available
                try:
                    from utils.wandb_logger import get_wandb_logger
                    wandb_logger = get_wandb_logger()
                except Exception:
                    wandb_logger = None
                if wandb_logger:
                    try:
                        import os as _os, glob, wandb as _wandb
                        samples_dir = _os.path.join(run_dir, "samples_used")
                        train_dir = _os.path.join(samples_dir, "train")
                        eval_dir = _os.path.join(samples_dir, "eval")
                        train_images = [_wandb.Image(p) for p in sorted(glob.glob(_os.path.join(train_dir, "*.png")))] if _os.path.exists(train_dir) else []
                        eval_images = [_wandb.Image(p) for p in sorted(glob.glob(_os.path.join(eval_dir, "*.png")))] if _os.path.exists(eval_dir) else []
                        payload = {}
                        if train_images:
                            payload['samples/train'] = train_images
                        if eval_images:
                            payload['samples/eval'] = eval_images
                        if payload:
                            wandb_logger._safe_log(payload)
                            print(f"[ OK ] Uploaded samples to WandB: train={len(train_images)}, eval={len(eval_images)}")
                        else:
                            print("[ INFO ] No sample images found to upload to WandB (main_specialist)")
                    except Exception as _e:
                        print(f"[ WARNING ] Failed to upload samples to WandB (main_specialist): {_e}")
        except Exception as e:
            print(f"[ WARNING ] Failed to generate samples_used in main_specialist: {e}")

    # ----------------------
    # EVALUATION
    # ----------------------
    if 'eval' in args.mode or 'all' in args.mode:
        if args.epoch is None:
            raise ValueError("--epoch must be specified for evaluation")
        # Load model (handling specialist checkpoints)
        model = None
        try:
            from models.base_model import LatentProgramNetwork
            if args.epoch == "phase_c_final":
                model_path = os.path.join(run_dir, 'full_joint.ckpt')
                if os.path.exists(model_path):
                    print(f"Loading final joint model from {model_path}")
                    model = LatentProgramNetwork().to(device)
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # Support legacy checkpoint with raw state_dict saved
                        model.load_state_dict(checkpoint)
                    model.eval()
                else:
                    model, _, _, _ = load_model(run_dir, epoch=None, device=device)
            else:
                epoch_int = None
                try:
                    epoch_int = int(args.epoch)
                except ValueError:
                    pass
                model, _, _, _ = load_model(run_dir, epoch=epoch_int, device=device)
        except Exception as e:
            print(f"Failed to load model using specialist checkpoints: {e}. Falling back to latest checkpoint.")
            model, _, _, _ = load_model(run_dir, epoch=None, device=device)

        print("\n=== RUNNING EVALUATION ===")
        eval_results = main_test(
            model, args.keys, run_dir,
            args.n_eval_samples, args.n_eval_queries, EVAL_SEED, device
        )
        save_evaluation_results(eval_results, run_dir)

        # --- Efficient Latent Validation (after evaluation) ---
        if LATENT_VALIDATION_AVAILABLE:
            print("\n=== RUNNING SPECIALIST LATENT VALIDATION ===")
            wandb_logger = None
            try:
                from utils.wandb_logger import get_wandb_logger
                wandb_logger = get_wandb_logger()
            except Exception:
                pass
            # Use a step counter that is always increasing (use eval epoch if int, else 1000)
            try:
                step_counter = int(args.epoch)
            except Exception:
                step_counter = 1000
            result = run_latent_validation_for_specialist(
                run_dir=run_dir,
                model=model,
                device=device,
                eval_keys=args.keys,
                n_samples_per_key=min(10, args.n_eval_samples),
                wandb_logger=wandb_logger,
                step_hint=step_counter + 1
            )
            print(f"Latent validation result: {result}")
            if not result.get('success', True):
                print(f"[ WARNING ] Latent validation failed or skipped: {result.get('reason', 'unknown reason')}")
        else:
            print("[ WARNING ] Latent validation skipped (scikit-learn not available)")

    # ----------------------
    # VISUALIZATION
    # ----------------------
    if 'visualize' in args.mode or 'all' in args.mode:
        viz_n = args.visualize_n_values or DEFAULT_VISUALIZE_N_VALUES
        if args.n_eval_queries and viz_n > args.n_eval_queries:
            viz_n = args.n_eval_queries
        epoch_for_viz = None
        try:
            epoch_for_viz = int(args.epoch)
        except (TypeError, ValueError):
            pass
        visualize_stored_results(run_dir, epoch=epoch_for_viz)


def print_help():
    print("""
=== SPECIALIST TRAINING HELP ===

This specialist training implements a 2-phase approach:
- Phase A: Train each encoder with its independent decoder on domain-specific data
- Phase B: Train shared decoder using PoE (Product of Experts) of all encoders

Usage examples:
  Train all phases:
    python main_specialist.py --mode train --file_name my_exp
  Train Phase A only:
    python main_specialist.py --mode train --file_name my_exp --phases A
  Evaluate final model (after Phase B):
    python main_specialist.py --mode eval --file_name my_exp --epoch phase_c_final
  Full workflow (train+eval+viz):
    python main_specialist.py --mode all --file_name my_exp

The final model uses PoE to combine specialized encoders for improved performance.
""")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        sys.exit(0)
    try:
        main_args()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 