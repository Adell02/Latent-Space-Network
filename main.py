import argparse
import torch
import os
import pickle
import numpy as np
from training import main_training
from evaluation import main_test
from utils.model_utils import (
    load_model,
    save_evaluation_results
)
from utils.visualizers import visualize_stored_results
from utils.settings_manager import init_settings
from utils.wandb_logger import init_wandb_for_mode, get_wandb_logger
import datetime

# Import latent validation functions (with graceful fallback)
try:
    from latent_validation import (
        latent_swap_test, 
        zero_random_latent_test, 
        latent_space_visualization, 
        create_visualization_plots
    )
    LATENT_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Latent validation not available: {e}")
    print("  To enable latent validation, install: pip install scikit-learn")
    LATENT_VALIDATION_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train, evaluate, or visualize the Latent Program Network')
    parser.add_argument('--mode', choices=['train', 'visualize', 'eval', 'all'], nargs='+', required=True,
                      help='Mode to run: train, visualize, evaluate (includes latent validation), or all')
    parser.add_argument('--file_name', type=str, help='Directory storing/containing model checkpoints and results',required=True)
    parser.add_argument('--settings', type=str, default='model_settings.json',
                      help='Settings file to use (default: model_settings.json)')
    parser.add_argument('--keys', type=str, nargs='+', 
                      help='Problem keys for evaluation (space-separated)')
    parser.add_argument('--n_eval_samples', type=int,
                      help='Numbers of input-output pairs to generate for Z optimisation during evaluation')
    parser.add_argument('--n_eval_queries', type=int,
                      help='Numbers of queries to do inference')
    parser.add_argument('--epoch', type=int,
                      help='Specific epoch to load for evaluation')
    parser.add_argument('--visualize_n_values', type=int,
                      help='Numbers of input-output pairs to generate for visualization')
    # Removed --use_reparameterized flag - always use mean vectors (μ) for consistency
    return parser.parse_args()

def main_args():
    args = parse_args()
    
    # Initialize settings with the specified file (sets global settings)
    print(f"Loading settings from: {args.settings}")
    settings = init_settings(args.settings)
    
    # Get project name and create unique run name
    project_name = settings.get_project_name()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_run_name = f"{project_name}_{timestamp}"
    
    # Set WANDB project env var for this session
    os.environ['WANDB_PROJECT_NAME'] = project_name
    
    # Get settings from settings manager
    data_settings = settings.get_data_settings()
    evaluation_settings = settings.get_evaluation_settings()
    wandb_settings = settings.get_wandb_settings()

    BASE_DIR = data_settings['run_base_dir']

    # Use settings defaults or command line overrides
    DEFAULT_EVAL_KEYS = args.keys or evaluation_settings['eval_keys']
    DEFAULT_EVAL_N_SAMPLES = args.n_eval_samples or evaluation_settings['eval_n_samples']
    DEFAULT_EVAL_N_QUERIES = args.n_eval_queries or evaluation_settings['eval_n_queries']
    DEFAULT_EVAL_EPOCH = args.epoch or evaluation_settings['eval_epoch']
    DEFAULT_VISUALIZE_N_VALUES = args.visualize_n_values or evaluation_settings['visualize_n_values']

    EVAL_SEED = data_settings['eval_seed']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not args.file_name:
        raise ValueError("--file_name must be specified")

    # Compose unique run directory using project name and timestamp
    run_dir = os.path.join(BASE_DIR, f"{args.file_name}_{unique_run_name}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Ensure run directory exists before initializing wandb
    from utils.model_utils import create_run_directory
    run_dir = create_run_directory(os.path.basename(run_dir))

    # Initialize wandb logging if enabled
    wandb_logger = None
    if wandb_settings.get('enabled', False):
        print("WandB logging enabled")
        wandb_logger = init_wandb_for_mode('main', run_dir)
        if wandb_logger:
            print(f"✓ WandB logging enabled for main.py: {wandb_logger.run.name}")
        else:
            print("⚠ WandB initialization failed, continuing without wandb")
    else:
        print("⚠ WandB logging is DISABLED in settings")

    # Step counter for wandb logging (since main.py is orchestration, not training)
    step_counter = 0
    
    try:
        if 'train' in args.mode or 'all' in args.mode:
            # Train the model
            print(f"Starting training with settings from: {args.settings}")
            
            # Log training start
            step_counter += 1
            if wandb_logger:
                try:
                    wandb_logger._safe_log({
                        'main/mode': args.mode,
                        'main/settings_file': args.settings,
                        'main/device': str(device),
                        'main_training/started': True,
                        'main_training/data_source': 'training_start'
                    }, step_hint=step_counter)
                    print(f"✓ Training start logged to WandB at step {step_counter}")
                except Exception as e:
                    print(f"⚠ Failed to log training start: {e}")
            
            results, model = main_training(args.file_name)
            print("Training complete. Results saved in the run directory.")
            
            # Log training completion metrics
            step_counter += 1
            if wandb_logger and results:
                try:
                    final_loss = results.get('epoch_losses', [0])[-1] if results.get('epoch_losses') else 0
                    total_epochs = len(results.get('epoch_losses', []))
                    wandb_logger._safe_log({
                        'main_training/completed': True,
                        'main_training/final_loss': final_loss,
                        'main_training/total_epochs': total_epochs,
                        'main_training/data_source': 'training_completion'
                    }, step_hint=step_counter)
                    print(f"✓ Training completion logged to WandB at step {step_counter}")
                except Exception as e:
                    print(f"⚠ Failed to log training completion: {e}")
        
        if 'eval' in args.mode or 'all' in args.mode:
            # Load the model
            if DEFAULT_EVAL_EPOCH is None:
                raise ValueError("--epoch must be specified for evaluation")
            if DEFAULT_EVAL_KEYS is None:
                print("No keys specified for evaluation, using default keys")
            if DEFAULT_EVAL_N_SAMPLES is None:
                print("No n_eval_samples specified for evaluation, using default n_eval_samples")
            if DEFAULT_EVAL_N_QUERIES is None:
                print("No n_eval_queries specified for evaluation, using default n_eval_queries")
            
            # Log evaluation start
            step_counter += 1
            if wandb_logger:
                try:
                    wandb_logger._safe_log({
                        'main_eval/started': True,
                        'main_eval/epoch_to_evaluate': DEFAULT_EVAL_EPOCH,
                        'main_eval/keys_to_evaluate': DEFAULT_EVAL_KEYS,
                        'main_eval/n_samples': DEFAULT_EVAL_N_SAMPLES,
                        'main_eval/n_queries': DEFAULT_EVAL_N_QUERIES,
                        'main_eval/data_source': f'evaluation_start_epoch_{DEFAULT_EVAL_EPOCH}'
                    }, step_hint=step_counter)
                    print(f"✓ Evaluation start logged to WandB at step {step_counter}")
                except Exception as e:
                    print(f"⚠ Failed to log evaluation start: {e}")
            
            model, _, _, _ = load_model(run_dir, epoch=DEFAULT_EVAL_EPOCH, device=device)
            
            # Run evaluation (now includes training latent data collection)
            print("\n=== RUNNING EVALUATION ===")
            # Always use mean vectors (μ) for consistency and efficiency
            print("Using mean (μ) vectors for latent visualization")
            eval_results = main_test(model, DEFAULT_EVAL_KEYS, run_dir, DEFAULT_EVAL_N_SAMPLES, DEFAULT_EVAL_N_QUERIES, EVAL_SEED, device)
                    
            # Save evaluation results
            save_evaluation_results(eval_results, run_dir)
            
            # Run latent validation as part of evaluation
            latent_validation_results = None
            latent_plots = {}
            
            if LATENT_VALIDATION_AVAILABLE:
                print("\n=== RUNNING LATENT VALIDATION ===")
                try:
                    # Generate test data for latent validation with key tracking
                    print("Generating latent validation test data...")
                    eval_keys_for_latent = DEFAULT_EVAL_KEYS[:2] if len(DEFAULT_EVAL_KEYS) > 1 else DEFAULT_EVAL_KEYS
                    n_samples_per_key = min(10, DEFAULT_EVAL_N_SAMPLES)  # Limit samples for efficiency
                    
                    all_inputs, all_outputs, sample_keys = [], [], []
                    for key in eval_keys_for_latent:
                        try:
                            from re_arc.main import generate_and_process_tasks
                            _, _, _, inputs, outputs = generate_and_process_tasks(key, n_samples_per_key)
                            all_inputs.extend(inputs)
                            all_outputs.extend(outputs)
                            sample_keys.extend([key] * len(inputs))  # Track which key each sample came from
                            print(f"Generated {len(inputs)} latent validation samples for key '{key}'")
                        except Exception as e:
                            print(f"Warning: Failed to generate latent validation data for key {key}: {e}")
                    
                    if all_inputs:
                        print(f"Running latent validation on {len(all_inputs)} samples from {len(eval_keys_for_latent)} keys...")
                        
                        # Run the 4 latent validation tests
                        print("  1. Latent Swap Test...")
                        swap_results = latent_swap_test(model, all_inputs, all_outputs, device, n_samples=5)
                        
                        print("  2. Zero/Random Latent Test...")
                        zero_random_results = zero_random_latent_test(model, all_inputs, all_outputs, device, n_samples=3)
                        
                        print("  3. Latent Space Visualization with Key Clustering...")
                        latents_pca, latents_tsne, properties, explained_variance = latent_space_visualization(
                            model, all_inputs, all_outputs, device, max_samples=min(30, len(all_inputs)), sample_keys=sample_keys
                        )
                        
                        print("  4. Creating enhanced latent validation plots...")
                        latent_plots = create_visualization_plots(
                            swap_results, zero_random_results, latents_pca, latents_tsne, properties, explained_variance, run_dir=run_dir
                        )
                        
                        # Store validation results for logging
                        latent_validation_results = {
                            'n_samples_tested': len(all_inputs),
                            'n_swap_tests': len(swap_results),
                            'n_zero_random_tests': len(zero_random_results['zero']),
                            'latent_space_samples': len(latents_pca) if latents_pca is not None else 0,
                            'test_keys': eval_keys_for_latent
                        }
                        
                        print(f"✓ Latent validation completed: {len(swap_results)} swaps, {len(zero_random_results['zero'])} zero/random tests")
                    else:
                        print("⚠ No data available for latent validation")
                        
                except Exception as e:
                    print(f"⚠ Latent validation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠ Latent validation skipped (scikit-learn not available)")
            
            # Log evaluation results to wandb (including latent validation)
            step_counter += 1
            if wandb_logger and eval_results:
                try:
                    print("Logging evaluation results to WandB...")
                    
                    # Log overall evaluation summary
                    total_keys_evaluated = len(eval_results)
                    
                    # Calculate average accuracy across all keys
                    all_grid_acc = []
                    all_shape_acc = []
                    all_overall_acc = []
                    all_exact_acc = []
                    
                    for key, key_results in eval_results.items():
                        if isinstance(key_results, dict) and 'metrics' in key_results:
                            metrics = key_results['metrics']
                            all_grid_acc.append(metrics.get('grid_accuracy', 0.0))
                            all_shape_acc.append(metrics.get('shape_accuracy', 0.0))
                            all_overall_acc.append(metrics.get('overall_accuracy', 0.0))
                            all_exact_acc.append(metrics.get('sample_exact_accuracy', 0.0))
                    
                    # Log summary metrics
                    log_dict = {
                        'main_eval/completed': True,
                        'main_eval/epoch_evaluated': DEFAULT_EVAL_EPOCH,
                        'main_eval/keys_evaluated': total_keys_evaluated,
                        'main_eval/eval_n_samples': DEFAULT_EVAL_N_SAMPLES,
                        'main_eval/eval_n_queries': DEFAULT_EVAL_N_QUERIES,
                        'main_eval/data_source': f'evaluation_epoch_{DEFAULT_EVAL_EPOCH}'
                    }
                    
                    # Add individual key metrics
                    for key, key_results in eval_results.items():
                        if isinstance(key_results, dict) and 'metrics' in key_results:
                            metrics = key_results['metrics']
                            log_dict.update({
                                f'main_eval_key_{key}/grid_accuracy': metrics.get('grid_accuracy', 0.0),
                                f'main_eval_key_{key}/shape_accuracy': metrics.get('shape_accuracy', 0.0),
                                f'main_eval_key_{key}/overall_accuracy': metrics.get('overall_accuracy', 0.0),
                                f'main_eval_key_{key}/sample_exact_accuracy': metrics.get('sample_exact_accuracy', 0.0),
                                f'main_eval_key_{key}/data_source': f'eval_key_{key}_epoch_{DEFAULT_EVAL_EPOCH}'
                            })
                    
                    # Add average metrics if we have results
                    if all_grid_acc:
                        log_dict.update({
                            'main_eval_average/grid_accuracy': np.mean(all_grid_acc),
                            'main_eval_average/shape_accuracy': np.mean(all_shape_acc),
                            'main_eval_average/overall_accuracy': np.mean(all_overall_acc),
                            'main_eval_average/sample_exact_accuracy': np.mean(all_exact_acc),
                            'main_eval_average/data_source': f'eval_average_epoch_{DEFAULT_EVAL_EPOCH}_keys_{len(all_grid_acc)}'
                        })
                    
                    # Add latent validation results if available
                    if latent_validation_results:
                        log_dict.update({
                            'latent_validation/epoch_tested': DEFAULT_EVAL_EPOCH,
                            'latent_validation/n_samples_tested': latent_validation_results['n_samples_tested'],
                            'latent_validation/n_swap_tests': latent_validation_results['n_swap_tests'],
                            'latent_validation/n_zero_random_tests': latent_validation_results['n_zero_random_tests'],
                            'latent_validation/latent_space_samples': latent_validation_results['latent_space_samples'],
                            'latent_validation/test_keys': latent_validation_results['test_keys'],
                            'latent_validation/data_source': f'latent_validation_epoch_{DEFAULT_EVAL_EPOCH}_keys_{latent_validation_results["test_keys"]}'
                        })
                        
                        # Add latent validation plots
                        import wandb
                        required_plots = ['latent_swap', 'zero_random_latent']
                        for plot_name in required_plots:
                            plot_path = latent_plots.get(plot_name)
                            if plot_path and os.path.exists(plot_path):
                                log_dict[f'latent_validation/{plot_name}'] = wandb.Image(plot_path)
                            else:
                                print(f"⚠ Warning: {plot_name}.png not found or not generated!")
                        
                        # Add interpretation guide
                        interpretation = """
                        LATENT VALIDATION INTERPRETATION GUIDE:
                        
                        🔴 BAD SIGNS (Decoder absorbing all load):
                        - Latent swaps produce normal-looking reconstructions
                        - Zero/random latents produce reasonable outputs  
                        - Latent space shows no semantic clustering
                        
                        🟢 GOOD SIGNS (Encoder learning meaningful latents):
                        - Latent swaps produce novel/different combinations
                        - Zero/random latents produce garbage/noise
                        - Latent space clusters by semantic properties (size, color, etc.)
                        
                        📊 WHAT TO LOOK FOR:
                        - Swap test: Different from targets = latents matter
                        - Zero/random test: Garbage output = latents matter
                        - Space visualization: Clustering = meaningful structure
                        """
                        log_dict['latent_validation/interpretation_guide'] = interpretation
                    
                    wandb_logger._safe_log(log_dict, step_hint=step_counter)
                    
                    # Clean up temporary plot files
                    # for plot_path in latent_plots.values():
                    #     try:
                    #         if os.path.exists(plot_path):
                    #             os.unlink(plot_path)
                    #     except:
                    #         pass
                    
                    log_message = f"✓ Evaluation results logged to WandB at step {step_counter} for {total_keys_evaluated} keys"
                    if latent_validation_results:
                        log_message += f" + latent validation ({latent_validation_results['n_samples_tested']} samples)"
                    print(log_message)
                    
                except Exception as e:
                    print(f"⚠ Failed to log evaluation results: {e}")
            
        if 'visualize' in args.mode or 'all' in args.mode:
            if DEFAULT_VISUALIZE_N_VALUES is None:
                print("No visualize_n_values specified for visualization, using default visualize_n_values")
            if DEFAULT_EVAL_N_QUERIES and DEFAULT_VISUALIZE_N_VALUES > DEFAULT_EVAL_N_QUERIES:
                print("visualize_n_values is greater than n_eval_queries, using n_eval_queries")
                DEFAULT_VISUALIZE_N_VALUES = DEFAULT_EVAL_N_QUERIES
            
            # Log visualization start
            step_counter += 1
            if wandb_logger:
                try:
                    wandb_logger._safe_log({
                        'main_visualization/started': True,
                        'main_visualization/epoch_to_visualize': DEFAULT_EVAL_EPOCH,
                        'main_visualization/n_values': DEFAULT_VISUALIZE_N_VALUES,
                        'main_visualization/data_source': f'visualization_start_epoch_{DEFAULT_EVAL_EPOCH}'
                    }, step_hint=step_counter)
                    print(f"✓ Visualization start logged to WandB at step {step_counter}")
                except Exception as e:
                    print(f"⚠ Failed to log visualization start: {e}")
            
            # Also run visualization
            print("\nVisualizing stored results...")
            visualize_stored_results(run_dir, epoch=DEFAULT_EVAL_EPOCH)
            
            # Log visualization completion
            step_counter += 1
            if wandb_logger:
                try:
                    wandb_logger._safe_log({
                        'main_visualization/completed': True,
                        'main_visualization/epoch_visualized': DEFAULT_EVAL_EPOCH,
                        'main_visualization/n_values': DEFAULT_VISUALIZE_N_VALUES,
                        'main_visualization/data_source': f'visualization_epoch_{DEFAULT_EVAL_EPOCH}'
                    }, step_hint=step_counter)
                    print(f"✓ Visualization completion logged to WandB at step {step_counter}")
                except Exception as e:
                    print(f"⚠ Failed to log visualization completion: {e}")

    except Exception as e:
        # Log error with step counter
        if wandb_logger:
            try:
                wandb_logger._safe_log({
                    'main/error': str(e),
                    'main/failed_mode': args.mode,
                    'main/error_step': step_counter
                }, step_hint=step_counter + 1)
                print(f"✓ Error logged to WandB at step {step_counter + 1}")
            except Exception as log_error:
                print(f"⚠ Failed to log error to WandB: {log_error}")
        raise
    
    finally:
        # Finish wandb run
        if wandb_logger:
            try:
                wandb_logger.finish()
                print("✓ WandB run finished")
            except Exception as e:
                print(f"⚠ Failed to finish WandB run: {e}")

if __name__ == "__main__":
    main_args()

