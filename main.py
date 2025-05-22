import argparse
import torch
import os
from models.base_model import main_training, main_test
from utils.model_utils import (
    load_model,
    save_evaluation_results
)
from utils.visualizers import visualize_stored_results
from utils.settings_manager import settings

# Get settings from settings manager
data_settings = settings.get_data_settings()
evaluation_settings = settings.get_evaluation_settings()

BASE_DIR = data_settings['run_base_dir']

DEFAULT_EVAL_KEYS = evaluation_settings['eval_keys']
DEFAULT_EVAL_N_SAMPLES = evaluation_settings['eval_n_samples']
DEFAULT_EVAL_N_QUERIES = evaluation_settings['eval_n_queries']
DEFAULT_EVAL_EPOCH = evaluation_settings['eval_epoch']

DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']

EVAL_SEED = data_settings['eval_seed']


def parse_args():
    parser = argparse.ArgumentParser(description='Train, evaluate, or visualize the Latent Program Network')
    parser.add_argument('--mode', choices=['train', 'visualize', 'eval', 'all'], nargs='+', required=True,
                      help='Mode to run: train, visualize, evaluate, or all')
    parser.add_argument('--file_name', type=str, help='Directory storing/containing model checkpoints and results',required=True)
    parser.add_argument('--keys', type=str, nargs='+', default=DEFAULT_EVAL_KEYS,
                      help='Problem keys for evaluation (space-separated)')
    parser.add_argument('--n_eval_samples', type=int, default=DEFAULT_EVAL_N_SAMPLES,
                      help='Numbers of input-output pairs to generate for Z optimisation during evaluation')
    parser.add_argument('--n_eval_queries', type=int, default=DEFAULT_EVAL_N_QUERIES,
                      help='Numbers of queries to do inference')
    parser.add_argument('--epoch', type=int, default=DEFAULT_EVAL_EPOCH,
                      help='Specific epoch to load for evaluation')
    parser.add_argument('--visualize_n_values', type=int, default=DEFAULT_VISUALIZE_N_VALUES,
                      help='Numbers of input-output pairs to generate for visualization')
    return parser.parse_args()
    

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not args.file_name:
        raise ValueError("--file_name must be specified")

    if 'train' in args.mode or 'all' in args.mode:
        # Train the model
        results, model = main_training(args.file_name)
        print("Training complete. Results saved in the run directory.")
    
    if 'eval' in args.mode or 'all' in args.mode:
        # Load the model
        if args.epoch is None:
            raise ValueError("--epoch must be specified for evaluation")
        if args.keys is None:
            print("No keys specified for evaluation, using default keys")
        if args.n_eval_samples is None:
            print("No n_eval_samples specified for evaluation, using default n_eval_samples")
        if args.n_eval_queries is None:
            print("No n_eval_queries specified for evaluation, using default n_eval_queries")
        
        model, _, _, _ = load_model(os.path.join(BASE_DIR, args.file_name), epoch=args.epoch, device=device)
        
        # Run evaluation
        print("\nRunning evaluation...")
        eval_results = main_test(model, args.keys, args.n_eval_samples, args.n_eval_queries, EVAL_SEED, device)
                
        # Save evaluation results
        save_evaluation_results(eval_results, os.path.join(BASE_DIR, args.file_name))
        
    if 'visualize' in args.mode or 'all' in args.mode:
        if args.visualize_n_values is None:
            print("No visualize_n_values specified for visualization, using default visualize_n_values")
        if args.visualize_n_values > args.n_eval_queries:
            print("visualize_n_values is greater than n_eval_queries, using n_eval_queries")
            args.visualize_n_values = args.n_eval_queries
        # Also run visualization
        print("\nVisualizing stored results...")
        visualize_stored_results(os.path.join(BASE_DIR, args.file_name))

if __name__ == "__main__":
    main()

