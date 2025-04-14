import argparse
import torch
import os
from models.base_model import main_training, evaluate_model_on_new_data
from utils.model_utils import (
    load_model,
    save_evaluation_results
)
from utils.visualizers import visualize_stored_results

BASE_DIR = 'runs_re_arc'
DEFAULT_RUN_DIR = 'original_w_gradient_ascent'

DEFAULT_KEYS = ['017c7c7b','00d62c1b','007bbfb7']
DEFAULT_N_VALUES = 100
DEFAULT_EPOCH = 49


def parse_args():
    parser = argparse.ArgumentParser(description='Train, evaluate, or visualize the Latent Program Network')
    parser.add_argument('--mode', choices=['train', 'visualize', 'eval', 'all'], nargs='+', required=True,
                      help='Mode to run: train, visualize, evaluate, or all')
    parser.add_argument('--run_dir', type=str, help='Directory containing model checkpoints and results')
    parser.add_argument('--keys', type=str, nargs='+', default=DEFAULT_KEYS,
                      help='Problem keys for evaluation (space-separated)')
    parser.add_argument('--n_values', type=int, default=DEFAULT_N_VALUES,
                      help='Numbers of examples to generate for evaluation (space-separated)')
    parser.add_argument('--epoch', type=int, default=DEFAULT_EPOCH,
                      help='Specific epoch to load for evaluation')
    return parser.parse_args()
    

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if 'train' in args.mode:
        # Train the model
        results, model = main_training()
        print("Training complete. Results saved in the run directory.")
    
    if 'eval' in args.mode:
        if not args.run_dir:
            raise ValueError("--run_dir must be specified for evaluation")
        
        # Load the model
        model, _, _, _ = load_model(os.path.join(BASE_DIR, args.run_dir), epoch=args.epoch, device=device)
        
        # Run evaluation
        print("\nRunning evaluation...")
        eval_results = evaluate_model_on_new_data(model, args.keys, args.n_values, device)
                
        # Save evaluation results
        save_evaluation_results(eval_results, os.path.join(BASE_DIR, args.run_dir))
        
    # Also run visualization
    print("\nVisualizing stored results...")
    visualize_stored_results(os.path.join(BASE_DIR, args.run_dir))

if __name__ == "__main__":
    main()

