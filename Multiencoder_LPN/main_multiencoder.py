import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local multi-encoder training and evaluation modules
import Multiencoder_LPN.training as multiencoder_training
import Multiencoder_LPN.evaluation as multiencoder_evaluation
from utils.model_utils import (
    save_evaluation_results
)
from utils.visualizers import visualize_stored_results
# Use local settings manager that points to multiencoder settings
import Multiencoder_LPN.settings_manager as multiencoder_settings
from models.multi_encoder_lpn import MultiEncoderLPN
from torch.optim import Adam

# Get settings from local settings manager
settings = multiencoder_settings.settings
data_settings = settings.get_data_settings()
evaluation_settings = settings.get_evaluation_settings()
model_architecture = settings.get_model_architecture()

BASE_DIR = data_settings['run_base_dir']

DEFAULT_EVAL_KEYS = evaluation_settings['eval_keys']
DEFAULT_EVAL_N_SAMPLES = evaluation_settings['eval_n_samples']
DEFAULT_EVAL_N_QUERIES = evaluation_settings['eval_n_queries']
DEFAULT_EVAL_EPOCH = evaluation_settings['eval_epoch']

DEFAULT_VISUALIZE_N_VALUES = evaluation_settings['visualize_n_values']

EVAL_SEED = data_settings['eval_seed']


def load_multi_encoder_model(run_dir, epoch=None, device='cuda'):
    """
    Load a MultiEncoderLPN model from a run directory.
    
    Args:
        run_dir (str): Path to the run directory
        epoch (int, optional): Specific epoch to load. If None, loads the latest checkpoint.
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded MultiEncoderLPN model
        optimizer: Loaded optimizer
        epoch: Epoch number of the loaded checkpoint
        loss: Loss value of the loaded checkpoint
    """
    # Initialize model with settings from multiencoder settings
    model = MultiEncoderLPN(
        num_encoders=model_architecture['num_encoders'],
        latent_dim=model_architecture['latent_dim'],
        hidden_dim=model_architecture['hidden_dim'],
        num_layers=model_architecture['num_layers'],
        num_heads=model_architecture['num_heads'],
        dropout=model_architecture['dropout'],
        encoder_max_length=model_architecture['encoder_max_length'],
        decoder_max_length=model_architecture['decoder_max_length']
    ).to(device)
    
    # Initialize optimizer
    training_settings = settings.get_training_settings()
    optimizer = Adam(model.parameters(), lr=training_settings['learning_rate'])
    
    if epoch is None:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
        latest_epoch = max([int(f.split('_')[1][5:].split('.')[0]) for f in checkpoints])
        checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch{latest_epoch}.pt')
    else:
        checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch{epoch}.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Set to evaluation mode
    model.eval()
    
    return model, optimizer, epoch, loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train, evaluate, or visualize the Multi-Encoder Latent Program Network')
    parser.add_argument('--mode', choices=['train', 'visualize', 'eval', 'all', 'encode'], nargs='+', required=True,
                      help='Mode to run: train, visualize, evaluate, encode (training sequences), or all')
    parser.add_argument('--file_name', type=str, help='Directory storing/containing model checkpoints and results',required=False,default="multiencoder_pattern_task")
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
    

def main_args():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Multi-encoder configuration: {model_architecture['num_encoders']} encoders")

    if not args.file_name:
        raise ValueError("--file_name must be specified")

    if 'train' in args.mode or 'all' in args.mode:
        # Train the multi-encoder model
        print("\n=== TRAINING MULTI-ENCODER MODEL ===")
        results, model = multiencoder_training.main_training(args.file_name)
        print("Multi-encoder training complete. Results saved in the run directory.")
    
    if 'encode' in args.mode or 'all' in args.mode:
        # Encode training sequences for latent space visualization
        print("\n=== ENCODING TRAINING SEQUENCES ===")
        if args.epoch is None:
            raise ValueError("--epoch must be specified for encoding")
        
        model, _, _, _ = load_multi_encoder_model(os.path.join(BASE_DIR, args.file_name), epoch=args.epoch, device=device)
        
        # Encode training sequences
        run_dir = os.path.join(BASE_DIR, args.file_name)
        encoded_data = multiencoder_evaluation.encode_training_sequences(model, run_dir, device=device, max_samples=500, batch_size=8)
        
        if encoded_data:
            # Save encoded data
            encoded_file = os.path.join(run_dir, 'encoded_training_latents.pkl')
            import pickle
            with open(encoded_file, 'wb') as f:
                pickle.dump(encoded_data, f)
            print(f"Encoded training latents saved to {encoded_file}")
        else:
            print("Failed to encode training sequences")
    
    if 'eval' in args.mode or 'all' in args.mode:
        # Load the multi-encoder model and evaluate
        print("\n=== EVALUATING MULTI-ENCODER MODEL ===")
        if args.epoch is None:
            raise ValueError("--epoch must be specified for evaluation")
        if args.keys is None:
            print("No keys specified for evaluation, using default keys")
        if args.n_eval_samples is None:
            print("No n_eval_samples specified for evaluation, using default n_eval_samples")
        if args.n_eval_queries is None:
            print("No n_eval_queries specified for evaluation, using default n_eval_queries")
        
        model, _, _, _ = load_multi_encoder_model(os.path.join(BASE_DIR, args.file_name), epoch=args.epoch, device=device)
        
        # Run multi-encoder evaluation
        print(f"Running multi-encoder evaluation with {model_architecture['num_encoders']} encoders...")
        eval_results = multiencoder_evaluation.main_test(model, args.keys, args.n_eval_samples, args.n_eval_queries, EVAL_SEED, device)
                
        # Save evaluation results
        save_evaluation_results(eval_results, os.path.join(BASE_DIR, args.file_name))
        
    if 'visualize' in args.mode or 'all' in args.mode:
        print("\n=== VISUALIZING RESULTS ===")
        if args.visualize_n_values is None:
            print("No visualize_n_values specified for visualization, using default visualize_n_values")
        if args.visualize_n_values > args.n_eval_queries:
            print("visualize_n_values is greater than n_eval_queries, using n_eval_queries")
            args.visualize_n_values = args.n_eval_queries
        # Also run visualization
        print("Visualizing stored multi-encoder results...")
        visualize_stored_results(os.path.join(BASE_DIR, args.file_name))

if __name__ == "__main__":
    main_args()

