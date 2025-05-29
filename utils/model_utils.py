import torch.nn as nn
import datetime
import os
import random
import numpy as np
import torch
import logging
import sys
from torch.utils.data import TensorDataset, DataLoader
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json


##############################
# Count Model Parameters
##############################
def count_model_parameters(model: nn.Module) -> None:
    total_params = 0
    breakdown = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            component = name.split('.')[0]
            breakdown[component] = breakdown.get(component, 0) + num_params

    print("=== Model Parameter Count ===")
    print(f"Total trainable parameters: {total_params:,}")
    for component, count in breakdown.items():
        print(f"{component}: {count:,} parameters")
    print("=============================")


##############################
# Create a Unique Run Directory
##############################
RUN_BASE_DIR = "runs_re_arc"    # Base directory to save run outputs
def create_run_directory(file_store_name=None,base_dir=RUN_BASE_DIR):
    if file_store_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    else:
        run_dir = os.path.join(base_dir, file_store_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


##############################
# Set Seed for Reproducibility
##############################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##############################
# Set Up Logging
##############################
def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Logs will be saved to: {log_file}")
    print(f"Logging configured. Logs will be saved to: {log_file}")
    return logger


##############################
# Prepare DataLoader
##############################
def prepare_dataloader(input_seqs, output_seqs, batch_size, shuffle=True):
    # Convert lists of numpy arrays to a single numpy array first, then to tensor
    if isinstance(input_seqs, list) and len(input_seqs) > 0 and isinstance(input_seqs[0], np.ndarray):
        try:
            # Ensure all sequences are of the same length before stacking, pad if necessary
            # For ARC, lengths should be consistent after processing (e.g., 902)
            max_len_input = max(len(s) for s in input_seqs) if input_seqs else 0
            max_len_output = max(len(s) for s in output_seqs) if output_seqs else 0
            
            # This padding assumes sequences are 1D. If they are multi-dimensional, adjust padding.
            # For ARC, this should be fine as they are flattened.
            padded_input_seqs = [np.pad(s, (0, max_len_input - len(s)), 'constant') if len(s) < max_len_input else s for s in input_seqs]
            padded_output_seqs = [np.pad(s, (0, max_len_output - len(s)), 'constant') if len(s) < max_len_output else s for s in output_seqs]

            input_tensor = torch.tensor(np.array(padded_input_seqs), dtype=torch.float32)
            output_tensor = torch.tensor(np.array(padded_output_seqs), dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Could not convert input/output sequences with np.array. Error: {e}. Falling back to slow method.")
            input_tensor = torch.FloatTensor(input_seqs) 
            output_tensor = torch.FloatTensor(output_seqs)
    elif isinstance(input_seqs, torch.Tensor) and isinstance(output_seqs, torch.Tensor):
        input_tensor = input_seqs.float()
        output_tensor = output_seqs.float()
    else: 
        print("Warning: Input sequences type not explicitly handled for optimized tensor conversion. Using slow torch.FloatTensor.")
        input_tensor = torch.FloatTensor(input_seqs)
        output_tensor = torch.FloatTensor(output_seqs)

    dataset = TensorDataset(input_tensor, output_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


##############################
# Save Checkpoint and Results
##############################
def save_checkpoint(model, optimizer, epoch, loss, run_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(run_dir, f'checkpoint_epoch{epoch}.pt'))


##############################
# Save Model Parameters
##############################
def save_model_params(run_dir):
    """
    Save model parameters to a pickle file.
    
    Args:
        run_dir: Directory to save parameters in
    """
    from utils.settings_manager import settings
    
    # Get all settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    evaluation_settings = settings.get_evaluation_settings()
    
    # Basic model parameters
    model_params = {
        # 'KEY': data_settings['key'], # Old: single key
        'TRAINING_KEYS': data_settings.get('training_keys', [data_settings.get('key')]), # New: list of keys
        'TRAINING_SEED': data_settings['training_seed'],
        'EVAL_SEED': data_settings['eval_seed'],
        'LATENT_DIM': model_architecture['latent_dim'],
        'HIDDEN_DIM': model_architecture['hidden_dim'],
        'NUM_LAYERS': model_architecture['num_layers'],
        'NUM_HEADS': model_architecture['num_heads'],
        'DROPOUT': model_architecture['dropout'],
        'MAX_LENGTH': model_architecture['max_length'],
        'ENCODER_MAX_LENGTH': model_architecture['encoder_max_length'],
        'DECODER_MAX_LENGTH': model_architecture['decoder_max_length'],
        'BATCH_SIZE': training_settings.get('batch_size', 16),
        'NUM_EPOCHS': training_settings['num_epochs'],
        'LEARNING_RATE': training_settings['learning_rate'],
        'BETA': training_settings['beta'],
        'n': data_settings['n'],
        'OPTIMIZE_Z': latent_optimization['training']['enabled'],
        'OPTIMIZE_Z_NUM_STEPS': latent_optimization['training']['num_steps'],
        'OPTIMIZE_Z_LR': latent_optimization['training']['learning_rate'],
        'OPTIMIZE_Z_INFERENCE': latent_optimization['inference']['enabled'],
        'OPTIMIZE_Z_INFERENCE_NUM_STEPS': latent_optimization['inference']['num_steps'],
        'OPTIMIZE_Z_INFERENCE_LR': latent_optimization['inference']['learning_rate'],
        'DEFAULT_EVAL_KEYS': evaluation_settings['eval_keys'], # This should already be a list
        'DEFAULT_EVAL_N_SAMPLES': evaluation_settings['eval_n_samples'],
        'DEFAULT_EVAL_N_QUERIES': evaluation_settings['eval_n_queries'],
        'DEFAULT_EVAL_EPOCH': evaluation_settings['eval_epoch']
    }
    
    # Add optimization method
    if 'method' in latent_optimization:
        model_params['OPTIMIZATION_METHOD'] = latent_optimization['method']
    
    # Add evolutionary parameters if available
    if 'evolutionary' in latent_optimization:
        evolutionary = latent_optimization['evolutionary']
        model_params['EVOLUTIONARY_POPULATION_SIZE'] = evolutionary.get('population_size', 20)
        model_params['EVOLUTIONARY_NUM_GENERATIONS'] = evolutionary.get('num_generations', 15)
        model_params['EVOLUTIONARY_MUTATION_STD'] = evolutionary.get('mutation_std', 0.1)
    
    # Add voronoi parameters if available
    if 'voronoi' in latent_optimization:
        voronoi = latent_optimization['voronoi']
        model_params['VORONOI_POPULATION_SIZE'] = voronoi.get('population_size', 20)
        model_params['VORONOI_NUM_GENERATIONS'] = voronoi.get('num_generations', 15)
        model_params['VORONOI_DIVERSITY_WEIGHT'] = voronoi.get('diversity_weight', 0.5)
        model_params['VORONOI_MUTATION_STD'] = voronoi.get('mutation_std', 0.1)
    
    params_file = os.path.join(run_dir, 'model_params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"Model parameters saved to {params_file}")
    
    # Also save the settings JSON file
    settings.save_settings(run_dir)


##############################
# Save Training JSON
##############################
def save_training_json(results, run_dir):
    """
    Save training results in a JSON format for better accessibility and readability.
    
    Args:
        results: Dictionary containing training results (should have tensors converted to lists/numbers)
        run_dir: Directory to save the JSON file
    """
    # Ensure all numpy arrays or tensors are converted to lists for JSON serialization
    # The main_training loop should already do this for items like 'input_sequences', 'latent_mus', etc.
    # Here we make a deep copy to safely attempt conversions for any missed items.
    results_copy = json.loads(json.dumps(results, default=lambda o: '' )) # basic attempt to make it serializable
                                                                       # a more robust solution would iterate and convert np/torch types

    json_file = os.path.join(run_dir, 'training_results.json')
    try:
        with open(json_file, 'w') as f:
            json.dump(results_copy, f, indent=4)
        print(f"Training results (JSON) saved to {json_file}")
    except TypeError as e:
        print(f"Error saving results to JSON: {e}. Some elements might not be serializable.")
        # Fallback: save problematic keys separately or log them
        problematic_keys = []
        for key, value in results.items():
            try:
                json.dumps(value)
            except TypeError:
                problematic_keys.append(key)
        print(f"Problematic keys for JSON serialization: {problematic_keys}")
        # Optionally, save a simplified version or just the serializable parts


##############################
# Save Results
##############################  
def save_results(results, run_dir):
    """Save results to a pickle file and a JSON file."""
    results_file_pkl = os.path.join(run_dir, 'results.pkl')
    with open(results_file_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results (pickle) saved to {results_file_pkl}")
    
    # Save as JSON
    save_training_json(results, run_dir) # Call the new JSON saving function
    
    # Also save model parameters (this already saves settings.json)
    save_model_params(run_dir)


##############################
# Perform Latent Analysis
##############################
def perform_latent_analysis(results, run_dir):
    all_mus = torch.cat(results['latent_mus'], dim=0).numpy()
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(all_mus)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(all_mus)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis')
    plt.title('Latent Space Visualization (t-SNE)')
    plt.savefig(os.path.join(run_dir, 'latent_space_visualization.png'))
    plt.close()


##############################
# Load Model
##############################
def load_model(run_dir, epoch=None, device='cuda'):
    """
    Load a model from a run directory.
    
    Args:
        run_dir (str): Path to the run directory
        epoch (int, optional): Specific epoch to load. If None, loads the latest checkpoint.
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: Epoch number of the loaded checkpoint
        loss: Loss value of the loaded checkpoint
    """
    from models.base_model import LatentProgramNetwork, compute_loss
    from torch.optim import Adam
    
    # Initialize model and optimizer
    model = LatentProgramNetwork().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)  # Default learning rate
    
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


def evaluate_model(model, samples_dataloader, queries_dataloader, device='cuda'):
    """
    Evaluate model performance on a dataloader.
    # ... (rest of docstring) ...
    """
    from models.base_model import compute_loss # Keep local import
    from utils.latent_functions import get_optimized_z # Keep local import
    from utils.settings_manager import settings # Keep local import

    latent_optimization = settings.get_latent_optimization()
    optimize_z_inference = latent_optimization['inference']['enabled']

    model.eval()
    shape_correct, shape_tokens = 0, 0
    grid_correct, grid_tokens = 0, 0
    sample_exact_correct = 0
    total_samples = 0
    
    support_losses = []
    query_losses = []
    support_reconstructions = []
    query_reconstructions = []
    z_optimization_logs = []

    z_from_last_sample_batch = None # Will store z from the last processed sample batch

    for batch_input_s, batch_target_s in samples_dataloader:
        batch_input_s = batch_input_s.to(device)
        batch_target_s = batch_target_s.to(device)
        
        current_z_for_this_sample_batch = None
        if optimize_z_inference:
            current_z_for_this_sample_batch, losses_opt = get_optimized_z(
                model, batch_input_s, batch_target_s, context='inference'
            )
            if losses_opt is not None: # get_optimized_z can return (z, None)
                 z_optimization_logs.append(losses_opt)
        else:
            with torch.no_grad():
                mu, log_var = model.encoder(batch_input_s, batch_target_s)
                current_z_for_this_sample_batch = model.reparameterize(mu, log_var)
        
        z_from_last_sample_batch = current_z_for_this_sample_batch

        # Calculate support loss (original way, which re-derives z internally via model call)
        # This part is not directly related to the RuntimeError but is a point of consistency.
        s_loss_val = compute_loss(model, batch_input_s, batch_target_s)
        support_losses.append(s_loss_val.item())

        # Calculate support reconstructions using the specific z for this sample batch
        if current_z_for_this_sample_batch is not None:
            with torch.no_grad():
                # 'current_z_for_this_sample_batch' already has the correct batch dim for these samples
                shape_logits_s, grid_logits_s = model.decoder(current_z_for_this_sample_batch, batch_input_s, target_seq=batch_target_s)
                support_reconstructions.append((
                    shape_logits_s.cpu(),
                    grid_logits_s.cpu()
                ))
        else:
             support_reconstructions.append((None, None)) # Should not happen if dataloader has items

    # After processing all sample batches:
    if z_from_last_sample_batch is None:
        # Check if queries_dataloader has any data. list(dataloader) can be slow if it's large.
        # A more efficient check might be needed if performance is critical here.
        # For now, a simple check or assuming queries_dataloader won't run if samples are empty.
        # If there are queries to process, but no z from samples, it's an issue.
        has_queries = False
        for _ in queries_dataloader: # Check if dataloader is not empty
            has_queries = True
            break
        
        if has_queries:
            print("Error in evaluate_model: No z obtained from samples_dataloader, but queries exist. Cannot proceed.")
            return {
                'metrics': {'error': 'No z from samples for query evaluation', 'support_loss': sum(support_losses)/len(support_losses) if support_losses else 0, 'query_loss': 0, 'shape_accuracy': 0, 'grid_accuracy': 0, 'overall_accuracy': 0, 'sample_exact_accuracy': 0, 'losses_gradient_ascent': z_optimization_logs, 'used_latent_optimization': optimize_z_inference},
                'reconstruction_results': {'support_reconstructions': support_reconstructions, 'query_reconstructions': []}
            }
        else: # No samples and no queries
            print("Warning in evaluate_model: samples_dataloader was empty. No z obtained. No queries to process.")
            # Fall through, query loop won't run.

    # If optimize_z_inference was false for the entire run, and no logs were added
    # (e.g., if samples_dataloader itself was empty or if no optimization happened)
    if not optimize_z_inference and not z_optimization_logs:
        z_optimization_logs.append(None)

    z_for_queries_prototype = None
    if z_from_last_sample_batch is not None:
        # Average the z from the last sample batch to create a [1, latent_dim] prototype
        z_for_queries_prototype = z_from_last_sample_batch.mean(dim=0, keepdim=True)

    for batch_input_q, batch_target_q in queries_dataloader:
        batch_input_q = batch_input_q.to(device)
        batch_target_q = batch_target_q.to(device)
        query_batch_size = batch_input_q.size(0)

        if z_for_queries_prototype is None:
            # This implies z_from_last_sample_batch was None, and the earlier check should have caught it if queries exist.
            # This is a safeguard.
            print("Critical Error: z_for_queries_prototype is None inside query loop. Skipping query batch.")
            continue
            
        with torch.no_grad():
            # Expand the [1, latent_dim] prototype to [query_batch_size, latent_dim]
            z_query_expanded = z_for_queries_prototype.expand(query_batch_size, -1)
            
            shape_logits, grid_logits = model.decoder(z_query_expanded, batch_input_q, target_seq=batch_target_q)
            
            # Compute query loss (original way, re-derives z from query data)
            q_loss_val = compute_loss(model, batch_input_q, batch_target_q)
            query_losses.append(q_loss_val.item())
            
            query_reconstructions.append((
                shape_logits.cpu(), # These are from z_query_expanded
                grid_logits.cpu()   # These are from z_query_expanded
            ))

            shape_pred = shape_logits.argmax(dim=-1)
            grid_pred = grid_logits.argmax(dim=-1)
            shape_tgt = batch_target_q[:, 900:902].long()
            grid_tgt = batch_target_q[:, :900].long()

            shape_correct += (shape_pred == shape_tgt).sum().item()
            shape_tokens += shape_tgt.numel()

            for i in range(query_batch_size):
                tgt_rows = int(batch_target_q[i, 900].item())
                tgt_cols = int(batch_target_q[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0 : # Check before indexing grid_pred and grid_tgt
                    grid_correct += (grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                    grid_tokens += active_pixels
                    if torch.all(shape_pred[i] == shape_tgt[i]) and torch.all(grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]):
                        sample_exact_correct += 1
                elif torch.all(shape_pred[i] == shape_tgt[i]): # If grid is empty, only shape matters
                     sample_exact_correct +=1


            total_samples += query_batch_size

    avg_support_loss = sum(support_losses) / len(support_losses) if support_losses else 0.0
    avg_query_loss = sum(query_losses) / len(query_losses) if query_losses else 0.0

    return {
        'metrics': {
            'support_loss': avg_support_loss,
            'query_loss': avg_query_loss,
            'shape_accuracy': shape_correct / shape_tokens if shape_tokens > 0 else 0.0,
            'grid_accuracy': grid_correct / grid_tokens if grid_tokens > 0 else 0.0,
            'overall_accuracy': (shape_correct + grid_correct) / (shape_tokens + grid_tokens) if (shape_tokens + grid_tokens) > 0 else 0.0,
            'sample_exact_accuracy': sample_exact_correct / total_samples if total_samples > 0 else 0.0,
            'losses_gradient_ascent': z_optimization_logs,
            'used_latent_optimization': optimize_z_inference,
        },
        'reconstruction_results': {
            'support_reconstructions': support_reconstructions,
            'query_reconstructions': query_reconstructions,
        }
    }



def save_evaluation_results(results, run_dir):
    """
    Save evaluation results to a pickle file.
    
    Args:
        results: Dictionary containing evaluation results
        run_dir: Directory to save results in
    """
    results_file = os.path.join(run_dir, 'evaluation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Evaluation results saved to {results_file}")
    
    # Also save model parameters
    save_model_params(run_dir)

def load_evaluation_results(run_dir):
    """
    Load evaluation results from a pickle file.
    
    Args:
        run_dir: Directory containing the results file
    
    Returns:
        dict: Loaded evaluation results
    """
    results_file = os.path.join(run_dir, 'evaluation_results.pkl')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"No evaluation results found in {run_dir}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results



