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
def create_run_directory(base_dir=RUN_BASE_DIR):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
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
def prepare_dataloader(input_seqs, output_seqs, batch_size):
    """Prepare DataLoader for training"""
    # Convert to PyTorch tensors (using FloatTensor here; adjust if needed)
    input_tensor = torch.FloatTensor(input_seqs)
    output_tensor = torch.FloatTensor(output_seqs)

    dataset = TensorDataset(input_tensor, output_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
# Save Results
##############################  
def save_results(results, run_dir):
    results_file = os.path.join(run_dir, 'results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)


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


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model performance on a dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing test data
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation metrics and visualizations
    """
    from models.base_model import OPTIMIZE_Z_INFERENCE_NUM_STEPS, OPTIMIZE_Z_INFERENCE_LR, compute_loss
    from utils.latent_functions import optimize_latent_z

    model.eval()
    shape_correct, shape_tokens = 0, 0
    grid_correct, grid_tokens = 0, 0
    sample_exact_correct = 0
    total_samples = 0
    
    # Track losses
    support_losses = []
    query_losses = []
    reconstructions = []
    
    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        batch_size = batch_input.size(0)

        # Split support (first sample) and query (rest)
        support_input = batch_input[0:1]
        support_target = batch_target[0:1]
        query_input = batch_input[1:]
        query_target = batch_target[1:]

        # Optimize z using only the support example
        with torch.enable_grad():
            z = optimize_latent_z(
                model,
                support_input,
                support_target,
                num_steps=OPTIMIZE_Z_INFERENCE_NUM_STEPS,
                lr=OPTIMIZE_Z_INFERENCE_LR
            )
            
            # Compute support loss
            support_loss = compute_loss(model, support_input, support_target)
            support_losses.append(support_loss.item())

        # Decode query examples using the same z
        with torch.no_grad():
            z_query = z.expand(query_input.size(0), -1)
            shape_logits, grid_logits = model.decoder(z_query, query_input, target_seq=query_target)
            
            # Compute query loss
            query_loss = compute_loss(model, query_input, query_target)
            query_losses.append(query_loss.item())
            
            reconstructions.append((shape_logits.cpu(), grid_logits.cpu()))

            shape_pred = shape_logits.argmax(dim=-1)
            grid_pred = grid_logits.argmax(dim=-1)
            shape_tgt = query_target[:, 900:902].long()
            grid_tgt = query_target[:, :900].long()

            shape_correct += (shape_pred == shape_tgt).sum().item()
            shape_tokens += shape_tgt.numel()

            for i in range(query_input.size(0)):
                tgt_rows = int(query_target[i, 900].item())
                tgt_cols = int(query_target[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                grid_correct += (grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]).sum().item()
                grid_tokens += active_pixels
                if torch.all(shape_pred[i] == shape_tgt[i]) and torch.all(grid_pred[i, :active_pixels] == grid_tgt[i, :active_pixels]):
                    sample_exact_correct += 1

            total_samples += query_input.size(0)

    # Compute average losses
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
        },
        'reconstruction_results': {
            'reconstructions': reconstructions,
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



