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