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
from utils.settings_manager import settings


##############################
# Count Model Parameters
##############################
def count_model_parameters(model: nn.Module, logger=None, exclude_independent_decoders=False) -> dict:
    """
    Count model parameters with detailed breakdown and return parameter information.
    If exclude_independent_decoders is True, skip parameters in independent decoders (for joint training).
    """
    total_params = 0
    trainable_params = 0
    breakdown = {}
    detailed_breakdown = {}
    # Count all parameters (trainable and non-trainable)
    for name, param in model.named_parameters():
        # Exclude independent decoders if requested
        if exclude_independent_decoders and 'multi_encoder.independent_decoders' in name:
            continue
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        # Component breakdown (first level)
        component = name.split('.')[0]
        breakdown[component] = breakdown.get(component, 0) + num_params
        # Detailed breakdown (full parameter name)
        detailed_breakdown[name] = {
            'shape': list(param.shape),
            'parameters': num_params,
            'trainable': param.requires_grad
        }
    # Calculate multi-encoder specific breakdown if applicable
    encoder_breakdown = {}
    decoder_params = 0
    shared_params = 0
    if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
        # Multi-encoder model analysis
        for name, param in model.named_parameters():
            if exclude_independent_decoders and 'multi_encoder.independent_decoders' in name:
                continue
            num_params = param.numel()
            if 'multi_encoder.encoders' in name:
                # Extract encoder index
                encoder_idx = name.split('.')[2]  # multi_encoder.encoders.0.something
                encoder_key = f'encoder_{encoder_idx}'
                encoder_breakdown[encoder_key] = encoder_breakdown.get(encoder_key, 0) + num_params
            elif 'multi_encoder.decoder' in name or 'decoder' in name:
                decoder_params += num_params
            else:
                shared_params += num_params
    # Create parameter info dictionary
    param_info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'component_breakdown': breakdown,
        'detailed_breakdown': detailed_breakdown,
        'is_multi_encoder': hasattr(model, 'is_multi_encoder') and model.is_multi_encoder,
        'num_encoders': getattr(model, 'num_encoders', 1) if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder else 1,
        'exclude_independent_decoders': exclude_independent_decoders
    }
    # Add multi-encoder specific info
    if param_info['is_multi_encoder']:
        param_info['encoder_breakdown'] = encoder_breakdown
        param_info['decoder_params'] = decoder_params
        param_info['shared_params'] = shared_params
        # Calculate per-encoder average
        if encoder_breakdown:
            total_encoder_params = sum(encoder_breakdown.values())
            avg_encoder_params = total_encoder_params / len(encoder_breakdown)
            param_info['avg_encoder_params'] = avg_encoder_params
            param_info['total_encoder_params'] = total_encoder_params
    # Print detailed information
    print("=" * 60)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model type: {'Multi-encoder' if param_info['is_multi_encoder'] else 'Single encoder'}")
    if exclude_independent_decoders:
        print("NOTE: Independent decoders are excluded from parameter count (joint training mode).")
    if param_info['is_multi_encoder']:
        print(f"Number of encoders: {param_info['num_encoders']}")
        print(f"Decoder parameters: {decoder_params:,}")
        print(f"Shared parameters: {shared_params:,}")
        print()
        print("Per-encoder breakdown:")
        for encoder_name, count in encoder_breakdown.items():
            print(f"  {encoder_name}: {count:,} parameters")
        if 'avg_encoder_params' in param_info:
            print(f"  Average per encoder: {param_info['avg_encoder_params']:,.0f} parameters")
        print()
    print("Component breakdown:")
    for component, count in breakdown.items():
        percentage = (count / total_params) * 100
        print(f"  {component}: {count:,} parameters ({percentage:.1f}%)")
    # Log detailed breakdown if logger is provided
    if logger:
        logger.info("=" * 60)
        logger.info("DETAILED MODEL PARAMETER ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Model type: {'Multi-encoder' if param_info['is_multi_encoder'] else 'Single encoder'}")
        if exclude_independent_decoders:
            logger.info("NOTE: Independent decoders are excluded from parameter count (joint training mode).")
        if param_info['is_multi_encoder']:
            logger.info(f"Number of encoders: {param_info['num_encoders']}")
            logger.info(f"Decoder parameters: {decoder_params:,}")
            logger.info(f"Shared parameters: {shared_params:,}")
            logger.info("Per-encoder breakdown:")
            for encoder_name, count in encoder_breakdown.items():
                logger.info(f"  {encoder_name}: {count:,} parameters")
            if 'avg_encoder_params' in param_info:
                logger.info(f"  Average per encoder: {param_info['avg_encoder_params']:,.0f} parameters")
        logger.info("Component breakdown:")
        for component, count in breakdown.items():
            percentage = (count / total_params) * 100
            logger.info(f"  {component}: {count:,} parameters ({percentage:.1f}%)")
        # Log top 10 largest parameter groups
        sorted_detailed = sorted(detailed_breakdown.items(), key=lambda x: x[1]['parameters'], reverse=True)
        logger.info("Top 10 largest parameter groups:")
        for name, info in sorted_detailed[:10]:
            logger.info(f"  {name}: {info['parameters']:,} parameters {info['shape']}")
    print("=" * 60)
    return param_info


##############################
# Create a Unique Run Directory
##############################
# Base directory comes from settings (falls back to 'runs')
RUN_BASE_DIR = settings.get_data_settings().get('run_base_dir', 'runs')

def create_run_directory(file_store_name=None, base_dir: str = None):
    """Create (or reuse) a run directory inside the configured base dir.

    Args:
        file_store_name: user-supplied folder name (or None → timestamp)
        base_dir: overrides settings value when provided.
    """
    if base_dir is None:
        base_dir = RUN_BASE_DIR
    if file_store_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    else:
        run_dir = os.path.join(base_dir, file_store_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Creating run directory at: {run_dir}")
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
# --------------------------------------------------
# Logging helpers (ASCII-only console to avoid cp1252 errors)
# --------------------------------------------------

class _AsciiFilter(logging.Filter):
    """Replace non-ASCII characters in log messages for Windows consoles."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = record.getMessage().encode('ascii', errors='replace').decode('ascii')
        except Exception:
            pass  # If anything goes wrong we keep original message
        return True

def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "training.log")

    # File handler (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    # Console handler with ASCII filter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(_AsciiFilter())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[console_handler, file_handler]
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
    checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Upload checkpoint to WandB if available
    try:
        from utils.wandb_logger import get_wandb_logger
        wandb_logger = get_wandb_logger()
        if wandb_logger and wandb_logger.is_initialized:
            wandb_logger.upload_checkpoint(checkpoint_path, epoch)
    except Exception as e:
        print(f"⚠ Could not upload checkpoint to wandb: {e}")


##############################
# Save Model Parameters
##############################
def save_model_params(run_dir, param_info=None):
    """
    Save model parameters to a pickle file.
    
    Args:
        run_dir: Directory to save parameters in
        param_info: Optional parameter count information from count_model_parameters()
    """
    # Get all settings
    data_settings = settings.get_data_settings()
    model_architecture = settings.get_model_architecture()
    training_settings = settings.get_training_settings()
    latent_optimization = settings.get_latent_optimization()
    evaluation_settings = settings.get_evaluation_settings()
    
    # Try to load parameter info from results if not provided
    if param_info is None:
        try:
            results_file = os.path.join(run_dir, 'results.pkl')
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                param_info = results.get('model_parameter_info', {})
        except Exception:
            param_info = {}
    
    # Basic model parameters
    model_params = {
        # 'KEY': data_settings['key'], # Old: single key
        'TRAINING_KEYS': data_settings.get('training_keys', [data_settings.get('key')]), # New: list of keys
        'TRAINING_SEED': data_settings['training_seed'],
        'EVAL_SEED': data_settings['eval_seed'],
        'DROPOUT': model_architecture['dropout'],
        'MAX_LENGTH': model_architecture['max_length'],
        'ENCODER_MAX_LENGTH': model_architecture['encoder_max_length'],
        'DECODER_MAX_LENGTH': model_architecture['decoder_max_length'],
        'NUM_ENCODERS': model_architecture.get('num_encoders', 1),  # Add NUM_ENCODERS parameter
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
    
    # Add parameter count information if available
    if param_info:
        model_params.update({
            'total_params': param_info.get('total_params', 'N/A'),
            'trainable_params': param_info.get('trainable_params', 'N/A'),
            'non_trainable_params': param_info.get('non_trainable_params', 'N/A'),
            'is_multi_encoder_model': param_info.get('is_multi_encoder', False),
            'num_encoders_actual': param_info.get('num_encoders', 1),
            'component_breakdown': param_info.get('component_breakdown', {}),
        })
        
        # Add multi-encoder specific parameters
        if param_info.get('is_multi_encoder', False):
            model_params.update({
                'encoder_breakdown': param_info.get('encoder_breakdown', {}),
                'decoder_params': param_info.get('decoder_params', 'N/A'),
                'shared_params': param_info.get('shared_params', 'N/A'),
                'avg_encoder_params': param_info.get('avg_encoder_params', 'N/A'),
                'total_encoder_params': param_info.get('total_encoder_params', 'N/A')
            })
    
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
    # Pass parameter info if available in results
    param_info = results.get('model_parameter_info', None)
    save_model_params(run_dir, param_info)


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
def load_model(run_dir, epoch=None, device='cuda', model_type='lpn'):
    """
    Load a model from a run directory.
    
    Args:
        run_dir (str): Path to the run directory
        epoch (int, optional): Specific epoch to load. If None, loads the latest checkpoint.
        device (str): Device to load the model on ('cuda' or 'cpu')
        model_type (str): Model type (legacy parameter - now automatically determined from settings)
    
    Returns:
        model: Loaded model (automatically configured for single or multi-encoder)
        optimizer: Loaded optimizer
        epoch: Epoch number of the loaded checkpoint
        loss: Loss value of the loaded checkpoint
    """
    from models.base_model import LatentProgramNetwork, compute_loss
    from torch.optim import Adam
    
    # LatentProgramNetwork automatically handles both single and multi-encoder configurations
    # based on the settings, so we don't need separate model types anymore
    model = LatentProgramNetwork().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)  # Default learning rate
    
    if epoch is None:
        # Check for full_joint.ckpt first
        full_joint_path = os.path.join(run_dir, 'full_joint.ckpt')
        if os.path.exists(full_joint_path):
            print("=== MODEL LOADING ===")
            print("Loading full_joint.ckpt")
            checkpoint_path = full_joint_path
        else:
            # Find the latest checkpoint
            checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch')]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {run_dir}")
            # Extract all available epochs and find the latest
            available_epochs = [int(f.split('_')[1][5:].split('.')[0]) for f in checkpoints]
            latest_epoch = max(available_epochs)
            print(f"=== MODEL LOADING ===")
            print(f"Available checkpoints: {sorted(available_epochs)}")
            print(f"No specific epoch requested - selecting latest epoch: {latest_epoch}")
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch{latest_epoch}.pt')
            epoch = latest_epoch
    else:
        print(f"=== MODEL LOADING ===")
        print(f"Loading specific epoch: {epoch}")
        checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch{epoch}.pt')
    
    if not os.path.exists(checkpoint_path):
        available_checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch')]
        available_epochs = [int(f.split('_')[1][5:].split('.')[0]) for f in available_checkpoints] if available_checkpoints else []
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\nAvailable epochs: {sorted(available_epochs)}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except (KeyError, ValueError) as e:
        if isinstance(e, KeyError):
            print(f"⚠ Warning: Optimizer state dict not found in checkpoint: {e}")
        else:
            print(f"⚠ Warning: Could not load optimizer state dict (possibly different parameter groups): {e}")
        print("   Proceeding without loading optimizer state.")
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Successfully loaded epoch {epoch} (training loss: {loss:.6f})")
    
    # Set to evaluation mode
    model.eval()
    
    # Print model configuration for debugging
    if hasattr(model, 'is_multi_encoder'):
        if model.is_multi_encoder:
            print(f"Loaded multi-encoder model with {model.num_encoders} encoders")
        else:
            print("Loaded single-encoder model")
    
    return model, optimizer, epoch, loss



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


##############################
# Collect Latent Data Helper
##############################
def collect_latent_data(model, dataloader, device, encoder_idx=None, max_samples=100, data_type=None):
    """Collect latent representations from a model.

    Args:
        model: The model providing the latent representations.
        dataloader: DataLoader yielding input/output pairs.
        device: Device to perform computation on.
        encoder_idx: Optional index of the encoder to use (``None`` for PoE or single encoder).
        max_samples: Maximum number of samples to collect.
        data_type: Optional string describing the data being collected.

    Returns:
        dict: Dictionary containing latent statistics and metadata.
    """
    model.eval()

    latent_data = {
        'latent_mus': [],
        'latent_log_vars': [],
        'latent_zs': [],
        'input_samples': [],
        'output_samples': [],
        'num_samples': 0
    }

    if encoder_idx is not None:
        latent_data['encoder_idx'] = encoder_idx
    if data_type is not None:
        latent_data['data_type'] = data_type

    with torch.no_grad():
        sample_count = 0
        for batch_input, batch_target in dataloader:
            if sample_count >= max_samples:
                break

            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            if hasattr(model, 'is_multi_encoder') and model.is_multi_encoder:
                if encoder_idx is not None:
                    mu, log_var = model(batch_input, batch_target, encoder_idx=encoder_idx)[1:3]
                else:
                    mu, log_var = model(batch_input, batch_target)[1:3]
            else:
                mu, log_var,_ = model.encoder(batch_input, batch_target)

            z = model.reparameterize(mu, log_var)

            batch_size = min(batch_input.size(0), max_samples - sample_count)
            latent_data['latent_mus'].append(mu[:batch_size].cpu().numpy())
            latent_data['latent_log_vars'].append(log_var[:batch_size].cpu().numpy())
            latent_data['latent_zs'].append(z[:batch_size].cpu().numpy())
            latent_data['input_samples'].append(batch_input[:batch_size].cpu().numpy())
            latent_data['output_samples'].append(batch_target[:batch_size].cpu().numpy())

            sample_count += batch_size

    if latent_data['latent_mus']:
        latent_data['latent_mus'] = np.concatenate(latent_data['latent_mus'], axis=0)
        latent_data['latent_log_vars'] = np.concatenate(latent_data['latent_log_vars'], axis=0)
        latent_data['latent_zs'] = np.concatenate(latent_data['latent_zs'], axis=0)
        latent_data['input_samples'] = np.concatenate(latent_data['input_samples'], axis=0)
        latent_data['output_samples'] = np.concatenate(latent_data['output_samples'], axis=0)
        latent_data['num_samples'] = len(latent_data['latent_mus'])

    return latent_data
