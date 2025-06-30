import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from typing import List, Tuple, Dict, Any

from utils.model_utils import prepare_dataloader


class MixedDomainsDataset(Dataset):
    """Dataset that mixes data from multiple encoders for round-robin training."""
    
    def __init__(self, encoder_datasets: List[Tuple[List, List]], num_encoders: int):
        """
        Args:
            encoder_datasets: List of (inputs, outputs) tuples for each encoder
            num_encoders: Number of encoders
        """
        self.encoder_datasets = []
        self.encoder_indices = []
        self.total_length = 0
        
        # Process each encoder's data
        for encoder_idx, (inputs, outputs) in enumerate(encoder_datasets):
            if inputs and outputs:
                # Convert to tensors if needed
                if isinstance(inputs[0], np.ndarray):
                    inputs = [torch.tensor(inp, dtype=torch.float32) for inp in inputs]
                    outputs = [torch.tensor(out, dtype=torch.float32) for out in outputs]
                
                self.encoder_datasets.append(list(zip(inputs, outputs)))
                self.encoder_indices.extend([encoder_idx] * len(inputs))
                self.total_length += len(inputs)
        
        # Create round-robin order for balanced sampling
        self._create_round_robin_order()
    
    def _create_round_robin_order(self):
        """Create round-robin sampling order to ensure balanced encoder training."""
        # Get max samples per encoder
        encoder_lengths = [len(dataset) for dataset in self.encoder_datasets]
        max_length = max(encoder_lengths) if encoder_lengths else 0
        
        self.round_robin_order = []
        
        # Build round-robin pattern
        for i in range(max_length):
            for encoder_idx, dataset in enumerate(self.encoder_datasets):
                if i < len(dataset):
                    # Add (encoder_idx, sample_idx) pairs
                    self.round_robin_order.append((encoder_idx, i))
    
    def __len__(self):
        return len(self.round_robin_order)
    
    def __getitem__(self, idx):
        encoder_idx, sample_idx = self.round_robin_order[idx]
        input_seq, output_seq = self.encoder_datasets[encoder_idx][sample_idx]
        
        # Return input, output, and encoder_idx for routing
        return input_seq, output_seq, encoder_idx


def create_mixed_domains_dataloader(encoder_datasets: List[Tuple[List, List]], 
                                  num_encoders: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader that mixes data from multiple encoders in round-robin fashion.
    
    Args:
        encoder_datasets: List of (inputs, outputs) tuples for each encoder
        num_encoders: Number of encoders
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader that yields (input_batch, output_batch, encoder_indices_batch)
    """
    dataset = MixedDomainsDataset(encoder_datasets, num_encoders)
    
    def collate_fn(batch):
        """Custom collate function to handle encoder indices."""
        inputs, outputs, encoder_indices = zip(*batch)
        
        # Stack tensors
        input_batch = torch.stack(inputs)
        output_batch = torch.stack(outputs)
        encoder_indices_batch = torch.tensor(encoder_indices, dtype=torch.long)
        
        return input_batch, output_batch, encoder_indices_batch
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def freeze_all_parameters(model: nn.Module) -> None:
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False


def freeze_encoders(model: nn.Module) -> None:
    """Freeze only encoder parameters, keep decoder trainable."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
        # Multi-encoder model
        for encoder in model.multi_encoder.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
    elif hasattr(model, 'encoder'):
        # Single encoder model
        for param in model.encoder.parameters():
            param.requires_grad = False


def freeze_decoder(model: nn.Module) -> None:
    """Freeze only decoder parameters, keep encoders trainable."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
        # Multi-encoder model
        for param in model.multi_encoder.decoder.parameters():
            param.requires_grad = False
    elif hasattr(model, 'decoder'):
        # Single/unified model
        for param in model.decoder.parameters():
            param.requires_grad = False


def unfreeze_all_parameters(model: nn.Module) -> None:
    """Unfreeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """Get list of trainable parameters."""
    return [param for param in model.parameters() if param.requires_grad]


def create_phase_optimizer(model: nn.Module, phase: str, base_lr: float, encoder_lr_mult: float = 0.05) -> torch.optim.Adam:
    """
    Create optimizer with appropriate parameter groups for different training phases.
    
    Args:
        model: The model to create optimizer for
        phase: Training phase ('pretrain', 'decoder', 'joint_ft')
        base_lr: Base learning rate
        encoder_lr_mult: Learning rate multiplier for encoders in joint fine-tuning
        
    Returns:
        Adam optimizer with appropriate parameter groups
    """
    if phase in ['pretrain', 'decoder']:
        # Single learning rate for all trainable parameters
        trainable_params = get_trainable_parameters(model)
        return Adam(trainable_params, lr=base_lr)
    
    elif phase == 'joint_ft':
        # Two parameter groups: encoders (reduced LR) and decoder (full LR)
        encoder_params = []
        decoder_params = []
        
        if hasattr(model, 'multi_encoder'):
            # Multi-encoder model
            for encoder in model.multi_encoder.encoders:
                encoder_params.extend([p for p in encoder.parameters() if p.requires_grad])
            decoder_params.extend([p for p in model.multi_encoder.decoder.parameters() if p.requires_grad])
        else:
            # Single encoder model
            if hasattr(model, 'encoder'):
                encoder_params.extend([p for p in model.encoder.parameters() if p.requires_grad])
            if hasattr(model, 'decoder'):
                decoder_params.extend([p for p in model.decoder.parameters() if p.requires_grad])
        
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': base_lr * encoder_lr_mult})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': base_lr})
        
        return Adam(param_groups)
    
    else:
        raise ValueError(f"Unknown phase: {phase}")


def setup_phase_training(model: nn.Module, phase: str) -> None:
    """
    Setup model parameters for specific training phase.
    
    Args:
        model: Model to setup
        phase: Training phase ('pretrain', 'decoder', 'joint_ft')
    """
    if phase == 'pretrain':
        # For pretraining, individual encoders will be handled in training loop
        unfreeze_all_parameters(model)
    elif phase == 'decoder':
        # Freeze encoders, unfreeze decoder
        freeze_encoders(model)
        # Ensure decoder is unfrozen
        if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
            for param in model.multi_encoder.decoder.parameters():
                param.requires_grad = True
        elif hasattr(model, 'decoder'):
            for param in model.decoder.parameters():
                param.requires_grad = True
    elif phase == 'joint_ft':
        # Unfreeze everything
        unfreeze_all_parameters(model)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def save_phase_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, phase: str, 
                         epoch: int, loss: float, run_dir: str, encoder_idx: int = None) -> str:
    """
    Save checkpoint for specific training phase.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        phase: Training phase
        epoch: Current epoch
        loss: Current loss
        run_dir: Run directory
        encoder_idx: Encoder index (for pretrain phase)
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'phase': phase,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if encoder_idx is not None:
        checkpoint['encoder_idx'] = encoder_idx
        checkpoint_path = os.path.join(run_dir, f'checkpoint_{phase}_encoder_{encoder_idx}_epoch{epoch}.pt')
    else:
        checkpoint_path = os.path.join(run_dir, f'checkpoint_{phase}_epoch{epoch}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_phase_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cuda') -> Tuple[int, float]:
    """
    Load checkpoint for specific training phase.
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def save_encoder_checkpoint(model: nn.Module, encoder_idx: int, run_dir: str) -> str:
    """Save individual encoder checkpoint."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
        encoder_state = model.multi_encoder.encoders[encoder_idx].state_dict()
    else:
        raise ValueError("Model does not have multi-encoder structure")
    
    checkpoint_path = os.path.join(run_dir, f'encoder_{encoder_idx}.ckpt')
    torch.save(encoder_state, checkpoint_path)
    return checkpoint_path


def load_encoder_checkpoint(model: nn.Module, encoder_idx: int, run_dir: str, device: str = 'cuda') -> None:
    """Load individual encoder checkpoint."""
    checkpoint_path = os.path.join(run_dir, f'encoder_{encoder_idx}.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    
    encoder_state = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
        model.multi_encoder.encoders[encoder_idx].load_state_dict(encoder_state)
    else:
        raise ValueError("Model does not have multi-encoder structure")


def load_all_encoder_checkpoints(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """Load all encoder checkpoints for Phase B and C."""
    if not hasattr(model, 'multi_encoder') or not hasattr(model.multi_encoder, 'encoders'):
        raise ValueError("Model does not have multi-encoder structure")
    
    num_encoders = len(model.multi_encoder.encoders)
    
    for encoder_idx in range(num_encoders):
        try:
            load_encoder_checkpoint(model, encoder_idx, run_dir, device)
            print(f"✓ Loaded encoder {encoder_idx} checkpoint")
        except FileNotFoundError:
            print(f"⚠ Encoder {encoder_idx} checkpoint not found - will start from random initialization")


def save_decoder_checkpoint(model: nn.Module, run_dir: str) -> str:
    """Save decoder checkpoint."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
        decoder_state = model.multi_encoder.decoder.state_dict()
    elif hasattr(model, 'decoder'):
        decoder_state = model.decoder.state_dict()
    else:
        raise ValueError("Model does not have decoder")
    
    checkpoint_path = os.path.join(run_dir, 'decoder.ckpt')
    torch.save(decoder_state, checkpoint_path)
    return checkpoint_path


def load_decoder_checkpoint(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """Load decoder checkpoint."""
    checkpoint_path = os.path.join(run_dir, 'decoder.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")
    
    decoder_state = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
        model.multi_encoder.decoder.load_state_dict(decoder_state)
    elif hasattr(model, 'decoder'):
        model.decoder.load_state_dict(decoder_state)
    else:
        raise ValueError("Model does not have decoder")


def save_full_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, run_dir: str) -> str:
    """Save full model checkpoint with complete structure."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    checkpoint_path = os.path.join(run_dir, 'full_joint.ckpt')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def print_parameter_status(model: nn.Module, phase: str) -> None:
    """Print parameter status for current phase."""
    param_counts = count_trainable_parameters(model)
    
    print(f"\n=== PARAMETER STATUS - {phase.upper()} PHASE ===")
    print(f"Total parameters: {param_counts['total_params']:,}")
    print(f"Trainable parameters: {param_counts['trainable_params']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable_params']:,}")
    print(f"Trainable ratio: {param_counts['trainable_params']/param_counts['total_params']:.1%}")
    print("=" * 50) 