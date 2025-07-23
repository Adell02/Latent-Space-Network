import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, IterableDataset
import os
import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any
from utils.model_utils import prepare_dataloader
from utils.data_preparation import transform_grid_to_sequence



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

class InfiniteARCDataset(IterableDataset):
    """Iterable dataset that generates or loads ARC examples on-the-fly."""

    def __init__(self, task_keys: List[str], batch_size: int, batches_per_epoch: int,
                 seed: int = 42, data_dir: str = None):
        self.task_keys = task_keys
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.data_dir = data_dir

        # Load pre-generated examples if available
        self.pre_generated = {}
        if data_dir:
            for key in task_keys:
                path = os.path.join(data_dir, f"{key}.json")
                if os.path.exists(path):
                    with open(path, "r") as fp:
                        try:
                            self.pre_generated[key] = json.load(fp)
                        except Exception:
                            self.pre_generated[key] = []

        # Retrieve generator functions
        self.generators = {k: getattr(__import__('re_arc.generators', fromlist=['']),'generate_' + k) for k in task_keys}
        self._epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch * self.batch_size

    def _sample_example(self, key: str, rng: random.Random) -> dict:
        examples = self.pre_generated.get(key)
        if examples:
            return rng.choice(examples)
        generator = self.generators[key]
        return generator(0, 1)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(len(self)):
            key = rng.choice(self.task_keys)
            example = self._sample_example(key, rng)
            input_seq = transform_grid_to_sequence(np.array(example['input']))
            output_seq = transform_grid_to_sequence(np.array(example['output']))
            yield torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)


def create_infinite_dataloader(task_keys: List[str], batch_size: int, batches_per_epoch: int,
                               seed: int = 42, data_dir: str = None) -> DataLoader:
    """Helper to create a DataLoader backed by ``InfiniteARCDataset``."""
    dataset = InfiniteARCDataset(task_keys, batch_size, batches_per_epoch, seed=seed, data_dir=data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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
    """Load individual encoder checkpoint with architecture mismatch handling."""
    checkpoint_path = os.path.join(run_dir, f'encoder_{encoder_idx}.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    
    encoder_state = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'encoders'):
        try:
            model.multi_encoder.encoders[encoder_idx].load_state_dict(encoder_state, strict=True)
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key(s)" in str(e):
                print(f"⚠ Architecture mismatch for encoder {encoder_idx}: {str(e)[:100]}...")
                print(f"⚠ This indicates config was changed between Phase A and Phase B")
                raise RuntimeError(f"Architecture mismatch when loading encoder {encoder_idx}. "
                                 f"This suggests the model configuration changed between phases. "
                                 f"Ensure the same frozen config is used for all phases.")
            else:
                raise e
    else:
        raise ValueError("Model does not have multi-encoder structure")


def load_all_encoder_checkpoints(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """Load all encoder checkpoints for Phase B and C with architecture mismatch handling."""
    if not hasattr(model, 'multi_encoder') or not hasattr(model.multi_encoder, 'encoders'):
        raise ValueError("Model does not have multi-encoder structure")
    
    num_encoders = len(model.multi_encoder.encoders)
    
    for encoder_idx in range(num_encoders):
        try:
            load_encoder_checkpoint(model, encoder_idx, run_dir, device)
            print(f"✓ Loaded encoder {encoder_idx} checkpoint")
        except FileNotFoundError:
            print(f"⚠ Encoder {encoder_idx} checkpoint not found - will start from random initialization")
        except RuntimeError as e:
            if "Architecture mismatch" in str(e):
                # This is our custom architecture mismatch error
                print(f"✗ Failed to load encoder {encoder_idx}: Architecture mismatch detected")
                print(f"✗ This indicates the model configuration changed between phases")
                raise e  # Re-raise to stop execution and force user to fix config
            else:
                # Other runtime errors
                print(f"⚠ Failed to load encoder {encoder_idx}: {e}")
                raise e


def save_decoder_checkpoint(model: nn.Module, run_dir: str) -> str:
    """Save shared decoder checkpoint."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'shared_decoder'):
        decoder_state = model.multi_encoder.shared_decoder.state_dict()
    elif hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
        decoder_state = model.multi_encoder.decoder.state_dict()
    elif hasattr(model, 'decoder'):
        decoder_state = model.decoder.state_dict()
    else:
        raise ValueError("Model does not have decoder")
    
    checkpoint_path = os.path.join(run_dir, 'shared_decoder.ckpt')
    torch.save(decoder_state, checkpoint_path)
    return checkpoint_path


def load_decoder_checkpoint(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """Load shared decoder checkpoint."""
    checkpoint_path = os.path.join(run_dir, 'shared_decoder.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Shared decoder checkpoint not found: {checkpoint_path}")
    
    decoder_state = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'shared_decoder'):
        model.multi_encoder.shared_decoder.load_state_dict(decoder_state)
    elif hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'decoder'):
        model.multi_encoder.decoder.load_state_dict(decoder_state)
    elif hasattr(model, 'decoder'):
        model.decoder.load_state_dict(decoder_state)
    else:
        raise ValueError("Model does not have decoder")


def save_independent_decoder_checkpoint(model: nn.Module, encoder_idx: int, run_dir: str) -> str:
    """Save individual independent decoder checkpoint."""
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'independent_decoders'):
        decoder_state = model.multi_encoder.independent_decoders[encoder_idx].state_dict()
    else:
        raise ValueError("Model does not have independent decoders")
    
    checkpoint_path = os.path.join(run_dir, f'independent_decoder_{encoder_idx}.ckpt')
    torch.save(decoder_state, checkpoint_path)
    return checkpoint_path


def load_independent_decoder_checkpoint(model: nn.Module, encoder_idx: int, run_dir: str, device: str = 'cuda') -> None:
    """Load individual independent decoder checkpoint with architecture mismatch handling."""
    checkpoint_path = os.path.join(run_dir, f'independent_decoder_{encoder_idx}.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Independent decoder checkpoint not found: {checkpoint_path}")
    
    decoder_state = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'multi_encoder') and hasattr(model.multi_encoder, 'independent_decoders'):
        try:
            model.multi_encoder.independent_decoders[encoder_idx].load_state_dict(decoder_state, strict=True)
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key(s)" in str(e):
                # Architecture mismatch - this happens when config changes between phases
                print(f"⚠ Architecture mismatch for independent decoder {encoder_idx}: {str(e)[:100]}...")
                print(f"⚠ This indicates config was changed between Phase A and Phase B")
                print(f"⚠ Skipping checkpoint loading - decoder will use random initialization")
                raise RuntimeError(f"Architecture mismatch when loading independent decoder {encoder_idx}. "
                                 f"This suggests the model configuration changed between phases. "
                                 f"Ensure the same frozen config is used for all phases.")
            else:
                # Re-raise other RuntimeErrors
                raise e
    else:
        raise ValueError("Model does not have independent decoders")


def save_all_independent_decoder_checkpoints(model: nn.Module, run_dir: str) -> List[str]:
    """Save all independent decoder checkpoints."""
    if not hasattr(model, 'multi_encoder') or not hasattr(model.multi_encoder, 'independent_decoders'):
        raise ValueError("Model does not have independent decoders")
    
    num_encoders = len(model.multi_encoder.independent_decoders)
    checkpoint_paths = []
    
    for encoder_idx in range(num_encoders):
        checkpoint_path = save_independent_decoder_checkpoint(model, encoder_idx, run_dir)
        checkpoint_paths.append(checkpoint_path)
    
    return checkpoint_paths


def load_all_independent_decoder_checkpoints(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """Load all independent decoder checkpoints with architecture mismatch handling."""
    if not hasattr(model, 'multi_encoder') or not hasattr(model.multi_encoder, 'independent_decoders'):
        raise ValueError("Model does not have independent decoders")
    
    num_encoders = len(model.multi_encoder.independent_decoders)
    
    for encoder_idx in range(num_encoders):
        try:
            load_independent_decoder_checkpoint(model, encoder_idx, run_dir, device)
            print(f"✓ Loaded independent decoder {encoder_idx} checkpoint")
        except FileNotFoundError:
            print(f"⚠ Independent decoder {encoder_idx} checkpoint not found - will start from random initialization")
        except RuntimeError as e:
            if "Architecture mismatch" in str(e):
                # This is our custom architecture mismatch error
                print(f"✗ Failed to load independent decoder {encoder_idx}: Architecture mismatch detected")
                print(f"✗ This indicates the model configuration changed between phases")
                raise e  # Re-raise to stop execution and force user to fix config
            else:
                # Other runtime errors
                print(f"⚠ Failed to load independent decoder {encoder_idx}: {e}")
                raise e


def initialize_shared_decoder_from_independent_decoders(model: nn.Module, run_dir: str, device: str = 'cuda') -> None:
    """
    Initialize shared decoder with averaged weights from all independent decoders.
    This provides a good starting point for Phase B training.
    
    Args:
        model: Multi-encoder model with independent and shared decoders
        run_dir: Run directory containing independent decoder checkpoints
        device: Device for loading checkpoints
    """
    if not hasattr(model, 'multi_encoder'):
        raise ValueError("Model does not have multi-encoder structure")
    
    if not hasattr(model.multi_encoder, 'independent_decoders'):
        raise ValueError("Model does not have independent decoders")
    
    if not hasattr(model.multi_encoder, 'shared_decoder'):
        raise ValueError("Model does not have shared decoder")
    
    num_encoders = len(model.multi_encoder.independent_decoders)
    print(f"\nInitializing shared decoder from {num_encoders} independent decoders...")
    
    # First, ensure all independent decoders are loaded
    load_all_independent_decoder_checkpoints(model, run_dir, device)
    
    # Get the shared decoder's state dict structure
    shared_decoder_state = model.multi_encoder.shared_decoder.state_dict()
    
    # Collect all independent decoder states
    independent_states = []
    for encoder_idx in range(num_encoders):
        independent_state = model.multi_encoder.independent_decoders[encoder_idx].state_dict()
        independent_states.append(independent_state)
    
    # Average the weights across all independent decoders
    averaged_state = {}
    for key in shared_decoder_state.keys():
        if key in independent_states[0]:  # Ensure the key exists in independent decoders
            # Stack tensors from all independent decoders and compute mean
            stacked_weights = torch.stack([state[key] for state in independent_states])
            averaged_state[key] = torch.mean(stacked_weights, dim=0)
            print(f"  ✓ Averaged parameter: {key} (shape: {averaged_state[key].shape})")
        else:
            # Keep original weights if not found in independent decoders
            averaged_state[key] = shared_decoder_state[key]
            print(f"  ⚠ Kept original parameter: {key} (not found in independent decoders)")
    
    # Load the averaged weights into the shared decoder
    model.multi_encoder.shared_decoder.load_state_dict(averaged_state)
    
    print("✓ Shared decoder initialized with averaged weights from independent decoders")
    print("  This provides a warm start for Phase B training based on Phase A knowledge")


def save_full_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, run_dir: str) -> str:
    """Save full model checkpoint with complete structure."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    # Only save optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
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