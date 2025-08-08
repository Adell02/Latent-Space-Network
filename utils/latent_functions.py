import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.settings_manager import settings
import logging
from models.base_model import get_latent_dim

# Set up logging
logger = logging.getLogger(__name__)

def optimize_latent_z(lpn, input_seq, target_seq, num_steps=None, lr=None, return_trajectory=False,
                     encoder_idx=None, use_independent_decoder=False):
    """
    Optimize latent z using gradient ascent to maximize p(y|x,z).
    Works for both single and multi-encoder models.
    """
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    
    # Use settings if parameters are not provided
    if num_steps is None:
        num_steps = latent_optimization['training']['num_steps']
    if lr is None:
        lr = latent_optimization['training']['learning_rate']
        
    batch_size = input_seq.size(0)
    device = input_seq.device
    
    # DEBUG: Check input data format
    logger.debug(f"    DEBUG: input_seq shape: {input_seq.shape}")
    logger.debug(f"    DEBUG: target_seq shape: {target_seq.shape}")
    logger.debug(f"    DEBUG: input_seq range: [{input_seq.min().item():.4f}, {input_seq.max().item():.4f}]")
    logger.debug(f"    DEBUG: target_seq range: [{target_seq.min().item():.4f}, {target_seq.max().item():.4f}]")
    
    # Check sequence lengths
    if input_seq.shape[1] != 902:
        logger.warning(f"    WARNING: input_seq length is {input_seq.shape[1]}, expected 902")
    if target_seq.shape[1] != 902:
        logger.warning(f"    WARNING: target_seq length is {target_seq.shape[1]}, expected 902")
    
    # Check target sequence format
    if not torch.is_tensor(target_seq):
        logger.warning(f"    WARNING: target_seq is not a tensor, type: {type(target_seq)}")
        target_seq = torch.tensor(target_seq, device=device, dtype=torch.float32)
    
    # Check for valid target values
    if torch.isnan(target_seq).any():
        logger.error(f"    ERROR: target_seq contains NaN values!")
    if torch.isinf(target_seq).any():
        logger.error(f"    ERROR: target_seq contains Inf values!")
    
    # Get initial latent representation from encoder
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if encoder_idx is not None:
                # Use specific encoder
                mu, log_var, _ = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
            else:
                # For multi-encoder, use PoE inference mode
                result = lpn(input_seq, target_seq)
                (shape_logits, grid_logits), mu, log_var, _ = result
        else:
            # For single encoder
            mu, log_var, _ = lpn.encoder(input_seq, target_seq)
    
    # DEBUG: Check if using VQ-VAE
    is_vq_vae = hasattr(lpn, 'is_using_vq_vae') and lpn.is_using_vq_vae()
    if is_vq_vae:
        logger.debug(f"    DEBUG: Model is using VQ-VAE")
        logger.debug(f"    DEBUG: mu shape: {mu.shape}, log_var shape: {log_var.shape}")
        # For VQ-VAE, mu contains quantized latents, log_var contains VQ loss
        z = mu.detach().clone().requires_grad_(True)
    else:
        # DEBUG: Check encoder outputs
        if torch.isnan(mu).any():
            logger.error(f"    ERROR: NaN detected in encoder mu!")
            logger.error(f"    ERROR: mu shape: {mu.shape}")
            logger.error(f"    ERROR: mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
        if torch.isnan(log_var).any():
            logger.error(f"    ERROR: NaN detected in encoder log_var!")
            logger.error(f"    ERROR: log_var shape: {log_var.shape}")
            logger.error(f"    ERROR: log_var range: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")
        
        # Initialize z with the encoder's output
        z = lpn.reparameterize(mu, log_var).detach().clone().requires_grad_(True)
    
    initial_z = z.detach().clone()
    
    # DEBUG: Check initial z
    if torch.isnan(z).any():
        logger.error(f"    ERROR: NaN detected in initial z!")
        logger.error(f"    ERROR: z shape: {z.shape}")
        logger.error(f"    ERROR: z range: [{z.min().item():.4f}, {z.max().item():.4f}]")
        if is_vq_vae:
            logger.error(f"    ERROR: VQ-VAE mode - z should be quantized latents")
        else:
            logger.error(f"    ERROR: This suggests NaN was introduced during reparameterization")
            logger.error(f"    ERROR: mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            logger.error(f"    ERROR: log_var range: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")

    # Compute individual initial losses for each sample
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if use_independent_decoder and encoder_idx is not None:
                # Use specific encoder's independent decoder
                shape_logits_init, grid_logits_init = lpn.multi_encoder.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
            else:
                # Use shared decoder (default for multi-encoder)
                shape_logits_init, grid_logits_init = lpn.multi_encoder.shared_decoder(z, input_seq, target_seq=target_seq)
        else:
            shape_logits_init, grid_logits_init = lpn.decoder(z, input_seq, target_seq=target_seq)
        
        # Compute per-sample initial losses
        initial_losses = []
        
        for i in range(batch_size):
            # Shape loss for this sample
            shape_pred_i = shape_logits_init[i:i+1]
            shape_tgt_i = target_seq[i:i+1, 900:902].long()
            shape_pred_reshaped = shape_pred_i.reshape(-1, 31)
            shape_tgt_reshaped = shape_tgt_i.reshape(-1)
            
            try:
                shape_loss_i = F.cross_entropy(shape_pred_reshaped, shape_tgt_reshaped)
            except Exception as e:
                logger.error(f"    ERROR: Sample {i} - shape loss computation failed: {e}")
                continue
            
            # Grid loss for this sample
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            
            if active_pixels > 0:
                grid_pred_i = grid_logits_init[i, :active_pixels]
                grid_tgt_i = target_seq[i, :active_pixels].long()
                
                try:
                    grid_loss_i = F.cross_entropy(grid_pred_i, grid_tgt_i)
                except Exception as e:
                    logger.error(f"    ERROR: Sample {i} - grid loss computation failed: {e}")
                    continue
            else:
                grid_loss_i = torch.tensor(0.0, device=device)
            
            sample_loss = (shape_loss_i + grid_loss_i).item()
            
            # Check for non-finite values
            if not torch.isfinite(torch.tensor(sample_loss)):
                logger.error(f"    ERROR: Sample {i} - Non-finite loss detected: {sample_loss}")
                continue
                
            initial_losses.append(sample_loss)
        
        if not initial_losses:
            logger.error(f"    ERROR: No initial losses computed! Empty batch or all samples failed.")
            return z, float('inf'), {} if return_trajectory else None
        
        # Check for non-finite losses
        if not all(torch.isfinite(torch.tensor(l)) for l in initial_losses):
            logger.error(f"    ERROR: Non-finite losses detected in initial_losses: {initial_losses}")
            return z, float('inf'), {} if return_trajectory else None
        
        # Compute batch average losses
        batch_avg_losses = [sum(initial_losses) / len(initial_losses)]

    # Initialize optimizer for z
    optimizer_z = torch.optim.Adam([z], lr=lr)
    
    # Initialize trajectory tracking
    individual_losses = [initial_losses]
    trajectory = {'z_vectors': [initial_z], 'losses': [batch_avg_losses[0]]} if return_trajectory else None

    pbar = tqdm(range(num_steps), desc="Gradient ascent", unit="step", leave=False)

    for step in pbar:
        optimizer_z.zero_grad()
        
        # DEBUG: Check inputs to decoder
        if torch.isnan(z).any():
            logger.error(f"    ERROR: NaN detected in z before decoder call!")
            logger.error(f"    ERROR: z shape: {z.shape}")
            logger.error(f"    ERROR: z range: [{z.min().item():.4f}, {z.max().item():.4f}]")
        if torch.isnan(input_seq).any():
            logger.error(f"    ERROR: NaN detected in input_seq before decoder call!")
        if torch.isnan(target_seq).any():
            logger.error(f"    ERROR: NaN detected in target_seq before decoder call!")
        
        # Decode using the current z
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if use_independent_decoder and encoder_idx is not None:
                # Use specific encoder's independent decoder
                shape_logits, grid_logits = lpn.multi_encoder.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
            else:
                # Use shared decoder (default for multi-encoder)
                shape_logits, grid_logits = lpn.multi_encoder.shared_decoder(z, input_seq, target_seq=target_seq)
        else:
            shape_logits, grid_logits = lpn.decoder(z, input_seq, target_seq=target_seq)

        # DEBUG: Check decoder outputs for NaN
        if torch.isnan(shape_logits).any():
            logger.error(f"    ERROR: NaN detected in shape_logits from decoder!")
            logger.error(f"    ERROR: shape_logits shape: {shape_logits.shape}")
            logger.error(f"    ERROR: shape_logits range: [{shape_logits.min().item():.4f}, {shape_logits.max().item():.4f}]")
        if torch.isnan(grid_logits).any():
            logger.error(f"    ERROR: NaN detected in grid_logits from decoder!")
            logger.error(f"    ERROR: grid_logits shape: {grid_logits.shape}")
            logger.error(f"    ERROR: grid_logits range: [{grid_logits.min().item():.4f}, {grid_logits.max().item():.4f}]")

        # Compute batch loss for backpropagation (keep existing approach)
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

        grid_loss_list = []
        for i in range(batch_size):
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits[i, :active_pixels],
                                       target_seq[i, :active_pixels].long())
                grid_loss_list.append(loss_i)

        grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else \
                   torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = shape_loss + grid_loss
        
        # Backward pass
        total_loss.backward()
        optimizer_z.step()
        
        # Compute individual sample losses for this step
        step_losses = []
        with torch.no_grad():
            for i in range(batch_size):
                # Shape loss for this sample
                shape_pred_i = shape_logits[i:i+1]
                shape_tgt_i = target_seq[i:i+1, 900:902].long()
                shape_loss_i = F.cross_entropy(shape_pred_i.reshape(-1, 31), shape_tgt_i.reshape(-1))
                
                # Grid loss for this sample
                tgt_rows = int(target_seq[i, 900].item())
                tgt_cols = int(target_seq[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0:
                    grid_loss_i = F.cross_entropy(grid_logits[i, :active_pixels],
                                                target_seq[i, :active_pixels].long())
                else:
                    grid_loss_i = torch.tensor(0.0, device=device)
                
                sample_loss = (shape_loss_i + grid_loss_i).item()
                step_losses.append(sample_loss)
            
        individual_losses.append(step_losses)
        batch_avg_loss = sum(step_losses) / len(step_losses)
        batch_avg_losses.append(batch_avg_loss)

        # Update trajectory - ensure losses is always a list for consistency
        if return_trajectory:
            trajectory['z_vectors'].append(z.detach().clone())
            # Keep losses as a list for consistency with other optimization methods
            trajectory['losses'].append(batch_avg_loss)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_avg_loss:.4f}',
            'shape': f'{shape_loss.item():.4f}',
            'grid': f'{grid_loss.item():.4f}'
        })

    if return_trajectory:
        return z, batch_avg_losses[-1], trajectory
    else:
        return z, batch_avg_losses[-1]

def optimize_latent_z_from_initial(lpn, input_seq, target_seq, initial_z, num_steps=None, lr=None, return_trajectory=False,
                                  encoder_idx=None, use_independent_decoder=False):
    """
    Optimize latent z using gradient ascent starting from a provided initial z vector.
    This ensures the trajectory starts from exactly the same point as the support sample.
    Works for both single and multi-encoder models.
    """
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    
    # Use settings if parameters are not provided
    if num_steps is None:
        num_steps = latent_optimization['training']['num_steps']
    if lr is None:
        lr = latent_optimization['training']['learning_rate']
        
    batch_size = input_seq.size(0)
    device = input_seq.device
    
    # Use the provided initial z vector (from support sample)
    z = initial_z.detach().clone()
    initial_z = z.detach().clone()

    # Compute individual initial losses for each sample
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if use_independent_decoder and encoder_idx is not None:
                # Use specific encoder's independent decoder
                shape_logits_init, grid_logits_init = lpn.multi_encoder.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
            else:
                # Use shared decoder (default for multi-encoder)
                shape_logits_init, grid_logits_init = lpn.multi_encoder.shared_decoder(z, input_seq, target_seq=target_seq)
        else:
            shape_logits_init, grid_logits_init = lpn.decoder(z, input_seq, target_seq=target_seq)
        
        # Compute per-sample initial losses
        initial_losses = []
        
        for i in range(batch_size):
            # Shape loss for this sample
            shape_pred_i = shape_logits_init[i:i+1]
            shape_tgt_i = target_seq[i:i+1, 900:902].long()
            shape_pred_reshaped = shape_pred_i.reshape(-1, 31)
            shape_tgt_reshaped = shape_tgt_i.reshape(-1)
            
            try:
                shape_loss_i = F.cross_entropy(shape_pred_reshaped, shape_tgt_reshaped)
            except Exception as e:
                logger.error(f"    ERROR: Sample {i} - shape loss computation failed: {e}")
                continue
            
            # Grid loss for this sample
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            
            if active_pixels > 0:
                grid_pred_i = grid_logits_init[i, :active_pixels]
                grid_tgt_i = target_seq[i, :active_pixels].long()
                
                try:
                    grid_loss_i = F.cross_entropy(grid_pred_i, grid_tgt_i)
                except Exception as e:
                    logger.error(f"    ERROR: Sample {i} - grid loss computation failed: {e}")
                    continue
            else:
                grid_loss_i = torch.tensor(0.0, device=device)
            
            sample_loss = (shape_loss_i + grid_loss_i).item()
            
            # Check for non-finite values
            if not torch.isfinite(torch.tensor(sample_loss)):
                logger.error(f"    ERROR: Sample {i} - Non-finite loss detected: {sample_loss}")
                continue
                
            initial_losses.append(sample_loss)
        
        if not initial_losses:
            logger.error(f"    ERROR: No initial losses computed! Empty batch or all samples failed.")
            return z, float('inf'), {} if return_trajectory else None
        
        # Check for non-finite losses
        if not all(torch.isfinite(torch.tensor(l)) for l in initial_losses):
            logger.error(f"    ERROR: Non-finite losses detected in initial_losses: {initial_losses}")
            return z, float('inf'), {} if return_trajectory else None
        
        # Compute batch average losses
        batch_avg_losses = [sum(initial_losses) / len(initial_losses)]

    # Initialize optimizer for z
    optimizer_z = torch.optim.Adam([z], lr=lr)
    
    # Initialize trajectory tracking
    individual_losses = [initial_losses]
    trajectory = {'z_vectors': [initial_z], 'losses': [batch_avg_losses[0]]} if return_trajectory else None

    pbar = tqdm(range(num_steps), desc="Gradient ascent", unit="step", leave=False)

    for step in pbar:
        optimizer_z.zero_grad()
        
        # Decode using the current z
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if use_independent_decoder and encoder_idx is not None:
                # Use specific encoder's independent decoder
                shape_logits, grid_logits = lpn.multi_encoder.independent_decoders[encoder_idx](z, input_seq, target_seq=target_seq)
            else:
                # Use shared decoder (default for multi-encoder)
                shape_logits, grid_logits = lpn.multi_encoder.shared_decoder(z, input_seq, target_seq=target_seq)
        else:
            shape_logits, grid_logits = lpn.decoder(z, input_seq, target_seq=target_seq)
        
        # Compute per-sample losses
        step_losses = []
        total_loss = 0
        
        for i in range(batch_size):
            # Shape loss for this sample
            shape_pred_i = shape_logits[i:i+1]
            shape_tgt_i = target_seq[i:i+1, 900:902].long()
            shape_loss_i = F.cross_entropy(shape_pred_i.reshape(-1, 31), shape_tgt_i.reshape(-1))
            
            # Grid loss for this sample
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                grid_loss_i = F.cross_entropy(grid_logits[i, :active_pixels],
                                            target_seq[i, :active_pixels].long())
            else:
                grid_loss_i = torch.tensor(0.0, device=device)
            
            sample_loss = shape_loss_i + grid_loss_i
            step_losses.append(sample_loss.item())
            total_loss += sample_loss
        
        # Backward pass
        total_loss.backward()
        optimizer_z.step()
        
        # Track losses
        batch_avg_loss = total_loss.item() / batch_size
        individual_losses.append(step_losses)
        batch_avg_losses.append(batch_avg_loss)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{batch_avg_loss:.4f}'})
        
        # Store trajectory if requested
        if return_trajectory:
            trajectory['z_vectors'].append(z.detach().clone())
            # Keep losses as a list for consistency with other optimization methods
            trajectory['losses'].append(batch_avg_losses[-1])
    
    
    if return_trajectory:
        return z.detach(), batch_avg_losses[-1], trajectory
    else:
        return z.detach(), batch_avg_losses[-1]

def evolutionary_optimize_latent_z(lpn, input_seq, target_seq, population_size=None,
                                   num_generations=None, mutation_std=None, return_trajectory=False):
    """
    Optimize latent z using an evolutionary algorithm with progress bar and trajectory logging.
    """
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    
    # Use settings if parameters are not provided
    evolutionary_settings = latent_optimization['evolutionary']
    if population_size is None:
        population_size = evolutionary_settings['population_size']
    if num_generations is None:
        num_generations = evolutionary_settings['num_generations']
    if mutation_std is None:
        mutation_std = evolutionary_settings['mutation_std']
        
    batch_size = input_seq.size(0)
        
    device = input_seq.device
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            # For multi-encoder, use PoE inference mode
            (shape_logits, grid_logits), mu, log_var, _ = lpn(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
        else:
            # For single encoder
            mu, log_var,_ = lpn.encoder(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
    
    # Compute initial loss
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            shape_logits_init, grid_logits_init = lpn.multi_encoder.decoder(initial_z, input_seq, target_seq=target_seq)
        else:
            shape_logits_init, grid_logits_init = lpn.decoder(initial_z, input_seq, target_seq=target_seq)
        
        shape_targets = target_seq[:, 900:902].long()
        shape_loss_init = F.cross_entropy(shape_logits_init.reshape(-1, 31), shape_targets.reshape(-1))
        
        grid_loss_list_init = []
        for i in range(batch_size):
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits_init[i, :active_pixels],
                                       target_seq[i, :active_pixels].long())
                grid_loss_list_init.append(loss_i)

        grid_loss_init = sum(grid_loss_list_init) / len(grid_loss_list_init) if grid_loss_list_init else \
                       torch.tensor(0.0, device=device)
        initial_loss = (shape_loss_init + grid_loss_init).item()
            
    population = initial_z.unsqueeze(0).repeat(population_size, 1, 1)
    population = population + torch.randn_like(population) * mutation_std
    best_candidate = None
    
    # Track trajectory information if requested
    trajectory = {
        'z_vectors': [initial_z.detach().clone()],
        'losses': [initial_loss],
        'best_fitness_per_generation': [initial_loss],
        'population_diversity': [],
        'encoder_mu': mu.detach().clone(),
        'encoder_log_var': log_var.detach().clone(),
        'initial_z': initial_z.detach().clone(),
        'method': 'evolutionary'
    } if return_trajectory else None
    
    
    # Create progress bar for generations
    pbar = tqdm(range(num_generations), desc="Evolutionary", unit="gen", leave=False)
    
    for gen in pbar:
        candidate_losses = []
        generation_candidates = []
        
        for i in range(population_size):
            candidate_z = population[i]
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                shape_logits, grid_logits = lpn.multi_encoder.decoder(candidate_z, input_seq, target_seq=target_seq)
            else:
                shape_logits, grid_logits = lpn.decoder(candidate_z, input_seq, target_seq=target_seq)
                
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='mean')
            grid_loss_list = []
            for j in range(batch_size):
                tgt_rows = int(target_seq[j, 900].item())
                tgt_cols = int(target_seq[j, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0:
                    loss_j = F.cross_entropy(grid_logits[j, :active_pixels], target_seq[j, :active_pixels].long(), reduction='mean')
                    grid_loss_list.append(loss_j)
            grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else torch.tensor(0.0, device=device)
            reconstruction_loss = shape_loss + grid_loss
            candidate_losses.append(reconstruction_loss.item())
            generation_candidates.append(candidate_z.detach().clone())
        
        sorted_indices = sorted(range(population_size), key=lambda i: candidate_losses[i])
        best_loss = candidate_losses[sorted_indices[0]]
        best_candidate = population[sorted_indices[0]].detach()
        
        # Track trajectory information
        if return_trajectory:
            trajectory['z_vectors'].append(best_candidate.detach().clone())
            trajectory['losses'].append(best_loss)
            trajectory['best_fitness_per_generation'].append(best_loss)
            
            # Calculate population diversity (average pairwise distance)
            pop_flat = torch.stack(generation_candidates).view(population_size, -1)
            pairwise_dists = torch.cdist(pop_flat, pop_flat, p=2)
            # Get upper triangular part (excluding diagonal) to avoid double counting
            upper_tri_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
            diversity = pairwise_dists[upper_tri_mask].mean().item()
            trajectory['population_diversity'].append(diversity)
        
        # Update progress bar
        pbar.set_postfix({
            'best_loss': f'{best_loss:.4f}',
            'avg_loss': f'{sum(candidate_losses)/len(candidate_losses):.4f}'
        })
        
        num_selected = population_size // 2
        selected_candidates = population[sorted_indices[:num_selected]]
        offspring = selected_candidates + torch.randn_like(selected_candidates) * mutation_std
        population = torch.cat([selected_candidates, offspring], dim=0)
        if population.size(0) > population_size:
            population = population[:population_size]
        elif population.size(0) < population_size:
            extra = best_candidate.unsqueeze(0).repeat(population_size - population.size(0), 1, 1)
            population = torch.cat([population, extra], dim=0)
    
    pbar.close()
    
    loss_improvement = initial_loss - best_loss
    
    if return_trajectory:
        return best_candidate, trajectory['losses'], trajectory
    else:
        return best_candidate, None

##############################
# New: Optimize latent z via a Voronoi‑Inspired Search
##############################
def voronoi_optimize_latent_z(lpn, input_seq, target_seq, population_size=None,
                              num_generations=None, diversity_weight=None,
                              mutation_std=None, return_trajectory=False):
    """
    Optimize latent z using a Voronoi-inspired approach with progress bar and trajectory logging.
    """
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    
    # Use settings if parameters are not provided
    voronoi_settings = latent_optimization['voronoi']
    if population_size is None:
        population_size = voronoi_settings['population_size']
    if num_generations is None:
        num_generations = voronoi_settings['num_generations']
    if diversity_weight is None:
        diversity_weight = voronoi_settings['diversity_weight']
    if mutation_std is None:
        mutation_std = voronoi_settings['mutation_std']
        
    batch_size = input_seq.size(0)
    logger.info(f"    Voronoi optimization: {num_generations} generations, "
          f"Pop size: {population_size}, Diversity weight: {diversity_weight}, "
          f"Mutation std: {mutation_std}, Batch size: {batch_size}")
        
    device = input_seq.device
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            # For multi-encoder, use PoE inference mode
            (shape_logits, grid_logits), mu, log_var, _ = lpn(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
        else:
            # For single encoder
            mu, log_var,_ = lpn.encoder(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
    
    # Compute initial loss
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            shape_logits_init, grid_logits_init = lpn.multi_encoder.decoder(initial_z, input_seq, target_seq=target_seq)
        else:
            shape_logits_init, grid_logits_init = lpn.decoder(initial_z, input_seq, target_seq=target_seq)
        
        shape_targets = target_seq[:, 900:902].long()
        shape_loss_init = F.cross_entropy(shape_logits_init.reshape(-1, 31), shape_targets.reshape(-1))
        
        grid_loss_list_init = []
        for i in range(batch_size):
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                loss_i = F.cross_entropy(grid_logits_init[i, :active_pixels],
                                       target_seq[i, :active_pixels].long())
                grid_loss_list_init.append(loss_i)

        grid_loss_init = sum(grid_loss_list_init) / len(grid_loss_list_init) if grid_loss_list_init else \
                       torch.tensor(0.0, device=device)
        initial_loss = (shape_loss_init + grid_loss_init).item()
            
    # Create an initial population
    population = initial_z.unsqueeze(0).repeat(population_size, 1, 1)
    population = population + torch.randn_like(population) * mutation_std

    def compute_diversity(pop):
        # For each candidate, compute the minimum Euclidean distance to any other candidate.
        # This serves as a proxy for the size of its Voronoi cell.
        pop_flat = pop.view(pop.size(0), -1)  # shape: (population_size, batch_size * latent_dim)
        diversity_scores = []
        for i in range(pop_flat.size(0)):
            candidate = pop_flat[i]
            # Compute distances to all other candidates
            dists = torch.norm(pop_flat - candidate.unsqueeze(0), dim=1)
            # Set self-distance to a large value so it is not the minimum.
            dists[i] = 1e6
            diversity_scores.append(dists.min().item())
        return diversity_scores

    best_candidate = None
    
    # Track trajectory information if requested
    trajectory = {
        'z_vectors': [initial_z.detach().clone()],
        'losses': [initial_loss],
        'combined_scores': [initial_loss],  # For Voronoi, track both loss and combined score
        'diversity_scores': [],
        'population_diversity': [],
        'encoder_mu': mu.detach().clone(),
        'encoder_log_var': log_var.detach().clone(),
        'initial_z': initial_z.detach().clone(),
        'method': 'voronoi',
        'diversity_weight': diversity_weight
    } if return_trajectory else None
    
    
    # Create progress bar for generations
    pbar = tqdm(range(num_generations), desc="Voronoi", unit="gen", leave=False)
    
    for gen in pbar:
        candidate_losses = []
        generation_candidates = []
        
        for i in range(population_size):
            candidate_z = population[i]
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                shape_logits, grid_logits = lpn.multi_encoder.decoder(candidate_z, input_seq, target_seq=target_seq)
            else:
                shape_logits, grid_logits = lpn.decoder(candidate_z, input_seq, target_seq=target_seq)
                
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='mean')
            grid_loss_list = []
            for j in range(batch_size):
                tgt_rows = int(target_seq[j, 900].item())
                tgt_cols = int(target_seq[j, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0:
                    loss_j = F.cross_entropy(grid_logits[j, :active_pixels], target_seq[j, :active_pixels].long(), reduction='mean')
                    grid_loss_list.append(loss_j)
            grid_loss = sum(grid_loss_list) / len(grid_loss_list) if grid_loss_list else torch.tensor(0.0, device=device)
            reconstruction_loss = shape_loss + grid_loss
            candidate_losses.append(reconstruction_loss.item())
            generation_candidates.append(candidate_z.detach().clone())
        
        diversity_scores = compute_diversity(population)
        # Combine reconstruction loss and diversity. Lower is better.
        combined_scores = [loss - diversity_weight * div for loss, div in zip(candidate_losses, diversity_scores)]
        sorted_indices = sorted(range(population_size), key=lambda i: combined_scores[i])
        best_score = combined_scores[sorted_indices[0]]
        best_loss = candidate_losses[sorted_indices[0]]
        best_candidate = population[sorted_indices[0]].detach()
        
        # Track trajectory information
        if return_trajectory:
            trajectory['z_vectors'].append(best_candidate.detach().clone())
            trajectory['losses'].append(best_loss)
            trajectory['combined_scores'].append(best_score)
            trajectory['diversity_scores'].append(diversity_scores[sorted_indices[0]])
            
            # Calculate population diversity (average pairwise distance)
            pop_flat = torch.stack(generation_candidates).view(population_size, -1)
            pairwise_dists = torch.cdist(pop_flat, pop_flat, p=2)
            # Get upper triangular part (excluding diagonal) to avoid double counting
            upper_tri_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
            pop_diversity = pairwise_dists[upper_tri_mask].mean().item()
            trajectory['population_diversity'].append(pop_diversity)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{best_loss:.4f}',
            'score': f'{best_score:.4f}',
            'diversity': f'{diversity_scores[sorted_indices[0]]:.3f}'
        })
        
        # Selection: take the top half based on combined score.
        num_selected = population_size // 2
        selected_candidates = population[sorted_indices[:num_selected]]
        offspring = selected_candidates + torch.randn_like(selected_candidates) * mutation_std
        population = torch.cat([selected_candidates, offspring], dim=0)
        if population.size(0) > population_size:
            population = population[:population_size]
        elif population.size(0) < population_size:
            extra = best_candidate.unsqueeze(0).repeat(population_size - population.size(0), 1, 1)
            population = torch.cat([population, extra], dim=0)
    
    pbar.close()
    
    loss_improvement = initial_loss - best_loss
    
    if return_trajectory:
        return best_candidate, trajectory['losses'], trajectory
    else:
        return best_candidate, None

##############################
# Helper: Choose Optimization Method
##############################
def get_optimized_z(lpn, input_seq, target_seq, num_steps=None, lr=None, context='training', 
                   return_trajectory=False, encoder_idx=None, use_independent_decoder=False):
    """
    Returns an optimized latent z using either gradient-based, evolutionary, or Voronoi-inspired search,
    depending on the optimization method setting.
    
    MEAN OPTIMIZATION: If num_steps=0, returns the mean of the posterior (no sampling).
    
    Args:
        lpn: The model
        input_seq: Input sequence tensor
        target_seq: Target sequence tensor
        num_steps: Number of optimization steps (overrides settings if provided)
        lr: Learning rate (overrides settings if provided)
        context: 'training' or 'inference' - determines which settings to use as defaults
        return_trajectory: Whether to return trajectory information
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
    """
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    optimization_method = latent_optimization.get('method', 'gradient')
    
    # Determine which settings to use based on context
    if context == 'inference':
        enabled = latent_optimization['inference']['enabled']
        default_num_steps = latent_optimization['inference']['num_steps']
        default_lr = latent_optimization['inference']['learning_rate']
    else:  # training context
        enabled = latent_optimization['training']['enabled']
        default_num_steps = latent_optimization['training']['num_steps']
        default_lr = latent_optimization['training']['learning_rate']
    
    # Use provided parameters or fall back to context-appropriate defaults
    final_num_steps = num_steps if num_steps is not None else default_num_steps
    final_lr = lr if lr is not None else default_lr
    
    # MEAN OPTIMIZATION: If steps=0, return mean of posterior (no sampling)
    if final_num_steps == 0:
        logger.info(f"    Mean optimization: num_steps=0, returning mean of posterior (no sampling)")
        with torch.no_grad():
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                if encoder_idx is not None:
                    # Use specific encoder
                    mu, log_var, _ = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                else:
                    # Use PoE inference mode
                    (shape_logits, grid_logits), mu, log_var, _ = lpn(input_seq, target_seq)
                # Return mean directly (no sampling)
                z = mu
            else:
                # For single encoder
                mu, log_var, _ = lpn.encoder(input_seq, target_seq)
                # Return mean directly (no sampling)
                z = mu
            
            if return_trajectory:
                # Create minimal trajectory for mean optimization
                trajectory = {
                    'z_vectors': [z.detach().clone()],
                    'losses': [0.0],  # No optimization loss for mean
                    'method': 'mean_optimization',
                    'encoder_mu': mu.detach().clone(),
                    'encoder_log_var': log_var.detach().clone(),
                    'initial_z': z.detach().clone()
                }
                return z, 0.0, trajectory
            else:
                return z, 0.0, None
    
    if enabled:
        logger.info(f"    Latent optimization ENABLED: method={optimization_method}, steps={final_num_steps}, lr={final_lr}")
        if optimization_method == "gradient":
            with torch.enable_grad():
                result = optimize_latent_z(lpn, input_seq, target_seq, 
                                       num_steps=final_num_steps, lr=final_lr, 
                                       return_trajectory=return_trajectory,
                                       encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder)
                logger.info(f"    Optimization completed, trajectory returned: {result[2] is not None if len(result) > 2 else False}")
                # Ensure we always return 3 values
                if len(result) == 2:
                    z, loss = result
                    return z, loss, None
                else:
                    return result
        elif optimization_method == "evolutionary":
            # For evolutionary and voronoi, we don't use num_steps/lr but their own parameters
            # However, we could map num_steps to num_generations if needed
            result = evolutionary_optimize_latent_z(lpn, input_seq, target_seq, 
                                                return_trajectory=return_trajectory)
            # Ensure we always return 3 values
            if len(result) == 2:
                z, loss = result
                return z, loss, None
            else:
                return result
        elif optimization_method == "voronoi":
            result = voronoi_optimize_latent_z(lpn, input_seq, target_seq,
                                           return_trajectory=return_trajectory)
            # Ensure we always return 3 values
            if len(result) == 2:
                z, loss = result
                return z, loss, None
            else:
                return result
        else:
            # Unknown method, fall back to basic sampling
            with torch.no_grad():
                if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                    if encoder_idx is not None:
                        # Use specific encoder
                        mu, log_var, _ = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                    else:
                        # Use PoE inference mode
                        (shape_logits, grid_logits), mu, log_var, _ = lpn(input_seq, target_seq)
                    z = lpn.reparameterize(mu, log_var)
                else:
                    # For single encoder
                    mu, log_var, _ = lpn.encoder(input_seq, target_seq)
                    z = lpn.reparameterize(mu, log_var)
                
                return z, 0.0, None
    else:
        # Optimization disabled, use basic sampling
        logger.info(f"    Latent optimization DISABLED, using basic sampling")
        with torch.no_grad():
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                if encoder_idx is not None:
                    # Use specific encoder
                    mu, log_var, _ = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                else:
                    # Use PoE inference mode
                    (shape_logits, grid_logits), mu, log_var, _ = lpn(input_seq, target_seq)
                z = lpn.reparameterize(mu, log_var)
            else:
                # For single encoder
                mu, log_var, _ = lpn.encoder(input_seq, target_seq)
                z = lpn.reparameterize(mu, log_var)
            
            return z, 0.0, None

def get_optimized_z_from_initial(lpn, input_seq, target_seq, initial_z, num_steps=None, lr=None, context='training', 
                                return_trajectory=False, encoder_idx=None, use_independent_decoder=False):
    """
    Returns an optimized latent z starting from a provided initial z vector.
    This ensures the trajectory starts from exactly the same point as the support sample.
    """
    # DEBUG: Check input types
    logger.debug(f"DEBUG: get_optimized_z_from_initial called")
    logger.debug(f"DEBUG: initial_z type: {type(initial_z)}")
    logger.debug(f"DEBUG: initial_z shape: {initial_z.shape if hasattr(initial_z, 'shape') else 'no shape'}")
    if hasattr(initial_z, 'cpu'):
        logger.debug(f"DEBUG: initial_z is tensor")
    else:
        logger.debug(f"DEBUG: initial_z is NOT tensor!")
        # Convert to tensor if it's not already
        if isinstance(initial_z, np.ndarray):
            logger.debug(f"DEBUG: Converting numpy array to tensor")
            initial_z = torch.tensor(initial_z, dtype=torch.float32, device=input_seq.device)
            logger.debug(f"DEBUG: Converted initial_z shape: {initial_z.shape}")
        else:
            logger.error(f"ERROR: initial_z is neither tensor nor numpy array!")
            raise ValueError(f"initial_z must be tensor or numpy array, got {type(initial_z)}")
    
    # Get settings from settings manager (moved inside function for sweep compatibility)
    latent_optimization = settings.get_latent_optimization()
    optimization_method = latent_optimization.get('method', 'gradient')
    
    # Determine which settings to use based on context
    if context == 'inference':
        enabled = latent_optimization['inference']['enabled']
        default_num_steps = latent_optimization['inference']['num_steps']
        default_lr = latent_optimization['inference']['learning_rate']
    else:  # training context
        enabled = latent_optimization['training']['enabled']
        default_num_steps = latent_optimization['training']['num_steps']
        default_lr = latent_optimization['training']['learning_rate']
    
    # Use provided parameters or fall back to context-appropriate defaults
    final_num_steps = num_steps if num_steps is not None else default_num_steps
    final_lr = lr if lr is not None else default_lr
    
    if enabled:
        logger.info(f"    Latent optimization ENABLED: method={optimization_method}, steps={final_num_steps}, lr={final_lr}")
        if optimization_method == "gradient":
            with torch.enable_grad():
                result = optimize_latent_z_from_initial(lpn, input_seq, target_seq, initial_z,
                                                     num_steps=final_num_steps, lr=final_lr, 
                                                     return_trajectory=return_trajectory,
                                                     encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder)
                logger.info(f"    Optimization completed, trajectory returned: {result[2] is not None if len(result) > 2 else False}")
                return result
        else:
            # For other methods, fall back to regular optimization
            logger.warning(f"    Warning: Using regular optimization for method {optimization_method}")
            return get_optimized_z(lpn, input_seq, target_seq, num_steps=final_num_steps, lr=final_lr,
                                 context=context, return_trajectory=return_trajectory,
                                 encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder)
    else:
        # Optimization disabled, return the initial z
        logger.info(f"    Latent optimization DISABLED, returning initial z")
        if return_trajectory:
            return initial_z, None, None
        else:
            return initial_z, None

def optimize_task_latent(lpn, support_samples, task_key, num_steps=None, lr=None, encoder_idx=None, use_independent_decoder=False):
    """
    Optimize ONE latent vector to explain ALL support samples for a given task.
    This is the key difference from per-sample optimization - it creates task-level clustering.
    
    Args:
        lpn: The model
        support_samples: List of (input_seq, target_seq) tuples for this task
        task_key: The task identifier
        num_steps: Number of optimization steps
        lr: Learning rate
        encoder_idx: Which encoder to use (for multi-encoder models)
        use_independent_decoder: Whether to use independent decoder
    
    Returns:
        task_latent: Single optimized latent representing the entire task
        final_loss: Final reconstruction loss for all support samples
        trajectory: Optimization trajectory (optional)
    """
    # Get settings from settings manager
    latent_optimization = settings.get_latent_optimization()
    
    # Use settings if parameters are not provided
    if num_steps is None:
        num_steps = latent_optimization['training']['num_steps']
    if lr is None:
        lr = latent_optimization['training']['learning_rate']
    
    if not support_samples:
        raise ValueError("No support samples provided for task optimization")
    
    device = support_samples[0][0].device
    
    # Initialize task latent from first support sample
    with torch.no_grad():
        first_input, first_target = support_samples[0]
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if encoder_idx is not None:
                mu, log_var, _ = lpn.multi_encoder.encoders[encoder_idx](first_input, first_target)
            else:
                result = lpn(first_input, first_target)
                (shape_logits, grid_logits), mu, log_var, _ = result
        else:
            mu, log_var, _ = lpn.encoder(first_input, first_target)
        
        # Initialize task latent
        task_latent = lpn.reparameterize(mu, log_var)
    
    # Detach and enable gradients
    task_latent = task_latent.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([task_latent], lr=lr)
    
    logger.info(f"    Task-level optimization for '{task_key}': {len(support_samples)} support samples, {num_steps} steps, LR: {lr}")
    
    # Track trajectory
    trajectory = {
        'z_vectors': [task_latent.detach().clone()],
        'losses': [],
        'task_key': task_key,
        'num_support_samples': len(support_samples)
    }
    
    # Optimization loop
    pbar = tqdm(range(num_steps), desc=f"Task opt: {task_key}", unit="step", leave=False)
    
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Compute loss for ALL support samples using the same task latent
        for input_seq, target_seq in support_samples:
            batch_size = input_seq.size(0)
            
            # Expand task latent to match batch size
            expanded_task_latent = task_latent.expand(batch_size, -1)
            
            # Forward pass through decoder
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                if use_independent_decoder and encoder_idx is not None:
                    shape_logits, grid_logits = lpn.multi_encoder.independent_decoders[encoder_idx](
                        expanded_task_latent, input_seq, target_seq=target_seq)
                else:
                    shape_logits, grid_logits = lpn.multi_encoder.shared_decoder(
                        expanded_task_latent, input_seq, target_seq=target_seq)
            else:
                shape_logits, grid_logits = lpn.decoder(expanded_task_latent, input_seq, target_seq=target_seq)
            
            # Compute reconstruction loss for this support sample
            # For task-level optimization, we need to compute loss manually
            # because compute_loss expects the model to handle the latent
            # But we're manually optimizing the latent outside the model

            # Manual loss computation (like in the original optimize_latent_z function):
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

            batch_size = target_seq.size(0)
            grid_loss_sum = 0.0
            active_samples = 0
            for i in range(batch_size):
                r, c = int(target_seq[i, 900].item()), int(target_seq[i, 901].item())
                n_pix = r * c
                if n_pix > 0:
                    grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long())
                    active_samples += 1
            grid_loss = grid_loss_sum / active_samples if active_samples > 0 else torch.tensor(0.0, device=device)
            loss = shape_loss + grid_loss
            total_loss += loss
        
        # Average loss across all support samples
        avg_loss = total_loss / len(support_samples)
        
        # Backward pass and optimize
        avg_loss.backward()
        optimizer.step()
        
        # Track progress
        loss_value = avg_loss.item()
        trajectory['losses'].append(loss_value)
        trajectory['z_vectors'].append(task_latent.detach().clone())
        
        pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    final_loss = trajectory['losses'][-1] if trajectory['losses'] else None
    logger.info(f"    Task '{task_key}' optimization complete. Final loss: {final_loss:.4f}")
    
    return task_latent.detach(), final_loss, trajectory