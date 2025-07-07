import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.settings_manager import settings

def optimize_latent_z(lpn, input_seq, target_seq, num_steps=None, lr=None, return_trajectory=False,
                     encoder_idx=None, use_independent_decoder=False):
    """
    Optimize latent z via gradient descent with per-sample loss tracking.
    
    Args:
        encoder_idx: Specific encoder to use (None for PoE inference in multi-encoder)
        use_independent_decoder: Whether to use independent decoder vs shared decoder
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
        
    # Get initial latent parameters from the model and compute initial z.
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            if encoder_idx is not None:
                # Use specific encoder
                mu, log_var = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
            else:
                # For multi-encoder, use PoE inference mode
                reconstruction, mu, log_var = lpn(input_seq, target_seq)
        else:
            # For single encoder
            mu, log_var = lpn.encoder(input_seq, target_seq)
    
    z = lpn.reparameterize(mu, log_var)
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
            shape_loss_i = F.cross_entropy(shape_pred_i.reshape(-1, 31), shape_tgt_i.reshape(-1))
            
            # Grid loss for this sample
            tgt_rows = int(target_seq[i, 900].item())
            tgt_cols = int(target_seq[i, 901].item())
            active_pixels = tgt_rows * tgt_cols
            if active_pixels > 0:
                grid_loss_i = F.cross_entropy(grid_logits_init[i, :active_pixels],
                                            target_seq[i, :active_pixels].long())
            else:
                grid_loss_i = torch.tensor(0.0, device=device)
            
            sample_loss = (shape_loss_i + grid_loss_i).item()
            initial_losses.append(sample_loss)

    # Detach z from the graph and enable gradients on it.
    z = z.detach().requires_grad_(True)
    optimizer_z = torch.optim.Adam([z], lr=lr)

    # Track individual sample losses and batch average
    individual_losses = [initial_losses]  # Each element is a list of losses for all samples at that step
    batch_avg_losses = [sum(initial_losses) / len(initial_losses)]
    
    # Track trajectory information if requested
    trajectory = {
        'z_vectors': [initial_z.detach().clone()],
        'losses': batch_avg_losses[0],  # Keep batch average for backward compatibility
        'individual_losses': individual_losses,  # NEW: Per-sample losses for each step
        'encoder_mu': mu.detach().clone(),
        'encoder_log_var': log_var.detach().clone(),
        'initial_z': initial_z.detach().clone()
    } if return_trajectory else None

    print(f"    Gradient ascent: {num_steps} steps, LR: {lr}, Batch size: {batch_size}")
    print(f"    Initial batch avg loss: {batch_avg_losses[0]:.4f}")

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

        reconstruction_loss = shape_loss + grid_loss
        
        # Compute individual sample losses for tracking (no gradients needed)
        with torch.no_grad():
            step_individual_losses = []
            for i in range(batch_size):
                # Shape loss for this sample
                shape_pred_i = shape_logits[i:i+1].detach()
                shape_tgt_i = target_seq[i:i+1, 900:902].long()
                shape_loss_i = F.cross_entropy(shape_pred_i.reshape(-1, 31), shape_tgt_i.reshape(-1))
                
                # Grid loss for this sample
                tgt_rows = int(target_seq[i, 900].item())
                tgt_cols = int(target_seq[i, 901].item())
                active_pixels = tgt_rows * tgt_cols
                if active_pixels > 0:
                    grid_loss_i = F.cross_entropy(grid_logits[i, :active_pixels].detach(),
                                                target_seq[i, :active_pixels].long())
                else:
                    grid_loss_i = torch.tensor(0.0, device=device)
                
                sample_loss = (shape_loss_i + grid_loss_i).item()
                step_individual_losses.append(sample_loss)
            
            individual_losses.append(step_individual_losses)
            batch_avg_losses.append(sum(step_individual_losses) / len(step_individual_losses))

        pbar.set_postfix({
            'avg_loss': f'{batch_avg_losses[-1]:.4f}',
            'min_loss': f'{min(step_individual_losses):.4f}',
            'max_loss': f'{max(step_individual_losses):.4f}'
        })

        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(z, 1.0)
        optimizer_z.step()

        # Store trajectory information after optimization step
        if return_trajectory:
            trajectory['z_vectors'].append(z.detach().clone())
            trajectory['losses'] = batch_avg_losses[-1]  # Keep scalar for compatibility
            trajectory['individual_losses'] = individual_losses

    pbar.close()

    loss_improvement = batch_avg_losses[0] - batch_avg_losses[-1]
    print(f"    ✓ Optimization complete: "
          f"Avg loss {batch_avg_losses[0]:.4f} → {batch_avg_losses[-1]:.4f} "
          f"(Δ: {loss_improvement:+.4f})")

    if return_trajectory:
        return z, batch_avg_losses, trajectory
    else:
        return z, batch_avg_losses

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
    print(f"    Evolutionary optimization: {num_generations} generations, "
          f"Pop size: {population_size}, Mutation std: {mutation_std}, Batch size: {batch_size}")
        
    device = input_seq.device
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            # For multi-encoder, use PoE inference mode
            reconstruction, mu, log_var = lpn(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
        else:
            # For single encoder
            mu, log_var = lpn.encoder(input_seq, target_seq)
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
    
    print(f"    Initial loss: {initial_loss:.4f}")
    
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
    print(f"    ✓ Evolutionary optimization complete: "
          f"Loss {initial_loss:.4f} → {best_loss:.4f} "
          f"(Δ: {loss_improvement:+.4f})")
    
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
    print(f"    Voronoi optimization: {num_generations} generations, "
          f"Pop size: {population_size}, Diversity weight: {diversity_weight}, "
          f"Mutation std: {mutation_std}, Batch size: {batch_size}")
        
    device = input_seq.device
    with torch.no_grad():
        if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
            # For multi-encoder, use PoE inference mode
            reconstruction, mu, log_var = lpn(input_seq, target_seq)
            initial_z = lpn.reparameterize(mu, log_var).detach()
        else:
            # For single encoder
            mu, log_var = lpn.encoder(input_seq, target_seq)
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
    
    print(f"    Initial loss: {initial_loss:.4f}")
    
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
    print(f"    ✓ Voronoi optimization complete: "
          f"Loss {initial_loss:.4f} → {best_loss:.4f} "
          f"(Δ: {loss_improvement:+.4f}), Best score: {best_score:.4f}")
    
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
    
    if enabled:
        if optimization_method == "gradient":
            with torch.enable_grad():
                return optimize_latent_z(lpn, input_seq, target_seq, 
                                       num_steps=final_num_steps, lr=final_lr, 
                                       return_trajectory=return_trajectory,
                                       encoder_idx=encoder_idx, use_independent_decoder=use_independent_decoder)
        elif optimization_method == "evolutionary":
            # For evolutionary and voronoi, we don't use num_steps/lr but their own parameters
            # However, we could map num_steps to num_generations if needed
            return evolutionary_optimize_latent_z(lpn, input_seq, target_seq, 
                                                return_trajectory=return_trajectory)
        elif optimization_method == "voronoi":
            return voronoi_optimize_latent_z(lpn, input_seq, target_seq,
                                           return_trajectory=return_trajectory)
        else:
            # Unknown method, fall back to basic sampling
            with torch.no_grad():
                if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                    if encoder_idx is not None:
                        # Use specific encoder
                        mu, log_var = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                    else:
                        # Use PoE inference mode
                        reconstruction, mu, log_var = lpn(input_seq, target_seq)
                    z = lpn.reparameterize(mu, log_var)
                else:
                    # For single encoder
                    mu, log_var = lpn.encoder(input_seq, target_seq)
                    z = lpn.reparameterize(mu, log_var)
            if return_trajectory:
                return z, None, None
            else:
                return z, None
    else:
        # Optimization disabled, just use basic sampling
        with torch.no_grad():
            if hasattr(lpn, 'is_multi_encoder') and lpn.is_multi_encoder:
                if encoder_idx is not None:
                    # Use specific encoder
                    mu, log_var = lpn.multi_encoder.encoders[encoder_idx](input_seq, target_seq)
                else:
                    # Use PoE inference mode
                    reconstruction, mu, log_var = lpn(input_seq, target_seq)
                z = lpn.reparameterize(mu, log_var)
            else:
                # For single encoder
                mu, log_var = lpn.encoder(input_seq, target_seq)
                z = lpn.reparameterize(mu, log_var)
        if return_trajectory:
            return z, None, None
        else:
            return z, None