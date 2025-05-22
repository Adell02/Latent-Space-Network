import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.settings_manager import settings

# Get settings from settings manager
latent_optimization = settings.get_latent_optimization()

def optimize_latent_z(lpn, input_seq, target_seq, num_steps=None, lr=None):
    """
    Optimize latent z via gradient descent with logging to verify optimization.
    """
    # Use settings if parameters are not provided
    if num_steps is None:
        num_steps = latent_optimization['training']['num_steps']
    if lr is None:
        lr = latent_optimization['training']['learning_rate']
        
    # Get initial latent parameters from the encoder and compute initial z.
    mu, log_var = lpn.encoder(input_seq, target_seq)
    z = lpn.reparameterize(mu, log_var)
    initial_z = z.detach().clone()

    # Detach z from the graph and enable gradients on it.
    z = z.detach().requires_grad_(True)

    # Create an optimizer for z.
    optimizer_z = torch.optim.Adam([z], lr=lr)

    # Track losses and z changes
    losses = []
    z_changes = []

    for step in range(num_steps):
        optimizer_z.zero_grad()
        # Decode using the current z.
        shape_logits, grid_logits = lpn.decoder(z, input_seq, target_seq=target_seq)

        # Compute loss on the shape tokens.
        shape_targets = target_seq[:, 900:902].long()
        shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

        # Compute grid loss
        batch_size = input_seq.size(0)
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
                   torch.tensor(0.0, device=input_seq.device, requires_grad=True)

        reconstruction_loss = shape_loss + grid_loss
        losses.append(reconstruction_loss.item())

        # Track how much z has changed
        z_delta = torch.norm(z - initial_z).item()
        z_changes.append(z_delta)

        # Log every few steps
        if step % 2 == 0:
            print(f"Step {step}: Loss = {reconstruction_loss.item():.4f}, "
                  f"Z change magnitude = {z_delta:.4f}")

        reconstruction_loss.backward()
        optimizer_z.step()

    # Final change in Z
    final_z_change = torch.norm(z - initial_z).item()
    print(f"\nZ optimization complete:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")
    print(f"Total Z change magnitude: {final_z_change:.4f}")

    return z.mean(dim=0, keepdim=True), losses

def evolutionary_optimize_latent_z(lpn, input_seq, target_seq, population_size=None,
                                   num_generations=None, mutation_std=None):
    """
    Optimize latent z using an evolutionary algorithm.
    """
    # Use settings if parameters are not provided
    evolutionary_settings = latent_optimization['evolutionary']
    if population_size is None:
        population_size = evolutionary_settings['population_size']
    if num_generations is None:
        num_generations = evolutionary_settings['num_generations']
    if mutation_std is None:
        mutation_std = evolutionary_settings['mutation_std']
        
    device = input_seq.device
    with torch.no_grad():
        mu, log_var = lpn.encoder(input_seq, target_seq)
        initial_z = lpn.reparameterize(mu, log_var).detach()
    population = initial_z.unsqueeze(0).repeat(population_size, 1, 1)
    population = population + torch.randn_like(population) * mutation_std
    best_candidate = None
    for gen in range(num_generations):
        candidate_losses = []
        for i in range(population_size):
            candidate_z = population[i]
            shape_logits, grid_logits = lpn.decoder(candidate_z, input_seq, target_seq=target_seq)
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='mean')
            batch_size = input_seq.size(0)
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
        sorted_indices = sorted(range(population_size), key=lambda i: candidate_losses[i])
        best_loss = candidate_losses[sorted_indices[0]]
        best_candidate = population[sorted_indices[0]].detach()
        print(f"Evolutionary Generation {gen}: Best Loss = {best_loss:.4f}")
        num_selected = population_size // 2
        selected_candidates = population[sorted_indices[:num_selected]]
        offspring = selected_candidates + torch.randn_like(selected_candidates) * mutation_std
        population = torch.cat([selected_candidates, offspring], dim=0)
        if population.size(0) > population_size:
            population = population[:population_size]
        elif population.size(0) < population_size:
            extra = best_candidate.unsqueeze(0).repeat(population_size - population.size(0), 1, 1)
            population = torch.cat([population, extra], dim=0)
    print("Evolutionary optimization complete.")
    return best_candidate, None

##############################
# New: Optimize latent z via a Voronoi‑Inspired Search
##############################
def voronoi_optimize_latent_z(lpn, input_seq, target_seq, population_size=None,
                              num_generations=None, diversity_weight=None,
                              mutation_std=None):
    """
    Optimize latent z using a Voronoi-inspired approach.
    In each generation, we evaluate each candidate's reconstruction loss and also compute a
    diversity score (approximated by the candidate's distance to its nearest neighbor in the population).
    We then select candidates that balance a low loss with high diversity.
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
        
    device = input_seq.device
    with torch.no_grad():
        mu, log_var = lpn.encoder(input_seq, target_seq)
        initial_z = lpn.reparameterize(mu, log_var).detach()
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
    for gen in range(num_generations):
        candidate_losses = []
        for i in range(population_size):
            candidate_z = population[i]
            shape_logits, grid_logits = lpn.decoder(candidate_z, input_seq, target_seq=target_seq)
            shape_targets = target_seq[:, 900:902].long()
            shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1), reduction='mean')
            batch_size = input_seq.size(0)
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
        diversity_scores = compute_diversity(population)
        # Combine reconstruction loss and diversity. Lower is better.
        combined_scores = [loss - diversity_weight * div for loss, div in zip(candidate_losses, diversity_scores)]
        sorted_indices = sorted(range(population_size), key=lambda i: combined_scores[i])
        best_score = combined_scores[sorted_indices[0]]
        best_candidate = population[sorted_indices[0]].detach()
        print(f"Voronoi Generation {gen}: Best Combined Score = {best_score:.4f}")
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
    print("Voronoi-inspired optimization complete.")
    return best_candidate, None

##############################
# Helper: Choose Optimization Method
##############################
def get_optimized_z(lpn, input_seq, target_seq):
    """
    Returns an optimized latent z using either gradient-based, evolutionary, or Voronoi-inspired search,
    depending on the optimization method setting.
    """
    optimization_method = latent_optimization.get('method', 'gradient')
    
    if latent_optimization['training']['enabled'] or latent_optimization['inference']['enabled']:
        if optimization_method == "gradient":
            with torch.enable_grad():
                return optimize_latent_z(lpn, input_seq, target_seq)
        elif optimization_method == "evolutionary":
            return evolutionary_optimize_latent_z(lpn, input_seq, target_seq)
        elif optimization_method == "voronoi":
            return voronoi_optimize_latent_z(lpn, input_seq, target_seq)
        else:
            mu, log_var = lpn.encoder(input_seq, target_seq)
            return lpn.reparameterize(mu, log_var), None
    else:
        mu, log_var = lpn.encoder(input_seq, target_seq)
        return lpn.reparameterize(mu, log_var), None