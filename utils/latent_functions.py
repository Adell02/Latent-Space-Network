import torch
import torch.nn as nn
import torch.nn.functional as F

def optimize_latent_z(lpn, input_seq, target_seq, num_steps=None, lr=None):
    """
    Optimize latent z via gradient descent with logging to verify optimization.
    """
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

    return z.detach()