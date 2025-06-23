"""Multi‑Encoder Latent‑Program Network
------------------------------------------------
*   K weight‑shared Transformer encoders take **disjoint or overlapping views** of an ARC specimen.
*   Each encoder outputs a diagonal‑Gaussian belief (μᵢ , log σ²ᵢ) **in one common latent basis**.
*   A closed‑form **Product‑of‑Experts** fuses these K Gaussians to (μ★ , σ★²).
*   A single Transformer decoder (inherited from the vanilla LPN) consumes z = μ★ (+ εσ★) at
    training time and z = μ★ at inference; inner‑loop latent search operates on that same z.

The module is drop‑in compatible with the existing training / inference scripts.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Optional

# -------------------------------------------------
#  Low‑level helper: diagonal‑Gaussian PoE
# -------------------------------------------------

def gaussian_poe(mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiply K diagonal Gaussians.
    Args
    -----
    mu      : (K, B, D)
    logvar  : (K, B, D)
    Returns
    -------
    fused_mu, fused_logvar : (B, D)
    """
    precision   = torch.exp(-logvar)            # Σ⁻¹
    fused_var   = 1. / precision.sum(0)         # (B,D)
    fused_mu    = fused_var * (precision * mu).sum(0)
    fused_logvar = fused_var.log()
    return fused_mu, fused_logvar

# -------------------------------------------------
#  Building blocks imported from the single‑encoder LPN
# -------------------------------------------------
from .single_encoder_lpn import TransformerEncoder, TransformerDecoder  # noqa

# -------------------------------------------------
#  Multi‑Encoder wrapper
# -------------------------------------------------
class MultiEncoderLPN(nn.Module):
    """K‑encoder → PoE → single decoder."""

    def __init__(
        self,
        num_encoders: int,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        encoder_max_length: int,
        decoder_max_length: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # ---- shared encoder weights ----
        prototype = TransformerEncoder(1, hidden_dim, num_layers, num_heads, dropout, encoder_max_length)
        self.encoders = nn.ModuleList([prototype for _ in range(num_encoders)])  # weight sharing via same ref
        self.decoder  = TransformerDecoder(1, hidden_dim, num_layers, num_heads, dropout)

    # -------------------------------------------------
    #  Re‑parameterisation
    # -------------------------------------------------
    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # -------------------------------------------------
    #  Forward (training / inference)
    # -------------------------------------------------
    def forward(
        self,
        input_views: List[Tuple[torch.Tensor, torch.Tensor]],  # [(x_i, y_i), ...] len = K
        *,
        training: bool = True,
        sample_latent: bool = True,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Args
        -----
        input_views : during **training** K *different* (x,y) pairs from the same ARC task;
                      during **inference** K *identical* copies of the single (x,y₀) pair.
        """
        K = len(self.encoders)
        assert K == len(input_views), "#encoders ≠ #views"
        mu_list, logvar_list = [], []
        for (enc, (x, y)) in zip(self.encoders, input_views):
            μ, logσ2 = enc(x, y)
            mu_list.append(μ)
            logvar_list.append(logσ2)
        mu_stack     = torch.stack(mu_list)        # (K,B,D)
        logvar_stack = torch.stack(logvar_list)
        mu_star, logvar_star = gaussian_poe(mu_stack, logvar_stack)
        z = self._reparam(mu_star, logvar_star, sample_latent and training)
        x0, y0 = input_views[0]                   # decoder always conditions on *one* input grid
        # Always provide target_seq for proper loss computation, even during inference
        shape_logits, grid_logits = self.decoder(z, x0, target_seq=y0)
        return (shape_logits, grid_logits), mu_star, logvar_star

# -------------------------------------------------
#  Convenience loss wrapper (matches existing API)
# -------------------------------------------------

def multinomial_loss(
    logits: Tuple[torch.Tensor, torch.Tensor],  # (shape_logits, grid_logits)
    target_seq: torch.Tensor,
    *,
    beta: float,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    shape_logits, grid_logits = logits
    shape_targets = target_seq[:, 900:902].long()
    shape_loss = F.cross_entropy(shape_logits.reshape(-1, 31), shape_targets.reshape(-1))

    batch_size = target_seq.size(0)
    grid_loss_sum, active = 0.0, 0
    for i in range(batch_size):
        r, c = map(int, target_seq[i, 900:902])
        n_pix = r * c
        if n_pix:
            grid_loss_sum += F.cross_entropy(grid_logits[i, :n_pix], target_seq[i, :n_pix].long(), reduction='sum')
            active += n_pix
    grid_loss = grid_loss_sum / active if active else torch.tensor(0.0, device=target_seq.device)
    recon = shape_loss + grid_loss
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / mu.size(0)
    return recon + beta * kl
