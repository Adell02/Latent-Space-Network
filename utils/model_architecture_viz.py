#!/usr/bin/env python3
"""
Model Architecture Visualization Utility

Generates torchviz diagrams for the model architecture and uploads to wandb.
"""

import torch
import tempfile
import os

def generate_architecture_visualizations(model, wandb_logger=None, device='cuda', global_step=None):
    """
    Generate torchviz architecture visualization for the model.
    
    Args:
        model: LatentProgramNetwork instance (single or multi-encoder)
        wandb_logger: WandB logger instance (optional)
        device: Device to run the model on
        global_step: Current training step for wandb logging
    """
    try:
        from torchviz import make_dot
        print("✓ torchviz available - generating architecture visualization...")
    except ImportError:
        print("⚠ torchviz not available - skipping architecture visualization")
        print("  Install with: pip install torchviz")
        return
    
    model.eval()
    
    try:
        # Enable gradient computation for better graph tracing
        model.train()  # Enable gradients
        
        # Create sample inputs in proper ARC format
        batch_size = 1
        seq_len = 902  # Standard sequence length
        
        # Generate proper ARC-formatted sequences directly (no in-place operations)
        # Grid tokens (0-899): discrete values 0-9 for pixel colors
        # Shape tokens (900-901): discrete values 0-30 for dimensions
        
        # Create grid data
        grid_data = torch.randint(0, 10, (batch_size, 900), device=device, dtype=torch.float32)
        shape_data = torch.tensor([[5, 5]], device=device, dtype=torch.float32)
        
        # Concatenate to form complete sequences
        input_seq = torch.cat([grid_data, shape_data], dim=1)
        target_seq = torch.cat([grid_data, shape_data], dim=1)
        
        # Enable gradients for proper tracing
        input_seq.requires_grad_(True)
        target_seq.requires_grad_(True)
        
        # Forward pass through the model with gradient computation enabled
        if hasattr(model, 'multi_encoder') and model.num_encoders > 1:
            # Multi-encoder model - trace through more components
            print(f"  Generating multi-encoder architecture diagram ({model.num_encoders} encoders)...")
            
            # Get individual encoder outputs for better tracing
            individual_mus = []
            individual_logvars = []
            
            for i in range(model.num_encoders):
                mu_i, logvar_i = model.multi_encoder.encoders[i](input_seq, target_seq)
                individual_mus.append(mu_i)
                individual_logvars.append(logvar_i)
            
            # PoE fusion
            mu_stack = torch.stack(individual_mus)
            logvar_stack = torch.stack(individual_logvars)
            
            # Get final model output
            (shape_logits, grid_logits), mu, logvar = model(input_seq, target_seq)
            
            # Create comprehensive output that traces through all components
            output = torch.cat([
                shape_logits.flatten(),
                grid_logits.flatten(),
                mu.flatten(),
                logvar.flatten(),
                torch.stack(individual_mus).flatten(),
                torch.stack(individual_logvars).flatten()
            ])
        else:
            # Single encoder model
            print("  Generating single encoder architecture diagram...")
            
            # Get intermediate encoder outputs for better tracing
            if hasattr(model, 'multi_encoder'):
                mu_enc, logvar_enc = model.multi_encoder.encoders[0](input_seq, target_seq)
            else:
                mu_enc, logvar_enc = model.encoder(input_seq, target_seq)
            
            # Get final model output
            (shape_logits, grid_logits), mu, logvar = model(input_seq, target_seq)
            
            # Create comprehensive output that traces through encoder and decoder
            output = torch.cat([
                shape_logits.flatten(),
                grid_logits.flatten(),
                mu.flatten(),
                logvar.flatten(),
                mu_enc.flatten(),
                logvar_enc.flatten()
            ])
        
        # Generate more detailed visualization
        dot = make_dot(output, params=dict(model.named_parameters()),
                      show_attrs=True, show_saved=True)
        
        # Improve visualization layout and readability
        dot.attr(rankdir='TB', size='20,24')  # Even larger size for complex architecture
        dot.attr('node', fontsize='7', style='filled', fillcolor='lightblue')
        dot.attr('edge', fontsize='5')
        
        # Add graph title with model info
        num_encoders = getattr(model, 'num_encoders', 1)
        latent_dim = getattr(model, 'latent_dim', 'unknown')
        is_vq_vae = hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae()
        vq_status = " (VQ-VAE)" if is_vq_vae else " (VAE)"
        dot.attr(label=f'Latent Program Network{vq_status}\\n{num_encoders} Encoder(s), Latent Dim: {latent_dim}')
        dot.attr(fontsize='12')
        
        # Save diagram
        viz_path = None
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            dot.format = 'png'
            dot.render(tmp_file.name.replace('.png', ''), cleanup=True)
            viz_path = tmp_file.name.replace('.png', '') + '.png'
        
        print(f"    ✓ Architecture visualization saved to {viz_path}")
        
        # Upload to wandb
        if wandb_logger and viz_path and os.path.exists(viz_path):
            try:
                import wandb
                
                log_dict = {
                    'architecture/model_diagram': wandb.Image(viz_path),
                    'architecture/num_encoders': getattr(model, 'num_encoders', 1),
                    'architecture/latent_dim': getattr(model, 'latent_dim', 'unknown'),
                    'architecture/visualization_generated': True
                }
                
                wandb_logger._safe_log(log_dict, step_hint=global_step or 0)
                print("    ✓ Architecture diagram uploaded to wandb")
                
            except Exception as e:
                print(f"    ⚠ Failed to upload to wandb: {e}")
        
        # Cleanup
        if viz_path and os.path.exists(viz_path):
            try:
                os.unlink(viz_path)
            except Exception as e:
                print(f"    ⚠ Cleanup warning: {e}")
        
        model.train()
        print("✓ Architecture visualization complete")
        
    except Exception as e:
        print(f"⚠ Architecture visualization failed: {e}")
        model.train()


def log_model_summary(model, wandb_logger=None, global_step=None):
    """
    Log a comprehensive model summary to wandb.
    
    Args:
        model: LatentProgramNetwork instance
        wandb_logger: WandB logger instance (optional)  
        global_step: Current training step for wandb logging
    """
    if not wandb_logger:
        return
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'model_summary/total_parameters': total_params,
            'model_summary/trainable_parameters': trainable_params,
            'model_summary/num_encoders': getattr(model, 'num_encoders', 1),
            'model_summary/latent_dimension': getattr(model, 'latent_dim', 'unknown'),
            'model_summary/parameter_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        wandb_logger._safe_log(summary, step_hint=global_step or 0)
        print(f"✓ Model summary logged: {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    except Exception as e:
        print(f"⚠ Failed to log model summary: {e}")