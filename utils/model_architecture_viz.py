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
        with torch.no_grad():
            # Create sample inputs
            batch_size = 1
            seq_len = 902  # Standard sequence length
            input_seq = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
            target_seq = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
            
            # Forward pass through the model
            if hasattr(model, 'multi_encoder') and model.num_encoders > 1:
                # Multi-encoder model - use PoE inference
                print(f"  Generating multi-encoder architecture diagram ({model.num_encoders} encoders)...")
                (shape_logits, grid_logits), mu, logvar = model(input_seq, target_seq)
            else:
                # Single encoder model
                print("  Generating single encoder architecture diagram...")
                (shape_logits, grid_logits), mu, logvar = model(input_seq, target_seq)
            
            # Create computation graph
            output = torch.cat([shape_logits.flatten(), grid_logits.flatten(), mu.flatten(), logvar.flatten()])
            dot = make_dot(output, params=dict(model.named_parameters()),
                          show_attrs=False, show_saved=False)
            dot.attr(rankdir='TB', size='12,16')
            dot.attr('node', fontsize='10')
            
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