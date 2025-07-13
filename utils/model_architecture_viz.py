#!/usr/bin/env python3
"""
Model Architecture Visualization Utility

Generates torchviz diagrams for Phase A and Phase B model configurations
and uploads them to wandb for documentation and debugging purposes.
"""

import torch
import tempfile
import os
from typing import Optional, Dict, Any

def generate_architecture_visualizations(model, wandb_logger=None, device='cuda', global_step=None):
    """
    Generate torchviz architecture visualizations for both Phase A and Phase B configurations.
    
    Args:
        model: LatentProgramNetwork instance
        wandb_logger: WandB logger instance (optional)
        device: Device to run the model on
        global_step: Current training step for wandb logging
    """
    try:
        from torchviz import make_dot
        print("✓ torchviz available - generating architecture visualizations...")
    except ImportError:
        print("⚠ torchviz not available - skipping architecture visualization")
        print("  Install with: pip install torchviz")
        return
    
    model.eval()
    
    # Create sample inputs (small batch for efficiency)
    batch_size = 1
    seq_len = model.multi_encoder.encoders[0].max_length if hasattr(model.multi_encoder.encoders[0], 'max_length') else 902
    
    with torch.no_grad():
        # Generate dummy input/target sequences
        input_seq = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
        target_seq = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
        
        visualizations = {}
        
        # === PHASE A: Individual Encoder + Independent Decoder ===
        try:
            print("  Generating Phase A architecture (Encoder + Independent Decoder)...")
            
            # Use first encoder with its independent decoder
            encoder_idx = 0
            (shape_logits, grid_logits), mu, logvar = model.multi_encoder.forward_single_encoder_with_independent_decoder(
                encoder_idx, input_seq, target_seq, sample_latent=False
            )
            
            # Create computation graph for Phase A
            phase_a_output = torch.cat([shape_logits.flatten(), grid_logits.flatten(), mu.flatten(), logvar.flatten()])
            phase_a_dot = make_dot(phase_a_output, params=dict(model.named_parameters()),
                                  show_attrs=False, show_saved=False)
            phase_a_dot.attr(rankdir='TB', size='12,16')
            phase_a_dot.attr('node', fontsize='10')
            
            # Save Phase A diagram
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                phase_a_dot.format = 'png'
                phase_a_dot.render(tmp_file.name.replace('.png', ''), cleanup=True)
                phase_a_path = tmp_file.name.replace('.png', '') + '.png'
                visualizations['phase_a'] = phase_a_path
            
            print(f"    ✓ Phase A visualization saved to {phase_a_path}")
            
        except Exception as e:
            print(f"    ⚠ Phase A visualization failed: {e}")
        
        # === PHASE B: PoE + Shared Decoder ===
        try:
            print("  Generating Phase B architecture (PoE + Shared Decoder)...")
            
            # Create input views for all encoders (PoE setup)
            input_views = [(input_seq, target_seq) for _ in range(model.num_encoders)]
            (shape_logits_poe, grid_logits_poe), mu_poe, logvar_poe = model.multi_encoder.forward_poe_with_shared_decoder(
                input_views, sample_latent=False
            )
            
            # Create computation graph for Phase B
            phase_b_output = torch.cat([shape_logits_poe.flatten(), grid_logits_poe.flatten(), 
                                       mu_poe.flatten(), logvar_poe.flatten()])
            phase_b_dot = make_dot(phase_b_output, params=dict(model.named_parameters()),
                                  show_attrs=False, show_saved=False)
            phase_b_dot.attr(rankdir='TB', size='12,16')
            phase_b_dot.attr('node', fontsize='10')
            
            # Save Phase B diagram
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                phase_b_dot.format = 'png'
                phase_b_dot.render(tmp_file.name.replace('.png', ''), cleanup=True)
                phase_b_path = tmp_file.name.replace('.png', '') + '.png'
                visualizations['phase_b'] = phase_b_path
            
            print(f"    ✓ Phase B visualization saved to {phase_b_path}")
            
        except Exception as e:
            print(f"    ⚠ Phase B visualization failed: {e}")
        
        # === UPLOAD TO WANDB ===
        if wandb_logger and visualizations:
            try:
                import wandb
                
                log_dict = {}
                
                if 'phase_a' in visualizations and os.path.exists(visualizations['phase_a']):
                    log_dict['architecture/phase_a_encoder_independent_decoder'] = wandb.Image(visualizations['phase_a'])
                    print("    ✓ Phase A architecture uploaded to wandb")
                
                if 'phase_b' in visualizations and os.path.exists(visualizations['phase_b']):
                    log_dict['architecture/phase_b_poe_shared_decoder'] = wandb.Image(visualizations['phase_b'])
                    print("    ✓ Phase B architecture uploaded to wandb")
                
                # Add model metadata
                log_dict.update({
                    'architecture/num_encoders': model.num_encoders,
                    'architecture/latent_dim': model.latent_dim,
                    'architecture/encoder_layers': getattr(model.multi_encoder.encoders[0], 'num_layers', 'unknown'),
                    'architecture/decoder_layers': getattr(model.multi_encoder.shared_decoder, 'num_layers', 'unknown'),
                    'architecture/visualization_generated': True
                })
                
                wandb_logger._safe_log(log_dict, step_hint=global_step or 0)
                print("    ✓ Architecture metadata logged to wandb")
                
            except Exception as e:
                print(f"    ⚠ Failed to upload to wandb: {e}")
        
        # === CLEANUP ===
        for viz_path in visualizations.values():
            try:
                if os.path.exists(viz_path):
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
        # Count parameters for each component
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Encoder parameters
        encoder_params = sum(sum(p.numel() for p in encoder.parameters()) 
                           for encoder in model.multi_encoder.encoders)
        
        # Independent decoder parameters
        independent_decoder_params = sum(sum(p.numel() for p in decoder.parameters())
                                       for decoder in model.multi_encoder.independent_decoders)
        
        # Shared decoder parameters
        shared_decoder_params = sum(p.numel() for p in model.multi_encoder.shared_decoder.parameters())
        
        summary = {
            'model_summary/total_parameters': total_params,
            'model_summary/trainable_parameters': trainable_params,
            'model_summary/encoder_parameters': encoder_params,
            'model_summary/independent_decoder_parameters': independent_decoder_params,
            'model_summary/shared_decoder_parameters': shared_decoder_params,
            'model_summary/num_encoders': model.num_encoders,
            'model_summary/latent_dimension': model.latent_dim,
            'model_summary/parameter_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        wandb_logger._safe_log(summary, step_hint=global_step or 0)
        print(f"✓ Model summary logged: {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    except Exception as e:
        print(f"⚠ Failed to log model summary: {e}") 