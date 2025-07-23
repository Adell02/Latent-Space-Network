#!/usr/bin/env python3
"""
Model Architecture Visualization Utility

Generates torchviz diagrams for the model architecture and uploads to wandb.
"""

import os
import tempfile
import torch
from pathlib import Path


def create_simple_architecture_diagram(model, device='cpu'):
    """Create a simple, clean architecture diagram showing model structure and parameters."""
    
    # Get model settings
    from utils.settings_manager import settings
    model_arch = settings.get_model_architecture()
    
    # Determine model type
    is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
    is_actually_single_encoder = hasattr(model, 'is_actually_single_encoder') and model.is_actually_single_encoder
    is_vq_vae = hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae()
    num_encoders = getattr(model, 'num_encoders', 1)
    
    # Use the correct logic for display
    display_as_multi_encoder = is_multi_encoder and not is_actually_single_encoder
    
    # Get architecture parameters
    latent_dim = model_arch.get('latent_dim', 128)
    encoder_hidden_dim = model_arch.get('encoder_hidden_dim', 96)
    decoder_hidden_dim = model_arch.get('decoder_hidden_dim', 96)
    encoder_layers = model_arch.get('encoder_layers', 2)
    decoder_layers = model_arch.get('decoder_layers', 2)
    encoder_heads = model_arch.get('encoder_heads', 6)
    decoder_heads = model_arch.get('decoder_heads', 6)
    dropout = model_arch.get('dropout', 1e-6)
    
    # Create Mermaid diagram
    mermaid_code = f"""
graph TD
    %% Input Layer
    Input["Input Sequence<br/>Shape: [B, 902]<br/>Grid: [0-9], Shape: [0-30]"] 
    
    %% Encoder Section
    """
    
    if display_as_multi_encoder:
        mermaid_code += f"""
    subgraph "Multi-Encoder ({num_encoders} Encoders)"
        """
        for i in range(num_encoders):
            mermaid_code += f"""
        Enc{i}["Encoder {i}<br/>Layers: {encoder_layers}<br/>Hidden: {encoder_hidden_dim}<br/>Heads: {encoder_heads}<br/>Dropout: {dropout:.1e}"]
        """
        mermaid_code += """
    end
    
    %% PoE Fusion
    PoE["Product of Experts<br/>Gaussian Fusion<br/>mu*, sigma*"]
    """
        
        # Connect encoders to PoE
        for i in range(num_encoders):
            mermaid_code += f"""
    Enc{i} --> PoE
    """
    else:
        mermaid_code += f"""
    subgraph "Single Encoder"
        Enc0["Transformer Encoder<br/>Layers: {encoder_layers}<br/>Hidden: {encoder_hidden_dim}<br/>Heads: {encoder_heads}<br/>Dropout: {dropout:.1e}"]
    end
    """
    
    # Latent Space
    if is_vq_vae:
        # Get VQ-VAE settings
        vq_settings = model_arch.get('vq_vae', {})
        num_embeddings = vq_settings.get('num_embeddings', 512)
        commitment_cost = vq_settings.get('commitment_cost', 0.25)
        
        mermaid_code += f"""
    
    %% VQ-VAE Latent Space
    subgraph "VQ-VAE Latent Space"
        VQLatent["Discrete Latent<br/>Codebook: {num_embeddings} embeddings<br/>Dim: {latent_dim}<br/>Commitment: {commitment_cost}"]
    end
    """
    else:
        mermaid_code += f"""
    
    %% VAE Latent Space
    subgraph "VAE Latent Space"
        VAELatent["Continuous Latent<br/>Dim: {latent_dim}<br/>mu, sigma^2 (Gaussian)<br/>KL Regularization"]
    end
    """
    
    # Decoder Section
    mermaid_code += f"""
    
    %% Decoder Section
    subgraph "Transformer Decoder"
        Dec["Decoder<br/>Layers: {decoder_layers}<br/>Hidden: {decoder_hidden_dim}<br/>Heads: {decoder_heads}<br/>Dropout: {dropout:.1e}"]
    end
    
    %% Output Layer
    subgraph "Output Predictions"
        ShapeOut["Shape Output<br/>Linear(hidden -> 31)<br/>Softmax over [0-30]"]
        GridOut["Grid Output<br/>Linear(hidden -> 10)<br/>Softmax over [0-9]"]
    end
    
    %% Final Output
    Output["Final Output<br/>Shape: [B, 2] + [B, 900]<br/>Cross-entropy Loss"]
    
    %% Connections
    Input --> Enc0
    """
    
    # Connect encoder(s) to latent space
    if display_as_multi_encoder:
        # Only add connections for encoders that actually exist
        for i in range(1, num_encoders):
            mermaid_code += f"""
    Input --> Enc{i}
    """
        
        if is_vq_vae:
            mermaid_code += """
    PoE --> VQLatent
    VQLatent --> Dec
    """
        else:
            mermaid_code += """
    PoE --> VAELatent
    VAELatent --> Dec
    """
    else:
        if is_vq_vae:
            mermaid_code += """
    Enc0 --> VQLatent
    VQLatent --> Dec
    """
        else:
            mermaid_code += """
    Enc0 --> VAELatent
    VAELatent --> Dec
    """
    
    # Final connections
    mermaid_code += """
    Dec --> ShapeOut
    Dec --> GridOut
    ShapeOut --> Output
    GridOut --> Output
    
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef encoderStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef latentStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef decoderStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef outputStyle fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    
    class Input inputStyle
    """
    
    # Apply styles
    if display_as_multi_encoder:
        for i in range(num_encoders):
            mermaid_code += f"""
    class Enc{i} encoderStyle
    """
        mermaid_code += """
    class PoE encoderStyle
    """
    else:
        mermaid_code += """
    class Enc0 encoderStyle
    """
    
    if is_vq_vae:
        mermaid_code += """
    class VQLatent latentStyle
    """
    else:
        mermaid_code += """
    class VAELatent latentStyle
    """
    
    mermaid_code += """
    class Dec decoderStyle
    class ShapeOut outputStyle
    class GridOut outputStyle
    class Output outputStyle
    """
    
    return mermaid_code.strip()


def generate_architecture_visualizations(model, wandb_logger=None, device='cpu', global_step=0):
    """Generate clean model architecture visualizations."""
    print(f"🏗️ Generating clean architecture visualization...")
    
    try:
        # Create simple architecture diagram
        mermaid_code = create_simple_architecture_diagram(model, device)
        
        # Try to render diagram if create_diagram is available
        try:
            # This would use the create_diagram tool if available
            print("Generated Mermaid diagram:")
            print(mermaid_code)
        except Exception as e:
            print(f"Diagram rendering info: {e}")
        
        if wandb_logger:
            # Log the diagram to wandb as HTML with Mermaid rendering
            import wandb
            mermaid_html = f"""
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{startOnLoad: true}});
            </script>
            """
            
            wandb_logger._safe_log({
                'model_architecture/diagram': wandb.Html(mermaid_html),
                'model_architecture/diagram_code': mermaid_code,
                'model_architecture/step': global_step
            }, step_hint=global_step)
            
            print(f"✓ Architecture diagram uploaded to WandB")
        
        print(f"✓ Clean architecture visualization generated successfully")
        
    except Exception as e:
        print(f"⚠ Architecture visualization failed: {e}")
        import traceback
        traceback.print_exc()


def log_model_summary(model, wandb_logger=None, global_step=0):
    """Log model parameter summary."""
    if not wandb_logger:
        return
    
    try:
        from utils.settings_manager import settings
        model_arch = settings.get_model_architecture()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model info
        is_multi_encoder = hasattr(model, 'is_multi_encoder') and model.is_multi_encoder
        is_actually_single_encoder = hasattr(model, 'is_actually_single_encoder') and model.is_actually_single_encoder
        is_vq_vae = hasattr(model, 'is_using_vq_vae') and model.is_using_vq_vae()
        num_encoders = getattr(model, 'num_encoders', 1)
        
        # Create summary
        summary = {
            'model_type': 'Multi-Encoder' if (is_multi_encoder and not is_actually_single_encoder) else 'Single-Encoder',
            'latent_type': 'VQ-VAE' if is_vq_vae else 'VAE',
            'num_encoders': num_encoders,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_dim': model_arch.get('latent_dim', 128),
            'encoder_layers': model_arch.get('encoder_layers', 2),
            'decoder_layers': model_arch.get('decoder_layers', 2),
            'encoder_hidden_dim': model_arch.get('encoder_hidden_dim', 96),
            'decoder_hidden_dim': model_arch.get('decoder_hidden_dim', 96),
        }
        
        # Log to wandb
        wandb_logger._safe_log({
            'model_summary': summary,
            'model_summary/step': global_step
        }, step_hint=global_step)
        
        print(f"✓ Model summary logged: {total_params:,} parameters, {summary['model_type']} + {summary['latent_type']}")
        
    except Exception as e:
        print(f"⚠ Model summary logging failed: {e}")


# Remove the problematic import section
def render_mermaid_diagram(mermaid_code):
    """Simple function to output Mermaid diagram code."""
    print("=== MERMAID DIAGRAM ===")
    print(mermaid_code)
    print("=== END DIAGRAM ===")
    return mermaid_code