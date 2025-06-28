#!/usr/bin/env python3
"""
Lean and efficient Weights & Biases (wandb) integration for latent space network.
Reuses existing visualization functions to avoid code duplication.
"""

import wandb
import torch
import os
from typing import Dict, Any, List, Optional

class WandbLogger:
    """Minimal wandb logger that reuses existing visualization functions."""
    
    def __init__(self, project_name: str, entity: str = None, api_key: str = None, 
                 log_interval: int = 1, config: Dict[str, Any] = None):
        self.project_name = project_name
        self.entity = entity
        self.api_key = api_key
        self.log_interval = log_interval
        self.config = config or {}
        self.run = None
        self.is_initialized = False
        
    def init(self, run_name: str = None, tags: List[str] = None):
        """Initialize wandb run."""
        if self.is_initialized:
            return
            
        # Set API key if provided
        if self.api_key:
            os.environ["WANDB_API_KEY"] = self.api_key
        
        try:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                tags=tags or ['latent-space-network'],
                config=self.config,
                reinit=True
            )
            self.is_initialized = True
            print(f"✓ Wandb initialized: {self.run.name}")
            return True
        except Exception as e:
            print(f"⚠ Wandb initialization failed: {e}")
            self.is_initialized = False
            return False
        
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics."""
        if not self.is_initialized:
            return
        wandb.log({**metrics, 'epoch': epoch}, step=epoch)
        
    def log_accuracy_metrics(self, epoch: int, accuracy_data: Dict[str, Any]):
        """Log accuracy metrics."""
        if not self.is_initialized:
            return
            
        if 'individual_encoders' in accuracy_data:
            # Multi-encoder case
            for encoder_idx, metrics in accuracy_data['individual_encoders'].items():
                log_data = {f'encoder_{encoder_idx}_{k}': v for k, v in metrics.items() if k != 'evaluation_name'}
                wandb.log({**log_data, 'epoch': epoch}, step=epoch)
        else:
            # Single encoder case
            log_data = {k: v for k, v in accuracy_data.items() if k not in ['epoch', 'evaluation_name']}
            wandb.log({**log_data, 'epoch': epoch}, step=epoch)
    
    def log_visualizations(self, run_dir: str, epoch: int, eval_results: Dict[str, Any] = None):
        """Log ONLY latent space visualization during training (as requested)."""
        if not self.is_initialized or epoch % self.log_interval != 0:
            return
            
        print(f"Logging latent space visualization for epoch {epoch}...")
        
        try:
            # Import visualization function
            from utils.visualizers import plot_comprehensive_latent_space
            
            # ONLY log latent space visualization during training
            try:
                plot_comprehensive_latent_space(None, eval_results, run_dir)
                latent_plot = os.path.join(run_dir, 'latent_space_visualization.png')
                if os.path.exists(latent_plot):
                    wandb.log({'latent_space_visualization': wandb.Image(latent_plot)}, step=epoch)
                    print("  ✓ Logged latent space visualization")
                else:
                    print("  ⚠ Latent space plot not found")
            except Exception as e:
                print(f"  ⚠ Could not create/log latent space visualization: {e}")
            
            print("  Note: Other plots will be uploaded when running 'visualize' command")
            
        except Exception as e:
            print(f"⚠ Error logging latent space visualization: {e}")
    
    def upload_all_plots(self, run_dir: str, epoch: int = None):
        """Upload all existing plots to wandb when running visualize command."""
        if not self.is_initialized:
            print("⚠ Wandb not initialized, cannot upload plots")
            return
            
        print(f"Uploading all plots from {run_dir} to wandb...")
        
        # Define plot files to look for and their wandb keys
        plot_files = [
            ('latent_space_visualization.png', 'latent_space_visualization'),
            ('epoch_accuracies.png', 'epoch_accuracies'),
            ('z_optimization_losses.png', 'z_optimization_losses'),
            ('multi_encoder_training_accuracies.png', 'multi_encoder_training_accuracies'),
            ('multi_encoder_accuracies.png', 'multi_encoder_accuracies'),
            ('poe_reconstruction_analysis.png', 'poe_reconstruction_analysis'),
        ]
        
        # Also look for trajectory reconstruction plots (pattern-based)
        import glob
        trajectory_plots = glob.glob(os.path.join(run_dir, 'multi_encoder_trajectory_reconstruction_sample_*.png'))
        
        uploaded_count = 0
        step = epoch if epoch else 0
        
        # Upload standard plots
        for filename, wandb_key in plot_files:
            plot_path = os.path.join(run_dir, filename)
            if os.path.exists(plot_path):
                try:
                    wandb.log({wandb_key: wandb.Image(plot_path)}, step=step)
                    print(f"  ✓ Uploaded {filename}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to upload {filename}: {e}")
        
        # Upload trajectory reconstruction plots
        for i, traj_plot in enumerate(trajectory_plots):
            if os.path.exists(traj_plot):
                try:
                    wandb_key = f'trajectory_reconstruction_sample_{i}'
                    wandb.log({wandb_key: wandb.Image(traj_plot)}, step=step)
                    print(f"  ✓ Uploaded {os.path.basename(traj_plot)}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to upload {os.path.basename(traj_plot)}: {e}")
        
        print(f"  📊 Total plots uploaded: {uploaded_count}")
        
        if uploaded_count == 0:
            print("  ⚠ No plots found to upload. Make sure to run visualizations first.")
            
        return uploaded_count
    
    def log_gradient_norms(self, model: torch.nn.Module, epoch: int):
        """Log gradient norms."""
        if not self.is_initialized:
            return
            
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[f'grad_norm/{name}'] = param.grad.norm().item()
        
        if grad_norms:
            wandb.log(grad_norms, step=epoch)
    
    def upload_checkpoint(self, checkpoint_path: str, epoch: int):
        """Upload model checkpoint as WandB artifact."""
        if not self.is_initialized:
            return
            
        try:
            import os
            
            # Create artifact with run name for better organization
            run_name = self.run.name if self.run else "unknown_run"
            artifact_name = f"{run_name}_checkpoint_epoch_{epoch}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model checkpoint at epoch {epoch} for run {run_name}",
                metadata={
                    "epoch": epoch,
                    "framework": "pytorch",
                    "run_name": run_name
                }
            )
            
            # Add the checkpoint file to the artifact
            artifact.add_file(checkpoint_path, name=f"checkpoint_epoch{epoch}.pt")
            
            # Log the artifact
            wandb.log_artifact(artifact)
            
            print(f"✓ Uploaded checkpoint for epoch {epoch} to wandb")
            
        except Exception as e:
            print(f"⚠ Failed to upload checkpoint to wandb: {e}")
    
    def upload_final_model(self, model_path: str, config_path: str = None):
        """Upload final trained model and configuration as WandB artifact."""
        if not self.is_initialized:
            return
            
        try:
            import os
            
            # Create artifact for final model with run name
            run_name = self.run.name if self.run else "unknown_run"
            artifact = wandb.Artifact(
                name=f"{run_name}_final_model",
                type="model",
                description=f"Final trained model for run {run_name}",
                metadata={
                    "framework": "pytorch",
                    "model_type": "latent_program_network",
                    "run_name": run_name
                }
            )
            
            # Add the model file
            if os.path.exists(model_path):
                artifact.add_file(model_path, name="final_model.pt")
                print(f"✓ Added model file: {model_path}")
            
            # Add configuration file if provided
            if config_path and os.path.exists(config_path):
                artifact.add_file(config_path, name="model_config.json")
                print(f"✓ Added config file: {config_path}")
            
            # Log the artifact
            wandb.log_artifact(artifact)
            
            print(f"✓ Uploaded final model to wandb")
            
        except Exception as e:
            print(f"⚠ Failed to upload final model to wandb: {e}")
    
    def finish(self):
        """Finish wandb run."""
        if self.is_initialized and self.run:
            self.run.finish()
            self.is_initialized = False
            print("✓ Wandb run finished")

# Global logger instance
_global_logger = None

def init_wandb_logger(project_name: str, entity: str = None, api_key: str = None,
                     run_name: str = None, tags: List[str] = None,
                     log_interval: int = 1, config: Dict[str, Any] = None) -> Optional[WandbLogger]:
    """Initialize global wandb logger."""
    global _global_logger
    
    _global_logger = WandbLogger(project_name, entity, api_key, log_interval, config)
    if _global_logger.init(run_name, tags):
        return _global_logger
    return None

def get_wandb_logger() -> Optional[WandbLogger]:
    """Get global wandb logger."""
    return _global_logger

def init_wandb_for_mode(mode: str, run_dir: str = None) -> Optional[WandbLogger]:
    """
    Initialize wandb for different modes (train, eval, visualize) using persistent settings.
    
    Args:
        mode: One of 'train', 'eval', 'visualize'
        run_dir: Run directory to extract project name and settings from
        
    Returns:
        WandbLogger instance or None if initialization fails
    """
    global _global_logger
    
    # If already initialized, return existing logger
    if _global_logger and _global_logger.is_initialized:
        print(f"✓ Using existing wandb session for {mode}")
        return _global_logger
    
    # Try to get wandb settings from settings manager
    try:
        from utils.settings_manager import settings
        wandb_settings = settings.get_wandb_settings()
        
        if not wandb_settings.get('enabled', False):
            print(f"⚠ Wandb not enabled in settings for {mode}")
            return None
            
        # Always prioritize environment variable for project name (especially for sweeps)
        project_name = os.environ.get('WANDB_PROJECT_NAME')
        
        # Only fall back to config if environment variable is not set
        if not project_name:
            project_name = wandb_settings.get('project_name')
            
        # Only use run_dir as last resort (and only if not in sweep mode)
        if not project_name and run_dir and not os.environ.get('WANDB_PROJECT_NAME'):
            project_name = os.path.basename(run_dir)
        
        # Final fallback
        if not project_name:
            project_name = f"latent-space-network-{mode}"
        
        # Create run name - use run_dir basename for run name, not project name
        if run_dir:
            run_name = f"{os.path.basename(run_dir)}_{mode}"
        else:
            run_name = f"{project_name}_{mode}"
        
        # Initialize wandb logger
        wandb_logger = init_wandb_logger(
            project_name=project_name,
            entity=wandb_settings.get('entity'),
            api_key=wandb_settings.get('api_key'),
            run_name=run_name,
            tags=[mode, 'latent-space-network'],
            log_interval=wandb_settings.get('log_interval', 1),
            config=settings.get_settings()
        )
        
        if wandb_logger:
            print(f"✓ Wandb initialized for {mode}: project={project_name}, run={run_name}")
        
        return wandb_logger
        
    except Exception as e:
        print(f"⚠ Could not initialize wandb for {mode}: {e}")
        return None 