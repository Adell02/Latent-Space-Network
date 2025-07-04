#!/usr/bin/env python3
"""
Lean and efficient Weights & Biases (wandb) integration for latent space network.
Reuses existing visualization functions to avoid code duplication.
"""

import wandb
import torch
import os
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

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
        
        # Make sure run name is safe and unique
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = _slugify(run_name)
        
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
        
    def _safe_log(self, data: Dict[str, Any], step_hint: Any = None):
        """Internal helper that logs without violating wandb step monotonicity."""
        if not self.is_initialized:
            return
        if isinstance(step_hint, int):
            wandb.log(data, step=step_hint)
        else:
            wandb.log(data)  # Let wandb autoincrement

    def log_training_metrics(self, epoch: Any, metrics: Dict[str, float]):
        """Log training metrics (epoch may be int or str)."""
        if not self.is_initialized:
            return
        self._safe_log({**metrics, 'epoch': epoch}, step_hint=epoch)
        
    def log_accuracy_metrics(self, epoch: int, accuracy_data: Dict[str, Any]):
        """Log accuracy metrics."""
        if not self.is_initialized:
            return
            
        if 'individual_encoders' in accuracy_data:
            # Multi-encoder case
            for encoder_idx, metrics in accuracy_data['individual_encoders'].items():
                log_data = {f'encoder_{encoder_idx}_{k}': v for k, v in metrics.items() if k != 'evaluation_name'}
                self._safe_log({**log_data, 'epoch': epoch}, step_hint=epoch)
        else:
            # Single encoder case
            log_data = {k: v for k, v in accuracy_data.items() if k not in ['epoch', 'evaluation_name']}
            self._safe_log({**log_data, 'epoch': epoch}, step_hint=epoch)
    
    def log_visualizations(self, run_dir: str, epoch: int, eval_results: Dict[str, Any] = None, trajectory_plots: Dict[str, str] = None):
        """Log visualizations including trajectory plots to wandb."""
        if not self.is_initialized or epoch % self.log_interval != 0:
            return
            
        print(f"Logging visualizations for epoch {epoch}...")
        
        try:
            # Import visualization function
            from utils.visualizers import plot_comprehensive_latent_space
            
            # Log latent space visualization during training
            try:
                plot_comprehensive_latent_space(None, eval_results, run_dir)
                latent_plot = os.path.join(run_dir, 'latent_space_visualization.png')
                if os.path.exists(latent_plot):
                    # Use consistent step parameter for slider visualization
                    self._safe_log({'latent_space': wandb.Image(latent_plot)}, step_hint=epoch)
                    print("  ✓ Logged latent space visualization")
                else:
                    print("  ⚠ Latent space plot not found")
            except Exception as e:
                print(f"  ⚠ Could not create/log latent space visualization: {e}")
            
            # Log trajectory plots if available
            if trajectory_plots:
                print(f"  Logging {len(trajectory_plots)} trajectory plots...")
                trajectory_logged = 0
                
                # Group all trajectory plots in a single log entry for slider visualization
                trajectory_images = {}
                
                for plot_key, plot_path in trajectory_plots.items():
                    if os.path.exists(plot_path):
                        try:
                            # Use consistent key naming for slider visualization
                            wandb_key = f'trajectory_{plot_key}'
                            trajectory_images[wandb_key] = wandb.Image(plot_path)
                            trajectory_logged += 1
                        except Exception as e:
                            print(f"    ⚠ Failed to prepare {plot_key}: {e}")
                    else:
                        print(f"    ⚠ Trajectory plot not found: {plot_path}")
                
                # Log all trajectory images with the same step for slider visualization
                if trajectory_images:
                    self._safe_log(trajectory_images, step_hint=epoch)
                    print(f"  ✓ Successfully logged {trajectory_logged}/{len(trajectory_plots)} trajectory plots")
                
            else:
                print("  No trajectory plots to log for this epoch")
            
        except Exception as e:
            print(f"⚠ Error logging visualizations: {e}")
    
    def upload_all_plots(self, run_dir: str, epoch: int = None):
        """Upload all existing plots to wandb when running visualize command."""
        if not self.is_initialized:
            print("⚠ Wandb not initialized, cannot upload plots")
            return
            
        print(f"Uploading all plots from {run_dir} to wandb...")
        
        # Define plot files to look for and their wandb keys
        plot_files = [
            ('latent_space_visualization.png', 'latent_space'),
            ('comprehensive_latent_space.png', 'latent_space_comprehensive'),
            ('epoch_accuracies.png', 'epoch_accuracies'),
            ('z_optimization_losses.png', 'z_optimization_losses'),
            ('multi_encoder_training_accuracies.png', 'multi_encoder_training_accuracies'),
            ('multi_encoder_accuracies.png', 'multi_encoder_accuracies'),
            ('poe_reconstruction_analysis.png', 'poe_reconstruction_analysis'),
        ]
        
        # Also look for trajectory reconstruction plots (pattern-based)
        import glob
        trajectory_plots = glob.glob(os.path.join(run_dir, 'multi_encoder_trajectory_reconstruction_sample_*.png'))
        # Look for new trajectory plots from training epochs
        epoch_trajectory_plots = glob.glob(os.path.join(run_dir, 'trajectory_epoch*_*.png'))
        
        uploaded_count = 0
        step = epoch if epoch else 0
        
        # Upload standard plots
        for filename, wandb_key in plot_files:
            plot_path = os.path.join(run_dir, filename)
            if os.path.exists(plot_path):
                try:
                    self._safe_log({wandb_key: wandb.Image(plot_path)}, step_hint=step)
                    print(f"  ✓ Uploaded {filename}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to upload {filename}: {e}")
        
        # Upload trajectory reconstruction plots
        for i, traj_plot in enumerate(trajectory_plots):
            if os.path.exists(traj_plot):
                try:
                    wandb_key = f'trajectory_reconstruction_sample_{i}'
                    self._safe_log({wandb_key: wandb.Image(traj_plot)}, step_hint=step)
                    print(f"  ✓ Uploaded {os.path.basename(traj_plot)}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to upload {os.path.basename(traj_plot)}: {e}")
        
        # Upload epoch trajectory plots with epoch-based steps for slider visualization
        # Group by key_sample pattern for consistent keys across epochs
        epoch_trajectory_dict = {}
        
        for epoch_traj_plot in epoch_trajectory_plots:
            if os.path.exists(epoch_traj_plot):
                try:
                    # Extract epoch and key_sample from filename
                    filename = os.path.basename(epoch_traj_plot)
                    # Format: trajectory_epoch{epoch}_{key}_sample{sample_idx}.png
                    parts = filename.replace('.png', '').split('_')
                    if len(parts) >= 4 and parts[0] == 'trajectory' and parts[1].startswith('epoch'):
                        epoch_num = int(parts[1].replace('epoch', ''))
                        key = parts[2]
                        sample_idx = parts[3].replace('sample', '')
                        
                        # Create consistent key for slider visualization
                        plot_key = f'{key}_sample{sample_idx}'
                        
                        # Group by key_sample
                        if plot_key not in epoch_trajectory_dict:
                            epoch_trajectory_dict[plot_key] = []
                        
                        # Store epoch and path
                        epoch_trajectory_dict[plot_key].append((epoch_num, epoch_traj_plot))
                except Exception as e:
                    print(f"  ⚠ Failed to process {os.path.basename(epoch_traj_plot)}: {e}")
        
        # Upload each trajectory plot with its corresponding epoch as step
        for plot_key, epoch_paths in epoch_trajectory_dict.items():
            for epoch_num, plot_path in epoch_paths:
                try:
                    wandb_key = f'trajectory_{plot_key}'
                    self._safe_log({wandb_key: wandb.Image(plot_path)}, step_hint=epoch_num)
                    print(f"  ✓ Uploaded {os.path.basename(plot_path)} (epoch {epoch_num})")
                    uploaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to upload {os.path.basename(plot_path)}: {e}")
        
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
            self._safe_log(grad_norms, step_hint=epoch)
    
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
            
            # Log the artifact (no training step associated)
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
            
            # Log the artifact (no training step associated)
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

# --------------------------------------------------
# Helper: sanitize a string so WandB run / artifact names are safe
# --------------------------------------------------
def _slugify(value: str) -> str:
    """Convert arbitrary string to safe slug for WandB names."""
    if value is None:
        return None
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value))[:128]