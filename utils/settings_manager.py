import json
import os
from typing import Dict, Any

class SettingsManager:
    
    def __init__(self, settings_file: str = "model_settings.json"):
        self.settings_file = settings_file
        self._settings = None

    def load_settings(self) -> None:
        """Load settings from JSON file."""
        if not os.path.exists(self.settings_file):
            raise FileNotFoundError(f"Settings file not found: {self.settings_file}")
        
        with open(self.settings_file, 'r') as f:
            self._settings = json.load(f)

    def get_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings

    def get_data_settings(self) -> Dict[str, Any]:
        """Get data-related settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['data_settings']
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Get model architecture settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['model_architecture']

    def get_training_settings(self) -> Dict[str, Any]:
        """Get training settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['training_settings']
        
    def get_training_data_settings(self) -> Dict[str, Any]:
        """Get training data source settings."""
        if self._settings is None:
            self.load_settings()
        training_settings = self._settings['training_settings']
        return {
            'data_source': training_settings.get('data_source', 'synthetic'),
            'use_ood_for_training': training_settings.get('use_ood_for_training', False)
        }

    def get_evaluation_data_settings(self) -> Dict[str, Any]:
        """Get evaluation data source settings."""
        if self._settings is None:
            self.load_settings()
        eval_settings = self._settings['evaluation_settings']
        return {
            'data_source': eval_settings.get('data_source', 'ood_original_arc'),
            'use_ood_for_evaluation': eval_settings.get('use_ood_for_evaluation', True)
        }

    def get_repulsion_loss_settings(self) -> dict:
        """Get repulsion loss settings."""
        return self._settings['repulsion_loss_settings']

    def get_solo_loss_settings(self) -> Dict[str, Any]:
        """Get solo loss settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['training_settings'].get('solo_loss', {
            'enabled': False,
            'lambda_solo': 0.1,
            'isolate_decoder_gradients': True,
            'log_frequency': 100
        })

    def get_wandb_settings(self) -> Dict[str, Any]:
        """Get wandb settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings.get('wandb_settings', {
            'enabled': False,
            'entity': None,
            'api_key': None,
            'log_interval': 1,
            'log_visualizations': True,
            'log_gradients': False,
            'log_trajectory_plots': True,
            'trajectory_max_samples': 3,
            'eval_log_interval': 10
        })

    def get_latent_optimization(self) -> Dict[str, Any]:
        """Get latent optimization settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['latent_optimization']

    def get_evaluation_settings(self) -> Dict[str, Any]:
        """Get evaluation settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['evaluation_settings']

    def get_specialist_training_settings(self) -> Dict[str, Any]:
        """Get specialist training settings."""
        if self._settings is None:
            self.load_settings()
        return self._settings['specialist_training']

    def get_enhanced_training(self) -> Dict[str, Any]:
        if self._settings is None:
            self.load_settings()
        return self._settings.get('enhanced_training', {})

    def get_project_name(self) -> str:
        """Get the WANDB project name from settings, or a default if not present."""
        if self._settings is None:
            self.load_settings()
        # Try wandb_settings, then top-level, then fallback
        if 'wandb_settings' in self._settings and 'project_name' in self._settings['wandb_settings']:
            return self._settings['wandb_settings']['project_name']
        if 'project_name' in self._settings:
            return self._settings['project_name']
        return 'latent-space-network'

    def save_settings(self, run_dir: str) -> None:
        """Save current settings to a run directory."""
        if self._settings is None:
            self.load_settings()
        settings_file = os.path.join(run_dir, self.settings_file.split("/")[-1])
        with open(settings_file, 'w') as f:
            json.dump(self._settings, f, indent=4)
        print(f"Settings saved to {settings_file}")

    def set_settings(self, new_settings: Dict[str, Any]) -> None:
        """Directly set the settings in memory (for sweeps)."""
        self._settings = new_settings

# Global settings instance - initialized with default
settings = SettingsManager()

def init_settings(settings_file: str = "model_settings.json") -> SettingsManager:
    """Initialize global settings with a specific file path."""
    global settings
    settings = SettingsManager(settings_file)
    return settings