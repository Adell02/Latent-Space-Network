import json
import os
from typing import Dict, Any

class SettingsManager:
    
    def __init__(self,settings_file: str = "model_settings.json"):
        self.settings_file = settings_file
        self._settings = None
        self.load_settings()

    def load_settings(self) -> None:
        """Load settings from JSON file."""
        if not os.path.exists(self.settings_file):
            raise FileNotFoundError(f"Settings file not found: {self.settings_file}")
        
        with open(self.settings_file, 'r') as f:
            self._settings = json.load(f)

    def get_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings

    def get_data_settings(self) -> Dict[str, Any]:
        """Get data-related settings."""
        return self._settings['data_settings']
    def get_model_architecture(self) -> Dict[str, Any]:
        """Get model architecture settings."""
        return self._settings['model_architecture']

    def get_training_settings(self) -> Dict[str, Any]:
        """Get training settings."""
        return self._settings['training_settings']

    def get_solo_loss_settings(self) -> Dict[str, Any]:
        """Get solo loss settings."""
        return self._settings['training_settings'].get('solo_loss', {
            'enabled': False,
            'lambda_solo': 0.1,
            'isolate_decoder_gradients': True,
            'log_frequency': 100
        })

    def get_wandb_settings(self) -> Dict[str, Any]:
        """Get wandb settings."""
        return self._settings['training_settings'].get('wandb', {
            'enabled': False,
            'entity': None,
            'api_key': None,
            'log_interval': 1,
            'log_visualizations': True,
            'log_gradients': False
        })

    def get_latent_optimization(self) -> Dict[str, Any]:
        """Get latent optimization settings."""
        return self._settings['latent_optimization']

    def get_evaluation_settings(self) -> Dict[str, Any]:
        """Get evaluation settings."""
        return self._settings['evaluation_settings']

    def save_settings(self, run_dir: str) -> None:
        """Save current settings to a run directory."""
        settings_file = os.path.join(run_dir, self.settings_file.split("/")[-1])
        with open(settings_file, 'w') as f:
            json.dump(self._settings, f, indent=4)
        print(f"Settings saved to {settings_file}")

# Create a global settings manager instance
settings = SettingsManager(settings_file="model_settings.json") 
#settings = SettingsManager(settings_file="LPN_reproduction/pattern_task_settings.json")