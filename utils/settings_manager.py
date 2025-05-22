import json
import os
from typing import Dict, Any

class SettingsManager:
    _instance = None
    _settings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._settings is None:
            self.load_settings()

    def load_settings(self, settings_file: str = "model_settings.json") -> None:
        """Load settings from JSON file."""
        if not os.path.exists(settings_file):
            raise FileNotFoundError(f"Settings file not found: {settings_file}")
        
        with open(settings_file, 'r') as f:
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

    def get_latent_optimization(self) -> Dict[str, Any]:
        """Get latent optimization settings."""
        return self._settings['latent_optimization']

    def get_evaluation_settings(self) -> Dict[str, Any]:
        """Get evaluation settings."""
        return self._settings['evaluation_settings']

    def save_settings(self, run_dir: str) -> None:
        """Save current settings to a run directory."""
        settings_file = os.path.join(run_dir, 'model_settings.json')
        with open(settings_file, 'w') as f:
            json.dump(self._settings, f, indent=4)
        print(f"Settings saved to {settings_file}")

# Create a global settings manager instance
settings = SettingsManager() 