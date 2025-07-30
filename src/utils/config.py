"""Configuration management utilities."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the plant recognition pipeline."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config.yaml in the config directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'yolo.model_path')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """Get YOLO-specific configuration."""
        return self._config.get('yolo', {})
    
    def get_filtering_config(self) -> Dict[str, Any]:
        """Get filtering configuration."""
        return self._config.get('filtering', {})
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality assessment configuration."""
        return self._config.get('quality', {})
    
    def get_similarity_config(self) -> Dict[str, Any]:
        """Get similarity detection configuration."""
        return self._config.get('similarity', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get final validation configuration."""
        return self._config.get('validation', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self._config.get('pipeline', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get('paths', {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


# Global configuration instance
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance