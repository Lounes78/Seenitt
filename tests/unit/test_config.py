"""Unit tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.utils.config import Config, get_config


class TestConfig:
    """Test configuration management functionality."""
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                'yolo': {'model_path': 'test_model.pt'},
                'quality': {'min_quality_score': 0.5}
            }
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            assert config.get('yolo.model_path') == 'test_model.pt'
            assert config.get('quality.min_quality_score') == 0.5
        finally:
            Path(config_path).unlink()
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            Config('nonexistent_config.yaml')
    
    def test_get_method_with_default(self):
        """Test get method with default values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {'existing_key': 'value'}
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            assert config.get('existing_key') == 'value'
            assert config.get('nonexistent_key', 'default') == 'default'
            assert config.get('nested.nonexistent.key', 42) == 42
        finally:
            Path(config_path).unlink()
    
    def test_specialized_config_getters(self):
        """Test specialized configuration getter methods."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                'yolo': {'model_path': 'test.pt'},
                'filtering': {'min_detections': 3},
                'quality': {'min_score': 0.3},
                'similarity': {'threshold': 0.85},
                'validation': {'api_endpoint': 'test'},
                'pipeline': {'batch_size': 32},
                'paths': {'data_dir': 'data'}
            }
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            assert config.get_yolo_config() == {'model_path': 'test.pt'}
            assert config.get_filtering_config() == {'min_detections': 3}
            assert config.get_quality_config() == {'min_score': 0.3}
            assert config.get_similarity_config() == {'threshold': 0.85}
            assert config.get_validation_config() == {'api_endpoint': 'test'}
            assert config.get_pipeline_config() == {'batch_size': 32}
            assert config.get_paths_config() == {'data_dir': 'data'}
        finally:
            Path(config_path).unlink()


def test_global_config_instance():
    """Test global configuration instance management."""
    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        test_config = {'test_key': 'test_value'}
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Test getting config instance
        config1 = get_config(config_path)
        config2 = get_config()  # Should return same instance
        
        assert config1 is config2
        assert config1.get('test_key') == 'test_value'
    finally:
        Path(config_path).unlink()