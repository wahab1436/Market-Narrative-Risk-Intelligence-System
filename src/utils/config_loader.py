"""
Simplified configuration loader without circular imports.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Simple configuration loader without dependencies on other modules.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        print(f"ConfigLoader initialized with config_dir: {config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_name: Name of config file (without extension)
        
        Returns:
            Dictionary containing configuration
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config or {}
        return self.configs[config_name]
    
    def get_config(self, config_name: str = "config") -> Dict[str, Any]:
        """
        Get configuration by name, load if not already loaded.
        
        Args:
            config_name: Name of config file
        
        Returns:
            Configuration dictionary
        """
        if config_name not in self.configs:
            return self.load_config(config_name)
        return self.configs[config_name]
    
    def get_scraping_config(self) -> Dict[str, Any]:
        """Get scraping configuration."""
        config = self.get_config("config")
        return config.get("scraping", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        config = self.get_config("config")
        return config.get("processing", {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get_config("feature_config")
    
    def get_model_config(self, model_type: str = None) -> Dict[str, Any]:
        """Get model configuration."""
        config = self.get_config("model_config")
        
        if model_type:
            # Try to find model config in nested structure
            for section_name, section in config.items():
                if isinstance(section, dict) and model_type in section:
                    return section[model_type]
            return {}
        
        return config
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        config = self.get_config("config")
        return config.get("dashboard", {})
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'scraping.max_retries')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        # Determine which config file contains the key
        if key_path.startswith('features.'):
            config_name = 'feature_config'
            key_path = key_path.replace('features.', '', 1)
        elif key_path.startswith('models.'):
            config_name = 'model_config'
            key_path = key_path.replace('models.', '', 1)
        else:
            config_name = 'config'
        
        config = self.get_config(config_name)
        
        # Traverse the path
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default


# Create a single global instance
_config_loader = None

def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """
    Get or create global config loader instance.
    
    Args:
        config_dir: Configuration directory
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader

# Create default instance
config_loader = get_config_loader()
