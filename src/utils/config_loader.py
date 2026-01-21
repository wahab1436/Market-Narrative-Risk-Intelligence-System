"""
Configuration loader for YAML/TOML configuration files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Load and manage configuration files.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
    
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
        
        self.configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
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
    
    def get_feature(self, *keys: str) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse in configuration
        
        Returns:
            Configuration value
        """
        config = self.configs.get("config", {})
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value


# Global config loader instance
config_loader = ConfigLoader()
