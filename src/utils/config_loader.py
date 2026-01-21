"""
Enhanced configuration loader with full integration support.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Advanced configuration loader with validation and hot-reloading support.
    Provides access to all configuration files with proper error handling.
    """
    
    def __init__(self, config_dir: str = "config", env: str = "development"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
            env: Environment (development, staging, production)
        """
        self.config_dir = Path(config_dir)
        self.env = env
        self.configs: Dict[str, Any] = {}
        self._load_all_configs()
        logger.info(f"ConfigLoader initialized for {env} environment")
    
    def _load_all_configs(self):
        """Load all configuration files on initialization."""
        config_files = [
            "config.yaml",           # Main configuration
            "feature_config.yaml",   # Feature engineering
            "model_config.yaml",     # Model hyperparameters
        ]
        
        for config_file in config_files:
            try:
                config_name = config_file.replace('.yaml', '').replace('.yml', '')
                self.configs[config_name] = self._load_yaml_file(config_file)
            except Exception as e:
                logger.warning(f"Failed to load {config_file}: {e}")
                self.configs[config_name] = {}
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            filename: Name of YAML file
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        logger.debug(f"Loaded config from {config_path}")
        return config
    
    def get_config(self, config_name: str = "config") -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of config file (without extension)
        
        Returns:
            Configuration dictionary
        
        Raises:
            KeyError: If config not found
        """
        if config_name not in self.configs:
            # Try to load it
            try:
                self.configs[config_name] = self._load_yaml_file(f"{config_name}.yaml")
            except FileNotFoundError:
                raise KeyError(f"Configuration '{config_name}' not found")
        
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
        config = self.get_config("feature_config")
        return config
    
    def get_model_config(self, model_type: str = None) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Args:
            model_type: Specific model type (linear, xgboost, etc.)
        
        Returns:
            Model configuration dictionary
        """
        config = self.get_config("model_config")
        
        if model_type:
            # Navigate to specific model config
            if model_type in config:
                return config[model_type]
            # Try to find in nested structure
            for section in config.values():
                if isinstance(section, dict) and model_type in section:
                    return section[model_type]
            logger.warning(f"Model config for '{model_type}' not found")
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
    
    def reload(self):
        """Reload all configuration files."""
        logger.info("Reloading all configuration files")
        self.configs.clear()
        self._load_all_configs()
    
    def validate_configs(self) -> List[str]:
        """
        Validate all configuration files.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required configs exist
        required_configs = ['config', 'feature_config', 'model_config']
        
        for config_name in required_configs:
            try:
                config = self.get_config(config_name)
                if not config:
                    errors.append(f"Config '{config_name}' is empty")
            except KeyError:
                errors.append(f"Required config '{config_name}' not found")
        
        # Validate specific sections
        try:
            # Validate scraping config
            scraping = self.get_scraping_config()
            if not scraping.get('news_url'):
                errors.append("Scraping config missing 'news_url'")
            
            # Validate model configs
            model_configs = self.get_model_config()
            if not model_configs:
                errors.append("Model configs are empty")
            
        except Exception as e:
            errors.append(f"Config validation error: {e}")
        
        if not errors:
            logger.info("All configurations validated successfully")
        else:
            logger.warning(f"Configuration validation errors: {errors}")
        
        return errors
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]):
        """
        Save configuration to file.
        
        Args:
            config_name: Name of config file (without extension)
            config_data: Configuration data to save
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # Create backup if exists
        if config_path.exists():
            backup_path = config_path.with_suffix('.yaml.bak')
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.debug(f"Created backup at {backup_path}")
        
        # Save new config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        # Update in-memory config
        self.configs[config_name] = config_data
        
        logger.info(f"Saved configuration to {config_path}")
    
    def export_to_json(self, output_path: Path) -> Path:
        """
        Export all configurations to a single JSON file.
        
        Args:
            output_path: Path to save JSON file
        
        Returns:
            Path to saved file
        """
        all_configs = {
            'metadata': {
                'exported_at': '2024-01-01T00:00:00Z',  # Should use datetime
                'environment': self.env,
                'config_dir': str(self.config_dir)
            },
            'configurations': self.configs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_configs, f, indent=2, default=str)
        
        logger.info(f"Exported all configs to {output_path}")
        return output_path


# Global config loader instance - accessible from anywhere
_config_loader_instance = None


def get_config_loader(config_dir: str = "config", env: str = None) -> ConfigLoader:
    """
    Get or create global config loader instance.
    
    Args:
        config_dir: Configuration directory
        env: Environment
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader_instance
    
    if _config_loader_instance is None:
        # Determine environment from environment variable
        import os
        env = env or os.getenv('APP_ENV', 'development')
        _config_loader_instance = ConfigLoader(config_dir, env)
    
    return _config_loader_instance


def get_config(config_name: str = "config") -> Dict[str, Any]:
    """
    Quick access function for getting configuration.
    
    Args:
        config_name: Configuration name
    
    Returns:
        Configuration dictionary
    """
    return get_config_loader().get_config(config_name)


def get_value(key_path: str, default: Any = None) -> Any:
    """
    Quick access function for getting configuration value.
    
    Args:
        key_path: Dot-separated key path
        default: Default value
    
    Returns:
        Configuration value
    """
    return get_config_loader().get_value(key_path, default)


# Export for easy import
config_loader = get_config_loader()
