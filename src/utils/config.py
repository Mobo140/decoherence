"""Configuration loading utilities."""
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, optionally merging with base config.
    
    Args:
        config_path: Path to YAML config file
        base_config_path: Optional path to base config to merge with
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path_obj = Path(config_path)
    if base_config_path is None and config_path_obj.parent.name == 'experiments':
        # Auto-detect base config
        base_config_path = config_path_obj.parent.parent / 'base.yaml'
    
    base_config = {}
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f) or {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    if base_config:
        config = merge_configs(base_config, config)
    
    return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries (override takes precedence).
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def parse_args_with_config() -> argparse.Namespace:
    """Parse command-line arguments with config file support.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Quantum decoherence prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Model type override')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for evaluation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    return parser.parse_args()
