"""Logging utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('quantum_decoh')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        log_path = Path(log_dir) / 'train.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
