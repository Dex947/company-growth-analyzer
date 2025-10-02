"""
Logging utilities using loguru.
Provides consistent logging across the project.
"""

from loguru import logger
import sys
from pathlib import Path
from config.config import LOGS_PATH

def setup_logger(log_file: str = "pipeline.log", level: str = "INFO"):
    """
    Configure the logger with file and console output.

    Args:
        log_file: Name of the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File handler
    log_path = LOGS_PATH / log_file
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    return logger

# Create default logger instance
log = setup_logger()
