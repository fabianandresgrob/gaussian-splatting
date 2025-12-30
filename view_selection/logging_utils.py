"""
Logging utilities for the view selection module.

Provides a consistent logging interface that:
- Uses Python's logging module for proper log management
- Supports file logging for ablation studies
- Integrates with tqdm progress bars without interference
- Allows configurable verbosity levels

Usage:
    from view_selection.logging_utils import get_logger, configure_logging
    
    # At module initialization
    logger = get_logger(__name__)
    
    # In training script (before creating selectors)
    configure_logging(log_dir="/path/to/output", level="INFO")
    
    # In selector code
    logger.info("Initialized with %d cameras", n_cameras)
    logger.debug("Camera %d selected with prob %.4f", uid, prob)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from functools import lru_cache


# Module-level configuration
_logging_configured = False
_log_dir: Optional[str] = None
_file_handler: Optional[logging.FileHandler] = None


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Custom handler that uses tqdm.write() to avoid interfering with progress bars.
    Falls back to normal stderr if tqdm is not available.
    """
    
    def __init__(self):
        super().__init__()
        try:
            from tqdm import tqdm
            self._tqdm_write = tqdm.write
        except ImportError:
            self._tqdm_write = None
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if self._tqdm_write is not None:
                self._tqdm_write(msg)
            else:
                sys.stderr.write(msg + self.terminator)
                sys.stderr.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance configured for view selection module
    """
    # Use a consistent prefix for all view_selection loggers
    if not name.startswith('view_selection'):
        name = f'view_selection.{name}'
    
    return logging.getLogger(name)


def configure_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
    log_file: Optional[str] = None,
    use_tqdm_handler: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the view selection module.
    
    Call this once at the start of training before creating any selectors.
    
    Args:
        log_dir: Directory to write log files. If provided, creates a timestamped
                 log file for this session.
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        console: Whether to output to console/stderr
        log_file: Specific log file path (overrides auto-generated name)
        use_tqdm_handler: Use tqdm-compatible output handler
        format_string: Custom format string for log messages
    """
    global _logging_configured, _log_dir, _file_handler
    
    # Get root logger for view_selection module
    logger = logging.getLogger('view_selection')
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    if console:
        if use_tqdm_handler:
            console_handler = TqdmLoggingHandler()
        else:
            console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir or log_file:
        if log_file:
            file_path = log_file
        else:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(log_dir, f'view_selection_{timestamp}.log')
        
        _file_handler = logging.FileHandler(file_path)
        _file_handler.setFormatter(formatter)
        _file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        logger.addHandler(_file_handler)
        _log_dir = log_dir or os.path.dirname(file_path)
        
        logger.info(f"View selection logging initialized. Log file: {file_path}")
    
    # Don't propagate to root logger
    logger.propagate = False
    
    _logging_configured = True


def get_log_dir() -> Optional[str]:
    """Get the current log directory, if configured."""
    return _log_dir


def log_config(logger: logging.Logger, config: Dict[str, Any], name: str = "Configuration") -> None:
    """
    Log a configuration dictionary in a readable format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary to log
        name: Name to use in the log message
    """
    logger.info(f"{name}:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_statistics(logger: logging.Logger, stats: Dict[str, Any], name: str = "Statistics") -> None:
    """
    Log statistics dictionary in a readable format.
    
    Args:
        logger: Logger instance
        stats: Statistics dictionary to log
        name: Name to use in the log message
    """
    logger.info(f"{name}:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


class SelectorLoggerMixin:
    """
    Mixin class that provides logging functionality for selectors.
    
    Inherit from this mixin to get consistent logging behavior:
    
        class MySelector(ViewSelector, SelectorLoggerMixin):
            def __init__(self, ...):
                super().__init__(...)
                self._init_logger()
                self.logger.info("MySelector initialized")
    """
    
    _logger: Optional[logging.Logger] = None
    
    def _init_logger(self) -> None:
        """Initialize the logger for this selector instance."""
        class_name = self.__class__.__name__
        self._logger = get_logger(class_name)
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger, initializing if needed."""
        if self._logger is None:
            self._init_logger()
        return self._logger
    
    def log_init(self, **kwargs) -> None:
        """Log initialization with key parameters."""
        class_name = self.__class__.__name__
        params = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.info(f"Initialized {class_name}: {params}")
    
    def log_selection(self, cam_uid: int, cam_name: str, probability: float, 
                      iteration: int, count: int) -> None:
        """Log a camera selection (at DEBUG level to avoid spam)."""
        self.logger.debug(
            f"Iter {iteration}: Selected camera {cam_uid} ({cam_name}) "
            f"prob={probability:.4f} count={count}"
        )
    
    def log_iteration_summary(self, iteration: int, stats: Dict[str, Any]) -> None:
        """Log summary statistics at an iteration checkpoint."""
        self.logger.info(f"Iteration {iteration} summary: {stats}")


# Convenience functions for common log patterns
def log_warning_once(logger: logging.Logger, message: str, key: Optional[str] = None) -> None:
    """
    Log a warning message only once (useful for repeated conditions).
    
    Args:
        logger: Logger instance
        message: Warning message
        key: Unique key for deduplication (defaults to message)
    """
    if not hasattr(log_warning_once, '_seen'):
        log_warning_once._seen = set()
    
    key = key or message
    if key not in log_warning_once._seen:
        log_warning_once._seen.add(key)
        logger.warning(message)
