"""
Sensitive Term Guard - Extract sensitive information before sending to LLMs.
"""

__version__ = "1.0.0"
__author__ = "Petar Ivanov"

from .exceptions import (
    ConfigurationError,
    ExtractionError,
    FileProcessingError,
    ScanningError,
    SensitiveTermGuardError,
    ValidationError,
)
from .extractors import BaseSensitiveTermExtractor, LLMEnhancedSensitiveTermExtractor
from .logging_config import get_logger, setup_logging
from .models import ExtractionConfig, ScannerConfig, TermScore
from .scanners import DomainSensitiveScanner
from .utils import load_terms_from_file, save_terms_to_file
from .validation import safe_read_file, validate_directory, validate_file_path

__all__ = [
    "BaseSensitiveTermExtractor",
    "LLMEnhancedSensitiveTermExtractor",
    "DomainSensitiveScanner",
    "TermScore",
    "ExtractionConfig",
    "ScannerConfig",
    "save_terms_to_file",
    "load_terms_from_file",
    "SensitiveTermGuardError",
    "ExtractionError",
    "ScanningError",
    "ConfigurationError",
    "ValidationError",
    "FileProcessingError",
    "setup_logging",
    "get_logger",
    "validate_file_path",
    "validate_directory",
    "safe_read_file",
]
