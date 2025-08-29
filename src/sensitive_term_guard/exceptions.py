"""
Custom exceptions for sensitive-term-guard.
"""


class SensitiveTermGuardError(Exception):
    """Base exception for all sensitive-term-guard errors."""

    pass


class ExtractionError(SensitiveTermGuardError):
    """Exception raised during term extraction."""

    pass


class ScanningError(SensitiveTermGuardError):
    """Exception raised during text scanning."""

    pass


class ConfigurationError(SensitiveTermGuardError):
    """Exception raised for configuration-related errors."""

    pass


class LLMError(SensitiveTermGuardError):
    """Exception raised for LLM-related errors."""

    pass


class FileProcessingError(SensitiveTermGuardError):
    """Exception raised during file processing."""

    pass


class ValidationError(SensitiveTermGuardError):
    """Exception raised for validation errors."""

    pass
