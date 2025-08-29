"""
Input validation utilities for sensitive-term-guard.
"""

import os
from pathlib import Path
from typing import List, Union

from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, SUPPORTED_TEXT_EXTENSIONS
from .exceptions import FileProcessingError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate and sanitize file path.

    Args:
        file_path: Input file path

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid or unsafe
    """
    try:
        path = Path(file_path).resolve()

        # Check if path exists
        if not path.exists():
            raise ValidationError(f"Path does not exist: {file_path}")

        # Prevent directory traversal attacks
        if ".." in str(path):
            raise ValidationError(f"Invalid path with directory traversal: {file_path}")

        return path

    except Exception as e:
        raise ValidationError(f"Invalid file path '{file_path}': {e}")


def validate_file_size(file_path: Path, max_size: int = MAX_FILE_SIZE) -> None:
    """
    Validate file size is within limits.

    Args:
        file_path: Path to file
        max_size: Maximum allowed file size in bytes

    Raises:
        ValidationError: If file is too large
    """
    try:
        size = file_path.stat().st_size
        if size > max_size:
            size_mb = size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            raise ValidationError(
                f"File too large: {size_mb:.1f}MB > {max_mb:.1f}MB limit"
            )
    except OSError as e:
        raise FileProcessingError(f"Cannot check file size: {e}")


def validate_file_extension(file_path: Path, allowed_extensions: set = None) -> None:
    """
    Validate file has allowed extension.

    Args:
        file_path: Path to file
        allowed_extensions: Set of allowed extensions (default: SUPPORTED_TEXT_EXTENSIONS)

    Raises:
        ValidationError: If extension not allowed
    """
    if allowed_extensions is None:
        allowed_extensions = SUPPORTED_TEXT_EXTENSIONS

    extension = file_path.suffix.lower()
    if extension not in allowed_extensions:
        raise ValidationError(
            f"Unsupported file extension '{extension}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )


def validate_directory(directory_path: Union[str, Path]) -> Path:
    """
    Validate directory path and check permissions.

    Args:
        directory_path: Path to directory

    Returns:
        Validated Path object

    Raises:
        ValidationError: If directory is invalid
    """
    path = validate_file_path(directory_path)

    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {directory_path}")

    if not os.access(path, os.R_OK):
        raise ValidationError(f"No read permission for directory: {directory_path}")

    return path


def validate_total_size(
    file_paths: List[Path], max_total: int = MAX_TOTAL_SIZE
) -> None:
    """
    Validate total size of all files is within limits.

    Args:
        file_paths: List of file paths
        max_total: Maximum total size in bytes

    Raises:
        ValidationError: If total size exceeds limit
    """
    total_size = 0
    for path in file_paths:
        try:
            total_size += path.stat().st_size
        except OSError:
            logger.warning(f"Cannot check size of {path}")
            continue

    if total_size > max_total:
        total_mb = total_size / (1024 * 1024)
        max_mb = max_total / (1024 * 1024)
        raise ValidationError(
            f"Total file size too large: {total_mb:.1f}MB > {max_mb:.1f}MB limit"
        )


def validate_text_content(content: str, max_length: int = 1_000_000) -> None:
    """
    Validate text content is reasonable.

    Args:
        content: Text content to validate
        max_length: Maximum allowed length

    Raises:
        ValidationError: If content is invalid
    """
    if not isinstance(content, str):
        raise ValidationError("Content must be a string")

    if len(content) > max_length:
        raise ValidationError(
            f"Text content too long: {len(content)} > {max_length} characters"
        )

    # Check for binary content (high ratio of non-printable characters)
    printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
    if len(content) > 100 and printable_chars / len(content) < 0.8:
        raise ValidationError("Content appears to be binary data")


def validate_config_value(key: str, value, expected_type, allowed_values=None):
    """
    Validate configuration value.

    Args:
        key: Configuration key name
        value: Value to validate
        expected_type: Expected type or tuple of types
        allowed_values: Optional list/set of allowed values

    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Config '{key}' must be {expected_type.__name__}, got {type(value).__name__}"
        )

    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"Config '{key}' value '{value}' not in allowed values: {allowed_values}"
        )


def safe_read_file(
    file_path: Path, encoding: str = "utf-8", max_size: int = MAX_FILE_SIZE
) -> str:
    """
    Safely read file with validation and error handling.

    Args:
        file_path: Path to file
        encoding: Text encoding to use
        max_size: Maximum file size allowed

    Returns:
        File content as string

    Raises:
        ValidationError: If file is invalid
        FileProcessingError: If reading fails
    """
    # Validate file
    validate_file_size(file_path, max_size)
    validate_file_extension(file_path)

    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        validate_text_content(content)
        return content

    except UnicodeDecodeError as e:
        raise FileProcessingError(f"Cannot decode file {file_path}: {e}")
    except OSError as e:
        raise FileProcessingError(f"Cannot read file {file_path}: {e}")
