"""
Utility functions for sensitive-term-guard.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .logging_config import get_logger
from .models import ExtractionConfig, LLMConfig, ScannerConfig, TermScore

logger = get_logger(__name__)


def save_terms_to_file(terms, filepath, include_metadata=True):
    """Save terms to a file with optional metadata."""
    dir_path = os.path.dirname(filepath)
    if dir_path:  # Only create directory if path has a directory component
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        if include_metadata:
            f.write("# Extracted sensitive terms\n")
            f.write(f"# Total terms: {len(terms)}\n\n")

        for term_score in terms:
            f.write(f"{term_score.term}\n")
            if include_metadata:
                f.write(
                    f"# Score: {term_score.sensitivity_score:.1f}, "
                    f"Frequency: {term_score.frequency}, "
                    f"Method: {term_score.extraction_method}\n"
                )

    logger.info(f"Saved {len(terms)} terms to {filepath}")


def load_terms_from_file(filepath: str) -> List[str]:
    """Load terms from a file (ignoring comments and metadata)."""
    terms = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    except FileNotFoundError:
        logger.error(f"Terms file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error loading terms file: {e}")

    return terms


def save_terms_to_separate_files(
    offline_terms: List[TermScore],
    llm_terms: Optional[List[TermScore]] = None,
    base_filepath: str = "domain-sensitive-terms",
    include_metadata: bool = True,
) -> Dict[str, str]:
    """Save extracted terms to separate files based on extraction method."""

    base_dir = os.path.dirname(base_filepath)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    files_created = {}

    # Save offline terms
    offline_file = f"{base_filepath}-offline.txt"
    save_terms_to_file(offline_terms, offline_file, include_metadata)
    files_created["offline_file"] = offline_file

    # Save LLM terms if provided
    if llm_terms:
        llm_file = f"{base_filepath}-llm.txt"
        save_terms_to_file(llm_terms, llm_file, include_metadata)
        files_created["llm_file"] = llm_file

    # Save combined terms
    combined_file = f"{base_filepath}.txt"
    all_terms = offline_terms.copy()
    if llm_terms:
        # Merge terms, avoiding duplicates
        offline_term_names = {term.term.lower() for term in offline_terms}
        for llm_term in llm_terms:
            if llm_term.term.lower() not in offline_term_names:
                all_terms.append(llm_term)

    # Sort combined terms by score
    all_terms.sort(key=lambda x: x.sensitivity_score, reverse=True)
    save_terms_to_file(all_terms, combined_file, include_metadata)
    files_created["combined_file"] = combined_file

    return files_created


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def save_config_to_file(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to YAML file."""
    dir_path = os.path.dirname(filepath)
    if dir_path:  # Only create directory if filepath has a directory part
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Saved configuration to {filepath}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        "extraction": {
            "min_score": 2.0,
            "max_terms": 200,
            "methods": ["ner", "patterns", "capitalization", "quotes"],
            "patterns": {
                "project_names": [
                    r"(?:Project|Operation)\s+([A-Z][a-zA-Z0-9_-]{2,})",
                ],
                "api_services": [
                    r"\b([a-zA-Z]+(?:API|Service|Engine))\b",
                ],
            },
        },
        "llm": {
            "endpoint": "http://localhost:9099/v1/chat/completions",
            "api_key": "your-api-key",
            "model": "your-model-name",
            "temperature": 0.1,
            "max_tokens": 300,
            "timeout": 30,
        },
        "scanning": {
            "redaction_text": "[SENSITIVE]",
            "case_sensitive": False,
            "match_type": "contains",
        },
    }


def config_dict_to_objects(config_dict: Dict[str, Any]) -> tuple:
    """Convert configuration dictionary to config objects."""
    extraction_config = ExtractionConfig(
        min_score=config_dict.get("extraction", {}).get("min_score", 2.0),
        max_terms=config_dict.get("extraction", {}).get("max_terms", 200),
        methods=config_dict.get("extraction", {}).get(
            "methods", ["ner", "patterns", "capitalization", "quotes"]
        ),
        patterns=config_dict.get("extraction", {}).get("patterns", {}),
    )

    llm_config = LLMConfig(
        endpoint=config_dict.get("llm", {}).get(
            "endpoint", "http://localhost:9099/v1/chat/completions"
        ),
        api_key=config_dict.get("llm", {}).get("api_key", "default-key"),
        model=config_dict.get("llm", {}).get("model", "default-model"),
        temperature=config_dict.get("llm", {}).get("temperature", 0.1),
        max_tokens=config_dict.get("llm", {}).get("max_tokens", 300),
        timeout=config_dict.get("llm", {}).get("timeout", 30),
    )

    scanner_config = ScannerConfig(
        redaction_text=config_dict.get("scanning", {}).get(
            "redaction_text", "[SENSITIVE]"
        ),
        case_sensitive=config_dict.get("scanning", {}).get("case_sensitive", False),
        match_type=config_dict.get("scanning", {}).get("match_type", "contains"),
    )

    return extraction_config, llm_config, scanner_config


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []

    # Validate extraction config
    extraction = config_dict.get("extraction", {})
    if "min_score" in extraction and not isinstance(
        extraction["min_score"], (int, float)
    ):
        errors.append("extraction.min_score must be a number")
    if "max_terms" in extraction and not isinstance(extraction["max_terms"], int):
        errors.append("extraction.max_terms must be an integer")
    if "methods" in extraction and not isinstance(extraction["methods"], list):
        errors.append("extraction.methods must be a list")

    # Validate LLM config
    llm = config_dict.get("llm", {})
    if "temperature" in llm and not isinstance(llm["temperature"], (int, float)):
        errors.append("llm.temperature must be a number")
    if "max_tokens" in llm and not isinstance(llm["max_tokens"], int):
        errors.append("llm.max_tokens must be an integer")
    if "timeout" in llm and not isinstance(llm["timeout"], int):
        errors.append("llm.timeout must be an integer")

    # Validate scanning config
    scanning = config_dict.get("scanning", {})
    if "case_sensitive" in scanning and not isinstance(
        scanning["case_sensitive"], bool
    ):
        errors.append("scanning.case_sensitive must be a boolean")
    if "match_type" in scanning and scanning["match_type"] not in [
        "exact",
        "contains",
        "regex",
    ]:
        errors.append("scanning.match_type must be one of: exact, contains, regex")

    return errors


def find_config_file(start_path: str = ".") -> Optional[str]:
    """Find configuration file in current directory or parent directories."""
    config_names = [
        "config.yml",
        "config.yaml",
        "sensitive-term-guard.yml",
        "sensitive-term-guard.yaml",
    ]

    path = Path(start_path).resolve()
    while path != path.parent:
        for config_name in config_names:
            config_path = path / config_name
            if config_path.exists():
                return str(config_path)
        path = path.parent

    return None


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries (later configs override earlier ones)."""
    result = {}

    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result
