"""
Data models for sensitive term extraction and scanning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ExtractionMethod(Enum):
    """Enumeration of extraction methods."""

    NER = "ner"
    PATTERN = "pattern"
    CAPITALIZATION = "capitalization"
    QUOTES = "quotes"
    LLM = "llm_analysis"
    COMBINED = "combined"


class ScanResult(Enum):
    """Enumeration of scan result types."""

    VALID = "valid"
    INVALID = "invalid"
    SANITIZED = "sanitized"


@dataclass
class TermScore:
    """Represents a sensitive term with its scoring and metadata."""

    term: str
    frequency: int
    sensitivity_score: float
    contexts: List[str]
    extraction_method: str
    category: Optional[str] = None

    def __str__(self) -> str:
        return f"TermScore(term='{self.term}', score={self.sensitivity_score:.1f})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ScannerResult:
    """Result of scanning text for sensitive terms."""

    is_valid: bool
    sanitized_prompt: str
    risk_score: float
    found_terms: List[str]
    scan_method: str

    def __str__(self) -> str:
        return f"ScannerResult(valid={self.is_valid}, risk={self.risk_score:.1f}, terms={len(self.found_terms)})"


@dataclass
class ExtractionConfig:
    """Configuration for term extraction."""

    min_score: float = 2.0
    max_terms: int = 200
    methods: List[str] = None
    patterns: dict = None

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["ner", "patterns", "capitalization", "quotes"]
        if self.patterns is None:
            self.patterns = {}


@dataclass
class LLMConfig:
    """Configuration for LLM enhancement."""

    endpoint: str = "http://localhost:9099/v1/chat/completions"
    api_key: str = "default-key"
    model: str = "default-model"
    temperature: float = 0.1
    max_tokens: int = 300
    timeout: int = 30


@dataclass
class ScannerConfig:
    """Configuration for text scanning."""

    redaction_text: str = "[SENSITIVE]"
    case_sensitive: bool = False
    match_type: str = "contains"  # exact, contains, regex
    terms_file: Optional[str] = None
