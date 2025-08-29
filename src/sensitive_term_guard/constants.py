"""
Constants and default configurations for sensitive-term-guard.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Petar Ivanov"

# File extensions
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".html",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
}
SUPPORTED_DOC_EXTENSIONS = {".docx", ".pdf"}

# Default patterns for extraction
DEFAULT_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "url": r"https?://[^\s/$.?#].[^\s]*",
    "api_key": r"\b[A-Z][A-Z0-9_]{10,}\b",
    "uuid": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "database_url": r"\b\w+://[^:\s]+:[^@\s]+@[^:\s]+:\d+/\w+\b",
    "internal_domain": r"\b[a-zA-Z0-9.-]+\.(?:internal|local|corp|company)\b",
    "project_name": r"(?:Project|Operation)\s+([A-Z][a-zA-Z0-9_-]{2,})",
    "system_name": r"\b[A-Z][a-zA-Z]*(?:API|Service|Engine|System|Platform|Hub)\b",
}

# Default extraction configuration
DEFAULT_EXTRACTION_CONFIG = {
    "min_score": 2.0,
    "max_terms": 200,
    "methods": ["ner", "patterns", "capitalization", "quotes"],
    "patterns": DEFAULT_PATTERNS,
}

# Default scanning configuration
DEFAULT_SCANNING_CONFIG = {
    "redaction_text": "[SENSITIVE]",
    "case_sensitive": False,
    "match_type": "contains",  # exact, contains, regex
}

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "endpoint": "http://localhost:11434/v1/chat/completions",
    "model": "llama3:8b",
    "api_key": "ollama",
    "temperature": 0.1,
    "max_tokens": 300,
    "timeout": 30,
}

# Risk scoring thresholds
RISK_THRESHOLDS = {
    "low": 2.0,
    "medium": 5.0,
    "high": 7.5,
    "critical": 9.0,
}

# NER entity types to extract
SENSITIVE_NER_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
}

# File size limits (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB

# Performance settings
BATCH_SIZE = 100
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30

# Logging settings
LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
