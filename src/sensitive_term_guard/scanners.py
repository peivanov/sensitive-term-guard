"""
Text scanners for identifying and anonymizing sensitive terms.
"""

import re
from typing import List, Optional, Set

from .logging_config import get_logger
from .models import ScannerConfig, ScannerResult

logger = get_logger(__name__)


class DomainSensitiveScanner:
    """Scanner for identifying and redacting domain-sensitive terms."""

    def __init__(
        self,
        terms_file: Optional[str] = None,
        terms_list: Optional[List[str]] = None,
        config: Optional[ScannerConfig] = None,
    ):
        """
        Initialize scanner with terms from file or list.

        Args:
            terms_file: Path to file containing sensitive terms (one per line)
            terms_list: List of sensitive terms
            config: Scanner configuration
        """
        self.config = config or ScannerConfig()
        self.sensitive_terms: Set[str] = set()

        if terms_file:
            self.load_terms_from_file(terms_file)
        elif terms_list:
            self.sensitive_terms = set(terms_list)

    def load_terms_from_file(self, filepath: str) -> None:
        """Load sensitive terms from a file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.sensitive_terms.add(line)
            logger.info(f"Loaded {len(self.sensitive_terms)} terms from {filepath}")
        except FileNotFoundError:
            logger.error(f"Terms file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading terms file: {e}")

    def add_terms(self, terms: List[str]) -> None:
        """Add terms to the sensitive terms set."""
        self.sensitive_terms.update(terms)

    def remove_terms(self, terms: List[str]) -> None:
        """Remove terms from the sensitive terms set."""
        for term in terms:
            self.sensitive_terms.discard(term)

    def scan(self, text: str) -> ScannerResult:
        """
        Scan text for sensitive terms and return sanitized version.

        Args:
            text: Text to scan

        Returns:
            ScannerResult with scanning results
        """
        if not self.sensitive_terms:
            return ScannerResult(
                is_valid=True,
                sanitized_prompt=text,
                risk_score=0.0,
                found_terms=[],
                scan_method="domain_sensitive",
            )

        found_terms = []
        sanitized_text = text
        total_matches = 0

        # Convert terms to appropriate format based on match type
        if self.config.match_type == "regex":
            # Terms are treated as regex patterns
            for term in self.sensitive_terms:
                try:
                    flags = 0 if self.config.case_sensitive else re.IGNORECASE
                    matches = list(re.finditer(term, sanitized_text, flags))
                    if matches:
                        found_terms.append(term)
                        total_matches += len(matches)
                        sanitized_text = re.sub(
                            term,
                            self.config.redaction_text,
                            sanitized_text,
                            flags=flags,
                        )
                except re.error:
                    # Invalid regex, treat as literal text
                    continue
        else:
            # Exact or contains matching
            for term in self.sensitive_terms:
                if self.config.match_type == "exact":
                    # Exact word matching
                    pattern = r"\b" + re.escape(term) + r"\b"
                else:
                    # Contains matching (default)
                    pattern = re.escape(term)

                flags = 0 if self.config.case_sensitive else re.IGNORECASE
                matches = list(re.finditer(pattern, sanitized_text, flags))

                if matches:
                    found_terms.append(term)
                    total_matches += len(matches)
                    sanitized_text = re.sub(
                        pattern, self.config.redaction_text, sanitized_text, flags=flags
                    )

        # Calculate risk score based on number of matches
        text_length = len(text.split())
        if text_length > 0:
            risk_score = min(total_matches / text_length * 10, 10.0)  # Cap at 10
        else:
            risk_score = 0.0

        is_valid = len(found_terms) == 0

        return ScannerResult(
            is_valid=is_valid,
            sanitized_prompt=sanitized_text,
            risk_score=risk_score,
            found_terms=found_terms,
            scan_method="domain_sensitive",
        )

    def validate_prompt(self, prompt: str) -> bool:
        """Quick validation without full scanning."""
        result = self.scan(prompt)
        return result.is_valid

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            "total_terms": len(self.sensitive_terms),
            "config": {
                "redaction_text": self.config.redaction_text,
                "case_sensitive": self.config.case_sensitive,
                "match_type": self.config.match_type,
            },
        }
