"""Tests for sensitive term scanners."""

import tempfile
from pathlib import Path

import pytest

from sensitive_term_guard.models import ScannerConfig
from sensitive_term_guard.scanners import DomainSensitiveScanner


class TestDomainSensitiveScanner:
    """Test cases for DomainSensitiveScanner."""

    def test_initialization_with_terms_list(self, sample_terms):
        """Test scanner initialization with terms list."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)
        assert len(scanner.sensitive_terms) == len(sample_terms)
        assert "Project Alpha" in scanner.sensitive_terms

    def test_initialization_with_terms_file(self, temp_dir, sample_terms):
        """Test scanner initialization with terms file."""
        # Create terms file
        terms_file = Path(temp_dir) / "terms.txt"
        terms_file.write_text("\n".join(sample_terms))

        scanner = DomainSensitiveScanner(terms_file=str(terms_file))
        assert len(scanner.sensitive_terms) == len(sample_terms)
        assert "Project Alpha" in scanner.sensitive_terms

    def test_initialization_with_config(self, sample_terms):
        """Test scanner initialization with custom config."""
        config = ScannerConfig(
            redaction_text="[REDACTED]", case_sensitive=True, match_type="exact"
        )
        scanner = DomainSensitiveScanner(terms_list=sample_terms, config=config)

        assert scanner.config.redaction_text == "[REDACTED]"
        assert scanner.config.case_sensitive == True
        assert scanner.config.match_type == "exact"

    def test_scan_with_sensitive_terms(self, sample_terms, sample_text):
        """Test scanning text containing sensitive terms."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)
        result = scanner.scan(sample_text)

        assert not result.is_valid  # Should find sensitive terms
        assert len(result.found_terms) > 0
        assert result.risk_score > 0
        assert "[SENSITIVE]" in result.sanitized_prompt

        # Check specific terms were found
        assert "Project Alpha" in result.found_terms
        assert "CustomerDB" in result.found_terms

    def test_scan_with_no_sensitive_terms(self, sample_terms):
        """Test scanning text with no sensitive terms."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)
        clean_text = "This is a normal text without any sensitive information."

        result = scanner.scan(clean_text)

        assert result.is_valid
        assert len(result.found_terms) == 0
        assert result.risk_score == 0
        assert result.sanitized_prompt == clean_text

    def test_scan_case_sensitive(self, sample_terms):
        """Test case-sensitive scanning."""
        config = ScannerConfig(case_sensitive=True)
        scanner = DomainSensitiveScanner(terms_list=sample_terms, config=config)

        # Should match exact case
        text1 = "Our Project Alpha system is running."
        result1 = scanner.scan(text1)
        assert "Project Alpha" in result1.found_terms

        # Should not match different case
        text2 = "Our project alpha system is running."
        result2 = scanner.scan(text2)
        assert "Project Alpha" not in result2.found_terms

    def test_scan_case_insensitive(self, sample_terms):
        """Test case-insensitive scanning (default)."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)

        # Should match different cases
        text1 = "Our project alpha system is running."
        result1 = scanner.scan(text1)
        assert len(result1.found_terms) > 0

        text2 = "Our PROJECT ALPHA system is running."
        result2 = scanner.scan(text2)
        assert len(result2.found_terms) > 0

    def test_scan_exact_match(self, sample_terms):
        """Test exact word matching."""
        config = ScannerConfig(match_type="exact")
        scanner = DomainSensitiveScanner(terms_list=sample_terms, config=config)

        # Should match whole word
        text1 = "The CustomerDB is our database."
        result1 = scanner.scan(text1)
        assert "CustomerDB" in result1.found_terms

        # Should not match partial word
        text2 = "The CustomerDatabase is extended."
        result2 = scanner.scan(text2)
        assert "CustomerDB" not in result2.found_terms

    def test_scan_contains_match(self, sample_terms):
        """Test contains matching (default)."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)

        # Should match partial occurrences
        text = "The CustomerDatabase extends CustomerDB functionality."
        result = scanner.scan(text)
        assert "CustomerDB" in result.found_terms

    def test_custom_redaction_text(self, sample_terms, sample_text):
        """Test custom redaction text."""
        config = ScannerConfig(redaction_text="[REDACTED]")
        scanner = DomainSensitiveScanner(terms_list=sample_terms, config=config)

        result = scanner.scan(sample_text)
        assert "[REDACTED]" in result.sanitized_prompt
        assert "[SENSITIVE]" not in result.sanitized_prompt

    def test_add_remove_terms(self, sample_terms):
        """Test adding and removing terms."""
        scanner = DomainSensitiveScanner(
            terms_list=sample_terms[:3]
        )  # Start with 3 terms

        # Add terms
        new_terms = ["NewTerm1", "NewTerm2"]
        scanner.add_terms(new_terms)
        assert "NewTerm1" in scanner.sensitive_terms
        assert "NewTerm2" in scanner.sensitive_terms

        # Remove terms
        scanner.remove_terms(["NewTerm1"])
        assert "NewTerm1" not in scanner.sensitive_terms
        assert "NewTerm2" in scanner.sensitive_terms

    def test_validate_prompt(self, sample_terms, sample_text):
        """Test quick prompt validation."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)

        # Should return False for text with sensitive terms
        assert not scanner.validate_prompt(sample_text)

        # Should return True for clean text
        clean_text = "This is a clean text."
        assert scanner.validate_prompt(clean_text)

    def test_get_stats(self, sample_terms):
        """Test scanner statistics."""
        config = ScannerConfig(redaction_text="[TEST]", case_sensitive=True)
        scanner = DomainSensitiveScanner(terms_list=sample_terms, config=config)

        stats = scanner.get_stats()

        assert stats["total_terms"] == len(sample_terms)
        assert stats["config"]["redaction_text"] == "[TEST]"
        assert stats["config"]["case_sensitive"] == True

    def test_risk_score_calculation(self, sample_terms):
        """Test risk score calculation."""
        scanner = DomainSensitiveScanner(terms_list=sample_terms)

        # Text with multiple sensitive terms should have higher risk
        high_risk_text = "Project Alpha uses CustomerDB and PhoenixEngine systems."
        result1 = scanner.scan(high_risk_text)

        # Text with single sensitive term should have lower risk
        low_risk_text = "The Project Alpha system is operational and working well with many other components."
        result2 = scanner.scan(low_risk_text)

        assert result1.risk_score >= result2.risk_score

    def test_empty_scanner(self):
        """Test scanner with no terms."""
        scanner = DomainSensitiveScanner(terms_list=[])
        text = "Any text should be valid."

        result = scanner.scan(text)
        assert result.is_valid
        assert len(result.found_terms) == 0
        assert result.risk_score == 0
        assert result.sanitized_prompt == text
