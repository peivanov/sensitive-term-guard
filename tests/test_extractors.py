"""Tests for sensitive term extractors."""

import pytest

from sensitive_term_guard.extractors import BaseSensitiveTermExtractor
from sensitive_term_guard.models import ExtractionConfig


class TestBaseSensitiveTermExtractor:
    """Test cases for BaseSensitiveTermExtractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = BaseSensitiveTermExtractor()
        assert extractor is not None
        assert extractor.config is not None
        assert extractor.config.min_score == 2.0
        assert extractor.config.max_terms == 200

    def test_initialization_with_config(self):
        """Test extractor initialization with custom config."""
        config = ExtractionConfig(min_score=3.0, max_terms=50)
        extractor = BaseSensitiveTermExtractor(config=config)
        assert extractor.config.min_score == 3.0
        assert extractor.config.max_terms == 50

    def test_extract_technical_patterns(self):
        """Test technical pattern extraction."""
        extractor = BaseSensitiveTermExtractor()
        text = """
        Our Project Alpha system uses the PhoenixEngine for processing.
        The CustomerDB stores data while AlphaDashboard shows metrics.
        """

        patterns = extractor.extract_technical_patterns(text)

        # Check if patterns were extracted
        assert len(patterns) > 0

        # Check for specific pattern categories
        found_terms = set()
        for category_terms in patterns.values():
            found_terms.update(category_terms)

        # Should find some of these terms
        expected_terms = {
            "Project Alpha",
            "PhoenixEngine",
            "CustomerDB",
            "AlphaDashboard",
        }
        found_matches = found_terms.intersection(expected_terms)
        assert len(found_matches) > 0

    def test_extract_capitalized_sequences(self):
        """Test capitalized sequence extraction."""
        extractor = BaseSensitiveTermExtractor()
        text = "The Internal Revenue Service handles Project Alpha teams."

        sequences = extractor.extract_capitalized_sequences(text)

        # Should extract proper noun sequences (not common words like "The")
        assert "Project Alpha" in sequences

        # Test that it extracts multi-word proper nouns
        text2 = "The Machine Learning Algorithm processes data for Digital Marketing Campaign."
        sequences2 = extractor.extract_capitalized_sequences(text2)

        # Should extract sequences of capitalized words
        assert len(sequences2) > 0
        # Check that common word patterns are filtered out
        assert "The Machine" not in sequences2

    def test_extract_quoted_terms(self):
        """Test quoted term extraction."""
        extractor = BaseSensitiveTermExtractor()
        text = """
        The system called "Project Alpha" uses the "CustomerDB" database.
        We also have 'Internal System' and `BetaNet` components.
        """

        quoted_terms = extractor.extract_quoted_terms(text)

        assert "Project Alpha" in quoted_terms
        assert "CustomerDB" in quoted_terms
        assert "Internal System" in quoted_terms
        assert "BetaNet" in quoted_terms

    def test_calculate_term_frequency_significance(self):
        """Test term frequency and significance calculation."""
        extractor = BaseSensitiveTermExtractor()
        terms = {"test_category": {"Project Alpha", "CustomerDB"}}
        text = "Project Alpha is mentioned twice. Project Alpha uses CustomerDB."

        term_scores = extractor.calculate_term_frequency_significance(terms, text)

        assert "Project Alpha" in term_scores
        assert "CustomerDB" in term_scores

        # Project Alpha should have higher frequency
        assert term_scores["Project Alpha"].frequency == 2
        assert term_scores["CustomerDB"].frequency == 1

        # Both should have positive sensitivity scores
        assert term_scores["Project Alpha"].sensitivity_score > 0
        assert term_scores["CustomerDB"].sensitivity_score > 0

    def test_filter_and_rank_terms(self):
        """Test term filtering and ranking."""
        extractor = BaseSensitiveTermExtractor()

        from sensitive_term_guard.models import TermScore

        term_scores = {
            "high_score": TermScore("Important Term", 5, 8.5, [], "test"),
            "low_score": TermScore("Common Word", 10, 1.0, [], "test"),
            "medium_score": TermScore("Project Alpha", 3, 4.5, [], "test"),
        }

        ranked_terms = extractor.filter_and_rank_terms(term_scores)

        # Should filter out low score terms
        term_names = [t.term for t in ranked_terms]
        assert "Important Term" in term_names
        assert "Project Alpha" in term_names
        assert "Common Word" not in term_names  # Below min_score

        # Should be ranked by score (highest first)
        assert ranked_terms[0].sensitivity_score >= ranked_terms[-1].sensitivity_score

    def test_extract_documents_from_directory(self, sample_documents):
        """Test document extraction from directory."""
        extractor = BaseSensitiveTermExtractor()
        documents = extractor.extract_documents_from_directory(sample_documents)

        assert len(documents) > 0

        # Check document structure
        doc = documents[0]
        assert "id" in doc
        assert "content" in doc
        assert "filename" in doc
        assert "path" in doc
        assert "extension" in doc

        # Check content was loaded
        assert len(doc["content"]) > 0

    def test_extract_sensitive_terms_offline(self, sample_documents):
        """Test complete offline extraction pipeline."""
        config = ExtractionConfig(
            min_score=1.0, max_terms=50
        )  # Lower threshold for testing
        extractor = BaseSensitiveTermExtractor(config=config)

        terms = extractor.extract_sensitive_terms_offline(sample_documents)

        # Should extract some terms
        assert len(terms) > 0

        # Check term structure
        term = terms[0]
        assert hasattr(term, "term")
        assert hasattr(term, "sensitivity_score")
        assert hasattr(term, "frequency")
        assert hasattr(term, "extraction_method")

        # Terms should be ranked by score
        if len(terms) > 1:
            assert terms[0].sensitivity_score >= terms[1].sensitivity_score
