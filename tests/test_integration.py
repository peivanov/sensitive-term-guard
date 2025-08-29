#!/usr/bin/env python3
"""
Enhanced integration tests for sensitive-term-guard.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from sensitive_term_guard.extractors import BaseSensitiveTermExtractor
from sensitive_term_guard.models import ExtractionConfig
from sensitive_term_guard.scanners import DomainSensitiveScanner


class TestEndToEndWorkflow:
    """Test complete workflows from extraction to scanning."""

    @pytest.fixture
    def sample_documents(self):
        """Create temporary directory with sample documents."""
        temp_dir = tempfile.mkdtemp()

        # Create sample documents
        docs = {
            "project_spec.txt": """
                Project Phoenix is our new initiative using the CortexAPI.
                Contact john.doe@company.internal for details.
                Database: postgresql://user:pass@db.internal:5432/prod
                API Key: PROD_API_KEY_2024_abc123
            """,
            "team_info.txt": """
                Team Lead: Sarah Chen (s.chen@company.internal)
                DevOps: Mike Johnson (m.johnson@company.internal)
                Security Contact: security@company.internal
                Internal domain: app.company.internal
            """,
            "config.txt": """
                Production Environment:
                - Load Balancer: lb-prod.company.internal
                - Cache: redis://cache.company.internal:6379
                - Monitoring: DataDog (API Key: DD_KEY_prod_xyz789)
            """,
        }

        for filename, content in docs.items():
            with open(Path(temp_dir) / filename, "w") as f:
                f.write(content.strip())

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_extraction_workflow(self, sample_documents):
        """Test complete extraction workflow."""
        # Configure extraction
        config = ExtractionConfig(
            min_score=1.0, max_terms=50, methods=["ner", "patterns", "capitalization"]
        )

        # Extract terms
        extractor = BaseSensitiveTermExtractor(config=config)
        terms = extractor.extract_sensitive_terms_offline(sample_documents)

        # Verify extraction
        assert len(terms) > 0

        # Check for expected sensitive terms
        term_strings = [term.term for term in terms]
        expected_terms = [
            "company.internal",
            "Phoenix",
            "CortexAPI",
            "john.doe@company.internal",
        ]

        found_expected = sum(
            1
            for expected in expected_terms
            if any(expected in term for term in term_strings)
        )
        assert found_expected >= len(expected_terms) * 0.5  # At least 50% found

    def test_full_scanning_workflow(self, sample_documents):
        """Test complete scanning workflow."""
        # Extract terms first
        extractor = BaseSensitiveTermExtractor()
        terms = extractor.extract_sensitive_terms_offline(sample_documents)

        # Create scanner
        scanner = DomainSensitiveScanner(
            terms_list=[term.term for term in terms[:20]]  # Use top 20 terms
        )

        # Test text with sensitive content
        test_text = """
        The Project Phoenix team is using CortexAPI for data processing.
        Contact john.doe@company.internal for system access.
        Database connection: postgresql://user:pass@db.internal:5432/prod
        """

        # Scan text
        result = scanner.scan(test_text)

        # Verify results
        assert not result.is_valid  # Should be invalid due to sensitive content
        assert result.risk_score > 0
        assert len(result.found_terms) > 0
        assert "[SENSITIVE]" in result.sanitized_prompt

    def test_performance_with_large_dataset(self, sample_documents):
        """Test performance with larger document sets."""
        # Create additional documents
        for i in range(10):
            with open(Path(sample_documents) / f"doc_{i}.txt", "w") as f:
                f.write(
                    f"""
                Document {i} contains Project Alpha{i} information.
                Team member: user{i}@company.internal
                System: System{i}API
                Database: db{i}.company.internal
                """
                )

        # Extract with time limit
        import time

        start_time = time.time()

        extractor = BaseSensitiveTermExtractor()
        terms = extractor.extract_sensitive_terms_offline(sample_documents)

        elapsed = time.time() - start_time

        # Should complete within reasonable time (adjust as needed)
        assert elapsed < 30.0  # 30 seconds max
        assert len(terms) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_directory(self):
        """Test extraction from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = BaseSensitiveTermExtractor()
            terms = extractor.extract_sensitive_terms_offline(temp_dir)
            assert len(terms) == 0

    def test_invalid_directory(self):
        """Test extraction from non-existent directory."""
        extractor = BaseSensitiveTermExtractor()
        terms = extractor.extract_sensitive_terms_offline("/non/existent/path")
        assert len(terms) == 0

    def test_scanner_with_empty_terms(self):
        """Test scanner with no terms."""
        scanner = DomainSensitiveScanner(terms_list=[])
        result = scanner.scan("This is test text with sensitive data.")

        assert result.is_valid  # Should be valid (no terms to match)
        assert result.risk_score == 0.0
        assert len(result.found_terms) == 0

    def test_scanner_with_empty_text(self):
        """Test scanner with empty text."""
        scanner = DomainSensitiveScanner(terms_list=["sensitive", "data"])
        result = scanner.scan("")

        assert result.is_valid
        assert result.risk_score == 0.0
        assert len(result.found_terms) == 0


@pytest.mark.slow
class TestLLMIntegration:
    """Test LLM integration (requires local LLM server)."""

    def test_llm_availability(self):
        """Test if LLM endpoint is available."""
        import asyncio

        import aiohttp

        async def check_llm():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:11434/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        return response.status == 200
            except:
                return False

        is_available = asyncio.run(check_llm())
        if not is_available:
            pytest.skip("LLM server not available")

    @pytest.mark.skipif(
        True,  # Skip by default - enable manually for LLM testing
        reason="LLM integration test - requires manual enable",
    )
    def test_llm_enhanced_extraction(self, sample_documents):
        """Test LLM-enhanced extraction (optional)."""
        from sensitive_term_guard.extractors import LLMEnhancedSensitiveTermExtractor

        # This test would require LLM server setup
        # Implementation depends on actual LLM integration
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
