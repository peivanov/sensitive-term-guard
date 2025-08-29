#!/usr/bin/env python3
"""
Basic usage example for sensitive-term-guard.
"""

import asyncio

from sensitive_term_guard import (
    BaseSensitiveTermExtractor,
    DomainSensitiveScanner,
    LLMEnhancedSensitiveTermExtractor,
    save_terms_to_file,
)


def basic_offline_extraction():
    """Example of basic offline term extraction."""
    print("=== Basic Offline Extraction ===")

    # Initialize extractor
    extractor = BaseSensitiveTermExtractor()

    # Extract terms from documents (replace with your document path)
    # terms = extractor.extract_sensitive_terms_offline("./test_documents")

    # For demonstration, create some sample terms
    from sensitive_term_guard.models import TermScore

    sample_terms = [
        TermScore("Project Alpha", 3, 8.5, ["Context 1"], "pattern_based"),
        TermScore("CustomerDB", 2, 7.2, ["Context 2"], "pattern_based"),
        TermScore(
            "Internal Revenue Allocation Model", 1, 9.1, ["Context 3"], "pattern_based"
        ),
    ]

    print(f"Extracted {len(sample_terms)} terms:")
    for term in sample_terms:
        print(f"  {term.term} (score: {term.sensitivity_score:.1f})")

    # Save terms to file
    save_terms_to_file(sample_terms, "sample-terms.txt", include_metadata=True)
    print("Terms saved to sample-terms.txt")

    return sample_terms


async def llm_enhanced_extraction():
    """Example of LLM-enhanced extraction."""
    print("\n=== LLM-Enhanced Extraction ===")

    # Initialize LLM-enhanced extractor
    extractor = LLMEnhancedSensitiveTermExtractor(
        llm_endpoint="http://localhost:9099/v1/chat/completions", api_key="your-api-key"
    )

    # For demonstration purposes, we'll skip actual LLM calls
    print("Note: This example skips actual LLM calls for demonstration")
    print("In practice, you would call:")
    print("  results = await extractor.extract_sensitive_terms_with_llm('./documents')")

    return []


def text_scanning_example():
    """Example of scanning and anonymizing text."""
    print("\n=== Text Scanning Example ===")

    # Create sample terms file
    sample_terms = ["Project Alpha", "CustomerDB", "Internal Revenue Allocation Model"]

    # Initialize scanner
    scanner = DomainSensitiveScanner(terms_list=sample_terms)

    # Test text
    test_text = """
    Our Project Alpha system integrates with the CustomerDB to provide
    real-time analytics. The Internal Revenue Allocation Model helps
    optimize our financial planning processes.
    """

    print("Original text:")
    print(test_text)

    # Scan the text
    result = scanner.scan(test_text)

    print(f"\nScan results:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Risk score: {result.risk_score:.1f}")
    print(f"  Found terms: {len(result.found_terms)}")

    print(f"\nSanitized text:")
    print(result.sanitized_prompt)


def configuration_example():
    """Example of using configuration objects."""
    print("\n=== Configuration Example ===")

    from sensitive_term_guard.models import ExtractionConfig, ScannerConfig

    # Create custom extraction configuration
    extraction_config = ExtractionConfig(
        min_score=3.0,
        max_terms=50,
        methods=["ner", "patterns"],
        patterns={
            "custom_ids": [r"CUST-[A-Z0-9]{6}"],
            "project_codes": [r"PROJ-[0-9]{4}"],
        },
    )

    # Create custom scanner configuration
    scanner_config = ScannerConfig(
        redaction_text="[REDACTED]", case_sensitive=True, match_type="exact"
    )

    print("Custom extraction config:")
    print(f"  Min score: {extraction_config.min_score}")
    print(f"  Max terms: {extraction_config.max_terms}")
    print(f"  Methods: {extraction_config.methods}")

    print("Custom scanner config:")
    print(f"  Redaction text: {scanner_config.redaction_text}")
    print(f"  Case sensitive: {scanner_config.case_sensitive}")
    print(f"  Match type: {scanner_config.match_type}")


async def main():
    """Run all examples."""
    print("Sensitive Term Guard - Usage Examples")
    print("=" * 40)

    # Run examples
    terms = basic_offline_extraction()
    await llm_enhanced_extraction()
    text_scanning_example()
    configuration_example()

    print("\n=== Examples Complete ===")
    print("For more advanced usage, see the CLI tools:")
    print("  sensitive-term-extract --help")
    print("  sensitive-term-scan --help")
    print("  sensitive-term-config --help")


if __name__ == "__main__":
    asyncio.run(main())
