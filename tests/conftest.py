"""Test configuration."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Our Project Alpha system integrates with the CustomerDB to provide
    real-time analytics. The Internal Revenue Allocation Model helps
    optimize our financial planning processes.

    The PhoenixEngine handles data processing while the AlphaDashboard
    provides visualization capabilities. Our BetaNet infrastructure
    supports the entire Operation Thunder initiative.
    """


@pytest.fixture
def sample_terms():
    """Sample sensitive terms for testing."""
    return [
        "Project Alpha",
        "CustomerDB",
        "Internal Revenue Allocation Model",
        "PhoenixEngine",
        "AlphaDashboard",
        "BetaNet",
        "Operation Thunder",
    ]


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    doc_dir = Path(temp_dir) / "documents"
    doc_dir.mkdir()

    # Create sample files
    (doc_dir / "doc1.txt").write_text(
        """
    Project Alpha is our main initiative for customer analytics.
    The CustomerDB stores all relevant customer information.
    """
    )

    (doc_dir / "doc2.md").write_text(
        """
    # Technical Documentation

    Our PhoenixEngine processes data efficiently.
    The AlphaDashboard provides real-time visualizations.
    """
    )

    (doc_dir / "config.yml").write_text(
        """
    system:
      name: "BetaNet"
      operation: "Operation Thunder"
    """
    )

    return str(doc_dir)
