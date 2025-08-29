# # 🛡️ Sensitive Term Guard

> **Sensitive term extraction and anonymization for LLM applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Sensitive Term Guard is a comprehensive Python package that automatically identifies and anonymizes sensitive information in documents before processing them with Large Language Models (LLMs). It combines multiple AI-powered extraction methods with enterprise-grade security to protect sensitive information while maintaining document utility.

## 🎯 Why Sensitive Term Guard?

When using LLMs for document processing, organizations risk exposing:
- **Project codenames** and internal initiatives
- **Employee names** and contact information
- **API endpoints** and system architectures
- **Database credentials** and connection strings
- **Client information** and partner details
- **Proprietary terminology** and trade secrets

## ✨ Key Features

### 🔍 **Multi-Modal Detection**
- **Named Entity Recognition (NER)**: Identifies people, organizations, locations using spaCy
- **Pattern Matching**: Detects project names, API services, system identifiers
- **Capitalization Analysis**: Finds proper nouns and technical terms
- **Quoted Content**: Extracts terms in quotes (often indicating importance)
- **LLM Enhancement**: Optional AI-powered analysis for context-aware detection

### 🛡️ **Enterprise Security**
- **Risk Scoring**: 0-10 risk assessment for documents
- **Configurable Sensitivity**: Adjustable thresholds for different security levels
- **Flexible Redaction**: Customizable replacement text and anonymization strategies
- **Audit Trail**: Detailed logging of what was detected and anonymized

### 🔧 **Developer Friendly**
- **Professional CLI Tools**: Rich terminal interfaces with progress indicators
- **Python API**: Full programmatic access for integration
- **YAML Configuration**: Flexible, version-controllable settings
- **Comprehensive Testing**: Validated with realistic organizational documents

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/peivanov/sensitive-term-guard.git
cd sensitive-term-guard
pip install -e .

# Install required models
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from sensitive_term_guard.extractors import BaseSensitiveTermExtractor
from sensitive_term_guard.models import ExtractionConfig

# Extract sensitive terms
config = ExtractionConfig()
extractor = BaseSensitiveTermExtractor(config)
terms = extractor.extract_documents_from_directory("documents/")

# Print results with risk scores
for term in terms:
    print(f"{term.term} (score: {term.score})")
```

### CLI Usage

```bash
# Extract sensitive terms from documents
sensitive-term-guard extract documents/ --output extracted_terms.json

# Scan and anonymize a document
sensitive-term-guard scan document.txt --output anonymized_document.txt

# Use custom configuration
sensitive-term-guard extract documents/ --config my_config.yml
```

## ⚙️ Configuration

Create a `config.yml` file to customize behavior:

```yaml
# Extraction settings
extraction:
  min_score: 2.0          # Minimum risk score to include term
  max_terms: 200          # Maximum terms to extract
  methods:                # Extraction methods to use
    - "ner"                 # Named Entity Recognition
    - "patterns"            # Pattern matching
    - "capitalization"      # Capitalized sequences
    - "quotes"              # Quoted terms
  patterns:               # Custom patterns for your organization
    project_names:
      - "Project \\w+"
      - "Operation \\w+"

# Scanner settings
scanner:
  replacement_text: "[REDACTED]"
  preserve_length: false
  min_term_length: 3

# LLM settings (optional)
llm:
  endpoint: "http://localhost:11434/v1/chat/completions"
  model: "llama3.1:8b"
  api_key: "ollama"
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/sensitive_term_guard

# Check code quality with pytest
python -m pytest tests/ --cov=src/sensitive_term_guard --cov-report=html
```

## 🔌 LLM Integration

Works with popular LLM providers:

```python
from sensitive_term_guard.extractors import LLMEnhancedSensitiveTermExtractor
from sensitive_term_guard.models import LLMConfig

# Configure for local Ollama
llm_config = LLMConfig(
    endpoint="http://localhost:11434/v1/chat/completions",
    model="llama3.1:8b",
    api_key="ollama"
)

# Enhanced extraction with AI analysis
extractor = LLMEnhancedSensitiveTermExtractor(
    extraction_config=ExtractionConfig(),
    llm_config=llm_config
)

terms = await extractor.extract_sensitive_terms_with_llm("documents/")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/what-a-great-feature`)
3. Run tests (`python -m pytest tests/`)
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

## 📚 Documentation

- **Examples**: Check `examples/` directory for usage patterns
- **Troubleshooting**: See `TROUBLESHOOTING.md` for common issues
- **API Documentation**: Comprehensive docstrings in source code

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
