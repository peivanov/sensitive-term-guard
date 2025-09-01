"""
Sensitive term extraction functionality.

This module provides classes for extracting sensitive terms from documents
using various methods including NER, pattern matching, and optional LLM analysis.
"""

import asyncio
import re
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Union, Tuple, Any

try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import spacy
import nltk
import aiohttp
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from .logging_config import get_logger
from .models import TermScore, ExtractionConfig, LLMConfig

# Setup logging
logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class BaseSensitiveTermExtractor:
    """Base class with LLM-free extraction methods"""

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.config = config or ExtractionConfig()

        # Initialize embedding model for similarity (optional)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.embedding_model = None

        # Load spaCy for NER with enhanced EntityRuler
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None  # Initialize NLTK
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize

        self.stop_words = set(stopwords.words("english"))
        self.word_tokenize = word_tokenize
        self.sent_tokenize = sent_tokenize

    def extract_documents_from_directory(
        self, directory_path: str, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract text from various document types in a directory"""
        documents = []
        supported_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".conf",
            ".cfg",
            ".ini",
            ".env",
            ".log",
            ".rst",
            ".toml",
            ".yang",
            ".bash",
            ".sh",
        }

        for file_path in Path(directory_path).rglob("*"):
            if (
                file_path.suffix.lower() in supported_extensions
                or file_path.suffix == ""
                or file_path.is_file()
            ):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content.strip():
                            documents.append(
                                {
                                    "id": str(file_path),
                                    "content": content,
                                    "filename": file_path.name,
                                    "path": str(file_path),
                                    "extension": (
                                        file_path.suffix.lower()
                                        if file_path.suffix
                                        else "no_extension"
                                    ),
                                }
                            )
                            if verbose:
                                logger.info(
                                    f"Loaded: {file_path.name} ({len(content)} chars)"
                                )
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        return documents

    def _setup_custom_entity_ruler(self):
        """Setup advanced EntityRuler with domain-specific patterns"""
        if not self.nlp:
            return None
            
        # Create EntityRuler with advanced configuration
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner", config={
                "phrase_matcher_attr": "LOWER",  # Case-insensitive matching
                "validate": True,  # Validate patterns
                "overwrite_ents": False,  # Don't overwrite existing entities
            })
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        # Define sophisticated domain-specific patterns
        patterns = [
            # PROJECT NAMES - Using token patterns for flexibility
            {"label": "PROJECT_NAME", "pattern": [
                {"LOWER": {"IN": ["project", "initiative", "program", "operation", "codename"]}},
                {"IS_TITLE": True}
            ]},
            {"label": "PROJECT_NAME", "pattern": [
                {"LOWER": "project"},
                {"IS_TITLE": True},
                {"IS_TITLE": True, "OP": "?"}  # Optional second word
            ]},
            
            # TECHNICAL SYSTEMS - Using simpler patterns that work
            {"label": "TECH_SYSTEM", "pattern": [
                {"IS_TITLE": True},
                {"LOWER": {"IN": ["engine", "api", "service", "system", "platform", "dashboard", "monitor", "database", "cache", "vault"]}}
            ]},
            {"label": "TECH_SYSTEM", "pattern": [
                {"IS_TITLE": True},
                {"LOWER": {"IN": ["cluster", "net", "grid", "hub"]}}
            ]},
            
            # CUSTOMER IDENTIFIERS - Simplified
            {"label": "CUSTOMER_ID", "pattern": [
                {"LOWER": "customer"},
                {"IS_UPPER": True}
            ]},
            
            # ENVIRONMENT NAMES - Using word lists instead of regex
            {"label": "ENVIRONMENT", "pattern": [
                {"LOWER": {"IN": ["prod", "staging", "dev", "test", "qa", "uat"]}},
                {"IS_ALPHA": True, "OP": "?"}
            ]},
            
            # CAMELCASE TECHNICAL TERMS - Simplified approach
            {"label": "CAMELCASE_TECH", "pattern": [
                {"IS_TITLE": True},
                {"IS_TITLE": True}
            ]},
            
            # COMPOUND SYSTEM NAMES
            {"label": "COMPOUND_SYSTEM", "pattern": [
                {"IS_TITLE": True},
                {"TEXT": {"IN": ["-", "_"]}},
                {"IS_TITLE": True}
            ]},
            
            # SECURITY CLASSIFICATIONS
            {"label": "SECURITY_CLASS", "pattern": [
                {"LOWER": {"IN": ["confidential", "restricted", "internal", "proprietary", "classified"]}},
                {"IS_ALPHA": True, "OP": "*"}
            ]},
            
            # REVENUE MODEL PATTERNS
            {"label": "REVENUE_MODEL", "pattern": [
                {"LOWER": "revenue"},
                {"LOWER": "allocation"},
                {"IS_TITLE": True}
            ]},
        ]
        
        # Add phrase patterns for common organizational terms
        # Use nlp.make_doc() to avoid unnecessary parsing as suggested by spaCy warning
        # These are generic organizational patterns that can be customized per deployment
        phrase_patterns = [
            # Generic infrastructure terms (can be customized via config)
            {"label": "ORG_SPECIFIC", "pattern": self.nlp.make_doc("Internal Use Only")},
            {"label": "ORG_SPECIFIC", "pattern": self.nlp.make_doc("Do Not Share")},
            {"label": "ORG_SPECIFIC", "pattern": self.nlp.make_doc("Confidential Information")},
            {"label": "ORG_SPECIFIC", "pattern": self.nlp.make_doc("Internal Revenue Allocation Model")},
            
            # Note: Add organization-specific terms via configuration file or environment variables
            # Examples that would be configured per deployment:
            # - Company-specific product names
            # - Internal system names  
            # - Project codenames
        ]
        
        # Add all patterns to ruler
        ruler.add_patterns(patterns + phrase_patterns)
        return ruler

    def extract_entities_with_ner(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract named entities using enhanced spaCy NER with EntityRuler.
        
        This method combines spaCy's built-in NER with custom EntityRuler patterns
        to provide better detection of organizational, technical, and sensitive terms.
        """
        entities = defaultdict(set)

        if not self.nlp:
            return entities

        # Setup custom EntityRuler if not already done
        if "entity_ruler" not in self.nlp.pipe_names:
            self._setup_custom_entity_ruler()

        # Process text in chunks to handle long documents
        max_chars = 1000000  # 1MB limit for spaCy
        if len(text) > max_chars:
            chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        else:
            chunks = [text]

        for chunk in chunks:
            try:
                doc = self.nlp(chunk)

                # Extract entities using enhanced spaCy NER with EntityRuler
                for ent in doc.ents:
                    clean_entity = ent.text.strip()

                    # Skip very short entities
                    if len(clean_entity) < 3:
                        continue

                    # Handle custom entity types from our EntityRuler patterns (high priority)
                    if ent.label_ in ['PROJECT_NAME', 'TECH_SYSTEM', 'CUSTOMER_ID', 'ENVIRONMENT', 
                                     'CAMELCASE_TECH', 'COMPOUND_SYSTEM', 'SECURITY_CLASS', 
                                     'REVENUE_MODEL', 'ORG_SPECIFIC']:
                        entities[ent.label_].add(clean_entity)
                        continue

                    # Legacy custom patterns (for backward compatibility)
                    elif ent.label_ in ['DB_CONNECTION', 'API_KEY', 'INTERNAL_DOMAIN']:
                        entities[ent.label_].add(clean_entity)
                        continue

                    # Focus on spaCy entities that indicate organizational terms
                    elif ent.label_ == 'ORG':
                        # Organizations - but filter out generic ones
                        if (not clean_entity.lower() in self.stop_words and
                            clean_entity not in {'Company', 'Corporation', 'Inc', 'LLC', 'Ltd', 'Team', 'Department', 'Group'} and
                            len(clean_entity) > 2):
                            entities['ORG'].add(clean_entity)

                    elif ent.label_ == 'PRODUCT':
                        # Products are often organization-specific systems
                        if (len(clean_entity) > 2 and
                            not any(generic in clean_entity.lower() for generic in ['product', 'service', 'solution', 'system'])):
                            entities['PRODUCT'].add(clean_entity)

                    elif ent.label_ == 'PERSON':
                        # People names - but only if they seem relevant (not common names)
                        if len(clean_entity) > 2 and ' ' in clean_entity:  # Full names are more specific
                            entities['PERSON'].add(clean_entity)

                    elif ent.label_ == 'EVENT':
                        # Events could be conferences, releases, projects
                        if len(clean_entity) > 3 and len(clean_entity.split()) <= 3:
                            entities['EVENT'].add(clean_entity)

                    elif ent.label_ == 'MONEY':
                        # Financial information
                        entities['MONEY'].add(clean_entity)

                    elif ent.label_ == 'PERCENT':
                        # Business metrics
                        entities['PERCENT'].add(clean_entity)

                # Add regex-based extraction for technical terms that spaCy/EntityRuler might miss

                # 1. Project/System names (CamelCase compounds)
                camel_pattern = r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*(?:Engine|API|Service|System|Platform|Hub|Vault|Net|Cluster|Dashboard|Analytics|Insight))\b'
                for match in re.finditer(camel_pattern, chunk):
                    term = match.group(1)
                    if len(term) > 4:
                        entities['TECH_SYSTEM'].add(term)

                # 1b. CamelCase system followed by a TitleCase word (e.g., 'DataVault Prime')
                tech_phrase_pattern = r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]+)\b"
                for match in re.finditer(tech_phrase_pattern, chunk):
                    term = match.group(1)
                    if len(term) > 6:
                        entities['TECH_SYSTEM_PHRASE'].add(term)

                # 1c. CamelCase system followed by a suffix word (e.g., 'AnalyticsHub API')
                tech_with_suffix_word = r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*\s+(?:API|Engine|Service|System|Platform|Dashboard|Console|Gateway|Server|Database|Manager|Controller|Monitor|Cluster))\b"
                for match in re.finditer(tech_with_suffix_word, chunk):
                    term = match.group(1)
                    if len(term) > 6:
                        entities['TECH_SYSTEM_PHRASE'].add(term)

                # 2. Project names with "Project" prefix
                project_pattern = r"\b(Project\s+[A-Z][a-zA-Z]+)\b"
                for match in re.finditer(project_pattern, chunk):
                    term = match.group(1)
                    entities["PROJECT_NAME"].add(term)

                # 3. Operation/Initiative names
                operation_pattern = r"\b(Operation\s+[A-Z][a-zA-Z]+)\b"
                for match in re.finditer(operation_pattern, chunk):
                    term = match.group(1)
                    entities["OPERATION_NAME"].add(term)

                # 4. Customer/Client identifiers
                customer_pattern = r"\b(Customer\s+[A-Z][a-zA-Z]*)\b"
                for match in re.finditer(customer_pattern, chunk):
                    term = match.group(1)
                    if not term.endswith(
                        ("s", "Service", "Support", "Team")
                    ):  # Avoid generic plural
                        entities["CUSTOMER_ID"].add(term)

            except Exception as e:
                logger.error(f"NER processing error: {e}")

        return entities

    def extract_technical_patterns(self, text: str) -> Dict[str, Set[str]]:
        """Extract technical terms using generic regex patterns that work with any organization's documents"""
        patterns = {
            "project_names": [
                # Generic project/operation patterns
                r"\b(Project\s+[A-Z][a-zA-Z0-9\s]{2,25})\b",
                r"\b(Operation\s+[A-Z][a-zA-Z0-9\s]{2,25})\b",
                r"\b(Initiative\s+[A-Z][a-zA-Z0-9\s]{2,25})\b",
                r"\b(Program\s+[A-Z][a-zA-Z0-9\s]{2,25})\b",
                r"\b(Campaign\s+[A-Z][a-zA-Z0-9\s]{2,25})\b",
                r"\b(Phase\s+[A-Z][a-zA-Z0-9\s]{2,15})\b",
            ],
            "system_names": [
                # Generic system naming patterns (compound words with technical suffixes)
                r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*(?:Engine|API|Service|System|Platform|Cloud|Vault|Hub|DB|Net))\b",
                # System names with spaces
                r"\b([A-Z][a-z]+\s+(?:Engine|API|Service|System|Platform|Cloud|Vault|Hub|Database|Server|Gateway))\b",
                # Technical compound names
                r"\b([A-Z][a-z]+(?:Processor|Manager|Controller|Handler|Monitor|Dashboard|Analytics|Cluster))\b",
            ],
            'code_names': [
                r'\b([A-Z][a-z]*[A-Z][a-zA-Z]*(?:Cluster|Net|System|Engine|API|Service|DB))\b',
                r'\b([A-Z]{2,}[-_][A-Z0-9]{2,})\b',
                r'\b(Code\s*[Nn]ame\s*:?\s*([A-Z][a-zA-Z0-9\s]{2,15}))\b',
            ],
            'api_services': [
                r'\b([a-zA-Z]+(?:API|Service|Engine|Manager|Controller|Handler|Processor))\b',
                r'\b([A-Z][a-zA-Z]*(?:DB|Database|Store|Cache|Vault))\b',
            ],
            'credentials_and_keys': [
                # Generic API key patterns
                r"\b([A-Z]{3,}_(?:DEV|STAGE|STAGING|PROD|PRODUCTION|API|KEY|SECRET|TOKEN)_[A-Z0-9_]+)\b",
                r"\b([A-Z]{3,}_[A-Z0-9]{4,}_[a-zA-Z0-9]{6,})\b",
                # Database connection strings (generic)
                r"((?:postgresql|mysql|mongodb|redis|clickhouse|oracle|mssql)://[a-zA-Z0-9_-]+:[^@\s]+@[a-zA-Z0-9.-]+(?::[0-9]+)?/[a-zA-Z0-9_]+)",
                # Generic secret patterns
                r"\b([A-Z0-9]{20,})\b",  # Long alphanumeric strings (often keys)
            ],
            "internal_infrastructure": [
                # Generic internal domain patterns
                r"\b([a-zA-Z0-9-]+\.(?:internal|corp|local|private|intra))\b",
                r"\b([a-zA-Z0-9-]+@[a-zA-Z0-9.-]+\.(?:internal|corp|local|private|intra))\b",
                # Generic service account patterns
                r"\b([a-zA-Z0-9-]+[-_](?:service|svc|dev|staging|prod|test|qa)@[a-zA-Z0-9.-]+)\b",
            ],
            "proprietary_terms": [
                # Generic proprietary naming patterns
                r"\b([A-Z][a-z]+(?:Algorithm|Framework|Protocol|Standard|Method|Process))\b",
                r"\b(Custom[A-Z][a-zA-Z]*)\b",
                r"\b(Proprietary[A-Z][a-zA-Z]*)\b",
                r"\b(Internal[A-Z][a-zA-Z]*)\b",
            ],
            "business_entities": [
                # Generic customer/client/partner patterns
                r"\b(Customer\s+[A-Z][a-zA-Z0-9\s]{2,20})\b",
                r"\b(Client\s+[A-Z][a-zA-Z0-9\s]{2,20})\b",
                r"\b(Partner\s+[A-Z][a-zA-Z0-9\s]{2,20})\b",
                r"\b(Vendor\s+[A-Z][a-zA-Z0-9\s]{2,20})\b",
            ],
            "technical_identifiers": [
                # Generic technical ID patterns
                r"\b([A-Z]{2,4}[-_][0-9]{3,}(?:[-_][A-Z0-9]{2,})?)\b",
                r"\b([A-Z]+[0-9]{4,}[A-Z]*)\b",
                r"\b([a-zA-Z]+_v[0-9]+(?:\.[0-9]+)*)\b",  # Version identifiers
                r"\b([A-Z]{3,}-[0-9]{3,})\b",  # Ticket/ID patterns
            ],
        }

        extracted_terms = defaultdict(set)

        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    term = match.group(1) if match.groups() else match.group(0)
                    term = term.strip()
                    if len(term) > 2 and not term.lower() in self.stop_words:
                        extracted_terms[category].add(term)

        return extracted_terms

    def extract_capitalized_sequences(self, text: str) -> Set[str]:
        """Extract sequences of capitalized words that might be proper nouns"""
        # Pattern for 2+ consecutive capitalized words
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        matches = re.findall(pattern, text)

        valid_sequences = set()
        for match in matches:
            # Filter out common phrases
            if not any(
                common in match.lower()
                for common in [
                    "the ",
                    "and ",
                    "or ",
                    "but ",
                    "in ",
                    "on ",
                    "at ",
                    "to ",
                    "for ",
                    "of ",
                    "with ",
                ]
            ):
                valid_sequences.add(match)

        return valid_sequences

    def extract_quoted_terms(self, text: str) -> Set[str]:
        """Extract terms in quotes which often indicate important concepts"""
        patterns = [
            r'"([^"]{3,30})"',  # Double quotes
            r"'([^']{3,30})'",  # Single quotes
            r"`([^`]{3,30})`",  # Backticks
        ]

        quoted_terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                clean_match = match.strip()
                if (
                    len(clean_match) > 2
                    and not clean_match.lower() in self.stop_words
                    and not clean_match.isdigit()
                ):
                    quoted_terms.add(clean_match)

        return quoted_terms

    def calculate_term_frequency_significance(
        self, terms: Dict[str, Set[str]], all_text: str
    ) -> Dict[str, TermScore]:
        """Calculate frequency and significance scores for terms with enhanced NER weighting"""
        term_scores = {}
        text_lower = all_text.lower()

        # Flatten all terms and track their extraction method
        all_terms = {}  # term -> extraction_method
        for category, category_terms in terms.items():
            for term in category_terms:
                all_terms[term] = category

        for term, extraction_method in all_terms.items():
            term_lower = term.lower()
            frequency = text_lower.count(term_lower)

            if frequency == 0:
                continue

            # Calculate significance score
            significance = 0

            # Base frequency score (normalized)
            freq_score = min(frequency / 10, 1.0)
            significance += freq_score * 2

            # Length score (longer terms are often more specific)
            length_score = min(len(term) / 20, 1.0)
            significance += length_score

            # Capitalization score
            if any(c.isupper() for c in term):
                significance += 1

            # Multi-word score
            if len(term.split()) > 1:
                significance += 1

            # Enhanced scoring based on extraction method and entity types
            if extraction_method.startswith("ner_"):
                # Prioritize organizational terms over technical infrastructure
                if "PROJECT_NAME" in extraction_method:
                    significance += 20  # Project names are top priority
                elif "OPERATION_NAME" in extraction_method:
                    significance += 18  # Operation names are very important
                elif 'TECH_SYSTEM' in extraction_method or 'TECH_SYSTEM_PHRASE' in extraction_method:
                    significance += 15  # Technical systems are highly valuable
                elif "CUSTOMER_ID" in extraction_method:
                    significance += 14  # Customer identifiers are important
                elif "ORG" in extraction_method:
                    significance += 12  # Organizations are important
                elif "PRODUCT" in extraction_method:
                    significance += 10  # Products can be organization-specific
                elif "EVENT" in extraction_method:
                    significance += 8  # Events could be internal conferences, releases
                elif "PERSON" in extraction_method:
                    significance += 6  # Person names can be sensitive
                elif "MONEY" in extraction_method:
                    significance += 8  # Financial information is sensitive
                elif "PERCENT" in extraction_method:
                    significance += 4  # Business metrics
                elif "DB_CONNECTION" in extraction_method:
                    significance += 6  # Database connections
                elif "API_KEY" in extraction_method:
                    significance += 18  # API keys
                elif "INTERNAL_DOMAIN" in extraction_method:
                    significance += 6  # Internal domains

                # Standard spaCy entities get lower scores
                elif "GPE" in extraction_method:  # Geopolitical entities
                    significance += 3
                elif "DATE" in extraction_method:
                    significance += 2
                elif "CARDINAL" in extraction_method or "ORDINAL" in extraction_method:
                    significance += 1
                elif "LAW" in extraction_method or "LANGUAGE" in extraction_method:
                    significance += 3

            elif extraction_method.startswith("pattern_"):
                # Pattern-based terms already have good scoring
                if "credentials_and_keys" in extraction_method:
                    significance += 10  # Credentials are critical
                elif "internal_infrastructure" in extraction_method:
                    significance += 8  # Internal infrastructure is important
                elif "system_names" in extraction_method:
                    significance += 7  # System names are valuable
                elif "project_names" in extraction_method:
                    significance += 7  # Project names are valuable
                elif "technical_identifiers" in extraction_method:
                    significance += 6  # Technical IDs are important
                elif "business_entities" in extraction_method:
                    significance += 5  # Business entities matter
                elif "proprietary_terms" in extraction_method:
                    significance += 6  # Proprietary terms are sensitive

            # Context analysis with enhanced sensitivity detection
            contexts = []
            sentences = self.sent_tokenize(all_text)
            for sentence in sentences:
                if term_lower in sentence.lower():
                    contexts.append(sentence.strip())

                    # Enhanced sensitivity indicators
                    sensitive_indicators = [
                        "confidential",
                        "internal",
                        "proprietary",
                        "classified",
                        "restricted",
                        "private",
                        "secret",
                        "sensitive",
                        "exclusive",
                        "do not share",
                        "nda",
                        "non-disclosure",
                        "internal use only",
                        "authorized personnel",
                        "clearance required",
                        "access control",
                        "company confidential",
                        "trade secret",
                        "proprietary information",
                    ]

                    sentence_lower = sentence.lower()
                    for indicator in sensitive_indicators:
                        if indicator in sentence_lower:
                            significance += 3  # Boost for sensitive context
                            break

                    # Boost for technical context
                    technical_indicators = [
                        "api",
                        "database",
                        "server",
                        "system",
                        "platform",
                        "infrastructure",
                        "deployment",
                        "production",
                        "staging",
                        "development",
                        "environment",
                        "credentials",
                        "authentication",
                        "authorization",
                        "security",
                    ]

                    for tech_indicator in technical_indicators:
                        if tech_indicator in sentence_lower:
                            significance += 1  # Small boost for technical context
                            break

            term_scores[term] = TermScore(
                term=term,
                frequency=frequency,
                sensitivity_score=significance,
                contexts=contexts[:3],
                extraction_method=extraction_method,
            )

        return term_scores

    def filter_and_rank_terms(
        self, term_scores: Dict[str, TermScore]
    ) -> List[TermScore]:
        """Filter and rank terms by sensitivity score - generic approach for any organization"""
        # Filter by minimum score
        filtered_terms = [
            score
            for score in term_scores.values()
            if score.sensitivity_score >= self.config.min_score
        ]

        # Comprehensive list of generic terms to filter out aggressively
        generic_terms = {
            # Basic technical terms
            "data",
            "system",
            "user",
            "file",
            "code",
            "name",
            "type",
            "value",
            "process",
            "service",
            "application",
            "method",
            "function",
            "class",
            "customer",
            "customers",
            "cust",
            "internal",
            "project",
            "security",
            "production",
            "infrastructure",
            "allocation",
            "revenue",
            "tech",
            # Additional generic terms
            "sec",
            "ratio",
            "engine",
            "dev",
            "platform",
            "manager",
            "team",
            "service",
            "contact",
            "custom",
            "develop",
            "compliance",
            "analytics",
            "engineer",
            "incident",
            "access",
            "prod",
            "information",
            "monitoring",
            "structure",
            "employee",
            "location",
            "product",
            "review",
            "format",
            "intel",
            "monitor",
            "level",
            "merge",
            "development",
            "integration",
            "document",
            "services",
            "client",
            "api",
            "classification",
            "management",
            "emergency",
            "assessment",
            "systems",
            "legal",
            "ram",
            "architect",
            "financial",
            "business",
            "analysis",
            "control",
            "clear",
            "department",
            "technical",
            "implement",
            "database",
            "projects",
            "primary",
            "update",
            "engineering",
            "clearance",
            "partner",
            "board",
            "external",
            "title",
            "network",
            "secret",
            "deploy",
            "email",
            "executive",
            "require",
            "detect",
            "phone",
            "admin",
            "test",
            "communication",
            "acquisition",
            "enterprise",
            "automated",
            "building",
            "account",
            "author",
            # Generic roles and titles
            "manager",
            "director",
            "lead",
            "senior",
            "junior",
            "principal",
            "specialist",
            "coordinator",
            "administrator",
            "analyst",
            "consultant",
        }

        # Generic patterns that indicate organization-specific terms
        organization_indicators = {
            "project_names",
            "system_names",
            "credentials_and_keys",
            "internal_infrastructure",
            "proprietary_terms",
            "business_entities",
            "technical_identifiers",
        }

        quality_filtered = []
        for term in filtered_terms:
            term_lower = term.term.lower().strip()

            # Skip very short terms or single characters
            if len(term_lower) < 3:
                continue

            # PRIORITIZE organizational terms over infrastructure
            is_organizational_term = any(
                [
                    "PROJECT_NAME" in term.extraction_method,
                    "OPERATION_NAME" in term.extraction_method,
                    "TECH_SYSTEM" in term.extraction_method,
                    "CUSTOMER_ID" in term.extraction_method,
                    term.extraction_method == "ner_ORG",
                    term.extraction_method == "ner_PRODUCT",
                    term.extraction_method == "ner_EVENT",
                ]
            )

            # DEPRIORITIZE infrastructure terms (but don't exclude them completely)
            is_infrastructure_term = any(
                [
                    "@" in term.term,  # Email addresses
                    "://" in term.term,  # Database URLs
                    "DB_CONNECTION" in term.extraction_method,
                    "API_KEY" in term.extraction_method,
                    "INTERNAL_DOMAIN" in term.extraction_method,
                ]
            )

            # Boost organizational terms significantly
            if is_organizational_term:
                term.sensitivity_score += 20

            # Reduce infrastructure terms (but keep the really important ones)
            elif is_infrastructure_term:
                if term.sensitivity_score < 15:  # Only keep high-scoring infrastructure
                    continue
                else:
                    term.sensitivity_score -= 10  # Reduce their priority

            # Skip purely generic terms unless they have extremely high scores
            if term_lower in generic_terms:
                if term.sensitivity_score > 25.0:  # Much higher threshold
                    quality_filtered.append(term)
                continue

            # Skip common single words that appear frequently
            if (
                " " not in term.term
                and term.frequency > 50
                and term.sensitivity_score < 15.0
            ):
                continue

            # Boost terms extracted by organization-specific patterns
            if any(
                indicator in term.extraction_method
                for indicator in organization_indicators
            ):
                term.sensitivity_score += 15  # Major boost for org-specific patterns

            # Boost terms with common organizational indicators (generic)
            if any(
                indicator in term.term.lower()
                for indicator in [".internal", ".corp", ".local", ".private"]
            ):
                term.sensitivity_score += 20

            # Boost credentials, API keys, database URLs, and technical identifiers
            if any(
                pattern in term.term
                for pattern in [
                    "_KEY_",
                    "_API_",
                    "://",
                    "@",
                    "_SECRET_",
                    "_TOKEN_",
                    "_PROD_",
                    "_DEV_",
                ]
            ):
                term.sensitivity_score += 25

            # Boost multi-word terms (usually more specific)
            if len(term.term.split()) > 1:
                term.sensitivity_score += 8

            # Boost terms with mixed case (likely proper nouns/systems)
            if any(c.isupper() for c in term.term[1:]) and any(
                c.islower() for c in term.term
            ):
                term.sensitivity_score += 5

            # Boost terms that contain technical patterns
            if re.search(
                r"[A-Z]{2,}[-_][A-Z0-9]+|[A-Z][a-z]+[A-Z][a-zA-Z]*(?:Engine|API|Service|System)",
                term.term,
            ):
                term.sensitivity_score += 10

            quality_filtered.append(term)

        # Remove duplicates (case-insensitive)
        seen_terms = set()
        unique_terms = []
        for term in quality_filtered:
            term_lower = term.term.lower().strip()
            if term_lower not in seen_terms and len(term_lower) > 2:
                seen_terms.add(term_lower)
                unique_terms.append(term)

        # Sort by sensitivity score (highest first)
        ranked_terms = sorted(
            unique_terms, key=lambda x: x.sensitivity_score, reverse=True
        )

        return ranked_terms[: self.config.max_terms]

    def extract_sensitive_terms_offline(self, directory_path: str) -> List[TermScore]:
        """Main method for offline sensitive term extraction"""
        logger.info(f"Processing documents from: {directory_path}")

        # Extract documents
        documents = self.extract_documents_from_directory(directory_path, verbose=True)
        logger.info(f"Found {len(documents)} documents")

        if not documents:
            return []

        # Combine all text for analysis
        all_text = "\n".join([doc["content"] for doc in documents])
        logger.info(f"Total text length: {len(all_text)} characters")

        # Extract terms using multiple methods
        all_terms = defaultdict(set)

        if "ner" in self.config.methods and self.nlp:
            logger.info("Extracting named entities...")
            ner_terms = self.extract_entities_with_ner(all_text)
            for category, terms in ner_terms.items():
                all_terms[f"ner_{category}"].update(terms)

        if "patterns" in self.config.methods:
            logger.info("Extracting technical patterns...")
            pattern_terms = self.extract_technical_patterns(all_text)
            for category, terms in pattern_terms.items():
                all_terms[f"pattern_{category}"].update(terms)

        if "capitalization" in self.config.methods:
            logger.info("Extracting capitalized sequences...")
            cap_terms = self.extract_capitalized_sequences(all_text)
            all_terms["capitalized_sequences"].update(cap_terms)

        if "quotes" in self.config.methods:
            logger.info("Extracting quoted terms...")
            quoted_terms = self.extract_quoted_terms(all_text)
            all_terms["quoted_terms"].update(quoted_terms)

        total_extracted = sum(len(terms) for terms in all_terms.values())
        logger.info(f"Extracted {total_extracted} terms using offline methods")

        # Calculate significance scores
        logger.info("Calculating term significance...")
        term_scores = self.calculate_term_frequency_significance(all_terms, all_text)

        # Filter and rank
        logger.info("Filtering and ranking terms...")
        ranked_terms = self.filter_and_rank_terms(term_scores)

        logger.info(f"Final filtered terms: {len(ranked_terms)}")
        return ranked_terms


class LLMEnhancedSensitiveTermExtractor(BaseSensitiveTermExtractor):
    """Enhanced extractor that can use LLM for better analysis"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[ExtractionConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        super().__init__(model_name, config)
        self.llm_config = llm_config or LLMConfig()

    async def analyze_chunk_with_llm(self, text_chunk: str) -> List[str]:
        """Use LLM to identify sensitive terms in a text chunk"""
        import aiohttp

        prompt = f"""Identify organization-specific sensitive terms that should be anonymized.
Focus on terms that would be specific to an organization or industry and should be anonymized.
These might include:
- Project names or codenames or operator names or internal system names
- Product identifiers or model numbers
- Internal systems or platforms
- Proprietary technology names, codes/identifiers
- Internal process names or methodologies
- Confidential program names or initiatives
- Internal team names or department names
- Internal tool names or software names
- Specific employee titles or roles
- Organization-specific terminology
- Pricing information or contract identifiers
- Client or partner names
- Customer/partner identifiers
- Security-related terminology specific to the organization

IMPORTANT: DO NOT include common technical terms and technologies that are widely known in the industry.

Only return the list of terms, one per line, with no explanations or categorization.
Focus on truly proprietary or organization-specific terminology only.

Text to analyze:
{text_chunk[:1500]}"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.llm_config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.llm_config.temperature,
                    "max_tokens": self.llm_config.max_tokens,
                    "stream": False,
                }

                async with session.post(
                    self.llm_config.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.llm_config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.llm_config.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = (
                            result.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )

                        # Handle reasoning tags
                        if content.strip().startswith("<think>"):
                            import re

                            think_match = re.search(
                                r"<think>(.*?)</think>", content, re.DOTALL
                            )
                            if think_match:
                                content = think_match.group(1)
                            else:
                                content = content.replace("<think>", "", 1)

                        # Extract terms from response
                        # TODO - improve extraction logic
                        filtered_terms = []
                        patterns = [
                            r'"([^"]{3,30})"',
                            r"Project\s+([A-Z][a-zA-Z0-9\s]{2,20})",
                            r"Operation\s+([A-Z][a-zA-Z0-9\s]{2,15})",
                            r"Customer\s+([A-Z][a-zA-Z0-9\s]{2,15})",
                            r"([A-Z][a-zA-Z]*Engine)",
                            r"([A-Z][a-zA-Z]*Dashboard)",
                            r"([A-Z]{2,}[-_][A-Z0-9]{2,})",
                        ]

                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            for match in matches:
                                clean_term = match.strip()
                                if 3 <= len(
                                    clean_term
                                ) <= 50 and not clean_term.lower() in [
                                    "project",
                                    "customer",
                                    "operation",
                                ]:
                                    filtered_terms.append(clean_term)

                        return list(set(filtered_terms))
                    else:
                        logger.error(f"LLM API error: {response.status}")

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")

        return []

    async def extract_sensitive_terms_with_llm(
        self, directory_path: str
    ) -> Dict[str, List[TermScore]]:
        """Extract terms with LLM and return separate offline/LLM results"""
        logger.info(f"Processing documents from: {directory_path}")

        # Get documents
        documents = self.extract_documents_from_directory(directory_path, verbose=True)
        if not documents:
            return {"offline": [], "llm_only": [], "combined": []}

        # Offline extraction
        logger.info("Running offline extraction...")
        offline_terms = self.extract_sensitive_terms_offline(directory_path)

        # LLM analysis
        logger.info("Running LLM analysis...")
        chunks = []
        for doc in documents[:3]:
            content = doc["content"]
            for i in range(0, len(content), 600):
                chunk = content[i : i + 1000]
                if chunk.strip():
                    chunks.append(chunk)

        all_llm_terms = []
        for i, chunk in enumerate(chunks[:5]):
            print(f"Analyzing chunk {i+1}/5")
            terms = await self.analyze_chunk_with_llm(chunk)
            all_llm_terms.extend(terms)
            await asyncio.sleep(0.5)

        # Convert LLM terms to TermScore objects
        term_counts = Counter(all_llm_terms)
        llm_term_scores = {}

        for term, frequency in term_counts.items():
            if len(term) > 2:
                base_score = frequency * 4
                quality_score = 0

                if len(term.split()) > 1:
                    quality_score += 4
                if any(
                    pattern in term for pattern in ["Project", "Operation", "Customer"]
                ):
                    quality_score += 5
                if any(c.isupper() for c in term):
                    quality_score += 1

                final_score = base_score + quality_score

                llm_term_scores[term] = TermScore(
                    term=term,
                    frequency=frequency,
                    sensitivity_score=final_score,
                    contexts=[],
                    extraction_method="llm_analysis",
                )

        llm_only_terms = [
            score
            for score in llm_term_scores.values()
            if score.sensitivity_score >= self.config.min_score
        ]
        llm_only_terms.sort(key=lambda x: x.sensitivity_score, reverse=True)
        llm_only_terms = llm_only_terms[: self.config.max_terms]

        # Create combined results
        combined_terms = {}
        for term in offline_terms:
            combined_terms[term.term] = term

        for term in llm_only_terms:
            if term.term in combined_terms:
                combined_terms[term.term].sensitivity_score += term.sensitivity_score
                combined_terms[term.term].extraction_method += " + llm_analysis"
            else:
                combined_terms[term.term] = term

        combined_list = sorted(
            combined_terms.values(), key=lambda x: x.sensitivity_score, reverse=True
        )
        combined_list = combined_list[: self.config.max_terms]

        return {
            "offline": offline_terms,
            "llm_only": llm_only_terms,
            "combined": combined_list,
        }
