# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### spaCy Model Not Found
```
OSError: [E050] Can't find model 'en_core_web_sm'
```
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### NLTK Data Missing
```
LookupError: Resource punkt_tab not found
```
**Solution:**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

#### Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution:**
- Reduce `max_terms` in configuration
- Process documents in smaller batches
- Increase system memory

### Runtime Issues

#### No Terms Extracted
**Symptoms:** `Extracted 0 terms` or very few terms found

**Solutions:**
1. **Check document quality:** Ensure documents contain organization-specific content
2. **Lower min_score:** Use `--min-score 1.0` for more sensitive detection
3. **Verify document format:** Ensure files are readable text format
4. **Check file permissions:** Ensure read access to document directory

#### Poor Anonymization Quality
**Symptoms:** Important terms not redacted or too many false positives

**Solutions:**
1. **Adjust sensitivity:** Modify `min_score` threshold
2. **Refine extraction methods:** Disable problematic methods in config
3. **Custom patterns:** Add domain-specific regex patterns
4. **Manual curation:** Review and edit extracted terms list

#### LLM Integration Issues
```
aiohttp.ClientError: Cannot connect to host
```
**Solutions:**
1. **Check LLM service:** Ensure Ollama/LLM server is running
2. **Verify endpoint:** Confirm correct URL and port
3. **Test connectivity:** `curl http://localhost:11434/api/tags`
4. **Check API key:** Ensure valid authentication

### Performance Issues

#### Slow Processing
**Solutions:**
1. **Reduce document size:** Process smaller file sets
2. **Optimize methods:** Disable unused extraction methods
3. **Use faster models:** Switch to lighter spaCy models
4. **Increase resources:** More CPU/memory

#### High Memory Usage
**Solutions:**
1. **Batch processing:** Process documents in smaller chunks
2. **Reduce max_terms:** Lower the maximum terms limit
3. **Clear cache:** Restart application periodically

### Configuration Issues

#### Config File Not Found
```
FileNotFoundError: config.yml not found
```
**Solution:**
```bash
sensitive-term-config init  # Create default config
```

#### Invalid Configuration
```
yaml.scanner.ScannerError: mapping values are not allowed here
```
**Solution:**
1. **Validate YAML:** Use online YAML validator
2. **Check indentation:** Ensure consistent spacing
3. **Reset config:** `sensitive-term-config init --force`

## Getting Help

1. **Check logs:** Enable verbose mode with `--verbose`
2. **Test with demo:** Use provided demo documents first
3. **Minimal example:** Test with simple single-file input
4. **Version check:** Ensure you're using the latest version
5. **Environment:** Try in a fresh virtual environment

## Debugging Steps

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Components
```python
# Test extractor only
from sensitive_term_guard.extractors import BaseSensitiveTermExtractor
extractor = BaseSensitiveTermExtractor()
terms = extractor.extract_sensitive_terms_offline("single_file.txt")

# Test scanner only
from sensitive_term_guard.scanners import DomainSensitiveScanner
scanner = DomainSensitiveScanner(terms_list=["test", "example"])
result = scanner.scan("This is a test example.")
```

### Performance Profiling
```bash
# Time extraction
time sensitive-term-extract documents/ --output terms.txt

# Memory usage
/usr/bin/time -v sensitive-term-extract documents/ --output terms.txt
```
