# Contributing to Company Growth Analyzer

Thank you for considering contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please create an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternative solutions considered
- How this benefits other users

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Include type hints where appropriate
   - Add comments explaining complex logic
4. **Test your changes**
   - Run existing tests: `pytest tests/`
   - Add new tests for new functionality
   - Ensure code passes linting: `flake8 src/`
5. **Commit your changes** (`git commit -m 'Add AmazingFeature'`)
   - Use clear, descriptive commit messages
   - Reference issue numbers if applicable
6. **Push to your branch** (`git push origin feature/AmazingFeature`)
7. **Open a Pull Request**
   - Describe your changes in detail
   - Link to related issues
   - Include screenshots/examples if relevant

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/company-growth-analyzer.git
cd company-growth-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ main.py

# Lint code
flake8 src/ main.py
```

## Code Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Comments**: Explain *why*, not *what*. Include critical questions and assumptions
- **Naming**: Descriptive names, avoid abbreviations

### Example

```python
def calculate_growth_rate(
    revenue_current: float,
    revenue_previous: float,
    periods: int = 1
) -> float:
    \"\"\"
    Calculate period-over-period revenue growth rate.

    Args:
        revenue_current: Current period revenue
        revenue_previous: Previous period revenue
        periods: Number of periods (default: 1)

    Returns:
        Growth rate as decimal (e.g., 0.15 for 15% growth)

    Raises:
        ValueError: If revenue_previous is zero

    Note:
        Growth rate can be misleading during turnarounds.
        A company going from -$100M to -$50M loss shows
        50% "growth" but is still unprofitable.
    \"\"\"
    if revenue_previous == 0:
        raise ValueError("Previous revenue cannot be zero")

    return (revenue_current - revenue_previous) / revenue_previous
```

## Testing

- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test component interactions
- **Mock External APIs**: Use fixtures/mocks for API calls
- **Edge Cases**: Test boundary conditions and error handling

Example test:

```python
def test_calculate_growth_rate():
    assert calculate_growth_rate(110, 100) == 0.10
    assert calculate_growth_rate(90, 100) == -0.10

    with pytest.raises(ValueError):
        calculate_growth_rate(100, 0)
```

## Adding New Data Sources

To add a new data source:

1. **Create collector class** inheriting from `BaseDataCollector`
2. **Implement `collect()` method**
3. **Add caching** using `_get_cached()` and `_set_cache()`
4. **Handle errors** gracefully with logging
5. **Update `DataAggregator`** to integrate new source
6. **Add tests** for the new collector
7. **Update documentation** in README.md

## Documentation

- Update README.md for significant changes
- Add entries to CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)
- Include inline code comments for complex logic
- Update docstrings when modifying function signatures

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Be respectful and constructive

Thank you for contributing!
