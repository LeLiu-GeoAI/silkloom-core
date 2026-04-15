name: SilkLoom Core
about: Lightweight, resilient LLM batch processing pipeline
topics:
  - python
  - llm
  - pipeline
  - workflow
  - batch-processing
  - openai

# Community

## Code of Conduct
Please treat all community members with respect and kindness.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues
- Check existing issues first
- Provide reproducible example
- Include Python version and dependencies

## Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/your-org/silkloom-core.git
cd silkloom-core
pip install -e ".[dev]"
```

## Testing

Tests will run automatically on PR via GitHub Actions. You can also run locally:

```bash
python -m pytest
# or just check imports
python -c "from silkloom_core import Pipeline; print('✓')"
```

## Release Process

1. Tag commit with semantic version: `git tag v0.2.0`
2. Push tag: `git push origin v0.2.0`
3. GitHub Actions builds and uploads to PyPI automatically
4. GitHub Release created with artifacts

## Questions?

Open an issue with the `question` label or start a discussion.
