## Summary

<!-- Describe the changes in 1-3 sentences. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactor (code change that neither fixes a bug nor adds a feature)
- [ ] Documentation update
- [ ] CI / DevEx improvement

## Checklist

- [ ] `ruff check .` passes with no errors
- [ ] `pytest tests/ -k "not integration"` passes
- [ ] Coverage has not decreased (`pytest --cov=tools --cov-report=term-missing`)
- [ ] I have updated documentation if needed
- [ ] I have added tests that prove my fix / feature works
