# Contributing to Medical Insurance Cost Prediction

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Found a bug? Let us know!
- **Feature Requests**: Have an idea for improvement?
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Help improve our documentation
- **Testing**: Add or improve test cases

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/medical-insurance-prediction.git
   cd medical-insurance-prediction
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

We follow PEP 8 style guidelines. Please ensure your code:

- Uses 4 spaces for indentation
- Has line lengths ≤ 88 characters
- Includes proper docstrings for functions and classes
- Uses meaningful variable and function names

### Code Formatting

We use `black` for code formatting:

```bash
pip install black
black .
```

### Linting

We use `flake8` for linting:

```bash
pip install flake8
flake8 src/ app/ tests/
```

### Type Hints

Please use type hints where appropriate:

```python
def predict_cost(age: int, bmi: float, smoker: bool) -> float:
    """Predict insurance cost based on input parameters."""
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov=app

# Run specific test file
python -m pytest tests/test_models.py
```

### Writing Tests

- Write tests for all new functions
- Use descriptive test names
- Include edge cases
- Mock external dependencies

Example test:

```python
def test_predict_cost_valid_input():
    """Test prediction with valid input parameters."""
    predictor = InsurancePredictor()
    result = predictor.predict({
        'age': 30,
        'bmi': 25.0,
        'smoker': 'no',
        'sex': 'male',
        'children': 1,
        'region': 'northeast'
    })
    assert isinstance(result, float)
    assert result > 0
```

## 📁 Project Structure

Understanding the project structure:

```
medical-insurance-prediction/
├── app/                    # Streamlit application
│   ├── main.py            # Main app file
│   ├── pages/             # Multi-page app components
│   └── utils/             # Utility functions
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training/prediction
│   └── visualization/     # Plotting functions
├── data/                  # Data files
│   ├── raw/               # Original data
│   └── processed/         # Processed data
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test files
└── docs/                  # Documentation
```

## 🔄 Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run the test suite** and ensure all tests pass
4. **Update CHANGELOG.md** with your changes
5. **Ensure code follows style guidelines**

### Pull Request Template

When submitting a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Approval** and merge

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if bug is fixed
3. **Gather relevant information** (OS, Python version, etc.)

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9.0]
- Package versions: [relevant packages]

**Screenshots**
If applicable, add screenshots

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've considered

**Additional Context**
Any other relevant information
```

## 📚 Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples where helpful
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting

### Building Documentation

If using Sphinx or similar:

```bash
cd docs/
make html
```

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📝 Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add feature: BMI category classification"
git commit -m "Fix: Handle missing values in age column"
git commit -m "Docs: Update installation instructions"

# Bad
git commit -m "Update"
git commit -m "Fix bug"
git commit -m "Changes"
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## 🔒 Security

### Reporting Security Issues

Please **DO NOT** report security vulnerabilities in public issues.

Instead:
1. Email security concerns to [security@yourproject.com]
2. Include detailed description
3. Provide steps to reproduce
4. Allow time for fix before public disclosure

### Security Guidelines

- Never commit sensitive data (API keys, passwords)
- Use environment variables for configuration
- Validate all user inputs
- Keep dependencies updated

## 🌟 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **README.md** acknowledgments section

## 📞 Getting Help

Need help contributing?

- **GitHub Discussions**: Ask questions
- **Issues**: Report problems
- **Email**: [maintainer@yourproject.com]
- **Discord/Slack**: [Community link if available]

## 📋 Development Setup

### Advanced Setup

For advanced development:

```bash
# Install development dependencies
pip install -e .
pip install pre-commit black flake8 pytest pytest-cov

# Setup pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### Environment Variables

Create `.env` file for development:

```bash
# Development settings
DEBUG=True
LOG_LEVEL=DEBUG
MODEL_PATH=models/
DATA_PATH=data/
```

### Database Setup (if applicable)

```bash
# Setup local database
python scripts/setup_db.py

# Run migrations
python scripts/migrate.py
```

## 🎯 Roadmap

Current development priorities:

1. **Model Improvements**
   - Add more sophisticated models
   - Implement ensemble methods
   - Add model interpretability features

2. **UI/UX Enhancements**
   - Improve mobile responsiveness
   - Add data visualization features
   - Implement user feedback system

3. **Performance Optimization**
   - Optimize model loading
   - Implement caching strategies
   - Add performance monitoring

4. **Testing & Quality**
   - Increase test coverage
   - Add integration tests
   - Implement automated testing

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [conduct@yourproject.com]. All complaints will be reviewed and investigated promptly and fairly.

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to Medical Insurance Cost Prediction! 🎉