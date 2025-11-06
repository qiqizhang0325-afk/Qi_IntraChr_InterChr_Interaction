# Contributing to QI Intra/Inter Chromosome Interaction Analysis

Thank you for your interest in contributing to this project! Please take a moment to
read this document to understand how you can contribute.

## Code of Conduct

Please note we have a [Code of Conduct](CODE_OF_CONDUCT.md),
please follow it in all your interactions with the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion, please:
1. Check if the issue already exists
2. Create a new issue with a clear description
3. Include steps to reproduce (for bugs)

### Contributing Code

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/Qi_Intra_InterChrInteraction
   cd Qi_Intra_InterChrInteraction
   ```

3. **Install the project**:
   
   **Using uv:**
   ```bash
   uv sync
   ```
   
   **Using pip:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate.bat    # Windows cmd
   # or: .venv\Scripts\Activate.ps1  # Windows PowerShell
   # or: source .venv/bin/activate   # Linux/macOS
   pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**:
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation if needed

6. **Run code quality checks**:
   ```bash
   uv run ruff check src/
   uv run ruff format src/
   uv run pytest
   ```

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request** on GitHub

## Code Style

- We use **ruff** for code formatting and linting
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes

## Testing

Please ensure all tests pass before submitting:
```bash
uv run pytest
```

## Questions?

If you have questions, please open an issue or contact the maintainers.