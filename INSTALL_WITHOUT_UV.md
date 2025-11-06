# Installation Without `uv`

> 📚 **For detailed installation instructions, see the [Wiki: Installation Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Installation)**

## Quick Install with `pip`

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
.venv\Scripts\activate.bat  # Windows cmd
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn psutil
```

## Notes

- Ensure Python >= 3.12
- PyTorch installation may take several minutes
- GPU support: Install PyTorch with CUDA from [pytorch.org](https://pytorch.org)

## Alternative: Using `uv` (Faster)

See [Wiki: Installation Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Installation) for `uv` installation method.
