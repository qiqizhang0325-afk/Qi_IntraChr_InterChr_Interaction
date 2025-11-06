# What is `uv`?

## Overview

`uv` is a fast Python package manager developed by Astral (the creators of `ruff`).

### Key features

1. Very fast (often 10–100x faster than `pip`)
2. Modern: supports `pyproject.toml`, manages virtual environments
3. Largely compatible with `pip` usage
4. Smarter dependency resolution

### In this project

We use `uv` to:
- Install dependencies (PyTorch, NumPy, Pandas, etc.)
- Manage the virtual environment
- Run commands (e.g., `uv run python src/main.py`)

## If you saw an error while installing `uv`

If `pip` reported no matching distribution for `uv`, typical causes are:
- Outdated `pip`
- Network issues
- Platform-specific builds unavailable

## Solutions

### Option 1 (recommended): Upgrade Python/pip and install `uv`

1) Install Python 3.10 or newer: https://www.python.org/downloads/
2) Restart your terminal
3) Verify: `python --version`
4) Install uv: `pip install uv`

### Option 2: Skip `uv` and just use `pip` (simplest)

```bash
python -m venv .venv
.venv\Scripts\activate.bat   # Windows cmd
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
python src/main.py
```

### Option 3: Use conda (if preferred)

```bash
conda create -n qi_env python=3.10
conda activate qi_env
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
python src/main.py
```

## Recommendation

Using `pip` is fully supported and easiest for most users. `uv` is optional and mainly improves speed.


