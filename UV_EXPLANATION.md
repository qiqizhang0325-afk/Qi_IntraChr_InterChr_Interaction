# What is `uv`?

## Overview

`uv` is a fast Python package manager and environment tool, similar to `pip`, but often 10–100x faster.

### Key capabilities

1. Install Python packages (as a faster drop-in for `pip`)
2. Manage virtual environments automatically
3. Run commands with the correct environment (e.g., `uv run python src/main.py`)

### In this project

We use `uv` to:
- Install project dependencies (PyTorch, NumPy, Pandas, Matplotlib, etc.)
- Manage a `.venv` environment automatically
- Run the project with `uv run`

## Important: `uv` is optional

You can use standard `pip` and a virtual environment instead. The project works fully without `uv`.

## If `pip install uv` fails

Common causes:
1. Outdated `pip`
2. Network issues
3. A specific `uv` build not available for your platform

### Solution A (recommended): Just use `pip`

```bash
# 1) Create venv
python -m venv .venv

# 2) Activate (Windows cmd)
.venv\Scripts\activate.bat

# 3) Install deps
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn

# 4) Run
python src/main.py
```

### Solution B: Upgrade `pip`, then install uv

```bash
python -m pip install --upgrade pip
pip install uv
```

### Solution C: Use the official installer

Download from the releases page: https://github.com/astral-sh/uv/releases

## Recommendation

Using `pip` is perfectly fine and simplest for most users. `uv` is a convenience for speed, not a requirement.


