# Virtual Environments Explained

## What is a virtual environment?

A virtual environment is an isolated Python environment that contains:
- Its own Python interpreter
- All packages required by your project (e.g., PyTorch, NumPy, Pandas)
- Specific package versions

## Why use a virtual environment?

Without isolation, different projects can conflict:
- Project A needs PyTorch 2.0.0, Project B needs PyTorch 1.9.0
- Global installs become messy and hard to manage

With a virtual environment, each project keeps its own dependencies:
```
ProjectA/.venv/ → torch 2.0.0, numpy 1.24.0, ...
ProjectB/.venv/ → torch 1.9.0, numpy 1.20.0, ...
YourProject/.venv/ → torch 2.0.0, numpy, pandas, ...
```

Benefits:
- Version isolation
- Clean system (no global pollution)
- Easier to share and reproduce environments

## How to use in this project

### Create a virtual environment
```bash
python -m venv .venv
```

### Activate it
```bash
# Windows cmd
.venv\Scripts\activate.bat
```
After activation:
- You will see `(.venv)` in your prompt
- `pip install ...` will install into this environment

### Install dependencies
```bash
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
```

### Run the project
```bash
python src/main.py
```

### Deactivate
```bash
deactivate
```

## Exporting dependencies (optional)
```bash
pip freeze > requirements.txt
```
Others can then run:
```bash
pip install -r requirements.txt
```

## Summary of benefits
- Isolation between projects
- Reproducibility
- Clean and maintainable setups
- Flexible version control per project

## Project structure reference
```
Qi_Intra_InterChrInteraction/
├── .venv/              # The virtual environment
├── data/               # Input data
├── results/            # Outputs
├── src/                # Source code
└── ...
```


