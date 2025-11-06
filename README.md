# QI Intra/Inter Chromosome Interaction Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17404819.svg)](https://doi.org/10.5281/zenodo.17404819)

Contents
-----------------
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Attribution](#attribution)
- [For Developers](#for-developers)

## Overview

This package provides a deep learning framework for analyzing intra-chromosome and inter-chromosome SNP (Single Nucleotide Polymorphism) interactions in genome-wide association studies (GWAS). The framework uses a combination of BiMamba (linear complexity state space models), CNN (Convolutional Neural Networks), and optional Transformer architectures to capture both local and long-range genetic interactions.

Key capabilities:
- **VCF File Processing**: Parse VCF files and extract SNP genotype data
- **Phenotype Simulation**: Simulate continuous or binary phenotypes with configurable heritability
- **Intra-chromosome Analysis**: Model interactions within chromosomes using BiMamba + CNN
- **Inter-chromosome Analysis**: Model cross-chromosome interactions using BiMamba + Cross-attention
- **Epistatic Interaction Detection**: Identify marker-pair level epistatic interactions
- **Main Effect Detection**: Identify SNPs with significant main effects

## Features

- **Linear Complexity**: BiMamba architecture enables processing of ultra-long sequences (millions of markers) with O(n) complexity
- **Dual-Path Architecture**: Separate models for intra-chromosome and inter-chromosome interactions
- **Flexible Phenotype Types**: Supports both continuous (quantitative) and binary (case/control) traits
- **Interpretable Outputs**: Provides main effect weights and epistatic interaction scores
- **Visualization**: Generates Manhattan plots and interaction heatmaps

## System Requirements

### OS Requirements

This package has been tested on:
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

### Python Dependencies

- Python >= 3.12
- PyTorch >= 2.0.0 (with CUDA support recommended for GPU acceleration)
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0

All dependencies including exact versions are specified in the [pyproject.toml](./pyproject.toml) file.

## Installation Guide

### Method 1: Using `uv` (Recommended - Faster)

This project can use `uv` as the package manager for faster installation. To install `uv`, check out [their documentation](https://docs.astral.sh/uv/getting-started/installation/).

#### Step 1: Clone the repository

```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
```

#### Step 2: Install `uv` (if not already installed)

- **Windows PowerShell**: 
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
- **Windows cmd / Linux/macOS**: 
  ```bash
  pip install uv
  ```

#### Step 3: Install dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies (PyTorch, NumPy, Pandas, Matplotlib, etc.).

### Method 2: Using `pip` (Alternative - Works for everyone)

If you prefer not to use `uv` or encounter installation issues, you can use the traditional `pip` method:

#### Step 1: Clone the repository

```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
```

#### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

#### Step 3: Activate the virtual environment

**Windows cmd:**
```bash
.venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your command prompt.

#### Step 4: Install dependencies

```bash
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
```

This will install all required packages. PyTorch installation may take a few minutes.

**Note**: For detailed pip installation instructions, see [INSTALL_WITHOUT_UV.md](INSTALL_WITHOUT_UV.md).

### Prepare your data

Place your VCF file in the `data/` directory:

```bash
# Copy your VCF file to data/ directory
# Default filename is test.vcf
copy your_file.vcf data\test.vcf
```

Or use a different filename and update the path in `src/main.py`.

**Note**: A test file `test.vcf` is already included in the repository, so you can skip this step if you want to test with the provided data.

### (Optional) Set up pre-commit hooks

```bash
uv run pre-commit install
```

This will set up code quality checks that run automatically before each commit.

## Quick Start

### Basic Usage

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
   cd Qi_Intra_InterChrInteraction
   ```

2. **Install dependencies** (choose one method):

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

3. **Prepare your VCF file**: 
   - Place your VCF file in the `data/` directory
   - The default filename is `test.vcf`
   - A test file `test.vcf` is already included in the repository
   - You can modify the filename in `src/main.py` if needed

4. **Run the analysis**:

   **Using uv:**
   ```bash
   uv run python src/main.py
   ```

   **Using pip (after activating virtual environment):**
   ```bash
   python src/main.py
   ```

**All results will be saved to the `results/` directory.**

5. **Configure parameters** (optional): Edit `src/main.py` to adjust:
   - VCF file path (default: `data/test.vcf`)
   - Model architecture (hidden dimensions, number of layers)
   - Training parameters (batch size, epochs, learning rate)
   - Phenotype type (continuous or binary)
   - Heritability

### Example Configuration

```python
# In src/main.py
# VCF file is automatically looked for in data/ directory
# vcf_path = os.path.join(data_dir, 'test.vcf')
PHENOTYPE_TYPE = 'continuous'  # or 'binary'
HERITABILITY = 0.9
BATCH_SIZE = 4
EPOCHS = 50
HIDDEN_DIM = 128
```

### Output Files

All results are saved to the `results/` directory:

**Text Files:**
- `results/main_effect_results_top10.txt`: Top 10 SNPs with highest main effect weights
- `results/main_effect_results_all.txt`: All SNPs with main effect weights (sorted by score)
- `results/epistatic_interactions_top10.txt`: Top 10 epistatic interaction pairs
- `results/epistatic_interactions_all.txt`: All epistatic interaction pairs (sorted by score)
- `results/training_history.txt`: Training history for all models

**Visualization Files:**
- `results/training_curves_intra_chr*.png`: Training curves for intra-chromosome models
- `results/training_curves_inter_*.png`: Training curves for inter-chromosome models
- `results/main_effect_manhattan_top10.png`: Manhattan plot of Top 10 main effects
- `results/main_effect_manhattan_all.png`: Manhattan plot of all main effects
- `results/epistatic_heatmap_top10.png`: Heatmap of Top 10 epistatic interactions
- `results/epistatic_heatmap_all.png`: Heatmap of all epistatic interactions

**Note**: To see example results, run the analysis with the provided test data in `data/` directory.

## Project Structure

```
.
├── data/                    # Data directory (place your VCF files here)
│   ├── .gitkeep
│   └── test.vcf  # Example VCF file
├── results/                  # Results directory (analysis outputs saved here)
│   ├── .gitkeep
│   ├── *.txt                 # Text results (main effects, interactions, training history)
│   └── *.png                 # Visualization files (plots, heatmaps)
├── src/
│   ├── __init__.py          # Package initialization and public API
│   ├── data_processor.py    # VCF file parsing and phenotype simulation
│   ├── dataset.py           # PyTorch Dataset class for SNP data
│   ├── model_components.py  # BiMambaBlock and PositionalEncoding
│   ├── models.py            # IntraChrModel and InterChrModel
│   ├── training.py          # Training functions and result integration
│   └── main.py              # Main execution script
└── tests/                   # Unit tests
```

## Attribution

### License

This project is released under the [Unlicense](LICENSE), allowing free use without restrictions.

### Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff). You can generate a citation using:

```bash
# Using cffconvert (if installed)
cffconvert -i CITATION.cff -f bibtex
```

### Publications

[Add your publication information here when available]

### Acknowledgements

[Add acknowledgements, funding information, etc.]

## For Developers

### Running Tests

```bash
uv run pytest
```

### Code Quality

The project uses:
- **ruff**: For linting and code formatting
- **pre-commit**: For automated code quality checks

To run linting manually:

```bash
uv run ruff check src/
uv run ruff format src/
```

### Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

1. Clone the repository
2. Install development dependencies:
   - **Using uv**: `uv sync`
   - **Using pip**: `python -m venv .venv` → activate → `pip install torch numpy pandas matplotlib seaborn scipy scikit-learn`
3. Set up pre-commit hooks: `uv run pre-commit install` (or `pre-commit install` if using pip)
4. Create a feature branch
5. Make your changes
6. Run tests: `uv run pytest` (or `pytest` if using pip)
7. Run the main script: `uv run python src/main.py` (or `python src/main.py` if using pip)
8. Submit a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

### Using as a Package

You can also import and use the modules directly:

```python
from src import VCFProcessor, IntraChrModel, InterChrModel, train_model

# Or import specific modules
from src.data_processor import VCFProcessor
from src.models import IntraChrModel, InterChrModel
```
