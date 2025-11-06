# QI Intra/Inter Chromosome Interaction Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17404819.svg)](https://doi.org/10.5281/zenodo.17404819)

> 📚 **Comprehensive documentation is available in the [Wiki](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki)**

Contents
-----------------
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

This package provides a deep learning framework for analyzing intra-chromosome and inter-chromosome SNP (Single Nucleotide Polymorphism) interactions in genome-wide association studies (GWAS). The framework uses a combination of BiMamba (linear complexity state space models), CNN (Convolutional Neural Networks), and optional Transformer architectures to capture both local and long-range genetic interactions.

Key capabilities:
- **VCF File Processing**: Parse VCF files and extract SNP genotype data
- **PLINK PED/MAP Support**: Load PLINK text formats (PED/MAP); auto-detected if `data/test_ped.ped` and `data/test_ped.map` exist
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

## Installation

**Requirements:** Python >= 3.12

### Quick Install

**Method 1: Using `uv` (recommended - faster):**
```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
uv sync
```

**Method 2: Using `pip`:**
```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
python -m venv .venv
.venv\Scripts\activate.bat  # Windows cmd
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate   # Linux/macOS
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn psutil
```

**Note:** For detailed installation instructions, troubleshooting, and GPU setup, see [Wiki: Installation Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Installation).

## Quick Start

1. **Install dependencies** (see [Installation](#installation) above)

2. **Prepare your data:**
   - **VCF**: Place `data/test.vcf` (or update path in `src/main.py`)
   - **PLINK PED/MAP**: Place `data/test_ped.ped` and `data/test_ped.map` (auto-detected if both exist)

3. **Run the analysis:**
   ```bash
   uv run python src/main.py  # or: python src/main.py (with venv activated)
   ```

4. **View results** in the `results/` directory:
   - Text files: `main_effect_results_*.txt`, `epistatic_interactions_*.txt`, `training_history.txt`, `performance_summary.txt`
   - Visualizations: `main_effect_manhattan_*.png`, `epistatic_heatmap_*.png`, `training_curves_*.png`

### Basic Configuration

Edit `src/main.py` to customize:

```python
# Training
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3

# Model
HIDDEN_DIM = 128
INTERACTION_MODE = 'main_nonzero'  # 'auto', 'main_nonzero', or 'all'

# Phenotype
PHENOTYPE_TYPE = 'continuous'  # or 'binary'
HERITABILITY = 0.8
```

📖 **For detailed guides:** See [Wiki: Quick Start](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Quick-Start) and [Wiki: Configuration](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Configuration)

## Documentation

📚 **Comprehensive documentation is available in the [Wiki](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki):**

### Getting Started
- [Installation Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Installation) - Step-by-step installation
- [Quick Start Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Quick-Start) - Get started in 5 minutes
- [Data Preparation](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Data-Preparation) - VCF and PED/MAP formats

### Usage
- [Configuration](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Configuration) - All configuration options
- [Model Architecture](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Model-Architecture) - Model design details
- [Output Files](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Output-Files) - Understanding results
- [Performance Monitoring](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Performance-Monitoring) - Runtime and memory

### Reference
- [Troubleshooting](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Troubleshooting) - Common issues
- [API Reference](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/API-Reference) - Programmatic usage
- [Contributing](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Contributing) - How to contribute
- [Code Structure](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Code-Structure) - Project organization

## Project Structure

```
.
├── data/              # Input data (VCF or PED/MAP)
├── results/           # Analysis outputs
├── src/               # Source code
│   ├── __init__.py
│   ├── data_processor.py
│   ├── dataset.py
│   ├── model_components.py
│   ├── intra_chr_model.py
│   ├── inter_chr_model.py
│   ├── models.py
│   ├── training.py
│   └── main.py
└── tests/             # Unit tests
```

📖 **Detailed structure:** See [Wiki: Code Structure](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Code-Structure)

## Citation

If you use this software in your research, please cite:

```bibtex
[Add your citation information here]
```

Citation information is also available in [CITATION.cff](CITATION.cff).

## License

This project is released under the [Unlicense](LICENSE), allowing free use without restrictions.

## For Developers

- **Running Tests:** `uv run pytest` (or `pytest`)
- **Code Quality:** Uses `ruff` for linting and formatting
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) and [Wiki: Contributing](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Contributing)
- **API Usage:** See [Wiki: API Reference](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/API-Reference)

## Links

- **GitHub Repository:** https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction
- **Wiki Documentation:** https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki
- **Issues:** https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/issues
"# Qi_Intra_InterChr_Interaction" 
