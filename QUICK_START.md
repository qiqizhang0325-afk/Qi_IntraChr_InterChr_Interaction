# Quick Start Guide

## 1) Clone the repository

```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
```

## 2) Install dependencies (choose one)

### Option A: Using `uv` (recommended for speed)

Install uv:
- Windows PowerShell: `irm https://astral.sh/uv/install.ps1 | iex`
- Windows cmd / Linux / macOS: `pip install uv`

Install project dependencies:
```bash
uv sync
```
This will:
- Create a virtual environment
- Install all dependencies (PyTorch, NumPy, Pandas, etc.)
- Generate a `uv.lock` file

### Option B: Using `pip`

```bash
# 1) Create a virtual environment
python -m venv .venv

# 2) Activate it
# Windows cmd:
.venv\Scripts\activate.bat
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3) Install dependencies
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
```

Note: After activation you should see `(.venv)` in your prompt. PyTorch installation can take a few minutes.

See more details in [INSTALL_WITHOUT_UV.md](INSTALL_WITHOUT_UV.md).

## 3) Prepare data

Copy your VCF file into `data/`:

```bash
# Default filename is test.vcf
copy your_file.vcf data\test.vcf
```

Or use a different filename and update the path in `src/main.py` accordingly.

## 4) Run the analysis

Using uv:
```bash
uv run python src/main.py
```

Using pip (activate the venv first):
```bash
# Windows cmd
.venv\Scripts\activate.bat
python src/main.py

# Windows PowerShell
.venv\Scripts\Activate.ps1
python src/main.py

# Linux/macOS
source .venv/bin/activate
python src/main.py
```

## 5) View results

All outputs are saved in `results/`:

- Text outputs:
  - `results/main_effect_results_top10.txt` — Top 10 main-effect markers
  - `results/main_effect_results_all.txt` — All main-effect markers
  - `results/epistatic_interactions_top10.txt` — Top 10 interaction pairs
  - `results/epistatic_interactions_all.txt` — All interaction pairs
  - `results/training_history.txt` — Training history

- Plots:
  - `results/main_effect_manhattan_top10.png` — Manhattan plot (Top 10)
  - `results/main_effect_manhattan_all.png` — Manhattan plot (All)
  - `results/epistatic_heatmap_top10.png` — Interaction heatmap (Top 10)
  - `results/epistatic_heatmap_all.png` — Interaction heatmap (All)
  - `results/training_curves_*.png` — Training curves

## Optional configuration

Edit `src/main.py` if you want to change parameters:

```python
# Model config
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128

# Phenotype config
PHENOTYPE_TYPE = 'continuous'  # or 'binary'
HERITABILITY = 0.9
```

## FAQ

### Q1: `uv` command not found

Install uv:
- Windows: `irm https://astral.sh/uv/install.ps1 | iex`
- Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Q2: VCF file not found

Ensure the file exists in `data/` as `test.vcf`, or update the path in `src/main.py`.

### Q3: CUDA error

If no GPU is available, the code will use CPU automatically. To force CPU:

```python
DEVICE = torch.device("cpu")
```

### Q4: Out of memory

Try reducing `BATCH_SIZE`, `EPOCHS`, or `HIDDEN_DIM`.

## End-to-end example

```bash
# 1) Clone
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction

# 2) Install (choose one)
# uv
uv sync
# pip
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn

# 3) Prepare data
copy my_data.vcf data\test.vcf

# 4) Run
uv run python src/main.py   # or: python src/main.py (with venv activated)

# 5) Check results in results/
```

