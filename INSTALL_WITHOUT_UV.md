# Install Without `uv` (using pip)

If you prefer not to install `uv`, you can use the standard `pip` workflow.

## Step 1: Create a virtual environment

```bash
python -m venv .venv
```

## Step 2: Activate the environment

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

After activation, you should see `(.venv)` in your prompt.

## Step 3: Install dependencies

Based on `pyproject.toml`:

```bash
pip install torch>=2.0.0 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 scipy>=1.10.0 scikit-learn>=1.3.0
```

## Step 4: Prepare data

```bash
copy your_file.vcf data\test.vcf
```

## Step 5: Run

```bash
python src/main.py
```

## Notes

- Ensure the virtual environment is activated (look for `(.venv)` in the prompt)
- PyTorch installation may take some time
- If a GPU is available, PyTorch will use it automatically

