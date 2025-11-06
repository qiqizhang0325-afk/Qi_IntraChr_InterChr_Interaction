# Install the `uv` Package Manager

## Windows

### Method 1: PowerShell (recommended)

1) Open PowerShell (not cmd.exe)
   - Press `Win + X` → Windows PowerShell / Terminal
   - Or search for "PowerShell" in the Start menu

2) Run the installer:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

3) Restart the terminal, or refresh PATH:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### Method 2: Install with pip
```bash
pip install uv
```

### Method 3: Install with pipx (isolated)
```bash
pip install pipx
pipx ensurepath
pipx install uv
```

### Method 4: Manual download
1) Visit: https://github.com/astral-sh/uv/releases
2) Download a Windows build (e.g., `uv-x86_64-pc-windows-msvc.zip`)
3) Extract to a directory (e.g., `C:\uv`)
4) Add that directory to PATH

## Verify installation

Open a new terminal and run:
```bash
uv --version
```
If a version is printed, the installation succeeded.

## Not using `uv`?

You can always use `pip` and `venv` instead:
```bash
python -m venv .venv
.venv\Scripts\activate.bat   # Windows cmd
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
python src/main.py
```


