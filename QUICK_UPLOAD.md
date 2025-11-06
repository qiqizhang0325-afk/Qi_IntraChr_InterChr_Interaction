# Quick Upload to GitHub

## Option 1: Use the batch script (simplest)

If you have an `upload_to_github.bat` script, double-click it to upload.

## Option 2: Run commands manually

From the project root:

```bash
# 1) Add files
git add .

# 2) Commit
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"

# 3) Set main branch
git branch -M main

# 4) Add remote
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git

# 5) Push
git push -u origin main
```

## Authentication

GitHub no longer accepts passwords — use a Personal Access Token or SSH.

### Personal Access Token (recommended)
1) https://github.com/settings/tokens → Generate new token (classic)
2) Select `repo` scope
3) When pushing:
   - Username: `qiqizhang0325-afk`
   - Password: paste your token

### SSH (optional)
```bash
git remote set-url origin git@github.com:qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
```

## Verify

After pushing, visit:
https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction

You should see all files.

