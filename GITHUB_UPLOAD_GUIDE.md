# GitHub Upload — Complete Guide

## Quick Start (recommended)

### Method 1: Command line (recommended)

Run these from the project root:

```bash
# 1) Initialize Git
git init

# 2) Add all files (respects .gitignore)
git add .

# 3) Commit
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"

# 4) Add remote
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git

# 5) Set main branch
git branch -M main

# 6) Push
git push -u origin main
```

### Method 2: GitHub Desktop (GUI)

1) Install [GitHub Desktop](https://desktop.github.com/)
2) File → Add Local Repository → choose your project folder
3) Click "Publish repository"
4) Repository name: `Qi_Intra_InterChrInteraction`
5) Account: `qiqizhang0325-afk`
6) Click "Publish Repository"

## Explanations

### `.gitignore`
- Prevents committing cache, venv, large data outputs, etc.
- The provided `.gitignore` already ignores `data/*` and `results/*` (except small keepers), `__pycache__/`, `.venv/`, etc.

### `README.md`
- Main documentation. Update as needed (author, repo URL, examples).

### `CHANGELOG.md`
- Records version changes. Update when you release versions.

### `CITATION.cff`
- Software citation metadata. Update author, repo URL, and version.

### `LICENSE`
- Defines usage permissions. Currently Unlicense.

### `pyproject.toml`
- Central config for metadata, dependencies, and tooling.

### `.github/workflows/cicd.yml`
- CI pipeline using `uv`. Optional but recommended.

## Authentication

If prompted for credentials:
- Use a Personal Access Token (classic) with `repo` scope (not your password)
- Or set up SSH and use the SSH remote URL

## Troubleshooting

- If the remote already has commits, either merge:
  ```bash
  git pull origin main --allow-unrelated-histories
  git push -u origin main
  ```
  or force push:
  ```bash
  git push -u origin main --force
  ```

## Future updates
```bash
git status
git add .
git commit -m "Describe your changes"
git push
```
