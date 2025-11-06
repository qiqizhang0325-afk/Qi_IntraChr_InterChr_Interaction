# Project Files Explanation

This document describes the purpose of the key files in a modern Python project.

## Required files

### 1) `.gitignore`
Purpose: Tell Git which files/dirs to ignore (caches, venvs, build artifacts, large data, etc.).

What to do: Already present. Update only if you need custom ignores.

### 2) `.pre-commit-config.yaml`
Purpose: Pre-commit hooks for formatting, linting, and tests.

What to do: Already configured. Activate with `pre-commit install`.

### 3) `README.md`
Purpose: Main documentation: overview, install, usage.

What to do: Keep updated (author info, repo URL, examples).

### 4) `CHANGELOG.md`
Purpose: Version history per release (Keep a Changelog style).

What to do: Update for each release with meaningful changes.

### 5) `CITATION.cff`
Purpose: Software citation metadata for researchers.

What to do: Ensure authors, version, and repository URL are correct.

### 6) `LICENSE`
Purpose: Usage permissions and legal clarity.

What to do: Keep the current Unlicense or choose another license if desired.

## Configuration

### 7) `pyproject.toml`
Purpose: Central configuration (metadata, dependencies, build system, tool configs).

What to do: Already updated; adjust authors/metadata as needed.

### 8) `.github/workflows/cicd.yml`
Purpose: CI pipeline (install deps with `uv`, run tests).

What to do: Optional but recommended for public repos.

## Summary table

| File | Required | Status | Action |
|------|----------|--------|--------|
| `.gitignore` | Yes | Present | Usually no changes needed |
| `.pre-commit-config.yaml` | Yes | Configured | `pre-commit install` |
| `README.md` | Yes | Present | Keep current |
| `CHANGELOG.md` | Recommended | Present | Update on releases |
| `CITATION.cff` | Recommended | Present | Verify authors/URL/version |
| `LICENSE` | Yes | Present | Keep or change license |
| `pyproject.toml` | Yes | Updated | Adjust metadata if needed |
| `.github/workflows/cicd.yml` | Recommended | Present | Keep for CI |

