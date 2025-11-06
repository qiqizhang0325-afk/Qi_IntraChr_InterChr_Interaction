# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-01-XX

### Added

- Modular code structure: Split monolithic script into separate modules
  - `data_processor.py`: VCF file parsing and phenotype simulation
  - `dataset.py`: PyTorch Dataset implementation
  - `model_components.py`: BiMambaBlock and PositionalEncoding classes
  - `models.py`: IntraChrModel and InterChrModel classes
  - `training.py`: Training functions and result integration
  - `main.py`: Main execution script
- Comprehensive README.md with installation and usage instructions
- Updated project dependencies in pyproject.toml
- GitHub Actions CI/CD pipeline using uv package manager

### Changed

- Refactored code to follow modern Python project structure
- Updated project name to `qi-intra-inter-chr`
- Improved code organization and documentation

## [1.1.1] - 2025-10-21

### Fixed

- Properly formatted `CITATION.cff` file for Zenodo syncing

## [1.1.0] - 2025-10-21

### Added

- Minimal CI/CD pipeline

### Changed

- Replaced package manager `hatch` with `uv`
- Expanded `README`

## [1.0.1] - 2024-07-08

### Added

- Added Zenodo DOI cross-links

## [1.0.0] - 2024-06-28

### Added

- Initial release of repository