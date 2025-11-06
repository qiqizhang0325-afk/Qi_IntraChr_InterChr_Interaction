# Metadata Files Explanation

This document explains the purpose of various metadata/config files and whether to keep or modify them.

## Categories

### Required / strongly recommended

#### 1) LICENSE
Purpose: Defines usage permissions for the software.

- Current: Unlicense (public domain-like, very permissive)
- Recommendation: Keep it. Projects without a license are legally "all rights reserved".
- You may switch to MIT/Apache-2.0 if preferred.

#### 2) CODE_OF_CONDUCT.md
Purpose: Community standards for participants.

- Keep if the project is public or accepts contributions
- Template is fine as-is

#### 3) CONTRIBUTING.md
Purpose: How to contribute (open PRs, run tests, style, etc.).

- Keep and customize if you accept contributions
- Otherwise optional

#### 4) .pre-commit-config.yaml
Purpose: Runs automated checks before commits (lint, format, tests).

- Already configured (ruff, pytest, YAML checks)
- Recommended to keep
- Install hooks: `pre-commit install`

### Optional / removable

#### 5) README_TEMPLATE.md
Purpose: Template only.

- Can be removed since `README.md` exists

## Summary table

| File | Keep? | Reason |
|------|-------|--------|
| LICENSE | Yes | Legal clarity |
| CODE_OF_CONDUCT.md | Yes (public) | Professionalism, community norms |
| CONTRIBUTING.md | Yes (if contributions) | Onboarding contributors |
| .pre-commit-config.yaml | Yes | Automated quality checks |
| README_TEMPLATE.md | No | Superseded by README.md |

## Action items

1) Keep `LICENSE` as-is (or change license if you prefer)
2) Keep `CODE_OF_CONDUCT.md`
3) Keep and customize `CONTRIBUTING.md`
4) Keep `.pre-commit-config.yaml`
5) Remove `README_TEMPLATE.md` if present

