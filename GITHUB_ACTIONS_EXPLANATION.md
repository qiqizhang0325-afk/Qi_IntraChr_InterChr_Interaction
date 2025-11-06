# GitHub Actions Workflow Explanation

## What is `.github/workflows/`?

This directory contains GitHub Actions configuration for automated CI (tests and checks).

## Current workflow

The file `.github/workflows/cicd.yml` runs on:
- Pushes to `main`
- Pull requests targeting `main`

It performs:
1) Checkout code
2) Set up Python 3.12
3) Install `uv`
4) Install dependencies (`uv sync`)
5) Run tests (`uv run pytest`)

## Do you need it?

Keep it if you:
- Plan to host on GitHub (adds trust with a green check)
- Collaborate with others
- Want automatic testing on every push/PR
- Follow modern Python best practices

You can remove it if you:
- Only use the project locally and do not push to GitHub
- Do not have tests or do not want automation

## Recommendation

Keep it. It’s free on GitHub, uses their infrastructure, and matches best practices.

## How to check results

After pushing, open the "Actions" tab in your GitHub repository to see workflow runs and status.

