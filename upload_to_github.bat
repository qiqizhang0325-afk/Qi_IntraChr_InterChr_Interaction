@echo off
echo ========================================
echo GitHub Upload Script
echo ========================================
echo.

echo Step 1: Adding all files...
git add .
echo.

echo Step 2: Committing changes...
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"
echo.

echo Step 3: Setting branch to main...
git branch -M main
echo.

echo Step 4: Adding remote repository...
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git 2>nul
if errorlevel 1 (
    echo Remote already exists, updating...
    git remote set-url origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
)
echo.

echo Step 5: Pushing to GitHub...
echo NOTE: You may need to enter your GitHub username and Personal Access Token
echo.
git push -u origin main
echo.

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Push failed!
    echo ========================================
    echo.
    echo Possible reasons:
    echo 1. Authentication failed - you need a Personal Access Token
    echo 2. Repository doesn't exist on GitHub - create it first
    echo 3. Network issues
    echo.
    echo See GITHUB_UPLOAD_GUIDE.md for detailed instructions.
) else (
    echo.
    echo ========================================
    echo SUCCESS! Code uploaded to GitHub
    echo ========================================
    echo.
    echo Visit: https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction
)

pause

