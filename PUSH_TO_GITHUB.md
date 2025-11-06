# Push to GitHub — Final Step

## ✅ Prerequisites

1. Git repository initialized
2. Files added
3. Changes committed
4. Branch set to `main`
5. Remote added

## 🔐 Final step: push to GitHub

GitHub requires authentication. Follow these steps:

### Step 1: Create a Personal Access Token

1) Go to: https://github.com/settings/tokens
2) Click "Generate new token" → "Generate new token (classic)"
3) Fill in:
   - Note: `Qi_Intra_InterChrInteraction`
   - Expiration: choose a duration (e.g., 90 days)
   - Scopes: select `repo`
4) Click "Generate token"
5) Copy the token immediately (it is shown only once)

### Step 2: Push

From the project root:
```bash
git push -u origin main
```
When prompted:
- Username: `qiqizhang0325-afk`
- Password: paste your Personal Access Token (not your GitHub password)

### Step 3: Verify

Visit:
https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction

You should see all files.

## Troubleshooting

### Auth failed
- Use a Personal Access Token, not your password
- Ensure the token has `repo` scope and is not expired

### Repository does not exist
- Create it first: https://github.com/new
- Name: `Qi_Intra_InterChrInteraction`
- Public or Private
- Do NOT initialize with README/.gitignore/LICENSE (you already have them)

### Using SSH (optional)
```bash
git remote set-url origin git@github.com:qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
git push -u origin main
```

## Future updates
```bash
git add .
git commit -m "Description of changes"
git push
```


