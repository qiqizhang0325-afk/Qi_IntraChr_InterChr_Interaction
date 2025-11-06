# Windows File Copy Guide

## Problem

This command failed in cmd:
```cmd
copy E:\modern_python-main\modern_python-main\data/test1.vcf data/zq_test.vcf
```

Reasons:
1) Windows cmd requires backslashes `\` (not `/`)
2) Paths with spaces should be quoted
3) Ensure the destination directory exists

## Correct commands

### Option 1: Use backslashes (recommended)
```cmd
copy E:\modern_python-main\modern_python-main\data\test1.vcf data\zq_test.vcf
```

### Option 2: Use quotes (safer)
```cmd
copy "E:\modern_python-main\modern_python-main\data\test1.vcf" "data\zq_test.vcf"
```

### Option 3: Change into the project directory first
```cmd
cd C:\Users\zhang\Qi_Intra_InterChrInteraction
copy "E:\modern_python-main\modern_python-main\data\test1.vcf" "data\zq_test.vcf"
```

## Full steps

1) Open cmd (not PowerShell)
2) Go to the project directory:
   ```cmd
   cd C:\Users\zhang\Qi_Intra_InterChrInteraction
   ```
3) Ensure the `data` directory exists:
   ```cmd
   if not exist data mkdir data
   ```
4) Copy the file:
   ```cmd
   copy "E:\modern_python-main\modern_python-main\data\test1.vcf" "data\zq_test.vcf"
   ```
5) Verify:
   ```cmd
   dir data
   ```

## Common errors

- "The syntax of the command is incorrect" → use `\` instead of `/`
- "The system cannot find the path specified" → check the source path
- "Access is denied" → ensure the destination directory exists and you have permissions

## Quick reference
```cmd
copy <SOURCE_PATH> <DEST_PATH>
copy E:\path\file.vcf C:\target\file.vcf
copy "E:\path with spaces\file.vcf" "C:\target\file.vcf"
```


