# Data Directory

Place your VCF or PLINK PED/MAP files in this directory.

## Test Data

VCF test data: a small `test.vcf` is included. You can use it to test the analysis pipeline:

```bash
# The test file is already in data/ directory
# Just run the analysis

# Using uv:
uv run python src/main.py

# Using pip (after activating virtual environment):
python src/main.py
```

## Default Inputs

- VCF default: `data/test.vcf`
- PED/MAP default (if both exist): `data/test_ped.ped` + `data/test_ped.map`

The program prefers PED/MAP if the default pair exists; otherwise it uses the VCF.

## Input Formats

### VCF
The VCF should follow the standard VCF format with:
- Header line starting with `#CHROM`
- Genotype data in columns 10 onwards
- Supported formats:
  - Integer format: directly 0/1/2 (homozygous reference/heterozygous/homozygous variant)
  - Float format (genotype dosage): will be automatically converted to 0/1/2

### PLINK PED/MAP (text)
- `.map`: 4 columns per SNP: `CHROM`, `SNP_ID`, genetic distance, `POS`
- `.ped`: first 6 columns (FID, IID, PID, MID, SEX, PHENO) then two allele columns per SNP
- Missing alleles like `0` are handled; genotypes are encoded as minor-allele counts (0/1/2)
- If PED phenotype (column 6) is present, it will be used; otherwise phenotype is simulated

## Your Own Data

When you add your own VCF files to this directory, they will be ignored by git (to avoid uploading large files). Only the test files `test.vcf`, `test_ped.ped`, and `test_ped.map` are tracked in the repository.

