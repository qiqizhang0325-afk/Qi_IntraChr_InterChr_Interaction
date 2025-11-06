# Data Directory

Place your VCF files in this directory.

## Test Data

A test VCF file (`test.vcf`) is included in this repository. You can use it to test the analysis pipeline:

```bash
# The test file is already in data/ directory
# Just run the analysis

# Using uv:
uv run python src/main.py

# Using pip (after activating virtual environment):
python src/main.py
```

## Default File

The default VCF file name is `test.vcf`. 

If you want to use a different file, modify the path in `src/main.py`:

```python
vcf_path = os.path.join(data_dir, 'your_file.vcf')
```

## VCF File Format

The VCF file should follow the standard VCF format with:
- Header line starting with `#CHROM`
- Genotype data in columns 10 onwards
- Supported formats:
  - Integer format: directly 0/1/2 (homozygous reference/heterozygous/homozygous variant)
  - Float format (genotype dosage): will be automatically converted to 0/1/2

## Your Own Data

When you add your own VCF files to this directory, they will be ignored by git (to avoid uploading large files). Only the test file `test.vcf` is tracked in the repository.

