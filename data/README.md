# Data Directory

Place your VCF files in this directory.

## Default File

The default VCF file name is `chr1_chr2_test.vcf`. 

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

