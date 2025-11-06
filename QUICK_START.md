# Quick Start

> 📚 **For detailed documentation, see the [Wiki](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki)**

## Minimal Setup

```bash
# 1. Clone and install
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
uv sync  # or: pip install torch numpy pandas matplotlib seaborn scipy scikit-learn psutil

# 2. Prepare data (place VCF or PED/MAP in data/)
# VCF: data/test.vcf
# PED/MAP: data/test_ped.ped + data/test_ped.map

# 3. Run
uv run python src/main.py  # or: python src/main.py
```

## Next Steps

- **Installation details:** [Wiki: Installation Guide](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Installation)
- **Data preparation:** [Wiki: Data Preparation](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Data-Preparation)
- **Configuration:** [Wiki: Configuration](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Configuration)
- **Understanding results:** [Wiki: Output Files](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Output-Files)
- **Troubleshooting:** [Wiki: Troubleshooting](https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction/wiki/Troubleshooting)
