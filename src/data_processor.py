"""Data preprocessing module for VCF file parsing and phenotype simulation."""

from itertools import combinations

import numpy as np
import pandas as pd


class VCFProcessor:
    """Processor for VCF files with phenotype simulation capabilities."""

    def __init__(self, vcf_path, phenotype_path=None):
        self.vcf_path = vcf_path
        self.phenotype_path = phenotype_path
        self.snp_data = None  # Store SNP genotype data (rows: SNPs, columns: samples)
        self.snp_info = None  # Store SNP information (CHROM, POS, ID)
        self.phenotype = None  # Store phenotype data (samples: phenotype values)
        # Intra-chromosome data blocks (key: chromosome ID, value: SNP genotype matrix)
        self.intra_chr_blocks = {}
        # Pairwise chromosome combinations (e.g., (chr1, chr2))
        self.inter_chr_pairs = []
        # Inter-chromosome data blocks (combination of SNPs from two chromosomes)
        self.inter_chr_blocks = []

    def parse_vcf(self):
        """
        Parse VCF file, extract SNP genotypes (0/1/2 encoding) and SNP information
        
        Supports two genotype formats:
        1. Integer format: directly 0/1/2
        2. Float format (genotype dosage): 0.0-0.5→0, 0.5-1.5→1, 1.5-2.0→2
        """
        # Read column names from header line (memory efficient)
        header_line = None
        header_line_idx = None
        
        with open(self.vcf_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if line.startswith('#CHROM'):
                    header_line = line.strip().split('\t')
                    header_line_idx = line_idx
                    break
        
        # If header line found, use its column names; otherwise use standard names
        if header_line is not None and header_line_idx is not None:
            # Use column names from header line (including sample names)
            all_cols = header_line
            chrom_col = '#CHROM'
            # Read data (starting after header line)
            vcf_df = pd.read_csv(
                self.vcf_path, sep='\t', skiprows=header_line_idx+1,
                names=all_cols, header=None,
                low_memory=False
            )
        else:
            # Header line not found, use standard column names
            print(
                "Warning: #CHROM header line not found, will use standard VCF column names"
            )
            # Read first data line to determine number of columns
            with open(self.vcf_path, 'r') as f:
                first_data_line = None
                for line in f:
                    line_stripped = line.strip()
                    if line_stripped and not line_stripped.startswith('#'):
                        first_data_line = line_stripped
                        break
                
                if first_data_line is None:
                    raise ValueError("No data rows found in VCF file")
                n_cols = len(first_data_line.split('\t'))
            
            # Standard VCF column names (first 9 columns)
            standard_cols = [
                'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'
            ]
            # Sample column names (starting from column 10, use default names)
            n_sample_cols = n_cols - 9
            sample_cols = [f'Sample_{i+1}' for i in range(n_sample_cols)]
            all_cols = standard_cols + sample_cols
            chrom_col = 'CHROM'
            
            # Read data (comment='#' automatically skips all comment lines)
            vcf_df = pd.read_csv(
                self.vcf_path, comment='#', sep='\t', header=None,
                names=all_cols,
                low_memory=False
            )
        
        # Extract basic SNP information
        self.snp_info = vcf_df[[chrom_col, 'POS', 'ID']].rename(
            columns={chrom_col: 'CHROM'}
        )
        
        # Extract genotype columns (starting from column 10, i.e., index 9)
        # Standard VCF format: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, Sample1, ...
        gt_cols = vcf_df.columns[9:].tolist()
        
        # Read genotype data (may be float)
        gt_data = vcf_df[gt_cols].values
        
        # Handle missing values (if any)
        if pd.isna(gt_data).any():
            print(f"Warning: Missing values detected, will fill with 0")
            gt_data = np.nan_to_num(gt_data, nan=0.0)
        
        # Convert to standard 0/1/2 encoding
        # If data is float (genotype dosage), need to convert to integer encoding
        if gt_data.dtype == np.float64 or gt_data.dtype == np.float32:
            # Genotype dosage conversion: 0.0-0.5→0, 0.5-1.5→1, 1.5-2.0→2
            self.snp_data = np.round(gt_data).astype(int)
            # Limit to 0-2 range
            self.snp_data = np.clip(self.snp_data, 0, 2)
            print(
                "Detected float genotype data (genotype dosage), "
                "converted to 0/1/2 encoding"
            )
        else:
            # If already integer, convert directly
            self.snp_data = gt_data.astype(int)
            # Limit to 0-2 range
            self.snp_data = np.clip(self.snp_data, 0, 2)
        
        # Verify genotype encoding range
        unique_values = np.unique(self.snp_data)
        if not np.all(np.isin(unique_values, [0, 1, 2])):
            print(
                f"Warning: Genotype values exceed 0/1/2 range, "
                f"detected values: {unique_values}"
            )
            self.snp_data = np.clip(self.snp_data, 0, 2)
        
        n_snps, n_samples = self.snp_data.shape
        print(f"Parsing complete: {n_snps} SNPs, {n_samples} samples")
        print(f"Genotype encoding statistics:")
        homo_ref_count = np.sum(self.snp_data == 0)
        homo_ref_pct = 100 * homo_ref_count / self.snp_data.size
        print(f"  - 0 (homozygous reference): {homo_ref_count} ({homo_ref_pct:.2f}%)")
        
        het_count = np.sum(self.snp_data == 1)
        het_pct = 100 * het_count / self.snp_data.size
        print(f"  - 1 (heterozygous): {het_count} ({het_pct:.2f}%)")
        
        homo_var_count = np.sum(self.snp_data == 2)
        homo_var_pct = 100 * homo_var_count / self.snp_data.size
        print(f"  - 2 (homozygous variant): {homo_var_count} ({homo_var_pct:.2f}%)")
        
        return self.snp_info, self.snp_data

    def simulate_phenotype(
        self, heritability=0.6, phenotype_type='continuous', normalize=True
    ):
        """
        Simulate phenotype data (GWAS-like phenotype, containing main effects +
            epistatic effects)
        
        Args:
            heritability (float): Heritability, range 0-1, represents the contribution
                proportion of genetic factors to phenotype variation
            phenotype_type (str): Phenotype type
                - 'continuous': Continuous quantitative trait (e.g., height, weight,
                    blood glucose concentration)
                - 'binary': Binary qualitative trait (e.g., disease/normal,
                    presence/absence)
            normalize (bool): Whether to normalize continuous phenotype (mean=0, std=1)
        
        Returns:
            phenotype (np.ndarray): Phenotype vector
                - Continuous: float array, shape (n_samples,)
                - Binary: integer array (0 or 1), shape (n_samples,)
        """
        n_samples = self.snp_data.shape[1]
        n_snps = self.snp_data.shape[0]
        
        # 1. Simulate main effects (randomly select 5 SNPs as main effect markers)
        main_effect_snps = np.random.choice(n_snps, 5, replace=False)
        main_effect = 0.3 * self.snp_data[main_effect_snps].sum(axis=0)
        
        # 2. Simulate epistatic effects (randomly select 3 SNP pairs)
        epistatic_pairs = [
            (np.random.choice(n_snps), np.random.choice(n_snps)) for _ in range(3)
        ]
        epistatic_effect = 0.5 * sum(
            [self.snp_data[i] * self.snp_data[j] for i, j in epistatic_pairs]
        )
        
        # 3. Simulate environmental noise
        noise = np.random.normal(0, 1, n_samples)
        
        # 4. Synthesize phenotype (heritability = heritability)
        # Genetic component = main effect + epistatic effect
        genetic_component = main_effect + epistatic_effect
        # Environmental component = noise
        environmental_component = noise
        # Total phenotype = heritability × genetic component + (1-heritability) × env
        # component
        phenotype = (
            heritability * genetic_component
            + (1 - heritability) * environmental_component
        )
        
        # 5. Process output according to phenotype type
        if phenotype_type == 'continuous':
            # Continuous quantitative trait: keep original continuous values
            if normalize:
                # Normalize: mean=0, std=1
                phenotype = (phenotype - phenotype.mean()) / (phenotype.std() + 1e-8)
            self.phenotype = phenotype
            print(
                f"Phenotype simulation complete: Continuous quantitative trait "
                f"(samples: {n_samples})"
            )
            print(f"  - Phenotype mean: {phenotype.mean():.4f}")
            print(f"  - Phenotype std: {phenotype.std():.4f}")
            print(f"  - Phenotype range: [{phenotype.min():.4f}, {phenotype.max():.4f}]")
            print(f"  - Heritability: {heritability:.2%}")
        elif phenotype_type == 'binary':
            # Binary qualitative trait: convert to 0/1 (e.g., disease/normal)
            # Use median as threshold (or other thresholds can be used)
            threshold = np.median(phenotype)
            self.phenotype = (phenotype > threshold).astype(int)
            cases = self.phenotype.sum()
            controls = n_samples - cases
            print(
                f"Phenotype simulation complete: Binary qualitative trait "
                f"(cases: {cases}, controls: {controls})"
            )
            print(f"  - Case proportion: {self.phenotype.mean():.2%}")
            print(f"  - Heritability: {heritability:.2%}")
        else:
            raise ValueError(
                f"phenotype_type must be 'continuous' or 'binary', "
                f"current: {phenotype_type}"
            )
        
        return self.phenotype

    def split_intra_inter_blocks(self):
        """Split intra-chr (within chromosomes) and inter-chr (between chromosome
        pairs) data blocks"""
        # 1. Split intra-chr blocks (group by chromosome)
        chromosomes = self.snp_info['CHROM'].unique()
        for chr_id in chromosomes:
            # Extract all SNP genotypes for this chromosome
            chr_snp_idx = self.snp_info['CHROM'] == chr_id
            self.intra_chr_blocks[chr_id] = self.snp_data[chr_snp_idx]
        snps_per_block = [
            self.intra_chr_blocks[chr_id].shape[0] for chr_id in chromosomes
        ]
        print(
            f"Intra-chr blocks: {len(self.intra_chr_blocks)} chromosomes, "
            f"SNPs per block: {snps_per_block}"
        )
        
        # 2. Generate inter-chr blocks (pairwise chromosome combinations)
        self.inter_chr_pairs = list(combinations(chromosomes, 2))
        for chr1, chr2 in self.inter_chr_pairs:
            # Merge SNPs from two chromosomes (rows: SNPs, columns: samples)
            chr1_snps = self.intra_chr_blocks[chr1]
            chr2_snps = self.intra_chr_blocks[chr2]
            inter_block = np.vstack([chr1_snps, chr2_snps])
            self.inter_chr_blocks.append((chr1, chr2, inter_block))
        print(f"Inter-chr blocks: {len(self.inter_chr_blocks)} pairwise combinations")
        return self.intra_chr_blocks, self.inter_chr_blocks



