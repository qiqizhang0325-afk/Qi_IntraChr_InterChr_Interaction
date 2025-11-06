"""Dataset class for SNP data compatible with PyTorch DataLoader."""

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SNPDataset(Dataset):
    """Dataset for SNP genotype matrices with phenotype labels."""

    def __init__(self, snp_matrix, phenotype):
        """
        Args:
            snp_matrix: SNP genotype matrix (n_snps × n_samples)
            phenotype: Phenotype vector (n_samples)
        """
        # Transpose to (n_samples × n_snps)
        self.snp_matrix = torch.tensor(snp_matrix.T, dtype=torch.float32)
        # (n_samples × 1)
        self.phenotype = torch.tensor(phenotype, dtype=torch.float32).unsqueeze(1)
        self.scaler = StandardScaler()
        # SNP feature standardization (by sample dimension)
        self.snp_matrix = torch.tensor(
            self.scaler.fit_transform(self.snp_matrix), dtype=torch.float32
        )

    def __len__(self):
        return self.snp_matrix.shape[0]

    def __getitem__(self, idx):
        return self.snp_matrix[idx], self.phenotype[idx]



