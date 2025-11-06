"""Inter-chromosome SNP interaction model."""

import numpy as np
import torch
import torch.nn as nn

try:
    from .model_components import BiMambaBlock, PositionalEncoding
except ImportError:
    from model_components import BiMambaBlock, PositionalEncoding


class InterChrModel(nn.Module):
    """
    Inter-chr Model: BiMamba (linear complexity) + Optional Transformer (fine-grained interactions)
    
    Design Philosophy (referencing claude.py):
    - BiMamba: Linear complexity O(n) for ultra-long sequences, capture long-range dependencies
    - Cross-chromosome attention: Connect features from two chromosomes
    - Optional Transformer: Provide fine-grained interaction capture on small-scale data
    - Main effect calculation: Based on encoded features that incorporate cross-chromosome information
      via cross-attention, ensuring main effects reflect inter-chromosome dependencies
    
    Advantages:
    - Linear complexity suitable for processing genome-wide data with millions of markers
    - BiMamba can efficiently encode global features of each chromosome
    - Cross-chromosome attention mechanism captures inter-chromosome interactions
    - Main effects are computed from features that have been influenced by cross-chromosome attention,
      making them more informative than independent per-chromosome calculations
    """
    def __init__(self, n_snps_chr1, n_snps_chr2, hidden_dim=128, max_epi_pairs=100, 
                 n_bimamba_layers=2, n_transformer_layers=0, n_heads=4, use_transformer=False):
        super(InterChrModel, self).__init__()
        self.n_snps_chr1 = n_snps_chr1
        self.n_snps_chr2 = n_snps_chr2
        self.hidden_dim = hidden_dim
        self.max_epi_pairs = max_epi_pairs
        self.use_transformer = use_transformer
        
        # 1. SNP feature embedding
        self.snp_embedding_chr1 = nn.Linear(1, hidden_dim)
        self.snp_embedding_chr2 = nn.Linear(1, hidden_dim)
        
        # 2. Positional encoding (separate for two chromosomes)
        self.pos_encoding_chr1 = PositionalEncoding(
            hidden_dim, dropout=0.1, max_len=n_snps_chr1 * 2
        )
        self.pos_encoding_chr2 = PositionalEncoding(
            hidden_dim, dropout=0.1, max_len=n_snps_chr2 * 2
        )
        
        # 3. BiMamba encoder: Linear complexity capture long-range dependencies (core module)
        # Each chromosome uses independent BiMamba encoder
        self.bimamba_chr1 = nn.ModuleList([
            BiMambaBlock(hidden_dim, d_state=16, dropout=0.1) 
            for _ in range(n_bimamba_layers)
        ])
        self.bimamba_chr2 = nn.ModuleList([
            BiMambaBlock(hidden_dim, d_state=16, dropout=0.1) 
            for _ in range(n_bimamba_layers)
        ])
        
        # 4. Optional Transformer encoder (for fine-grained interactions)
        if use_transformer and n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder_chr1 = nn.TransformerEncoder(
                encoder_layer, num_layers=n_transformer_layers
            )
            self.transformer_encoder_chr2 = nn.TransformerEncoder(
                encoder_layer, num_layers=n_transformer_layers
            )
        else:
            self.transformer_encoder_chr1 = None
            self.transformer_encoder_chr2 = None
        
        # 4. Cross-chromosome attention (focus on high-potential interaction SNP pairs,
        # referencing Geneformer's attention mechanism)
        # Global attention: for phenotype prediction (using global representation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=0.1
        )
        # Fine-grained cross-attention: each SNP attends to all SNPs in the other chromosome
        # This allows each SNP to incorporate cross-chromosome information based on relevance
        self.cross_attention_fine = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 5. Marker-pair level epistatic score layer (non-additive effects of cross-chromosome SNP pairs)
        self.epistatic_cross = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 5. Main effect capture layer (based on encoded features with cross-chromosome information)
        # Extract main effect features from each SNP's encoded representation
        # This will use features that have been influenced by cross-chromosome attention
        self.main_effect_per_snp_chr1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)
        )
        self.main_effect_per_snp_chr2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)
        )
        # Aggregate main effect features across all SNPs
        # First aggregate within each chromosome
        self.main_effect_aggregate_chr1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.main_effect_aggregate_chr2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Then fuse cross-chromosome main effects (concatenate and learn fusion)
        self.main_effect_fusion = nn.Sequential(
            nn.Linear(32, 16),  # 16 + 16 = 32
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Main effect weights (learnable parameters for weighting)
        self.main_weights_chr1 = nn.Parameter(torch.randn(n_snps_chr1))
        self.main_weights_chr2 = nn.Parameter(torch.randn(n_snps_chr2))
        
        # 6. Main effect + epistatic fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 7. Output layer
        self.output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input (batch × (n_snps_chr1 + n_snps_chr2))
        Returns:
            pred: Phenotype prediction value (batch × 1)
            main_weights: Main effect weights ((n_snps_chr1 + n_snps_chr2,))
            epistatic_pairs: Cross-chromosome epistatic interaction pair list
                ((chr1_idx, chr2_idx) tuples)
            epistatic_scores: Corresponding epistatic scores for interaction pairs (tensor)
        """
        # Split SNPs from two chromosomes
        x_chr1 = x[:, :self.n_snps_chr1]  # (batch × n_snps_chr1)
        x_chr2 = x[:, self.n_snps_chr1:]  # (batch × n_snps_chr2)
        
        # 1. SNP feature embedding
        # (batch × n_snps_chr1 × hidden_dim)
        x_embed_chr1 = self.snp_embedding_chr1(x_chr1.unsqueeze(-1))
        # (batch × n_snps_chr2 × hidden_dim)
        x_embed_chr2 = self.snp_embedding_chr2(x_chr2.unsqueeze(-1))
        
        # 2. Positional encoding
        x_pos_chr1 = self.pos_encoding_chr1(x_embed_chr1)
        x_pos_chr2 = self.pos_encoding_chr2(x_embed_chr2)
        
        # 3. BiMamba encoding: Linear complexity capture long-range dependencies (core)
        # Chromosome 1
        x_mamba_chr1 = x_pos_chr1
        for bimamba_layer in self.bimamba_chr1:
            x_mamba_chr1 = x_mamba_chr1 + bimamba_layer(x_mamba_chr1)  # Residual connection
        
        # Chromosome 2
        x_mamba_chr2 = x_pos_chr2
        for bimamba_layer in self.bimamba_chr2:
            x_mamba_chr2 = x_mamba_chr2 + bimamba_layer(x_mamba_chr2)
        
        # 4. Optional Transformer encoding (for fine-grained interactions)
        if self.use_transformer and self.transformer_encoder_chr1 is not None:
            chr1_feat = self.transformer_encoder_chr1(x_mamba_chr1)
            chr2_feat = self.transformer_encoder_chr2(x_mamba_chr2)
            chr1_feat = x_mamba_chr1 + chr1_feat  # Fuse BiMamba and Transformer
            chr2_feat = x_mamba_chr2 + chr2_feat
        else:
            chr1_feat = x_mamba_chr1
            chr2_feat = x_mamba_chr2
        
        # 5. Cross-chromosome attention (two levels)
        # 5a. Fine-grained cross-attention: Each SNP attends to all SNPs in the other chromosome
        # This allows each SNP to incorporate cross-chromosome information based on relevance
        # Chr1: each SNP in chr1 attends to all SNPs in chr2
        chr1_feat_with_attn, attn_weights_1to2_fine = self.cross_attention_fine(
            chr1_feat, chr2_feat, chr2_feat
        )  # (batch × n_snps_chr1 × hidden_dim), each SNP gets personalized cross-chr info
        # Chr2: each SNP in chr2 attends to all SNPs in chr1
        chr2_feat_with_attn, attn_weights_2to1_fine = self.cross_attention_fine(
            chr2_feat, chr1_feat, chr1_feat
        )  # (batch × n_snps_chr2 × hidden_dim), each SNP gets personalized cross-chr info
        
        # Residual connection: combine original features with cross-attention enhanced features
        chr1_feat_enhanced = chr1_feat + chr1_feat_with_attn  # (batch × n_snps_chr1 × hidden_dim)
        chr2_feat_enhanced = chr2_feat + chr2_feat_with_attn  # (batch × n_snps_chr2 × hidden_dim)
        
        # 5b. Global cross-attention: for phenotype prediction (using global representation)
        # Use global average pooling as query, local features as key and value
        chr1_global = chr1_feat_enhanced.mean(dim=1, keepdim=True)  # (batch × 1 × hidden_dim)
        chr2_global = chr2_feat_enhanced.mean(dim=1, keepdim=True)  # (batch × 1 × hidden_dim)
        
        # Global cross-chromosome attention: for phenotype prediction
        attn_output_chr1, attn_weights_1to2 = self.cross_attention(
            chr1_global, chr2_feat_enhanced, chr2_feat_enhanced
        )  # (batch × 1 × hidden_dim)
        attn_output_chr2, attn_weights_2to1 = self.cross_attention(
            chr2_global, chr1_feat_enhanced, chr1_feat_enhanced
        )  # (batch × 1 × hidden_dim)
        
        # 5. Main effect calculation (based on encoded features with cross-chromosome information)
        # Now each SNP's features contain personalized cross-chromosome information
        # via fine-grained attention, not just a simple broadcast
        
        # Extract main effect features from each SNP's encoded representation
        # These features now contain personalized cross-chromosome information via fine-grained attention
        # Each SNP has different cross-chr information based on its relevance to the other chromosome
        main_effect_features_chr1 = self.main_effect_per_snp_chr1(chr1_feat_enhanced)  # (batch × n_snps_chr1 × 16)
        main_effect_features_chr2 = self.main_effect_per_snp_chr2(chr2_feat_enhanced)  # (batch × n_snps_chr2 × 16)
        
        # Main effect weights (between 0-1) - learnable parameters
        main_weights_chr1 = torch.sigmoid(self.main_weights_chr1)  # (n_snps_chr1)
        main_weights_chr2 = torch.sigmoid(self.main_weights_chr2)  # (n_snps_chr2)
        
        # Weighted aggregation: use main_weights to aggregate main effect features
        # This is not simple weighted sum, but weighted aggregation that preserves information
        # Chromosome 1: weighted aggregation
        main_weights_chr1_expanded = main_weights_chr1.unsqueeze(0).unsqueeze(-1)  # (1 × n_snps_chr1 × 1)
        weighted_main_features_chr1 = main_effect_features_chr1 * main_weights_chr1_expanded
        x_main_chr1_raw = weighted_main_features_chr1.sum(dim=1)  # (batch × 16)
        x_main_chr1 = self.main_effect_aggregate_chr1(x_main_chr1_raw)  # (batch × 16)
        
        # Chromosome 2: weighted aggregation
        main_weights_chr2_expanded = main_weights_chr2.unsqueeze(0).unsqueeze(-1)  # (1 × n_snps_chr2 × 1)
        weighted_main_features_chr2 = main_effect_features_chr2 * main_weights_chr2_expanded
        x_main_chr2_raw = weighted_main_features_chr2.sum(dim=1)  # (batch × 16)
        x_main_chr2 = self.main_effect_aggregate_chr2(x_main_chr2_raw)  # (batch × 16)
        
        # Fuse cross-chromosome main effects: concatenate and learn optimal fusion
        # This allows the model to learn how to combine main effects from two chromosomes
        # rather than simple addition
        x_main_combined = torch.cat([x_main_chr1, x_main_chr2], dim=1)  # (batch × 32)
        x_main = self.main_effect_fusion(x_main_combined)  # (batch × 16)
        
        # Concatenate main weights for output
        main_weights = torch.cat([main_weights_chr1, main_weights_chr2], dim=0)
        
        # 6. Cross-chromosome marker-pair level epistatic scores
        # (sample SNP pairs with high attention weights)
        epistatic_pairs = []
        epistatic_scores = []
        
        # Sampling strategy: Prioritize SNPs with high main effect weights,
        # combined with attention weights
        k_top_chr1 = min(50, self.n_snps_chr1)
        k_top_chr2 = min(50, self.n_snps_chr2)
        
        if k_top_chr1 > 0:
            top_snps_chr1 = torch.topk(main_weights_chr1, k=k_top_chr1)[1].cpu().numpy()
        else:
            top_snps_chr1 = np.arange(self.n_snps_chr1)
        
        if k_top_chr2 > 0:
            top_snps_chr2 = torch.topk(main_weights_chr2, k=k_top_chr2)[1].cpu().numpy()
        else:
            top_snps_chr2 = np.arange(self.n_snps_chr2)
        
        n_sample_pairs = min(self.max_epi_pairs, len(top_snps_chr1) * len(top_snps_chr2))
        
        # Sample SNP pairs (combine attention weight information)
        sampled_pairs = set()
        if n_sample_pairs > 0 and len(top_snps_chr1) > 0 and len(top_snps_chr2) > 0:
            max_attempts = n_sample_pairs * 3
            for _ in range(max_attempts):
                if len(sampled_pairs) >= n_sample_pairs:
                    break
                i = np.random.choice(top_snps_chr1)
                j = np.random.choice(top_snps_chr2)
                if (i, j) not in sampled_pairs:
                    sampled_pairs.add((i, j))
        
        # Calculate marker-pair epistatic scores (using enhanced features with cross-chr attention)
        for i, j in sampled_pairs:
            # Concatenate enhanced features of two SNPs (batch × 2*hidden_dim)
            # These features contain personalized cross-chromosome information
            pair_feat = torch.cat([chr1_feat_enhanced[:, i], chr2_feat_enhanced[:, j]], dim=1)
            # Calculate epistatic score (non-additive effect)
            epi_score = torch.sigmoid(
                self.epistatic_cross(pair_feat)
            ).mean(dim=0).squeeze()  # Scalar
            i_val = i.item() if isinstance(i, torch.Tensor) else i
            j_val = j.item() if isinstance(j, torch.Tensor) else j
            epistatic_pairs.append((i_val, j_val))
            epistatic_scores.append(epi_score)
        
        # Organize epistatic results (Top-K)
        if epistatic_scores:
            # Ensure all scores are scalars, then stack into 1D tensor
            epistatic_scores_flat = []
            for score in epistatic_scores:
                if score.dim() > 0:
                    score = score.squeeze()
                if score.dim() == 0:  # Scalar
                    epistatic_scores_flat.append(score)
                else:
                    epistatic_scores_flat.append(score.item())
            
            if epistatic_scores_flat:
                # Convert to tensor and ensure 1D
                if isinstance(epistatic_scores_flat[0], torch.Tensor):
                    epistatic_scores = torch.stack(epistatic_scores_flat)
                else:
                    epistatic_scores = torch.tensor(epistatic_scores_flat, device=x.device)
                
                # Ensure 1D tensor
                if epistatic_scores.dim() == 0:
                    epistatic_scores = epistatic_scores.unsqueeze(0)
                
                # Ensure top_k does not exceed actual score count, and at least 1
                actual_len = (
                    epistatic_scores.shape[0] if epistatic_scores.dim() > 0 else 1
                )
                top_k = min(self.max_epi_pairs, actual_len)
                
                if top_k > 0 and actual_len > 0:
                    top_indices = torch.topk(epistatic_scores, k=top_k)[1]
                    epistatic_pairs = [epistatic_pairs[idx.item()] for idx in top_indices]
                    epistatic_scores = epistatic_scores[top_indices]
                else:
                    # If top_k is 0 or no scores, return empty list
                    epistatic_pairs = []
                    epistatic_scores = torch.tensor([0.0], device=x.device)
            else:
                epistatic_pairs = []
                epistatic_scores = torch.tensor([0.0], device=x.device)
        else:
            epistatic_pairs = []
            epistatic_scores = torch.tensor([0.0], device=x.device)
        
        # 7. Fuse features (for phenotype prediction)
        # Combine BiMamba's global representation and cross-chromosome attention
        fused_feat = torch.cat([
            attn_output_chr1.squeeze(1),  # Cross-chr attention output for chr1
            attn_output_chr2.squeeze(1)   # Cross-chr attention output for chr2
        ], dim=1)  # (batch × hidden_dim*2)
        x_fusion = self.fusion(fused_feat)  # (batch × 32)
        
        # 8. Phenotype prediction
        pred = torch.sigmoid(self.output(x_fusion))  # (batch × 1)
        
        return pred, main_weights, epistatic_pairs, epistatic_scores



