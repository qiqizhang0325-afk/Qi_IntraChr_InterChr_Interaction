"""Models for intra-chromosome and inter-chromosome SNP interaction analysis."""

import numpy as np
import torch
import torch.nn as nn

try:
    from .model_components import BiMambaBlock, PositionalEncoding
except ImportError:
    from model_components import BiMambaBlock, PositionalEncoding

class IntraChrModel(nn.Module):
    """
    Intra-chr Model: BiMamba (linear complexity) + Deep CNN (local features) +
    Optional Transformer (fine-grained interactions)
    
    Design Philosophy (referencing claude.py and Geneformer):
    1. BiMamba: Linear complexity O(n) for ultra-long sequences (millions of
       markers), capture long-range dependencies
    2. Deep CNN: Extract local feature patterns
    3. Optional Transformer: Provide fine-grained interaction capture on
       small-scale data (or use hybrid)
    
    Advantages:
    - BiMamba's linear complexity suitable for large-scale data
    - Can process all markers on entire chromosome
    - Combine CNN and BiMamba to capture multi-level features
    """
    def __init__(self, n_snps, hidden_dim=128, max_epi_pairs=200, 
                 n_cnn_layers=4, n_bimamba_layers=2, n_transformer_layers=0, 
                 n_heads=8, use_transformer=False):
        super(IntraChrModel, self).__init__()
        self.n_snps = n_snps
        self.hidden_dim = hidden_dim
        self.max_epi_pairs = max_epi_pairs
        self.use_transformer = use_transformer  # Whether to use Transformer (optional)
        
        # 1. SNP embedding layer
        self.snp_embedding = nn.Linear(1, hidden_dim)
        
        # 2. Deep 1D-CNN: Extract local features (short-range interactions)
        cnn_layers = []
        in_channels = 1
        out_channels = hidden_dim // 2
        
        for i in range(n_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
            if i < n_cnn_layers - 1:
                out_channels = min(out_channels * 2, hidden_dim)
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_dim = out_channels
        
        # 3. Feature projection: Project CNN output to BiMamba's hidden_dim
        self.feature_projection = nn.Linear(self.cnn_output_dim, hidden_dim)
        
        # 4. Positional encoding
        self.position_embedding = PositionalEncoding(
            hidden_dim, dropout=0.1, max_len=n_snps * 2
        )
        
        # 5. BiMamba encoder: Linear complexity capture long-range dependencies
        # (core module)
        # Use multi-layer BiMamba to capture global dependencies of entire chromosome
        self.bimamba_layers = nn.ModuleList([
            BiMambaBlock(hidden_dim, d_state=16, dropout=0.1) 
            for _ in range(n_bimamba_layers)
        ])
        
        # 6. Optional Transformer encoder (for fine-grained interactions,
        # suitable for small-scale data)
        if use_transformer and n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_transformer_layers
            )
        else:
            self.transformer_encoder = None
        
        # 5. Main effect capture layer
        self.main_effect = nn.Sequential(
            nn.Linear(n_snps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 16)
        )
        
        # 6. Marker-pair level epistatic score layer (capture non-additive effects)
        # Input: concatenated features of two SNPs
        self.epistatic_pair = nn.Linear(hidden_dim * 2, 1)
        
        # 7. Epistatic capture layer (fuse CNN + Transformer features + main effects)
        # Input dimension: cnn_output_dim + hidden_dim + 16
        self.epistatic = nn.Sequential(
            nn.Linear(self.cnn_output_dim + hidden_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 8. Output layer (phenotype prediction + main effect weights +
        # epistatic scores)
        self.output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Phenotype prediction (use sigmoid activation for binary classification)
            nn.Linear(16, 1)
        )
        # Main effect weights (interpretable output)
        self.main_effect_weights = nn.Parameter(torch.randn(n_snps))

    def forward(self, x):
        """
        Args:
            x: Input (batch × n_snps)
        Returns:
            pred: Phenotype prediction value (batch × 1)
            main_weights: Main effect weights (n_snps)
            epistatic_pairs: Epistatic interaction pair list ((i,j) index tuples)
            epistatic_scores: Corresponding epistatic scores for interaction pairs (tensor)
        """
        # 1. SNP embedding
        x_embed = self.snp_embedding(x.unsqueeze(-1))  # (batch × n_snps × hidden_dim)
        
        # 2. Deep 1D-CNN extract local features (add channel dimension)
        x_cnn = x.unsqueeze(1)  # (batch × 1 × n_snps)
        x_cnn = self.cnn(x_cnn).transpose(1, 2)  # (batch × n_snps × cnn_output_dim)
        
        # 3. Feature projection: Project CNN features to BiMamba dimension
        x_cnn_proj = self.feature_projection(x_cnn)  # (batch × n_snps × hidden_dim)
        
        # 4. Fuse embedding features and CNN features
        x_combined = x_embed + x_cnn_proj  # Residual connection
        
        # 5. Positional encoding
        x_pos = self.position_embedding(x_combined)  # (batch × n_snps × hidden_dim)
        
        # 6. BiMamba encoding: Linear complexity capture long-range dependencies (core)
        x_mamba = x_pos
        for bimamba_layer in self.bimamba_layers:
            x_mamba = x_mamba + bimamba_layer(x_mamba)  # Residual connection
        
        # 7. Optional Transformer encoding (for fine-grained interactions)
        if self.use_transformer and self.transformer_encoder is not None:
            x_transformer = self.transformer_encoder(x_mamba)  # (batch × n_snps × hidden_dim)
            x_final = x_mamba + x_transformer  # Fuse BiMamba and Transformer features
        else:
            x_final = x_mamba  # Only use BiMamba features
        
        # 8. Main effect calculation
        x_main = self.main_effect(x)  # (batch × 16)
        # Main effect weights (between 0-1)
        main_weights = torch.sigmoid(self.main_effect_weights)
        
        # 9. Marker-pair level epistatic scores (extract from BiMamba/Transformer features)
        # BiMamba has already captured long-range dependencies, can extract interaction scores
        all_epistatic_pairs = []
        all_epistatic_scores = []
        
        # Use BiMamba-encoded features to extract interaction pairs
        # If SNP count is not too large, calculate all pairs; otherwise sample
        n_pairs_to_sample = min(
            self.max_epi_pairs, self.n_snps * (self.n_snps - 1) // 2
        )
        
        if self.n_snps <= 100:  # Small scale: calculate all pairs
            for i in range(self.n_snps):
                for j in range(i + 1, self.n_snps):
                    # Concatenate BiMamba features of two SNPs
                    # (batch × 2*hidden_dim)
                    pair_feat = torch.cat([x_final[:, i], x_final[:, j]], dim=1)
                    # Scalar
                    epi_score = torch.sigmoid(
                        self.epistatic_pair(pair_feat)
                    ).mean(dim=0).squeeze()
                    all_epistatic_pairs.append((i, j))
                    all_epistatic_scores.append(epi_score)
        else:  # Large scale: Sampling strategy (prioritize SNPs with high main effect weights)
            # Ensure top_snps has at least one element
            k_top = min(200, self.n_snps)
            if k_top > 0:
                top_snps = torch.topk(main_weights, k=k_top)[1].cpu().numpy()
            else:
                top_snps = np.arange(self.n_snps)
            
            sampled_pairs = set()
            max_attempts = n_pairs_to_sample * 3
            for _ in range(max_attempts):
                if len(sampled_pairs) >= n_pairs_to_sample:
                    break
                i = np.random.choice(top_snps)
                j = np.random.choice(self.n_snps)
                if i != j and (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
                    if i < j:
                        sampled_pairs.add((i, j))
                    else:
                        sampled_pairs.add((j, i))
            
            for i, j in sampled_pairs:
                pair_feat = torch.cat([x_final[:, i], x_final[:, j]], dim=1)
                epi_score = torch.sigmoid(
                    self.epistatic_pair(pair_feat)
                ).mean(dim=0).squeeze()  # Scalar
                all_epistatic_pairs.append((i, j))
                all_epistatic_scores.append(epi_score)
        
        # Organize epistatic results (Top-K)
        if all_epistatic_scores:
            # Ensure all scores are scalars, then stack into 1D tensor
            all_epistatic_scores_flat = []
            for score in all_epistatic_scores:
                if score.dim() > 0:
                    score = score.squeeze()
                if score.dim() == 0:  # Scalar
                    all_epistatic_scores_flat.append(score)
                else:
                    all_epistatic_scores_flat.append(score.item())
            
            if all_epistatic_scores_flat:
                # Convert to tensor and ensure 1D
                if isinstance(all_epistatic_scores_flat[0], torch.Tensor):
                    epistatic_scores = torch.stack(all_epistatic_scores_flat)
                else:
                    epistatic_scores = torch.tensor(all_epistatic_scores_flat, device=x.device)
                
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
                    epistatic_pairs = [all_epistatic_pairs[idx.item()] for idx in top_indices]
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
        
        # 10. Global epistatic score (for loss function)
        # Fuse CNN local features + BiMamba global features + main effects
        x_flat = torch.cat([
            x_cnn.mean(dim=1),  # Mean of CNN local features
            x_final.mean(dim=1),  # Mean of BiMamba global features
            x_main  # Main effects
        ], dim=1)  # (batch × (cnn_output_dim + hidden_dim + 16))
        # (batch × 1)
        epistatic_score_global = torch.sigmoid(
            self.epistatic(x_flat).mean(dim=1, keepdim=True)
        )
        
        # 11. Phenotype prediction
        pred = torch.sigmoid(self.output(self.epistatic(x_flat)))  # (batch × 1)
        
        return pred, main_weights, epistatic_pairs, epistatic_scores



class InterChrModel(nn.Module):
    """
    Inter-chr Model: BiMamba (linear complexity) + Optional Transformer (fine-grained interactions)
    
    Design Philosophy (referencing claude.py):
    - BiMamba: Linear complexity O(n) for ultra-long sequences, capture long-range dependencies
    - Cross-chromosome attention: Connect features from two chromosomes
    - Optional Transformer: Provide fine-grained interaction capture on small-scale data
    
    Advantages:
    - Linear complexity suitable for processing genome-wide data with millions of markers
    - BiMamba can efficiently encode global features of each chromosome
    - Cross-chromosome attention mechanism captures inter-chromosome interactions
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
        self.cross_attention = nn.MultiheadAttention(
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
        # Main effect weights
        self.main_weights_chr1 = nn.Parameter(torch.randn(n_snps_chr1))
        self.main_weights_chr2 = nn.Parameter(torch.randn(n_snps_chr2))

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
        
        # 5. Cross-chromosome attention (focus on high-potential interaction SNP pairs,
        # referencing Geneformer's attention mechanism)
        # Use global average pooling as query, local features as key and value
        chr1_global = chr1_feat.mean(dim=1, keepdim=True)  # (batch × 1 × hidden_dim)
        chr2_global = chr2_feat.mean(dim=1, keepdim=True)  # (batch × 1 × hidden_dim)
        
        # Cross-chromosome attention: Chr1's global representation attends to Chr2's
        # local features
        attn_output_chr1, attn_weights_1to2 = self.cross_attention(
            chr1_global, chr2_feat, chr2_feat
        )  # (batch × 1 × hidden_dim)
        attn_output_chr2, attn_weights_2to1 = self.cross_attention(
            chr2_global, chr1_feat, chr1_feat
        )  # (batch × 1 × hidden_dim)
        
        # 5. Main effect weights
        main_weights_chr1 = torch.sigmoid(self.main_weights_chr1)
        main_weights_chr2 = torch.sigmoid(self.main_weights_chr2)
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
        
        # Calculate marker-pair epistatic scores (using BiMamba/Transformer encoded features)
        for i, j in sampled_pairs:
            # Concatenate BiMamba features of two SNPs (batch × 2*hidden_dim)
            pair_feat = torch.cat([chr1_feat[:, i], chr2_feat[:, j]], dim=1)
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

# -------------------------- 4. Training Function (Separate Training + Result Integration) --------------------------