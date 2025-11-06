"""Intra-chromosome SNP interaction model."""

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
    4. Main effect and interaction calculations are based on encoded features
       (CNN + BiMamba), ensuring that both utilize the rich contextual information
    
    Advantages:
    - BiMamba's linear complexity suitable for large-scale data
    - Can process all markers on entire chromosome
    - Combine CNN and BiMamba to capture multi-level features
    - Main effects and interactions are computed from encoded features, capturing
      contextual dependencies
    """
    def __init__(self, n_snps, hidden_dim=128, max_epi_pairs=200, 
                 n_cnn_layers=4, n_bimamba_layers=2, n_transformer_layers=0, 
                 n_heads=8, use_transformer=False,
                 interaction_mode='auto', main_weight_threshold=0.5,
                 main_weight_threshold_mode='absolute'):
        super(IntraChrModel, self).__init__()
        self.n_snps = n_snps
        self.hidden_dim = hidden_dim
        self.max_epi_pairs = max_epi_pairs
        self.use_transformer = use_transformer  # Whether to use Transformer (optional)
        # Interaction mode:
        #  - 'main_nonzero': only pairs among SNPs with main weight > threshold
        #  - 'all': pairs among all SNPs
        #  - 'auto': previous behavior (small: all pairs; large: sample biased by main weights)
        self.interaction_mode = interaction_mode
        self.main_weight_threshold = float(main_weight_threshold)
        # Threshold mode for 'main_nonzero':
        #  - 'absolute': use absolute threshold value (default: 0.5)
        #  - 'median': use median of main_weights as threshold (equivalent to percentile 0.5)
        #  - 'mean': use mean of main_weights as threshold
        #  - 'percentile': use percentile (threshold value as percentile, e.g., 0.5 = median, 0.75 = 75th percentile)
        # Note: 'percentile' with 0.5 is equivalent to 'median', but 'median' is more explicit
        self.main_weight_threshold_mode = main_weight_threshold_mode
        
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
        
        # 5. Main effect capture layer (based on encoded features)
        # Extract main effect features from each SNP's encoded representation
        self.main_effect_per_snp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)
        )
        # Aggregate main effect features across all SNPs
        self.main_effect_aggregate = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        
        # 8. Main effect calculation (based on encoded features)
        # Extract main effect features from each SNP's encoded representation
        # x_final: (batch × n_snps × hidden_dim)
        # Apply main effect extraction to each SNP
        main_effect_features = self.main_effect_per_snp(x_final)  # (batch × n_snps × 16)
        
        # Main effect weights (between 0-1) - learnable parameters
        main_weights = torch.sigmoid(self.main_effect_weights)  # (n_snps)
        
        # Weighted aggregation: use main_weights to aggregate main effect features
        # Expand main_weights for broadcasting: (n_snps) -> (1 × n_snps × 1)
        main_weights_expanded = main_weights.unsqueeze(0).unsqueeze(-1)  # (1 × n_snps × 1)
        # Weighted sum: (batch × n_snps × 16) * (1 × n_snps × 1) -> (batch × n_snps × 16)
        weighted_main_features = main_effect_features * main_weights_expanded
        # Sum over SNPs: (batch × n_snps × 16) -> (batch × 16)
        x_main_weighted = weighted_main_features.sum(dim=1)  # (batch × 16)
        # Apply aggregation layer
        x_main = self.main_effect_aggregate(x_main_weighted)  # (batch × 16)
        
        # 9. Marker-pair level epistatic scores (extract from BiMamba/Transformer features)
        # BiMamba has already captured long-range dependencies, can extract interaction scores
        all_epistatic_pairs = []
        all_epistatic_scores = []
        
        # Use BiMamba-encoded features to extract interaction pairs
        # Determine candidate SNP indices based on interaction_mode
        if self.interaction_mode in ('main_nonzero', 'all'):
            if self.interaction_mode == 'main_nonzero':
                # Calculate threshold based on mode
                if self.main_weight_threshold_mode == 'median':
                    threshold = main_weights.median()
                elif self.main_weight_threshold_mode == 'mean':
                    threshold = main_weights.mean()
                elif self.main_weight_threshold_mode == 'percentile':
                    # threshold value is interpreted as percentile (0-1)
                    threshold = torch.quantile(main_weights, self.main_weight_threshold)
                else:  # 'absolute'
                    threshold = self.main_weight_threshold
                
                # Select SNPs with main weight > threshold
                candidate = (main_weights > threshold).nonzero(as_tuple=True)[0].cpu().numpy()
                if candidate.size < 2:
                    # 阈值过严时退化为按主效取前2个，保证成对
                    k = min(2, self.n_snps)
                    candidate = torch.topk(main_weights, k=k)[1].cpu().numpy()
            else:  # 'all'
                candidate = np.arange(self.n_snps)

            n_candidate = int(len(candidate))
            # 计算 candidate 内所有互作对（不采样，不截断）
            for ii in range(n_candidate):
                i = int(candidate[ii])
                for jj in range(ii + 1, n_candidate):
                    j = int(candidate[jj])
                    pair_feat = torch.cat([x_final[:, i], x_final[:, j]], dim=1)
                    epi_score = torch.sigmoid(self.epistatic_pair(pair_feat)).mean(dim=0).squeeze()
                    all_epistatic_pairs.append((i, j))
                    all_epistatic_scores.append(epi_score)
        else:
            # 'auto'（兼容旧逻辑）：小规模全对，大规模按主效加权采样
            n_pairs_to_sample = min(self.max_epi_pairs, self.n_snps * (self.n_snps - 1) // 2)

            if self.n_snps <= 200:
                for i in range(self.n_snps):
                    for j in range(i + 1, self.n_snps):
                        pair_feat = torch.cat([x_final[:, i], x_final[:, j]], dim=1)
                        epi_score = torch.sigmoid(self.epistatic_pair(pair_feat)).mean(dim=0).squeeze()
                        all_epistatic_pairs.append((i, j))
                        all_epistatic_scores.append(epi_score)
            else:
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
                    epi_score = torch.sigmoid(self.epistatic_pair(pair_feat)).mean(dim=0).squeeze()
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
                # 明确模式下返回全部互作；auto 模式保留 top-k 截断
                if self.interaction_mode in ('main_nonzero', 'all'):
                    top_k = actual_len
                else:
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



