"""Training functions and result integration."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, epochs=50,
    phenotype_type='binary'
):
    """
    Train a single model (intra or inter), including training and validation sets
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        epochs: Number of training epochs
        phenotype_type: Phenotype type ('binary' or 'continuous')
    
    Returns:
        model: Trained model
        best_metric: Best evaluation metric (AUC for binary, R² for continuous)
        history: Training history dictionary, containing loss and metric history
    """
    model.to(device)
    best_metric = 0.0
    history = {
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': []
    }
    
    for epoch in range(epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for snps, labels in train_loader:
            snps, labels = snps.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward propagation
            pred, main_weights, epi_pairs, epi_scores = model(snps)
            # Loss function
            loss_cls = criterion(pred, labels)
            loss_main = 0.01 * torch.norm(main_weights)
            loss_epi = 0.01 * epi_scores.mean() if len(epi_scores) > 0 else 0.0
            total_loss_batch = loss_cls + loss_main + loss_epi
            
            # Backward propagation
            total_loss_batch.backward()
            optimizer.step()
            
            train_loss += total_loss_batch.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
        
        # Calculate training set metrics
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels).flatten()
        avg_train_loss = train_loss / len(train_loader)
        
        if phenotype_type == 'binary':
            train_metric = roc_auc_score(train_labels, train_preds)
            metric_name = 'AUC'
        else:
            if len(train_preds) > 1 and np.std(train_preds) > 1e-8 and np.std(train_labels) > 1e-8:
                corr, _ = pearsonr(train_preds, train_labels)
                train_metric = corr ** 2
            else:
                train_metric = 0.0
            metric_name = 'R²'
        
        history['train_loss'].append(avg_train_loss)
        history['train_metric'].append(train_metric)
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for snps, labels in val_loader:
                snps, labels = snps.to(device), labels.to(device)
                pred, main_weights, epi_pairs, epi_scores = model(snps)
                
                loss_cls = criterion(pred, labels)
                loss_main = 0.01 * torch.norm(main_weights)
                loss_epi = 0.01 * epi_scores.mean() if len(epi_scores) > 0 else 0.0
                total_loss_batch = loss_cls + loss_main + loss_epi
                
                val_loss += total_loss_batch.item()
                val_preds.extend(pred.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
        
        # Calculate validation set metrics
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels).flatten()
        avg_val_loss = val_loss / len(val_loader)
        
        if phenotype_type == 'binary':
            val_metric = roc_auc_score(val_labels, val_preds)
        else:
            if len(val_preds) > 1 and np.std(val_preds) > 1e-8 and np.std(val_labels) > 1e-8:
                corr, _ = pearsonr(val_preds, val_labels)
                val_metric = corr ** 2
            else:
                val_metric = 0.0
        
        history['val_loss'].append(avg_val_loss)
        history['val_metric'].append(val_metric)
        
        # Update best model
        if val_metric > best_metric:
            best_metric = val_metric
        
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Train {metric_name}: {train_metric:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val {metric_name}: {val_metric:.4f} | "
            f"Best Val {metric_name}: {best_metric:.4f}"
        )
    
    return model, best_metric, history



def integrate_results(intra_results, inter_results, snp_info):
    """Integrate main effect markers and epistatic interaction pairs
    (distinguish intra/inter, marker-pair level)"""
    # 1. Integrate main effect markers (Top10)
    all_main = []
    # Intra-chr main effects
    for chr_id, (main_weights, snp_idx, _, _) in intra_results.items():
        chr_snps = snp_info.iloc[snp_idx].reset_index(drop=True)
        for idx, (_, row) in enumerate(chr_snps.iterrows()):
            if idx < len(main_weights):
                all_main.append({
                    'SNP_ID': row['ID'],
                    'CHROM': chr_id,
                    'Main_Effect_Weight': (
                        main_weights[idx].item()
                        if isinstance(main_weights[idx], torch.Tensor)
                        else main_weights[idx]
                    ),
                    'Type': 'intra-chr'
                })
    # Inter-chr main effects
    for (chr1, chr2), (
        main_weights, chr1_snp_idx, chr2_snp_idx, _, _
    ) in inter_results.items():
        # Chromosome 1 SNPs
        chr1_snps = snp_info.iloc[chr1_snp_idx].reset_index(drop=True)
        for idx, (_, row) in enumerate(chr1_snps.iterrows()):
            if idx < len(main_weights):
                weight = (
                    main_weights[idx].item()
                    if isinstance(main_weights[idx], torch.Tensor)
                    else main_weights[idx]
                )
                all_main.append({
                    'SNP_ID': row['ID'],
                    'CHROM': chr1,
                    'Main_Effect_Weight': weight,
                    'Type': 'inter-chr'
                })
        # Chromosome 2 SNPs
        chr2_snps = snp_info.iloc[chr2_snp_idx].reset_index(drop=True)
        chr1_len = len(chr1_snp_idx)
        for idx, (_, row) in enumerate(chr2_snps.iterrows()):
            if chr1_len + idx < len(main_weights):
                weight_idx = chr1_len + idx
                weight = (
                    main_weights[weight_idx].item()
                    if isinstance(main_weights[weight_idx], torch.Tensor)
                    else main_weights[weight_idx]
                )
                all_main.append({
                    'SNP_ID': row['ID'],
                    'CHROM': chr2,
                    'Main_Effect_Weight': weight,
                    'Type': 'inter-chr'
                })
    
    # Create full results (all SNPs) and top10 results
    main_df_all = (
        pd.DataFrame(all_main)
        .drop_duplicates('SNP_ID')
        .sort_values('Main_Effect_Weight', ascending=False)
    )
    main_df_top10 = main_df_all.head(10)
    
    # 2. Integrate epistatic interaction pairs (all pairs, marker-pair level)
    all_epi = []
    # Intra-chr epistatic
    for chr_id, (_, snp_idx, epi_pairs, epi_scores) in intra_results.items():
        chr_snps = snp_info.iloc[snp_idx].reset_index(drop=True)
        chr_snp_ids = chr_snps['ID'].values
        for (i, j), score in zip(epi_pairs, epi_scores):
            if i < len(chr_snp_ids) and j < len(chr_snp_ids):
                score_val = score.item() if isinstance(score, torch.Tensor) else score
                all_epi.append({
                    'SNP1': chr_snp_ids[i],
                    'SNP2': chr_snp_ids[j],
                    'CHROM1': chr_id,
                    'CHROM2': chr_id,
                    'Pair_Type': 'intra-chr',
                    'Epistatic_Score': score_val
                })
    
    # Inter-chr epistatic
    for (chr1, chr2), (
        _, chr1_snp_idx, chr2_snp_idx, epi_pairs, epi_scores
    ) in inter_results.items():
        chr1_snps = snp_info.iloc[chr1_snp_idx].reset_index(drop=True)
        chr2_snps = snp_info.iloc[chr2_snp_idx].reset_index(drop=True)
        chr1_snp_ids = chr1_snps['ID'].values
        chr2_snp_ids = chr2_snps['ID'].values
        for (i, j), score in zip(epi_pairs, epi_scores):
            if i < len(chr1_snp_ids) and j < len(chr2_snp_ids):
                score_val = score.item() if isinstance(score, torch.Tensor) else score
                all_epi.append({
                    'SNP1': chr1_snp_ids[i],
                    'SNP2': chr2_snp_ids[j],
                    'CHROM1': chr1,
                    'CHROM2': chr2,
                    'Pair_Type': 'inter-chr',
                    'Epistatic_Score': score_val
                })
    
    # Create full results (all pairs) and top10 results
    epi_df_all = pd.DataFrame(all_epi).sort_values('Epistatic_Score', ascending=False)
    
    # Create Top10: separately for intra-chr and inter-chr (each gets Top10)
    intra_epi_all = epi_df_all[epi_df_all['Pair_Type'] == 'intra-chr']
    inter_epi_all = epi_df_all[epi_df_all['Pair_Type'] == 'inter-chr']
    
    intra_epi_top10 = intra_epi_all.head(10) if len(intra_epi_all) > 0 else pd.DataFrame()
    inter_epi_top10 = inter_epi_all.head(10) if len(inter_epi_all) > 0 else pd.DataFrame()
    
    # Combine intra and inter Top10
    epi_df_top10 = pd.concat([intra_epi_top10, inter_epi_top10], ignore_index=True)
    # Re-sort by score to maintain order
    if len(epi_df_top10) > 0:
        epi_df_top10 = epi_df_top10.sort_values('Epistatic_Score', ascending=False)
    
    return main_df_top10, epi_df_top10, main_df_all, epi_df_all

# -------------------------- 5. Main Function (Pipeline Integration) --------------------------
