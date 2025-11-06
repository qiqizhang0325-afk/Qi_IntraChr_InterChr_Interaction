"""Main function for intra/inter-chromosome SNP interaction analysis."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Imports - handle both direct execution and module import
try:
    # Try relative imports first (when used as a module)
    from .data_processor import VCFProcessor
    from .dataset import SNPDataset
    from .models import InterChrModel, IntraChrModel
    from .training import integrate_results, train_model
except ImportError:
    # Fall back to absolute imports (when run directly)
    from data_processor import VCFProcessor
    from dataset import SNPDataset
    from models import InterChrModel, IntraChrModel
    from training import integrate_results, train_model

if __name__ == "__main__":
    # Configuration parameters
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root directory (one level up from src/)
    project_root = os.path.dirname(script_dir)
    
    # Create data and results directories if they don't exist
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # VCF file path - look in data/ directory
    vcf_path = os.path.join(data_dir, 'test.vcf')
    VCF_PATH = vcf_path  # Your test VCF file path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    # Model architecture configuration (optimized for large-scale data)
    HIDDEN_DIM = 128  # Hidden layer dimension
    N_CNN_LAYERS = 4  # Number of CNN layers (for local feature extraction)
    N_BIMAMBA_LAYERS = 2  # Number of BiMamba layers (linear complexity, core module)
    N_TRANSFORMER_LAYERS = 0  # Number of Transformer layers (optional, 0 means not used)
    USE_TRANSFORMER = False  # Whether to use Transformer (False=BiMamba only, True=hybrid)
    # Note: For large-scale data (millions of markers), recommend USE_TRANSFORMER=False
    # For small-scale data (<10K markers), can set USE_TRANSFORMER=True

    # Phenotype type configuration
    PHENOTYPE_TYPE = 'continuous'  # 'continuous' or 'binary'
    HERITABILITY = 0.9  # Heritability
    
    # Step 1: Data preprocessing
    processor = VCFProcessor(VCF_PATH)
    snp_info, snp_data = processor.parse_vcf()
    # Simulate phenotype: can choose continuous or binary
    phenotype = processor.simulate_phenotype(
        heritability=HERITABILITY, 
        phenotype_type=PHENOTYPE_TYPE,
        normalize=True  # Whether to normalize continuous phenotype
    )
    intra_blocks, inter_blocks = processor.split_intra_inter_blocks()

    # Step 2: Train intra-chr models separately
    # Select loss function according to phenotype type
    if PHENOTYPE_TYPE == 'binary':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  # Continuous phenotype uses mean squared error loss
    
    # key: chromosome ID, value: (main effect weights, SNP indices, epi_pairs, epi_scores)
    intra_results = {}
    # key: chromosome ID, value: training history
    intra_histories = {}
    
    for chr_id, chr_snps in intra_blocks.items():
        print(f"\n===== Training intra-chr model (Chromosome {chr_id}) =====")
        # Get SNP indices for this chromosome
        chr_snp_idx = snp_info[snp_info['CHROM'] == chr_id].index.values
        # Build dataset
        dataset = SNPDataset(chr_snps, phenotype)
        # Split training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model (using BiMamba, suitable for large-scale data)
        model = IntraChrModel(
            n_snps=chr_snps.shape[0],
            hidden_dim=HIDDEN_DIM,
            n_cnn_layers=N_CNN_LAYERS,
            n_bimamba_layers=N_BIMAMBA_LAYERS,
            n_transformer_layers=N_TRANSFORMER_LAYERS,
            use_transformer=USE_TRANSFORMER
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Train (pass phenotype type parameter)
        trained_model, best_metric, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS, 
            phenotype_type=PHENOTYPE_TYPE
        )
        intra_histories[chr_id] = history
        
        # Save results (use validation set to get final results)
        trained_model.eval()
        with torch.no_grad():
            # Use all validation set samples for inference
            all_val_snps = []
            for snps, _ in val_loader:
                all_val_snps.append(snps)
            if all_val_snps:
                val_snps_batch = torch.cat(all_val_snps, dim=0).to(DEVICE)
                _, main_weights, epi_pairs, epi_scores = trained_model(val_snps_batch)
                # Average batch results
                if isinstance(main_weights, torch.Tensor) and main_weights.dim() > 1:
                    main_weights = main_weights.mean(dim=0)
            else:
                # If no validation set, use first training sample
                sample_snps = dataset[0][0].unsqueeze(0).to(DEVICE)
                _, main_weights, epi_pairs, epi_scores = trained_model(sample_snps)
        intra_results[chr_id] = (main_weights, chr_snp_idx, epi_pairs, epi_scores)

    # Step 3: Train inter-chr models separately
    # key: (chr1, chr2), value: (main effect weights, chr1/chr2 SNP indices, epi_pairs, epi_scores)
    inter_results = {}
    # key: (chr1, chr2), value: training history
    inter_histories = {}
    
    for chr1, chr2, inter_snps in inter_blocks:
        print(f"\n===== Training inter-chr model ({chr1}×{chr2}) =====")
        # Get SNP indices for two chromosomes
        chr1_snp_idx = snp_info[snp_info['CHROM'] == chr1].index.values
        chr2_snp_idx = snp_info[snp_info['CHROM'] == chr2].index.values
        n_snps_chr1 = len(chr1_snp_idx)
        n_snps_chr2 = len(chr2_snp_idx)
        # Build dataset
        dataset = SNPDataset(inter_snps, phenotype)
        # Split training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model (using BiMamba, suitable for large-scale data)
        model = InterChrModel(
            n_snps_chr1=n_snps_chr1,
            n_snps_chr2=n_snps_chr2,
            hidden_dim=HIDDEN_DIM,
            n_bimamba_layers=N_BIMAMBA_LAYERS,
            n_transformer_layers=N_TRANSFORMER_LAYERS,
            use_transformer=USE_TRANSFORMER
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Train (pass phenotype type parameter)
        trained_model, best_metric, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS,
            phenotype_type=PHENOTYPE_TYPE
        )
        inter_histories[(chr1, chr2)] = history
        
        # Save results (use validation set to get final results)
        trained_model.eval()
        with torch.no_grad():
            all_val_snps = []
            for snps, _ in val_loader:
                all_val_snps.append(snps)
            if all_val_snps:
                val_snps_batch = torch.cat(all_val_snps, dim=0).to(DEVICE)
                _, main_weights, epi_pairs, epi_scores = trained_model(val_snps_batch)
                if isinstance(main_weights, torch.Tensor) and main_weights.dim() > 1:
                    main_weights = main_weights.mean(dim=0)
            else:
                sample_snps = dataset[0][0].unsqueeze(0).to(DEVICE)
                _, main_weights, epi_pairs, epi_scores = trained_model(sample_snps)
        inter_results[(chr1, chr2)] = (main_weights, chr1_snp_idx, chr2_snp_idx, epi_pairs, epi_scores)

    # Step 4: Result integration and output
    main_df_top10, epistatic_df_top10, main_df_all, epistatic_df_all = integrate_results(
        intra_results, inter_results, snp_info
    )
    print("\n===== Top10 Main Effect Markers =====")
    print(main_df_top10)
    print("\n===== Top10 Epistatic Interaction Pairs =====")
    print(epistatic_df_top10)
    print(f"\n  - Intra-chr pairs: {len(epistatic_df_top10[epistatic_df_top10['Pair_Type'] == 'intra-chr'])}")
    print(f"  - Inter-chr pairs: {len(epistatic_df_top10[epistatic_df_top10['Pair_Type'] == 'inter-chr'])}")
    
    # Save results to text files
    print("\n===== Saving Results to Files =====")
    # Save Top10 results
    main_df_top10.to_csv(
        os.path.join(results_dir, 'main_effect_results_top10.txt'),
        sep='\t', index=False, float_format='%.6f'
    )
    epistatic_df_top10.to_csv(
        os.path.join(results_dir, 'epistatic_interactions_top10.txt'),
        sep='\t', index=False, float_format='%.6f'
    )
    print(f"Top10 main effect results saved to: results/main_effect_results_top10.txt")
    print(f"Top10 epistatic interaction results saved to: results/epistatic_interactions_top10.txt")
    
    # Save all results
    main_df_all.to_csv(
        os.path.join(results_dir, 'main_effect_results_all.txt'),
        sep='\t', index=False, float_format='%.6f'
    )
    epistatic_df_all.to_csv(
        os.path.join(results_dir, 'epistatic_interactions_all.txt'),
        sep='\t', index=False, float_format='%.6f'
    )
    print(f"All main effect results saved to: results/main_effect_results_all.txt ({len(main_df_all)} SNPs)")
    print(f"All epistatic interaction results saved to: results/epistatic_interactions_all.txt ({len(epistatic_df_all)} pairs)")
    
    # Save training history to text file (use UTF-8 encoding to avoid Windows GBK issues)
    # Use R2 instead of R² to avoid encoding issues
    metric_name = 'R2' if PHENOTYPE_TYPE == 'continuous' else 'AUC'
    with open(os.path.join(results_dir, 'training_history.txt'), 'w', encoding='utf-8') as f:
        f.write("Training History Record\n")
        f.write("=" * 80 + "\n\n")
        
        # Intra-chr model history
        for chr_id, history in intra_histories.items():
            f.write(f"Intra-chr Model - Chromosome {chr_id}\n")
            f.write("-" * 80 + "\n")
            header = (
                f"{'Epoch':<10} {'Train Loss':<15} "
                f"{'Train ' + metric_name:<15} {'Val Loss':<15} "
                f"{'Val ' + metric_name:<15}\n"
            )
            f.write(header)
            f.write("-" * 80 + "\n")
            for epoch in range(len(history['train_loss'])):
                f.write(
                    f"{epoch+1:<10} {history['train_loss'][epoch]:<15.6f} "
                    f"{history['train_metric'][epoch]:<15.6f} "
                    f"{history['val_loss'][epoch]:<15.6f} "
                    f"{history['val_metric'][epoch]:<15.6f}\n"
                )
            f.write("\n")
        
        # Inter-chr model history
        for (chr1, chr2), history in inter_histories.items():
            f.write(f"Inter-chr Model - {chr1}×{chr2}\n")
            f.write("-" * 80 + "\n")
            header = (
                f"{'Epoch':<10} {'Train Loss':<15} "
                f"{'Train ' + metric_name:<15} {'Val Loss':<15} "
                f"{'Val ' + metric_name:<15}\n"
            )
            f.write(header)
            f.write("-" * 80 + "\n")
            for epoch in range(len(history['train_loss'])):
                f.write(
                    f"{epoch+1:<10} {history['train_loss'][epoch]:<15.6f} "
                    f"{history['train_metric'][epoch]:<15.6f} "
                    f"{history['val_loss'][epoch]:<15.6f} "
                    f"{history['val_metric'][epoch]:<15.6f}\n"
                )
            f.write("\n")
    
    print(f"Training history saved to: results/training_history.txt")
    
    # Plot training curves (loss and R²)
    print("\n===== Plotting Training Curves =====")
    # Plot training curves for all intra-chr models
    for chr_id, history in intra_histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Loss curve
        ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Loss Curve - Chromosome {chr_id}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # R²/AUC curve
        metric_name_display = 'R²' if PHENOTYPE_TYPE == 'continuous' else 'AUC'
        ax2.plot(epochs_range, history['train_metric'], 'b-', label=f'Train {metric_name_display}', linewidth=2)
        ax2.plot(epochs_range, history['val_metric'], 'r-', label=f'Val {metric_name_display}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name_display, fontsize=12)
        ax2.set_title(f'{metric_name_display} Curve - Chromosome {chr_id}', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f'training_curves_intra_chr{chr_id}.png'),
            dpi=300, bbox_inches='tight'
        )
        print(f"Training curves saved to: results/training_curves_intra_chr{chr_id}.png")
        plt.close()
    
    # Plot training curves for all inter-chr models
    for (chr1, chr2), history in inter_histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Loss curve
        ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Loss Curve - {chr1}×{chr2}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # R²/AUC curve
        metric_name_display = 'R²' if PHENOTYPE_TYPE == 'continuous' else 'AUC'
        ax2.plot(epochs_range, history['train_metric'], 'b-', label=f'Train {metric_name_display}', linewidth=2)
        ax2.plot(epochs_range, history['val_metric'], 'r-', label=f'Val {metric_name_display}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name_display, fontsize=12)
        ax2.set_title(f'{metric_name_display} Curve - {chr1}×{chr2}', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f'training_curves_inter_{chr1}_{chr2}.png'),
            dpi=300, bbox_inches='tight'
        )
        print(f"Training curves saved to: results/training_curves_inter_{chr1}_{chr2}.png")
        plt.close()

    # Step 5: Result visualization (GWAS-style, distinguish intra/inter)
    # Main effect Manhattan plot - Top10
    plt.figure(figsize=(12, 6))
    snp_info_main_top10 = (
        snp_info.merge(
            main_df_top10[['SNP_ID', 'Main_Effect_Weight']],
            left_on='ID', right_on='SNP_ID', how='left'
        ).fillna(0)
    )
    for chr_id in sorted(snp_info['CHROM'].unique()):
        chr_data = snp_info_main_top10[snp_info_main_top10['CHROM'] == chr_id]
        plt.scatter(chr_data['POS'], chr_data['Main_Effect_Weight'], label=f'Chr{chr_id}', s=30)
    plt.xlabel('Position (bp)', fontsize=12)
    plt.ylabel('Main Effect Weight', fontsize=12)
    plt.title('Main Effect SNPs - Top10 (Manhattan Plot Style)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, 'main_effect_manhattan_top10.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    # Main effect Manhattan plot - All
    plt.figure(figsize=(14, 6))
    snp_info_main_all = (
        snp_info.merge(
            main_df_all[['SNP_ID', 'Main_Effect_Weight']],
            left_on='ID', right_on='SNP_ID', how='left'
        ).fillna(0)
    )
    for chr_id in sorted(snp_info['CHROM'].unique()):
        chr_data = snp_info_main_all[snp_info_main_all['CHROM'] == chr_id]
        plt.scatter(chr_data['POS'], chr_data['Main_Effect_Weight'], label=f'Chr{chr_id}', s=10, alpha=0.6)
    plt.xlabel('Position (bp)', fontsize=12)
    plt.ylabel('Main Effect Weight', fontsize=12)
    plt.title('Main Effect SNPs - All (Manhattan Plot Style)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, 'main_effect_manhattan_all.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("Main effect Manhattan plots saved: results/main_effect_manhattan_top10.png, results/main_effect_manhattan_all.png")

    # Epistatic interaction heatmap - Top10 (distinguish intra/inter)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Intra-chr interaction heatmap - Top10
    intra_epi_top10 = epistatic_df_top10[epistatic_df_top10['Pair_Type'] == 'intra-chr'].head(10)
    if not intra_epi_top10.empty:
        snps_intra = list(set(intra_epi_top10['SNP1']) | set(intra_epi_top10['SNP2']))
        snp_idx_intra = {snp: i for i, snp in enumerate(snps_intra)}
        heatmap_intra = np.zeros((len(snps_intra), len(snps_intra)))
        for _, row in intra_epi_top10.iterrows():
            i = snp_idx_intra[row['SNP1']]
            j = snp_idx_intra[row['SNP2']]
            heatmap_intra[i, j] = row['Epistatic_Score']
            heatmap_intra[j, i] = row['Epistatic_Score']
        sns.heatmap(heatmap_intra, xticklabels=snps_intra, yticklabels=snps_intra, 
                    cmap='RdBu_r', annot=True, ax=ax1, fmt='.3f')
        ax1.set_title('Intra-chromosome Epistatic Interactions - Top10', fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'No intra-chr interactions found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Intra-chromosome Epistatic Interactions - Top10', fontsize=12)
    
    # Inter-chr interaction heatmap - Top10
    inter_epi_top10 = epistatic_df_top10[epistatic_df_top10['Pair_Type'] == 'inter-chr'].head(10)
    if not inter_epi_top10.empty:
        snps_inter = list(set(inter_epi_top10['SNP1']) | set(inter_epi_top10['SNP2']))
        snp_idx_inter = {snp: i for i, snp in enumerate(snps_inter)}
        heatmap_inter = np.zeros((len(snps_inter), len(snps_inter)))
        for _, row in inter_epi_top10.iterrows():
            i = snp_idx_inter[row['SNP1']]
            j = snp_idx_inter[row['SNP2']]
            heatmap_inter[i, j] = row['Epistatic_Score']
            heatmap_inter[j, i] = row['Epistatic_Score']
        sns.heatmap(heatmap_inter, xticklabels=snps_inter, yticklabels=snps_inter, 
                    cmap='RdBu_r', annot=True, ax=ax2, fmt='.3f')
        ax2.set_title('Inter-chromosome Epistatic Interactions - Top10', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No inter-chr interactions found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Inter-chromosome Epistatic Interactions - Top10', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, 'epistatic_heatmap_top10.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    # Epistatic interaction heatmap - All (distinguish intra/inter)
    # For all results, we'll create a more compact visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Intra-chr interaction heatmap - All
    intra_epi_all = epistatic_df_all[epistatic_df_all['Pair_Type'] == 'intra-chr']
    if not intra_epi_all.empty:
        # For large datasets, we might need to sample or use a different visualization
        # Here we'll show all pairs but with smaller labels
        snps_intra_all = list(set(intra_epi_all['SNP1']) | set(intra_epi_all['SNP2']))
        # Limit to reasonable number for visualization (max 50 SNPs)
        if len(snps_intra_all) > 50:
            # Get top scoring pairs and their SNPs
            top_intra_pairs = intra_epi_all.head(100)  # Get top 100 pairs
            snps_intra_all = list(set(top_intra_pairs['SNP1']) | set(top_intra_pairs['SNP2']))
            intra_epi_all = top_intra_pairs
        
        snp_idx_intra_all = {snp: i for i, snp in enumerate(snps_intra_all)}
        heatmap_intra_all = np.zeros((len(snps_intra_all), len(snps_intra_all)))
        for _, row in intra_epi_all.iterrows():
            if row['SNP1'] in snp_idx_intra_all and row['SNP2'] in snp_idx_intra_all:
                i = snp_idx_intra_all[row['SNP1']]
                j = snp_idx_intra_all[row['SNP2']]
                heatmap_intra_all[i, j] = row['Epistatic_Score']
                heatmap_intra_all[j, i] = row['Epistatic_Score']
        
        # Use smaller font for many SNPs
        fontsize = 6 if len(snps_intra_all) > 20 else 8
        sns.heatmap(heatmap_intra_all, xticklabels=snps_intra_all, yticklabels=snps_intra_all, 
                    cmap='RdBu_r', annot=False, ax=ax1, fmt='.3f', 
                    cbar_kws={'label': 'Epistatic Score'})
        ax1.set_title(f'Intra-chromosome Epistatic Interactions - All ({len(intra_epi_all)} pairs)', fontsize=12)
        ax1.tick_params(axis='x', labelsize=fontsize, rotation=90)
        ax1.tick_params(axis='y', labelsize=fontsize)
    else:
        ax1.text(0.5, 0.5, 'No intra-chr interactions found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Intra-chromosome Epistatic Interactions - All', fontsize=12)
    
    # Inter-chr interaction heatmap - All
    inter_epi_all = epistatic_df_all[epistatic_df_all['Pair_Type'] == 'inter-chr']
    if not inter_epi_all.empty:
        snps_inter_all = list(set(inter_epi_all['SNP1']) | set(inter_epi_all['SNP2']))
        # Limit to reasonable number for visualization
        if len(snps_inter_all) > 50:
            top_inter_pairs = inter_epi_all.head(100)
            snps_inter_all = list(set(top_inter_pairs['SNP1']) | set(top_inter_pairs['SNP2']))
            inter_epi_all = top_inter_pairs
        
        snp_idx_inter_all = {snp: i for i, snp in enumerate(snps_inter_all)}
        heatmap_inter_all = np.zeros((len(snps_inter_all), len(snps_inter_all)))
        for _, row in inter_epi_all.iterrows():
            if row['SNP1'] in snp_idx_inter_all and row['SNP2'] in snp_idx_inter_all:
                i = snp_idx_inter_all[row['SNP1']]
                j = snp_idx_inter_all[row['SNP2']]
                heatmap_inter_all[i, j] = row['Epistatic_Score']
                heatmap_inter_all[j, i] = row['Epistatic_Score']
        
        fontsize = 6 if len(snps_inter_all) > 20 else 8
        sns.heatmap(heatmap_inter_all, xticklabels=snps_inter_all, yticklabels=snps_inter_all, 
                    cmap='RdBu_r', annot=False, ax=ax2, fmt='.3f',
                    cbar_kws={'label': 'Epistatic Score'})
        ax2.set_title(f'Inter-chromosome Epistatic Interactions - All ({len(inter_epi_all)} pairs)', fontsize=12)
        ax2.tick_params(axis='x', labelsize=fontsize, rotation=90)
        ax2.tick_params(axis='y', labelsize=fontsize)
    else:
        ax2.text(0.5, 0.5, 'No inter-chr interactions found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Inter-chromosome Epistatic Interactions - All', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, 'epistatic_heatmap_all.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("Epistatic interaction heatmaps saved: results/epistatic_heatmap_top10.png, results/epistatic_heatmap_all.png")
    
