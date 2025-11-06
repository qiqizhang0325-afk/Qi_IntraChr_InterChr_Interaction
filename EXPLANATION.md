# Detailed Logic Explanation

## Q1: How are the top main-effect markers and epistatic pairs selected?

They are selected based on the model-learned weights and scores, not the synthetic (ground-truth) markers used when simulating the phenotype.

### 1) Phenotype simulation (`data_processor.py`)

```python
# Randomly pick true main-effect markers and interaction pairs
main_effect_snps = np.random.choice(n_snps, 5, replace=False)
main_effect = 0.3 * self.snp_data[main_effect_snps].sum(axis=0)

epistatic_pairs = [(np.random.choice(n_snps), np.random.choice(n_snps)) for _ in range(3)]
epistatic_effect = 0.5 * sum(self.snp_data[i] * self.snp_data[j] for i, j in epistatic_pairs)
```

Notes:
- These are the ground-truth loci/pairs, unknown to the model.
- The model only sees SNPs and phenotypes, and must learn patterns.

### 2) Training (`training.py`)

The model is optimized to predict the phenotype. We also regularize learned main-effect weights and epistatic scores.

- `best_metric` (AUC or R²) is used only for reporting validation performance.
- It is not used to pick SNPs or interactions.

### 3) Result extraction (`main.py`)

After training, we evaluate the trained model on validation data and extract:
- `main_weights`: learned importance of SNPs
- `epi_pairs`: candidate interaction pairs
- `epi_scores`: learned interaction scores

### 4) Integration (`training.py::integrate_results`)

- Main effects are ranked by `Main_Effect_Weight` (descending)
- Epistatic pairs are ranked by `Epistatic_Score` (descending)
- Top lists are then produced from these rankings

Conclusion:
- Top main-effect markers = highest learned `Main_Effect_Weight`
- Top epistatic pairs = highest learned `Epistatic_Score`
- Not chosen by the simulation choices, and not chosen by `best_metric`

---

## Q2: Why can labels in the heatmap differ from those in the text output?

Previously, the heatmap displayed Top 6 per type while the text listed Top 10 overall, so the displayed SNP set could differ. The code was updated to ensure consistency:
- We now build Top 10 separately for intra-chr and inter-chr and then merge.
- Heatmaps for Top 10 use `.head(10)` consistently.

---

## `training.py` Overview

### `train_model`

Trains a single model (intra-chr or inter-chr) for multiple epochs.

Inputs:
- `model`: `IntraChrModel` or `InterChrModel`
- `train_loader`, `val_loader`
- `criterion`, `optimizer`, `device`
- `epochs`, `phenotype_type` (binary or continuous)

Per-epoch loop:
- Forward: `pred, main_weights, epi_pairs, epi_scores = model(snps)`
- Loss: `loss = cls_loss + 0.01*||main_weights|| + 0.01*mean(epi_scores)`
- Backprop: `loss.backward(); optimizer.step(); optimizer.zero_grad()`
- Validation: compute AUC (binary) or R² (continuous) for reporting

Return values:
- `model`: fully trained model (after all epochs)
- `best_metric`: best validation AUC/R² observed
- `history`: training/validation loss and metric curves

Important: We return the final trained model, not a checkpoint tied to `best_metric`.

### `integrate_results`

Aggregates results across models and produces DataFrames:
- Main effects: deduplicate by `SNP_ID`, sort by `Main_Effect_Weight` desc, provide both Top 10 and All
- Epistatic pairs: sort by `Epistatic_Score` desc, provide both Top 10 (built per type) and All

Returns:
- `main_df_top10`, `epistatic_df_top10`, `main_df_all`, `epistatic_df_all`

---

## Summary

- Detection is driven by learned weights/scores, not simulation choices or `best_metric`.
- Heatmap vs text mismatch was due to different Top-N limits; code now aligns them.
- `train_model` handles optimization and reporting; `integrate_results` consolidates outputs for saving and plotting.

