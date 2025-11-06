# 代码逻辑详细解释

## 问题1: 检测到的主效位点和互作对是基于什么选择的？

### 答案：基于模型学习到的权重和分数，而不是模拟时指定的

让我详细解释整个流程：

### 1. 表型模拟阶段（`data_processor.py`）

```python
# 随机选择5个SNP作为主效位点
main_effect_snps = np.random.choice(n_snps, 5, replace=False)
main_effect = 0.3 * self.snp_data[main_effect_snps].sum(axis=0)

# 随机选择3个SNP对作为互作对
epistatic_pairs = [
    (np.random.choice(n_snps), np.random.choice(n_snps)) for _ in range(3)
]
epistatic_effect = 0.5 * sum(
    [self.snp_data[i] * self.snp_data[j] for i, j in epistatic_pairs]
)
```

**重要点**：
- 这些是**真实的主效位点和互作对**（ground truth）
- 但模型**不知道**这些信息
- 模型只能看到SNP数据和表型数据，需要自己学习

### 2. 模型训练阶段（`training.py`）

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, phenotype_type='binary'):
    # ...
    for epoch in range(epochs):
        # 训练阶段
        for snps, labels in train_loader:
            # 模型前向传播
            pred, main_weights, epi_pairs, epi_scores = model(snps)
            
            # 损失函数
            loss_cls = criterion(pred, labels)  # 预测损失
            loss_main = 0.01 * torch.norm(main_weights)  # 主效权重正则化
            loss_epi = 0.01 * epi_scores.mean()  # 互作分数正则化
            total_loss_batch = loss_cls + loss_main + loss_epi
            
            # 反向传播，更新模型参数
            total_loss_batch.backward()
            optimizer.step()
        
        # 验证阶段
        # 计算验证集上的AUC或R²
        if val_metric > best_metric:
            best_metric = val_metric
```

**关键理解**：
- `best_metric` 只是用来**评估模型性能**（AUC或R²）
- **不是**用来选择主效位点或互作对的
- 模型在整个训练过程中都在学习，最终模型是训练完所有epoch后的模型

### 3. 结果提取阶段（`main.py`）

```python
# 训练完成后，使用验证集提取结果
trained_model.eval()
with torch.no_grad():
    val_snps_batch = torch.cat(all_val_snps, dim=0).to(DEVICE)
    _, main_weights, epi_pairs, epi_scores = trained_model(val_snps_batch)
```

**这里提取的是**：
- `main_weights`: 模型学习到的每个SNP的主效权重（模型参数）
- `epi_pairs`: 模型认为重要的互作对（基于epistatic_score排序）
- `epi_scores`: 对应的互作分数

### 4. 结果整合阶段（`training.py` - `integrate_results`）

```python
def integrate_results(intra_results, inter_results, snp_info):
    # 1. 整合主效位点
    # 按照 Main_Effect_Weight 排序，取Top10
    main_df = (
        pd.DataFrame(all_main)
        .drop_duplicates('SNP_ID')
        .sort_values('Main_Effect_Weight', ascending=False)  # ← 按权重排序
        .head(10)
    )
    
    # 2. 整合互作对
    # 按照 Epistatic_Score 排序，取Top10
    epi_df = pd.DataFrame(all_epi).sort_values(
        'Epistatic_Score', ascending=False  # ← 按分数排序
    ).head(10)
```

**总结**：
- ✅ **检测到的主效位点**：按照模型学习到的 `Main_Effect_Weight` 排序选择Top10
- ✅ **检测到的互作对**：按照模型学习到的 `Epistatic_Score` 排序选择Top10
- ❌ **不是**基于模拟时指定的位点（模型不知道这些信息）
- ❌ **不是**基于 `best_metric`（best_metric只用于评估模型性能）

### 模型如何学习主效位点和互作对？

模型通过以下方式学习：

1. **主效权重** (`main_effect_weights`): 
   - 是模型的可学习参数 `nn.Parameter(torch.randn(n_snps))`
   - 通过反向传播自动更新
   - 训练后，权重大的SNP被认为是主效位点

2. **互作分数** (`epistatic_scores`):
   - 模型对每个SNP对计算互作分数
   - 通过 `epistatic_pair` 网络层计算
   - 分数高的对被认为是互作对

---

## 问题2: 为什么热图中的标记名和文本文件中的不一样？

### 原因：热图只显示Top6，而文本文件显示Top10

让我查看代码：

### 文本文件生成（`main.py`）

```python
# 保存所有Top10的结果
epistatic_df.to_csv('epistatic_interactions.txt', sep='\t', index=False, float_format='%.6f')
```

`epistatic_df` 包含Top10的互作对。

### 热图生成（`main.py`）

```python
# Intra-chr interaction heatmap
intra_epi = epistatic_df[epistatic_df['Pair_Type'] == 'intra-chr'].head(6)  # ← 只取Top6
if not intra_epi.empty:
    snps_intra = list(set(intra_epi['SNP1']) | set(intra_epi['SNP2']))
    # ... 生成热图

# Inter-chr interaction heatmap  
inter_epi = epistatic_df[epistatic_df['Pair_Type'] == 'inter-chr'].head(6)  # ← 只取Top6
if not inter_epi.empty:
    snps_inter = list(set(inter_epi['SNP1']) | set(inter_epi['SNP2']))
    # ... 生成热图
```

**问题所在**：
1. 文本文件保存了**所有Top10**的互作对
2. 热图只显示了**Top6**的互作对（`.head(6)`）
3. 热图显示的SNP是这Top6对中出现的所有SNP的并集

**举例说明**：
- 文本文件可能有10个互作对：Pair1, Pair2, ..., Pair10
- 热图只显示前6个：Pair1, Pair2, ..., Pair6
- 如果Pair7-Pair10包含了一些新的SNP，这些SNP不会出现在热图中

### 解决方案

如果你想热图显示所有Top10，可以修改代码：

```python
# 修改前
intra_epi = epistatic_df[epistatic_df['Pair_Type'] == 'intra-chr'].head(6)

# 修改后
intra_epi = epistatic_df[epistatic_df['Pair_Type'] == 'intra-chr'].head(10)
```

---

## training.py 详细解释

### 函数1: `train_model`

这是核心训练函数，负责训练单个模型（intra-chr或inter-chr）。

#### 输入参数

```python
def train_model(
    model,              # 模型实例（IntraChrModel或InterChrModel）
    train_loader,      # 训练数据加载器
    val_loader,         # 验证数据加载器
    criterion,          # 损失函数（BCELoss或MSELoss）
    optimizer,          # 优化器（Adam）
    device,             # 设备（CPU或GPU）
    epochs=50,          # 训练轮数
    phenotype_type='binary'  # 表型类型（'binary'或'continuous'）
):
```

#### 训练流程

**1. 初始化**
```python
model.to(device)  # 将模型移到GPU或CPU
best_metric = 0.0  # 记录最佳验证指标
history = {...}    # 记录训练历史
```

**2. 每个Epoch的训练阶段**

```python
for epoch in range(epochs):
    model.train()  # 设置为训练模式
    
    for snps, labels in train_loader:
        # 前向传播
        pred, main_weights, epi_pairs, epi_scores = model(snps)
        
        # 计算损失
        loss_cls = criterion(pred, labels)  # 分类/回归损失
        loss_main = 0.01 * torch.norm(main_weights)  # 主效权重正则化
        loss_epi = 0.01 * epi_scores.mean()  # 互作分数正则化
        total_loss = loss_cls + loss_main + loss_epi
        
        # 反向传播
        total_loss.backward()
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清零梯度
```

**损失函数组成**：
- `loss_cls`: 主要损失，确保预测准确
- `loss_main`: 正则化项，防止主效权重过大
- `loss_epi`: 正则化项，控制互作分数

**3. 每个Epoch的验证阶段**

```python
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度（节省内存）
        for snps, labels in val_loader:
            pred, main_weights, epi_pairs, epi_scores = model(snps)
            # 计算验证损失和指标
```

**4. 计算评估指标**

```python
if phenotype_type == 'binary':
    # 二分类：使用AUC（ROC曲线下面积）
    val_metric = roc_auc_score(val_labels, val_preds)
else:
    # 连续型：使用R²（决定系数）
    corr, _ = pearsonr(val_preds, val_labels)
    val_metric = corr ** 2
```

**5. 记录最佳指标**

```python
if val_metric > best_metric:
    best_metric = val_metric  # 更新最佳指标
    # 注意：这里没有保存模型，最终返回的是训练完所有epoch的模型
```

#### 返回值

```python
return model, best_metric, history
```

- `model`: 训练完成后的模型（训练了所有epoch）
- `best_metric`: 验证集上的最佳指标值
- `history`: 包含训练历史的字典

**重要**：返回的模型是训练完所有epoch的模型，不是best_metric对应的模型。

---

### 函数2: `integrate_results`

这个函数整合所有模型（intra-chr和inter-chr）的结果。

#### 输入参数

```python
def integrate_results(intra_results, inter_results, snp_info):
    """
    intra_results: dict
        key: 染色体ID (如 'chr1')
        value: (main_weights, snp_idx, epi_pairs, epi_scores)
    
    inter_results: dict
        key: (chr1, chr2) 元组
        value: (main_weights, chr1_snp_idx, chr2_snp_idx, epi_pairs, epi_scores)
    
    snp_info: DataFrame
        包含SNP信息（CHROM, POS, ID）
    """
```

#### 处理流程

**1. 整合主效位点**

```python
# 收集所有主效权重
all_main = []
for chr_id, (main_weights, snp_idx, _, _) in intra_results.items():
    # 将权重和SNP信息关联
    for idx, weight in enumerate(main_weights):
        all_main.append({
            'SNP_ID': snp_info.iloc[snp_idx[idx]]['ID'],
            'CHROM': chr_id,
            'Main_Effect_Weight': weight,
            'Type': 'intra-chr'
        })

# 排序并取Top10
main_df = (
    pd.DataFrame(all_main)
    .drop_duplicates('SNP_ID')  # 去重（同一个SNP可能出现在多个模型中）
    .sort_values('Main_Effect_Weight', ascending=False)  # 按权重降序
    .head(10)  # 取Top10
)
```

**2. 整合互作对**

```python
# 收集所有互作对
all_epi = []
for chr_id, (_, snp_idx, epi_pairs, epi_scores) in intra_results.items():
    chr_snp_ids = snp_info.iloc[snp_idx]['ID'].values
    for (i, j), score in zip(epi_pairs, epi_scores):
        all_epi.append({
            'SNP1': chr_snp_ids[i],
            'SNP2': chr_snp_ids[j],
            'CHROM1': chr_id,
            'CHROM2': chr_id,
            'Pair_Type': 'intra-chr',
            'Epistatic_Score': score
        })

# 排序并取Top10
epi_df = pd.DataFrame(all_epi).sort_values(
    'Epistatic_Score', ascending=False
).head(10)
```

#### 返回值

```python
return main_df, epi_df
```

- `main_df`: Top10主效位点的DataFrame
- `epi_df`: Top10互作对的DataFrame

---

## 总结

1. **检测结果基于模型学习**：主效位点和互作对是根据模型学习到的权重和分数选择的，不是基于模拟时指定的位点。

2. **best_metric的作用**：只用于评估模型性能，不用于选择位点。

3. **热图vs文本文件**：热图只显示Top6，文本文件显示Top10，所以标记名可能不同。

4. **训练流程**：模型通过反向传播学习，最终返回训练完所有epoch的模型。

5. **结果整合**：`integrate_results`函数将所有模型的结果合并，按权重/分数排序取Top10。

