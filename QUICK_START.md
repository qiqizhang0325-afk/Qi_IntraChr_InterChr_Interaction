# 快速开始指南

## 从 GitHub 克隆项目到本地

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction
```

### 步骤 2: 安装依赖

项目使用 `uv` 作为包管理器。如果你还没有安装 `uv`，请先安装：

**Windows:**

**方法 1: 使用 PowerShell**（推荐）
```powershell
# 打开 PowerShell（不是 cmd.exe）
irm https://astral.sh/uv/install.ps1 | iex
```

**方法 2: 使用 pip**（如果已有 Python）
```bash
pip install uv
```

**方法 3: 不使用 uv**（使用传统 pip）
参见 [INSTALL_WITHOUT_UV.md](INSTALL_WITHOUT_UV.md)

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后安装项目依赖：

```bash
uv sync
```

这会：
- 创建虚拟环境
- 安装所有依赖（PyTorch, NumPy, Pandas 等）
- 创建 `uv.lock` 文件（锁定依赖版本）

### 步骤 3: 准备数据文件

将你的 VCF 文件放到 `data/` 目录：

```bash
# 将你的 VCF 文件复制到 data/ 目录
# 默认文件名是 test.vcf
copy your_file.vcf data/test.vcf
```

或者使用不同的文件名，然后修改 `src/main.py` 中的路径。

### 步骤 4: 运行分析

```bash
uv run python src/main.py
```

或者激活虚拟环境后运行：

```bash
# Windows
.venv\Scripts\activate
python src/main.py

# Linux/macOS
source .venv/bin/activate
python src/main.py
```

### 步骤 5: 查看结果

所有结果会保存在 `results/` 目录：

- **文本结果**：
  - `results/main_effect_results_top10.txt` - Top10 主效位点
  - `results/main_effect_results_all.txt` - 所有主效位点
  - `results/epistatic_interactions_top10.txt` - Top10 互作对
  - `results/epistatic_interactions_all.txt` - 所有互作对
  - `results/training_history.txt` - 训练历史

- **可视化结果**：
  - `results/main_effect_manhattan_top10.png` - Top10 Manhattan 图
  - `results/main_effect_manhattan_all.png` - 所有位点 Manhattan 图
  - `results/epistatic_heatmap_top10.png` - Top10 热图
  - `results/epistatic_heatmap_all.png` - 所有互作热图
  - `results/training_curves_*.png` - 训练曲线

## 配置参数（可选）

如果需要修改参数，编辑 `src/main.py`：

```python
# 模型配置
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128

# 表型配置
PHENOTYPE_TYPE = 'continuous'  # 或 'binary'
HERITABILITY = 0.9
```

## 常见问题

### 问题 1: uv 命令不存在

安装 uv：
- Windows: `irm https://astral.sh/uv/install.ps1 | iex`
- Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 问题 2: 找不到 VCF 文件

确保 VCF 文件在 `data/` 目录中，文件名是 `test.vcf`，或者修改 `src/main.py` 中的路径。

### 问题 3: CUDA 错误

如果没有 GPU，代码会自动使用 CPU。如果想强制使用 CPU，可以在 `src/main.py` 中修改：

```python
DEVICE = torch.device("cpu")  # 强制使用 CPU
```

### 问题 4: 内存不足

如果数据很大，可以：
- 减少 `BATCH_SIZE`
- 减少 `EPOCHS`
- 使用更小的 `HIDDEN_DIM`

## 完整工作流程示例

```bash
# 1. 克隆项目
git clone https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
cd Qi_Intra_InterChrInteraction

# 2. 安装依赖
uv sync

# 3. 准备数据（将你的 VCF 文件放到 data/ 目录）
# 假设你的文件是 my_data.vcf
copy my_data.vcf data/test.vcf

# 4. 运行分析
uv run python src/main.py

# 5. 查看结果
# 打开 results/ 目录查看所有结果文件
```

