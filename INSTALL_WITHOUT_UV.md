# 不使用 uv 的安装方法

如果你不想安装 uv，可以使用传统的 pip 方法：

## 步骤 1: 创建虚拟环境

```bash
python -m venv .venv
```

## 步骤 2: 激活虚拟环境

**Windows cmd:**
```bash
.venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

激活后，命令行前面会显示 `(.venv)`。

## 步骤 3: 安装依赖

根据 `pyproject.toml` 中的依赖，手动安装：

```bash
pip install torch>=2.0.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0
```

或者一次性安装：

```bash
pip install torch>=2.0.0 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 scipy>=1.10.0 scikit-learn>=1.3.0
```

## 步骤 4: 准备数据

```bash
copy your_file.vcf data\test.vcf
```

## 步骤 5: 运行

```bash
python src/main.py
```

## 注意事项

- 确保虚拟环境已激活（命令行前有 `(.venv)`）
- PyTorch 安装可能需要一些时间
- 如果有 GPU，PyTorch 会自动检测并使用

