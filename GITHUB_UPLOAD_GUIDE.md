# GitHub 上传完整指南

## 快速开始（推荐方式）

### 方式 1: 使用命令行（推荐）

在项目根目录下依次执行以下命令：

```bash
# 1. 初始化 Git 仓库
git init

# 2. 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 3. 提交更改
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"

# 4. 添加远程仓库
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git

# 5. 设置主分支为 main
git branch -M main

# 6. 推送到 GitHub
git push -u origin main
```

### 方式 2: 使用 GitHub Desktop（图形界面）

1. 下载并安装 [GitHub Desktop](https://desktop.github.com/)
2. 打开 GitHub Desktop
3. File → Add Local Repository → 选择你的项目文件夹
4. 点击 "Publish repository"
5. 输入仓库名称：`Qi_Intra_InterChrInteraction`
6. 选择账户：`qiqizhang0325-afk`
7. 点击 "Publish Repository"

## 详细步骤说明

### 步骤 1: 初始化 Git 仓库

```bash
git init
```

这会创建一个 `.git` 文件夹，用于跟踪文件变更。

### 步骤 2: 配置 Git 用户信息（如果还没有配置）

```bash
git config user.name "Qi Zhang"
git config user.email "your@email.com"
```

或者全局配置（所有项目都使用）：

```bash
git config --global user.name "Qi Zhang"
git config --global user.email "your@email.com"
```

### 步骤 3: 检查要上传的文件

```bash
git status
```

这会显示：
- 绿色：将被添加的文件
- 红色：未跟踪的文件
- 灰色：被 .gitignore 忽略的文件

### 步骤 4: 添加文件

```bash
git add .
```

这会添加所有文件（除了 `.gitignore` 中指定的文件）。

**注意**：`.gitignore` 已经配置好，会忽略：
- `data/*` - 数据文件（VCF 文件很大，不会上传）
- `results/*` - 结果文件（运行后生成，不会上传）
- `__pycache__/` - Python 缓存
- `.venv/` - 虚拟环境

### 步骤 5: 提交更改

```bash
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"
```

提交信息应该描述这次更改的内容。

### 步骤 6: 添加远程仓库

```bash
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
```

如果已经存在 origin，先删除再添加：

```bash
git remote remove origin
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
```

### 步骤 7: 推送到 GitHub

```bash
# 设置主分支为 main
git branch -M main

# 推送到 GitHub
git push -u origin main
```

`-u` 参数会设置上游分支，以后可以直接使用 `git push`。

## 认证问题解决

### 如果遇到认证失败

GitHub 不再支持密码认证，需要使用：

#### 方法 1: Personal Access Token (推荐)

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 选择权限：至少选择 `repo` 权限
4. 生成 token 并复制
5. 推送时，用户名输入：`qiqizhang0325-afk`，密码输入：`你的token`

#### 方法 2: SSH 密钥

1. 生成 SSH 密钥：
   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   ```

2. 添加 SSH 密钥到 GitHub：
   - 复制 `~/.ssh/id_ed25519.pub` 的内容
   - GitHub Settings → SSH and GPG keys → New SSH key

3. 更改远程 URL：
   ```bash
   git remote set-url origin git@github.com:qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
   ```

## 验证上传

上传成功后：

1. 访问：https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction
2. 检查文件是否都在
3. 检查 README.md 是否正确显示

## 后续更新

以后更新代码时：

```bash
# 1. 查看更改
git status

# 2. 添加更改的文件
git add .

# 3. 提交
git commit -m "Description of your changes"

# 4. 推送
git push
```

## 重要提示

1. **不要上传大文件**：
   - VCF 文件应该在 `data/` 目录，已被 `.gitignore` 忽略
   - 如果 VCF 文件很大，确保它在 `data/` 目录中

2. **不要上传敏感信息**：
   - API 密钥
   - 密码
   - 个人信息

3. **检查 .gitignore**：
   - 确保 `data/*` 和 `results/*` 被忽略
   - 确保 `__pycache__/` 被忽略

## 一键上传脚本

你也可以创建一个批处理文件 `upload_to_github.bat`：

```batch
@echo off
git init
git add .
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"
git branch -M main
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
git push -u origin main
pause
```

然后双击运行即可。
