# 快速上传到 GitHub

## 方法 1: 使用批处理脚本（最简单）

我已经为你创建了 `upload_to_github.bat` 文件，直接双击运行即可！

## 方法 2: 手动执行命令

在项目根目录打开命令行，依次执行：

```bash
# 1. 添加所有文件
git add .

# 2. 提交
git commit -m "Initial commit: QI Intra/Inter Chromosome Interaction Analysis"

# 3. 设置主分支
git branch -M main

# 4. 添加远程仓库
git remote add origin https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git

# 5. 推送到 GitHub
git push -u origin main
```

## 重要：GitHub 认证

GitHub 不再支持密码登录，你需要：

### 选项 1: Personal Access Token（推荐）

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 选择权限：勾选 `repo`（完整仓库访问权限）
4. 生成并复制 token
5. 执行 `git push` 时：
   - Username: `qiqizhang0325-afk`
   - Password: `粘贴你的token`（不是密码！）

### 选项 2: 使用 SSH

```bash
# 更改远程 URL 为 SSH
git remote set-url origin git@github.com:qiqizhang0325-afk/Qi_Intra_InterChrInteraction.git
```

## 验证

上传成功后，访问：
https://github.com/qiqizhang0325-afk/Qi_Intra_InterChrInteraction

你应该能看到所有文件！

