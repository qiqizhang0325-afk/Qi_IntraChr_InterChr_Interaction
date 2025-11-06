# 元数据文件说明 (Metadata Files Explanation)

本文档解释项目中各个元数据文件的作用，以及是否需要修改或删除。

## 文件分类

### ✅ **必需文件**（建议保留）

#### 1. LICENSE
**作用**：定义软件的使用许可，告诉别人如何使用你的代码。

**当前状态**：使用 Unlicense（公共领域，完全自由使用）

**是否需要修改**：
- ✅ **建议保留**：没有 LICENSE 的文件在法律上默认保留所有权利，别人无法使用
- ⚠️ **可选修改**：如果你想要其他许可证（如 MIT、Apache 2.0），可以替换
- 如果 Unlicense 符合你的需求，**无需修改**

**重要性**：⭐⭐⭐⭐⭐（非常重要）

---

### ✅ **推荐保留**（提升项目专业性）

#### 2. CODE_OF_CONDUCT.md
**作用**：定义项目的行为准则，说明参与项目的规则和期望。

**当前状态**：标准的 Contributor Covenant 行为准则

**是否需要修改**：
- ✅ **建议保留**：特别是如果你计划：
  - 在 GitHub 上公开发布
  - 接受其他人的贡献
  - 与团队协作
- ⚠️ **可以删除**：如果只是个人项目，不打算接受贡献
- 如果保留，**通常无需修改**（已经是标准模板）

**重要性**：⭐⭐⭐⭐（如果公开发布）

---

#### 3. CONTRIBUTING.md
**作用**：说明如何为项目做贡献（提交代码、报告问题等）。

**当前状态**：模板文件，包含 TODO

**是否需要修改**：
- ✅ **建议保留并更新**：如果你希望别人贡献代码
- ⚠️ **可以删除**：如果只是个人项目，不接受贡献
- 如果保留，**需要修改**：添加你的具体贡献指南

**重要性**：⭐⭐⭐（如果接受贡献）

---

#### 4. .pre-commit-config.yaml
**作用**：配置 pre-commit hooks，在每次提交代码前自动运行代码质量检查。

**当前状态**：已配置好，包含：
- YAML 文件检查
- ruff 代码格式化和 linting
- pytest 测试

**是否需要修改**：
- ✅ **强烈建议保留**：自动确保代码质量
- ⚠️ **可以删除**：如果你不想使用自动化检查
- 如果保留，**通常无需修改**（已经配置好了）

**使用方法**：
```bash
uv run pre-commit install  # 安装 hooks
```

**重要性**：⭐⭐⭐⭐（提升代码质量）

---

### ❌ **可以删除**

#### 5. README_TEMPLATE.md
**作用**：README 的模板文件，用于创建新项目时的参考。

**当前状态**：模板文件，你已经有了实际的 README.md

**是否需要修改**：
- ❌ **可以删除**：你已经有了实际的 README.md，这个模板不再需要
- 或者保留作为参考（但通常不需要）

**重要性**：⭐（仅作为参考）

---

## 总结和建议

### 推荐配置（公开发布的项目）

| 文件 | 操作 | 原因 |
|------|------|------|
| LICENSE | ✅ 保留 | 必需，定义使用许可 |
| CODE_OF_CONDUCT.md | ✅ 保留 | 提升专业性，规范行为 |
| CONTRIBUTING.md | ✅ 保留并更新 | 帮助他人贡献代码 |
| .pre-commit-config.yaml | ✅ 保留 | 自动代码质量检查 |
| README_TEMPLATE.md | ❌ 删除 | 已有实际 README |

### 个人项目配置

| 文件 | 操作 | 原因 |
|------|------|------|
| LICENSE | ✅ 保留 | 必需 |
| CODE_OF_CONDUCT.md | ⚠️ 可选 | 个人项目可以删除 |
| CONTRIBUTING.md | ❌ 删除 | 不接受贡献 |
| .pre-commit-config.yaml | ✅ 保留 | 提升代码质量 |
| README_TEMPLATE.md | ❌ 删除 | 不需要 |

## 具体操作建议

### 1. LICENSE
**建议**：保留，无需修改（除非你想换其他许可证）

### 2. CODE_OF_CONDUCT.md
**建议**：保留，无需修改（已经是标准模板）

### 3. CONTRIBUTING.md
**建议**：如果保留，需要更新内容：

```markdown
# Contributing to QI Intra/Inter Chromosome Interaction Analysis

Thank you for your interest in contributing!

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

## Code Style

We use ruff for code formatting and linting. Please ensure your code passes:
- `uv run ruff check src/`
- `uv run ruff format src/`
```

### 4. .pre-commit-config.yaml
**建议**：保留，无需修改（已经配置好了）

### 5. README_TEMPLATE.md
**建议**：删除（你已经有了实际的 README.md）

## 我的最终建议

对于你的项目（公开发布到 GitHub）：

1. ✅ **保留 LICENSE** - 必需
2. ✅ **保留 CODE_OF_CONDUCT.md** - 提升专业性
3. ✅ **保留并更新 CONTRIBUTING.md** - 帮助他人贡献
4. ✅ **保留 .pre-commit-config.yaml** - 自动代码质量检查
5. ❌ **删除 README_TEMPLATE.md** - 不再需要

这样可以：
- 符合开源项目最佳实践
- 提升项目专业性
- 方便他人使用和贡献
- 自动确保代码质量

