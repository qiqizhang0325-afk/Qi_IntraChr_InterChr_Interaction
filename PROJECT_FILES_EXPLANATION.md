# 项目文件说明 (Project Files Explanation)

本文档解释了现代 Python 项目中各个重要文件的作用。

## 必需文件 (Required Files)

### 1. `.gitignore`
**作用**: 告诉 Git 哪些文件和目录不应该被版本控制。

**为什么需要**: 
- 避免提交临时文件、缓存文件、编译文件等到仓库
- 减少仓库大小
- 保护敏感信息（如 API 密钥、密码等）

**包含的内容**:
- Python 缓存文件 (`__pycache__/`, `*.pyc`)
- 虚拟环境 (`venv/`, `.env`)
- IDE 配置文件 (`.idea/`, `.vscode/`)
- 构建产物 (`dist/`, `build/`)
- 测试覆盖率报告 (`.coverage`, `htmlcov/`)

**你需要做什么**: 
- ✅ 已经存在，通常不需要修改
- 如果需要忽略特定文件，可以添加

### 2. `.pre-commit-config.yaml`
**作用**: 配置 pre-commit hooks，在每次提交代码前自动运行代码质量检查。

**为什么需要**:
- 自动格式化代码
- 检查代码风格和潜在错误
- 运行测试
- 确保代码质量一致性

**包含的检查**:
- `check-yaml`: 检查 YAML 文件格式
- `ruff`: Python 代码 linting 和格式化
- `pytest`: 运行单元测试

**你需要做什么**:
- ✅ 已经配置好
- 运行 `uv run pre-commit install` 来激活
- 每次提交前会自动运行这些检查

### 3. `README.md`
**作用**: 项目的主要文档，介绍项目、安装方法、使用方法等。

**为什么需要**:
- 帮助用户快速了解项目
- 提供安装和使用指南
- 展示项目功能
- GitHub 会自动在仓库首页显示

**你需要做什么**:
- ✅ 已创建，包含项目说明
- 根据实际情况更新：
  - 作者信息
  - 仓库 URL
  - 使用示例
  - 联系方式

### 4. `CHANGELOG.md`
**作用**: 记录项目的版本变更历史。

**为什么需要**:
- 让用户了解每个版本的变化
- 帮助追踪 bug 修复和新功能
- 遵循 [Keep a Changelog](https://keepachangelog.com/) 标准

**格式**:
- 按版本号组织
- 分类：Added, Changed, Deprecated, Removed, Fixed, Security

**你需要做什么**:
- ✅ 已创建初始版本
- 每次发布新版本时更新
- 记录重要的变更

### 5. `CITATION.cff`
**作用**: 提供软件引用信息，方便其他研究者引用你的工作。

**为什么需要**:
- 遵循 FAIR 原则（Findable, Accessible, Interoperable, Reusable）
- 让其他人能够正确引用你的软件
- GitHub 和 Zenodo 可以自动识别

**你需要做什么**:
- ✅ 已创建模板
- **必须更新**:
  - 作者姓名和 ORCID
  - 仓库 URL
  - 版本号
  - 其他元数据

### 6. `LICENSE`
**作用**: 定义软件的使用许可。

**为什么需要**:
- 明确他人如何使用你的代码
- 保护你的权利或允许自由使用
- 没有 LICENSE 的文件在法律上默认保留所有权利

**当前状态**:
- ✅ 使用 Unlicense（公共领域，完全自由使用）

**你需要做什么**:
- 如果 Unlicense 符合你的需求，无需修改
- 如果需要其他许可证（如 MIT, Apache 2.0），可以替换

## 配置文件 (Configuration Files)

### 7. `pyproject.toml`
**作用**: 现代 Python 项目的核心配置文件。

**包含内容**:
- 项目元数据（名称、版本、描述）
- 依赖项列表
- 构建系统配置
- 工具配置（ruff, pytest 等）

**你需要做什么**:
- ✅ 已更新项目名称和依赖
- 根据需要更新作者信息

### 8. `.github/workflows/cicd.yml`
**作用**: GitHub Actions CI/CD 工作流配置。

**为什么需要**:
- 自动运行测试
- 检查代码质量
- 确保代码可以正确安装和运行

**你需要做什么**:
- ✅ 已配置使用 uv
- 通常不需要修改，除非需要添加更多测试步骤

## 总结

| 文件 | 必需性 | 状态 | 需要做什么 |
|------|--------|------|-----------|
| `.gitignore` | ✅ 必需 | ✅ 已存在 | 通常无需修改 |
| `.pre-commit-config.yaml` | ✅ 必需 | ✅ 已配置 | 运行 `uv run pre-commit install` |
| `README.md` | ✅ 必需 | ✅ 已创建 | 更新作者和仓库信息 |
| `CHANGELOG.md` | ✅ 推荐 | ✅ 已创建 | 每次发布时更新 |
| `CITATION.cff` | ✅ 推荐 | ✅ 已创建 | **必须更新作者信息** |
| `LICENSE` | ✅ 必需 | ✅ 已存在 | 确认许可类型 |
| `pyproject.toml` | ✅ 必需 | ✅ 已更新 | 更新作者信息 |
| `.github/workflows/cicd.yml` | ✅ 推荐 | ✅ 已配置 | 通常无需修改 |

## 下一步操作

1. **更新 CITATION.cff**: 替换作者信息和仓库 URL
2. **更新 README.md**: 根据实际情况调整内容
3. **运行 pre-commit**: `uv run pre-commit install`
4. **测试安装**: `uv sync` 然后 `uv run pytest`
5. **提交代码**: 所有文件都已准备好

