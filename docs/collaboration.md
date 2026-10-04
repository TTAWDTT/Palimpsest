# 协作说明

[仓库首页](../README.md) · [开发与复算](README.md) · [研究结论](../outputs/README.md)

代码仓库的本机位置为 `E:\ai_image_origin_research\repo`，GitHub 为私有仓库 `TTAWDTT/ai-image-origin-research`。数据、环境和模型位于同级的 `data/`、`envs/`、`models/`；本机 `work/` 产物仍按复算指南管理。

原 C 盘 Codex 任务路径通过目录链接指向 E 盘仓库，避免当前任务使用两份不同的代码。目录链接只保存路径关系，代码及产物存储在 E 盘。

## 先一起整理

当前整理版作为共同起点。后续优先核查这三件事，再讨论研究恢复：

1. 研究报告中的有效结论、失败证据和待验证假设是否能清楚区分。
2. 通用模型接口是否易于理解，哪些历史实验需要保留为复算入口。
3. 下一阶段只选择一个范围明确、能够被真实数据验证的目标。

研究目前保持暂停。恢复模拟研发、下载、全量评测或训练的具体范围，待共同商定。

## 提交与讨论

- `main` 保存可阅读、经过相应检查的版本；新整理或研发使用 `ttawdtt/<主题>` 分支。
- 一次提交围绕一个具体目的，说明改动及其验证；较大的改动通过 pull request 讨论。
- 阅读文档从 `outputs/README.md` 进入；代码修改遵循 `docs/README.md` 的阶段划分。
- 原始图像、权重、虚拟环境、下载档案和本机缓存由 `.gitignore` 排除，报告和必要的关键指标进入 Git。
- 历史指标、清单和代码指纹保留原口径；目录整理不代表实验已重新运行。

本机运行：

```powershell
Set-Location E:\ai_image_origin_research\repo
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
python tools/check_layout.py
```
