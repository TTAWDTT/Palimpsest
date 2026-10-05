# 协作说明

[仓库首页](../README.md) · [开发与复算](README.md) · [研究结论](../reports/README.md)

代码仓库的本机位置为 `E:\ai_image_origin_research\repo`，GitHub 为公开仓库 `TTAWDTT/Palimpsest`。数据、环境和模型位于同级的 `data/`、`envs/`、`models/`；本机 `work/` 产物仍按复算指南管理。

代码和本机产物已复制并校验到 E 盘，后续以 E 盘仓库为准。原 C 盘 Codex 任务目录被进程占用，目前仍是旧副本，尚未切换成目录链接；不要在旧副本继续修改。

2026-10-04 迁移检查：8,237 个文件（1,103,849,931 字节）复制后逐文件 SHA-256 一致；E 盘环境重建后 82 项测试通过，目录与文档链接检查通过。仓库之后由用户公开；数据、权重、虚拟环境和本机缓存不提交。

前轮 PR 已合并，公开后的 [main CI](https://github.com/TTAWDTT/Palimpsest/actions/runs/37195711527) 在 `21a4b1e` 上通过。此前账户限制记录属于当时的私有仓库状态。

关闭 Codex 后，可从独立 PowerShell 完成切换：

```powershell
Set-Location E:\ai_image_origin_research\repo
powershell -NoProfile -ExecutionPolicy Bypass -File tools\finish_repo_migration.ps1 -Finalize
```

[收尾脚本](../tools/finish_repo_migration.ps1) 默认只校验，指定 `-Finalize` 才改目录。它会核对迁移审计、把 C 盘旧目录改名为备份、建立指向 E 盘的目录链接；不删除备份、不终止进程。若旧副本已发生修改或仍被占用，脚本会停止。备份待后续确认后清理。

## 先一起整理

本轮模块化已完成，见[整理记录](refactor_2026-10-04.md)和[代码结构](architecture.md)。后续优先核查这三件事，再讨论研究恢复：

1. 研究报告中的有效结论、失败证据和待验证假设是否能清楚区分。
2. 通用模型接口是否易于理解，哪些历史实验需要保留为复算入口。
3. 下一阶段只选择一个范围明确、能够被真实数据验证的目标。

研究目前保持暂停。恢复模拟研发、下载、全量评测或训练的具体范围，待共同商定。

### 2026-10-05：阶段更新

上述暂停记录属于此前整理阶段。用户已要求转向传播后 AI／非 AI 判别算法，并授权制定详细开发流程，见[当前计划](robust_ai_detection_plan.md)。本次交付为设计文档；数据使用历史审计、候选特征实验与算法实现尚待依计划执行。Simulation 研发与神经模型训练继续暂缓，旧全量 baseline 直接复用。

## 提交与讨论

2026-10-05算法迭代更新：用户已授权按流程实施，并使用BootLoops技能/工具辅助。首轮结果见[算法记录](../reports/04_算法研发/首轮局部统计算法_开发筛选与真实拍屏诊断_2026-10-05.md)：局部统计原型已实现，当前规则的真实拍屏稳健性未通过。Simulation和神经训练继续暂缓；研究不设置自动定时探索。

- `main` 保存可阅读、经过相应检查的版本；新整理或研发使用 `ttawdtt/<主题>` 分支。
- 一次提交围绕一个具体目的，说明改动及其验证；较大的改动通过 pull request 讨论。
- 阅读文档从 `reports/README.md` 进入；代码修改遵循 `docs/README.md` 的阶段划分。
- 原始图像、权重、虚拟环境、下载档案和本机缓存由 `.gitignore` 排除，报告和必要的关键指标进入 Git。
- 历史指标、清单和代码指纹保留原口径；目录整理不代表实验已重新运行。

本机运行：

```powershell
Set-Location E:\ai_image_origin_research\repo
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
python tools/check_layout.py
```
