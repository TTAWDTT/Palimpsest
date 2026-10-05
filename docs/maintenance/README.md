# 维护记录

[当前实验入口](../../experiments/README.md) · [代码结构](../architecture.md)

| 文件 | 用途 |
|---|---|
| [处理清单](experiment_refactor_checklist.md) | 本轮审查结论、处理顺序与完成情况 |
| [处理结果](experiment_refactor_result.md) | 修复、提取、验证证据与保留边界 |
| [迁移表](experiment_migration.csv) | 旧文件到当前入口及提取模块 |
| [入口目录表](experiment_catalog.csv) | 当前角色、研究问题、数据范围与目的 |
| [审查原始记录](inventory_before.json) | 整理前的模块、导入和顶层操作；供机器核对 |
| [迁移映射](experiment_migration.json) | 静态检查使用的旧入口映射 |
| [提取映射](experiment_extractions.json) | 静态检查使用的共享实现归属 |

`legacy_experiment_indexes/` 保留上一轮分类的历史导航，当前阅读使用实验首页。Git 版本 `21a4b1e` 保存本轮整理前源码；此目录不包含原始数据、权重或本机断点。
