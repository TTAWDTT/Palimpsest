# 数据准备 / dragotti

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_audit_bundle.py](prepare_audit_bundle.py) | 准备输入或物化对照 | Freeze the small verified Dragotti Range probe as one machine-readable record. |
| [prepare_range_members.py](prepare_range_members.py) | 准备输入或物化对照 | Extract a few verified members from Dragotti's public ZIP via HTTP Range. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ROOT = REPO_ROOT`
