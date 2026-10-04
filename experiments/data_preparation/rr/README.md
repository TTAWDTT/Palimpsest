# 数据准备 / rr

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_registry_simulation_split.py](prepare_registry_simulation_split.py) | 准备输入或物化对照 | Record RR sources already used for simulation development and reserve the rest. |
| [prepare_test.py](prepare_test.py) | 准备输入或物化对照 | Safely extract the verified RRDataset test archive and index every file. |
| [prepare_trainval.ps1](prepare_trainval.ps1) | 准备输入或物化对照 | 历史 PowerShell 入口；执行前核对长度/MD5/断点要求。 |
| [audit_manifest_provenance.py](audit_manifest_provenance.py) | 核查结构、配对或假设 | Summarize paired RR test provenance signals available from the verified manifest. |
| [audit_rewind_archive.py](audit_rewind_archive.py) | 核查结构、配对或假设 | Match ReWIND's image archive to its official CSV and verify extracted MD5s. |
| [audit_test.py](audit_test.py) | 核查结构、配对或假设 | Validate RRDataset test images and derive source-grouped inference manifests. |
| [audit_trainval_archive.py](audit_trainval_archive.py) | 核查结构、配对或假设 | Safely extract and index the verified RRDataset train/validation archive. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/RRDataset_test.tar.gz'`
- `MANIFEST = DATA_ROOT / 'manifests/rr_test_archive_files.csv'`
- `MANIFEST = DATA_ROOT / 'manifests/rr_test_files.csv'`
- `ROOT = DATA_ROOT / 'derived/rr_test'`

## 跨问题协议依赖

- [origin_detection/encoding_shortcuts/rr/run_oracle_codec_control.py](../../origin_detection/encoding_shortcuts/rr/run_oracle_codec_control.py)
- [origin_detection/platform_statistics/rr/protocol.py](../../origin_detection/platform_statistics/rr/protocol.py)
