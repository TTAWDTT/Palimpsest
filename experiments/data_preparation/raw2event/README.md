# 数据准备 / raw2event

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_download_cifar10_python.py](prepare_download_cifar10_python.py) | 准备输入或物化对照 | Fetch an official-MD5 CIFAR-10 archive mirror for exact stimulus lookup. |
| [prepare_download_first_raw_tar_member.py](prepare_download_first_raw_tar_member.py) | 准备输入或物化对照 | Range-extract only the first raw TAR member, not its 17.9 GB archive. |
| [prepare_download_pair.py](prepare_download_pair.py) | 准备输入或物化对照 | Download and verify selected official Raw2Event RAW/RGB/metadata prefixes. |
| [prepare_download_phase_confirmation.py](prepare_download_phase_confirmation.py) | 准备输入或物化对照 | Fetch only the ten frozen phase-confirmation Raw2Event capture triples. |
| [prepare_download_process_split.py](prepare_download_process_split.py) | 准备输入或物化对照 | Download only frozen calibration/development Raw2Event capture prefixes. |
| [prepare_download_second_tar_member.py](prepare_download_second_tar_member.py) | 准备输入或物化对照 | Range-extract one independently selected RAW TAR member with a known direct counterpart. |
| [prepare_registry_phase_confirmation.py](prepare_registry_phase_confirmation.py) | 准备输入或物化对照 | Freeze ten new, unopened Raw2Event sources for a single phase check. |
| [prepare_registry_process_split.py](prepare_registry_process_split.py) | 准备输入或物化对照 | Freeze a small, content-disjoint process-calibration split before more videos. |
| [prepare_source_paths.py](prepare_source_paths.py) | 准备输入或物化对照 | Audit whether Raw2Event filenames deterministically identify CIFAR-10 sources. |
| [audit_metadata_tar.py](audit_metadata_tar.py) | 核查结构、配对或假设 | Read the beginning of the official uncompressed metadata TAR by HTTP Range. |
| [audit_probe.py](audit_probe.py) | 核查结构、配对或假设 | Read-only structural and pixel audit of one official Raw2Event prefix. |
| [audit_tar_neighbors.py](audit_tar_neighbors.py) | 核查结构、配对或假设 | Read TAR member headers only via HTTP Range; never unpack whole archive. |
| [audit_tar_vs_direct.py](audit_tar_vs_direct.py) | 核查结构、配对或假设 | Compare one independently packaged Raw2Event RAW clip with direct views. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = WORK_DIR / 'cifar10_python_download_audit.json'`
- `AUDIT = WORK_DIR / 'raw2event_phase_confirmation_freeze.json'`
- `AUDIT = WORK_DIR / 'raw2event_process_split_v1_audit.json'`
- `MANIFEST = DATA_ROOT / 'manifests/raw2event_phase_confirmation_v1.csv'`
- `ROOT = DATA_ROOT / 'raw/cifar10_official'`
- `ROOT = DATA_ROOT / 'raw/raw2event_probe'`
- `ROOT = DATA_ROOT / 'raw/raw2event_probe/raw_tar_member'`
- `SPLIT = DATA_ROOT / 'manifests/raw2event_process_split_v1.csv'`
- `TARGET = ROOT / 'cifar-10-python.tar.gz'`
