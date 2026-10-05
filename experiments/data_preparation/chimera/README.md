# 数据准备 / chimera

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_bfree_manifest.py](prepare_bfree_manifest.py) | 准备输入或物化对照 | Decode all source and genuine recapture images and sign an inference manifest. |
| [prepare_data_ranges.py](prepare_data_ranges.py) | 准备输入或物化对照 | Download the public Zenodo data archive with checked independent byte ranges. |
| [prepare_registry_simulation_split.py](prepare_registry_simulation_split.py) | 准备输入或物化对照 | Freeze source-level, stratified Chimera splits before simulator calibration. |
| [prepare_size_control.py](prepare_size_control.py) | 准备输入或物化对照 | Create a frozen 256x256 recapture control for Chimera B-Free evaluation. |
| [audit_archive.py](audit_archive.py) | 核查结构、配对或假设 | Audit official Chimera Zenodo data archive without extracting it wholesale. |
| [audit_content_pairs.py](audit_content_pairs.py) | 核查结构、配对或假设 | Check released Chimera filename pairing with image-content pHashes. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/chimera_data.tar.gz'`
- `AUDIT = WORK_DIR / 'chimera_simulation_source_split.json'`
- `MANIFEST = DATA_ROOT / 'manifests/chimera_bfree_manifest.csv'`
- `ROOT = DATA_ROOT / 'derived/chimera_paired'`
