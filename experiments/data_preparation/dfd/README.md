# 数据准备 / dfd

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_color.ps1](prepare_color.ps1) | 准备输入或物化对照 | 历史 PowerShell 入口；执行前核对长度/MD5/断点要求。 |
| [audit_all_image_metadata.py](audit_all_image_metadata.py) | 核查结构、配对或假设 | Collect per-image header metadata from the fully verified DFD B/W archive. |
| [audit_color_fullpage_metadata.py](audit_color_fullpage_metadata.py) | 核查结构、配对或假设 | Extract and inspect one complete DFD color TIFF per printer folder. |
| [audit_color_inventory.py](audit_color_inventory.py) | 核查结构、配对或假设 | Inventory the verified DFD color archive without extracting 25 GB of scans. |
| [audit_color_links.py](audit_color_links.py) | 核查结构、配对或假设 | Audit how cropped PNGs relate by name to DFD color whole-sheet TIFFs. |
| [audit_color_non_tiff_fullpages.py](audit_color_non_tiff_fullpages.py) | 核查结构、配对或假设 | Verify DFD color's six exceptional whole-page PNG scans and encoded DPI. |
| [audit_color_scanner_pairs.py](audit_color_scanner_pairs.py) | 核查结构、配对或假设 | Check image-content pairing for DFD's six exceptional full-page PNGs. |
| [audit_halftone.py](audit_halftone.py) | 核查结构、配对或假设 | Audit DFD's public B/W scanned-sheet archive without unpacking it. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/dfd_halftone/HalftoneImages-BW.tar.gz'`
- `ARCHIVE = DATA_ROOT / 'raw/dfd_halftone_color/HalftoneImages-Color.tar.gz'`
