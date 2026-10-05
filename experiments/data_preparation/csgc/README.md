# 数据准备 / csgc

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

历史获取与准入工具。目录/索引可见不等于档案已下载或逐图验证；下载只在显式执行时发生。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先检查官方索引与档案校验要求，再准备必要成员，随后核查像素/配对并冻结清单。每个数据集的可用性以报告为准。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_cross_resolution_pairs.py](prepare_cross_resolution_pairs.py) | 准备输入或物化对照 | Range-extract one CSGC source/scan pair per SPI and compare templates. |
| [prepare_multi_zip.py](prepare_multi_zip.py) | 准备输入或物化对照 | Index all three CSGC official ZIPs from verified HTTP Range tail fragments. |
| [audit_additional_pairs.py](audit_additional_pairs.py) | 核查结构、配对或假设 | Spot-check additional cross-SPI CSGC pair mappings by HTTP Range + CRC. |
| [audit_archive_2400spi.py](audit_archive_2400spi.py) | 核查结构、配对或假设 | Full-entry audit of official CSGC 2400-spi paired archive. |
| [audit_cross_spi_twelve_sources.py](audit_cross_spi_twelve_sources.py) | 核查结构、配对或假设 | Range audit 12 predetermined CSGC holdout IDs across 4800/9600 SPI. |
| [audit_pair_content.py](audit_pair_content.py) | 核查结构、配对或假设 | Check that CSGC's same-number source/scan entries are content paired. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/csgc/CSGC_scan2400spi.zip'`
