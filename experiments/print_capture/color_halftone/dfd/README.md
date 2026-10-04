# 彩色网点与扫描 / dfd

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_color_fullpage_tile.py](run_color_fullpage_tile.py) | 执行明确的实验或对照 | Measure one color tile and blank-paper control in an 800-ppi DFD full scan. |
| [run_color_matched_printers.py](run_color_matched_printers.py) | 执行明确的实验或对照 | Same-named color chart/driver setting across two HP CLJ5550 printer IDs. |
| [run_color_multitile_transfer.py](run_color_multitile_transfer.py) | 执行明确的实验或对照 | Pre-fixed four-color ROI spectral peaks across D5/D6 same nominal print setup. |
| [run_color_sample_spectra.py](run_color_sample_spectra.py) | 执行明确的实验或对照 | Exploratory 2D spectra of one selected DFD color scan per printer folder. |
