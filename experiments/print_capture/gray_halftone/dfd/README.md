# 灰度网点与扫描 / dfd

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_lattice_forward_transfer.py](run_lattice_forward_transfer.py) | 执行明确的实验或对照 | Calibrate one effective square halftone lattice on D5; predict D6 peaks. |
| [run_resampling_peak_probe.py](run_resampling_peak_probe.py) | 执行明确的实验或对照 | Check whether publication-like resizing can erase genuine print lattices. |
| [run_scanner_pair.py](run_scanner_pair.py) | 执行明确的实验或对照 | Extract two scanner-labeled copies of one DFD print for a visual audit. |
| [run_tile_control.py](run_tile_control.py) | 执行明确的实验或对照 | Compare a printed uniform tile with blank paper in one DFD scan. |
| [run_tone_lattice.py](run_tone_lattice.py) | 执行明确的实验或对照 | Observed peak stability across gray tiles in two matched DFD P3 scans. |
| [evaluate_print_scan_prototype.py](evaluate_print_scan_prototype.py) | 评测、比较或汇总 | One fixed-condition sanity comparison, not a calibrated printer validation. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/dfd_halftone/HalftoneImages-BW.tar.gz'`

## 跨问题协议依赖

- [print_capture/color_halftone/dfd/run_color_matched_printers.py](../../color_halftone/dfd/run_color_matched_printers.py)
