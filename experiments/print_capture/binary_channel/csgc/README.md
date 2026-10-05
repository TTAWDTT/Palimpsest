# 二值模板有效通道 / csgc

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [fit_bounded_power_tone.py](fit_bounded_power_tone.py) | 仅在既定校准部分拟合 | Diagnostic: can one bounded reflectance-plus-power curve explain CSGC? |
| [fit_effective_channel.py](fit_effective_channel.py) | 仅在既定校准部分拟合 | Estimate a *composite* blur/linear-tone diagnostic on known CSGC pairs. |
| [fit_nonlinear_tone.py](fit_nonlinear_tone.py) | 仅在既定校准部分拟合 | Test a bounded monotone *composite* transfer on held-out CSGC templates. |
| [evaluate_cross_spi_four_sources.py](evaluate_cross_spi_four_sources.py) | 评测、比较或汇总 | Small falsification probe: transfer 2400-spi composite parameters to other SPI. |
| [evaluate_cross_spi_twelve_sources.py](evaluate_cross_spi_twelve_sources.py) | 评测、比较或汇总 | Fixed-parameter CSGC forward transfer on 12 cross-SPI digital templates. |
| [evaluate_forward.py](evaluate_forward.py) | 评测、比较或汇总 | Run physical-coordinate binary print-scan prototype on 850 held-out codes. |
| [evaluate_forward_residuals.py](evaluate_forward_residuals.py) | 评测、比较或汇总 | Probe held-out CSGC residual structure of the fixed forward prototype. |
| [evaluate_residual_conditioning.py](evaluate_residual_conditioning.py) | 评测、比较或汇总 | Diagnostic of which fixed-input structures explain CSGC forward residuals. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/csgc/CSGC_scan2400spi.zip'`
- `PROBE = DATA_ROOT / 'derived/csgc_probe'`
