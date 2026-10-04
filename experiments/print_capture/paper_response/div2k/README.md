# 纸面与相机响应 / div2k

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_photo_paper.py](run_photo_paper.py) | 执行明确的实验或对照 | Fixed continuous-tone print hypothesis on the three verified DIV2K-SCAN pairs. |
| [run_print_camera.py](run_print_camera.py) | 执行明确的实验或对照 | Fixed-parameter, non-calibrated print->paper->camera probe on three real pairs. |
| [run_print_hypotheses_100.py](run_print_hypotheses_100.py) | 执行明确的实验或对照 | Fixed, uncalibrated print->camera hypotheses on audited DIV2K-SCAN pairs. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = WORK_DIR / 'div2k_scan_100_pair_audit.json'`

## 跨问题协议依赖

- [data_preparation/div2k/audit_scan_test.py](../../../data_preparation/div2k/audit_scan_test.py)
