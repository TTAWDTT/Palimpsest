# 再数字化统计原型 / rr

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

RR 发布包内部的历史统计试验；缺逐图物理方式标签，内部留出不是独立实拍验证。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成数据准备与已知精确重叠排除，核对冻结来源角色；拟合仅用校准部分，物化与评价使用同源对照。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_redigital_encoding_strata.py](audit_redigital_encoding_strata.py) | 核查结构、配对或假设 | Describe RR redigital geometry by observable JPEG subsampling stratum. |
| [audit_redigital_label_encoding.py](audit_redigital_label_encoding.py) | 核查结构、配对或假设 | Audit JPEG header profiles by source label in released RR redigital images. |
| [audit_redigital_spectrum.py](audit_redigital_spectrum.py) | 核查结构、配对或假设 | Measure paired low/mid/high frequency changes in RR redigital strata. |
| [run_color_pilot.py](run_color_pilot.py) | 执行明确的实验或对照 | Test a class-blind photometric simulator on RR's observable 4:2:0 stratum. |
| [run_spectral_pilot.py](run_spectral_pilot.py) | 执行明确的实验或对照 | Pilot a class-balanced phase-randomized residual for RR 4:2:0 images. |
| [evaluate_redigital_420_color_critic.py](evaluate_redigital_420_color_critic.py) | 评测、比较或汇总 | Evaluate whether five paired features distinguish RR 4:2:0 from color simulation. |
| [evaluate_redigital_detector_strata.py](evaluate_redigital_detector_strata.py) | 评测、比较或汇总 | Describe fixed-detector degradation in RR JPEG subsampling strata. |
| [evaluate_redigital_spectral_critic.py](evaluate_redigital_spectral_critic.py) | 评测、比较或汇总 | Compare small RR redigital simulations with source-disjoint two-sample critics. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `MANIFEST = DATA_ROOT / 'manifests/rr_test_files.csv'`
- `ROOT = DATA_ROOT / 'derived/rr_test'`

## 跨问题协议依赖

- [origin_detection/platform_statistics/rr/evaluate_simulator_critic.py](../../platform_statistics/rr/evaluate_simulator_critic.py)
- [origin_detection/platform_statistics/rr/protocol.py](../../platform_statistics/rr/protocol.py)
