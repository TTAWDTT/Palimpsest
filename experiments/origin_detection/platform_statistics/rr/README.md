# 平台传播统计原型 / rr

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

RR 发布包内部的历史统计试验；缺逐图物理方式标签，内部留出不是独立实拍验证。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成数据准备与已知精确重叠排除，核对冻结来源角色；拟合仅用校准部分，物化与评价使用同源对照。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_transfer_simulation.py](prepare_transfer_simulation.py) | 准备输入或物化对照 | Write development-set transfer simulations for fixed-detector score tests. |
| [audit_inputs.py](audit_inputs.py) | 核查结构、配对或假设 | Characterize paired RR transformations without assigning unknown process labels. |
| [audit_jpeg_profiles.py](audit_jpeg_profiles.py) | 核查结构、配对或假设 | Match published RR JPEG quantization profiles to Pillow encodings. |
| [audit_simulation_materialization.py](audit_simulation_materialization.py) | 核查结构、配对或假设 | Verify rendered JPEGs reproduce the simulator's recorded paired features. |
| [run_fidelity.py](run_fidelity.py) | 执行明确的实验或对照 | Run the historical RR resize/JPEG fidelity pilot; not device validation. |
| [evaluate_simulator_critic.py](evaluate_simulator_critic.py) | 评测、比较或汇总 | Measure whether simple paired features distinguish real RR processing from simulation. |
| [evaluate_transfer_geometry_effect.py](evaluate_transfer_geometry_effect.py) | 评测、比较或汇总 | Stratify frozen origin detectors by observed RR transfer geometry. |
| [protocol.py](protocol.py) | 共享协议；不直接执行 | Fit a class-blind resize/JPEG simulator and audit its held-out RR fidelity. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = BASE / 'work' / 'rr_transformation_provenance_audit.json'`
- `MANIFEST = DATA_ROOT / 'manifests/rr_transfer_simulation_dev_bfree.csv'`
