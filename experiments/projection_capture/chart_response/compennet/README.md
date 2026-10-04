# 色卡有效光度 / compennet

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_channel_basis.py](run_channel_basis.py) | 执行明确的实验或对照 | Falsify an additive projector-channel/surface-mixing hypothesis. |
| [run_channel_transfer.py](run_channel_transfer.py) | 执行明确的实验或对照 | Transfer pure-channel tone curves while recalibrating four target charts. |
| [run_chart_model.py](run_chart_model.py) | 执行明确的实验或对照 | Test chart-only projector forward models on held-out textured inputs. |
| [run_full_lut.py](run_full_lut.py) | 执行明确的实验或对照 | High-capacity pixelwise 3D-LUT control for the chart-only forward probe. |
| [run_transfer.py](run_transfer.py) | 执行明确的实验或对照 | Can a chart-calibrated effective projector response transfer to another setup? |
| [evaluate_light_setups.py](evaluate_light_setups.py) | 评测、比较或汇总 | Compare true structured-light decodes on two cloud_np CompenNet++ setups. |
| [evaluate_residuals.py](evaluate_residuals.py) | 评测、比较或汇总 | Diagnose where uniform-chart prediction fails on real projected textures. |
| [benchmark_forward_speed.py](benchmark_forward_speed.py) | 测量数值方法耗时 | CPU simulation-kernel timing on one 256px CompenNet input. |
| [protocol.py](protocol.py) | 共享协议；不直接执行 | Test chart-only projector forward models on held-out textured inputs. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/compennet/CompenNetDataset.zip'`
- `AUDIT = WORK_DIR / 'compennet_full_audit.json'`

## 跨问题协议依赖

- [projection_capture/structured_light/compennetpp/evaluate_geometry_surrogates.py](../../structured_light/compennetpp/evaluate_geometry_surrogates.py)
- [projection_capture/structured_light/compennetpp/protocol.py](../../structured_light/compennetpp/protocol.py)
