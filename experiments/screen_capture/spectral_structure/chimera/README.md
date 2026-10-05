# 频谱结构与混杂 / chimera

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_native_bandpower.py](run_native_bandpower.py) | 执行明确的实验或对照 | Contrast native recapture high-band power with pure digital resampling. |
| [run_native_channel_coherence.py](run_native_channel_coherence.py) | 执行明确的实验或对照 | Compare high-band RGB channel coherence in real recapture and digital controls. |
| [run_native_frequency.py](run_native_frequency.py) | 执行明确的实验或对照 | Search for repeatable native-pixel spectral peaks in true screen recaptures. |
| [run_residual_spectrum.py](run_residual_spectrum.py) | 执行明确的实验或对照 | Probe unfitted real-minus-proxy spatial residuals on development sources. |
| [evaluate_highband_confounders.py](evaluate_highband_confounders.py) | 评测、比较或汇总 | Test whether native high-band excess tracks source content and scene. |
| [protocol.py](protocol.py) | 共享协议；不直接执行 | Search for repeatable native-pixel spectral peaks in true screen recaptures. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `GEOMETRY = WORK_DIR / 'chimera_fixed_publication_geometry.json'`
- `ROOT = DATA_ROOT / 'derived/chimera_paired'`
- `SPLIT = DATA_ROOT / 'manifests/chimera_simulation_source_split.csv'`

## 跨问题协议依赖

- [screen_capture/photometry/chimera/evaluate_composite_photometry.py](../../photometry/chimera/evaluate_composite_photometry.py)
- [screen_capture/photometry/chimera/protocol.py](../../photometry/chimera/protocol.py)
