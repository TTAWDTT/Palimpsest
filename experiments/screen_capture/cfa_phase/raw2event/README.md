# CFA 相位与时间诊断 / raw2event

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [run_cfa_phase.py](run_cfa_phase.py) | 执行明确的实验或对照 | Four CFA phase hypotheses on the same cached screen irradiance fields. |
| [run_flat_field_grid.py](run_flat_field_grid.py) | 执行明确的实验或对照 | Exploratory LCD-grid spectral check in blank top bands of real RAW frames. |
| [run_grid_phase_motion.py](run_grid_phase_motion.py) | 执行明确的实验或对照 | Check whether a flat-field spectral peak follows screen motion or sensor. |
| [run_temporal_lag_control.py](run_temporal_lag_control.py) | 执行明确的实验或对照 | Test whether flat-region temporal residuals grow with frame lag. |
| [run_temporal_residual.py](run_temporal_residual.py) | 执行明确的实验或对照 | Explore temporal RAW differences in locally flat CFA planes. |
| [evaluate_phase_confirmation.py](evaluate_phase_confirmation.py) | 评测、比较或汇总 | One-pass source-disjoint confirmation of the frozen Raw2Event CFA hypotheses. |
| [evaluate_summary_cfa_phase.py](evaluate_summary_cfa_phase.py) | 评测、比较或汇总 | Source-level paired descriptions for the exploratory CFA phase probe. |
| [evaluate_summary_phase_confirmation.py](evaluate_summary_phase_confirmation.py) | 评测、比较或汇总 | Paired, source-level summary of the one-pass phase confirmation sample. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `GEOMETRY = WORK_DIR / 'raw2event_content_registered_geometry.json'`
- `MANIFEST = DATA_ROOT / 'manifests/raw2event_phase_confirmation_v1.csv'`
- `ROOT = DATA_ROOT / 'raw/raw2event_probe/frames_raw'`
- `SPLIT = DATA_ROOT / 'manifests/raw2event_process_split_v1.csv'`

## 跨问题协议依赖

- [data_preparation/raw2event/prepare_download_cifar10_python.py](../../../data_preparation/raw2event/prepare_download_cifar10_python.py)
- [screen_capture/raw_geometry/raw2event/audit_split_content.py](../../raw_geometry/raw2event/audit_split_content.py)
- [screen_capture/raw_geometry/raw2event/audit_split_first_frames.py](../../raw_geometry/raw2event/audit_split_first_frames.py)
- [screen_capture/raw_geometry/raw2event/fit_content_geometry.py](../../raw_geometry/raw2event/fit_content_geometry.py)
- [screen_capture/source_matching/raw2event/protocol.py](../../source_matching/raw2event/protocol.py)
- [screen_capture/source_to_raw/raw2event/protocol.py](../../source_to_raw/raw2event/protocol.py)
- [screen_capture/spectral_response/raw2event/protocol.py](../../spectral_response/raw2event/protocol.py)
