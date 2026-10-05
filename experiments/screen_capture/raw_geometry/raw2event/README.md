# RAW/RGB 几何对应 / raw2event

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_content_feature_geometry.py](audit_content_feature_geometry.py) | 核查结构、配对或假设 | Independent full-frame feature check of frozen calibration RAW/RGB affine. |
| [audit_sensor_geometry.py](audit_sensor_geometry.py) | 核查结构、配对或假设 | Check whether one calibration-only RAW-to-ISP transform transfers to new captures. |
| [audit_split_content.py](audit_split_content.py) | 核查结构、配对或假设 | Independently test the filename-to-CIFAR mapping on frozen captured sources. |
| [audit_split_first_frames.py](audit_split_first_frames.py) | 核查结构、配对或假设 | Audit the first frames of the frozen Raw2Event calibration/development split. |
| [fit_content_geometry.py](fit_content_geometry.py) | 仅在既定校准部分拟合 | Diagnose content registration using known source and captured ISP-RGB only. |
| [evaluate_geometry_transfer.py](evaluate_geometry_transfer.py) | 评测、比较或汇总 | Exploratory RAW prediction change from replacing unstable per-Tag extrapolation. |
| [evaluate_global_geometry_process.py](evaluate_global_geometry_process.py) | 评测、比较或汇总 | Recalibrate effective RAW counts after replacing RAW/RGB Tag extrapolation. |
| [evaluate_raw_registration.py](evaluate_raw_registration.py) | 评测、比较或汇总 | Target-RAW geometry upper bound; diagnostic only, never a heldout score. |
| [plot_registration_diagnostic.py](plot_registration_diagnostic.py) | 生成诊断图 | Visualize selected source/RGB/RAW registration failures for diagnosis. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `ARCHIVE = DATA_ROOT / 'raw/cifar10_official/cifar-10-python.tar.gz'`
- `GEOMETRY = WORK_DIR / 'raw2event_process_split_first_frame_audit.json'`
- `MANIFEST = DATA_ROOT / 'manifests/raw2event_phase_confirmation_v1.csv'`
- `REFERENCE_AUDIT = WORK_DIR / 'raw2event_probe_pixel_audit.json'`
- `SPLIT = DATA_ROOT / 'manifests/raw2event_process_split_v1.csv'`

## 跨问题协议依赖

- [data_preparation/raw2event/prepare_download_cifar10_python.py](../../../data_preparation/raw2event/prepare_download_cifar10_python.py)
- [screen_capture/cfa_phase/raw2event/evaluate_phase_confirmation.py](../../cfa_phase/raw2event/evaluate_phase_confirmation.py)
- [screen_capture/cfa_phase/raw2event/run_cfa_phase.py](../../cfa_phase/raw2event/run_cfa_phase.py)
- [screen_capture/source_matching/raw2event/protocol.py](../../source_matching/raw2event/protocol.py)
- [screen_capture/source_to_raw/raw2event/protocol.py](../../source_to_raw/raw2event/protocol.py)
- [screen_capture/spectral_response/raw2event/protocol.py](../../spectral_response/raw2event/protocol.py)
