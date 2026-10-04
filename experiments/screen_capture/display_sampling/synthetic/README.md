# 显示采样与积分 / synthetic

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

虚拟输入上的结构、收敛或速度诊断；没有真实设备标定，不能作为实拍保真度证据。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先审阅虚拟参数和积分/采样约定，再运行对应数值对照；收敛和速度是独立验证问题。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_alias_aperture_sweep.py](audit_alias_aperture_sweep.py) | 核查结构、配对或假设 | Check coupled screen-grid and image-detail response of the prototype. |
| [run_airy_alias.py](run_airy_alias.py) | 执行明确的实验或对照 | Structural Airy/alias sweep at conditional LCD projection pitches. |
| [run_aperture_alias_noise_tradeoff.py](run_aperture_alias_noise_tradeoff.py) | 执行明确的实验或对照 | Virtual aperture intervention: presampling LCD alias and photon budget. |
| [run_aperture_throughput.py](run_aperture_throughput.py) | 执行明确的实验或对照 | Ideal fixed-geometry aperture/exposure intervention, not device calibration. |
| [run_display_prefilter.py](run_display_prefilter.py) | 执行明确的实验或对照 | Prototype a fast tilted-screen display-raster prefilter approximation. |
| [run_display_prefilter_stress.py](run_display_prefilter_stress.py) | 执行明确的实验或对照 | Stress the fast display-prefilter candidate beyond near-front-facing poses. |
| [run_projective_convergence.py](run_projective_convergence.py) | 执行明确的实验或对照 | Check numerical convergence before designing a fast tilted-screen renderer. |
| [benchmark_analytic.py](benchmark_analytic.py) | 测量数值方法耗时 | Compare the analytic frontal Gaussian screen renderer with fine quadrature. |
