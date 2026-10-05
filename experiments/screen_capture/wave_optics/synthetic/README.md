# 波光学离焦与预滤波 / synthetic

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

虚拟输入上的结构、收敛或速度诊断；没有真实设备标定，不能作为实拍保真度证据。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先审阅虚拟参数和积分/采样约定，再运行对应数值对照；收敛和速度是独立验证问题。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_wave_prefilter_cli.py](audit_wave_prefilter_cli.py) | 核查结构、配对或假设 | Exercise the public single-image CLI for the bounded wave prefilter path. |
| [run_wave_display_prefilter.py](run_wave_display_prefilter.py) | 执行明确的实验或对照 | Experimental fast display-domain integration of a wave-optical screen PSF. |
| [run_wave_optical_defocus_psf.py](run_wave_optical_defocus_psf.py) | 执行明确的实验或对照 | Compare a scalar circular-pupil PSF to the current Airy*disk approximation. |
| [run_wave_prefilter_scale.py](run_wave_prefilter_scale.py) | 执行明确的实验或对照 | Independent virtual display-scale and emitter-fill stress for wave prefilter. |
| [plot_wave_optical_defocus.py](plot_wave_optical_defocus.py) | 生成诊断图 | Visualize optical-PSF approximation error under one virtual setup. |
| [benchmark_wave_defocus_render.py](benchmark_wave_defocus_render.py) | 测量数值方法耗时 | Measure first/repeated forward-render cost for the two optical branches. |
| [benchmark_wave_prefilter_tiled.py](benchmark_wave_prefilter_tiled.py) | 测量数值方法耗时 | Virtual larger-frame wave prefilter tile seam and runtime probe. |

## 跨问题协议依赖

- [screen_capture/focus_response/synthetic/run_physical_focus_display_lattice.py](../../focus_response/synthetic/run_physical_focus_display_lattice.py)
