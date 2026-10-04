# 拍屏数值与速度

[返回实验目录](../README.md)

虚拟输入上的积分、光学、曝光、对照及性能实验。

所有 Python 命令从仓库根目录以 `python -m experiments.screen_numerics.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [audit_fdnet_real_screen_pairs.py](audit_fdnet_real_screen_pairs.py) | Fetch and verify the small real-screen focus/defocus examples in FDNet authors' repo. |
| [benchmark_screen_analytic.py](benchmark_screen_analytic.py) | Compare the analytic frontal Gaussian screen renderer with fine quadrature. |
| [benchmark_wave_defocus_render.py](benchmark_wave_defocus_render.py) | Measure first/repeated forward-render cost for the two optical branches. |
| [benchmark_wave_prefilter_tiled.py](benchmark_wave_prefilter_tiled.py) | Virtual larger-frame wave prefilter tile seam and runtime probe. |
| [check_screen_alias_aperture_sweep.py](check_screen_alias_aperture_sweep.py) | Check coupled screen-grid and image-detail response of the prototype. |
| [evaluate_fdnet_focus_pairs.py](evaluate_fdnet_focus_pairs.py) | Small exploratory optical-focus intervention audit on FDNet author's examples. |
| [plot_fdnet_focus_comparison.py](plot_fdnet_focus_comparison.py) | Render a small, attributed visual comparison of verified author examples. |
| [plot_wave_optical_defocus.py](plot_wave_optical_defocus.py) | Visualize optical-PSF approximation error under one virtual setup. |
| [probe_aperture_alias_noise_tradeoff.py](probe_aperture_alias_noise_tradeoff.py) | Virtual aperture intervention: presampling LCD alias and photon budget. |
| [probe_aperture_throughput.py](probe_aperture_throughput.py) | Ideal fixed-geometry aperture/exposure intervention, not device calibration. |
| [probe_focus_before_vs_after_sampling.py](probe_focus_before_vs_after_sampling.py) | Counterexample: pre-sensor optical blur vs post-capture Gaussian blur. |
| [probe_focused_diffraction_boundary.py](probe_focused_diffraction_boundary.py) | Compare wave prefilter and fine integration away from optical crop edges. |
| [probe_focused_diffraction_prefilter.py](probe_focused_diffraction_prefilter.py) | Numerical check of fast wave prefilter for in-focus high-f-number cases. |
| [probe_physical_focus_display_lattice.py](probe_physical_focus_display_lattice.py) | Virtual physical focus intervention with coupled projection and optical PSF. |
| [probe_screen_airy_alias.py](probe_screen_airy_alias.py) | Structural Airy/alias sweep at conditional LCD projection pitches. |
| [probe_screen_display_prefilter.py](probe_screen_display_prefilter.py) | Prototype a fast tilted-screen display-raster prefilter approximation. |
| [probe_screen_display_prefilter_stress.py](probe_screen_display_prefilter_stress.py) | Stress the fast display-prefilter candidate beyond near-front-facing poses. |
| [probe_screen_projective_convergence.py](probe_screen_projective_convergence.py) | Check numerical convergence before designing a fast tilted-screen renderer. |
| [probe_wave_display_prefilter.py](probe_wave_display_prefilter.py) | Experimental fast display-domain integration of a wave-optical screen PSF. |
| [probe_wave_optical_defocus_psf.py](probe_wave_optical_defocus_psf.py) | Compare a scalar circular-pupil PSF to the current Airy*disk approximation. |
| [probe_wave_prefilter_scale.py](probe_wave_prefilter_scale.py) | Independent virtual display-scale and emitter-fill stress for wave prefilter. |
| [projective_area_reference.py](../../src/palimpsest/simulation/reference.py) | Slow exact no-blur pixel-area reference for projective screen emitters. |
| [smoke_wave_prefilter_cli.py](smoke_wave_prefilter_cli.py) | Exercise the public single-image CLI for the bounded wave prefilter path. |
