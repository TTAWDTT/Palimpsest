# Raw2Event RAW

[返回实验目录](../README.md)

数字源对应、RAW 几何、CFA、计数映射和参数可辨识性。

所有 Python 命令从仓库根目录以 `python -m experiments.raw2event.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [analyze_raw2event_split_comparison.py](analyze_raw2event_split_comparison.py) | Paired descriptive analysis of the frozen Raw2Event forward-RAW probes. |
| [audit_raw2event_content_feature_geometry.py](audit_raw2event_content_feature_geometry.py) | Independent full-frame feature check of frozen calibration RAW/RGB affine. |
| [audit_raw2event_probe.py](audit_raw2event_probe.py) | Read-only structural and pixel audit of one official Raw2Event prefix. |
| [audit_raw2event_sensor_geometry.py](audit_raw2event_sensor_geometry.py) | Check whether one calibration-only RAW-to-ISP transform transfers to new captures. |
| [audit_raw2event_split_first_frames.py](audit_raw2event_split_first_frames.py) | Audit the first frames of the frozen Raw2Event calibration/development split. |
| [audit_raw2event_tar_vs_direct.py](audit_raw2event_tar_vs_direct.py) | Compare one independently packaged Raw2Event RAW clip with direct views. |
| [diagnose_raw2event_raw_registration.py](diagnose_raw2event_raw_registration.py) | Target-RAW geometry upper bound; diagnostic only, never a heldout score. |
| [evaluate_raw2event_display_prefilter.py](evaluate_raw2event_display_prefilter.py) | Evaluate a fixed fast display-prefilter numerical approximation on real ROIs. |
| [evaluate_raw2event_geometry_transfer.py](evaluate_raw2event_geometry_transfer.py) | Exploratory RAW prediction change from replacing unstable per-Tag extrapolation. |
| [evaluate_raw2event_global_geometry_process.py](evaluate_raw2event_global_geometry_process.py) | Recalibrate effective RAW counts after replacing RAW/RGB Tag extrapolation. |
| [evaluate_raw2event_phase_confirmation.py](evaluate_raw2event_phase_confirmation.py) | One-pass source-disjoint confirmation of the frozen Raw2Event CFA hypotheses. |
| [evaluate_raw2event_source_to_raw_split.py](evaluate_raw2event_source_to_raw_split.py) | Frozen 10/10 cross-content RAW test for the screen forward renderer. |
| [evaluate_raw2event_spectral_mix.py](evaluate_raw2event_spectral_mix.py) | Fit effective display-primary to RAW-CFA coupling on frozen calibration sources. |
| [fetch_cifar10_python.py](fetch_cifar10_python.py) | Fetch an official-MD5 CIFAR-10 archive mirror for exact stimulus lookup. |
| [fetch_raw2event_first_raw_tar_member.py](fetch_raw2event_first_raw_tar_member.py) | Range-extract only the first raw TAR member, not its 17.9 GB archive. |
| [fetch_raw2event_pair.py](fetch_raw2event_pair.py) | Download and verify selected official Raw2Event RAW/RGB/metadata prefixes. |
| [fetch_raw2event_phase_confirmation.py](fetch_raw2event_phase_confirmation.py) | Fetch only the ten frozen phase-confirmation Raw2Event capture triples. |
| [fetch_raw2event_process_split.py](fetch_raw2event_process_split.py) | Download only frozen calibration/development Raw2Event capture prefixes. |
| [fetch_raw2event_second_tar_member.py](fetch_raw2event_second_tar_member.py) | Range-extract one independently selected RAW TAR member with a known direct counterpart. |
| [freeze_raw2event_phase_confirmation.py](freeze_raw2event_phase_confirmation.py) | Freeze ten new, unopened Raw2Event sources for a single phase check. |
| [freeze_raw2event_process_split.py](freeze_raw2event_process_split.py) | Freeze a small, content-disjoint process-calibration split before more videos. |
| [index_raw2event_source_paths.py](index_raw2event_source_paths.py) | Audit whether Raw2Event filenames deterministically identify CIFAR-10 sources. |
| [inspect_raw2event_metadata_tar.py](inspect_raw2event_metadata_tar.py) | Read the beginning of the official uncompressed metadata TAR by HTTP Range. |
| [inspect_raw2event_tar_neighbors.py](inspect_raw2event_tar_neighbors.py) | Read TAR member headers only via HTTP Range; never unpack whole archive. |
| [match_raw2event_cifar_source.py](match_raw2event_cifar_source.py) | Search the MD5-verified official CIFAR-10 archive for captured screen stimuli. |
| [probe_raw2event_cfa_phase.py](probe_raw2event_cfa_phase.py) | Four CFA phase hypotheses on the same cached screen irradiance fields. |
| [probe_raw2event_display_identifiability.py](probe_raw2event_display_identifiability.py) | Explore whether virtual display pitch and blur are identifiable from known RAW. |
| [probe_raw2event_fine_convergence.py](probe_raw2event_fine_convergence.py) | Check whether the fast-vs-fine spectral gap comes from fine-grid aliasing. |
| [probe_raw2event_flat_field_grid.py](probe_raw2event_flat_field_grid.py) | Exploratory LCD-grid spectral check in blank top bands of real RAW frames. |
| [probe_raw2event_grid_phase_motion.py](probe_raw2event_grid_phase_motion.py) | Check whether a flat-field spectral peak follows screen motion or sensor. |
| [probe_raw2event_isp_transfer.py](probe_raw2event_isp_transfer.py) | Fit a minimal RAW-to-RGB color chain on one Raw2Event prefix; test on another. |
| [probe_raw2event_source_to_raw.py](probe_raw2event_source_to_raw.py) | Test the existing screen forward renderer against two known-source real RAW frames. |
| [probe_raw2event_temporal_lag_control.py](probe_raw2event_temporal_lag_control.py) | Test whether flat-region temporal residuals grow with frame lag. |
| [probe_raw2event_temporal_residual.py](probe_raw2event_temporal_residual.py) | Explore temporal RAW differences in locally flat CFA planes. |
| [refine_raw2event_content_geometry.py](refine_raw2event_content_geometry.py) | Diagnose content registration using known source and captured ISP-RGB only. |
| [render_raw2event_registration_diagnostic.py](render_raw2event_registration_diagnostic.py) | Visualize selected source/RGB/RAW registration failures for diagnosis. |
| [summarize_raw2event_cfa_phase.py](summarize_raw2event_cfa_phase.py) | Source-level paired descriptions for the exploratory CFA phase probe. |
| [summarize_raw2event_phase_confirmation.py](summarize_raw2event_phase_confirmation.py) | Paired, source-level summary of the one-pass phase confirmation sample. |
| [verify_raw2event_split_content.py](verify_raw2event_split_content.py) | Independently test the filename-to-CIFAR mapping on frozen captured sources. |
