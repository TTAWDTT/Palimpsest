> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# RRDataset 与早期统计模拟

[返回实验目录](../../../experiments/README.md)

数据准入、配对、传播统计和历史统计模拟；旧原型不代表真实物理过程。

所有 Python 命令从仓库根目录以 `python -m experiments.rr.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [analyze_rr_redigital_detector_strata.py](../../../experiments/origin_detection/redigital_statistics/rr/evaluate_redigital_detector_strata.py) | Describe fixed-detector degradation in RR JPEG subsampling strata. |
| [analyze_rr_simulation_inputs.py](../../../experiments/origin_detection/platform_statistics/rr/audit_inputs.py) | Characterize paired RR transformations without assigning unknown process labels. |
| [analyze_rr_transfer_geometry_effect.py](../../../experiments/origin_detection/platform_statistics/rr/evaluate_transfer_geometry_effect.py) | Stratify frozen origin detectors by observed RR transfer geometry. |
| [audit_rewind_archive.py](../../../experiments/data_preparation/rr/audit_rewind_archive.py) | Match ReWIND's image archive to its official CSV and verify extracted MD5s. |
| [audit_rr_manifest_provenance.py](../../../experiments/data_preparation/rr/audit_manifest_provenance.py) | Summarize paired RR test provenance signals available from the verified manifest. |
| [audit_rr_redigital_encoding_strata.py](../../../experiments/origin_detection/redigital_statistics/rr/audit_redigital_encoding_strata.py) | Describe RR redigital geometry by observable JPEG subsampling stratum. |
| [audit_rr_redigital_label_encoding.py](../../../experiments/origin_detection/redigital_statistics/rr/audit_redigital_label_encoding.py) | Audit JPEG header profiles by source label in released RR redigital images. |
| [audit_rr_redigital_spectrum.py](../../../experiments/origin_detection/redigital_statistics/rr/audit_redigital_spectrum.py) | Measure paired low/mid/high frequency changes in RR redigital strata. |
| [audit_rr_test.py](../../../experiments/data_preparation/rr/audit_test.py) | Validate RRDataset test images and derive source-grouped inference manifests. |
| [audit_rr_trainval_archive.py](../../../experiments/data_preparation/rr/audit_trainval_archive.py) | Safely extract and index the verified RRDataset train/validation archive. |
| [audit_rr_transformation_provenance.py](../../../experiments/origin_detection/encoding_shortcuts/rr/audit_transformation_provenance.py) | Audit observable encoding provenance of RR transformed test images. |
| [download_rr_trainval.ps1](../../../experiments/data_preparation/rr/prepare_trainval.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [evaluate_rr_redigital_420_color_critic.py](../../../experiments/origin_detection/redigital_statistics/rr/evaluate_redigital_420_color_critic.py) | Evaluate whether five paired features distinguish RR 4:2:0 from color simulation. |
| [evaluate_rr_redigital_spectral_critic.py](../../../experiments/origin_detection/redigital_statistics/rr/evaluate_redigital_spectral_critic.py) | Compare small RR redigital simulations with source-disjoint two-sample critics. |
| [evaluate_rr_simulator_critic.py](../../../experiments/origin_detection/platform_statistics/rr/evaluate_simulator_critic.py) | Measure whether simple paired features distinguish real RR processing from simulation. |
| [extract_rr_test.py](../../../experiments/data_preparation/rr/prepare_test.py) | Safely extract the verified RRDataset test archive and index every file. |
| [freeze_rr_simulation_split.py](../../../experiments/data_preparation/rr/prepare_registry_simulation_split.py) | Record RR sources already used for simulation development and reserve the rest. |
| [interpret_rr_jpeg_profiles.py](../../../experiments/origin_detection/platform_statistics/rr/audit_jpeg_profiles.py) | Match published RR JPEG quantization profiles to Pillow encodings. |
| [materialize_rr_transfer_simulation.py](../../../experiments/origin_detection/platform_statistics/rr/prepare_transfer_simulation.py) | Write development-set transfer simulations for fixed-detector score tests. |
| [metadata_shortcut_diagnostic.py](../../../experiments/origin_detection/encoding_shortcuts/rr/evaluate_metadata.py) | Measure how much ReWIND labels leak through format and image dimensions. |
| [rr_format_shortcut.py](../../../experiments/origin_detection/encoding_shortcuts/rr/run_format_shortcut.py) | Quantify the RRDataset train/val label shortcut from decoded file format. |
| [rr_oracle_codec_control.py](../../../experiments/origin_detection/encoding_shortcuts/rr/run_oracle_codec_control.py) | Test how much of RR processing one resize/JPEG pass can explain. |
| [rr_redigital_420_color_pilot.py](../../../experiments/origin_detection/redigital_statistics/rr/run_color_pilot.py) | Test a class-blind photometric simulator on RR's observable 4:2:0 stratum. |
| [rr_redigital_420_spectral_residual_pilot.py](../../../experiments/origin_detection/redigital_statistics/rr/run_spectral_pilot.py) | Pilot a class-balanced phase-randomized residual for RR 4:2:0 images. |
| [rr_simulator_v0.py](../../../experiments/origin_detection/platform_statistics/rr/run_fidelity.py) | Fit a class-blind resize/JPEG simulator and audit its held-out RR fidelity. |
| [rr_test_format_control.py](../../../experiments/origin_detection/encoding_shortcuts/rr/evaluate_format_control.py) | Apply the train/val format-only PNG→AI, JPEG→real rule to RR test. |
| [verify_rr_simulation_materialization.py](../../../experiments/origin_detection/platform_statistics/rr/audit_simulation_materialization.py) | Verify rendered JPEGs reproduce the simulator's recorded paired features. |
