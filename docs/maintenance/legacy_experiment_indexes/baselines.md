> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# 检测 baseline

[返回实验目录](../../../experiments/README.md)

B-Free、D3、Benford-RF 和评分汇总；官方模型使用独立环境。

所有 Python 命令从仓库根目录以 `python -m experiments.baselines.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [benford_rr.py](../../../experiments/origin_detection/baselines/run_benford_rr.py) | DCT first-digit statistics plus random forest on RRDataset. |
| [compare_rr_full_baselines.py](../../../experiments/origin_detection/baselines/evaluate_rr_full_baselines.py) | Audit matched RRDataset predictions and compare full baseline results. |
| [continue_bfree.ps1](../../../experiments/origin_detection/baselines/resume_rewind_bfree.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [continue_rr_bfree.ps1](../../../experiments/origin_detection/baselines/resume_rr_bfree.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [evaluate_bfree_by_dataset.py](../../../experiments/origin_detection/baselines/evaluate_bfree_by_dataset.py) | Summarize local and published B-Free decisions by ReWIND upstream dataset. |
| [evaluate_bfree_local.py](../../../experiments/origin_detection/baselines/evaluate_bfree_local.py) | Compare locally inferred B-Free logits with QuAD's published per-file scores. |
| [evaluate_matched_rr_pilot.py](../../../experiments/origin_detection/baselines/evaluate_matched_rr_pilot.py) | Compare four baselines on one prespecified 600-source RRDataset pilot. |
| [evaluate_rr_bfree.py](../../../experiments/origin_detection/baselines/evaluate_rr_bfree.py) | Evaluate B-Free on every verified RRDataset test condition and paired source. |
| [evaluate_rr_transfer_simulation_bfree.py](../../../experiments/origin_detection/baselines/evaluate_rr_transfer_simulation_bfree.py) | Compare simulated and released transfer effects on fixed B-Free scores. |
| [fourier_knn_rr.py](../../../experiments/origin_detection/baselines/run_fourier_knn_rr.py) | Adapt Dzanic et al.'s spectral features and 5-NN to RRDataset. |
| [run_bfree_baseline.py](../../../experiments/origin_detection/baselines/run_bfree.py) | Run the unmodified B-Free network with per-image end-to-end timings. |
| [run_d3_rr.py](../../../experiments/origin_detection/baselines/run_d3_rr.py) | Evaluate official D3 head and CLIP ViT-L/14 on RRDataset single images. |
| [score_published_logits.py](../../../experiments/origin_detection/baselines/evaluate_published_logits.py) | Recompute single-image baselines from QuAD's published detector logits. |
| [score_rewind_available_subset.py](../../../experiments/origin_detection/baselines/evaluate_rewind_available_subset.py) | Score the QuAD-published logits on files expected in its public ReWIND ZIP. |
| [summarize_bfree_latency.py](../../../experiments/origin_detection/baselines/evaluate_summary_bfree_latency.py) | Report ReWIND batch-one B-Free timings by patch-projection execution mode. |
| [validate_bfree_crop_first.py](../../../experiments/origin_detection/baselines/audit_bfree_crop_first.py) | Check patch-aligned crop-first B-Free logits against the original forward. |
