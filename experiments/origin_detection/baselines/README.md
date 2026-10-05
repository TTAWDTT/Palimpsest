# 固定检测器 baseline / baselines

[实验首页](../../README.md) · [命名与执行约定](../../../docs/experiment_conventions.md) · [研究报告](../../../reports/README.md)

## 范围与状态

新统一缓存汇总见 [evaluate_rr_suite.py](evaluate_rr_suite.py)：`--scope baselines` 复用三个全量 baseline，`--scope all` 加入[数字传播对照](../propagation_controls/rr/README.md)。该入口不启动推理；本轮只显式运行新对照图片的 B-Free。

历史固定检测器评测；官方环境、权重和断点均有独立约束。推理/特征实现已提取到 `palimpsest.detection.baselines`，本目录保留选择数据、调度、训练旧 RF 和汇总协议。研究训练和全量复算暂停；少量接入核对见[接口验证](../integration/rr/README.md)。

## 前置条件与执行顺序

先核对数据准入/清单和权重，再显式推理，最后校验结果覆盖、固定阈值指标、配对变化及含解码延迟。

数据、权重、结果根目录由 [路径配置](../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_bfree_crop_first.py](audit_bfree_crop_first.py) | 核查结构、配对或假设 | Check patch-aligned crop-first B-Free logits against the original forward. |
| [run_benford_rr.py](run_benford_rr.py) | 执行明确的实验或对照 | DCT first-digit statistics plus random forest on RRDataset. |
| [run_bfree.py](run_bfree.py) | 执行明确的实验或对照 | Run the unmodified B-Free network with per-image end-to-end timings. |
| [run_d3_rr.py](run_d3_rr.py) | 执行明确的实验或对照 | Evaluate official D3 head and CLIP ViT-L/14 on RRDataset single images. |
| [run_fourier_knn_rr.py](run_fourier_knn_rr.py) | 执行明确的实验或对照 | Adapt Dzanic et al.'s spectral features and 5-NN to RRDataset. |
| [evaluate_bfree_by_dataset.py](evaluate_bfree_by_dataset.py) | 评测、比较或汇总 | Summarize local and published B-Free decisions by ReWIND upstream dataset. |
| [evaluate_bfree_local.py](evaluate_bfree_local.py) | 评测、比较或汇总 | Compare locally inferred B-Free logits with QuAD's published per-file scores. |
| [evaluate_matched_rr_pilot.py](evaluate_matched_rr_pilot.py) | 评测、比较或汇总 | Compare four baselines on one prespecified 600-source RRDataset pilot. |
| [evaluate_published_logits.py](evaluate_published_logits.py) | 评测、比较或汇总 | Recompute single-image baselines from QuAD's published detector logits. |
| [evaluate_rewind_available_subset.py](evaluate_rewind_available_subset.py) | 评测、比较或汇总 | Score the QuAD-published logits on files expected in its public ReWIND ZIP. |
| [evaluate_rr_bfree.py](evaluate_rr_bfree.py) | 评测、比较或汇总 | Evaluate B-Free on every verified RRDataset test condition and paired source. |
| [evaluate_rr_full_baselines.py](evaluate_rr_full_baselines.py) | 评测、比较或汇总 | Audit matched RRDataset predictions and compare full baseline results. |
| [evaluate_rr_transfer_simulation_bfree.py](evaluate_rr_transfer_simulation_bfree.py) | 评测、比较或汇总 | Compare simulated and released transfer effects on fixed B-Free scores. |
| [evaluate_summary_bfree_latency.py](evaluate_summary_bfree_latency.py) | 评测、比较或汇总 | Report ReWIND batch-one B-Free timings by patch-projection execution mode. |
| [resume_rewind_bfree.ps1](resume_rewind_bfree.ps1) | 显式恢复历史推理 | 历史 PowerShell 入口；执行前核对长度/MD5/断点要求。 |
| [resume_rr_bfree.ps1](resume_rr_bfree.ps1) | 显式恢复历史推理 | 历史 PowerShell 入口；执行前核对长度/MD5/断点要求。 |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `MANIFEST = DATA_ROOT / 'manifests/rr_test_files.csv'`
- `ROOT = DATA_ROOT / 'derived/rr_test'`
- `ROOT = REPO_ROOT`
