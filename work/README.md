# 历史复算工作区

此目录保留已有脚本的原路径，因为报告里的复算命令和文件指纹引用这些路径。Git 只跟踪顶层 Python/PowerShell 脚本与环境锁文件；同目录的 CSV、JSON、日志、模型、虚拟环境及第三方仓库属于本机中间产物。阅读研究结论请从 [`outputs/README.md`](../outputs/README.md) 开始。

| 任务 | 主要入口 |
|---|---|
| RR 测试包准入与配对 | `audit_rr_test.py`、`audit_rr_trainval_archive.py`、`freeze_rr_simulation_split.py` |
| 三种全量 baseline | `run_bfree_baseline.py`、`run_d3_rr.py`、`benford_rr.py`、`compare_rr_full_baselines.py` |
| 真实扫描配对审计 | `audit_descan_pairs.py`、`audit_descan_near_duplicates.py` |
| 早期 simulation 原型 | `rr_simulator_v0.py`、`materialize_rr_transfer_simulation.py`、`evaluate_rr_simulator_critic.py` |
| 下载与断点恢复 | `download_verified.ps1`、`continue_rr_bfree.ps1`、`download_descan_calibration.py` |
| Dragotti 官方 ZIP 小样本审计 | `probe_dragotti_range.py`、`analyze_dragotti_probe.py`、`build_dragotti_audit_bundle.py`；机器可读结果见 `outputs/03_过程模拟/`。 |

这是历史代码导航，不代表所有脚本都是当前推荐方法。新物理过程模块在完成受控测量协议后另建稳定入口，避免继续把研究版脚本堆在此目录。
