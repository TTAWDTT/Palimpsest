# 检测器分数迁移 / chimera

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

历史固定检测器评测；官方模型环境、权重及旧断点指纹均有独立约束。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先核对数据准入/清单和权重，再显式推理，最后校验结果覆盖、固定阈值指标、配对变化及含解码延迟。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [evaluate_bfree.py](evaluate_bfree.py) | 评测、比较或汇总 | Evaluate source-paired B-Free performance on verified Chimera recaptures. |
| [evaluate_score_contraction.py](evaluate_score_contraction.py) | 评测、比较或汇总 | Separate score compression from common offset on frozen Chimera development. |
| [evaluate_simulation_score_transfer.py](evaluate_simulation_score_transfer.py) | 评测、比较或汇总 | Compare fixed B-Free source-score changes on real vs composite proxy recaptures. |
| [evaluate_size_control.py](evaluate_size_control.py) | 评测、比较或汇总 | Compare native and 256x256 Chimera recaptures under fixed B-Free weights. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = WORK_DIR / 'chimera_composite_sim_dev256_audit.json'`
- `SPLIT = DATA_ROOT / 'manifests/chimera_simulation_source_split.csv'`
