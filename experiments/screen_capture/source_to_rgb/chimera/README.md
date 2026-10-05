# 数字源到发布 RGB / chimera

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [prepare_virtual_mac.py](prepare_virtual_mac.py) | 准备输入或物化对照 | Render a frozen, unfitted screen-camera hypothesis on Chimera development. |
| [fit_virtual_effective_mac.py](fit_virtual_effective_mac.py) | 仅在既定校准部分拟合 | Fit only effective exposure/blur to real RGB observables on calibration IDs. |
| [evaluate_virtual_mac.py](evaluate_virtual_mac.py) | 评测、比较或汇总 | Evaluate one frozen virtual forward chain against paired real Chimera Mac recaptures. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = WORK_DIR / 'chimera_virtual_screen_mac_dev256_audit.json'`
- `MANIFEST = DATA_ROOT / 'manifests/chimera_virtual_screen_mac_dev256_manifest.csv'`
- `SPLIT = DATA_ROOT / 'manifests/chimera_simulation_source_split.csv'`

## 跨问题协议依赖

- [origin_detection/score_transfer/chimera/evaluate_bfree.py](../../../origin_detection/score_transfer/chimera/evaluate_bfree.py)
- [origin_detection/score_transfer/chimera/evaluate_simulation_score_transfer.py](../../../origin_detection/score_transfer/chimera/evaluate_simulation_score_transfer.py)
- [screen_capture/geometry/chimera/run_observables.py](../../geometry/chimera/run_observables.py)
