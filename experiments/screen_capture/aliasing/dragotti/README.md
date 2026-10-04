# 子像素混叠与衍射 / dragotti

[实验首页](../../../README.md) · [命名与执行约定](../../../../docs/experiment_conventions.md) · [研究报告](../../../../reports/README.md)

## 范围与状态

公开配对数据上的历史开发/诊断。有效参数与观察性关联不能自动解释为已识别的真实物理设备参数。 研究暂停，本轮仅整理代码。

## 前置条件与执行顺序

先完成该数据集准备和内容/几何审计，核对冻结划分；在校准部分拟合，再对既定开发/留出部分评价。不同文件可能代表替代假设，不构成一条必须全部运行的流水线。

数据、权重、结果根目录由 [路径配置](../../../../configs/README.md) 指定。先阅读脚本中的冻结指纹、来源清单和参数，不批量执行本目录。`protocol.py` 供入口复用，不启动实验。已有结果名称保持原值；新代码不能绕过旧断点校验。

## 入口清单

| 文件 | 角色 | 目的 |
|---|---|---|
| [audit_diffraction_alias_budget.py](audit_diffraction_alias_budget.py) | 核查结构、配对或假设 | Conditional diffraction/alias budget for Dragotti EOS600D recapture. |
| [run_alias.py](run_alias.py) | 执行明确的实验或对照 | Conservative screen-lattice alias probe on verified Dragotti color-chart crops. |
| [run_diffraction_otf.py](run_diffraction_otf.py) | 执行明确的实验或对照 | Conditional ideal diffraction cutoff for a published recapture setting. |
| [run_predicted_alias_spectrum.py](run_predicted_alias_spectrum.py) | 执行明确的实验或对照 | Exploratory screen-grid alias-band probe on verified real recaptures. |

## 声明的路径与前置记录

以下为源代码常量的静态摘录，完整约束仍以入口及共享协议为准。

- `AUDIT = Path('outputs/03_过程模拟/02_拍屏实测/Dragotti跨场景几何复核_2026-09-24.json')`
- `AUDIT = Path('outputs/03_过程模拟/02_拍屏实测/Dragotti跨设备配对审计_2026-09-24.json')`

## 跨问题协议依赖

- [screen_capture/geometry/dragotti/audit_pairs.py](../../geometry/dragotti/audit_pairs.py)
