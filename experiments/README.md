# 实验工作流

[仓库首页](../README.md) · [命名与执行约定](../docs/experiment_conventions.md) · [迁移清单](../docs/maintenance/experiment_migration.csv)

目录按“研究什么”划分，数据集是问题内的协议范围，文件名说明执行角色。`src/palimpsest/` 保存可复用实现；问题内 `protocol.py` 保存冻结设置与专用共享步骤。当前推进[传播后 AI／非 AI 算法迭代](../docs/robust_ai_detection_plan.md)，已实现[统计候选筛选](origin_detection/robust_statistics/rr/README.md)与[固定规则拍屏诊断](origin_detection/robust_statistics/chimera/README.md)。已有[统一缓存评测](../docs/propagation_evaluation.md)可复用；历史探索不自动运行。

| 主题 | 内容 |
|---|---|
| [来源判别](origin_detection/README.md) | 固定 baseline、检测器分数迁移、RR 统计模拟与编码捷径 |
| [拍屏过程](screen_capture/README.md) | 来源对应、RAW 几何、CFA/ISP、光度、采样、混叠、对焦与波光学 |
| [打印与拍纸](print_capture/README.md) | 二值通道、网点、纸面响应及积分诊断 |
| [投影相机](projection_capture/README.md) | 结构光几何、色卡光度与原始响应 |
| [数据准备](data_preparation/README.md) | 获取、完整性、内容配对与冻结清单 |

## 找一个实验

1. 选择研究问题，阅读其 README 的状态、前置条件和入口表。
2. 核对源码里的指纹、窗口、插值、划分和输出位置。
3. 在仓库根目录显式运行相应 Python 模块；例如 `python -m experiments.screen_capture.source_to_raw.raw2event.run_two_sources`。该命令会运行研究实验，不是目录检查。

`prepare`、`audit`、`fit`、`run`、`evaluate`、`plot`、`benchmark` 表达执行角色。多个变体用具体限定词区分，不继续新增泛称 `probe` 的文件。历史文件仍有冻结常量，尚未提供统一调度或可任意切换协议的配置接口。

完整入口列表见 [目录表](../docs/maintenance/experiment_catalog.csv)，旧路径对应见 [迁移表](../docs/maintenance/experiment_migration.csv)。精确复算旧结果应恢复其记录的 Git 版本与环境；本轮改名与模块提取不重签旧断点。
