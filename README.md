# 传播后图像原始来源判别

判断一张经过平台传播或物理再数字化的图像，其**原始视觉内容**来自自然摄影还是 AI 生成。

**从 [研究导航](outputs/README.md) 开始。**那里只列当前结论、阅读顺序和每类报告的入口；早期试验已单独归档，避免与现阶段结论混在一起。

## 仓库地图

| 位置 | 内容 |
|---|---|
| [`outputs/`](outputs/README.md) | 可阅读的研究结论，按研究问题、数据与基线、过程模拟、历史探索分组。 |
| [`docs/`](docs/controlled_screen_capture_protocol.md) | 采集协议、[首轮操作单](docs/controlled_screen_capture_runbook.md)与记录规范。 |
| [`sources/`](sources/2026-09-24_process_physics_sources.md) | 原始论文和官方数据来源的证据索引。 |
| [`origin_simulation/`](origin_simulation/capture_kit.py) | 拍屏前向原型、受控采集校准包，以及[棋盘几何测量](origin_simulation/geometry_measure.py)。 |
| [`work/`](work/README.md) | 既有复算脚本及本机中间产物；从导航页进入，不建议把文件列表当阅读顺序。 |
| `E:\ai_image_origin_research\data\` | 大型原始档案、解包图像与冻结清单，不进入 Git。 |

目前已完成 RRDataset 三种 baseline 的全量评测。Simulation 已有第一个拍屏空间链物理结构原型，但尚未经过真实设备校准；早期统计模拟也不能当作已验证的物理过程。

本项目从 2026-09-24 才开始使用独立 Git。较早实验的形成时间由原始记录及文件指纹说明，首次提交不是事前预注册。既有 `work/` 脚本路径维持原样，以免破坏报告中的复算命令。
