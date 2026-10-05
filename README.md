# Palimpsest

研究图像经过数字传播与物理再拍后，如何判断其原始视觉内容来自自然摄影还是 AI 生成。

**当前推进传播后 AI／非 AI 非深度学习算法的迭代**，见[开发流程](docs/robust_ai_detection_plan.md)与[首轮结果](reports/04_算法研发/首轮局部统计算法_开发筛选与真实拍屏诊断_2026-10-05.md)。已完成局部统计原型、数据用途审计和真实处理筛选；固定规则在Chimera拍屏诊断失败，尚无经过验证的自研稳健方法。Simulation 研发与神经模型训练暂缓；已有全量 baseline 不重复推理。

[第二轮配对方向投影](reports/04_算法研发/第二轮配对方向投影_RR改善与真实拍屏失败_2026-10-05.md)已完成：RR开发改善没有迁移到真实拍屏，当前构造与速度预算均未通过。代码、参数和反例可复算。

本机代码：`E:\ai_image_origin_research\repo`。GitHub：[TTAWDTT/Palimpsest](https://github.com/TTAWDTT/Palimpsest)（公开）。

## 阅读入口

| 要了解什么 | 入口 |
|---|---|
| 当前证据、失败结果与待验证问题 | [研究结论](reports/README.md) |
| 当前算法开发步骤、数据划分与验收条件 | [算法研发流程](docs/robust_ai_detection_plan.md) |
| 首轮算法结果、失败与下一轮约束 | [算法研发记录](reports/04_算法研发/README.md) |
| 已有缓存汇总、数字传播对照协议 | [统一评测与传播对照](docs/propagation_evaluation.md) |
| 公共模块如何划分、实验依赖什么 | [代码结构](docs/architecture.md) |
| 从源码目录理解业务链路与扩展位置 | [源码导览](src/README.md) |
| 区域定位、来源判别、可调用链与能力缺口 | [推理接入指南](docs/detection_pipeline.md) |
| 整理清单、接入验证与研究待办 | [本轮处理记录](docs/maintenance/detection_foundation_checklist.md) |
| 模型约定、安装和历史复算 | [开发指南](docs/README.md) |
| 某个研究问题的实验入口 | [实验工作流](experiments/README.md) |
| 配置数据、权重和结果路径 | [配置说明](configs/README.md) |
| 一起审阅、整理和商定下一步 | [协作说明](docs/collaboration.md) |

## 项目结构

```text
src/palimpsest/
  contracts.py      RGB、坐标、掩膜、原始来源预测协议
  localization/     区域定位/分割，baselines 为已有候选方法
  detection/        原始自然摄影/AI 内容判别
    algorithms/     自研非深度学习原型（稳健性未通过）
    models/         自研神经模型（待开发）
    baselines/      B-Free、D3、Benford 适配
  pipelines/        解码→定位→裁切→来源推理与计时
  training/         显式训练与标定接口（实现待开发）
  simulation/       前向模型与渲染入口
  data/             清单校验与明确约定的图像变换
  evaluation/       分类、同源配对、延迟与描述性统计
  io/               文件校验、CSV 与代码指纹
  paths.py          本机路径配置
configs/            路径模板与虚拟模型配置
experiments/        按问题组织的历史工作流；角色入口与 protocol.py
reports/            研究结论、指标与图表
sources/            文献和官方来源记录
work/               本机结果、日志与第三方代码（多数不提交）
tests/              各公共模块的针对性验证
tools/             维护工具
```

公共库不依赖实验脚本。具体数据集选择与拟合留在 `experiments/`；同一算法的通用实现从公共模块复用。原始数据、模型权重和独立 baseline 环境默认位于仓库同级目录，见[配置说明](configs/README.md)。

## 安装与检查

要求 Python 3.11 及以上；在仓库根目录运行：

```powershell
uv sync --locked --extra dev --extra classical
uv run --locked ruff check src experiments tests tools
uv run --locked python tools/check_layout.py
uv run --locked --extra dev --extra classical pytest -q
uv run --locked palimpsest-render --help
uv run --locked palimpsest-simulate --help
uv run --locked palimpsest-detect --help
```

两种渲染入口接受数字显示帧和来源图→显示→相机→发布配置，虚拟参数用于检验假设。`palimpsest-detect` 提供图片来源链，执行时需相应外部代码/权重和推理环境；定位候选不保证是承载图像的物件。核心检查不下载权重、不执行全量推理或研究训练。

重构没有重算历史实验。要恢复已有断点，遵循[完整性说明](docs/README.md#历史实验与结果完整性)，不要修改指纹来绕过校验。
