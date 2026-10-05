# Palimpsest

研究图像经过数字传播与物理再拍后，如何判断其原始视觉内容来自自然摄影还是 AI 生成。

**研究目前暂停。** 已有数据审计、三种 baseline 和前向模型原型；尚未完成 simulation 的充分真实验证，也尚未形成自研的高速稳健检测算法或模型。

本机代码：`E:\ai_image_origin_research\repo`。GitHub：[TTAWDTT/Palimpsest](https://github.com/TTAWDTT/Palimpsest)（公开）。

## 阅读入口

| 要了解什么 | 入口 |
|---|---|
| 当前证据、失败结果与待验证问题 | [研究结论](reports/README.md) |
| 公共模块如何划分、实验依赖什么 | [代码结构](docs/architecture.md) |
| 模型约定、安装和历史复算 | [开发指南](docs/README.md) |
| 某个研究问题的实验入口 | [实验工作流](experiments/README.md) |
| 配置数据、权重和结果路径 | [配置说明](configs/README.md) |
| 一起审阅、整理和商定下一步 | [协作说明](docs/collaboration.md) |

## 项目结构

```text
src/palimpsest/
  simulation/       前向模型与渲染入口
  data/             清单校验与明确约定的图像变换
  evaluation/       分类、同源配对、延迟与描述性统计
  io/               文件校验、CSV 与代码指纹
  paths.py          本机路径配置
src/origin_simulation/  历史模块命令的兼容入口
configs/            路径模板与虚拟模型配置
experiments/        按问题组织的历史工作流；角色入口与 protocol.py
reports/            研究结论、指标与图表
sources/            文献和官方来源记录
work/               本机结果、日志与第三方代码（多数不提交）
tests/              simulation、data、evaluation、infrastructure
tools/             维护工具
```

公共库不依赖实验脚本。具体数据集选择与拟合留在 `experiments/`；同一算法的通用实现从公共模块复用。原始数据、模型权重和独立 baseline 环境默认位于仓库同级目录，见[配置说明](configs/README.md)。

## 安装与检查

要求 Python 3.11 及以上；在仓库根目录运行：

```powershell
uv sync --locked --extra dev
uv run --locked ruff check src experiments tests tools
uv run --locked python tools/check_layout.py
uv run --locked pytest -q
uv run --locked palimpsest-render --help
uv run --locked palimpsest-pipeline --help
```

两种渲染入口分别接受数字显示帧和完整的来源图→显示→相机→发布配置。虚拟参数未经真机标定时，输出只能用于检验模型假设。上述检查不执行数据下载、全量推理或训练。

重构没有重算历史实验。要恢复已有断点，遵循[完整性说明](docs/README.md#历史实验与结果完整性)，不要修改指纹来绕过校验。
