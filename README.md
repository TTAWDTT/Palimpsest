# 传播后图像原始来源判别

研究经过缩放、压缩、平台传播和物理再数字化后，如何判断图像的原始内容来自自然摄影还是 AI 生成。

**当前研究暂停。** 已完成数据审计和三种 baseline；过程模拟器仍是未完成真实设备验证的研究原型。尚未形成自研的高速稳健检测算法或模型。

本机代码位置：`E:\ai_image_origin_research\repo`。GitHub 私有仓库：[TTAWDTT/ai-image-origin-research](https://github.com/TTAWDTT/ai-image-origin-research)。

## 从这里开始

| 需要了解什么 | 入口 |
|---|---|
| 已完成什么、哪些结论成立 | [研究结论](outputs/README.md) |
| simulation 的原理、真实验证和失败证据 | [过程模拟](outputs/03_过程模拟/README.md) |
| 代码如何组织、怎样复算 | [开发与复算指南](docs/README.md) |
| 某个数据集的实验脚本 | [实验目录](experiments/README.md) |
| 一起整理仓库、记录待讨论事项 | [协作说明](docs/collaboration.md) |

## 仓库结构

```text
origin_simulation/     可复用的前向模型、处理流水线和 CLI
experiments/           按数据集/用途分类的历史实验脚本
tests/                 模型行为和数值回归测试
docs/                  开发指南、模型约定和受控采集协议
outputs/               研究报告、关键指标和配图
sources/               文献与官方数据来源索引
work/                  本机结果、日志、缓存和第三方代码（不提交）
```

大型数据、模型权重和冻结清单位于 `E:\ai_image_origin_research\`。阅读文档和运行模型单元测试不需要连接 E 盘。

## 安装与检查

在仓库根目录运行，要求 Python 3.11 及以上：

```powershell
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
uv run --locked python -m origin_simulation.render_cli --help
uv run --locked python -m origin_simulation.pipeline_cli --help
```

两个 CLI 分别接受“已知数字显示帧”和“来源图→显示→相机→发布”的配置；参数未经真机标定时，输出只能用于检验模型假设。历史实验有额外数据及环境要求，见[复算指南](docs/README.md#历史实验与结果完整性)。
