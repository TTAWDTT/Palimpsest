# 开发与复算指南

[仓库首页](../README.md) · [研究结论](../outputs/README.md) · [实验脚本](../experiments/README.md)

## 代码阅读顺序

先读 [screen_pipeline.py](../origin_simulation/screen_pipeline.py)，了解完整的来源图→显示→相机→发布流程；再读 [screen_capture.py](../origin_simulation/screen_capture.py) 的 `render_screen_capture`，了解相机内部阶段。

| 阶段 | 文件 | 责任 |
|---|---|---|
| 输入与显示栅格 | [screen_pipeline.py](../origin_simulation/screen_pipeline.py) | 数字图缩放、放置、量化，串接拍摄与发布 |
| 参数与结果类型 | [_screen/parameters.py](../origin_simulation/_screen/parameters.py) | 参数单位、允许范围、互斥条件与输出数组 |
| 面板发光 | [_screen/emission.py](../origin_simulation/_screen/emission.py) | RGB 发光区域和共址机制负对照 |
| 光学 | [_screen/optics.py](../origin_simulation/_screen/optics.py)、[optical_psf.py](../origin_simulation/optical_psf.py) | 模糊、薄透镜、衍射和瞳孔积分 |
| 空间采样 | [_screen/spatial.py](../origin_simulation/_screen/spatial.py) | 细网格积分、解析积分及受限快速路径 |
| 曝光时间 | [_screen/temporal.py](../origin_simulation/_screen/temporal.py) | PWM 与逐行曝光积分 |
| 传感器与 ISP | [screen_capture.py](../origin_simulation/screen_capture.py)、[sensor.py](../origin_simulation/sensor.py) | 光子噪声、读出增益、Bayer、去马赛克与色调 |
| 发布处理 | [publication.py](../origin_simulation/publication.py) | 裁切、缩放、PNG/JPEG 编码 |

`screen_capture` 的公共入口和参数名保留；历史实验使用的私有函数也暂保留显式导出。新内部实现直接引用所属模块，避免依赖入口文件中的私有函数。

其他路径：`print_scan.py` 为单色打印扫描；`color_print_scan.py` 为 CMYK 代理；`photo_paper.py` 为连续色调代理；`print_camera.py` 把纸面接入相机。有效范围见[打印约定](print_scan_model_contract.md)。投影实验位于 `experiments/projector/`，见[投影约定](projector_camera_model_contract.md)。

## 运行和验证

```powershell
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
uv run --locked python -m origin_simulation.render_cli --help
uv run --locked python -m origin_simulation.pipeline_cli --help
python tools/check_layout.py
```

单帧显示输入配置示例：[virtual_screen_example.json](virtual_screen_example.json)。它是虚拟参数，不代表校准设备。CLI 要求新输出前缀，避免覆盖已有文件。

`check_layout.py` 静态检查仓库内导入与文档链接，不执行实验、不访问数据盘。

- 通用成像机制放 `origin_simulation/`；具体数据集路径、来源选择、拟合和结果统计放 `experiments/<主题>/`。
- 每个机制写明输入单位、假设和适用范围。模型不能访问真假标签来选取传播参数。
- 先验证相关行为；结构测试、数值参照和真实数据验证分别报告。
- 结果明确校准、开发和留出来源。研究暂停期间，不自动启动下载、全量推理或训练。

## 历史实验与结果完整性

所有实验在仓库根目录执行。Python 统一使用模块方式，例如：

```powershell
python -m experiments.baselines.evaluate_rr_bfree --help
python -m experiments.chimera.evaluate_chimera_virtual_screen_mac
```

第二条会读取既有结果，需要对应本机产物；不是无数据示例。部分旧实验没有 CLI，会直接读写固定路径，运行前查看脚本头部路径常量和对应报告。深度学习 baseline 使用各自官方依赖环境，最小模型环境不包含 torch、timm、sklearn 等全部研究依赖。

2026-09-26 整理时，212 个 Python/PowerShell 脚本从 `work/` 迁入十个实验主题；修改了导入、路径和排版。CSV/JSON 指标、冻结清单与模型权重未重新生成。历史指纹对应当时代码，**整理不构成旧实验重新运行或重新验收**。

整理前版本：`0f675d6d544de407dae5c162e26100245edf79ef`。恢复要求原代码指纹的断点时，应在该版本的独立 checkout 中恢复原环境。新布局下不手工覆盖旧指纹；RAW 模拟缓存现在对整个 `origin_simulation/` 包计算代码指纹，避免漏掉已拆分的模块。

## 数据采集与文献

- [受控拍屏采集协议](controlled_screen_capture_protocol.md)
- [首轮采集操作单](controlled_screen_capture_runbook.md)
- [原始文献与官方数据证据](../sources/2026-09-24_process_physics_sources.md)

用户暂时无法拍摄，采集保持待办状态。
