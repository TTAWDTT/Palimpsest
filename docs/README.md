# 开发与复算指南

[仓库首页](../README.md) · [研究结论](../reports/README.md) · [实验脚本](../experiments/README.md)

先读[代码结构](architecture.md)理解完整任务、当前能力和依赖方向。区域与来源推理见[接入指南](detection_pipeline.md)；模拟器先读[事件目录说明](../src/palimpsest/simulation/README.md)，下面给出拍屏原型的阅读顺序。

## 代码阅读顺序

先读 [screen_capture/pipeline.py](../src/palimpsest/simulation/screen_capture/pipeline.py)，了解完整的来源图→显示→相机→发布流程；再读 [screen_capture/capture.py](../src/palimpsest/simulation/screen_capture/capture.py) 的 `render_screen_capture`，了解相机内部阶段。

| 阶段 | 文件 | 责任 |
|---|---|---|
| 输入与显示栅格 | [screen_capture/pipeline.py](../src/palimpsest/simulation/screen_capture/pipeline.py) | 数字图缩放、放置、量化，串接拍摄与发布 |
| 参数与结果类型 | [screen_capture/_render/parameters.py](../src/palimpsest/simulation/screen_capture/_render/parameters.py) | 参数单位、允许范围、互斥条件与输出数组 |
| 面板发光 | [screen_capture/_render/emission.py](../src/palimpsest/simulation/screen_capture/_render/emission.py) | RGB 发光区域和共址机制负对照 |
| 光学 | [screen_capture/_render/optics.py](../src/palimpsest/simulation/screen_capture/_render/optics.py)、[shared/optical_psf.py](../src/palimpsest/simulation/shared/optical_psf.py) | 模糊、薄透镜、衍射和瞳孔积分 |
| 空间采样 | [screen_capture/_render/spatial.py](../src/palimpsest/simulation/screen_capture/_render/spatial.py) | 细网格积分、解析积分及受限快速路径 |
| 曝光时间 | [screen_capture/_render/temporal.py](../src/palimpsest/simulation/screen_capture/_render/temporal.py) | PWM 与逐行曝光积分 |
| 传感器与 ISP | [screen_capture/capture.py](../src/palimpsest/simulation/screen_capture/capture.py)、[shared/sensor.py](../src/palimpsest/simulation/shared/sensor.py) | 光子噪声、读出增益、Bayer、去马赛克与色调 |
| 发布处理 | [digital/publication.py](../src/palimpsest/simulation/digital/publication.py) | 裁切、缩放、PNG/JPEG 编码 |

`screen_capture/capture.py` 提供拍屏入口。空间/时间实现位于事件内的 `_render/`，跨事件共用机制在 `shared/`；所有调用直接引用实现，旧平铺路径不保留转发。

其他事件：`digital/publication.py` 为数字裁切/缩放/编码；`print_scan/` 提供单色与 CMYK 打印扫描原型；`print_capture/` 提供连续色调相纸与拍纸面原型。有效范围见[打印约定](print_scan_model_contract.md)。截屏和独立拍投影前向模块待开发；已有投影实验位于 `experiments/projection_capture/`，见[投影约定](projector_camera_model_contract.md)。

## 运行和验证

```powershell
uv sync --locked --extra dev --extra classical
uv run --locked --extra dev --extra classical python -m pytest -q
uv run --locked python -m palimpsest.simulation.screen_capture.render_cli --help
uv run --locked python -m palimpsest.simulation.screen_capture.pipeline_cli --help
uv run --locked python tools/check_layout.py
```

单帧显示输入配置示例：[virtual_screen_example.json](../configs/screen/virtual_screen.example.json)。它是虚拟参数，不代表校准设备。CLI 要求新输出前缀，避免覆盖已有文件。

`check_layout.py` 静态检查仓库内导入与文档链接，不执行实验、不访问数据盘。

- 通用成像机制放 `src/palimpsest/simulation/`；具体数据集路径、来源选择、拟合和结果统计放 `experiments/<主题>/`。
- 每个机制写明输入单位、假设和适用范围。模型不能访问真假标签来选取传播参数。
- 先验证相关行为；结构测试、数值参照和真实数据验证分别报告。
- 结果明确校准、开发和留出来源。研究暂停期间，不自动启动下载、全量推理或训练。

## 历史实验与结果完整性

所有实验在仓库根目录执行。Python 统一使用模块方式，例如：

```powershell
python -m experiments.origin_detection.baselines.evaluate_rr_bfree --help
python -m experiments.screen_capture.source_to_rgb.chimera.evaluate_virtual_mac
```

第二条会读取既有结果，需要对应本机产物；不是无数据示例。部分旧实验没有 CLI，显式执行入口时读取结果；运行前查看对应报告。路径由 `palimpsest.paths` 统一配置，历史冻结清单内部的绝对路径仍按原记录保留。深度学习 baseline 使用各自官方依赖环境，最小模型环境不包含 torch、timm、sklearn 等全部研究依赖。

使用独立 baseline Python 时，将仓库和 `src` 加入模块路径：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
```

显式执行的 PowerShell 续跑脚本会加载同一配置。它们仍要求历史指纹一致；整理后不能用它们覆盖旧结果。

2026-09-26 整理时，212 个 Python/PowerShell 脚本从 `work/` 迁入十个实验主题；修改了导入、路径和排版。CSV/JSON 指标、冻结清单与模型权重未重新生成。历史指纹对应当时代码，**整理不构成旧实验重新运行或重新验收**。

整理前版本：`0f675d6d544de407dae5c162e26100245edf79ef`。恢复要求原代码指纹的断点时，应在该版本的独立 checkout 中恢复原环境。新布局下不手工覆盖旧指纹；RAW 模拟缓存现在对整个 `src/palimpsest/simulation/` 包计算代码指纹，避免漏掉已拆分的模块。

2026-10-04 重构前版本：`649c601b6bc15a1e132cbb3c25f63ed4fb342c40`。本轮采用 `src/palimpsest`，提取共享评测、数据和文件校验；报告从 `outputs/` 移入 `reports/`。新代码指纹改变，旧数据、结果和指标保留原值。

2026-10-05 事件目录重构前版本：`9c5ffcb`。simulation 按事件归档并提取共用函数，直接更新实验、测试和 CLI 调用；未重跑旧研究结果或改写已冻结的工件指纹。

## 数据采集与文献

- [受控拍屏采集协议](controlled_screen_capture_protocol.md)
- [首轮采集操作单](controlled_screen_capture_runbook.md)
- [原始文献与官方数据证据](../sources/2026-09-24_process_physics_sources.md)

用户暂时无法拍摄，采集保持待办状态。
