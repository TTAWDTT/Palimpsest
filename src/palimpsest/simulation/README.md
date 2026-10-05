# 按处理事件组织的 simulation

[源码导览](../../README.md) · [架构](../../../docs/architecture.md) · [实验目录](../../../experiments/README.md)

simulation 为来源判别提供可组合的处理工具。同一事件允许不同显式参数与随机种子；不要求输出逐像素复现数据集中某一次处理。本次只是目录重构与共用代码提取，没有新增物理模型、重新训练或重新生成研究结果。

## 目录与状态

```text
simulation/
├── digital/                   数字裁切、缩放与 PNG/JPEG 重存
│   └── publication.py
├── screen_capture/            拍屏：显示→光学→相机采样→ISP
│   ├── capture.py             拍屏渲染入口
│   ├── pipeline.py            来源图→显示栅格→拍屏→数字发布
│   ├── _render/               发光、光学、空间/时间采样内部实现
│   ├── calibration/           采集图案、几何测量与参数转换
│   ├── reference.py           小场景数值积分参照
│   ├── render_cli.py
│   └── pipeline_cli.py
├── print_scan/                打印后扫描
│   ├── monochrome.py          单色打印与扫描原型
│   └── color.py               CMYK 纸面与彩色扫描原型
├── print_capture/             拍打印件
│   ├── photo_paper.py         连续色调相纸原型
│   └── camera.py              反射纸面→相机
├── screenshot/                截屏事件，待开发
├── projection_capture/        拍投影事件，独立前向实现待开发
└── shared/                    事件共用的机制与基础函数
    ├── image.py               RGB 范围、sRGB 转换、重采样与物理尺度模糊
    ├── sensor.py              Bayer、去马赛克与亮度锐化
    ├── optical_psf.py         瞳孔与离焦 PSF
    └── units.py               物理单位换算
```

| 事件 | 可调用入口 | 当前边界 |
|---|---|---|
| 数字处理 | [apply_publication](digital/publication.py) | 显式参数执行裁切、缩放和编码；不是某个社交平台的完整规则 |
| 拍屏 | [render_screen_capture](screen_capture/capture.py)、[run_screen_pipeline](screen_capture/pipeline.py) | 已有受限前向原型；尚未完成充分的真实设备验证 |
| 打印扫描 | [simulate_print_scan](print_scan/monochrome.py)、[simulate_color_print_scan](print_scan/color.py) | 单色/彩色结构原型，包含未校准的纸面与扫描假设 |
| 拍打印件 | [simulate_photo_paper_surface](print_capture/photo_paper.py)、[photograph_print_surface](print_capture/camera.py) | 有效纸面与相机原型；纸面也可来自 print_scan/color 的公共表面生成函数 |
| 截屏 | screenshot | 只有预留包；不能把拍屏或 PNG 保存当作已实现的截屏事件 |
| 拍投影 | projection_capture | 只有预留包；已有相关实验不构成完整事件模拟器 |

## 组合与共用边界

事件模块直接调用 shared。shared 不导入事件模块，不选择数据来源，不读取 AI/自然摄影标签。数字发布是独立事件，拍屏、扫描和拍打印件的输出都可以再调用它；不在每个事件中重复实现 JPEG 编码。

例如拍屏组合为：

```text
来源图 → screen_capture.pipeline 显示放置
      → screen_capture.capture 拍屏原型
      → digital.publication 裁切/缩放/编码
```

直接引用实现，不保留旧平铺路径的兼容文件：

```python
from palimpsest.simulation.digital.publication import (
    PublicationParameters,
    apply_publication,
)
from palimpsest.simulation.screen_capture.pipeline import run_screen_pipeline
```

具体 RR 校准参数池、来源划分、抽样和评价留在 [platform_statistics/rr](../../../experiments/origin_detection/platform_statistics/rr/README.md)。它们是数据集实验协议，不是另一种物理事件，不放入 shared。

## 扩展与维护

- 新事件在对应事件包内实现；未实现的包不提供假的模拟结果。
- 多个事件真正共用的机制放 shared；事件参数与特定假设留在事件模块。
- 更新实验、测试和 CLI 的直接调用，避免新增旧路径转发或入口私有助手转发。
- 测试按事件组织在 tests/simulation；结构测试通过不等于真实设备验证完成。
- 代码移动会改变历史代码指纹。既有数据、分数、缓存与清单不重写；严格复算旧结果需使用报告记录的历史 Git 版本。
