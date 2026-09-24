# 拍屏前向链新增可切换的 ISP 亮度锐化，Chimera 检验未支持单因解释

日期：2026-09-25。**状态：结构假设已实现；真实设备参数未标定。**在已有的显示发光→光学→CFA/RAW→简化去马赛克→sRGB 链之后，增加默认关闭的亮度边缘增强。这样能明确检查“后期 ISP 锐化”与原生拍屏高频的关系，但用 Chimera 真图做冻结分组检验时，单一高斯反遮罩锐化仍不足以解释两设备的宽频差和通道结构。

## 机制与实现位置

[Xu 等的相机 ISP 示意，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf)把去马赛克、降噪、颜色/色调处理、边缘增强及压缩放在 RAW 之后的链中；[RawVDemoiré 原论文](https://papers.nips.cc/paper_files/paper/2023/file/7f05193e5487287a890df7fbc3554427-Paper-Conference.pdf)也指出屏幕和 CFA 的混叠经过 ISP 后会进一步改变外观。因此“图像中的高频”不应统统塞回显示格子或传感器噪声。

`origin_simulation/screen_capture.py` 现在支持参数 `isp_luma_sharpen_amount`（默认 0）与 `isp_luma_sharpen_sigma_pixels`（默认 1）。模型先对模拟 RAW 做双线性去马赛克和 sRGB 转换，再计算编码 RGB 的亮度 `Y′=0.2126R′+0.7152G′+0.0722B′`，令 `Δ=a[Y′−Gaussianσ(Y′)]`，把同一个 `Δ` 加给三个 RGB 通道并裁剪到 `[0,1]`。参数 `a` 和 `σ` 是虚拟的 ISP 假设，不是 Chimera 的实测设备设置。输出增加 `srgb_before_sharpen` 中间量，CLI 的可选 NPZ 也保存它；RAW、辐照度和传感器噪声不会被这个末段操作回写。

新结构测试验证默认关闭时输出保持不变；打开时仅改变最终 RGB、不改变 RAW；非饱和像素的 R/G/B 增量相同，负强度被拒绝。拍屏及采集工具相关测试 **22 项通过**。这个测试只证明代码因果位置和数值性质，**不证明某台相机 ISP 实际采用此滤波器**。

## 对真实 Chimera 拍屏的独立检验

为了避免直接把检测器分数当作锐化参数，另外做了一个**发布图空间的负对照**：从 256×256 数字原图作此前固定的仿射，再放大到原生发布分辨率；在量化到 8 位前施加相同形式的亮度锐化（固定 σ=1 发布像素）。这是对未知显示/相机/发布链的粗代理，**不是运行上述物理渲染器，也不是相机 ISP 校准**。使用 240 个校准来源，只以 0.30–0.45 周期/发布像素的**亮度功率绝对 dB 误差**从强度 `{0, .25, .5, .75, 1, 1.5, 2, 3, 5}` 选择；120 个开发来源只用于固定参数验收。840 个 reserved 来源未读取。Chimera 数据审计见[实包报告](Chimera真实拍屏配对实包审计_2026-09-25.md)。

| 设备 | 校准最小误差强度 | 开发亮度功率平均绝对误差，强度 0→选定 | 开发 RGB MAE，0→选定 | 候选像素裁剪比例均值，0→选定 |
|---|---:|---:|---:|---:|
| MacBook→iPhone | 5，达到搜索边界 | 8.78→8.42 dB | 0.0833→0.0835 | 0.15%→0.65% |
| LG→Blackfly | 5，达到搜索边界 | 7.61→5.31 dB | 0.0647→0.0665 | 0.21%→0.89% |

Mac 的改善很小；Monitor 的亮度频带改善约 2.30 dB，但仍留 5.31 dB 误差，RGB 像素误差反而增加。所选强度在两设备都落在**探索上限**；继续把强度调得更大不能被视为物理识别，且会进一步带来过冲/裁剪。开发集 RGB 跨通道一致性绝对误差的平均值 Mac 从 0.441 到 0.429，Monitor 从 0.485 到 0.308，仍不为零；色差功率绝对误差 Mac 从 6.55 到 6.46 dB，Monitor 从 3.65 到 3.59 dB，变化很小。对照核与颜色/ISP 流程未知，数值不能外推为镜头或 ISP 参数。

**决定：**保留默认关闭的 ISP 锐化分支，供将来有 RAW 或受控拍屏图时做机制消融；不把 `a=5` 写进设备配置，也不据此生成最终训练集。下一版真正值得投入的过程验证，应在同源真实图上同时检查源亮度依赖、高频亮度/色差功率、RGB 互频谱、边缘过冲及固定检测器的同源分数转移。如果缺 RAW，发布图只能约束这些复合观测量，不能把 ISP 与显示、镜头及作者重采样独立标定。

复算：`python work/test_chimera_luma_sharpen_hypothesis.py` 得 `work/chimera_luma_sharpen_hypothesis.json`。通道和内容诊断见[前一报告](Chimera高频的内容混杂与RGB通道结构_2026-09-25.md)。代码验证：`python -m pytest tests/test_screen_capture.py tests/test_screen_capture_projective.py tests/test_capture_kit.py -q`。
