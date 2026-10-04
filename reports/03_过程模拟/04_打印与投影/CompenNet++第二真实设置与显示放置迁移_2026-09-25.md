# CompenNet++ 第二真实设置：显示放置可沿用，几何与颜色须重测

2026-09-25。接续[首组原生画幅前向实验](CompenNet++原生画幅几何光度联合前向实验_2026-09-25.md)。**结论：**对官方第二组 `light3/pos1/cloud_np`，冻结首组由训练图估计的 256 方图显示放置方式，另用该组结构光和 13 张纯色参考标定，在同样编号的 20 张真实相机测试图上得到有效内容区 RGB MAE **0.04351**。同组四色端点方案为 **0.14278**，同组 13 色但只用单应几何为 **0.08779**。这验证了**模型形式**在另一组真实条件下仍有用；它没有证明首组的几何、光度参数可以原封不动转移，也不是屏摄结果。

## 对应关系与审计

两组均来自 [CompenNet++ 作者公开档案与采集说明](https://github.com/BingyaoHuang/CompenNet-plusplus)。本组另外取得 **72 个源图/相机 RGB PNG 成员**：14 个参考色输入及其采集图、训练图 1–2 及其采集图、测试图 20 张及其采集图。72/72 成员逐一核对官方 ZIP 中央目录 CRC、解压长度及 E 盘本地 SHA-256；其中 **36 个投影输入文件**也与首组同名源图逐字节 SHA 一致。本组 **42 对结构光源图/相机图**此前已独立完成 **84/84** 成员 CRC 核验。官方 **11.48 GB** ZIP 未全包验收，不能把逐成员审计写成整包验证。

这两组同名 `pos1/cloud_np` 不是单独改变照明的控制实验。[跨设置结构光比较](CompenNet++跨设置几何与光照不可混同_2026-09-25.md)显示相同投影码的相机位置中位偏移约 **112 像素**；此处因此为第二组**重新解码结构光几何**。两组相机黑图的全画幅 RGB 平均值也明显不同，但光照、姿态、曝光、表面区域共同变化，不作单因子归因。官方 `cam/raw` 指未经几何变换的相机 **RGB PNG**，不是传感器 RAW。

## 冻结与新标定

- **沿用首组：**显示放置变换 `u=0.427619x−43.109`、`v=0.427374y−0.346`，以及“逐点结构光几何 + 13 张纯色档位构成每通道有效响应”的模型形式。两组共享相同 256×256 投影输入。
- **第二组重新取得：**其自身 42 对结构光及黑、R/G/B 各四档的 13 张参考采集图。颜色参考为第二组特有，因此下面不是零标定跨设备预测。
- **仅作诊断：**用第二组训练图 1–2 和四色端点响应再拟合显示放置，所得 `u=0.433196x−44.522`、`v=0.427835y−0.188`。四色模型的训练 MAE 从沿用首组放置时 **0.11134** 略降至 **0.11113**。这不使用测试图，但四色响应在暗档明显失真，几何拟合也可能吸收光度残差。
- **测试完全固定：**使用与首组相同的 20 个纹理编号和已核过 SHA 的源图；各方式在第二组共同的 **24,255 个**高置信相机内容像素上评测，占 640×480 画幅 **7.90%**。该区域小于首组，两个设置的绝对 MAE **不可直接按同一采样分布比较**。

| 第二组真实相机测试，20 张逐图等权平均 MAE | 沿用首组显示放置 | 第二组两张训练图重拟合放置 |
|---|---:|---:|
| 黑 + 纯 R/G/B 四色线性叠加，第二组逐点结构光 | 0.14278 | 0.14421 |
| **13 色通道阶梯，第二组逐点结构光** | **0.04351** | 0.04942 |
| 13 色通道阶梯，第二组结构光拟合**单应几何** | 0.08779 | — |

13 色阶梯比四色端点在 **20/20** 张测试图更准；逐点几何比单应映射也在 **20/20** 张更准。冻结首组放置参数比第二组用四色端点重拟合的放置参数在 **20/20** 张更准。一个谨慎的解释是：实际显示缩放/居中较稳定，而有缺陷的光度模型会把颜色误差错归给几何，导致训练误差略降、留出误差上升。此处不能据此唯一证明具体投影软件如何缩放；只说明**色彩与几何联拟合需要独立校准或反证**。

第二组结构光高置信掩膜有 **37,964** 相机像素，但最终两种显示放置的共同内容区仅 **24,255**。该组单应拟合 3 像素阈值的随机一致性内点率 **37.82%**，也提示同名布幕上的局部几何不能被单一平面模型充分表达。结构光低对比、遮挡和未覆盖区域被排除，结果不能推广到整个相机画幅。

## 研究边界与下一步

第一、第二组都支持“局部几何 + 输入通道非线性色阶”这个**条件前向结构**，但每组仍需自己的结构光/色块采集。测得的是投影、表面与 RGB 相机发布链合成后的有效响应；没有分离 DLP 时序、光谱、BRDF、PSF、传感器 CFA 或 ISP。它也不能代替手机拍屏、打印扫描或 AI/自然来源标签。下一步应在**不借测试纹理调参**的前提下，比较少量混色参考能否稳健解释留出混色和高频纹理，以及颜色标定减少后跨设置性能如何变化。

复算程序为 `experiments/projector/evaluate_compennetpp_cross_setup_display.py`，机器可读逐图结果为 `work/compennetpp_cross_setup_display_probe.json`。第二组新图在 `E:\ai_image_origin_research\data\derived\compennetpp_raw_ref_probe_light3_pos1_cloud_np`，此前结构光图在 `E:\ai_image_origin_research\data\derived\compennetpp_sl_light3_pos1_cloud`。在前一实验和第二组结构光审计已完成的前提下：

```powershell
uv run python -m experiments.projector.probe_compennetpp_raw_reference_pairs --setup light3/pos1/cloud_np --ref-numbers 1 5 21 101 125 --train-numbers 1 2 --test-numbers 1 2 3 --output-json work/compennetpp_light3_initial_probe.json
uv run python -m experiments.projector.probe_compennetpp_raw_reference_pairs --setup light3/pos1/cloud_np --ref-numbers 2 3 4 6 11 16 26 51 76 --test-numbers --output-json work/compennetpp_light3_primary_curve_probe.json
uv run python -m experiments.projector.probe_compennetpp_raw_reference_pairs --setup light3/pos1/cloud_np --ref-numbers --test-numbers 10 35 37 40 48 88 97 98 112 113 114 116 153 183 185 189 198 --output-json work/compennetpp_light3_test20_extension_probe.json
uv run python -m experiments.projector.evaluate_compennetpp_cross_setup_display
```
