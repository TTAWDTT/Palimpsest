# ImageNet-ES：用真实同源跨光圈拍屏约束通量模拟

2026-09-25

## 为什么这套数据重要

[ImageNet-ES 原论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Baek_Unexplored_Faces_of_Robustness_and_Out-of-Distribution_Covariate_Shifts_in_Environment_CVPR_2024_paper.pdf)在暗室把已知数字图显示到屏幕，再用真实相机系统改变光圈、快门、ISO 与灯光。[作者补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Baek_Unexplored_Faces_of_CVPR_2024_supplemental.pdf)说明设备是 55 英寸 4K LG OLED55B3FNA、Canon EOS RP 与 RF 24–105 mm 镜头，屏幕—相机名义距离约 1 m；对焦为 AF，作者将相机原画幅中的有效图像区域裁出发布。因此它提供**已知数字源→真实拍屏**的同源、条件干预，不必依赖用户当前无法完成的手机拍摄。它是 OLED，**不能**用我们假定的竖直 RGB LCD 发光条去解释屏幕格点。

这套数据不直接解决 AI 图像来源判别：作者选的是 Tiny-ImageNet 自然图像，也没有 AI/真实来源二分类标签。它的价值是给拍屏模拟的**光圈／快门／增益与成片响应子链**一组外部约束。

## 远程档案和逐图映射审计

[官方 Hugging Face 档案](https://huggingface.co/datasets/edw2n/ImageNet-ES)为 `ImageNet-ES.zip`，本次 HEAD 长度 **26,267,837,212 字节**、ETag `ea8e6692bd861bf91c2e37dbe4e12f4f0ce822e41effe8587fcd3fa0889a2b6d`。通过 HTTP Range 读取 ZIP 中央目录，核到 **212,003 个文件成员：212,000 JPEG、3 JSON**。该数目包括作者发布的拍屏条件图、数字参考图及训练图；**整包未下载，也未逐张验收所有 212,003 个成员**。

作者仓库提交 `a46c92cc8f77dc1d4d7d44743e1aa0f70d89` 的[测试参数表](https://github.com/Edw2n/ImageNet-ES/blob/a46c92cc8f77dc1d4d7d44743e1aa0f70d89/settings/grid-options-3x3x3.csv)把 `param_4/13/22` 分别映射到同一 **ISO 250、1/60 秒**的 f/5、f/9、f/16；`param_5/14/23` 是 ISO 2000、同快门的对应组。测试发布图以相同 `类别/原文件名` 放在 `es-test/param_control/l5/param_ID/`，数字参考在 `es-test/sampled_tin_no_resize2/`。最初检查了四个来源的两档 ISO 先导组；随后固定二十个不同类别的 f/5、f/9、f/16 组用于本分析。**这二十组的 80 个文件**均经 ZIP 成员尺寸/CRC 检查、逐文件 SHA-256 记录并解码；先导组另有十二个 ISO 2000 文件，不纳入下文统计。数据存于 `E:\ai_image_origin_research\data\derived\imagenet_es_probe`；逐文件结果在 `work/imagenet_es_20class_aperture_manifest.json`，代码在 `experiments/imagenet_es/audit_imagenet_es_remote_index.py`、`experiments/imagenet_es/extract_imagenet_es_control_pairs.py`。

二十个不同类别各取哈希排序的一个原图，**已作为本项目开发样本使用**。作者 test split 的其余类别尚未打开用于本分析；以后评测本项目模拟方法，不得把这二十组再叫作独立外部测试。发布 JPEG 的同源三光圈图尺寸逐组一致，但所查图片均无 EXIF；这批数据没有提供逐张线性 RAW、传感器黑电平、镜头实测 T-stop、实际焦距或 OLED 发光谱。

已将这二十组冻结为 `E:\ai_image_origin_research\data\manifests\imagenet_es_aperture_development.csv`，逐图含官方 ZIP 成员名、来源键、名义光圈/ISO/快门和本地 SHA-256。登记表 SHA-256 为 `5eb98b60d4342ddf27716b3db51fc74207ebc7526689b80144365633ff3b6d1e`；生成器 `experiments/imagenet_es/freeze_imagenet_es_aperture_registry.py`重新核对了 80 张本地文件的哈希和 20 组配对完整性。

## 真实干预与简单过程模型的比较

先取关灯 `l5`、ISO 250、1/60 秒条件；同一来源、同一灯光，只改变参数表光圈。20/20 组的**整幅 JPEG 平均编码亮度**按 f/5＞f/9＞f/16 单调下降。三档编码亮度的跨来源中位分别为 **0.403、0.155、0.048**；接近黑色的像素比例均值由 **8.0%→15.8%→30.1%**。这一部分支持“光圈干预改变了发布图”，但 JPEG 是相机处理后的数值，不能直接解释为光子计数。

为压力测试[我们的相对光圈通量链](../03_前向模型/拍屏光圈到光子计数的相对通量链_2026-09-25.md)，做两种**明确标注假设**的比较：

| 量 | f/9 相对 f/5 | f/16 相对 f/5 |
|---|---:|---:|
| 理想瞳孔面积 `N⁻²`，只对线性传感器光量成立 | 0.3086 | 0.0977 |
| 把 JPEG 当标准 sRGB 逆变换，在同位置平滑后取像素比；18 个有效来源的中位 | 0.1908 | 0.0232 |
| 上述来源中位的 bootstrap 95% 区间¹ | 0.1713–0.2104 | 0.0207–0.0286 |

¹ 只在二十个内容来源中重采样；两个来源因共同非暗非饱和掩膜不足而未进入比值。它**不**代表跨设备或跨环境的不确定度。逆标准 sRGB 不等于逆 Canon JPEG ISP，因此本表不能推翻 `N⁻²` 的传感器光学规律，也不能由差值反推出实际 T-stop。

另设一个可证伪的**最简单成片基线**：把真实 f/5 发布 JPEG 当作已经标准 sRGB 编码的线性场景测量，逆 sRGB→乘 `(5/N)²`→再编码，直接预测相同位置 f/9 或 f/16。与真实配对 JPEG 比，预测平均亮度/真实平均亮度的来源中位为 **1.34、2.33**；分别在 **19/20、20/20** 个来源预测偏亮，逐图 RGB MAE 来源中位约 **0.0509、0.0653**（`[0,1]` 单位）。这表明“理想光圈平方比＋标准 sRGB”**不足以**模拟这批发布 JPEG 的亮度。可能的缺口包括实际镜头透过率、相机曝光和内部色调响应、显示发光、AF、裁切与图像配准；现有发布内容无法把它们逐项辨识。原始逐图统计与程序：`work/imagenet_es_20class_aperture_evaluation.json`、`experiments/imagenet_es/evaluate_imagenet_es_aperture_pairs.py`。

## 对研发路线的影响和已有工作的区别

这给模拟器设置了两层目标：**传感器层**应检验光圈、快门、光子泊松噪声的相对关系；**发布图层**还须单独检验色调、裁切、JPEG 与亮度分布，不能把理想 `N⁻²` 直接当成 JPEG 真值。ImageNet-ES 只允许检验后一层及定性干预方向；要辨认物理光子率仍需要线性 RAW、暗帧和曝光记录。对 OLED 屏幕子像素、视角、对焦、莫尔纹过程，本次样本未给出可签发的参数。

[ImageNet-sES / CycleGAN-ES（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/papers/Kim_ImageNet-sES_A_First_Systematic_Study_of_Sensor-Environment_Simulation_Anchored_by_WACV_2026_paper.pdf)已经在这套真实拍屏数据上研究过模拟：每种环境/传感器条件分别训练无配对 CycleGAN，在 sRGB 层学习样式并报告保真和鲁棒性收益；作者明确承认尚无显式 RAW/ISP 过程建模。因此本项目不能把“用 ImageNet-ES 生成模拟图”本身称为新贡献。可能的区别必须由**同源干预的因果预测、分环节可校准性及来源检测器迁移**来证明，目前尚未证明。
