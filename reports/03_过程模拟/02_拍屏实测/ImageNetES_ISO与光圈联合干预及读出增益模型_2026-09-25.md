# ImageNet-ES：ISO 与光圈联合干预，分离入射光和读出增益

2026-09-25

## 数据口径

承接[同源跨光圈检查](ImageNetES真实光圈干预与拍屏通量模型外部反证_2026-09-25.md)，从 [ImageNet-ES 官方档案](https://huggingface.co/datasets/edw2n/ImageNet-ES)对已固定的二十个来源补取 ISO 2000 条件。现在每个来源有数字参考及六张真实 OLED 拍屏发布 JPEG：名义光圈 f/5、f/9、f/16 × ISO 250、2000，均为关灯 `l5` 和 1/60 秒。参数映射按[作者测试网格](https://github.com/Edw2n/ImageNet-ES/blob/a46c92cc8f77dc1d4d7d44743e1aa0f70d89/settings/grid-options-3x3x3.csv)。合计 **20 个来源、120 张拍屏 JPEG、20 张数字参考**；140 个选取的 ZIP 成员逐一核对尺寸、CRC、SHA-256 并解码，不代表全档案逐图核验。

冻结登记表为 `E:\ai_image_origin_research\data\manifests\imagenet_es_aperture_iso_development.csv`，SHA-256 `ea51cc7ee65b1a3541a0be29f3408e10365ba9215f339c26e053b2c7ef5298a7`。这二十组均为**开发样本**；没有将作者其余 test 来源用于拟合或本次统计。提取、登记、评价的可复算脚本分别为 `experiments/imagenet_es/extract_imagenet_es_control_pairs.py`、`experiments/imagenet_es/freeze_imagenet_es_aperture_registry.py`、`experiments/imagenet_es/evaluate_imagenet_es_iso_aperture.py`；逐图摘要在 `work/imagenet_es_20class_iso_aperture_evaluation.json`。

## 观察到的发布 JPEG 变化

| 光圈 | ISO 250：平均编码亮度中位 | ISO 2000：平均编码亮度中位 | ISO 2000：近白像素比例中位 |
|---|---:|---:|---:|
| f/5 | 0.403 | 0.766 | 34.2% |
| f/9 | 0.155 | 0.559 | 0.015% |
| f/16 | 0.048 | 0.284 | 0% |

亮度为每图 Rec.709 权重的 JPEG 编码 RGB 均值，再在二十个来源间取中位；近白定义为三个通道都 ≥0.98。在每个光圈，**20/20 来源**的 ISO 2000 成片亮于 ISO 250。f/5、ISO 2000 有明显高光截断，因此不能用整图亮度比估计 ISO 的线性增益。

把 f/16、ISO 2000 与 f/5、ISO 250 配对，名义的 `ISO 比 × 理想入瞳面积比` 为 `8×(5/16)²=0.78125`；实际两张 JPEG 的平均编码亮度比，来源中位为 **0.752**。若将 f/5、ISO 250 的发布 JPEG 逆标准 sRGB 后乘 0.78125，再编码去预测 f/16、ISO 2000，预测平均亮度/真实平均亮度中位为 **1.187**，逐图 RGB MAE 中位为 **0.0655**。这些数字只说明该简单**成片**变换不等于真实相机的发布结果；名义 ISO 值未必等于已测得的模拟增益，标准 sRGB 逆变换也不是 Canon 相机 ISP 的逆过程。

## 对模拟器的过程修正

屏幕发光、光学入瞳、快门和传感器量子噪声先决定收集的光电子；ISO 对读出链的作用不能在这一阶段当作“额外光子”。当前模拟器新增**默认 1** 的 `sensor_analog_gain_relative`：在光电子的泊松采样及简化读出噪声之后、归一化 ADC 截断之前放大信号。同一虚拟平场上，增益 ×8 与曝光电子数 ×8 可以产生相同无噪声输出均值，但前者光子数不变，量化前的相对散粒噪声更大；增益也会更早触及输出上限。现有 `full_well_electrons` 仍是 unity-gain 等效归一化上限，该阶段未标定具体相机 ADC/转换增益。

结构检验覆盖：同均值不同散粒噪声、增益造成的输出截断，以及非法参数拒绝。完整项目测试 `python -m pytest -q` 结果为 **82 passed**。这个组件是**相机读出假设**，不是 EOS RP 的 ISO 250/2000 标定；真实相机可能同时改变模拟增益、数字增益、降噪和 JPEG 色调，发布包没有逐张 RAW、黑电平或增益拆分。

## 下一道可证伪关口

在其余来源保持冻结的前提下，用这二十组开发样本约束一个明确的“传感器信号 → JPEG 色调/黑位/截断”子模型，再对**未用过的来源**和光圈／ISO 组合检验亮度、颜色、饱和区域及空间残差。任何从 JPEG 拟合的响应只能叫**有效发布图响应**；若没有 RAW 与设备记录，不能把拟合参数称为真实传感器量子效率、T-stop 或 ISO 模拟增益。OLED 子像素和几何/对焦仍需另行证据。
