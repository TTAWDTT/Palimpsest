# 冻结 CuRe 特征上的自研判别头

本目录的方法使用神经网络。作者发布的 CuRe 编码器和适配器保持冻结，我们拟合统计判别头及统一阈值；这不是纯非神经网络算法，也不是官方 CuRe 判别分数。

## 第69轮候选的业务链

文件 → 作者预处理 → 一次冻结编码器前向 → 3920维描述符 → 五个拟合内判别方向 → 20项一／二次展开 → 同源风险与分数方差读出 → 一个全局阈值。

- 特征提取复用 [FrozenCureQuantiles](../../../representations/frozen_cure_quantiles.py)，保留原3600维并增加320维投影分位数，不另做编码器前向。
- [quantile_score_consistency.py](quantile_score_consistency.py)定义可保存、加载的五方向规则与拟合器；此前三／四方向和软间隔变体也集中在本目录。
- [detector.py](detector.py)组合完整特征提取器与已拟合规则，提供 `CureQuantileDetector.predict_file(path)`。
- 通用风险、间隔和标定求解器在 [algorithms/readouts](../../../algorithms/readouts/README.md)。它们也可用于手工像素特征，不加载神经网络。
- 实验在 [frozen_features/cure](../../../../../../experiments/origin_detection/frozen_features/cure/README.md)，数据划分、拟合和标定不进入推理模块。

当前最佳联合候选为第69轮最低域汇总BA85%、最大同源下降2.5个百分点，尚未满足下降≤2个百分点或独立真实数据验证。第70—72轮为后续失败变体；目录归属不改变历史结果。

## 显式加载与文件预测

```python
from pathlib import Path
from palimpsest.detection.models.frozen_features.cure.detector import (
    CureQuantileDetector,
)

# 路径和第三方源码指纹由调用者明确提供；不自动下载或选择参数。
detector = CureQuantileDetector.from_paths(
    Path(rule_path),
    Path(vendor_path),
    Path(adapter_path),
    Path(backbone_path),
    source_pins=verified_source_pins,
)
try:
    result = detector.predict_file(Path(image_path))
    print(result.prediction.origin, result.prediction.score, result.end_to_end_ms)
finally:
    detector.close()
```

`rule_path`必须是已拟合五方向规则；本机第69轮文件是 `work/robust_statistics/paired_ba_calibration/selected_rule.json`，不随源码发布。骨干、作者适配器与vendor代码仍在仓库外或忽略目录。`from_paths`先核规则布局，然后沿用已有编码器的权重／源码核验。

预测只接收当前文件，不读取来源标签、原图或处理条件。作者预处理按PNG后缀增加编码，因此此处保留文件接口，尚未实现 `OriginDetector.predict(rgb)` 或接入区域CLI。它不会通过临时文件偷偷改变区域输入的处理协议。

`prediction.timing_ms`只列读出；外层 `feature_extraction_ms`包括文件解码、预处理、编码器及统计，`end_to_end_ms`包括上述阶段和读出，不含加载与结果序列化。原始评分不填 `ai_probability`，严格 `score > threshold` 判AI，等号判自然。

## 迁移与历史复算

本次直接迁移代码及更新调用，不保留旧路径转发模块，不重拟合参数、不重推理图片。规则JSON的schema、kind和数值布局保持原样。

历史收据的代码SHA仍指向原执行版本 `d279b7e965e2ac8ad2f63b4110cd46cace55addb`。目录迁移使当前源码指纹改变；不能修改旧收据来绕过检查。完整历史复算应使用原提交，当前路径用于后续开发。[迁移记录](../../../../../../docs/maintenance/cure_method_migration.json)记录直接移动与共用规则抽取。
