# CuRe五方向读出实验

[公共实现](../../../../src/palimpsest/detection/models/frozen_features/cure/README.md) · [历史报告](../../../../reports/04_算法研发/README.md)

| 轮次 | 入口／协议 | 研究变化 |
|---|---|---|
| 67 | [quantile_score_consistency](quantile_score_consistency/README.md) | 五方向／20项同源一致性读出 |
| 68 | [directed_score_margin](directed_score_margin/README.md) | 有向处理margin损失 |
| 69 | [paired_ba_calibration](paired_ba_calibration/README.md) | BA／配对跌幅联合阈值政策；85%／2.5pp未达标 |
| 70 | [rate_score_readout](rate_score_readout/README.md) | 分组平滑分类率代理 |
| 71 | [split_rate_score](split_rate_score/README.md) | 方向与读出按来源分拆 |
| 72 | [complementary_split](complementary_split/README.md) | 两个互补分拆的固定平均 |

当前调用形式：`python -m experiments.origin_detection.frozen_features.cure.paired_ba_calibration.run_iteration --help`。其他入口同样使用新包前缀，不存在兼容转发脚本。

本次迁移不重启实验。私有结果仍在原 `work/robust_statistics/<问题>/`，历史代码／数据指纹保持原值；冻结收据在新源码上可能拒绝，这是预期的来源检查。完整历史复算使用原执行提交，下一轮运行必须另签当前代码与输出目录，不能覆盖旧实验。
