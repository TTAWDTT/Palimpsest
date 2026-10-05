# 固定投影规则的真实拍屏对照

[RR拟合与门槛](../rr/README.md) · [共用图像推理入口](../../robust_statistics/chimera/README.md)

推理复用`robust_statistics.chimera.run_algorithm --projection`，而本目录负责从签发清单及固定缓存比较第二轮、旧颜色规则和B-Free。没有重复的图像推理器或特征提取器。

```powershell
uv run --locked python -m experiments.origin_detection.paired_projection.chimera.evaluate_comparison
uv run --locked python -m experiments.origin_detection.robust_statistics.rr.benchmark_algorithm --projection
```

输出：`work/robust_statistics/chimera_paired_projection/comparison.json`。每条件给出同来源方法BA差及来源级分层bootstrap区间，并按三种已知场景作事后诊断；不能由场景差异推断成像因果关系。**旧Chimera结果已经暴露，固定参数对照不是新盲测**；不把此处数据用于改阈值、换rank或翻转方向。

比较复用`evaluation.cached`的覆盖/标签/有限数审计、`classification`的指标和`pairing.paired_change`的来源重采样。AUC不另写算法；数值的已知排序与错位控制已有测试，投影控制见[第二轮测试](../../../../tests/detection/test_paired_projection.py)。CPU参考仅比较明确协议，各方法历史GPU/CPU计时不是公平速度排名。
