# 共用传统读出与标定

这些模块只接收数值特征与训练／标定记录，不加载编码器，不调用Torch。输入若来自神经网络，完整方法仍是神经方法。

| 模块 | 职责 |
|---|---|
| [stable_rule.py](stable_rule.py) | 显式线性规则、JSON保存加载、配对ridge拟合 |
| [source_view_risk.py](source_view_risk.py) | 同源视图的平均或平滑较差分类风险 |
| [consistent_source_risk.py](consistent_source_risk.py) | 分类风险与同源分数方差约束 |
| [source_hinge.py](source_hinge.py) | 有限视图软间隔线性规划 |
| [paired_threshold.py](paired_threshold.py) | 基于BA与同源下降的全局阈值标定及显式回退 |
| [rate_penalty.py](rate_penalty.py) | 分组平滑错误率与处理后增量代理 |

它们可服务像素算法和冻结神经特征方法。数值求解通过不能证明真实传播稳定性。像素残差的 `StableDetector` 保留在相邻 `paired_stability.py`，不会混入共用规则载体。
