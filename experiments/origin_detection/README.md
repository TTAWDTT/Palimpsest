# 来源判别

[实验首页](../README.md) · [约定](../../docs/experiment_conventions.md)

| 研究问题/范围 | 协议数据 | 文件数 |
|---|---|---:|
| [固定检测器 baseline](baselines/README.md) | baselines | 17 |
| [稳健统计算法首轮筛选](robust_statistics/rr/README.md) | rr开发来源，非最终测试 | 5个入口/协议 |
| [固定规则真实拍屏诊断](robust_statistics/chimera/README.md) | chimera，参数不回调 | 1个入口 |
| [配对处理方向投影](paired_projection/rr/README.md) | rr既有开发来源，两个候选及不投影参考 | 1个入口 |
| [投影规则同来源方法比较](paired_projection/chimera/README.md) | chimera固定缓存，真实拍屏失败记录 | 1个入口 |
| [灰度顺序统计跨数据开发](ordinal_statistics/README.md) | rr既有角色＋已暴露chimera，两个预登记规则 | 审计/提取/筛选统一入口 |
| [同源数字传播对照](propagation_controls/rr/README.md) | rr | 3个入口/协议 |
| [接口与真实权重接入核对](integration/rr/README.md) | rr | 1 |
| [编码捷径与对照](encoding_shortcuts/rr/README.md) | rr | 5 |
| [平台传播统计原型](platform_statistics/rr/README.md) | rr | 8 |
| [再数字化统计原型](redigital_statistics/rr/README.md) | rr | 8 |
| [检测器分数迁移](score_transfer/chimera/README.md) | chimera | 4 |

本轮同来源缓存baseline对照：`python -m experiments.origin_detection.robust_statistics.evaluate_development_comparison`；不重新推理。

[phase_statistics](phase_statistics/README.md)：第四轮闭合相位及多原型预登记、签名缓存与速度入口，失败结果已归档。

[frozen_readout](frozen_readout/README.md)：第五轮，冻结灰度顺序／相位缓存上的固定RF与来源标签置乱诊断；不解码图像。

[residual_statistics](residual_statistics/README.md)：第六轮，256尺度的归一化RGB残差共生与顺序统计；原图／处理联合标定、两个消融候选、便携树推理、重编码与文件速度对照。

[第七轮训练干预](residual_training/README.md)：保持第六轮表示，四个预登记权重／编码对照，另留JPEG70评测。

[第八轮配对分数稳定](paired_stability/README.md)：冻结表示与来源，零约束和两个稳定惩罚，保留信号抹除反例。

[第九轮阈值标定](threshold_calibration/README.md)：三个冻结读出、两个已见标定范围，不改变特征或权重；加入编码后两类准入。

[第十轮像素条件读出](conditional_residual/README.md)：按来源交叉拟合的编码门控和条件线性读出；人工通道反转对照先行，真实判别仍失败。

[第十一轮小波包络](wavelet_envelopes/README.md)：固定一阶／二阶／合并／残差合并四表示，检验较大范围的空间调制能否保留来源线索。

`threshold_diagnostics/audit_gaussian_shift.py`：独立高斯平移反例，核验AUC稳定并不保证固定阈值BA稳定。

- [`gaussian_readout`](gaussian_readout/README.md)：第十二轮固定均值、方差和协方差读出，复用签名表示。
- `threshold_diagnostics/audit_rate_boundaries.py`：从真假整数正确数追加审计比例边界，保存原收据。

[第十三轮组风险与配对](group_margin/README.md)：固定凸目标四消融，错源配对与来源标签置乱控制。

[第十四轮核读出](kernel_readout/README.md)：固定随机Fourier非线性容量，完整可移植map+head规则，不将核近似误作物理稳定保证。

- [color_relations](color_relations/README.md)：第十五轮固定RGB偏序/联合残差，检验旧独立通道表示遗漏的信息；人工控制与完整开发评测已完成，强编码退化仍失败。

## 继续迭代

[冻结CLIP表示](frozen_clip/README.md)固定现有预训练编码器，检查其类别信号与处理退化；[冻结读出](frozen_readout/README.md)提供公共逐候选评测。

[来源内风险](source_view_risk/README.md)：第十七轮固定2×2对照，六头完整评价已完成；[小型冻结编码器](compact_frozen_encoder/README.md)：第十八轮，固定一个较小既有神经表示，运行控制与提取入口。

[语义核](semantic_kernel/README.md)、[同源慢子空间](slow_subspace/README.md)、[来源分数一致性](source_score_consistency/README.md)：第十九至二十一轮，完整开发评测和失败记录；不继续事后调参。

[小型语义编码器](compact_semantic_encoder/README.md)：第二十二轮CLIP-B/16已完成，最低汇总BA81.11%、细分下降11.25pp，未采用。

[DEAR-r局部法医对照](dear_baseline/README.md)：第二十三轮固定作者权重，区分原生OOM与经数值核对的空间分块执行。

[中间表示](intermediate_encoder/README.md)：第二十四轮固定中点LN2与来源风险头；人工存储控制通过，CUDA和像素提取尚待序列执行。
