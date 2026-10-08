# 有向margin条件和实现边界

训练已知同来源before／after且标签相同。记signed标签t∈{−1,+1}，共同阈值τ、评分s。若`t*(s_before-s_after)≤0`，Fake的after score不小于before；Real的after score不大于before。因此before判对时after也判对，Real使用score≤τ的tie规则。有限边权均正、每个正部平方为0时条件逐边成立。

这只是已知评分／边上的条件命题，未证明物理通道满足它。正部平方损失很小也不足：Fake before=ε、after=−ε，在τ0时所有案例翻转，损失4ε²可任意小。数值拟合及未知通道无需满足零违反，绝对BA约束和独立验证不能删除。

对固定映射和标签，平方正部作用于w的线性差，因此凸、梯度为`2 Σ edge_weight max(0,t*Δz·w) t*Δz`，共同bias消去。原source风险／ridge仍使用真实来源组；错误配对只改变这个附加项，标签符号来自当前拟合labels，可为source-null，不能回用metadata真标签。

此项借用已知pairwise ranking／hinge思路。[Joachims2002原始§4.1](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf)在检索偏好上定义pairwise difference，本文只定向读摘要／§4.1及4.2开始，未复现原SVM、代码、全篇实验。检索效果不支持图像传播稳定性；本任务需要自己的明确边、负控和实证。

已检查现有variance/source-risk工具：可复用source风险及优化器，但对称variance会同时惩罚改善与退化，且value-only fitter无处理标签，不能表达所需方向。新增训练行context分发并扩展共享source-panel CV，保留数值fitter接口；没有通过嵌入猜条件或在预测时传原图。具体数据／版本／计数见[事前实验](../../experiments/origin_detection/directed_score_margin/README.md)。
