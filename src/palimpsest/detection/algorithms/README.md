# 自研非深度学习判别算法

当前处于候选探索阶段，开发次序、候选信号、数据划分与验收见[算法研发流程](../../../../docs/robust_ai_detection_plan.md)。任务是输入经过未知处理的图像，判断 AI／非 AI，并衡量处理后的绝对准确率、同源下降与速度。

已有[局部统计规则原型](local_statistics/README.md)，用于首轮候选筛选，尚无通过完整真实验证的自研方法。新增方法实现 `palimpsest.detection.interfaces.OriginDetector`，用共用 `Prediction` 输出分数、阈值与计时。手工特征和决策规则属于这里；外部论文基线属于相邻 `baselines/`。

第二轮新增同特征上的[配对处理方向投影协议](../../../../experiments/origin_detection/paired_projection/rr/README.md)，只检验两个预先固定的秩，不把线性子空间行为视作图像物理不变性。

应验证原图、平台传播和真实再数字化的来源配对下降及 CPU 端到端延迟。原型需要显式标定参数，目录不提供默认预测。

第三轮[灰度顺序统计](ordinal_statistics/README.md)实现28个轨道/相等频率的等权规则；跨数据开发中未通过处理后准确度门槛，属于可复算失败候选。
