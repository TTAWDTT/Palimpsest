# 自研非深度学习判别算法

当前处于设计阶段，开发次序、候选信号、数据划分与验收见[算法研发流程](../../../../docs/robust_ai_detection_plan.md)。任务是输入经过未知处理的图像，判断 AI／非 AI，并衡量处理后的绝对准确率、同源下降与速度。

待开发。新增方法实现 `palimpsest.detection.interfaces.OriginDetector`，用共用 `Prediction` 输出分数、阈值与计时。手工特征和决策规则属于这里；外部论文基线属于相邻 `baselines/`。

应验证原图、平台传播和真实再数字化的来源配对下降及 CPU 端到端延迟。目前没有可调用的自研算法，目录不提供默认预测。
