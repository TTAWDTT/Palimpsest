# 含神经网络的自研判别方法

[frozen_features](frozen_features/README.md)：已有冻结CuRe编码器＋自研统计头，当前85%／2.5pp候选属于这条路线，尚未达标。冻结参数不改变整套方法包含神经网络的性质。

[trainable](trainable/README.md)：后续训练神经参数的模型路线，待开发。

特征提取器在相邻 `representations/`，通用传统风险与标定在 `algorithms/readouts/`。模型包负责组合方法、专用规则和推理接入。CuRe保留文件预处理接口，尚未接入区域RGB链。
