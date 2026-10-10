# 统计表示与传统判别读出

当前处于候选探索阶段，开发次序、候选信号、数据划分与验收见[算法研发流程](../../../../docs/robust_ai_detection_plan.md)。任务是输入经过未知处理的图像，判断 AI／非 AI，并衡量处理后的绝对准确率、同源下降与速度。

已有[局部统计规则原型](local_statistics/README.md)，用于首轮候选筛选，尚无通过完整真实验证的自研方法。新增方法实现 `palimpsest.detection.interfaces.OriginDetector`，用共用 `Prediction` 输出分数、阈值与计时。手工特征和决策规则属于这里；外部论文基线属于相邻 `baselines/`。

第二轮新增同特征上的[配对处理方向投影协议](../../../../experiments/origin_detection/paired_projection/rr/README.md)，只检验两个预先固定的秩，不把线性子空间行为视作图像物理不变性。

应验证原图、平台传播和真实再数字化的来源配对下降及 CPU 端到端延迟。原型需要显式标定参数，目录不提供默认预测。

第三轮[灰度顺序统计](ordinal_statistics/README.md)实现28个轨道/相等频率的等权规则；跨数据开发中未通过处理后准确度门槛，属于可复算失败候选。

第四轮[闭合相位与多原型](phase_statistics/README.md)使用6个偶相位矩和固定标签标定；跨数据开发未过门槛，不提供成功检测的默认参数。

第六轮[归一化残差与顺序统计](residual_statistics/README.md)固定256尺度、RGB三尺度二阶差分共生；两个候选由原图和处理图联合标定，仍需通过真实准确度及速度门槛。

## 公共特征森林

[forest.py](forest.py)保存明确的二叉树JSON，并用NumPy遍历树做像素特征之后的分数推理，不依赖sklearn或pickle。拟合仍在实验侧用sklearn；相邻detector负责图片预处理和预测协议。

适用于固定浅层森林的单图推理、参数可读性及保存加载核验。分数是树叶AI比例的平均减0.5；这个比例未独立校准。输入按sklearn树约定转float32，节点切分使用`<=`，最终AI判断仍为`score > threshold`。树维度、概率、循环／共享节点、输入非有限或模式不符时抛错，不给出默认为自然图的结果。

信任前要求手写树真值、阈值等号和float32边界控制通过，与sklearn输出的误差≤1e-12，并验证JSON保存加载与故意破坏参数的拒绝。它们验证树推理与导出；数据来源、标签质量及真实泛化需要各实验另审。当前工具的控制在`tests/detection/test_residual_statistics.py`，不会修改已冻结的第三至第五轮实现。

[paired_stability.py](paired_stability.py)只承担像素残差适配；通用规则、配对ridge、同源风险和阈值标定在[readouts](readouts/README.md)，两条路线复用。

[conditional_residual.py](conditional_residual.py)：像素统计门控＋条件读出，参数显式保存；第十轮未过准确度筛选，不作为默认检测器。

[wavelet_envelopes.py](wavelet_envelopes.py)：固定离散Morlet模值及二阶包络统计；第十一轮开发中。不是完整散射或拍屏不变性证明。

`gaussian_readout.py` 提供非神经的共享协方差、Gaussian朴素贝叶斯与收缩QDA显式规则；仅用fit统计，无目标批次适应，稳健性尚待验证。

`group_margin.py` 以凸logistic目标检验均值风险、熵组风险与配对约束；复用显式`StableRule`推理/导出，目标及参数由实验收据区分。

`kernel_readout.py` 提供固定Fourier映射和显式嵌套读出，先验证核近似及从原特征到最终分数的复算，当前稳健性尚未通过。

`color_relations.py`：RTPO启发的显式适配，351维RGB偏序/平局与同位置联合残差；固定网格的有限关系不变性不扩展到颜色混合/缩放。暂无合格部署规则。

## 与神经方法的边界

共用求解器是非神经数值模块，但输入来自神经编码器时，完整方法仍含神经网络。CuRe三／四／五方向规则及第67—72轮组合迁入相邻[models/frozen_features/cure](../models/frozen_features/cure/README.md)，旧路径直接消除，不保留转发模块。

非神经路线继续使用局部、顺序、相位、残差、小波和RGB联合统计。推荐接续[第15轮](../../../../reports/04_算法研发/第十五轮RGB联合关系_原始关联提高但强编码仍失败_2026-10-06.md)，第6轮作为较简单对照；这些方法均尚未通过最终稳健性验收。

通用的 `slow_subspace.py`、`source_partition.py` 和核／协方差数值操作仍可接受明确的非神经特征。模块本身的求解方式不能代表整套输入链是否使用神经网络。
