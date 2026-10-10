# 原始视觉内容判别

实现 `OriginDetector.predict(uint8 RGB) -> Prediction`：分数越大越偏 AI，严格超过声明阈值才判 AI。自然摄影/AI 标签描述原始视觉内容，经过拍屏不会改变其标签。

| 目录 | 当前状态 |
|---|---|
| `baselines/` | B-Free、D3、DEAR-r研究权重适配；Benford 特征和已拟合森林的数值导出/推理 |
| [algorithms](algorithms/README.md) | 局部统计规则原型可显式加载参数；首轮真实拍屏稳健性未通过，无默认成熟方法 |
| [representations](representations/README.md) | 已有冻结CLIP／CuRe神经表示，不自动输出来源判定；下游读出与稳健性评测在实验侧 |
| [models](models/README.md) | 冻结CuRe＋自研统计头已有候选；trainable神经训练路线待开发 |

所有方法使用同一 `Prediction`，明确 raw logit、未校准概率差和阈值。未校准分数不填 `ai_probability`。具体环境、协议差异和安装见仓库 `docs/detection_pipeline.md`。
