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
