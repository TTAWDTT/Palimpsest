# 算法研发

[研发流程](../../docs/robust_ai_detection_plan.md) · [固定 baseline](../02_数据与基线/README.md) · [实验入口](../../experiments/origin_detection/robust_statistics/rr/README.md)

先看[第二轮配对方向投影](第二轮配对方向投影_RR改善与真实拍屏失败_2026-10-05.md)：RR内部改善没有迁移到Chimera拍屏，速度预算未过，当前构造停止推进。此前[首轮局部统计迭代](首轮局部统计算法_开发筛选与真实拍屏诊断_2026-10-05.md)保留颜色规则的失败记录。

区分开发筛选、阈值标定、固定规则外部诊断和最终独立验证；不能用一个通过的筛选门槛代表整个目标完成。
