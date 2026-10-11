# 第九轮：单阈值的类别最差正确率标定

[第八轮](../paired_stability/README.md) · [研发流程](../../../docs/robust_ai_detection_plan.md)

计算前固定六候选：第八轮λ=0/1/10三个规则，各用raw或raw＋JPEG90的threshold视图标定。禁止修改任何表示、权重、标准化参数、来源角色或selection阈值反馈；不以JPEG70作标定，不打开RR reserved。

旧maximin BA可能在最难视图偏向某一类别。新单阈值最大化**所有标定视图、真假两类正确率的最小值**；同值先最大化最差BA，再取abs阈值小、数值小。预测仍是score>threshold；不需要知道处理、来源或类别。阈值候选为所有标定分数和最小分数下方nextafter，枚举一次，无搜索扩张。

继承第八轮七门槛，另要求JPEG90与JPEG70后的六个域聚合条件（包含编码原图）的真假正确率都≥55%。不能仅用BA≥55%遮蔽某一类失败。没有独立数据新颖性或泛化认证；这是目标函数与验收条件对齐的开发实验。

对照保留第八轮旧阈值结果及SHA。先人工已知阈值和等号边界、常数反例与身份损坏控制，再六次标定／selection复算；不重新训练规则、不解码图、不重跑baseline。全部失败不另作速度benchmark。

不重新拟合null：第八轮固定来源置乱的AUC门槛失败，改变常数阈值不能改变其排序。若父null AUC检查不再失败，则本轮拒绝解释，而不是假定新阈值安全。

```powershell
python -m experiments.origin_detection.threshold_calibration.run_iteration
```
